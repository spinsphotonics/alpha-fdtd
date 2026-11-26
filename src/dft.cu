// SPDX-License-Identifier: MIT
// dft.cu -- afdtd_dft: the DFT of a recording file (record.cpp) on one GPU.
//     out[f][c][x][y][z] = sum over frames k of v_c(k, x, y, z) exp(-i omega_f tau_c(k)),  tau = first_step + k every (E), minus 1/2 (H')
// The box is cut into chunks of frame rows; a chunk is one contiguous byte range of every frame, so the file is read once in all.
// Per chunk and block of frames: the frames are read (threads, double-buffered against the GPU), decoded on the GPU into X[t][v]
// (float32; v = the selected components of the chunk's cells in (c, x, y, z) order), the phase table W[f][t] = exp(-i omega_f tau(t))
// is made on the GPU from a float64 phase, and Y[f][v] += W X runs in float32 tiles.  The chunk's Y then goes to `out` or the file.
#include "alpha_fdtd.h"
#include "internal.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace afdtd {
namespace {

struct Hdr { int64_t n[3], off[3], shape[3], every, first_step, frames, row0, rows, z0, planes, pair0, pairs, frame_bytes, stride, data_offset; int dtype; };

bool jget(const std::string& s, const char* key, int64_t* v, int k)
{
  const std::string q = std::string("\"") + key + "\":";
  size_t p = s.find(q); if (p == std::string::npos) return false;
  p += q.size();
  for (int i = 0; i < k; ++i) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '[' || s[p] == ',')) ++p;
    char* e = nullptr; v[i] = strtoll(s.c_str() + p, &e, 10);
    if (e == s.c_str() + p) return false;
    p = (size_t)(e - s.c_str());
  }
  return true;
}
Hdr read_header(int fd, const char* path)
{
  char b[4097]; memset(b, 0, sizeof b);
  if (pread(fd, b, 4096, 0) != 4096) fail(AFDTD_EIO, "%s: cannot read a 4096-byte header", path);
  const std::string s(b);
  if (s.find("\"format\": \"alpha-fdtd-record\"") == std::string::npos) fail(AFDTD_EINVAL, "%s is not an alpha-fdtd recording", path);
  Hdr h; int64_t one;
  bool ok = jget(s, "domain", h.n, 3) && jget(s, "offset", h.off, 3) && jget(s, "shape", h.shape, 3) && jget(s, "every", &h.every, 1) &&
            jget(s, "first_step", &h.first_step, 1) && jget(s, "frames", &h.frames, 1) && jget(s, "row0", &h.row0, 1) && jget(s, "rows", &h.rows, 1) &&
            jget(s, "z0", &h.z0, 1) && jget(s, "planes", &h.planes, 1) && jget(s, "pair0", &h.pair0, 1) && jget(s, "pairs", &h.pairs, 1) &&
            jget(s, "frame_bytes", &h.frame_bytes, 1) && jget(s, "frame_stride", &h.stride, 1) && jget(s, "data_offset", &h.data_offset, 1) &&
            jget(s, "version", &one, 1) && one == 1;
  if (!ok) fail(AFDTD_EINVAL, "%s: the header lacks a field or has an unknown version", path);
  h.dtype = s.find("\"dtype\": \"f32\"") != std::string::npos ? AFDTD_F32 : s.find("\"dtype\": \"f16\"") != std::string::npos ? AFDTD_F16 : AFDTD_BF16;
  // frames in the file: the header's count, or (a recording whose recorder was not flushed) what the file size holds
  const off_t end = lseek(fd, 0, SEEK_END);
  const int64_t have = end > h.data_offset ? (int64_t)((end - h.data_offset) / h.stride) : 0;
  if (h.frames == 0 || h.frames > have) h.frames = have;
  if (h.frames < 1) fail(AFDTD_EINVAL, "%s holds no frames", path);
  const int64_t want = h.dtype == AFDTD_F32 ? 6 * h.shape[0] * h.shape[1] * h.shape[2] * 4 : (int64_t)h.rows * h.planes * h.pairs * 24 * 24;
  if (want != h.frame_bytes) fail(AFDTD_EINVAL, "%s: inconsistent frame layout", path);
  return h;
}

struct Chunk { int ra, rb, ya, yb, xa, xb; };   // native frames: rows [ra, rb) (relative), box y [ya, yb); canonical: box x [xa, xb)

// raw frames -> X[t][v], v = ((ci * sx + x) * ny + y) * sz + z over the chunk's cells
__global__ void decode_k(const unsigned short* raw, float* X, int T, size_t chunk_bytes, int ncomp, const int* comps, int sx, int ny, int sz,
                         int bx0, int by0, int bz0, int ra, int row0, int planes, int pairs, int pair0, int z0f, int dt)
{
  const size_t V = (size_t)ncomp * sx * ny * sz, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= V * (size_t)T) return;
  const int t = (int)(i / V); size_t v = i % V;
  const int z = (int)(v % sz); v /= sz; const int yl = (int)(v % ny); v /= ny; const int xl = (int)(v % sx); const int ci = (int)(v / sx);
  const int x = bx0 + xl, y = by0 + yl, zz = bz0 + z, c = comps[ci];
  const size_t e = (((((size_t)(y / 24 - row0 - ra) * planes + (zz - z0f)) * pairs + (x / 2 - pair0)) * 24 + y % 24) * 12 + 2 * c + (x & 1));
  const unsigned short b = raw[(size_t)t * (chunk_bytes / 2) + e];
  X[i] = dt == AFDTD_F16 ? __half2float(__ushort_as_half(b)) : __uint_as_float((unsigned)b << 16);
}
// W[f][t] = exp(-i omega_f tau(t)), tau(t) = t0 + (k0 + t) every - half
__global__ void phase_k(float2* W, const double* om, int nf, int T, double t0, int64_t k0, int64_t every, double half)
{
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)nf * T) return;
  const int f = (int)(i / T), t = (int)(i % T);
  const double th = om[f] * (t0 + (double)(k0 + t) * (double)every - half);
  double s, c; sincos(th, &s, &c);
  W[i] = make_float2((float)c, (float)-s);
}
// Y[f][v] += sum_t W[f][t] X[t][v] for f < nf, v in [0, V) (row strides ldx, ldy): tiles of 32 f x 128 v, 32 t per step; each thread 4 f x 4 v
constexpr int FT = 32, VT = 128, KT = 32;
__global__ void __launch_bounds__(256) gemm_k(float2* Y, const float2* W, const float* X, int nf, size_t V, int T, size_t ldx, size_t ldy)
{
  __shared__ float2 sw[KT][FT];
  __shared__ __align__(16) float sx[KT][VT];
  const int tid = threadIdx.x, tf = tid / 32, tv = tid % 32;
  const int f0 = blockIdx.y * FT; const size_t v0 = (size_t)blockIdx.x * VT;
  float2 acc[4][4];
  for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) acc[a][b] = make_float2(0.f, 0.f);
  for (int k0 = 0; k0 < T; k0 += KT) {
    for (int i = tid; i < KT * FT; i += 256) { const int t = i % KT, f = i / KT; sw[t][f] = (f0 + f < nf && k0 + t < T) ? W[(size_t)(f0 + f) * T + k0 + t] : make_float2(0.f, 0.f); }
    for (int i = tid; i < KT * VT; i += 256) { const int t = i / VT, v = i % VT; sx[t][v] = (k0 + t < T && v0 + v < V) ? X[(size_t)(k0 + t) * ldx + v0 + v] : 0.f; }
    __syncthreads();
#pragma unroll 8
    for (int t = 0; t < KT; ++t) {
      const float4 xv = *reinterpret_cast<const float4*>(&sx[t][tv * 4]);
      const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        const float2 w = sw[t][tf * 4 + a];
#pragma unroll
        for (int b = 0; b < 4; ++b) { acc[a][b].x = fmaf(w.x, xs[b], acc[a][b].x); acc[a][b].y = fmaf(w.y, xs[b], acc[a][b].y); }
      }
    }
    __syncthreads();
  }
  for (int a = 0; a < 4; ++a) {
    const int f = f0 + tf * 4 + a; if (f >= nf) continue;
    for (int b = 0; b < 4; ++b) {
      const size_t v = v0 + tv * 4 + b; if (v >= V) continue;
      float2& y = Y[(size_t)f * ldy + v]; y.x += acc[a][b].x; y.y += acc[a][b].y;
    }
  }
}

struct Gpu {                    // device buffers, freed on any exit
  std::vector<void*> d; std::vector<void*> h;
  template <class T> T* dev(size_t n) { void* p = nullptr; AF_CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); d.push_back(p); return (T*)p; }
  template <class T> T* host(size_t n) { void* p = pinned_alloc(std::max<size_t>(n, 1) * sizeof(T)); if (!p) fail(AFDTD_ENOMEM, "pinned host memory"); h.push_back(p); return (T*)p; }
  ~Gpu() { for (void* p : d) cudaFree(p); for (void* p : h) pinned_free(p); }
};
struct Fd { int fd = -1; ~Fd() { if (fd >= 0) ::close(fd); } };
struct Joiner { std::thread t; ~Joiner() { if (t.joinable()) t.join(); } };   // a failure in the loop never leaves a thread joinable

void pread_all(int fd, char* dst, size_t n, off_t at, const char* path)
{
  while (n) {
    const ssize_t r = pread(fd, dst, n, at);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) fail(AFDTD_EIO, "%s: read failed at byte %lld: %s", path, (long long)at, r == 0 ? "end of file" : strerror(errno));
    dst += r; n -= (size_t)r; at += r;
  }
}
void pwrite_all(int fd, const char* src, size_t n, off_t at, const char* path)
{
  while (n) {
    const ssize_t w = pwrite(fd, src, n, at);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) fail(AFDTD_EIO, "%s: write failed: %s", path, strerror(errno));
    src += w; n -= (size_t)w; at += w;
  }
}
// a .npy (version 1.0) header for complex64 data of the given shape; returns its length (a multiple of 64)
std::string npy_header(const int64_t* shape, int nd)
{
  std::string d = "{'descr': '<c8', 'fortran_order': False, 'shape': (";
  for (int i = 0; i < nd; ++i) d += std::to_string(shape[i]) + ", ";
  d += "), }";
  size_t tot = 10 + d.size() + 1; const size_t pad = (64 - tot % 64) % 64;
  d += std::string(pad, ' ') + "\n";
  std::string h = "\x93NUMPY"; h += (char)1; h += (char)0;
  const uint16_t L = (uint16_t)d.size(); h += (char)(L & 0xFF); h += (char)(L >> 8);
  return h + d;
}

}  // namespace

void dft_file(const char* path, const double* omega, int nfreq, unsigned comps, int device, size_t gpu_bytes, float* out, const char* out_path)
{
  if (!path) fail(AFDTD_EINVAL, "path is NULL");
  if (nfreq < 1 || !omega) fail(AFDTD_EINVAL, "nfreq = %d (>= 1) and omega are required", nfreq);
  if (!comps || (comps & ~63u)) fail(AFDTD_EINVAL, "components = 0x%x: a non-empty mask of (1 << AFDTD_EX) ... (1 << AFDTD_HZ)", comps);
  if (!out == !(out_path && *out_path)) fail(AFDTD_EINVAL, "give exactly one of out and out_path");
  for (int f = 0; f < nfreq; ++f) if (!std::isfinite(omega[f])) fail(AFDTD_EINVAL, "omega[%d] is not finite", f);
  Fd in; in.fd = ::open(path, O_RDONLY);
  if (in.fd < 0) fail(AFDTD_EIO, "cannot open %s: %s", path, strerror(errno));
  const Hdr H = read_header(in.fd, path);
  int dc = 0; if (cudaGetDeviceCount(&dc) != cudaSuccess || device < 0 || device >= dc) { cudaGetLastError(); fail(AFDTD_EUNSUPPORTED, "no CUDA device %d", device); }
  AF_CK(cudaSetDevice(device));
  std::vector<int> cl; for (int c = 0; c < 6; ++c) if (comps & (1u << c)) cl.push_back(c);
  const int nc = (int)cl.size(); const int sx = (int)H.shape[0], sy = (int)H.shape[1], sz = (int)H.shape[2];
  const int64_t NT = H.frames;
  size_t fr = 0, tt = 0; AF_CK(cudaMemGetInfo(&fr, &tt));
  const size_t budget = gpu_bytes ? std::min(gpu_bytes, fr) : (size_t)(fr * 0.85);
  // chunks: of frame rows (native frames: each a contiguous range of every frame), or of x planes (canonical FP32 frames: one
  // contiguous range per component); Y (nfreq * values * 8) plus X and the raw data of a block of TB frames must fit
  const bool canon = H.dtype == AFDTD_F32;
  const size_t row_bytes = canon ? (size_t)sy * sz * 4 : (size_t)H.planes * H.pairs * 576;   // one frame row, or one x plane of one component
  const int64_t NROW = canon ? sx : H.rows;
  const int TB = (int)std::min<int64_t>(NT, 64);                            // frames per block
  auto chunk_ys = [&](int ra, int rb, int& ya, int& yb) {
    ya = std::max<int64_t>(0, (H.row0 + ra) * 24 - H.off[1]); yb = std::min<int64_t>(sy, (H.row0 + rb) * 24 - H.off[1]); };
  auto cost = [&](int r, int nf) {
    const size_t cells = canon ? (size_t)r * sy * sz : (size_t)sx * std::min<size_t>((size_t)sy, (size_t)r * 24) * sz;
    const size_t raw = canon ? (size_t)TB * nc * r * row_bytes : (size_t)TB * r * row_bytes;
    return (size_t)nf * nc * cells * 8 + (size_t)TB * nc * cells * 4 + 2 * raw + (size_t)nf * TB * 8 * 2; };
  // the file is read once per block of frequencies: all frequencies with chunks of frame rows when one row fits, else one row
  // per chunk and as many frequencies per block as fit
  int R = (int)NROW, NF = nfreq;
  while (R > 1 && cost(R, NF) > budget) R = (R + 1) / 2;
  while (NF > 1 && cost(R, NF) > budget) NF = (NF + 1) / 2;
  if (cost(R, NF) > budget) fail(AFDTD_ENOMEM, "the DFT of one frame row at one frequency needs %zu bytes of GPU memory, %zu are available", cost(R, NF), budget);
  std::vector<Chunk> ch;
  for (int ra = 0; ra < NROW; ra += R) {
    Chunk c; c.ra = ra; c.rb = std::min<int>((int)NROW, ra + R);
    if (canon) { c.xa = c.ra; c.xb = c.rb; c.ya = 0; c.yb = sy; ch.push_back(c); continue; }
    c.xa = 0; c.xb = sx; chunk_ys(c.ra, c.rb, c.ya, c.yb); if (c.yb > c.ya) ch.push_back(c);
  }
  // output
  Fd of; size_t data0 = 0;
  const int64_t oshape[5] = {nfreq, nc, sx, sy, sz};
  if (!out) {
    of.fd = ::open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (of.fd < 0) fail(AFDTD_EIO, "cannot open %s for writing: %s", out_path, strerror(errno));
    const std::string h = npy_header(oshape, 5); data0 = h.size();
    pwrite_all(of.fd, h.data(), h.size(), 0, out_path);
    if (ftruncate(of.fd, (off_t)(data0 + (size_t)nfreq * nc * sx * sy * sz * 8))) fail(AFDTD_EIO, "%s: cannot size the file: %s", out_path, strerror(errno));
  }
  Gpu G;
  int* dcomps = G.dev<int>(6); AF_CK(cudaMemcpy(dcomps, cl.data(), nc * sizeof(int), cudaMemcpyHostToDevice));
  double* dom = G.dev<double>(nfreq); AF_CK(cudaMemcpy(dom, omega, nfreq * sizeof(double), cudaMemcpyHostToDevice));
  const size_t rawcap = (size_t)TB * R * row_bytes * (canon ? nc : 1), Vcap = canon ? (size_t)nc * R * sy * sz : (size_t)nc * sx * std::min<size_t>(sy, (size_t)R * 24) * sz;
  char* hraw[2] = {G.host<char>(rawcap), G.host<char>(rawcap)};
  unsigned short* draw = G.dev<unsigned short>(canon ? 1 : rawcap / 2 + 1);
  float* dX = G.dev<float>((size_t)TB * Vcap);
  float2* dW[2] = {G.dev<float2>((size_t)NF * TB), G.dev<float2>((size_t)NF * TB)};
  float2* dY = G.dev<float2>((size_t)NF * Vcap);
  float2* hY = G.host<float2>((size_t)NF * Vcap);
  for (int fa = 0; fa < nfreq; fa += NF)
  for (const Chunk& c : ch) {
    const int nfb = std::min(NF, nfreq - fa);
    const int ny = c.yb - c.ya, nxc = c.xb - c.xa; const size_t cells = (size_t)nxc * ny * sz, V = (size_t)nc * cells;
    const size_t cb = canon ? (size_t)nc * nxc * row_bytes : (size_t)(c.rb - c.ra) * row_bytes;     // bytes per frame of the chunk
    AF_CK(cudaMemset(dY, 0, (size_t)nfb * V * sizeof(float2)));
    // block b of frames is read into hraw[b & 1] by a thread while the GPU works on block b - 1
    const int64_t NB = (NT + TB - 1) / TB;
    auto read_block = [&](int64_t b, char* dst) {
      const int64_t k0 = b * TB, k1 = std::min<int64_t>(NT, k0 + TB);
      const int nth = (int)std::min<int64_t>(8, k1 - k0); std::vector<std::thread> th; std::vector<std::string> er((size_t)nth);
      for (int i = 0; i < nth; ++i) th.emplace_back([&, i] {
        try {
          for (int64_t k = k0 + i; k < k1; k += nth) {
            const off_t f0 = (off_t)(H.data_offset + k * H.stride);
            if (!canon) { pread_all(in.fd, dst + (size_t)(k - k0) * cb, cb, f0 + (off_t)((int64_t)c.ra * row_bytes), path); continue; }
            for (int ci = 0; ci < nc; ++ci)                  // component cl[ci], x planes [xa, xb): contiguous; lands as X[t][ci][x][y][z]
              pread_all(in.fd, dst + (size_t)(k - k0) * cb + (size_t)ci * nxc * row_bytes, (size_t)nxc * row_bytes,
                        f0 + (off_t)(((size_t)cl[ci] * sx + c.xa) * row_bytes), path);
          }
        } catch (const Error& e) { er[(size_t)i] = e.what(); } });
      for (auto& t : th) t.join();
      for (auto& e : er) if (!e.empty()) fail(AFDTD_EIO, "%s", e.c_str());
    };
    read_block(0, hraw[0]);
    for (int64_t b = 0; b < NB; ++b) {
      const int64_t k0 = b * TB; const int T = (int)std::min<int64_t>(TB, NT - k0);
      if (canon) AF_CK(cudaMemcpy(dX, hraw[b & 1], (size_t)T * cb, cudaMemcpyHostToDevice));    // already X[t][v]
      else AF_CK(cudaMemcpy(draw, hraw[b & 1], (size_t)T * cb, cudaMemcpyHostToDevice));
      std::string nerr; Joiner next;                  // next joins before nerr is destroyed
      if (b + 1 < NB) next.t = std::thread([&] { try { read_block(b + 1, hraw[(b + 1) & 1]); } catch (const Error& e) { nerr = e.what(); } });
      if (!canon)
        decode_k<<<(unsigned)((V * T + 255) / 256), 256>>>(draw, dX, T, cb, nc, dcomps, sx, ny, sz, (int)H.off[0], (int)H.off[1] + c.ya, (int)H.off[2],
                                                            c.ra, (int)H.row0, (int)H.planes, (int)H.pairs, (int)H.pair0, (int)H.z0, H.dtype);
      for (int e = 0; e < 2; ++e) phase_k<<<(unsigned)(((size_t)nfb * T + 255) / 256), 256>>>(dW[e], dom + fa, nfb, T, (double)H.first_step, k0, H.every, e ? 0.5 : 0.0);
      for (int ci = 0; ci < nc; ++ci) {
        dim3 grid((unsigned)((cells + VT - 1) / VT), (unsigned)((nfb + FT - 1) / FT));
        gemm_k<<<grid, 256>>>(dY + (size_t)ci * cells, dW[cl[ci] >= 3], dX + (size_t)ci * cells, nfb, cells, T, V, V);
      }
      AF_CK(cudaGetLastError());
      AF_CK(cudaDeviceSynchronize());
      if (next.t.joinable()) next.t.join();
      if (!nerr.empty()) fail(AFDTD_EIO, "%s", nerr.c_str());
    }
    AF_CK(cudaMemcpy(hY, dY, (size_t)nfb * V * sizeof(float2), cudaMemcpyDeviceToHost));
    // scatter: piece (f, ci, x) of ny * sz values to out[(((f * nc + ci) * sx + x) * sy + ya) * sz] (one piece per (f, ci) when the
    // chunk is the whole box in y); in parallel: buffered writes at scattered offsets are bound by one thread's copying otherwise
    const bool whole = canon || ny == sy;                                   // one piece per (f, ci)
    const size_t npiece = (size_t)nfb * nc * (whole ? 1 : sx);
    std::vector<std::string> werr(16);
    const unsigned nth = (unsigned)std::min<size_t>(16, npiece);
    { std::vector<std::thread> th;
      for (unsigned w = 0; w < nth; ++w) th.emplace_back([&, w] {
        try {
          for (size_t k = w; k < npiece; k += nth) {
            const size_t x = whole ? 0 : k % sx, fc = whole ? k : k / sx;
            const int fl = (int)(fc / nc), ci = (int)(fc % nc), f = fa + fl;
            const size_t n = whole ? cells : (size_t)ny * sz;
            const float2* src = hY + (size_t)fl * V + (size_t)ci * cells + x * ny * sz;
            const size_t at = ((((size_t)f * nc + ci) * sx + x + c.xa) * sy + (whole ? 0 : c.ya)) * (size_t)sz;
            if (out) memcpy(out + 2 * at, src, n * sizeof(float2)); else pwrite_all(of.fd, (const char*)src, n * sizeof(float2), (off_t)(data0 + at * 8), out_path);
          }
        } catch (const Error& e) { werr[w] = e.what(); } });
      for (auto& t : th) t.join(); }
    for (auto& e : werr) if (!e.empty()) fail(AFDTD_EIO, "%s", e.c_str());
  }
}

}  // namespace afdtd
