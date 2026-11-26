// SPDX-License-Identifier: MIT
// inject.cu -- injectors (alpha_fdtd.h section 8) against the oracle tests/oracle/reference.cu, bit for bit.
//
//   afdtd_inject_test NX NY NZ SPLITS FIRST ROWS M BOX [gpus=N] [seed=S] [rec=EVERY] [neg=skip] [prec=P]
//
// BOX is x0,y0,z0,sx,sy,sz.  Random lossy state (as exact.cu), a random basis of M fields over the box, random weights with every third
// row all zero (skipped).  The oracle runs the reference steps and, after each step t = FIRST + 7 k with a live row, applies the rule of
// section 8 with its own kernel.  Compared: every value of the final state; with rec=EVERY also every frame of a recorder on the box.
//   neg=skip   negative control: the oracle leaves out the last live injection; passes only if mismatches are found
#include <alpha_fdtd.h>
#include "reference.cu"
#include "prec.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"passed\": false, \"error\": \"CUDA %s line %d\"}\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
static bool on_wall(int c, int64_t x, int64_t y, int64_t z) { const int a = c % 3; return (x == 0 && a != 0) || (y == 0 && a != 1) || (z == 0 && a != 2); }

struct OBox { int x0, y0, z0, sx, sy, sz; };
__device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float tof(__half v) { return __half2float(v); }
__device__ __forceinline__ float tof(float v) { return v; }
template <class T> __device__ __forceinline__ T fromf(float v);
template <> __device__ __forceinline__ __nv_bfloat16 fromf<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __half fromf<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float fromf<float>(float v) { return v; }
// the section 8 rule on the oracle's arrays: one thread per cell of the box
template <class T> __global__ void ref_inject(RefFieldsT<T> F, RefGrid g, OBox b, const float* G, const float* w, int M)
{
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, cells = (size_t)b.sx * b.sy * b.sz;
  if (i >= cells) return;
  const int z = b.z0 + (int)(i % b.sz), y = b.y0 + (int)(i / b.sz % b.sy), x = b.x0 + (int)(i / ((size_t)b.sz * b.sy));
  const size_t idx = ((size_t)x * g.ny + y) * g.nz + z;
  for (int c = 0; c < 6; ++c) {
    if (c < 3 && ((x == 0 && c != 0) || (y == 0 && c != 1) || (z == 0 && c != 2))) continue;
    float acc = 0.f;
    for (int j = 0; j < M; ++j) acc = __fadd_rn(acc, __fmul_rn(w[j], G[((size_t)j * 6 + c) * cells + i]));
    F.f[c][idx] = fromf<T>(__fadd_rn(tof(F.f[c][idx]), acc));
  }
}

struct Frames { std::map<int64_t, std::vector<uint16_t>> f; std::mutex m; };
static int on_frame(void* u, int64_t step, const void* d, size_t n)
{ Frames* F = (Frames*)u; std::lock_guard<std::mutex> l(F->m); F->f[step].assign((const uint16_t*)d, (const uint16_t*)d + n / 2); return 0; }

template <class U, class T> int run(int argc, char** argv, const Prec& P);
int main(int argc, char** argv)
{
  std::string prec = "bf16";
  for (int i = 9; i < argc; ++i) if (!strncmp(argv[i], "prec=", 5)) prec = argv[i] + 5;
  const Prec P = prec_of(prec);
  if (P.bits == 32) return run<uint32_t, float>(argc, argv, P);
  if (P.dtype == AFDTD_F16) return run<uint16_t, __half>(argc, argv, P);
  return run<uint16_t, __nv_bfloat16>(argc, argv, P);
}
template <class U, class T> int run(int argc, char** argv, const Prec& P)
{
  if (argc < 9) { fprintf(stderr, "usage: afdtd_inject_test NX NY NZ SPLITS FIRST ROWS M BOX [gpus=N] [seed=S] [rec=EVERY] [neg=skip]\n"); return 2; }
  const int64_t nx = atoll(argv[1]), ny = atoll(argv[2]), nz = atoll(argv[3]), first = atoll(argv[5]), rows = atoll(argv[6]);
  const int M = atoi(argv[7]); const std::string splits = argv[4];
  OBox bx; if (sscanf(argv[8], "%d,%d,%d,%d,%d,%d", &bx.x0, &bx.y0, &bx.z0, &bx.sx, &bx.sy, &bx.sz) != 6) { fprintf(stderr, "bad box\n"); return 2; }
  int ngpus = 1; uint64_t seed = 11; int64_t every = 0; std::string neg;
  for (int i = 9; i < argc; ++i) {
    const std::string a = argv[i];
    if (!a.compare(0, 5, "gpus=")) ngpus = atoi(a.c_str() + 5);
    else if (!a.compare(0, 5, "seed=")) seed = strtoull(a.c_str() + 5, nullptr, 10);
    else if (!a.compare(0, 4, "rec=")) every = atoll(a.c_str() + 4);
    else if (!a.compare(0, 4, "neg=")) neg = a.substr(4);
    else if (!a.compare(0, 5, "prec=")) continue;
    else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  std::string name = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + " steps " + splits + " first " + std::to_string(first) +
                     " rows " + std::to_string(rows) + " M " + std::to_string(M) + " box " + argv[8] + " gpus " + std::to_string(ngpus) +
                     (every ? " rec " + std::to_string(every) : "") + (neg.empty() ? "" : " neg=" + neg) + (P.name != "bf16" ? " " + P.name : "");
  int have = 0;
  if (cudaGetDeviceCount(&have) != cudaSuccess || have < ngpus) { printf("{\"test\": \"%s\", \"skipped\": \"%d GPU(s) visible\"}\n", name.c_str(), have); return 77; }
  std::vector<int64_t> sp; int64_t total = 0;
  for (size_t p = 0; p < splits.size();) { size_t q = splits.find(',', p); if (q == std::string::npos) q = splits.size(); sp.push_back(atoll(splits.substr(p, q - p).c_str())); total += sp.back(); p = q + 1; }

  const size_t n = (size_t)nx * ny * nz, cells = (size_t)bx.sx * bx.sy * bx.sz;
  std::vector<U> in(12 * n), ref, got(12 * n);
  std::mt19937_64 rng(seed);
  auto u01 = [&] { return (double)(rng() >> 11) / 9007199254740992.0; };
  for (int c = 0; c < 12; ++c) for (size_t i = 0; i < n; ++i)
    in[c * n + i] = (U)quantize(P, c < 6 ? (float)(2 * u01() - 1) : c < 9 ? (float)(0.6 + 0.4 * u01()) : (float)(0.3 * u01()));
  ref = in;
  for (int c = 0; c < 12; ++c) {
    if (c >= 3 && c < 6) continue;
    for (size_t i = 0; i < n; ++i) if (on_wall(c, (int64_t)(i / ((size_t)ny * nz)), (int64_t)(i / nz % ny), (int64_t)(i % nz))) ref[c * n + i] = 0;
  }
  std::vector<float> G((size_t)M * 6 * cells), W((size_t)rows * M);
  for (auto& v : G) v = (float)(0.2 * u01() - 0.1);
  for (int64_t k = 0; k < rows; ++k) for (int j = 0; j < M; ++j) W[(size_t)k * M + j] = k % 3 == 2 ? 0.f : (float)(2 * u01() - 1);

  afdtd_options o; memset(&o, 0, sizeof o); o.ngpus = ngpus; o.flags = P.flag;
  std::string err;
  afdtd_plan p = afdtd_plan_create(nx, ny, nz, &o);
  if (!p) { printf("{\"test\": \"%s\", \"passed\": false, \"error\": \"%s\"}\n", name.c_str(), afdtd_last_error()); return 1; }
  afdtd_box b; memset(&b, 0, sizeof b);
  b.shape[0] = nx; b.shape[1] = ny; b.shape[2] = nz;
  for (int c = 0; c < 12; ++c) { b.array[c].data = in.data() + c * n; b.array[c].dtype = P.dtype; }
  if (afdtd_set(p, &b)) err = std::string("set: ") + afdtd_last_error();
  afdtd_injector_options io; memset(&io, 0, sizeof io);
  io.offset[0] = bx.x0; io.offset[1] = bx.y0; io.offset[2] = bx.z0; io.shape[0] = bx.sx; io.shape[1] = bx.sy; io.shape[2] = bx.sz;
  io.nbasis = M; io.basis = G.data(); io.first_step = first; io.rows = rows; io.weights = W.data();
  int iid = -1; if (err.empty() && (iid = afdtd_inject(p, &io)) < 0) err = std::string("inject: ") + afdtd_last_error();
  Frames fr; int rid = -1; afdtd_recorder_info ri; memset(&ri, 0, sizeof ri);
  if (every && err.empty()) {
    afdtd_recorder_options ro; memset(&ro, 0, sizeof ro);
    for (int a = 0; a < 3; ++a) { ro.offset[a] = io.offset[a]; ro.shape[a] = io.shape[a]; }
    ro.every = every; ro.callback = on_frame; ro.user = &fr;
    if ((rid = afdtd_record(p, &ro)) < 0) err = std::string("record: ") + afdtd_last_error();
  }
  for (int64_t s : sp) if (err.empty() && afdtd_execute(p, s)) err = std::string("execute: ") + afdtd_last_error();
  if (rid >= 0 && err.empty() && (afdtd_record_info(p, rid, &ri) || afdtd_record_close(p, rid))) err = std::string("record close: ") + afdtd_last_error();
  if (iid >= 0 && err.empty() && afdtd_inject_close(p, iid)) err = std::string("inject close: ") + afdtd_last_error();
  for (int c = 0; c < 12; ++c) { b.array[c].data = got.data() + c * n; b.array[c].dtype = P.dtype; }
  if (err.empty() && afdtd_get(p, &b)) err = std::string("get: ") + afdtd_last_error();
  afdtd_plan_destroy(p);

  // oracle
  RefFieldsT<T> F; RefGrid g = {(int)nx, (int)ny, (int)nz};
  CK(cudaSetDevice(0));
  for (int c = 0; c < 12; ++c) { CK(cudaMalloc(&F.f[c], n * sizeof(U))); CK(cudaMemcpy(F.f[c], ref.data() + c * n, n * sizeof(U), cudaMemcpyHostToDevice)); }
  float *dG = nullptr, *dw = nullptr; CK(cudaMalloc(&dG, G.size() * 4)); CK(cudaMemcpy(dG, G.data(), G.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMalloc(&dw, (size_t)M * 4));
  int64_t last_live = -1; for (int64_t k = 0; k < rows; ++k) if (k % 3 != 2 && first + 7 * k <= total) last_live = first + 7 * k;
  dim3 gr, bl; reference_dims(g, gr, bl);
  uint64_t fmism = 0, fcomp = 0, applied = 0; std::vector<U> st(6 * n);
  for (int64_t t = 0; t <= total; ++t) {
    if (t > 0) { reference_step_h_t<T><<<gr, bl>>>(F, g); reference_step_e_t<T><<<gr, bl>>>(F, g); }
    if (t >= first && (t - first) % 7 == 0 && (t - first) / 7 < rows && ((t - first) / 7) % 3 != 2 && !(neg == "skip" && t == last_live)) {
      CK(cudaMemcpy(dw, W.data() + (size_t)((t - first) / 7) * M, (size_t)M * 4, cudaMemcpyHostToDevice));
      ref_inject<T><<<(unsigned)((cells + 255) / 256), 256>>>(F, g, bx, dG, dw, M); ++applied;
    }
    if (every && t % every == 0) {
      CK(cudaDeviceSynchronize());
      for (int c = 0; c < 6; ++c) CK(cudaMemcpy(st.data() + c * n, F.f[c], n * sizeof(U), cudaMemcpyDeviceToHost));
      auto it = fr.f.find(t);
      if (it == fr.f.end()) { ++fmism; continue; }
      for (int c = 0; c < 6; ++c) for (int x = bx.x0; x < bx.x0 + bx.sx; ++x) for (int y = bx.y0; y < bx.y0 + bx.sy; ++y) for (int z = bx.z0; z < bx.z0 + bx.sz; ++z) {
        uint32_t have;
        if (ri.dtype == AFDTD_F32) { const size_t q = (((size_t)c * bx.sx + (x - bx.x0)) * bx.sy + (y - bx.y0)) * bx.sz + (z - bx.z0);
                                     memcpy(&have, reinterpret_cast<const char*>(it->second.data()) + 4 * q, 4); }
        else have = it->second[(size_t)(((((y / 24 - ri.row0) * ri.planes + (z - ri.z0)) * ri.pairs + (x / 2 - ri.pair0)) * 24 + y % 24) * 12 + 2 * c + (x & 1))];
        ++fcomp; fmism += have != (uint32_t)st[c * n + ((size_t)x * ny + y) * nz + z];
      }
    }
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  for (int c = 0; c < 6; ++c) CK(cudaMemcpy(st.data() + c * n, F.f[c], n * sizeof(U), cudaMemcpyDeviceToHost));
  uint64_t mism = 0; for (int c = 0; c < 6; ++c) for (size_t i = 0; i < n; ++i) mism += got[c * n + i] != st[c * n + i];
  for (int c = 0; c < 12; ++c) CK(cudaFree(F.f[c]));
  bool passed = err.empty() && applied > 0;
  passed = neg == "skip" ? passed && mism > 0 : passed && !mism && !fmism;
  for (char& q : err) if (q == '"' || q == '\\') q = '\'';
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("{\"test\": \"%s\", \"passed\": %s, \"injections\": %llu, \"mismatches\": %llu, \"frame_values\": %llu, \"frame_mismatches\": %llu, \"gpu\": \"%s\", "
         "\"error\": \"%s\"}\n", name.c_str(), passed ? "true" : "false", (unsigned long long)applied, (unsigned long long)mism, (unsigned long long)fcomp,
         (unsigned long long)fmism, pr.name, err.c_str());
  return passed ? 0 : 1;
}
