// SPDX-License-Identifier: MIT
// boxio.cpp -- caller boxes <-> the state: element types and strides, the lower-wall rule, the host-resident domain (allocation,
// row layout, checksums).  Host code only; the GPU side of a box is Object::box.
#include "alpha_fdtd.h"
#include "internal.hpp"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

namespace afdtd {

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& f)
{
  const int nt = (int)std::max<int64_t>(1, std::min<int64_t>({n, (int64_t)std::max(1u, std::thread::hardware_concurrency()), 64}));
  if (nt == 1) { if (n > 0) f(0, n); return; }
  std::vector<std::thread> th; std::exception_ptr ep = nullptr; std::mutex m;
  for (int t = 0; t < nt; ++t) th.emplace_back([&, t] {
    try { f(n * t / nt, n * (t + 1) / nt); } catch (...) { std::lock_guard<std::mutex> l(m); if (!ep) ep = std::current_exception(); } });
  for (auto& t : th) t.join();
  if (ep) std::rethrow_exception(ep);
}

// ------------------------------------------------------------------------------------------------ element conversion
static inline uint16_t f32_to_bf16(uint32_t u)       // round to nearest, ties to even; NaN stays NaN (quiet)
{
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x0040u);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static inline bool on_wall(int c, int64_t x, int64_t y, int64_t z)   // tangential E, Ca, Cb on a lower wall (alpha_fdtd.h section 2)
{
  if (c >= 3 && c < 6) return false;
  const int a = c % 3;
  return (x == 0 && a != 0) || (y == 0 && a != 1) || (z == 0 && a != 2);
}
static inline int64_t elem(const BoxSpec& b, int c, int64_t i, int64_t j, int64_t k)
{
  if (b.flags & AFDTD_BOX_STRIDED) return i * b.a[c].stride[0] + j * b.a[c].stride[1] + k * b.a[c].stride[2];
  return (i * b.shape[1] + j) * b.shape[2] + k;
}
static inline uint16_t read_bits(const BoxSpec& b, int c, int64_t e)
{
  if (b.a[c].dtype == AFDTD_BF16) return static_cast<const uint16_t*>(b.a[c].data)[e];
  uint32_t u; memcpy(&u, static_cast<const char*>(b.a[c].data) + 4 * e, 4); return f32_to_bf16(u);
}
static inline void write_bits(const BoxSpec& b, int c, int64_t e, uint16_t v)
{
  if (b.a[c].dtype == AFDTD_BF16) static_cast<uint16_t*>(b.a[c].data)[e] = v;
  else { const uint32_t u = (uint32_t)v << 16; memcpy(static_cast<char*>(b.a[c].data) + 4 * e, &u, 4); }
}

// ------------------------------------------------------------------------------------------------ host-resident domain
void host_alloc(HostDomain& h)
{
  const size_t tot = h.bytes;
  char* m = (char*)mmap(nullptr, tot, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (m == MAP_FAILED) fail(AFDTD_ENOMEM, "host memory: mmap of %zu bytes failed", tot);
  madvise(m, tot, MADV_HUGEPAGE);
  h.rows = m;
  h.zero = (char*)mmap(nullptr, h.RSB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (h.zero == MAP_FAILED) { h.zero = nullptr; fail(AFDTD_ENOMEM, "host memory: mmap failed"); }
  memset(h.zero, 0, h.RSB);
  parallel_for((int64_t)(tot / 4096), [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; ++i) m[i * 4096] = 0; });   // first touch, in parallel
  std::atomic<int> next(0), err(0);
  const int NJ = h.G.NJ;
  std::vector<std::thread> th;
  for (int t = 0; t < 16; ++t) th.emplace_back([&, t] {
    cudaSetDevice(h.devs[t % h.devs.size()]);
    for (int i; (i = next++) <= NJ && !err;)
      if ((i < NJ ? cudaHostRegister(h.rows + (size_t)i * h.HRS, h.HRS, cudaHostRegisterPortable) : cudaHostRegister(h.zero, h.RSB, cudaHostRegisterPortable)) != cudaSuccess) err = i + 1;
  });
  for (auto& t : th) t.join();
  if (err) { cudaGetLastError(); fail(AFDTD_ENOMEM, "host memory: pinning %zu bytes failed (cudaHostRegister)", tot); }
}
void host_free(HostDomain& h)
{
  if (h.rows) {
    parallel_for(h.G.NJ, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; ++i) cudaHostUnregister(h.rows + (size_t)i * h.HRS); });
    munmap(h.rows, h.bytes); h.rows = nullptr;
  }
  if (h.zero) { cudaHostUnregister(h.zero); munmap(h.zero, h.RSB); h.zero = nullptr; }
  cudaGetLastError();
}
// the 16-bit half of cell (x, y, z), component c, in the host rows (lane: y % 24, or 24..29 for the repeated coefficient lanes)
static inline uint16_t* host_at(const HostDomain& h, int c, int64_t x, int J, int lane, int64_t z)
{
  const Geo& G = h.G;
  const size_t zi = (size_t)(z + kSOFF), gp = (size_t)(x / 2 + kGOFF);
  char* row = h.rows + (size_t)J * h.HRS;
  if (c < 6) return reinterpret_cast<uint16_t*>(row + ((zi * G.PLREC + gp * kS + lane) * 6 + c) * 4) + (x & 1);
  return reinterpret_cast<uint16_t*>(row + h.RSB + zi * G.PS + (gp * 30 + lane) * kCREC + (c - 6) * 4) + (x & 1);
}
static inline uint64_t mix64(uint64_t x)
{
  x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
void host_checksums(const HostDomain& h, uint64_t out[6])
{
  const Geo& G = h.G; const uint64_t n = (uint64_t)G.nx * G.ny * G.nz;
  std::vector<uint64_t> s((size_t)G.nx * 6, 0);
  parallel_for(G.nx, [&](int64_t a, int64_t b) {
    for (int64_t x = a; x < b; ++x)
      for (int64_t y = 0; y < G.ny; ++y)
        for (int64_t z = 0; z < G.nz; ++z) {
          const uint64_t idx = ((uint64_t)x * G.ny + y) * G.nz + z;
          for (int c = 0; c < 6; ++c) s[x * 6 + c] += mix64(((((uint64_t)(c % 3)) * n + idx) << 32) + *host_at(h, c, x, (int)(y / kS), (int)(y % kS), z));
        }
  });
  for (int c = 0; c < 6; ++c) { out[c] = 0; for (int64_t x = 0; x < G.nx; ++x) out[c] += s[x * 6 + c]; }
}

// ------------------------------------------------------------------------------------------------ boxes
static const char* kName[12] = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Cax", "Cay", "Caz", "Cbx", "Cby", "Cbz"};

void box_io(Plan& p, const BoxSpec& b, int dir)
{
  // ---- checks: nothing is written before every component passed
  int nc = 0;
  for (int a = 0; a < 3; ++a)
    if (b.shape[a] < 1 || b.off[a] < 0 || b.off[a] > p.n[a] - b.shape[a])
      fail(AFDTD_EINVAL, "box offset (%lld, %lld, %lld) shape (%lld, %lld, %lld) is not inside the %lld x %lld x %lld domain", (long long)b.off[0], (long long)b.off[1],
           (long long)b.off[2], (long long)b.shape[0], (long long)b.shape[1], (long long)b.shape[2], (long long)p.n[0], (long long)p.n[1], (long long)p.n[2]);
  if (b.flags & ~AFDTD_BOX_STRIDED) fail(AFDTD_EINVAL, "unknown box flag bits 0x%x", b.flags & ~AFDTD_BOX_STRIDED);
  for (int c = 0; c < 12; ++c) {
    if (!b.a[c].data) continue;
    ++nc;
    if (b.a[c].dtype != AFDTD_BF16 && b.a[c].dtype != AFDTD_F32) fail(AFDTD_EINVAL, "%s: dtype %d is not AFDTD_BF16 (1) or AFDTD_F32 (2)", kName[c], b.a[c].dtype);
    if (b.flags & AFDTD_BOX_STRIDED)
      for (int a = 0; a < 3; ++a) {
        if (b.a[c].stride[a] < 0) fail(AFDTD_EINVAL, "%s: negative stride %lld", kName[c], (long long)b.a[c].stride[a]);
        if (dir == 1 && b.a[c].stride[a] == 0 && b.shape[a] > 1) fail(AFDTD_EINVAL, "%s: zero stride on get (the writes would overlap)", kName[c]);
      }
  }
  if (!nc) fail(AFDTD_EINVAL, "the box gives no array (every data pointer is NULL)");
  const int64_t sx = b.shape[0], sy = b.shape[1], sz = b.shape[2], x0 = b.off[0], y0 = b.off[1], z0 = b.off[2];
  // ---- lower walls: count (and under STRICT_WALLS refuse) before anything is written
  uint64_t wc = 0, wf = 0; int64_t first[4] = {-1, 0, 0, 0};
  if (dir == 0 && (x0 == 0 || y0 == 0 || z0 == 0))
    for (int c = 0; c < 12; ++c) {
      if (!b.a[c].data || (c >= 3 && c < 6)) continue;
      for (int w = 0; w < 3; ++w) {
        if (b.off[w] != 0 || w == c % 3) continue;
        const int64_t hi[3] = {w == 0 ? 1 : sx, w == 1 ? 1 : sy, w == 2 ? 1 : sz};
        for (int64_t i = 0; i < hi[0]; ++i) for (int64_t j = 0; j < hi[1]; ++j) for (int64_t k = 0; k < hi[2]; ++k) {
          const int64_t pos[3] = {x0 + i, y0 + j, z0 + k};
          bool earlier = false;                        // counted on a wall w2 < w already
          for (int w2 = 0; w2 < w; ++w2) if (w2 != c % 3 && pos[w2] == 0) earlier = true;
          if (earlier) continue;
          const uint16_t v = read_bits(b, c, elem(b, c, i, j, k));
          if (c >= 6) { wc += v != 0; continue; }
          if ((v & 0x7FFFu) == 0) continue;
          if (!wf++) { first[0] = c; first[1] = pos[0]; first[2] = pos[1]; first[3] = pos[2]; }
        }
      }
    }
  if (wf && (p.flags & AFDTD_STRICT_WALLS)) {
    const uint16_t v = read_bits(b, (int)first[0], elem(b, (int)first[0], first[1] - x0, first[2] - y0, first[3] - z0));
    uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4);
    fail(AFDTD_EWALLS, "AFDTD_STRICT_WALLS: %llu E value(s) on the lower walls are not 0 (first: %s = %g at (%lld, %lld, %lld)); index 0 is the PEC wall, "
         "put fields at index >= 1; nothing was written", (unsigned long long)wf, kName[first[0]], f, (long long)first[1], (long long)first[2], (long long)first[3]);
  }
  // ---- copy in slabs of x planes through a dense BF16 staging buffer
  const size_t plane = (size_t)sy * sz;
  const int64_t xs = std::max<int64_t>(1, (int64_t)(((size_t)256 << 20) / (2 * plane * nc)));
  std::vector<uint16_t> stage((size_t)std::min(xs, sx) * plane * nc);
  for (int64_t xa = 0; xa < sx; xa += xs) {
    const int64_t xb = std::min(sx, xa + xs); const size_t cnt = (size_t)(xb - xa) * plane;
    uint16_t* comp[12]; int i = 0;
    for (int c = 0; c < 12; ++c) comp[c] = b.a[c].data ? stage.data() + cnt * i++ : nullptr;
    const int64_t off[3] = {x0 + xa, y0, z0}, shp[3] = {xb - xa, sy, sz};
    if (dir == 0)
      parallel_for(xb - xa, [&](int64_t a, int64_t e) {
        for (int c = 0; c < 12; ++c) if (comp[c])
          for (int64_t ii = a; ii < e; ++ii) for (int64_t j = 0; j < sy; ++j) {
            uint16_t* d = comp[c] + ((size_t)ii * sy + j) * sz;
            for (int64_t k = 0; k < sz; ++k) d[k] = on_wall(c, x0 + xa + ii, y0 + j, z0 + k) ? 0 : read_bits(b, c, elem(b, c, xa + ii, j, k));
          }
      });
    if (!p.on_host) p.obj->box(p.mem, off, shp, comp, dir);
    else
      parallel_for(xb - xa, [&](int64_t a, int64_t e) {
        for (int c = 0; c < 12; ++c) if (comp[c])
          for (int64_t ii = a; ii < e; ++ii) for (int64_t j = 0; j < sy; ++j) {
            const int64_t x = off[0] + ii, y = y0 + j; const int J = (int)(y / kS), l = (int)(y % kS);
            uint16_t* s = comp[c] + ((size_t)ii * sy + j) * sz;
            for (int64_t k = 0; k < sz; ++k) {
              uint16_t* t = host_at(p.host, c, x, J, l, z0 + k);
              if (dir == 1) { s[k] = *t; continue; }
              *t = s[k];
              if (c >= 6 && J > 0 && l + kS < 30) *host_at(p.host, c, x, J - 1, l + kS, z0 + k) = s[k];      // repeated lanes of row J - 1
            }
          }
      });
    if (dir == 1)
      parallel_for(xb - xa, [&](int64_t a, int64_t e) {
        for (int c = 0; c < 12; ++c) if (comp[c])
          for (int64_t ii = a; ii < e; ++ii) for (int64_t j = 0; j < sy; ++j) {
            const uint16_t* s = comp[c] + ((size_t)ii * sy + j) * sz;
            for (int64_t k = 0; k < sz; ++k) write_bits(b, c, elem(b, c, xa + ii, j, k), s[k]);
          }
      });
  }
  if (dir == 0) { p.wz_coef = wc; p.wz_field = wf; }
}

}  // namespace afdtd
