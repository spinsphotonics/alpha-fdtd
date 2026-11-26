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
uint16_t f32_to_f16(uint32_t u)                       // IEEE half: round to nearest, ties to even; overflow to inf; NaN quiet
{
  const uint32_t s = (u >> 16) & 0x8000u, a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return (uint16_t)(s | 0x7E00u | ((a >> 13) & 0x3FFu));
  if (a >= 0x477FF000u) return (uint16_t)(s | 0x7C00u);                  // >= 65520 (the tie above 65504 rounds to even: inf)
  if (a >= 0x38800000u) {                                                  // normal half
    const uint32_t r = a - 0x38000000u, rem = r & 0x1FFFu; uint32_t m = r >> 13;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) ++m;
    return (uint16_t)(s | m);
  }
  if (a < 0x33000000u) return (uint16_t)s;                                // below 2^-25: +-0
  const uint32_t e = a >> 23, mant = (a & 0x7FFFFFu) | 0x800000u, sh = 126 - e;   // subnormal half: mant * 2^(e - 150) / 2^-24
  uint32_t m = mant >> sh; const uint32_t rem = mant & ((1u << sh) - 1u), half = 1u << (sh - 1);
  if (rem > half || (rem == half && (m & 1u))) ++m;
  return (uint16_t)(s | m);
}
uint32_t f16_to_f32(uint16_t h)                        // exact
{
  const uint32_t s = (uint32_t)(h & 0x8000u) << 16; uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  if (e == 0x1F) return s | 0x7F800000u | (m << 13);
  if (e == 0) { if (!m) return s; int k = 0; while (!(m & 0x400u)) { m <<= 1; ++k; } return s | ((uint32_t)(113 - k) << 23) | ((m & 0x3FFu) << 13); }
  return s | ((e + 112) << 23) | (m << 13);
}
// the state's bit pattern (16 or 32 bits) from a caller element, and back
static inline uint32_t to_state(int prec, int dtype, const void* data, int64_t e)
{
  uint32_t f;                                          // the value as float32 bits
  if (dtype == AFDTD_F32) memcpy(&f, static_cast<const char*>(data) + 4 * e, 4);
  else {
    const uint16_t v = static_cast<const uint16_t*>(data)[e];
    if (dtype == prec) return v;
    f = dtype == AFDTD_BF16 ? (uint32_t)v << 16 : f16_to_f32(v);
  }
  return prec == AFDTD_F32 ? f : prec == AFDTD_BF16 ? f32_to_bf16(f) : f32_to_f16(f);
}
static inline void from_state(int prec, int dtype, void* data, int64_t e, uint32_t v)
{
  if (dtype == prec && dtype != AFDTD_F32) { static_cast<uint16_t*>(data)[e] = (uint16_t)v; return; }
  const uint32_t f = prec == AFDTD_F32 ? v : prec == AFDTD_BF16 ? v << 16 : f16_to_f32((uint16_t)v);
  if (dtype == AFDTD_F32) memcpy(static_cast<char*>(data) + 4 * e, &f, 4);
  else static_cast<uint16_t*>(data)[e] = dtype == AFDTD_BF16 ? f32_to_bf16(f) : f32_to_f16(f);
}
static inline float state_float(int prec, uint32_t v)
{ const uint32_t f = prec == AFDTD_F32 ? v : prec == AFDTD_BF16 ? v << 16 : f16_to_f32((uint16_t)v); float x; memcpy(&x, &f, 4); return x; }
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


// ------------------------------------------------------------------------------------------------ host-resident domain
// Pinned host memory for copies at the full host-link rate: 2 MiB-aligned anonymous memory with transparent huge pages,
// touched, then registered, as for a host-memory domain.
void* pinned_alloc(size_t bytes)
{
  const size_t A = (size_t)2 << 20, len = rup(bytes, A) + A;
  char* m = (char*)mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (m == MAP_FAILED) return nullptr;
  char* p = (char*)(((uintptr_t)m + sizeof(size_t) * 2 + A - 1) / A * A);     // room before p for the mapping's start and length
  if (p - m < (ptrdiff_t)(2 * sizeof(size_t))) p += A;
  madvise(p, rup(bytes, A), MADV_HUGEPAGE);
  parallel_for((int64_t)((bytes + 4095) / 4096), [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; ++i) p[i * 4096] = 0; });
  if (cudaHostRegister(p, rup(bytes, 4096), cudaHostRegisterPortable) != cudaSuccess) { cudaGetLastError(); munmap(m, len); return nullptr; }
  reinterpret_cast<size_t*>(p)[-1] = len; reinterpret_cast<char**>(p)[-2] = m;
  return p;
}
void pinned_free(void* q) noexcept
{
  if (!q) return;
  char* p = (char*)q; const size_t len = reinterpret_cast<size_t*>(p)[-1]; char* m = reinterpret_cast<char**>(p)[-2];
  cudaHostUnregister(p); munmap(m, len);
}

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
    if (b.a[c].dtype != AFDTD_BF16 && b.a[c].dtype != AFDTD_F32 && b.a[c].dtype != AFDTD_F16)
      fail(AFDTD_EINVAL, "%s: dtype %d is not AFDTD_BF16 (1), AFDTD_F32 (2) or AFDTD_F16 (3)", kName[c], b.a[c].dtype);
    if (b.flags & AFDTD_BOX_STRIDED)
      for (int a = 0; a < 3; ++a) {
        if (b.a[c].stride[a] < 0) fail(AFDTD_EINVAL, "%s: negative stride %lld", kName[c], (long long)b.a[c].stride[a]);
        if (dir == 1 && b.a[c].stride[a] == 0 && b.shape[a] > 1) fail(AFDTD_EINVAL, "%s: zero stride on get (the writes would overlap)", kName[c]);
      }
  }
  if (!nc) fail(AFDTD_EINVAL, "the box gives no array (every data pointer is NULL)");
  const int64_t sx = b.shape[0], sy = b.shape[1], sz = b.shape[2], x0 = b.off[0], y0 = b.off[1], z0 = b.off[2];
  const int prec = p.prec; const uint32_t nosign = prec == AFDTD_F32 ? 0x7FFFFFFFu : 0x7FFFu;
  auto rd = [&](int c, int64_t e) { return to_state(prec, b.a[c].dtype, b.a[c].data, e); };
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
          const uint32_t v = rd(c, elem(b, c, i, j, k));
          if (c >= 6) { wc += v != 0; continue; }
          if ((v & nosign) == 0) continue;
          if (!wf++) { first[0] = c; first[1] = pos[0]; first[2] = pos[1]; first[3] = pos[2]; }
        }
      }
    }
  if (wf && (p.flags & AFDTD_STRICT_WALLS)) {
    const float f = state_float(prec, rd((int)first[0], elem(b, (int)first[0], first[1] - x0, first[2] - y0, first[3] - z0)));
    fail(AFDTD_EWALLS, "AFDTD_STRICT_WALLS: %llu E value(s) on the lower walls are not 0 (first: %s = %g at (%lld, %lld, %lld)); index 0 is the PEC wall, "
         "put fields at index >= 1; nothing was written", (unsigned long long)wf, kName[first[0]], f, (long long)first[1], (long long)first[2], (long long)first[3]);
  }
  // ---- copy in slabs of x planes through a dense staging buffer of the state's element type (16 or 32 bits)
  const size_t plane = (size_t)sy * sz, es = prec == AFDTD_F32 ? 4 : 2;
  const int64_t xs = std::max<int64_t>(1, (int64_t)(((size_t)256 << 20) / (es * plane * nc)));
  std::vector<uint32_t> stage(((size_t)std::min(xs, sx) * plane * nc * es + 3) / 4);
  auto put = [&](char* base, size_t i, uint32_t v) { if (es == 4) reinterpret_cast<uint32_t*>(base)[i] = v; else reinterpret_cast<uint16_t*>(base)[i] = (uint16_t)v; };
  auto get = [&](const char* base, size_t i) -> uint32_t { return es == 4 ? reinterpret_cast<const uint32_t*>(base)[i] : reinterpret_cast<const uint16_t*>(base)[i]; };
  for (int64_t xa = 0; xa < sx; xa += xs) {
    const int64_t xb = std::min(sx, xa + xs); const size_t cnt = (size_t)(xb - xa) * plane;
    uint16_t* comp[12]; char* cb[12]; int i = 0;
    for (int c = 0; c < 12; ++c) { cb[c] = b.a[c].data ? reinterpret_cast<char*>(stage.data()) + cnt * es * i++ : nullptr; comp[c] = reinterpret_cast<uint16_t*>(cb[c]); }
    const int64_t off[3] = {x0 + xa, y0, z0}, shp[3] = {xb - xa, sy, sz};
    if (dir == 0)
      parallel_for(xb - xa, [&](int64_t a, int64_t e) {
        for (int c = 0; c < 12; ++c) if (cb[c])
          for (int64_t ii = a; ii < e; ++ii) for (int64_t j = 0; j < sy; ++j) {
            const size_t d = ((size_t)ii * sy + j) * sz;
            for (int64_t k = 0; k < sz; ++k) put(cb[c], d + k, on_wall(c, x0 + xa + ii, y0 + j, z0 + k) ? 0u : rd(c, elem(b, c, xa + ii, j, k)));
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
        for (int c = 0; c < 12; ++c) if (cb[c])
          for (int64_t ii = a; ii < e; ++ii) for (int64_t j = 0; j < sy; ++j) {
            const size_t d = ((size_t)ii * sy + j) * sz;
            for (int64_t k = 0; k < sz; ++k) from_state(prec, b.a[c].dtype, b.a[c].data, elem(b, c, xa + ii, j, k), get(cb[c], d + k));
          }
      });
  }
  if (dir == 0) { p.wz_coef = wc; p.wz_field = wf; }
}

}  // namespace afdtd
