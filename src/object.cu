// SPDX-License-Identifier: MIT
// object.cu -- everything that touches one generated kernel text.  CMake compiles this file once per (kernel text, architecture)
// with -DKERN="kernels/<name>.cuh" -DAFDTD_NS=<namespace> -DAFDTD_ARCH="sm_XX" -DAFDTD_CC=<10 major + minor>; each copy lives in its
// own namespace and exports one afdtd::Object.  With -DAFDTD_FP16 the same text runs in IEEE half precision (the state's 16-bit values
// are FP16 bit patterns; every operation is the FP16 one), else in BF16.
//
//   in GPU memory   1 GPU: plain allocations, one persistent launch per <= 256 trips.  2..8 GPUs: one virtual window over the
//                   devices (y stripes, the seams read through peer mappings), one persistent launch per device.
//   in host memory  the helix: a ring of rows on the GPUs, each row loaded once per round, advanced Q trips, written back.
//   tail            steps % 7 single steps with the reference formula on the same layout (exact for every split of the steps).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "internal.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace AFDTD_NS {
#include KERN

#if SK != 7 || SS != 24 || SOFF != 20 || SGOFF != 8 || SCREC != 24 || SSH != 3 || SW0 != -2 || SNCOPY != 2 || SXGS != 2176 || SYGS != 512
#error "kernel text does not have the state layout of internal.hpp"
#endif
#if !SSPLIT || !SDUP || !SPM || !SCT || !SCP || SPAL || SMXY
#error "the executors serve the split + dup + pm + ct + cp layout with per-cell coefficients"
#endif
#ifndef SM80
#define SM80 0
#endif

using afdtd::Geo; using afdtd::Knobs; using afdtd::HostDomain; using afdtd::rup; using afdtd::kGR; using afdtd::fail;
typedef unsigned long long U64;
#ifdef AFDTD_FP16
typedef __half GT;
#define AFDTD_PREC_CODE 3            /* AFDTD_F16 */
#else
typedef __nv_bfloat16 GT;
#define AFDTD_PREC_CODE 1            /* AFDTD_BF16 */
#endif
#define CK AF_CK
#define CU AF_CU
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void need(bool ok, const char* what) { if (!ok) fail(-8 /* AFDTD_EINTERNAL */, "internal check failed: %s", what); }

// threads whose first exception is rethrown in the caller
static void par(int n, const std::function<void(int)>& f)
{
  std::vector<std::thread> th; std::exception_ptr ep = nullptr; std::mutex m;
  for (int i = 0; i < n; ++i) th.emplace_back([&, i] { try { f(i); } catch (...) { std::lock_guard<std::mutex> l(m); if (!ep) ep = std::current_exception(); } });
  for (auto& t : th) t.join();
  if (ep) std::rethrow_exception(ep);
}

// ================================================================================================ device helpers
__device__ __forceinline__ GT g_sub(GT a, GT b) { return __hsub(a, b); }
__device__ __forceinline__ GT g_mul(GT a, GT b) { return __hmul(a, b); }
__device__ __forceinline__ GT g_fma(GT a, GT b, GT c) { return __hfma(a, b, c); }
__host__ __device__ __forceinline__ U64 mix64(U64 x)
{
  x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
__device__ __forceinline__ void rec_pos(size_t t, const Geo& G, int& J, int& zi, int& gp, int& l)
{
  J = (int)(t / (size_t)G.ROWREC); const size_t rem = t % (size_t)G.ROWREC;
  zi = (int)(rem / (size_t)G.PLREC); const long pr = (long)(rem % (size_t)G.PLREC);
  gp = (int)(pr / SS); l = (int)(pr % SS);
}
__device__ __forceinline__ size_t rec_of(int x, int y, int z, const Geo& G)
{
  return (size_t)(y / SS) * G.ROWREC + (size_t)(z + SOFF) * G.PLREC + (long)(x / 2 + SGOFF) * SS + y % SS;
}
__device__ __forceinline__ unsigned short* coe_at(unsigned char* coe, size_t crow, const Geo& G, int J, int l, int x, int z, int w)
{
  unsigned* o = reinterpret_cast<unsigned*>(coe + (size_t)J * crow + (size_t)(z + SOFF) * G.PS + ((size_t)(x / 2 + SGOFF) * 30 + l) * SCREC);
  return reinterpret_cast<unsigned short*>(o + w) + (x & 1);
}
// every half-record that is not an owned cell back to +0; owned cells untouched
__global__ void scrub_k(Geo G, unsigned* buf, size_t t0, size_t t1)
{
  const size_t t = t0 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= t1) return;
  int J, zi, gp, l; rec_pos(t, G, J, zi, gp, l);
  const int z = zi - SOFF, y = J * SS + l, x0 = 2 * (gp - SGOFF);
  const bool row_in = z >= 0 && z < G.nz && y < G.ny && zi < G.NZP;
  unsigned keep = 0;
  for (int h = 0; h < 2; ++h) { const int x = x0 + h; if (row_in && x >= 0 && x < G.nx) keep |= 0xFFFFu << (16 * h); }
  if (keep == 0xFFFFFFFFu) return;
  for (int c = 0; c < 6; ++c) { const unsigned v = buf[t * 6 + c]; if (v & ~keep) buf[t * 6 + c] = v & keep; }
}
struct P12 { unsigned short* f[12]; };
// dense box slab [xa, xb) of box (bx0, by0, bz0) x (., bny, bnz) <-> state.  dir 0: in, 1: out.  NULL components are skipped.
__global__ void box_k(Geo G, unsigned* fld, unsigned char* coe, size_t crow, long bx0, long by0, long bz0, long bny, long bnz, long xa, long xb,
                      P12 S, int dir)
{
  const size_t li = (size_t)blockIdx.x * blockDim.x + threadIdx.x, plane = (size_t)bny * bnz;
  if (li >= (size_t)(xb - xa) * plane) return;
  const int x = (int)(bx0 + xa + (long)(li / plane)), y = (int)(by0 + (long)((li % plane) / bnz)), z = (int)(bz0 + (long)(li % bnz));
  unsigned short* fw = reinterpret_cast<unsigned short*>(fld + rec_of(x, y, z, G) * 6) + (x & 1);
  const int J = y / SS, l = y % SS;
  for (int c = 0; c < 12; ++c) {
    if (!S.f[c]) continue;
    if (c < 6) { if (dir == 0) fw[2 * c] = S.f[c][li]; else S.f[c][li] = fw[2 * c]; continue; }
    unsigned short* o = coe_at(coe, crow, G, J, l, x, z, c - 6);
    if (dir == 1) { S.f[c][li] = *o; continue; }
    *o = S.f[c][li];
    if (J > 0 && l + SS < 30) *coe_at(coe, crow, G, J - 1, l + SS, x, z, c - 6) = S.f[c][li];      // repeated lanes of row J - 1
  }
}
// checksums over the state, cells with y in [y0, y0 + nys): sum[c], c < 6
__global__ void esum_k(Geo G, const unsigned* buf, U64* sum, int y0, int nys)
{
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nl = (size_t)G.nx * nys * G.nz, n = (size_t)G.nx * G.ny * G.nz;
  U64 s[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 64; ++i) {
    const size_t li = t * 64 + i;
    if (li >= nl) break;
    const int z = (int)(li % G.nz), y = y0 + (int)((li / G.nz) % nys), x = (int)(li / ((size_t)G.nz * nys));
    const size_t idx = ((size_t)x * G.ny + y) * G.nz + z;
    const unsigned* o = buf + rec_of(x, y, z, G) * 6;
    for (int c = 0; c < 6; ++c) s[c] += mix64(((((U64)(c % 3)) * n + idx) << 32) + ((o[c] >> (16 * (x & 1))) & 0xFFFFu));
  }
  if (t * 64 < nl) for (int c = 0; c < 6; ++c) atomicAdd(&sum[c], s[c]);
}

// ================================================================================================ the step tail
// The reference steps (H half step, then E half step, one correctly rounded BF16 operation each, neighbours outside the domain +0)
// on the state layout.  Field rows [Jlo, Jhi) are present, row J of the buffer at (J - Jb0) ROWREC records and (J - Jb0) crow bytes;
// neighbours in absent rows read +0, so only rows with both neighbour rows present are exact.  One thread per record (an x pair at
// one (y, z)); the H kernel writes only H words and reads only E words of other records (E kernel: the converse): no race.
struct TailGeo { int nx, ny, nz; long ROWREC, PLREC; size_t PS, crow; int Jb0, Jlo, Jhi; };
#ifdef AFDTD_FP16
__device__ __forceinline__ GT tl_bf(unsigned short b) { return __ushort_as_half(b); }
__device__ __forceinline__ unsigned short tl_bits(GT v) { return __half_as_ushort(v); }
#else
__device__ __forceinline__ GT tl_bf(unsigned short b) { return __ushort_as_bfloat16(b); }
__device__ __forceinline__ unsigned short tl_bits(GT v) { return __bfloat16_as_ushort(v); }
#endif
__device__ __forceinline__ GT tl_h(unsigned w, int h) { return tl_bf((unsigned short)((w >> (16 * h)) & 0xFFFFu)); }
__device__ __forceinline__ bool tl_rec(const TailGeo& G, int Ja, size_t t, int& J, int& l, int& z, int& x0, int& y, size_t& r, size_t& cofs)
{
  J = Ja + (int)(t / (size_t)G.ROWREC);
  const size_t rem = t % (size_t)G.ROWREC;
  const int zi = (int)(rem / (size_t)G.PLREC); const long pr = (long)(rem % (size_t)G.PLREC);
  const int gp = (int)(pr / SS); l = (int)(pr % SS);
  z = zi - SOFF; y = J * SS + l; x0 = 2 * (gp - SGOFF);
  r = (size_t)(J - G.Jb0) * G.ROWREC + rem;
  cofs = (size_t)(J - G.Jb0) * G.crow + (size_t)zi * G.PS + ((size_t)gp * 30 + l) * SCREC;
  return z >= 0 && z < G.nz && y < G.ny && x0 >= 0 && x0 < G.nx;
}
__global__ void tail_h_k(TailGeo G, unsigned* fld, int Ja, int Jb)
{
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)(Jb - Ja) * G.ROWREC) return;
  int J, l, z, x0, y; size_t r, co;
  if (!tl_rec(G, Ja, t, J, l, z, x0, y, r, co)) return;
  const unsigned* R = fld + r * 6;
  const unsigned ew[3] = {R[0], R[1], R[2]};
  unsigned w[3] = {R[3], R[4], R[5]};
  const bool yin = y + 1 < G.ny && (l + 1 < SS || J + 1 < G.Jhi), zin = z + 1 < G.nz;
  const size_t ry = l + 1 < SS ? r + 1 : r + (size_t)G.ROWREC - (SS - 1), rz = r + (size_t)G.PLREC, rx = r + SS;
  const unsigned exy = yin ? fld[ry * 6 + 0] : 0u, ezy = yin ? fld[ry * 6 + 2] : 0u;
  const unsigned exz = zin ? fld[rz * 6 + 0] : 0u, eyz = zin ? fld[rz * 6 + 1] : 0u;
  const bool xin1 = x0 + 2 < G.nx;
  const unsigned eyx = xin1 ? fld[rx * 6 + 1] : 0u, ezx = xin1 ? fld[rx * 6 + 2] : 0u;
  for (int h = 0; h < 2; ++h) {
    const int x = x0 + h;
    if (x >= G.nx) continue;
    const GT ex = tl_h(ew[0], h), ey = tl_h(ew[1], h), ez = tl_h(ew[2], h);
    const bool xp = x + 1 < G.nx;
    const GT ez_xp = !xp ? tl_bf(0) : h == 0 ? tl_h(ew[2], 1) : tl_h(ezx, 0), ey_xp = !xp ? tl_bf(0) : h == 0 ? tl_h(ew[1], 1) : tl_h(eyx, 0);
    const GT ez_yp = tl_h(ezy, h), ex_yp = tl_h(exy, h), ey_zp = tl_h(eyz, h), ex_zp = tl_h(exz, h);
    const GT cc[3] = {g_sub(g_sub(ez_yp, ez), g_sub(ey_zp, ey)), g_sub(g_sub(ex_zp, ex), g_sub(ez_xp, ez)), g_sub(g_sub(ey_xp, ey), g_sub(ex_yp, ex))};
    for (int a = 0; a < 3; ++a) {
      const unsigned nb = tl_bits(g_sub(tl_h(w[a], h), cc[a]));
      w[a] = (w[a] & ~(0xFFFFu << (16 * h))) | (nb << (16 * h));
    }
  }
  fld[r * 6 + 3] = w[0]; fld[r * 6 + 4] = w[1]; fld[r * 6 + 5] = w[2];
}
__global__ void tail_e_k(TailGeo G, unsigned* fld, const unsigned char* coe, int Ja, int Jb)
{
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)(Jb - Ja) * G.ROWREC) return;
  int J, l, z, x0, y; size_t r, co;
  if (!tl_rec(G, Ja, t, J, l, z, x0, y, r, co)) return;
  const unsigned* cw = reinterpret_cast<const unsigned*>(coe + co);
  const unsigned* R = fld + r * 6;
  unsigned w[3] = {R[0], R[1], R[2]};
  const unsigned hw[3] = {R[3], R[4], R[5]};
  const bool yin = y >= 1 && (l > 0 || J - 1 >= G.Jlo), zin = z >= 1;
  const size_t ry = l > 0 ? r - 1 : r - (size_t)G.ROWREC + (SS - 1), rz = r - (size_t)G.PLREC, rx = r - SS;
  const unsigned hxy = yin ? fld[ry * 6 + 3] : 0u, hzy = yin ? fld[ry * 6 + 5] : 0u;
  const unsigned hxz = zin ? fld[rz * 6 + 3] : 0u, hyz = zin ? fld[rz * 6 + 4] : 0u;
  const bool xin1 = x0 >= 2;
  const unsigned hyx = xin1 ? fld[rx * 6 + 4] : 0u, hzx = xin1 ? fld[rx * 6 + 5] : 0u;
  for (int h = 0; h < 2; ++h) {
    const int x = x0 + h;
    if (x >= G.nx) continue;
    const GT hx = tl_h(hw[0], h), hy = tl_h(hw[1], h), hz = tl_h(hw[2], h);
    const GT hz_xm = h == 1 ? tl_h(hw[2], 0) : tl_h(hzx, 1), hy_xm = h == 1 ? tl_h(hw[1], 0) : tl_h(hyx, 1);
    const GT hz_ym = tl_h(hzy, h), hx_ym = tl_h(hxy, h), hy_zm = tl_h(hyz, h), hx_zm = tl_h(hxz, h);
    const GT cc[3] = {g_sub(g_sub(hz, hz_ym), g_sub(hy, hy_zm)), g_sub(g_sub(hx, hx_zm), g_sub(hz, hz_xm)), g_sub(g_sub(hy, hy_xm), g_sub(hx, hx_ym))};
    for (int a = 0; a < 3; ++a) {
      const unsigned nb = tl_bits(g_fma(tl_h(cw[3 + a], h), cc[a], g_mul(tl_h(cw[a], h), tl_h(w[a], h))));
      w[a] = (w[a] & ~(0xFFFFu << (16 * h))) | (nb << (16 * h));
    }
  }
  fld[r * 6 + 0] = w[0]; fld[r * 6 + 1] = w[1]; fld[r * 6 + 2] = w[2];
}
static unsigned tail_grid(const TailGeo& G, int Ja, int Jb) { return (unsigned)(((size_t)(Jb - Ja) * G.ROWREC + 255) / 256); }

#if !SM80
static CUtensorMap tensor_map(void* base, cuuint64_t d0, const Geo& G, int rows, cuuint64_t s1, cuuint64_t s2, cuuint32_t b0, cuuint32_t b1, cuuint64_t s0)
{
  CUtensorMap m; memset(&m, 0, sizeof m);
  cuuint64_t dims[4] = {d0, (cuuint64_t)G.NPAIR, (cuuint64_t)G.NZP, (cuuint64_t)rows}, str[3] = {s0, s1, s2};
  cuuint32_t box[4] = {b0, b1, 1u, 1u}, es[4] = {1u, 1u, 1u, 1u};
  CU(afdtd::drv().TensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, base, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                       CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
static CUtensorMap map_fields(void* base, const Geo& G, int rows)
{ return tensor_map(base, 6 * SS, G, rows, (cuuint64_t)G.PLREC * 24, (cuuint64_t)G.ROWREC * 24, 6 * SGH, 9, 6 * SS * 4); }
static CUtensorMap map_coefs(void* base, const Geo& G, int rows, size_t crow)
{ return tensor_map(base, 6 * 32, G, rows, (cuuint64_t)G.NPAIR * 768, crow, 180, 8, 6 * 32 * 4); }
#endif
static void launch_stream(const KP& P, unsigned grid, cudaStream_t st, void* fld, void* coe, const Geo& G, int frows, int crows, size_t crow)
{
#if SM80
  (void)fld; (void)coe; (void)G; (void)frows; (void)crows; (void)crow;
  stream<<<grid, SNTHR, SSMEM, st>>>(P);
#else
  stream<<<grid, SNTHR, SSMEM, st>>>(P, map_fields(fld, G, frows), map_coefs(coe, G, crows, crow));
#endif
  CK(cudaGetLastError());
}

// ================================================================================================ virtual memory
struct Seg { CUdeviceptr va; size_t sz; CUmemGenericAllocationHandle h; int dev; };
struct Vmm {
  std::vector<CUmemAccessDesc> acc; std::vector<Seg> segs;
  void init(const std::vector<int>& devs)
  {
    acc.resize(devs.size());
    for (size_t d = 0; d < devs.size(); ++d) { acc[d] = CUmemAccessDesc(); acc[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc[d].location.id = devs[d]; acc[d].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
  }
  static CUmemAllocationProp prop(int dev)
  {
    CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = dev; return p;
  }
  void map(CUdeviceptr va, size_t sz, int dev)
  {
    const afdtd::Driver& D = afdtd::drv(); CUmemAllocationProp p = prop(dev); Seg g; g.va = va; g.sz = sz; g.dev = dev;
    CK(cudaSetDevice(dev)); CU(D.MemCreate(&g.h, sz, &p, 0));
    if (D.MemMap(va, sz, 0, g.h, 0) != CUDA_SUCCESS) { D.MemRelease(g.h); fail(-5, "cuMemMap failed"); }
    segs.push_back(g);
    CU(D.MemSetAccess(va, sz, acc.data(), acc.size()));
    CK(cudaMemset((void*)va, 0, sz));
  }
  void map_by(CUdeviceptr va, size_t total, const std::function<int(size_t)>& owner)      // owner(offset) = device of that granule
  {
    size_t q0 = 0; const size_t nq = total / kGR;
    while (q0 < nq) { const int o = owner(q0 * kGR); size_t q1 = q0 + 1; while (q1 < nq && owner(q1 * kGR) == o) ++q1; map(va + q0 * kGR, (q1 - q0) * kGR, o); q0 = q1; }
  }
  void release() { for (auto& g : segs) { cudaSetDevice(g.dev); afdtd::drv().MemUnmap(g.va, g.sz); afdtd::drv().MemRelease(g.h); } segs.clear(); }
};

// ================================================================================================ domain in GPU memory
struct Dev { int dev; std::vector<int> ord; int nb = 0; int* dord = nullptr; size_t ord_cap = 0; long long* dtim = nullptr; size_t tim_cap = 0; unsigned* dtick = nullptr; };
struct Mem {
  Geo G; Knobs k; std::vector<int> devs; int ngpu = 1, nsm = 0; bool multi = false;
  // 1 GPU
  unsigned* FLD = nullptr; unsigned char *COE = nullptr, *XP = nullptr, *YH = nullptr; unsigned* prog = nullptr;
  size_t nrec = 0, CROW = 0, nxp = 0, nyh = 0, prog_cap = 0;
  // 2..8 GPUs: state window (fields, coefficients) and pool window (YH, XP, progress words)
  std::vector<int> J0, sr; size_t RSB = 0, FB = 0, CRSB = 0, CFB = 0, YHB = 0, XPB = 0, PGB = 0;
  CUdeviceptr base = 0, aux = 0, yhb = 0, xpb = 0, pgb = 0; int GAPX = 0, GAPY = 0, SSTX = 0, SSTY = 0; long PST = 0; Vmm state, pools;
  std::vector<Dev> L; int cur_passes = -1; Knobs cur_k; std::vector<U64*> dsum; unsigned short* stage = nullptr; size_t stage_cap = 0;
  // agents: one non-blocking stream and one seam event per GPU, a pinned array of DONE words, each pass's block order per GPU
  std::vector<cudaStream_t> ast; std::vector<cudaEvent_t> aev; unsigned* hdone = nullptr; size_t hdone_n = 0;
  std::vector<cudaStream_t> mst;     // the persistent launches: non-blocking, so legacy-stream work elsewhere does not wait for them
  std::vector<std::vector<int>> posv; std::vector<int> nppv; bool patched = false;
  size_t crow() const { return multi ? CRSB : CROW; }
  unsigned char* coe() const { return multi ? reinterpret_cast<unsigned char*>(base + FB) : COE; }
  unsigned* fld() const { return multi ? reinterpret_cast<unsigned*>(base) : FLD; }
  int Ja(int g) const { return multi ? J0[g] : 0; }
  int Jb(int g) const { return multi ? J0[g + 1] : G.NJ; }
};
static void sync_all(Mem& s) { for (int d : s.devs) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); } }
static int max_npp(const Mem& s) { int m = 0; for (int g = 0; g < s.ngpu; ++g) m = std::max(m, s.G.NI * (s.J0[g + 1] - s.J0[g])); return m; }

static void pools_single(Mem& s)
{
  CK(cudaSetDevice(s.devs[0]));
  if (s.XP) { CK(cudaFree(s.XP)); s.XP = nullptr; }
  if (s.YH) { CK(cudaFree(s.YH)); s.YH = nullptr; }
  s.nxp = (size_t)(s.k.mx + 1) * s.G.XBUF; s.nyh = (size_t)(s.k.mx + 1) * s.G.YBUF;
  CK(cudaMalloc(&s.XP, s.nxp)); CK(cudaMalloc(&s.YH, s.nyh));
}
static void pools_multi(Mem& s)
{
  const int NG = s.ngpu;
  if (s.aux) { s.pools.release(); afdtd::drv().MemAddressFree(s.aux, s.YHB + s.XPB + s.PGB); s.aux = 0; }
  s.SSTX = s.k.mx + 1 + 2 * s.GAPX; s.SSTY = s.k.mx + 1 + 2 * s.GAPY;
  s.YHB = rup((size_t)NG * s.SSTY * s.G.YBUF, kGR); s.XPB = rup((size_t)NG * s.SSTX * s.G.XBUF, kGR); s.PGB = (size_t)NG * s.PST * 4;
  CU(afdtd::drv().MemAddressReserve(&s.aux, s.YHB + s.XPB + s.PGB, 0, 0, 0));
  s.yhb = s.aux; s.xpb = s.yhb + s.YHB; s.pgb = s.xpb + s.XPB;
  s.pools.map_by(s.yhb, s.YHB, [&](size_t off) { return s.devs[std::min((size_t)NG - 1, off / ((size_t)s.SSTY * s.G.YBUF))]; });
  s.pools.map_by(s.xpb, s.XPB, [&](size_t off) { return s.devs[std::min((size_t)NG - 1, off / ((size_t)s.SSTX * s.G.XBUF))]; });
  s.pools.map_by(s.pgb, s.PGB, [&](size_t off) { return s.devs[std::min((size_t)NG - 1, off / ((size_t)s.PST * 4))]; });
}
static void alloc_single(Mem& s)
{
  const Geo& G = s.G;
  s.nrec = (size_t)(G.NJ + 1) * G.ROWREC; s.CROW = (size_t)G.NZP * G.NPAIR * 32 * SCREC;
  CK(cudaSetDevice(s.devs[0]));
  CK(cudaMalloc(&s.FLD, s.nrec * 24)); CK(cudaMalloc(&s.COE, (size_t)G.NJ * s.CROW));
  CK(cudaMemset(s.FLD, 0, s.nrec * 24)); CK(cudaMemset(s.COE, 0, (size_t)G.NJ * s.CROW));
  pools_single(s);
}
static void alloc_multi(Mem& s)
{
  const int NG = s.ngpu; const Geo& G = s.G;
  s.state.init(s.devs); s.pools.init(s.devs);
  s.RSB = (size_t)G.ROWREC * 24; s.FB = (size_t)(G.NJ + 1) * s.RSB; s.CRSB = s.RSB / 3 * 4; s.CFB = (size_t)(G.NJ + 1) * s.CRSB;
  s.J0.assign(NG + 1, 0);
  for (int g = 0; g <= NG; ++g) s.J0[g] = (int)((long)g * G.NJ / NG);
  for (int g = 0; g < NG; ++g) if (s.J0[g + 1] <= s.J0[g]) fail(-1, "ny = %d is too small for %d GPUs", G.ny, NG);
  s.sr.assign(G.NJ + 1, 0); for (int g = 0; g < NG; ++g) for (int J = s.J0[g]; J < s.J0[g + 1]; ++J) s.sr[J] = g; s.sr[G.NJ] = NG - 1;
  s.GAPX = (int)((kGR + G.XBUF - 1) / G.XBUF); s.GAPY = (int)((kGR + G.YBUF - 1) / G.YBUF);
  s.PST = (long)rup((size_t)2 * max_npp(s) * afdtd::kPMAX + 64, (size_t)1 << 19);      // blocks, DONE word, agent words
  CU(afdtd::drv().MemAddressReserve(&s.base, s.FB + s.CFB, 0, 0, 0));
  s.state.map_by(s.base, s.FB, [&](size_t off) { return s.devs[s.sr[off / s.RSB]]; });
  s.state.map_by(s.base + s.FB, s.CFB, [&](size_t off) { return s.devs[s.sr[off / s.CRSB]]; });
  pools_multi(s);
  sync_all(s);
}

// block tables: the order of pillar-passes and their guards (strip order: x strips of width tx, I fastest inside a strip)
static long strip_key(int id, int NI, long tx) { const long I = id % NI, J = id / NI; return ((I / tx) * 100000L + J) * 4096 + I % tx; }
static void upload_tables(Mem& s, Dev& l)
{
  CK(cudaSetDevice(l.dev));
  if (l.ord_cap < l.ord.size()) { if (l.dord) CK(cudaFree(l.dord)); l.dord = nullptr; CK(cudaMalloc(&l.dord, l.ord.size() * sizeof(int))); l.ord_cap = l.ord.size(); }
  CK(cudaMemcpy(l.dord, l.ord.data(), l.ord.size() * sizeof(int), cudaMemcpyHostToDevice));
  if (l.tim_cap < (size_t)l.nb * 16) { if (l.dtim) CK(cudaFree(l.dtim)); l.dtim = nullptr; CK(cudaMalloc(&l.dtim, (size_t)l.nb * 16 * sizeof(long long))); l.tim_cap = (size_t)l.nb * 16; }
  if (!l.dtick) CK(cudaMalloc(&l.dtick, 64));
  if (!s.multi && s.prog_cap < 2 * (size_t)l.nb + 64) { if (s.prog) CK(cudaFree(s.prog)); s.prog = nullptr; CK(cudaMalloc(&s.prog, (2 * (size_t)l.nb + 64) * 4)); s.prog_cap = 2 * (size_t)l.nb + 64; }
}
static void tables_single(Mem& s, int passes)
{
  const int NI = s.G.NI, NJ = s.G.NJ, NPP = NI * NJ, nblk = NPP * passes;
  const int MX = std::min(s.k.mx, nblk), GRID = std::min(s.k.grid == 0 ? s.nsm : s.k.grid, nblk);
  if (MX < nblk && MX <= 2 * (std::min(NI, NJ) + 2) + GRID) fail(-8, "pool size %d too small for grid %d", MX, GRID);
  std::vector<int> pil(NPP); for (int i = 0; i < NPP; ++i) pil[i] = i;
  std::sort(pil.begin(), pil.end(), [&](int x, int y) { return strip_key(x, NI, s.k.tx) < strip_key(y, NI, s.k.tx); });
  std::vector<int> pos(NPP); for (int i = 0; i < NPP; ++i) pos[pil[i]] = i;
  s.posv.assign(1, pos); s.nppv.assign(1, NPP);
  const int DONE = nblk;
  Dev& l = s.L[0]; l.ord.assign((size_t)nblk * 16, 0); l.nb = nblk;
  auto blk = [&](int p, int I, int J) { return p * NPP + pos[J * NI + I]; };
  for (int b = 0; b < nblk; ++b) {
    const int p = b / NPP, id = pil[b % NPP], I = id % NI, J = id / NI;
    int* o = &l.ord[(size_t)b * 16];
    o[0] = I; o[1] = J;
    o[2] = I > 0 ? blk(p, I - 1, J) % MX : MX; o[3] = b % MX;
    o[4] = J > 0 ? blk(p, I, J - 1) % MX : MX; o[5] = b % MX;
    o[6] = I > 0 ? blk(p, I - 1, J) : DONE; o[7] = J > 0 ? blk(p, I, J - 1) : DONE; o[8] = b;
    o[9] = p > 0 ? blk(p - 1, std::min(I + 1, NI - 1), std::min(J + 1, NJ - 1)) : DONE;
    o[10] = DONE; o[11] = DONE;
    if (b >= MX) {
      const int b2 = b - MX, p2 = b2 / NPP, id2 = pil[b2 % NPP], I2 = id2 % NI, J2 = id2 / NI;
      o[10] = I2 + 1 < NI ? blk(p2, I2 + 1, J2) : b2;
      o[11] = J2 + 1 < NJ ? blk(p2, I2, J2 + 1) : b2;
    }
    o[12] = DONE; o[13] = DONE;
    if (s.k.gate && GRID < nblk) {
      const int cx = I + 1 < NI ? blk(p, I + 1, J) : -1, cy = J + 1 < NJ ? blk(p, I, J + 1) : -1;
      if (cx >= 0 && cx / GRID == b / GRID && cx - GRID >= 0) o[12] = cx - GRID;
      if (cy >= 0 && cy / GRID == b / GRID && cy - GRID >= 0) o[13] = cy - GRID;
    }
    o[14] = DONE; o[15] = DONE;
    for (int j = 9; j < 14; ++j) if (o[j] != DONE && o[j] >= b) fail(-8, "guard %d of block %d is not earlier", j, b);
    if ((o[2] != MX && (b - o[6] >= MX || o[6] > b)) || (o[4] != MX && (b - o[7] >= MX || o[7] > b))) fail(-8, "pool size %d too small for strip width %d", MX, s.k.tx);
  }
  upload_tables(s, l);
}
static void tables_multi(Mem& s, int passes)
{
  const int NG = s.ngpu, NI = s.G.NI, NJ = s.G.NJ;
  std::vector<int> NPP(NG), nblk(NG); int maxblk = 0;
  for (int g = 0; g < NG; ++g) { NPP[g] = NI * (s.J0[g + 1] - s.J0[g]); nblk[g] = NPP[g] * passes; maxblk = std::max(maxblk, nblk[g]); }
  const int MX = std::min(s.k.mx, maxblk), GRIDA = s.k.grid == 0 ? s.nsm : s.k.grid;
  for (int g = 0; g < NG; ++g) if (MX < nblk[g] && MX <= 2 * (std::min(NI, s.J0[g + 1] - s.J0[g]) + 2) + GRIDA) fail(-8, "pool size %d too small", MX);
  const long PST = s.PST;
  std::vector<std::vector<int>> pil(NG), pos(NG);
  for (int g = 0; g < NG; ++g) {
    pil[g].resize(NPP[g]); pos[g].resize(NPP[g]);
    for (int i = 0; i < NPP[g]; ++i) pil[g][i] = i;
    std::sort(pil[g].begin(), pil[g].end(), [&](int x, int y) { return strip_key(x, NI, s.k.tx) < strip_key(y, NI, s.k.tx); });
    for (int i = 0; i < NPP[g]; ++i) pos[g][pil[g][i]] = i;
  }
  s.posv = pos; s.nppv = NPP;
  auto LB = [&](int p, int I, int J) { const int g = s.sr[J]; return p * NPP[g] + pos[g][(J - s.J0[g]) * NI + I]; };
  auto PR = [&](int p, int I, int J) { return (int)(s.sr[J] * PST + LB(p, I, J)); };
  auto DONE = [&](int g) { return (int)(g * PST + nblk[g]); };
  auto XS = [&](int g, int q) { return g * s.SSTX + s.GAPX + q; };
  auto YS = [&](int g, int q) { return g * s.SSTY + s.GAPY + q; };
  for (int g = 0; g < NG; ++g) {
    Dev& l = s.L[g]; l.ord.assign((size_t)nblk[g] * 16, 0); l.nb = nblk[g];
    const int GRD = std::min(GRIDA, nblk[g]);
    for (int b = 0; b < nblk[g]; ++b) {
      const int p = b / NPP[g], id = pil[g][b % NPP[g]], I = id % NI, J = s.J0[g] + id / NI;
      int* o = &l.ord[(size_t)b * 16];
      o[0] = I; o[1] = J;
      o[2] = I > 0 ? XS(g, LB(p, I - 1, J) % MX) : XS(g, MX); o[3] = XS(g, b % MX);
      o[4] = J > 0 ? YS(s.sr[J - 1], LB(p, I, J - 1) % MX) : YS(0, MX); o[5] = YS(g, b % MX);
      o[6] = I > 0 ? PR(p, I - 1, J) : DONE(g); o[7] = J > 0 ? PR(p, I, J - 1) : DONE(g); o[8] = PR(p, I, J);
      o[9] = p > 0 ? PR(p - 1, std::min(I + 1, NI - 1), std::min(J + 1, NJ - 1)) : DONE(g);
      o[10] = DONE(g); o[11] = DONE(g);
      if (b >= MX) {
        const int b2 = b - MX, p2 = b2 / NPP[g], id2 = pil[g][b2 % NPP[g]], I2 = id2 % NI, J2 = s.J0[g] + id2 / NI;
        o[10] = I2 + 1 < NI ? PR(p2, I2 + 1, J2) : PR(p2, I2, J2);
        o[11] = J2 + 1 < NJ ? PR(p2, I2, J2 + 1) : PR(p2, I2, J2);
      }
      o[12] = DONE(g); o[13] = DONE(g); o[14] = DONE(g); o[15] = DONE(g);
      if (s.k.gate && GRD < nblk[g]) {
        if (I + 1 < NI) { const int cx = LB(p, I + 1, J); if (cx / GRD == b / GRD && cx - GRD >= 0) o[12] = (int)(g * PST + cx - GRD); }
        if (J + 1 < s.J0[g + 1]) { const int cy = LB(p, I, J + 1); if (cy / GRD == b / GRD && cy - GRD >= 0) o[13] = (int)(g * PST + cy - GRD); }
      }
      for (int j = 9; j < 14; ++j) if (o[j] != DONE(g) && o[j] / PST == g && o[j] % PST >= b) fail(-8, "guard %d of block %d is not earlier", j, b);
      if (I > 0 && (b - LB(p, I - 1, J) >= MX || LB(p, I - 1, J) > b)) fail(-8, "pool size %d too small for strip width %d", MX, s.k.tx);
      if (J > s.J0[g] && (b - LB(p, I, J - 1) >= MX || LB(p, I, J - 1) > b)) fail(-8, "pool size %d too small", MX);
    }
    upload_tables(s, l);
  }
}

static KP kparams(const Mem& s, const Dev& l, unsigned* prog, unsigned char* xp, unsigned char* yh, int pmode)
{
  KP P; memset(&P, 0, sizeof P);
  P.fld = reinterpret_cast<Rec*>(s.fld()); P.coe = s.coe(); P.xp = xp; P.yh = yh; P.prog = prog; P.tim = l.dtim; P.ord = l.dord; P.nblk = l.nb;
  P.pmode = pmode; P.tick = l.dtick; P.ROWREC = s.G.ROWREC; P.PLREC = s.G.PLREC; P.XBUF = (long)s.G.XBUF; P.YBUF = (long)s.G.YBUF; P.NB = s.G.NB;
  P.NPAIR = s.G.NPAIR; P.CROW = (long)s.crow();
  return P;
}
// ================================================================================================ agents inside a launch
// Block (p, I, J) writes x pairs [8I - 3, 8I + 5) of row J at time 7(p + 1) and reads pairs [8I, 8I + 9) of row J and lanes 0..7 of
// row J + 1; it starts once its guards ord[9..13] read DONE.  (p, I, J) done implies (p, I', J') done for I' <= I, J' <= J (the x and
// y producer chains).  So the state of a region at time 7(p + 1) is final once one late enough block of pass p is done, and stays so
// until the blocks of pass p + 1 that write it (copies) or touch it at all (writes) start.  Those "gated" blocks get their ord[9]
// (previous-pass guard, (p, I + 1, J + 1)) redirected to an agent word, written DONE by the GPU's agent stream after: a stream wait
// on the late block (which also covers the redirected guard), the work, and (last row of a GPU) the next GPU's part, through an
// event.  The agent stream needs no SM: stream wait, copies (copy engines), and DONE written by a host-to-device copy (a memset
// could run as a kernel, behind the persistent grid that holds every SM).
// A boundary's work runs in parts, in pass order: first all writes (injections, one part), then the copies split by strip of the
// schedule (x strips of tx pillars, the order in which a pass runs): each copy part waits only for its own strip's writers and holds
// back only its own strip's blocks of the next pass, so it has about a pass to finish, and the parts pipeline on the copy engines.
// A block held back by several parts waits for the last of them (the agent stream runs the parts in order).
struct AgPart { int Itop = 0, q0 = -1, q1 = -1; std::vector<int> jtop, w0, w1; std::vector<char> seam, work; };   // q0 < 0: the writes
struct AgB { int p; int64_t t; std::vector<AgPart> parts; };
static inline long blk_local(const Mem& s, int g, int p, int I, int J) { return (long)p * s.nppv[g] + s.posv[g][(size_t)(J - s.Ja(g)) * s.G.NI + I]; }
static inline long prog_index(const Mem& s, int g, long local) { return s.multi ? (long)g * s.PST + local : local; }
static inline unsigned* prog_base(const Mem& s) { return s.multi ? reinterpret_cast<unsigned*>(s.pgb) : s.prog; }

// the boundaries of a launch of P passes from plan step step0 that have agent work; patch[g] = (ord index, new value)
static void plan_agents(Mem& s, int P, int64_t step0, const afdtd::AgentHooks& h, std::vector<AgB>& B,
                        std::vector<std::vector<std::pair<long, int>>>& patch, std::vector<int>& nW)
{
  const int NI = s.G.NI, NJ = s.G.NJ, NG = s.ngpu, TX = std::max(1, s.k.tx), NS = (NI + TX - 1) / TX;
  std::vector<afdtd::Region> rg; std::vector<int> owner;            // per block (I, J) of the next pass: the last part holding it, -1
  patch.assign(NG, {}); nW.assign(NG, 0);
  for (int p = 0; p < P; ++p) {
    const int64_t t = step0 + 7 * (int64_t)(p + 1);
    if (!h.want(t)) continue;
    rg.clear(); h.regions(t, rg);
    const bool gate = p + 1 < P;                          // the next pass is in this launch
    if (gate) owner.assign((size_t)NI * NJ, -1);
    AgB b; b.p = p; b.t = t;
    auto new_part = [&](int q0, int q1) { AgPart a; a.q0 = q0; a.q1 = q1; a.jtop.assign(NG, -1); a.w0.assign(NG, 0); a.w1.assign(NG, 0);
                                          a.seam.assign(NG, 0); a.work.assign(NG, 0); return a; };
    // one region's share of a part: writers' rows (work) and the blocks of the next pass it holds back (rows Jg0..J1-1, pillars Ig0..Ig1)
    auto add = [&](AgPart& a, int ai, int J0, int J1, int Jg0, int Ig0, int Ig1, int& Imax) {
      Imax = std::max(Imax, Ig1);
      for (int g = 0; g < NG; ++g) {
        const int Ja = s.Ja(g), Jb = s.Jb(g);
        if (std::max(J0, Ja) < std::min(J1, Jb)) { a.work[g] = 1; a.jtop[g] = std::max(a.jtop[g], std::min(J1, Jb) - 1); }
        const int lo = std::max(Jg0, Ja), hi = std::min(J1, Jb);
        if (gate && lo < hi) {
          a.jtop[g] = std::max(a.jtop[g], hi - 1);
          for (int J = lo; J < hi; ++J) for (int I = Ig0; I <= Ig1; ++I) owner[(size_t)J * NI + I] = std::max(owner[(size_t)J * NI + I], ai);
          if (hi == Jb && Jb < NJ) a.seam[g] = 1;          // held blocks in g's last row: their (p, I + 1, Jb) is on GPU g + 1
        }
      }
    };
    std::vector<int> imax;
    {   // the writes: one part
      AgPart a = new_part(-1, -1); int Imax = -1;
      for (const auto& r : rg) {
        if (!r.write) continue;
        const int J0 = std::max(0, r.J0), J1 = std::min(NJ, r.J1), q0 = std::max(0, r.q0), q1 = r.q1;
        if (J0 >= J1 || q0 >= q1) continue;
        const int Iw0 = std::min(NI - 1, (q0 + 3) / 8), Iw1 = std::min(NI - 1, (q1 + 2) / 8);
        const int Ig0 = std::max(0, std::min(Iw0, q0 / 8 - 1)), Ig1 = std::max(Iw1, std::min(NI - 1, (q1 - 1) / 8));
        add(a, (int)b.parts.size(), J0, J1, std::max(0, J0 - 1), Ig0, Ig1, Imax);
      }
      if (Imax >= 0) { b.parts.push_back(a); imax.push_back(Imax); }
    }
    for (int st = 0; st < NS; ++st) {                     // the copies, strip by strip
      const int Is0 = st * TX, Is1 = std::min(NI, Is0 + TX) - 1;
      const int qs0 = std::max(0, 8 * Is0 - 3), qs1 = 8 * (Is1 + 1) - 3;        // pairs whose writer is in the strip
      AgPart a = new_part(qs0, Is1 == NI - 1 ? (1 << 30) : qs1); int Imax = -1;
      for (const auto& r : rg) {
        if (r.write) continue;
        const int J0 = std::max(0, r.J0), J1 = std::min(NJ, r.J1), q0 = std::max(r.q0, a.q0), q1 = std::min(r.q1, a.q1);
        if (J0 >= J1 || q0 >= q1) continue;
        const int Iw0 = std::min(NI - 1, (q0 + 3) / 8), Iw1 = std::min(NI - 1, (q1 + 2) / 8);
        add(a, (int)b.parts.size(), J0, J1, J0, Iw0, Iw1, Imax);
      }
      if (Imax >= 0) { b.parts.push_back(a); imax.push_back(Imax); }
    }
    if (b.parts.empty()) continue;
    for (size_t k = 0; k < b.parts.size(); ++k) {
      AgPart& a = b.parts[k];
      a.Itop = std::min(NI - 1, imax[k] + 1);
      for (int g = 0; g < NG; ++g) if (a.jtop[g] >= 0) a.jtop[g] = std::min(s.Jb(g) - 1, a.jtop[g] + 1);
      for (int g = 0; g + 1 < NG; ++g) if (a.seam[g]) a.jtop[g + 1] = std::max(a.jtop[g + 1], s.Ja(g + 1));
    }
    if (gate)                                             // agent words: each held block gets one, from the last part holding it
      for (size_t k = 0; k < b.parts.size(); ++k)
        for (int g = 0; g < NG; ++g) {
          AgPart& a = b.parts[k];
          a.w0[g] = nW[g];
          const long base = (long)s.L[g].nb + 64;
          for (int J = s.Ja(g); J < s.Jb(g); ++J) for (int I = 0; I < NI; ++I)
            if (owner[(size_t)J * NI + I] == (int)k) patch[g].emplace_back(blk_local(s, g, p + 1, I, J) * 16 + 9, (int)prog_index(s, g, base + nW[g]++));
          a.w1[g] = nW[g];
        }
    B.push_back(std::move(b));
  }
}
// block tables on the device: the pristine ones, or with agent words in ord[9] of the gated blocks
static void upload_ord(Mem& s, const std::vector<std::vector<std::pair<long, int>>>* patch)
{
  bool any = false; if (patch) for (auto& v : *patch) any |= !v.empty();
  if (!any && !s.patched) return;
  for (int g = 0; g < s.ngpu; ++g) {
    Dev& l = s.L[g]; CK(cudaSetDevice(l.dev));
    if (!any) { CK(cudaMemcpy(l.dord, l.ord.data(), l.ord.size() * sizeof(int), cudaMemcpyHostToDevice)); continue; }
    std::vector<int> o = l.ord;
    for (auto& pv : (*patch)[g]) o[(size_t)pv.first] = pv.second;
    CK(cudaMemcpy(l.dord, o.data(), o.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  s.patched = any;
}
static void need_done_words(Mem& s, size_t n)
{
  if (s.hdone_n >= n) return;
  if (s.hdone) CK(cudaFreeHost(s.hdone)); s.hdone = nullptr; s.hdone_n = 0;
  CK(cudaHostAlloc(&s.hdone, std::max<size_t>(n, 1024) * 4, cudaHostAllocPortable));
  s.hdone_n = std::max<size_t>(n, 1024);
  for (size_t i = 0; i < s.hdone_n; ++i) s.hdone[i] = 0x3fffffffu;
}
static afdtd::AgentDev agent_dev(const Mem& s, int g)
{ afdtd::AgentDev d{g, s.devs[g], s.Ja(g), s.Jb(g), reinterpret_cast<char*>(s.fld()), s.G.ROWREC, s.G.PLREC, s.ast[g]}; d.prec = AFDTD_PREC_CODE; return d; }
// enqueue the agent work of every boundary, in order (the host may block in h.begin: back-pressure from the sinks)
static void run_agents(Mem& s, const std::vector<AgB>& B, const afdtd::AgentHooks& h)
{
  unsigned* pb = prog_base(s);
  for (const AgB& b : B) {
    h.begin(b.t);
    for (const AgPart& a : b.parts)
      for (int g = s.ngpu - 1; g >= 0; --g) {
        if (a.jtop[g] < 0) continue;
        CK(cudaSetDevice(s.devs[g])); cudaStream_t st = s.ast[g];
        const long tgt = prog_index(s, g, blk_local(s, g, b.p, a.Itop, a.jtop[g]));
        CU(afdtd::drv().StreamWaitValue32((CUstream)st, (CUdeviceptr)(pb + tgt), 0x3fffffffu, CU_STREAM_WAIT_VALUE_GEQ));
        if (a.work[g]) h.work(b.t, agent_dev(s, g), a.q0, a.q1);
        if (g > 0 && a.seam[g - 1]) CK(cudaEventRecord(s.aev[g], st));
        if (a.seam[g]) CK(cudaStreamWaitEvent(st, s.aev[g + 1], 0));
        if (a.w1[g] > a.w0[g])
          CK(cudaMemcpyAsync(pb + prog_index(s, g, (long)s.L[g].nb + 64 + a.w0[g]), s.hdone, (size_t)(a.w1[g] - a.w0[g]) * 4, cudaMemcpyHostToDevice, st));
      }
    h.end(b.t);
  }
}
// after a failure while enqueuing: every agent word DONE (a copy on a fresh stream), so that the launch can finish
static void release_agents(Mem& s, const std::vector<int>& nW) noexcept
{
  unsigned* pb = prog_base(s);
  for (int g = 0; g < s.ngpu; ++g) {
    if (!nW[g] || !s.hdone) continue;
    cudaSetDevice(s.devs[g]); cudaStream_t ds = nullptr;
    if (cudaStreamCreateWithFlags(&ds, cudaStreamNonBlocking) != cudaSuccess) continue;
    cudaMemcpyAsync(pb + prog_index(s, g, (long)s.L[g].nb + 64), s.hdone, (size_t)nW[g] * 4, cudaMemcpyHostToDevice, ds);
    cudaStreamSynchronize(ds); cudaStreamDestroy(ds);
  }
}

// one launch of `passes` trips over the whole domain (from plan step step0, agents h or none); returns the seconds of the launches
static double launch(Mem& s, int passes, int64_t step0, const afdtd::AgentHooks* h)
{
  if (s.cur_passes != passes || memcmp(&s.cur_k, &s.k, sizeof s.k)) { (s.multi ? tables_multi : tables_single)(s, passes); s.cur_passes = passes; s.cur_k = s.k; s.patched = false; }
  std::vector<AgB> B; std::vector<std::vector<std::pair<long, int>>> patch; std::vector<int> nW(s.ngpu, 0);
  if (h) plan_agents(s, passes, step0, *h, B, patch, nW);
  upload_ord(s, B.empty() ? nullptr : &patch);
  size_t maxw = 0; for (int w : nW) maxw = std::max(maxw, (size_t)w);
  if (maxw) need_done_words(s, maxw);
  auto agents = [&]() { if (B.empty()) return; try { run_agents(s, B, *h); } catch (...) { release_agents(s, nW); sync_all(s); throw; } };
  const Geo& G = s.G;
  if (!s.multi) {
    Dev& l = s.L[0]; const int nblk = l.nb;
    CK(cudaSetDevice(l.dev));
    CK(cudaMemset(s.XP, 0, s.nxp)); CK(cudaMemset(s.YH, 0, s.nyh));
    std::vector<unsigned> hp((size_t)nblk + 64 + nW[0], 0u); hp[nblk] = 0x3fffffffu;
    CK(cudaMemcpy(s.prog, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(l.dtim, 0, (size_t)nblk * 16 * sizeof(long long))); CK(cudaMemset(l.dtick, 0, 64));
    CK(cudaDeviceSynchronize());
    const int GRID = std::min(s.k.grid == 0 ? s.nsm : s.k.grid, nblk);
    const double t0 = now();
    launch_stream(kparams(s, l, s.prog, s.XP, s.YH, 0), GRID, s.mst[0], s.FLD, s.COE, G, G.NJ + 1, G.NJ, s.CROW);
    agents();
    CK(cudaDeviceSynchronize());
    const double t1 = now();
    scrub_k<<<(unsigned)((s.nrec + 255) / 256), 256>>>(G, s.FLD, 0, s.nrec);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    return t1 - t0;
  }
  const int NG = s.ngpu;
  for (auto& g : s.pools.segs) { CK(cudaSetDevice(g.dev)); CK(cudaMemsetAsync((void*)g.va, 0, g.sz)); }
  sync_all(s);
  unsigned* prog = reinterpret_cast<unsigned*>(s.pgb);
  for (int g = 0; g < NG; ++g) { CK(cudaSetDevice(s.devs[g])); const unsigned dn = 0x3fffffffu; CK(cudaMemcpy(prog + g * s.PST + s.L[g].nb, &dn, 4, cudaMemcpyHostToDevice)); }
  for (auto& l : s.L) { CK(cudaSetDevice(l.dev)); CK(cudaMemset(l.dtim, 0, (size_t)l.nb * 16 * sizeof(long long))); CK(cudaMemset(l.dtick, 0, 64)); }
  sync_all(s);
  const int GRIDA = s.k.grid == 0 ? s.nsm : s.k.grid;
  std::vector<double> tend(NG, 0);
  const double t0 = now();
  for (auto& l : s.L) {
    CK(cudaSetDevice(l.dev));
    launch_stream(kparams(s, l, prog, reinterpret_cast<unsigned char*>(s.xpb), reinterpret_cast<unsigned char*>(s.yhb), 0), std::min(GRIDA, l.nb),
                  s.mst[&l - s.L.data()], s.fld(), s.coe(), G, G.NJ + 1, G.NJ, s.CRSB);
  }
  agents();
  std::vector<cudaError_t> er(NG, cudaSuccess);
  { std::vector<std::thread> th;
    for (int i = 0; i < NG; ++i) th.emplace_back([&, i]() { cudaSetDevice(s.L[i].dev); er[i] = cudaDeviceSynchronize(); tend[i] = now(); });
    for (auto& t : th) t.join(); }
  for (auto e : er) CK(e);
  double t1 = 0; for (double t : tend) t1 = std::max(t1, t);
  for (int g = 0; g < NG; ++g) {
    const size_t a = (size_t)s.J0[g] * G.ROWREC, b = (size_t)(s.J0[g + 1] + (g == NG - 1 ? 1 : 0)) * G.ROWREC;
    CK(cudaSetDevice(s.devs[g])); scrub_k<<<(unsigned)((b - a + 255) / 256), 256>>>(G, s.fld(), a, b); CK(cudaGetLastError());
  }
  sync_all(s);
  return t1 - t0;
}

static void m_destroy(void* st)
{
  Mem* s = static_cast<Mem*>(st); if (!s) return;
  for (auto& l : s->L) { cudaSetDevice(l.dev); cudaFree(l.dord); cudaFree(l.dtim); cudaFree(l.dtick); }
  for (size_t g = 0; g < s->ast.size(); ++g) { cudaSetDevice(s->devs[g]); if (s->ast[g]) cudaStreamDestroy(s->ast[g]); if (s->aev[g]) cudaEventDestroy(s->aev[g]);
                                               if (g < s->mst.size() && s->mst[g]) cudaStreamDestroy(s->mst[g]); }
  if (s->hdone) cudaFreeHost(s->hdone);
  for (size_t d = 0; d < s->dsum.size(); ++d) { cudaSetDevice(s->devs[d]); cudaFree(s->dsum[d]); }
  if (!s->devs.empty()) { cudaSetDevice(s->devs[0]); cudaFree(s->stage); }
  if (!s->multi) { if (!s->devs.empty()) cudaSetDevice(s->devs[0]); cudaFree(s->FLD); cudaFree(s->COE); cudaFree(s->XP); cudaFree(s->YH); cudaFree(s->prog); }
  else {
    s->state.release(); s->pools.release();
    if (s->base) afdtd::drv().MemAddressFree(s->base, s->FB + s->CFB);
    if (s->aux) afdtd::drv().MemAddressFree(s->aux, s->YHB + s->XPB + s->PGB);
  }
  delete s;
}
static void* m_create(const Geo& G, const int* devs, int n, const Knobs& k)
{
  Mem* s = new Mem;
  try {
    s->G = G; s->k = k; s->devs.assign(devs, devs + n); s->ngpu = n; s->multi = n > 1;
    CK(cudaDeviceGetAttribute(&s->nsm, cudaDevAttrMultiProcessorCount, devs[0]));
    for (int d : s->devs) { CK(cudaSetDevice(d)); CK(cudaFuncSetAttribute(stream, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SSMEM)); }
    if (s->multi) alloc_multi(*s); else { s->J0 = {0, G.NJ}; alloc_single(*s); }
    s->L.resize(n); for (int g = 0; g < n; ++g) s->L[g].dev = devs[g];
    s->dsum.assign(n, nullptr);
    for (int g = 0; g < n; ++g) { CK(cudaSetDevice(devs[g])); CK(cudaMalloc(&s->dsum[g], 64)); }
    s->ast.assign(n, nullptr); s->aev.assign(n, nullptr); s->mst.assign(n, nullptr);
    for (int g = 0; g < n; ++g) { CK(cudaSetDevice(devs[g])); CK(cudaStreamCreateWithFlags(&s->ast[g], cudaStreamNonBlocking));
      CK(cudaStreamCreateWithFlags(&s->mst[g], cudaStreamNonBlocking));
      CK(cudaEventCreateWithFlags(&s->aev[g], cudaEventDisableTiming)); }
    return s;
  } catch (...) { m_destroy(s); throw; }
}
static void m_set_knobs(void* st, const Knobs& k)
{
  Mem& s = *static_cast<Mem*>(st);
  const bool pools = k.mx != s.k.mx;
  s.k = k;
  if (pools) (s.multi ? pools_multi : pools_single)(s);
}
static void m_box(void* st, const int64_t off[3], const int64_t shape[3], uint16_t* const comp[12], int dir)
{
  Mem& s = *static_cast<Mem*>(st);
  const size_t plane = (size_t)shape[1] * shape[2];
  int nc = 0; for (int c = 0; c < 12; ++c) nc += comp[c] != nullptr;
  if (!nc) return;
  const long xs = std::max(1L, (long)(((size_t)64 << 20) / (2 * plane * nc)));      // x planes per slab (64 MiB of staging)
  const size_t cap = (size_t)std::min<int64_t>(xs, shape[0]) * plane * nc;
  CK(cudaSetDevice(s.devs[0]));
  if (s.stage_cap < cap) { if (s.stage) CK(cudaFree(s.stage)); s.stage = nullptr; CK(cudaMalloc(&s.stage, cap * 2)); s.stage_cap = cap; }
  for (long xa = 0; xa < shape[0]; xa += xs) {
    const long xb = std::min<long>(shape[0], xa + xs); const size_t cnt = (size_t)(xb - xa) * plane;
    P12 S; int i = 0;
    for (int c = 0; c < 12; ++c) S.f[c] = comp[c] ? s.stage + cnt * i++ : nullptr;
    if (dir == 0) for (int c = 0; c < 12; ++c) if (comp[c]) CK(cudaMemcpy(S.f[c], comp[c] + (size_t)xa * plane, cnt * 2, cudaMemcpyHostToDevice));
    box_k<<<(unsigned)((cnt + 255) / 256), 256>>>(s.G, s.fld(), s.coe(), s.crow(), off[0], off[1], off[2], shape[1], shape[2], xa, xb, S, dir);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    if (dir == 1) for (int c = 0; c < 12; ++c) if (comp[c]) CK(cudaMemcpy(comp[c] + (size_t)xa * plane, S.f[c], cnt * 2, cudaMemcpyDeviceToHost));
  }
}
// the agents' work at t on the state as it is (no launch running), synchronous
static void m_agents_now(void* st, int64_t t, const afdtd::AgentHooks& h)
{
  Mem& s = *static_cast<Mem*>(st);
  std::vector<afdtd::Region> rg; h.regions(t, rg);
  sync_all(s);
  h.begin(t);
  for (int g = 0; g < s.ngpu; ++g) {
    bool any = false; for (auto& r : rg) any |= std::max(r.J0, s.Ja(g)) < std::min(r.J1, s.Jb(g));
    if (!any) continue;
    CK(cudaSetDevice(s.devs[g])); h.work(t, agent_dev(s, g), -1, -1); h.work(t, agent_dev(s, g), 0, 1 << 30);   // writes, then copies
  }
  h.end(t);
  for (int g = 0; g < s.ngpu; ++g) { CK(cudaSetDevice(s.devs[g])); CK(cudaStreamSynchronize(s.ast[g])); }
}
// passes trips from plan step step0.  Agent work runs inside the launches (stream waits); without stream memory operations in the
// driver, each launch ends at the next boundary with work instead (slower: a drain and a scrub per boundary).
static double m_advance(void* st, int64_t passes, int64_t step0, const afdtd::AgentHooks* h)
{
  Mem& s = *static_cast<Mem*>(st);
  // CUDA_LAUNCH_BLOCKING makes a launch return only when it is done, before the agent work could be enqueued: end the launches
  // at the boundaries instead
  static const bool blocking = [] { const char* e = getenv("CUDA_LAUNCH_BLOCKING"); return e && atoi(e) != 0; }();
  const bool inl = h && afdtd::drv().StreamWaitValue32 && !blocking;
  double sec = 0;
  for (int64_t done = 0; done < passes;) {
    int p = (int)std::min<int64_t>(afdtd::kPMAX, passes - done);
    if (h && !inl) { int q = 1; while (q < p && !h->want(step0 + 7 * (done + q))) ++q; p = q; }
    sec += launch(s, p, step0 + 7 * done, inl ? h : nullptr);
    done += p;
    if (h && !inl && h->want(step0 + 7 * done)) m_agents_now(st, step0 + 7 * done, *h);
  }
  return sec;
}
// H half step on every GPU, all synchronised, then the E half step (the reference's step order); several GPUs read the seam rows
// through the window
static void m_tail(void* st, int steps)
{
  Mem& s = *static_cast<Mem*>(st);
  TailGeo T; T.nx = s.G.nx; T.ny = s.G.ny; T.nz = s.G.nz; T.ROWREC = s.G.ROWREC; T.PLREC = s.G.PLREC;
  T.PS = s.G.PS; T.crow = s.crow(); T.Jb0 = 0; T.Jlo = 0; T.Jhi = s.G.NJ;
  sync_all(s);
  for (int k = 0; k < steps; ++k)
    for (int half = 0; half < 2; ++half) {
      for (int g = 0; g < s.ngpu; ++g) {
        const int Ja = s.Ja(g), Jb = s.Jb(g);
        CK(cudaSetDevice(s.devs[g]));
        if (half == 0) tail_h_k<<<tail_grid(T, Ja, Jb), 256>>>(T, s.fld(), Ja, Jb); else tail_e_k<<<tail_grid(T, Ja, Jb), 256>>>(T, s.fld(), s.coe(), Ja, Jb);
        CK(cudaGetLastError());
      }
      sync_all(s);
    }
}
static void m_checksums(void* st, uint64_t out[6])
{
  Mem& s = *static_cast<Mem*>(st);
  U64 tot[6] = {0, 0, 0, 0, 0, 0};
  for (int g = 0; g < s.ngpu; ++g) {
    const int y0 = s.Ja(g) * SS, nys = std::min(s.Jb(g) * SS, s.G.ny) - y0;
    CK(cudaSetDevice(s.devs[g])); CK(cudaMemset(s.dsum[g], 0, 64));
    const size_t nl = (size_t)s.G.nx * nys * s.G.nz;
    esum_k<<<(unsigned)((nl / 64 + 256) / 256), 256>>>(s.G, s.fld(), s.dsum[g], y0, nys);
    CK(cudaGetLastError());
    U64 h[8]; CK(cudaMemcpy(h, s.dsum[g], 64, cudaMemcpyDeviceToHost));
    for (int c = 0; c < 6; ++c) tot[c] += h[c];
  }
  for (int c = 0; c < 6; ++c) out[c] = tot[c];
}
static void m_zero(void* st)
{
  Mem& s = *static_cast<Mem*>(st);
  if (!s.multi) { CK(cudaSetDevice(s.devs[0])); CK(cudaMemset(s.FLD, 0, s.nrec * 24)); CK(cudaMemset(s.COE, 0, (size_t)s.G.NJ * s.CROW)); }
  else for (auto& g : s.state.segs) { CK(cudaSetDevice(g.dev)); CK(cudaMemsetAsync((void*)g.va, 0, g.sz)); }
  sync_all(s);
}

// ================================================================================================ domain in host memory (the helix)
// The domain lives in host memory as rows (fields, then coefficients).  A ring of NR = nslot * S rows is resident on the GPUs (slot q
// on device q % D); virtual row v (NJV = NJ + zero rows, a multiple of NR) maps statically to ring position v mod NR.  Row g may run
// trip p once row g + 1 holds trip p and row g - 1 has done trip p.  Each row is loaded once per round, advanced Q trips, written
// back.  Block (p, I, g) has the time key (p + g) DL + p lam (+ an x skew); launch L takes the keys in [L DL, (L + 1) DL), alternating
// between two streams per device so that launch L + 1 fills the SMs launch L releases; launch L + 2 waits for launch L everywhere.
struct Blk { int p, I; long g; double key; };
struct Ref { int L, d, loc; long seq; };
static inline U64 bkey(long g, int p, int I) { return ((U64)g << 26) | ((U64)p << 13) | (U64)I; }

static double h_run(HostDomain& h, int passes, int Q)
{
  const int D = (int)h.devs.size(), S = h.ring.S, NI = h.G.NI, NJ = h.G.NJ, MX = 400, PF = 2;
  const int NSLOT = h.ring.nslot, NR = NSLOT * S, NJV = (int)rup((size_t)NJ + 1, (size_t)NR);
  const double LAM = h.ring.lam; const size_t chunk = (size_t)64 << 20;
  const Geo& G = h.G; const int NB = G.NB; const size_t RSB = h.RSB, CSB = h.CSB, CROW = h.CROW, HRS = h.HRS, XBUF = G.XBUF, YBUF = G.YBUF;
  auto DEV = [&](int d) { return h.devs[d]; };
  int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, DEV(0)));
  const int GRID = nsm;
  need(passes >= 1 && Q >= 1, "helix passes");
  if (afdtd::ring_bad(Q, NR, NI, NJ, D, passes, LAM, GRID)) fail(-8, "host-memory ring of %d rows too small for depth %d", NR, Q);
  const int rounds = (passes + Q - 1) / Q, Qlast = passes - (rounds - 1) * Q;
  const long GTOT = (long)rounds * NJV;
  const double DL = (double)Q * NI / ((double)GRID * D), LL = DL;
  auto vrow = [&](long g) { return (int)(g % NJV); };
  auto tgt = [&](long g) { const int v = vrow(g); return v < NJ ? (g / NJV == rounds - 1 ? Qlast : Q) : 0; };
  auto dev = [&](long g) { return ((vrow(g) / S) % NSLOT) % D; };
  auto keyf = [&](int p, long g) { return (double)(p + g) * DL + p * LAM; };
  const double EPS = 24.0 / NB;
  auto keyb = [&](int p, int I, long g) { return keyf(p, g) + I * EPS; };
  auto LK = [&](double t) { return (long)std::floor(t / LL); };
  auto needl = [&](long g) -> long { const int v = vrow(g); if (v > NJ) return -1; if (v == 0) return LK(keyf(0, g)); return LK(keyf(0, g - 1)); };
  auto rel = [&](long g) -> long { const int v = vrow(g); if (v > NJ) return -1; if (v == NJ) return LK(keyf(tgt(g - 1) - 1, g - 1)); return LK(keyf(tgt(g) - 1, g)); };
  const long LMAX = LK(keyf(Qlast - 1, (long)(rounds - 1) * NJV + NJ - 1));
  auto window = [&](long L, std::vector<std::vector<Blk>>& out) {
    for (auto& x : out) x.clear();
    for (int p = 0; p < Q; ++p) {
      long g = std::max(0l, (long)std::floor((L * LL - p * LAM) / DL) - p - 1);
      while (g < GTOT && LK(keyf(p, g)) < L) ++g;
      for (; g < GTOT && LK(keyf(p, g)) == L; ++g) { if (p >= tgt(g)) continue; const int d = dev(g); for (int I = 0; I < NI; ++I) out[d].push_back(Blk{p, I, g, keyb(p, I, g)}); }
    }
    for (auto& x : out) std::sort(x.begin(), x.end(), [](const Blk& u, const Blk& w) { return u.key != w.key ? u.key < w.key : u.g != w.g ? u.g < w.g : u.p != w.p ? u.p < w.p : u.I < w.I; });
  };
  long PLb = 0, total_blocks = 0; std::vector<long> nbd(D, 0);
  { std::vector<std::vector<Blk>> w(D); for (long L = 0; L <= LMAX; ++L) { window(L, w); for (int d = 0; d < D; ++d) { PLb = std::max(PLb, (long)w[d].size()); total_blocks += (long)w[d].size(); nbd[d] += (long)w[d].size(); } } }
  need(total_blocks == (long)NI * passes * NJ, "helix block count");
  const long PL = (long)rup((size_t)PLb + 64, 64); long nbmax = 0; for (int d = 0; d < D; ++d) nbmax = std::max(nbmax, nbd[d]);
  const long PST = (long)rup((size_t)(nbmax + 64) * 4, kGR) / 4;
  if ((double)PST * D >= 2.0e9) fail(-1, "host-memory run too long for one call (%d trips): split it into several afdtd_execute calls", passes);

  // ---- device state: one virtual window for rows (fields, coefficients), one for the pools; released at the end of the call
  const afdtd::Driver& DR = afdtd::drv();
  Vmm vm; vm.init(h.devs);
  const size_t FW = (size_t)NJV * RSB, CW = (size_t)NJV * CROW;
  CUdeviceptr fbase = 0, cbase = 0, aux = 0;
  std::vector<CUmemGenericAllocationHandle> hF(NSLOT, 0), hC(NSLOT, 0);
  const int GAPX = (int)((kGR + XBUF - 1) / XBUF), GAPY = (int)((kGR + YBUF - 1) / YBUF);
  const long SSTY = (long)S * NI + 2 * GAPY, SSTX = MX + 1 + 2 * GAPX, YZST = 1 + 2 * GAPY;
  const size_t YHB = rup(((size_t)NSLOT * SSTY + (size_t)D * YZST) * YBUF, kGR), XPB = rup((size_t)D * SSTX * XBUF, kGR), PGB = (size_t)D * PST * 4;
  std::vector<unsigned*> dtick(D, nullptr); std::vector<int*> dord(D, nullptr); std::vector<long long*> dtim(D, nullptr);
  std::vector<cudaStream_t> sc0(D, nullptr), sc1(D, nullptr), sh2d(D, nullptr), sd2h(D, nullptr);
  const int NEV = 8;
  std::vector<std::vector<cudaEvent_t>> evd(D, std::vector<cudaEvent_t>(NEV, nullptr)), evk(D, std::vector<cudaEvent_t>(NEV, nullptr));
  std::vector<cudaEvent_t> evload(NR, nullptr), evfree(NR, nullptr);
  int mapped_rows = 0;
  auto cleanup = [&]() {
    for (int d = 0; d < D; ++d) { cudaSetDevice(DEV(d)); cudaDeviceSynchronize(); }
    for (int s2 = 0; s2 < mapped_rows; ++s2) { DR.MemUnmap(fbase + (size_t)s2 * S * RSB, (size_t)S * RSB); DR.MemUnmap(cbase + (size_t)s2 * S * CROW, (size_t)S * CROW); }
    for (int q = 0; q < NSLOT; ++q) { if (hF[q]) DR.MemRelease(hF[q]); if (hC[q]) DR.MemRelease(hC[q]); }
    vm.release();
    if (fbase) DR.MemAddressFree(fbase, FW); if (cbase) DR.MemAddressFree(cbase, CW); if (aux) DR.MemAddressFree(aux, YHB + XPB + PGB);
    for (int d = 0; d < D; ++d) { cudaSetDevice(DEV(d)); cudaFree(dord[d]); cudaFree(dtim[d]); cudaFree(dtick[d]);
      for (cudaStream_t s2 : {sc0[d], sc1[d], sh2d[d], sd2h[d]}) if (s2) cudaStreamDestroy(s2);
      for (int k = 0; k < NEV; ++k) { if (evd[d][k]) cudaEventDestroy(evd[d][k]); if (evk[d][k]) cudaEventDestroy(evk[d][k]); } }
    for (int r = 0; r < NR; ++r) { cudaSetDevice(DEV(((r / S) % NSLOT) % D)); if (evload[r]) cudaEventDestroy(evload[r]); if (evfree[r]) cudaEventDestroy(evfree[r]); }
  };
  try {
    CU(DR.MemAddressReserve(&fbase, FW, 0, 0, 0)); CU(DR.MemAddressReserve(&cbase, CW, 0, 0, 0));
    for (int q = 0; q < NSLOT; ++q) { const int d = DEV(q % D); CK(cudaSetDevice(d)); CUmemAllocationProp pr = Vmm::prop(d);
      CU(DR.MemCreate(&hF[q], (size_t)S * RSB, &pr, 0)); CU(DR.MemCreate(&hC[q], (size_t)S * CROW, &pr, 0)); }
    for (int s2 = 0; s2 < NJV / S; ++s2) { const int q = s2 % NSLOT;      // virtual stripe s2 -> slot s2 mod NSLOT
      CU(DR.MemMap(fbase + (size_t)s2 * S * RSB, (size_t)S * RSB, 0, hF[q], 0));
      if (DR.MemMap(cbase + (size_t)s2 * S * CROW, (size_t)S * CROW, 0, hC[q], 0) != CUDA_SUCCESS) { DR.MemUnmap(fbase + (size_t)s2 * S * RSB, (size_t)S * RSB); fail(-5, "cuMemMap failed"); }
      ++mapped_rows;
      CU(DR.MemSetAccess(fbase + (size_t)s2 * S * RSB, (size_t)S * RSB, vm.acc.data(), D)); CU(DR.MemSetAccess(cbase + (size_t)s2 * S * CROW, (size_t)S * CROW, vm.acc.data(), D)); }
    for (int q = 0; q < NSLOT; ++q) { CK(cudaSetDevice(DEV(q % D))); CK(cudaMemset((void*)(fbase + (size_t)q * S * RSB), 0, (size_t)S * RSB)); CK(cudaMemset((void*)(cbase + (size_t)q * S * CROW), 0, (size_t)S * CROW)); }
    CU(DR.MemAddressReserve(&aux, YHB + XPB + PGB, 0, 0, 0));
    const CUdeviceptr yhb = aux, xpb = aux + YHB, pgb = xpb + XPB;
    vm.map_by(yhb, YHB, [&](size_t off) { const size_t i = off / YBUF; return DEV(i < (size_t)NSLOT * SSTY ? (int)((i / SSTY) % D) : (int)std::min((size_t)D - 1, (i - (size_t)NSLOT * SSTY) / YZST)); });
    vm.map_by(xpb, XPB, [&](size_t off) { return DEV((int)std::min((size_t)D - 1, off / ((size_t)SSTX * XBUF))); });
    vm.map_by(pgb, PGB, [&](size_t off) { return DEV((int)std::min((size_t)D - 1, off / ((size_t)PST * 4))); });
    unsigned* prog = reinterpret_cast<unsigned*>(pgb);
    auto DONE = [&](int d) { return (int)(d * PST + nbd[d]); };
    auto YHS = [&](int v, int I) { return (int)(((v / S) % NSLOT) * SSTY + GAPY + (v % S) * NI + I); };
    auto YZERO = [&](int d) { return (int)((long)NSLOT * SSTY + (long)d * YZST + GAPY); };
    auto XS = [&](int d, long s2) { return (int)(d * SSTX + GAPX + s2); };
    for (int d = 0; d < D; ++d) { CK(cudaSetDevice(DEV(d))); const unsigned dn = 0x3fffffffu; CK(cudaMemcpy(prog + DONE(d), &dn, 4, cudaMemcpyHostToDevice)); }

    // ---- every block table of the run, built before the run and uploaded once
    std::vector<std::vector<int>> ordAll(D); for (int d = 0; d < D; ++d) ordAll[d].assign((size_t)nbd[d] * 16, 0);
    std::vector<std::vector<long>> loff(D, std::vector<long>(LMAX + 2, 0)), lcnt(D, std::vector<long>(LMAX + 1, 0));
    std::vector<std::vector<long>> lrows(LMAX + 1);            // rows (and ghost rows) touched by launch L
    {
      std::vector<std::vector<Blk>> win(D);
      std::unordered_map<U64, Ref> mp[2];
      std::vector<std::vector<Blk>> xring(D, std::vector<Blk>(MX)); std::vector<long> seqc(D, 0);
      for (long L = 0; L <= LMAX; ++L) {
        window(L, win);
        auto& cur = mp[L & 1]; auto& prv = mp[(L - 1) & 1]; cur.clear();
        for (int d = 0; d < D; ++d) for (size_t i = 0; i < win[d].size(); ++i) { const Blk& b = win[d][i]; cur[bkey(b.g, b.p, b.I)] = Ref{(int)L, d, (int)i, seqc[d] + (long)i}; }
        auto pidx = [&](int p, int I, long g, int dme, long ime, double kme) -> int {     // progress word of block (p, I, g) seen from (dme, ime)
          const long Lb = LK(keyf(p, g));
          if (Lb <= L - 2) return DONE(dme);
          need(Lb <= L, "dependency in a later launch");
          auto& m = Lb == L ? cur : prv; auto it = m.find(bkey(g, p, I)); need(it != m.end(), "dependency block");
          const Ref& r = it->second;
          if (Lb == L && r.d == dme) need(r.loc < ime, "dependency later in the same launch");
          if (Lb == L && r.d != dme) need(keyb(p, I, g) < kme, "cross-device dependency not earlier");
          return (int)(r.d * PST + r.seq); };
        for (int d = 0; d < D; ++d) {
          loff[d][L] = seqc[d]; lcnt[d][L] = (long)win[d].size();
          for (size_t i = 0; i < win[d].size(); ++i) {
            const Blk& b = win[d][i]; const int p = b.p, I = b.I, v = vrow(b.g); const long g = b.g, seq = seqc[d] + (long)i; int* o = &ordAll[d][(size_t)seq * 16];
            if (I == 0) { lrows[L].push_back(g); lrows[L].push_back(g + 1); }
            o[0] = I; o[1] = v;
            if (I > 0) { auto it = cur.find(bkey(g, p, I - 1)); need(it != cur.end(), "XP producer");
              need(seq - it->second.seq < MX && it->second.d == d, "XP ring"); o[2] = XS(d, it->second.seq % MX); o[6] = pidx(p, I - 1, g, d, (long)i, b.key); }
            else { o[2] = XS(d, MX); o[6] = DONE(d); }
            o[3] = XS(d, seq % MX);
            o[4] = v > 0 ? YHS(v - 1, I) : YZERO(d); o[5] = YHS(v, I);
            o[7] = v > 0 ? pidx(p, I, g - 1, d, (long)i, b.key) : DONE(d);
            o[8] = (int)(d * PST + seq);
            o[9] = p > 0 ? pidx(p - 1, std::min(I + 1, NI - 1), v + 1 < NJ ? g + 1 : g, d, (long)i, b.key) : DONE(d);
            o[10] = DONE(d);
            if (seq >= MX) { const Blk x = xring[d][seq % MX]; o[10] = pidx(x.p, x.I + 1 < NI ? x.I + 1 : x.I, x.g, d, (long)i, b.key); }
            xring[d][seq % MX] = b;
            o[11] = DONE(d);
            { int rp = -1; long rg = -1;              // the reader of what my YH slot holds now
              if (p > 0) { rp = p - 1; rg = g + 1; if (v + 1 >= NJ) rg = -1; }
              else if (g - NR >= 0 && vrow(g - NR) < NJ - 1) { rg = g - NR + 1; rp = tgt(g - NR) - 1; }
              if (rg >= 0 && rp >= 0) o[11] = pidx(rp, I, rg, d, (long)i, b.key); }
            o[12] = o[13] = o[14] = o[15] = DONE(d);
          }
          seqc[d] += (long)win[d].size();
        }
      }
      for (int d = 0; d < D; ++d) need(seqc[d] == nbd[d], "per-device block count");
      for (auto& v : lrows) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); }
    }
    for (int d = 0; d < D; ++d) { CK(cudaSetDevice(DEV(d)));
      CK(cudaFuncSetAttribute(stream, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SSMEM));
      CK(cudaMalloc(&dtick[d], (size_t)(LMAX + 1) * 128)); CK(cudaMemset(dtick[d], 0, (size_t)(LMAX + 1) * 128));
      CK(cudaMalloc(&dord[d], ordAll[d].size() * sizeof(int) + 64)); CK(cudaMemcpy(dord[d], ordAll[d].data(), ordAll[d].size() * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMalloc(&dtim[d], (size_t)4 * PL * 16 * sizeof(long long)));
      CK(cudaStreamCreateWithFlags(&sc0[d], cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sc1[d], cudaStreamNonBlocking));
      CK(cudaStreamCreateWithFlags(&sh2d[d], cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sd2h[d], cudaStreamNonBlocking));
      for (int k = 0; k < NEV; ++k) { CK(cudaEventCreateWithFlags(&evd[d][k], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&evk[d][k], cudaEventDisableTiming)); } }
    for (auto& x : ordAll) std::vector<int>().swap(x);
    std::vector<char> evload_set(NR, 0), evfree_set(NR, 0);
    for (int r = 0; r < NR; ++r) { CK(cudaSetDevice(DEV(((r / S) % NSLOT) % D))); CK(cudaEventCreateWithFlags(&evload[r], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&evfree[r], cudaEventDisableTiming)); }

    // ---- run
    long next_load = 0;
    auto issue_load = [&](long g) {
      const int v = vrow(g), r = (int)(g % NR), d = dev(g); CK(cudaSetDevice(DEV(d))); cudaStream_t st = sh2d[d];
      if (evfree_set[r]) CK(cudaStreamWaitEvent(st, evfree[r], 0));
      if (v <= NJ) {
        char* fdst = (char*)(fbase + (size_t)v * RSB); char* cdst = (char*)(cbase + (size_t)v * CROW);
        const char* src = v == NJ ? h.zero : h.rows + (size_t)v * HRS;
        for (size_t o = 0; o < RSB; o += chunk) CK(cudaMemcpyAsync(fdst + o, src + o, std::min(chunk, RSB - o), cudaMemcpyHostToDevice, st));
        if (v < NJ) for (size_t o = 0; o < CSB; o += chunk) CK(cudaMemcpyAsync(cdst + o, src + RSB + o, std::min(chunk, CSB - o), cudaMemcpyHostToDevice, st));
      }
      CK(cudaEventRecord(evload[r], st)); evload_set[r] = 1; };
    auto issue_store = [&](long g, long L) {
      const int v = vrow(g), r = (int)(g % NR), d = dev(g); CK(cudaSetDevice(DEV(d))); cudaStream_t st = sd2h[d];
      for (int e = 0; e < D; ++e) CK(cudaStreamWaitEvent(st, evd[e][L % NEV], 0));
      if (v < NJ) { const char* src = (const char*)(fbase + (size_t)v * RSB); char* dst = h.rows + (size_t)v * HRS;
        for (size_t o = 0; o < RSB; o += chunk) CK(cudaMemcpyAsync(dst + o, src + o, std::min(chunk, RSB - o), cudaMemcpyDeviceToHost, st)); }
      CK(cudaEventRecord(evfree[r], st)); evfree_set[r] = 1; };
    long last_rel_issued = -1;              // launches whose stores are issued
    auto pump_loads = [&](long upto_launch, bool must) {
      while (next_load < GTOT) { const long g = next_load, nd = needl(g);
        if (nd < 0) { if (g - NR >= 0 && rel(g - NR) > last_rel_issued) break; issue_load(g); ++next_load; continue; }
        if (nd > upto_launch) break;
        if (g - NR >= 0 && rel(g - NR) > last_rel_issued) { need(!must, "ring slot not free for a needed row"); break; }
        issue_load(g); ++next_load; } };
    std::vector<std::vector<char>> kstarted(D, std::vector<char>(NEV, 0));
    // watchdog: a launch not finished after kStall seconds is reported as a stall (AFDTD_EDEVICE) instead of hanging the caller.  The
    // run is first released by writing DONE into every progress word (a copy: no SM needed), so the kernels end and the device stays
    // usable; the state is invalid.  The message names the stalled launches (blocks done, blocks claimed) for a bug report.
    const double kStall = 120.0;
    auto stalled = [&](long L) {
      std::string msg; char b[160];
      std::vector<unsigned> pg((size_t)PST * D); cudaStream_t ds = nullptr;
      CK(cudaSetDevice(DEV(0))); CK(cudaStreamCreateWithFlags(&ds, cudaStreamNonBlocking));
      CK(cudaMemcpyAsync(pg.data(), prog, pg.size() * 4, cudaMemcpyDeviceToHost, ds)); CK(cudaStreamSynchronize(ds));
      for (long Lx = L; Lx <= std::min(LMAX, L + 1); ++Lx) for (int d = 0; d < D; ++d) {
        unsigned tk = 0; long nd = 0;
        CK(cudaMemcpyAsync(&tk, dtick[d] + Lx * 32, 4, cudaMemcpyDeviceToHost, ds)); CK(cudaStreamSynchronize(ds));
        for (long i = 0; i < lcnt[d][Lx]; ++i) nd += pg[(size_t)d * PST + loff[d][Lx] + i] >= 0x3fffffffu;
        snprintf(b, sizeof b, "; launch %ld GPU %d: %ld of %ld blocks done, %u claims", Lx, d, nd, lcnt[d][Lx], tk); msg += b;
      }
      std::fill(pg.begin(), pg.end(), 0x3fffffffu);
      CK(cudaMemcpyAsync(prog, pg.data(), pg.size() * 4, cudaMemcpyHostToDevice, ds)); CK(cudaStreamSynchronize(ds)); cudaStreamDestroy(ds);
      fail(-5 /* AFDTD_EDEVICE */, "host-memory execute stalled (%d passes, depth %d, %ld launches%s); the plan's state is invalid: please report it",
           passes, Q, LMAX + 1, msg.c_str());
    };
    auto wait_ev = [&](cudaEvent_t e, long L) {
      const double t0 = now();
      for (cudaError_t r; (r = cudaEventQuery(e)) != cudaSuccess;) {
        if (r != cudaErrorNotReady) CK(r);
        if (now() - t0 > kStall) stalled(L);
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
    };
    pump_loads(0, true); pump_loads(PF, false);
    const double T0 = now();
    for (long L = 0; L <= LMAX; ++L) {
      if (L >= 3) for (int d = 0; d < D; ++d) wait_ev(evd[d][(L - 3) % NEV], L - 3);
      pump_loads(L, true);
      for (int d = 0; d < D; ++d) {
        const long nb = lcnt[d][L];
        CK(cudaSetDevice(DEV(d))); cudaStream_t st = (L & 1) ? sc1[d] : sc0[d];
        for (int e = 0; e < D; ++e) { if (L >= 1 && kstarted[e][(L - 1) % NEV]) CK(cudaStreamWaitEvent(st, evk[e][(L - 1) % NEV], 0));
          if (L >= 2) CK(cudaStreamWaitEvent(st, evd[e][(L - 2) % NEV], 0)); }
        kstarted[d][L % NEV] = 0;
        if (nb > 0) {
          for (long g : lrows[L]) { const int r = (int)(g % NR); if (evload_set[r]) CK(cudaStreamWaitEvent(st, evload[r], 0)); }
          // Launch L - 1 must have a block resident on this GPU before launch L can take SMs: launch L's blocks wait for launch L - 1's,
          // and with no CTA of L - 1 resident, CTAs of L holding every SM would wait forever (seen on H100).  One resident CTA of a
          // persistent launch runs all of its blocks, so this is enough.  Without stream memory operations: wait for L - 1 to finish.
          if (L >= 1 && lcnt[d][L - 1] > 0) {
            if (DR.StreamWaitValue32) CU(DR.StreamWaitValue32((CUstream)st, (CUdeviceptr)(dtick[d] + (L - 1) * 32), 1u, CU_STREAM_WAIT_VALUE_GEQ));
            else CK(cudaStreamWaitEvent(st, evd[d][(L - 1) % NEV], 0));
          }
          CK(cudaEventRecord(evk[d][L % NEV], st)); kstarted[d][L % NEV] = 1;
          KP P; memset(&P, 0, sizeof P); P.nblk = (int)nb; P.pmode = 1; P.tick = dtick[d] + L * 32; P.fld = reinterpret_cast<Rec*>(fbase); P.coe = reinterpret_cast<unsigned char*>(cbase);
          P.xp = reinterpret_cast<unsigned char*>(xpb); P.yh = reinterpret_cast<unsigned char*>(yhb); P.prog = prog; P.tim = dtim[d] + (size_t)(L % 4) * PL * 16; P.ord = dord[d] + (size_t)loff[d][L] * 16;
          P.ROWREC = G.ROWREC; P.PLREC = G.PLREC; P.XBUF = (long)XBUF; P.YBUF = (long)YBUF; P.NB = NB; P.NPAIR = G.NPAIR; P.CROW = (long)CROW;
          launch_stream(P, (unsigned)std::min((long)GRID, nb), st, (void*)fbase, (void*)cbase, G, NJV, NJV, CROW);
        }
        CK(cudaEventRecord(evd[d][L % NEV], st));
      }
      for (long g = std::max(0l, next_load - 2l * NR - 2); g < next_load; ++g) if (rel(g) == L) issue_store(g, L);   // rows released by launch L
      last_rel_issued = L;
      pump_loads(L + 1 + PF, false);
    }
    for (long L = std::max(0l, LMAX - 2); L <= LMAX; ++L) for (int d = 0; d < D; ++d) wait_ev(evd[d][L % NEV], L);
    for (int d = 0; d < D; ++d) { CK(cudaSetDevice(DEV(d))); CK(cudaDeviceSynchronize()); }
    const double wall = now() - T0;
    cleanup();
    return wall;
  } catch (...) { cleanup(); throw; }
}

// The step tail as a sweep over the host rows: each GPU takes a contiguous range and walks it in blocks of rows with one halo row on
// each side.  A lower halo must hold the PRE-tail state of its row, which the previous block has already written back, so each block
// saves its last row before computing; the halos at range boundaries are saved for every range before any GPU starts.  steps < 24:
// the wrong cells at the outer faces of the halos never reach the owned rows.
static void h_tail(HostDomain& h, int r)
{
  if (r <= 0) return;
  need(r < SS, "tail steps < 24");
  const Geo& G = h.G; const int NJ = G.NJ; const size_t RSB = h.RSB, CSB = h.CSB, CROW = h.CROW, HRS = h.HRS;
  const int D = std::min((int)h.devs.size(), NJ);
  size_t fr = 0, tt = 0; CK(cudaSetDevice(h.devs[0])); CK(cudaMemGetInfo(&fr, &tt));
  const int TB = (int)std::max<long long>(1, std::min<long long>(64, (long long)(fr * 0.6 / (RSB + CROW)) - 4));
  std::vector<int> R0(D + 1); for (int d = 0; d <= D; ++d) R0[d] = (int)((long)d * NJ / D);
  std::vector<unsigned char*> savelo(D, nullptr), savehi(D, nullptr);
  auto free_saves = [&]() { for (int d = 0; d < D; ++d) { cudaSetDevice(h.devs[d]); cudaFree(savelo[d]); cudaFree(savehi[d]); } };
  try {
    for (int d = 0; d < D; ++d) { CK(cudaSetDevice(h.devs[d])); CK(cudaMalloc(&savelo[d], RSB)); CK(cudaMalloc(&savehi[d], RSB));
      if (R0[d] > 0) CK(cudaMemcpy(savelo[d], h.rows + (size_t)(R0[d] - 1) * HRS, RSB, cudaMemcpyHostToDevice));
      if (R0[d + 1] < NJ) CK(cudaMemcpy(savehi[d], h.rows + (size_t)R0[d + 1] * HRS, RSB, cudaMemcpyHostToDevice)); }
    // two buffer sets on two streams: block b + 1's loads overlap block b's kernels and write-back.  cur[q] holds the pre-tail last row
    // of the block that saved it; events order the save before the next block's lower-halo copy, and that copy before the block after
    // it overwrites cur[q].
    const int TB2 = std::max(1, TB / 2);
    par(D, [&](int d) {
      CK(cudaSetDevice(h.devs[d])); cudaStream_t st[2] = {nullptr, nullptr}; unsigned char *fb[2] = {nullptr, nullptr}, *cb[2] = {nullptr, nullptr}, *cu[2] = {nullptr, nullptr};
      cudaEvent_t saved[2] = {nullptr, nullptr}, used[2] = {nullptr, nullptr}; bool has_used[2] = {false, false};
      auto rel_all = [&]() { for (int q = 0; q < 2; ++q) { if (st[q]) cudaStreamSynchronize(st[q]); cudaFree(fb[q]); cudaFree(cb[q]); cudaFree(cu[q]);
        if (st[q]) cudaStreamDestroy(st[q]); if (saved[q]) cudaEventDestroy(saved[q]); if (used[q]) cudaEventDestroy(used[q]); } };
      try {
        for (int q = 0; q < 2; ++q) { CK(cudaStreamCreateWithFlags(&st[q], cudaStreamNonBlocking)); CK(cudaMalloc(&fb[q], (size_t)(TB2 + 2) * RSB)); CK(cudaMalloc(&cb[q], (size_t)(TB2 + 2) * CROW));
          CK(cudaMalloc(&cu[q], RSB)); CK(cudaEventCreateWithFlags(&saved[q], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&used[q], cudaEventDisableTiming)); }
        CK(cudaMemcpy(cu[1], savelo[d], RSB, cudaMemcpyDeviceToDevice));
        int b = 0;
        for (int J0 = R0[d]; J0 < R0[d + 1]; J0 += TB2, ++b) {
          const int J1 = std::min(J0 + TB2, R0[d + 1]), q = b & 1; cudaStream_t S = st[q];
          auto frow = [&](int J) { return fb[q] + (size_t)(J - (J0 - 1)) * RSB; };
          auto crow = [&](int J) { return cb[q] + (size_t)(J - (J0 - 1)) * CROW; };
          for (int J = J0; J < J1; ++J) { CK(cudaMemcpyAsync(frow(J), h.rows + (size_t)J * HRS, RSB, cudaMemcpyHostToDevice, S)); CK(cudaMemcpyAsync(crow(J), h.rows + (size_t)J * HRS + RSB, CSB, cudaMemcpyHostToDevice, S)); }
          if (J1 < NJ) { if (J1 == R0[d + 1]) CK(cudaMemcpyAsync(frow(J1), savehi[d], RSB, cudaMemcpyDeviceToDevice, S)); else CK(cudaMemcpyAsync(frow(J1), h.rows + (size_t)J1 * HRS, RSB, cudaMemcpyHostToDevice, S));
            CK(cudaMemcpyAsync(crow(J1), h.rows + (size_t)J1 * HRS + RSB, CSB, cudaMemcpyHostToDevice, S)); }
          if (J0 > 0) { CK(cudaMemcpyAsync(crow(J0 - 1), h.rows + (size_t)(J0 - 1) * HRS + RSB, CSB, cudaMemcpyHostToDevice, S));
            if (b > 0) CK(cudaStreamWaitEvent(S, saved[1 - q], 0));
            CK(cudaMemcpyAsync(frow(J0 - 1), cu[1 - q], RSB, cudaMemcpyDeviceToDevice, S)); }
          CK(cudaEventRecord(used[1 - q], S)); has_used[1 - q] = true;
          if (b > 0 && has_used[q]) CK(cudaStreamWaitEvent(S, used[q], 0));
          CK(cudaMemcpyAsync(cu[q], frow(J1 - 1), RSB, cudaMemcpyDeviceToDevice, S));       // pre-tail state of the next block's lower halo
          CK(cudaEventRecord(saved[q], S));
          TailGeo T; T.nx = G.nx; T.ny = G.ny; T.nz = G.nz; T.ROWREC = G.ROWREC; T.PLREC = G.PLREC; T.PS = G.PS; T.crow = CROW;
          T.Jb0 = J0 - 1; T.Jlo = std::max(0, J0 - 1); T.Jhi = std::min(NJ, J1 + 1);
          unsigned* F = reinterpret_cast<unsigned*>(fb[q]);       // buffer row 0 = global row J0 - 1
          for (int s2 = 0; s2 < r; ++s2) {
            tail_h_k<<<tail_grid(T, T.Jlo, T.Jhi), 256, 0, S>>>(T, F, T.Jlo, T.Jhi);
            tail_e_k<<<tail_grid(T, T.Jlo, T.Jhi), 256, 0, S>>>(T, F, cb[q], T.Jlo, T.Jhi);
          }
          CK(cudaGetLastError());
          for (int J = J0; J < J1; ++J) CK(cudaMemcpyAsync(h.rows + (size_t)J * HRS, frow(J), RSB, cudaMemcpyDeviceToHost, S));
        }
        for (int q = 0; q < 2; ++q) CK(cudaStreamSynchronize(st[q]));
      } catch (...) { rel_all(); throw; }
      rel_all(); });
  } catch (...) { free_saves(); throw; }
  free_saves();
}

}  // namespace AFDTD_NS

extern const afdtd::Object AFDTD_SYM;
const afdtd::Object AFDTD_SYM = {AFDTD_NAME, AFDTD_ARCH, AFDTD_CC, AFDTD_NS::m_create, AFDTD_NS::m_destroy, AFDTD_NS::m_set_knobs, AFDTD_NS::m_box,
                                 AFDTD_NS::m_advance, AFDTD_NS::m_agents_now, AFDTD_NS::m_tail, AFDTD_NS::m_checksums, AFDTD_NS::m_zero, AFDTD_NS::h_run, AFDTD_NS::h_tail,
                                 AFDTD_PREC_CODE};
