// SPDX-License-Identifier: MIT
// f32.cu -- the FP32 executor (AFDTD_FP32): the state as 12 dense float arrays (component, then x, y, z in C order, z fastest), one
// kernel per half step computing the update of alpha_fdtd.h section 1 with correctly rounded float operations in its order.  1 to 8
// GPUs: x slabs of one virtual-memory window (the seam planes are read through peer mappings); events order a GPU's H half step after
// its x + 1 neighbour's E half step of the previous step, and its E half step after its x - 1 neighbour's H half step.  Agent work
// (recorders' copies, injectors' adds) runs in stream order between the steps: there is no persistent launch to work beside.  One
// Object for every supported architecture (plain CUDA, SASS per family).
#include "alpha_fdtd.h"
#include "internal.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace afdtd {
namespace f32 {

typedef unsigned long long U64;
struct D3 { int nx, ny, nz, xa, xb; size_t n; };          // the domain; this GPU's x slab [xa, xb); cells per component

__device__ __forceinline__ float at(const float* p, int i, int j, int k, const D3& g)
{ return (i < 0 || i >= g.nx || j < 0 || j >= g.ny || k < 0 || k >= g.nz) ? 0.f : p[((size_t)i * g.ny + j) * g.nz + k]; }
// grid-stride over y and x: ny and the x slab may exceed 65535 (the gridDim.y, gridDim.z limits)
#define F32_CELL const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= g.nz) return; \
                 for (int i = g.xa + (int)blockIdx.z; i < g.xb; i += gridDim.z) for (int j = blockIdx.y; j < g.ny; j += gridDim.y) { \
                 const size_t x = ((size_t)i * g.ny + j) * g.nz + k;
#define F32_END }
__global__ void step_h(float* s, D3 g)
{
  F32_CELL
  float* const E = s; float* const H = s + 3 * g.n;
  const float ex = E[x], ey = E[g.n + x], ez = E[2 * g.n + x];
  const float cx = __fsub_rn(__fsub_rn(at(E + 2 * g.n, i, j + 1, k, g), ez), __fsub_rn(at(E + g.n, i, j, k + 1, g), ey));
  const float cy = __fsub_rn(__fsub_rn(at(E, i, j, k + 1, g), ex), __fsub_rn(at(E + 2 * g.n, i + 1, j, k, g), ez));
  const float cz = __fsub_rn(__fsub_rn(at(E + g.n, i + 1, j, k, g), ey), __fsub_rn(at(E, i, j + 1, k, g), ex));
  H[x] = __fsub_rn(H[x], cx); H[g.n + x] = __fsub_rn(H[g.n + x], cy); H[2 * g.n + x] = __fsub_rn(H[2 * g.n + x], cz);
  F32_END
}
__global__ void step_e(float* s, D3 g)
{
  F32_CELL
  float* const E = s; const float* const H = s + 3 * g.n; const float* const Ca = s + 6 * g.n; const float* const Cb = s + 9 * g.n;
  const float hx = H[x], hy = H[g.n + x], hz = H[2 * g.n + x];
  const float cx = __fsub_rn(__fsub_rn(hz, at(H + 2 * g.n, i, j - 1, k, g)), __fsub_rn(hy, at(H + g.n, i, j, k - 1, g)));
  const float cy = __fsub_rn(__fsub_rn(hx, at(H, i, j, k - 1, g)), __fsub_rn(hz, at(H + 2 * g.n, i - 1, j, k, g)));
  const float cz = __fsub_rn(__fsub_rn(hy, at(H + g.n, i - 1, j, k, g)), __fsub_rn(hx, at(H, i, j - 1, k, g)));
  E[x] = __fmaf_rn(Cb[x], cx, __fmul_rn(Ca[x], E[x]));
  E[g.n + x] = __fmaf_rn(Cb[g.n + x], cy, __fmul_rn(Ca[g.n + x], E[g.n + x]));
  E[2 * g.n + x] = __fmaf_rn(Cb[2 * g.n + x], cz, __fmul_rn(Ca[2 * g.n + x], E[2 * g.n + x]));
  F32_END
}
__host__ __device__ __forceinline__ U64 mix64(U64 x)
{
  x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
// checksums of the six fields over the cells with x in [xa, xb) (alpha_fdtd.h: the state's bit patterns, 32 bits here)
__global__ void sums_k(const unsigned* s, D3 g, U64* out)
{
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x, n0 = (size_t)g.xa * g.ny * g.nz, n1 = (size_t)g.xb * g.ny * g.nz;
  U64 a[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 64; ++i) {
    const size_t idx = n0 + t * 64 + i;
    if (idx >= n1) break;
    for (int c = 0; c < 6; ++c) a[c] += mix64(((((U64)(c % 3)) * g.n + idx) << 32) + s[c * g.n + idx]);
  }
  if (n0 + t * 64 < n1) for (int c = 0; c < 6; ++c) atomicAdd(&out[c], a[c]);
}

struct St {
  int nx, ny, nz, ng; size_t n, bytes, win = 0; std::vector<int> devs, X0;
  CUdeviceptr base = 0; std::vector<CUmemGenericAllocationHandle> h; struct Map { CUdeviceptr va; size_t sz; int dev; }; std::vector<Map> maps;
  float* s = nullptr;                                     // the state: 12 * n floats
  std::vector<cudaStream_t> st; std::vector<cudaEvent_t> evh, eve, evx; std::vector<U64*> dsum;
  D3 d3(int g) const { D3 d; d.nx = nx; d.ny = ny; d.nz = nz; d.n = n; d.xa = X0[g]; d.xb = X0[g + 1]; return d; }
};
static void sync_all(St& s) { for (size_t g = 0; g < s.devs.size(); ++g) { AF_CK(cudaSetDevice(s.devs[g])); AF_CK(cudaStreamSynchronize(s.st[g])); } }

static void destroy(void* p)
{
  St* s = static_cast<St*>(p); if (!s) return;
  for (size_t g = 0; g < s->devs.size(); ++g) {
    cudaSetDevice(s->devs[g]);
    if (g < s->st.size() && s->st[g]) { cudaStreamSynchronize(s->st[g]); cudaStreamDestroy(s->st[g]); }
    if (g < s->evh.size() && s->evh[g]) cudaEventDestroy(s->evh[g]);
    if (g < s->eve.size() && s->eve[g]) cudaEventDestroy(s->eve[g]);
    if (g < s->evx.size() && s->evx[g]) cudaEventDestroy(s->evx[g]);
    if (g < s->dsum.size()) cudaFree(s->dsum[g]);
  }
  if (s->ng == 1) { if (s->s) { cudaSetDevice(s->devs[0]); cudaFree(s->s); } }
  else {
    const Driver& D = drv();
    for (auto& m : s->maps) D.MemUnmap(m.va, m.sz);
    for (auto x : s->h) D.MemRelease(x);
    if (s->base) D.MemAddressFree(s->base, s->win);
  }
  delete s;
}
// x slabs: GPU g owns [X0[g], X0[g + 1]); in the window, a granule belongs to the GPU owning the x plane of its first byte
static void* create(const Geo& G, const int* devs, int n, const Knobs&)
{
  St* s = new St;
  try {
    s->nx = G.nx; s->ny = G.ny; s->nz = G.nz; s->ng = n; s->devs.assign(devs, devs + n);
    s->n = (size_t)G.nx * G.ny * G.nz; s->bytes = 12 * s->n * sizeof(float);
    s->X0.assign(n + 1, 0); for (int g = 0; g <= n; ++g) s->X0[g] = (int)((long)g * G.nx / n);
    if (n == 1) {
      AF_CK(cudaSetDevice(devs[0])); AF_CK(cudaMalloc(&s->s, s->bytes)); AF_CK(cudaMemset(s->s, 0, s->bytes));
    } else {
      const Driver& D = drv();
      const size_t plane = (size_t)G.ny * G.nz * sizeof(float), comp = s->n * sizeof(float);
      s->win = rup(s->bytes, kGR);
      AF_CU(D.MemAddressReserve(&s->base, s->win, 0, 0, 0));
      std::vector<CUmemAccessDesc> acc(n);
      for (int g = 0; g < n; ++g) { memset(&acc[g], 0, sizeof acc[g]); acc[g].location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc[g].location.id = devs[g]; acc[g].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
      auto owner = [&](size_t off) { const size_t x = (off % comp) / plane; int g = 0; while (g + 1 < n && (size_t)s->X0[g + 1] <= x) ++g; return g; };
      for (size_t q0 = 0, nq = s->win / kGR; q0 < nq;) {
        const int o = owner(q0 * kGR); size_t q1 = q0 + 1; while (q1 < nq && owner(q1 * kGR) == o) ++q1;
        CUmemAllocationProp pr; memset(&pr, 0, sizeof pr); pr.type = CU_MEM_ALLOCATION_TYPE_PINNED; pr.location.type = CU_MEM_LOCATION_TYPE_DEVICE; pr.location.id = devs[o];
        CUmemGenericAllocationHandle hh; AF_CK(cudaSetDevice(devs[o])); AF_CU(D.MemCreate(&hh, (q1 - q0) * kGR, &pr, 0)); s->h.push_back(hh);
        AF_CU(D.MemMap(s->base + q0 * kGR, (q1 - q0) * kGR, 0, hh, 0)); s->maps.push_back({s->base + q0 * kGR, (q1 - q0) * kGR, devs[o]});
        AF_CU(D.MemSetAccess(s->base + q0 * kGR, (q1 - q0) * kGR, acc.data(), n));
        q0 = q1;
      }
      s->s = reinterpret_cast<float*>(s->base);
      for (auto& m : s->maps) { AF_CK(cudaSetDevice(m.dev)); AF_CK(cudaMemset((void*)m.va, 0, m.sz)); }
      for (int g = 0; g < n; ++g) { AF_CK(cudaSetDevice(devs[g])); AF_CK(cudaDeviceSynchronize()); }
    }
    s->st.assign(n, nullptr); s->evh.assign(n, nullptr); s->eve.assign(n, nullptr); s->evx.assign(n, nullptr); s->dsum.assign(n, nullptr);
    for (int g = 0; g < n; ++g) {
      AF_CK(cudaSetDevice(devs[g])); AF_CK(cudaStreamCreateWithFlags(&s->st[g], cudaStreamNonBlocking));
      AF_CK(cudaEventCreateWithFlags(&s->evh[g], cudaEventDisableTiming)); AF_CK(cudaEventCreateWithFlags(&s->eve[g], cudaEventDisableTiming)); AF_CK(cudaEventCreateWithFlags(&s->evx[g], cudaEventDisableTiming));
      AF_CK(cudaMalloc(&s->dsum[g], 64));
    }
    return s;
  } catch (...) { destroy(s); throw; }
}
static void set_knobs(void*, const Knobs&) {}
// a dense box (32-bit values, NULL components skipped) in (dir 0) or out (dir 1): one 3-D copy per component and GPU slab
static void box(void* p, const int64_t off[3], const int64_t shape[3], uint16_t* const comp16[12], int dir)
{
  St& s = *static_cast<St*>(p);
  sync_all(s);
  for (int c = 0; c < 12; ++c) {
    float* hb = reinterpret_cast<float*>(comp16[c]);
    if (!hb) continue;
    for (int g = 0; g < s.ng; ++g) {
      const int64_t xa = std::max<int64_t>(off[0], s.X0[g]), xb = std::min<int64_t>(off[0] + shape[0], s.X0[g + 1]);
      if (xa >= xb) continue;
      AF_CK(cudaSetDevice(s.devs[g]));
      float* dv = s.s + (size_t)c * s.n;
      cudaMemcpy3DParms m; memset(&m, 0, sizeof m);
      cudaPitchedPtr dp = make_cudaPitchedPtr(dv, (size_t)s.nz * 4, (size_t)s.nz, (size_t)s.ny);
      cudaPitchedPtr hp = make_cudaPitchedPtr(hb, (size_t)shape[2] * 4, (size_t)shape[2], (size_t)shape[1]);
      const cudaPos dpos = make_cudaPos((size_t)off[2] * 4, (size_t)off[1], (size_t)xa), hpos = make_cudaPos(0, 0, (size_t)(xa - off[0]));
      if (dir == 0) { m.srcPtr = hp; m.srcPos = hpos; m.dstPtr = dp; m.dstPos = dpos; m.kind = cudaMemcpyHostToDevice; }
      else { m.srcPtr = dp; m.srcPos = dpos; m.dstPtr = hp; m.dstPos = hpos; m.kind = cudaMemcpyDeviceToHost; }
      m.extent = make_cudaExtent((size_t)shape[2] * 4, (size_t)shape[1], (size_t)(xb - xa));
      AF_CK(cudaMemcpy3DAsync(&m, s.st[g]));
    }
  }
  sync_all(s);
}
static AgentDev agent_dev(const St& s, int g)
{
  AgentDev d; memset(&d, 0, sizeof d);
  d.g = g; d.dev = s.devs[g]; d.fld = reinterpret_cast<char*>(s.s); d.st = s.st[g]; d.prec = AFDTD_F32;
  d.nx = s.nx; d.ny = s.ny; d.nz = s.nz; d.Xa = s.X0[g]; d.Xb = s.X0[g + 1]; d.Ja = 0; d.Jb = 0;
  return d;
}
// one step on every GPU; then, when h wants it, the agent work at plan step t (injections, then copies) on each GPU's stream
static void one_step(St& s, int64_t t, const AgentHooks* h)
{
  const unsigned bz = 128;
  for (int g = 0; g < s.ng; ++g) {
    AF_CK(cudaSetDevice(s.devs[g]));
    if (g + 1 < s.ng) AF_CK(cudaStreamWaitEvent(s.st[g], s.eve[g + 1], 0));         // E of x + 1 (last step) is final; H of mine read it
    const D3 d = s.d3(g);
    step_h<<<dim3((s.nz + bz - 1) / bz, std::min(s.ny, 65535), std::min(d.xb - d.xa, 65535)), bz, 0, s.st[g]>>>(s.s, d);
    AF_CK(cudaEventRecord(s.evh[g], s.st[g]));
  }
  for (int g = 0; g < s.ng; ++g) {
    AF_CK(cudaSetDevice(s.devs[g]));
    if (g > 0) AF_CK(cudaStreamWaitEvent(s.st[g], s.evh[g - 1], 0));                // H of x - 1 is final, and done reading my E
    const D3 d = s.d3(g);
    step_e<<<dim3((s.nz + bz - 1) / bz, std::min(s.ny, 65535), std::min(d.xb - d.xa, 65535)), bz, 0, s.st[g]>>>(s.s, d);
    AF_CK(cudaEventRecord(s.evx[g], s.st[g]));               // my E half step is done (it has read H of x - 1)
  }
  AF_CK(cudaGetLastError());
  const bool work = h && h->want(t);
  if (work) h->begin(t);
  for (int g = 0; g < s.ng; ++g) {
    AF_CK(cudaSetDevice(s.devs[g]));
    if (work && g + 1 < s.ng) AF_CK(cudaStreamWaitEvent(s.st[g], s.evx[g + 1], 0));   // x + 1 has read my H (its E half step) before my injection changes it
    if (work) { const AgentDev a = agent_dev(s, g); h->work(t, a, -1, -1); h->work(t, a, 0, 1 << 30); }
    AF_CK(cudaEventRecord(s.eve[g], s.st[g]));
  }
  if (work) h->end(t);
}
static double advance(void* p, int64_t passes, int64_t step0, const AgentHooks* h)
{
  St& s = *static_cast<St*>(p);
  const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  for (int64_t k = 1; k <= 7 * passes; ++k) one_step(s, step0 + k, h);
  sync_all(s);
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
}
static void tail(void* p, int steps) { St& s = *static_cast<St*>(p); for (int k = 0; k < steps; ++k) one_step(s, 0, nullptr); sync_all(s); }
static void agents_now(void* p, int64_t t, const AgentHooks& h)
{
  St& s = *static_cast<St*>(p);
  sync_all(s);
  h.begin(t);
  for (int g = 0; g < s.ng; ++g) { AF_CK(cudaSetDevice(s.devs[g])); const AgentDev a = agent_dev(s, g); h.work(t, a, -1, -1); h.work(t, a, 0, 1 << 30); }
  h.end(t);
  sync_all(s);
}
static void checksums(void* p, uint64_t out[6])
{
  St& s = *static_cast<St*>(p);
  sync_all(s);
  U64 tot[6] = {0, 0, 0, 0, 0, 0};
  for (int g = 0; g < s.ng; ++g) {
    AF_CK(cudaSetDevice(s.devs[g])); AF_CK(cudaMemsetAsync(s.dsum[g], 0, 64, s.st[g]));
    const D3 d = s.d3(g); const size_t cells = (size_t)(d.xb - d.xa) * s.ny * s.nz;
    sums_k<<<(unsigned)((cells / 64 + 256) / 256), 256, 0, s.st[g]>>>(reinterpret_cast<const unsigned*>(s.s), d, s.dsum[g]);
    AF_CK(cudaGetLastError());
    U64 hv[8]; AF_CK(cudaMemcpyAsync(hv, s.dsum[g], 64, cudaMemcpyDeviceToHost, s.st[g])); AF_CK(cudaStreamSynchronize(s.st[g]));
    for (int c = 0; c < 6; ++c) tot[c] += hv[c];
  }
  for (int c = 0; c < 6; ++c) out[c] = tot[c];
}
static void zero(void* p)
{
  St& s = *static_cast<St*>(p); sync_all(s);
  if (s.ng == 1) { AF_CK(cudaSetDevice(s.devs[0])); AF_CK(cudaMemset(s.s, 0, s.bytes)); }
  else for (auto& m : s.maps) { AF_CK(cudaSetDevice(m.dev)); AF_CK(cudaMemset((void*)m.va, 0, m.sz)); }
  for (int d : s.devs) { AF_CK(cudaSetDevice(d)); AF_CK(cudaDeviceSynchronize()); }
}

}  // namespace f32

extern const Object kF32Object;
const Object kF32Object = {"f32", "sm_80+", 0, f32::create, f32::destroy, f32::set_knobs, f32::box, f32::advance, f32::agents_now, f32::tail,
                           f32::checksums, f32::zero, nullptr, nullptr, AFDTD_F32};
size_t f32_bytes(int64_t nx, int64_t ny, int64_t nz, int ngpus)        // per GPU, the busiest
{
  const size_t n = (size_t)nx * ny * nz, plane = (size_t)ny * nz * 4 * 12;
  return ngpus == 1 ? 12 * n * 4 : (size_t)((nx + ngpus - 1) / ngpus) * plane + 12 * kGR + 64;
}

}  // namespace afdtd
