// SPDX-License-Identifier: MIT
// inject.cu -- injectors (alpha_fdtd.h section 8): at plan step t = first_step + 7 k (row k of the weights, if not all zero), after
// the step, every E and H' value v of the box (tangential E on the lower walls excepted: it stays +0)
//     acc = 0;  for j = 0 .. nbasis - 1:  acc = acc + w[k][j] * G_j(v)        (float32, each * and + correctly rounded)
//     v   = bf16_rn( f32(v) + acc )                                           (the + in float32, then round to nearest even)
// The adds run on the agent stream (object.cu) inside the launches, on the one SM the persistent grid leaves free while an
// injector has work in the call (plan.cpp).
#include "alpha_fdtd.h"
#include "internal.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace afdtd {

struct Injector {
  int64_t off[3], shape[3], first = 0, nrows = 0, last = -1;
  int M = 0, J0 = 0, J1 = 0, q0 = 0, q1 = 0;
  size_t cells = 0;
  std::vector<float> w;                          // nrows * M
  std::vector<char> live;                        // row k has a non-zero weight
  std::vector<int> devs; std::vector<float*> G;  // basis per device: device memory, or mapped pinned host memory (one copy, shared)
  float* host = nullptr; bool mapped = false;
  bool due(int64_t t) const
  {
    if (t <= last || t < first || (t - first) % kK) return false;
    const int64_t k = (t - first) / kK;
    return k < nrows && live[(size_t)k];
  }
  ~Injector()
  {
    for (size_t g = 0; g < G.size(); ++g) if (G[g] && G[g] != host) { cudaSetDevice(devs[g]); cudaFree(G[g]); }
    if (host) cudaFreeHost(host);
  }
};

namespace {
constexpr int kMaxBasis = 64;
struct Wts { float w[kMaxBasis]; };
__device__ __forceinline__ bool wall_e(int c, int x, int y, int z) { return c < 3 && ((x == 0 && c != 0) || (y == 0 && c != 1) || (z == 0 && c != 2)); }
// cells of the box with y in [ya, yb): one thread per cell, all six components
__global__ void inject_k(char* fld, long ROWREC, long PLREC, const float* G, size_t cells, int M, Wts W, int x0, int y0, int z0, int sx, int sy,
                         int sz, int ya, int yb, int prec)
{
  const size_t ny = (size_t)(yb - ya), i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)sx * ny * sz) return;
  const int z = z0 + (int)(i % sz), y = ya + (int)((i / sz) % ny), x = x0 + (int)(i / (sz * ny));
  const size_t li = ((size_t)(x - x0) * sy + (y - y0)) * sz + (z - z0);
  const size_t rec = (size_t)(y / kS) * ROWREC + (size_t)(z + kSOFF) * PLREC + (size_t)(x / 2 + kGOFF) * kS + y % kS;
  unsigned short* r = reinterpret_cast<unsigned short*>(fld + rec * 24) + (x & 1);
  for (int c = 0; c < 6; ++c) {
    if (wall_e(c, x, y, z)) continue;
    float acc = 0.f;
    for (int j = 0; j < M; ++j) acc = __fadd_rn(acc, __fmul_rn(W.w[j], G[((size_t)j * 6 + c) * cells + li]));
    if (prec == AFDTD_F16) r[2 * c] = __half_as_ushort(__float2half_rn(__fadd_rn(__half2float(__ushort_as_half(r[2 * c])), acc)));
    else r[2 * c] = __bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(__bfloat162float(__ushort_as_bfloat16(r[2 * c])), acc)));
  }
}
// FP32 plans: dense arrays (component, then x, y, z), the cells of the box with x in [xa, xb)
__global__ void inject_f32_k(float* s, size_t N, int ny, int nz, const float* G, size_t cells, int M, Wts W, int x0, int y0, int z0, int sx, int sy,
                             int sz, int xa, int xb)
{
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nxs = (size_t)(xb - xa);
  if (i >= nxs * sy * sz) return;
  const int z = z0 + (int)(i % sz), y = y0 + (int)(i / sz % sy), x = xa + (int)(i / ((size_t)sz * sy));
  const size_t li = ((size_t)(x - x0) * sy + (y - y0)) * sz + (z - z0), cell = ((size_t)x * ny + y) * nz + z;
  for (int c = 0; c < 6; ++c) {
    if (wall_e(c, x, y, z)) continue;
    float acc = 0.f;
    for (int j = 0; j < M; ++j) acc = __fadd_rn(acc, __fmul_rn(W.w[j], G[((size_t)j * 6 + c) * cells + li]));
    s[(size_t)c * N + cell] = __fadd_rn(s[(size_t)c * N + cell], acc);
  }
}
Injector& get(Plan& p, int id)
{
  if (id < 0 || id >= (int)p.inj.size() || !p.inj[(size_t)id]) fail(AFDTD_EINVAL, "no open injector with id %d", id);
  return *p.inj[(size_t)id];
}
}  // namespace

int inject_open(Plan& p, const InjOptions& o)
{
  if (p.on_host) fail(AFDTD_EINVAL, "injectors need the domain in GPU memory (this plan keeps it in host memory)");
  for (int a = 0; a < 3; ++a)
    if (o.shape[a] < 1 || o.off[a] < 0 || o.off[a] + o.shape[a] > p.n[a])
      fail(AFDTD_EINVAL, "injector box offset %lld shape %lld on axis %d is outside the %lld-cell domain", (long long)o.off[a], (long long)o.shape[a], a, (long long)p.n[a]);
  if (o.M < 1 || o.M > kMaxBasis) fail(AFDTD_EINVAL, "nbasis = %d (1 to %d)", o.M, kMaxBasis);
  if (!o.basis || !o.weights) fail(AFDTD_EINVAL, "basis and weights are required");
  if (o.first < 0 || o.first % kK) fail(AFDTD_EINVAL, "first_step = %lld: a multiple of 7 (>= 0) is required", (long long)o.first);
  if (o.nrows < 1) fail(AFDTD_EINVAL, "rows = %lld (>= 1)", (long long)o.nrows);
  auto j = std::make_shared<Injector>();
  for (int a = 0; a < 3; ++a) { j->off[a] = o.off[a]; j->shape[a] = o.shape[a]; }
  j->M = o.M; j->first = o.first; j->nrows = o.nrows; j->last = p.steps_done - 1;
  j->cells = (size_t)o.shape[0] * o.shape[1] * o.shape[2];
  j->J0 = (int)(o.off[1] / kS); j->J1 = (int)((o.off[1] + o.shape[1] - 1) / kS + 1);
  j->q0 = (int)(o.off[0] / 2); j->q1 = (int)((o.off[0] + o.shape[0] - 1) / 2 + 1);
  j->w.assign(o.weights, o.weights + (size_t)o.nrows * o.M);
  j->live.assign((size_t)o.nrows, 0);
  for (int64_t k = 0; k < o.nrows; ++k) for (int i = 0; i < o.M; ++i) if (j->w[(size_t)k * o.M + i] != 0.f) j->live[(size_t)k] = 1;
  const size_t bytes = (size_t)o.M * 6 * j->cells * sizeof(float);
  j->devs = p.devs; j->G.assign(p.devs.size(), nullptr);
  bool dev_ok = true;
  for (size_t g = 0; g < p.devs.size() && dev_ok; ++g) {
    AF_CK(cudaSetDevice(p.devs[g]));
    size_t fr = 0, tt = 0; AF_CK(cudaMemGetInfo(&fr, &tt));
    if (fr < bytes + ((size_t)256 << 20) || cudaMalloc(&j->G[g], bytes) != cudaSuccess) { cudaGetLastError(); j->G[g] = nullptr; dev_ok = false; break; }
    AF_CK(cudaMemcpy(j->G[g], o.basis, bytes, cudaMemcpyHostToDevice));
  }
  if (!dev_ok) {                                  // the GPUs are full (the domain): the basis in mapped pinned host memory
    for (size_t g = 0; g < p.devs.size(); ++g) if (j->G[g]) { cudaSetDevice(p.devs[g]); cudaFree(j->G[g]); j->G[g] = nullptr; }
    if (cudaHostAlloc(&j->host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
      cudaGetLastError(); j->host = nullptr; fail(AFDTD_ENOMEM, "the injector basis (%zu bytes) fits neither on the GPUs nor in pinned host memory", bytes);
    }
    memcpy(j->host, o.basis, bytes); j->mapped = true;
    for (size_t g = 0; g < p.devs.size(); ++g) j->G[g] = j->host;      // unified addressing: the same pointer on every device
  }
  // Load the kernel now, with no persistent launch running: under CUDA's lazy module loading the first launch of a kernel may need a
  // context synchronization, which would wait forever behind a persistent grid that waits for this kernel.
  for (size_t g = 0; g < p.devs.size(); ++g) {
    AF_CK(cudaSetDevice(p.devs[g]));
    Wts W; memset(&W, 0, sizeof W);
    inject_k<<<1, 32>>>(nullptr, 0, 0, j->G[g], 0, 0, W, 0, 0, 0, 0, 1, 1, 0, 0, AFDTD_BF16);
    inject_f32_k<<<1, 32>>>(nullptr, 0, 1, 1, j->G[g], 0, 0, W, 0, 0, 0, 0, 1, 1, 0, 0);
    AF_CK(cudaGetLastError()); AF_CK(cudaDeviceSynchronize());
  }
  p.inj.push_back(j);
  return (int)p.inj.size() - 1;
}
void inject_close(Plan& p, int id) { get(p, id); p.inj[(size_t)id].reset(); }
void inject_close_all(Plan& p) noexcept { for (auto& j : p.inj) j.reset(); }
bool inject_active(const Plan& p) { for (auto& j : p.inj) if (j) return true; return false; }
bool inject_due_between(const Plan& p, int64_t t0, int64_t t1)
{
  for (auto& j : p.inj) if (j) for (int64_t t = std::max<int64_t>(t0, j->first) / kK * kK; t <= t1; t += kK) if (t >= t0 && j->due(t)) return true;
  return false;
}

AgentHooks inject_hooks(Plan& p)
{
  AgentHooks h; Plan* pp = &p;
  h.want = [pp](int64_t t) { for (auto& j : pp->inj) if (j && j->due(t)) return true; return false; };
  h.regions = [pp](int64_t t, std::vector<Region>& out) { for (auto& j : pp->inj) if (j && j->due(t)) out.push_back(Region{j->J0, j->J1, j->q0, j->q1, true}); };
  h.begin = [](int64_t) {};
  h.work = [pp](int64_t t, const AgentDev& d, int, int) {
    for (auto& jj : pp->inj) {
      if (!jj || !jj->due(t)) continue;
      Injector& j = *jj;
      if (d.prec == AFDTD_F32) {
        const int xa = (int)std::max<int64_t>(j.off[0], d.Xa), xb = (int)std::min<int64_t>(j.off[0] + j.shape[0], d.Xb);
        if (xa >= xb) continue;
        Wts W; memset(&W, 0, sizeof W);
        const int64_t k = (t - j.first) / kK;
        for (int i = 0; i < j.M; ++i) W.w[i] = j.w[(size_t)k * j.M + i];
        const size_t n = (size_t)(xb - xa) * (size_t)j.shape[1] * (size_t)j.shape[2];
        inject_f32_k<<<(unsigned)((n + 255) / 256), 256, 0, d.st>>>(reinterpret_cast<float*>(d.fld), (size_t)d.nx * d.ny * d.nz, (int)d.ny, (int)d.nz, j.G[(size_t)d.g],
                                                                    j.cells, j.M, W, (int)j.off[0], (int)j.off[1], (int)j.off[2], (int)j.shape[0], (int)j.shape[1],
                                                                    (int)j.shape[2], xa, xb);
        AF_CK(cudaGetLastError());
        continue;
      }
      const int ya = (int)std::max<int64_t>(j.off[1], (int64_t)d.Ja * kS), yb = (int)std::min<int64_t>(j.off[1] + j.shape[1], (int64_t)d.Jb * kS);
      if (ya >= yb) continue;
      Wts W; memset(&W, 0, sizeof W);
      const int64_t k = (t - j.first) / kK;
      for (int i = 0; i < j.M; ++i) W.w[i] = j.w[(size_t)k * j.M + i];
      const size_t n = (size_t)j.shape[0] * (size_t)(yb - ya) * (size_t)j.shape[2];
      inject_k<<<(unsigned)((n + 255) / 256), 256, 0, d.st>>>(d.fld, d.ROWREC, d.PLREC, j.G[(size_t)d.g], j.cells, j.M, W, (int)j.off[0], (int)j.off[1],
                                                              (int)j.off[2], (int)j.shape[0], (int)j.shape[1], (int)j.shape[2], ya, yb, d.prec);
      AF_CK(cudaGetLastError());
    }
  };
  h.end = [pp](int64_t t) { for (auto& j : pp->inj) if (j && j->due(t)) j->last = t; };
  return h;
}

}  // namespace afdtd
