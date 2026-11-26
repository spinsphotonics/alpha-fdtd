// SPDX-License-Identifier: MIT
// plan.cpp -- devices, memory mode, schedule choice (ESTIMATE by rules and a cost model, MEASURE / PATIENT by timing on the plan's own
// GPUs), the per-process self-check of every (kernel object, device), execute.
#include "alpha_fdtd.h"
#include "internal.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <random>
#include <set>

namespace afdtd {

void fail(int code, const char* fmt, ...)
{
  char b[600]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
  throw Error(code, b);
}
void cuda_fail(int err, const char* what, int line)
{
  const bool oom = err == (int)cudaErrorMemoryAllocation || err == -(int)CUDA_ERROR_OUT_OF_MEMORY;
  const char* s = err >= 0 ? cudaGetErrorString((cudaError_t)err) : "driver API error";
  fail(oom ? AFDTD_ENOMEM : AFDTD_EDEVICE, "CUDA: %s (%d) in %s (line %d)", s, err, what, line);
}

std::recursive_mutex& device_lock(int dev)
{
  static std::recursive_mutex m[256];
  return m[(unsigned)dev % 256];
}
const Driver& drv()
{
  static Driver d = [] {
    Driver x; memset(&x, 0, sizeof x);
    auto get = [](const char* name, void** fn) {
      cudaDriverEntryPointQueryResult q;
      if (cudaGetDriverEntryPointByVersion(name, fn, 12000, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess) *fn = nullptr;
    };
    get("cuMemCreate", (void**)&x.MemCreate); get("cuMemRelease", (void**)&x.MemRelease); get("cuMemMap", (void**)&x.MemMap);
    get("cuMemUnmap", (void**)&x.MemUnmap); get("cuMemSetAccess", (void**)&x.MemSetAccess);
    get("cuMemAddressReserve", (void**)&x.MemAddressReserve); get("cuMemAddressFree", (void**)&x.MemAddressFree);
    get("cuTensorMapEncodeTiled", (void**)&x.TensorMapEncodeTiled);
    get("cuStreamWaitValue32", (void**)&x.StreamWaitValue32);
    return x;
  }();
  if (!d.MemCreate || !d.MemMap || !d.MemAddressReserve || !d.TensorMapEncodeTiled) fail(AFDTD_EUNSUPPORTED, "the NVIDIA driver lacks the virtual memory or tensor map API");
  return d;
}

// ------------------------------------------------------------------------------------------------ host-memory ring rule
// Row g of the run reuses the ring position of row g - NR, which must be released 1 + slack launches before row g is needed.
long ring_bad(int Q, int NR, int NI, int NJ, int D, int passes, double lam, int grid)
{
  const int NJV = (int)rup((size_t)NJ + 1, (size_t)NR), rounds = (passes + Q - 1) / Q, Qlast = passes - (rounds - 1) * Q, slack = 2;
  const double DL = (double)Q * NI / ((double)grid * D);
  auto key = [&](int p, long g) { return (double)(p + g) * DL + p * lam; };
  auto LK = [&](double t) { return (long)std::floor(t / DL); };
  auto tgt = [&](long g) { return g % NJV < NJ ? (g / NJV == rounds - 1 ? Qlast : Q) : 0; };
  auto need = [&](long g) -> long { const int v = (int)(g % NJV); return v > NJ ? -1 : v == 0 ? LK(key(0, g)) : LK(key(0, g - 1)); };
  auto rel = [&](long g) -> long { const int v = (int)(g % NJV); return v > NJ ? -1 : v == NJ ? LK(key(tgt(g - 1) - 1, g - 1)) : LK(key(tgt(g) - 1, g)); };
  long bad = 0;
  for (long g = NR; g < (long)rounds * NJV; ++g) { const long nd = need(g), rl = rel(g - NR); if (nd >= 0 && rl >= 0 && nd - rl < 1 + slack) ++bad; }
  return bad;
}
static int max_Q(int NR, int NI, int NJ, int D, double lam, int grid)
{
  int lo = 0, hi = NR;
  while (lo < hi) { const int m = (lo + hi + 1) / 2; if (!ring_bad(m, NR, NI, NJ, D, std::max(4 * NR, m), lam, grid)) lo = m; else hi = m - 1; }
  return lo;
}

// ------------------------------------------------------------------------------------------------ memory
// GPU memory kept free per GPU beyond the state: CUDA's lazily loaded modules and launch resources (256 MiB) and the box staging
// buffer of afdtd_set / afdtd_get (64 MiB, or one x plane of 12 components when that is larger).
static size_t reserve_bytes(int64_t ny, int64_t nz) { return ((size_t)256 << 20) + std::max((size_t)64 << 20, (size_t)(24 * ny * nz)); }
// Pool size (blocks in flight): the measured fast floor (1100 on 1 GPU, 384 on several), or with small = true only the legal minimum
// (domains whose pools would not fit otherwise: long z, where every pool slab holds a whole z column).
static int mx_rule(int NI, int rows, int grid, int tx, int ngpus, int NJ, bool small = false)
{
  return std::max({small ? 0 : ngpus == 1 ? 1100 : 384, 2 * (std::min(NI, rows) + 2) + grid + 16, rows * tx + 64, 2 * (std::min(NI, NJ) + 2) + grid + 1});
}
static size_t mem_bytes(const Geo& G, int ngpus, int mx)       // busiest GPU, domain in GPU memory (the executor's allocations)
{
  if (ngpus == 1) {
    const size_t nb = (size_t)G.NI * G.NJ * kPMAX;
    return (size_t)(G.NJ + 1) * G.ROWREC * 24 + (size_t)G.NJ * G.NZP * G.NPAIR * 32 * kCREC + (size_t)(mx + 1) * (G.XBUF + G.YBUF) + nb * (64 + 128 + 4) + 1024;
  }
  const size_t rsb = (size_t)G.ROWREC * 24, crsb = rsb / 3 * 4, gx = (kGR + G.XBUF - 1) / G.XBUF, gy = (kGR + G.YBUF - 1) / G.YBUF;
  const size_t rows = (size_t)(G.NJ + ngpus - 1) / ngpus + 1, nb = (size_t)G.NI * rows * kPMAX;
  return rows * (rsb + crsb) + (mx + 1 + 2 * gx) * G.XBUF + (mx + 1 + 2 * gy) * G.YBUF + nb * (64 + 128) + rup(nb + 64, (size_t)1 << 19) * 4;
}
static size_t ring_bytes(const Geo& G, int S, int spg)          // per GPU, domain in host memory (one execute's allocations)
{
  const size_t RSB = (size_t)G.ROWREC * 24, CROW = rup((size_t)G.NZP * G.PS, kGR);
  const size_t gx = (kGR + G.XBUF - 1) / G.XBUF, gy = (kGR + G.YBUF - 1) / G.YBUF;
  return (size_t)spg * S * (RSB + CROW) + (400 + 1 + 2 * gx) * G.XBUF + (size_t)spg * (S * G.NI + 2 * gy) * G.YBUF + ((size_t)3 << 29);
}
static void host_layout(HostDomain& h)
{
  const Geo& G = h.G;
  h.RSB = (size_t)G.ROWREC * 24; h.CSB = (size_t)G.NZP * G.PS; h.CROW = rup(h.CSB, kGR); h.HRS = h.RSB + rup(h.CSB, 4096);
  h.bytes = (size_t)G.NJ * h.HRS;
}
void bytes_needed(int64_t nx, int64_t ny, int64_t nz, int ngpus, size_t* gpu, size_t* host, unsigned flags)
{
  if (flags & AFDTD_FP32) { if (gpu) *gpu = f32_bytes(nx, ny, nz, ngpus) + reserve_bytes(ny, nz); if (host) *host = 0; return; }
  const Geo G = geometry(nx, ny, nz, ngpus > 1), Gh = geometry(nx, ny, nz, true);
  const int rows = (G.NJ + ngpus - 1) / ngpus;
  if (gpu) *gpu = std::min(mem_bytes(G, ngpus, mx_rule(G.NI, rows, 148, 12, ngpus, G.NJ)), mem_bytes(G, ngpus, mx_rule(G.NI, rows, 148, 12, ngpus, G.NJ, true)))
                 + reserve_bytes(ny, nz);
  HostDomain h; h.G = Gh; host_layout(h);
  if (host) *host = h.bytes + h.RSB;
}

// ------------------------------------------------------------------------------------------------ cost model (prediction only)
// Per GPU: min(SM side, memory floor).  SM side = SMs x clock / (cycles per owned cell update per SM), measured per kernel family on
// the committed kernels: sm_80 1.06, sm_90 0.734 (H200, SM-bound), sm_100 / sm_103 0.64 (B200, B300).  Memory floor = HBM bandwidth /
// 10.15 B per update (H100, memory-bound).  Several GPUs x 0.965; host memory x 0.87 (H100, execute calls of 7 qmax steps; shorter calls are bound by the host link).
static double predict(int dev, int cc, int ngpus, bool on_host)
{
  int clk = 0, mclk = 0, bus = 0, nsm = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev); cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaGetLastError();
  const double cyc = cc < 90 ? 1.06 : cc < 100 ? 0.734 : 0.64;
  double r = nsm * clk * 1e3 / cyc;
  if (mclk > 0 && bus > 0) r = std::min(r, 2.0 * mclk * 1e3 * bus / 8 / 10.15);
  return r * ngpus * (ngpus > 1 ? 0.965 : 1.0) * (on_host ? 0.87 : 1.0);
}

// ------------------------------------------------------------------------------------------------ objects and candidates
static const Object* object_for(int cc, const char* name, int prec)
{
  for (const Object* o : objects()) if (o->cc == cc && o->prec == prec && (!name || !strcmp(o->name, name))) return o;
  return nullptr;
}
struct Cand { const Object* obj; Knobs k; };
// Selection rules, from timing these kernels on each GPU: H100 / H200 strip width 6 (gate 1 on 1 GPU, 0 on several); B200 / B300 grid = SMs,
// the smallest strip width dividing the grid with tx x rows >= 1.5 grid, gate 0; A100 the cp.async text without L2 hints, tx 12, gate 0.
static std::vector<Cand> candidates(const Plan& p, const Geo& G, int rigor)
{
  const int ng = (int)p.devs.size(), rows = (G.NJ + ng - 1) / ng, grid = p.nsm;
  std::vector<Cand> out;
  auto add = [&](const char* obj, int g, int tx, int gate) {
    const Object* o = object_for(p.cc, obj, p.prec);
    if (!o) return;
    Knobs k; k.grid = g; k.tx = tx; k.gate = gate; k.mx = mx_rule(G.NI, rows, g, tx, ng, G.NJ, p.small_pools);
    for (auto& c : out) if (c.obj == o && !memcmp(&c.k, &k, sizeof k)) return;
    out.push_back({o, k});
  };
  std::vector<int> divs; for (int d = 2; d <= grid / 2; ++d) if (grid % d == 0) divs.push_back(d);
  if (p.cc >= 100) {
    int tx = grid; for (int d = 1; d <= grid; ++d) if (grid % d == 0 && d * rows >= 1.5 * grid) { tx = d; break; }
    add(nullptr, grid, tx, 0);
    if (rigor >= AFDTD_MEASURE) { for (int t : {4, 37, 74}) if (grid % t == 0) add(nullptr, grid, t, 0); add(nullptr, 144, 12, 0); }
    if (rigor >= AFDTD_PATIENT) for (int t : divs) for (int g : {0, 1}) if (g == 0 || ng == 1) add(nullptr, grid, t, g);
  } else if (p.cc >= 90) {
    add(nullptr, grid, 6, ng == 1 ? 1 : 0);
    if (rigor >= AFDTD_MEASURE) for (int t : {12, 6, 22, 33}) for (int g : {1, 0}) if (g == 0 || ng == 1) add(nullptr, grid, t, g);
    if (rigor >= AFDTD_PATIENT) for (int t : divs) for (int g : {1, 0}) if (g == 0 || ng == 1) add(nullptr, grid, t, g);
  } else {
    add("zs7_cpasync_l2h0", grid, 12, 0);
    if (rigor >= AFDTD_MEASURE)
      for (const char* o : {"zs7_cpasync_l2h0", "zs7_cpasync_l2h3"}) for (int g : {0, 1}) for (int t : {12, 6, 18}) if (g == 0 || ng == 1) add(o, grid, t, g);
    if (rigor >= AFDTD_PATIENT) for (const char* o : {"zs7_cpasync_l2h0", "zs7_cpasync_l2h3"}) for (int t : divs) add(o, grid, t, 0);
  }
  return out;
}
static std::string sched_text(const Plan& p)
{
  char b[200];
  if (p.on_host) snprintf(b, sizeof b, "obj=%s host S=%d slots=%d lam=%g qmax=%d", p.obj->name, p.host.ring.S, p.host.ring.nslot, p.host.ring.lam, p.host.ring.qmax);
  else snprintf(b, sizeof b, "obj=%s grid=%d tx=%d gate=%d mx=%d", p.obj->name, p.knobs.grid, p.knobs.tx, p.knobs.gate, p.knobs.mx);
  return b;
}
// A wisdom schedule, checked like a planned one: syntax, ranges, a legal pool size and memory that fits (GPU memory), a ring no
// larger than the one planned and a legal depth (host memory).  false: the record is not used.
static bool parse_sched(Plan& p, const std::string& s, const Geo& G, size_t budget)
{
  char obj[64] = {0}; Knobs k; int S = 0, slots = 0, qmax = 0; double lam = 0;
  if (p.on_host) { if (sscanf(s.c_str(), "obj=%63s host S=%d slots=%d lam=%lf qmax=%d", obj, &S, &slots, &lam, &qmax) != 5) return false; }
  else if (sscanf(s.c_str(), "obj=%63s grid=%d tx=%d gate=%d mx=%d", obj, &k.grid, &k.tx, &k.gate, &k.mx) != 5) return false;
  const Object* o = object_for(p.cc, obj, p.prec);
  if (!o) return false;
  p.obj = o;
  if (p.on_host) {
    if (S != 1 || slots < (int)p.devs.size() || slots % (int)p.devs.size() || slots > p.host.ring.nslot || qmax < 1 || lam <= 0) return false;
    if (qmax > max_Q(slots, p.host.G.NI, p.host.G.NJ, (int)p.devs.size(), lam, p.nsm)) return false;
    p.host.ring.S = S; p.host.ring.nslot = slots; p.host.ring.lam = lam; p.host.ring.qmax = qmax;
  } else {
    const int ng = (int)p.devs.size(), rows = (G.NJ + ng - 1) / ng;
    if (k.grid < 1 || k.grid > p.nsm || k.tx < 1 || k.gate < 0 || k.gate > 1) return false;
    if (k.mx < mx_rule(G.NI, rows, k.grid, k.tx, ng, G.NJ, true) || mem_bytes(G, ng, k.mx) > budget) return false;
    p.knobs = k;
  }
  return true;
}
static std::string wisdom_key(const Plan& p)
{
  char b[400];
  snprintf(b, sizeof b, "kset=%s gpu=\"%s\" cc=%d.%d sm=%d ngpus=%d size=%lldx%lldx%lld mode=%s%s", AFDTD_KSET, p.gpu_name.c_str(), p.cc / 10, p.cc % 10,
           p.nsm, (int)p.devs.size(), (long long)p.n[0], (long long)p.n[1], (long long)p.n[2], p.on_host ? "host" : "gpu", p.prec == AFDTD_F16 ? " prec=fp16" : "");
  return b;
}

// ------------------------------------------------------------------------------------------------ self-check
// Before an object is first used on a device in this process: a small random lossy problem, 2 trips + 3 steps through the executor
// against 17 single reference steps (the tail kernels), every value compared.  A difference means a broken install, driver or GPU.
static void self_check(const Object* o, int dev, const Knobs& kp)
{
  static std::mutex m; static std::set<std::pair<const Object*, int>> done;
  std::lock_guard<std::mutex> l(m);
  if (done.count({o, dev})) return;
  const int64_t n[3] = {70, 101, 61}, off[3] = {0, 0, 0}, N = n[0] * n[1] * n[2];
  const Geo G = geometry(n[0], n[1], n[2], false);
  Knobs k = kp; k.mx = mx_rule(G.NI, G.NJ, k.grid, k.tx, 1, G.NJ);
  std::vector<uint16_t> in((size_t)12 * N), a((size_t)12 * N), b((size_t)12 * N);
  std::mt19937_64 rng(12345);
  auto bf = [o](float f) { uint32_t u; memcpy(&u, &f, 4); return o->prec == AFDTD_F16 ? f32_to_f16(u) : (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); };
  for (int c = 0; c < 12; ++c)
    for (int64_t i = 0; i < N; ++i) {
      const int64_t x = i / (n[1] * n[2]), y = i / n[2] % n[1], z = i % n[2], pos[3] = {x, y, z};
      const double u = (double)(rng() >> 11) / 9007199254740992.0;
      const bool wall = (c < 3 || c >= 6) && ((pos[0] == 0 && c % 3 != 0) || (pos[1] == 0 && c % 3 != 1) || (pos[2] == 0 && c % 3 != 2));
      in[(size_t)c * N + i] = wall ? 0 : bf(c < 6 ? (float)(2 * u - 1) : c < 9 ? (float)(0.5 + 0.5 * u) : (float)(0.3 * u));
    }
  auto run = [&](std::vector<uint16_t>& out, bool engine) {
    void* st = o->create(G, &dev, 1, k);
    try {
      uint16_t* cp[12]; for (int c = 0; c < 12; ++c) cp[c] = in.data() + (size_t)c * N;
      o->box(st, off, n, cp, 0);
      if (engine) { o->advance(st, 2, 0, nullptr); o->tail(st, 3); } else o->tail(st, 17);
      for (int c = 0; c < 12; ++c) cp[c] = out.data() + (size_t)c * N;
      o->box(st, off, n, cp, 1);
    } catch (...) { o->destroy(st); throw; }
    o->destroy(st);
  };
  run(a, true); run(b, false);
  size_t bad = 0; for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
  for (size_t i = (size_t)6 * N; i < in.size(); ++i) bad += a[i] != in[i];
  if (bad) fail(AFDTD_EINTERNAL, "self-check failed: kernel %s (%s) on device %d differs from the reference steps in %zu values", o->name, o->arch, dev, bad);
  done.insert({o, dev});
}

// ------------------------------------------------------------------------------------------------ plan
static void check_devices(Plan& p, int ngpus, const int* devices)
{
  int have = 0;
  if (cudaGetDeviceCount(&have) != cudaSuccess || have == 0) { cudaGetLastError(); fail(AFDTD_EUNSUPPORTED, "no CUDA device (or no NVIDIA driver)"); }
  for (int i = 0; i < ngpus; ++i) {
    const int d = devices ? devices[i] : i;
    if (d < 0 || d >= have) fail(AFDTD_EINVAL, "device %d does not exist (%d visible)", d, have);
    for (int j = 0; j < i; ++j) if (p.devs[j] == d) fail(AFDTD_EINVAL, "device %d listed twice", d);
    p.devs.push_back(d);
  }
  for (int d : p.devs) {
    int mj = 0, mn = 0, nsm = 0; cudaDeviceProp pr;
    AF_CK(cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, d)); AF_CK(cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, d));
    AF_CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d)); AF_CK(cudaGetDeviceProperties(&pr, d));
    const int cc = 10 * mj + mn;
    if (!object_for(cc, nullptr, AFDTD_BF16))
      fail(AFDTD_EUNSUPPORTED, "device %d (%s, compute capability %d.%d) is not supported (8.0, 9.0, 10.0, 10.3) or not built into this library", d, pr.name, mj, mn);
    if (d == p.devs[0]) { p.cc = cc; p.nsm = nsm; p.gpu_name = pr.name; }
    else if (cc != p.cc || nsm != p.nsm || p.gpu_name != pr.name) fail(AFDTD_EUNSUPPORTED, "devices %d and %d are different GPU models", p.devs[0], d);
  }
  for (int d : p.devs) for (int e : p.devs) if (d != e) {
    int ok = 0; AF_CK(cudaDeviceCanAccessPeer(&ok, d, e));
    if (!ok) fail(AFDTD_EUNSUPPORTED, "no peer access from device %d to device %d", d, e);
    AF_CK(cudaSetDevice(d)); const cudaError_t r = cudaDeviceEnablePeerAccess(e, 0);
    if (r != cudaSuccess && r != cudaErrorPeerAccessAlreadyEnabled) AF_CK(r);
    cudaGetLastError();
  }
}
static size_t min_free(const Plan& p)
{
  size_t m = SIZE_MAX;
  for (int d : p.devs) { size_t f = 0, t = 0; AF_CK(cudaSetDevice(d)); AF_CK(cudaMemGetInfo(&f, &t)); m = std::min(m, f); }
  return m;
}
static double gib(size_t b) { return b / 1073741824.0; }

Plan* plan_create(int64_t nx, int64_t ny, int64_t nz, int ngpus, const int* devices, size_t host_budget, int rigor, unsigned flags)
{
  const int64_t n[3] = {nx, ny, nz};
  for (int a = 0; a < 3; ++a) if (n[a] < 16 || n[a] > (1 << 20)) fail(AFDTD_EINVAL, "size %lld x %lld x %lld: each axis needs 16 to 1048576 cells", (long long)nx, (long long)ny, (long long)nz);
  if ((double)nx * ny * nz > 4e15) fail(AFDTD_EINVAL, "size %lld x %lld x %lld: too many cells", (long long)nx, (long long)ny, (long long)nz);
  if (rigor < AFDTD_ESTIMATE || rigor > AFDTD_WISDOM_ONLY) fail(AFDTD_EINVAL, "rigor %d is not AFDTD_ESTIMATE, _MEASURE, _PATIENT or _WISDOM_ONLY", rigor);
  const unsigned known = AFDTD_FORCE_HOST | AFDTD_STRICT_WALLS | AFDTD_FP16 | AFDTD_FP32;
  if (flags & ~known) fail(AFDTD_EINVAL, "unknown flag bits 0x%x (a rigor value passed as a flag?)", flags & ~known);
  if ((flags & AFDTD_FP16) && (flags & AFDTD_FP32)) fail(AFDTD_EINVAL, "AFDTD_FP16 and AFDTD_FP32 together: choose one precision");
  if (ngpus < 0 || ngpus > 8) fail(AFDTD_EINVAL, "ngpus = %d: 1 to 8 (0 means 1)", ngpus);
  if (ngpus == 0) { ngpus = 1; if (devices) fail(AFDTD_EINVAL, "devices given with ngpus = 0"); }
  Plan* pp = new Plan;
  try {
    Plan& p = *pp;
    p.n[0] = nx; p.n[1] = ny; p.n[2] = nz; p.flags = flags;
    p.prec = (flags & AFDTD_FP32) ? AFDTD_F32 : (flags & AFDTD_FP16) ? AFDTD_F16 : AFDTD_BF16;
    check_devices(p, ngpus, devices);
    if (p.prec == AFDTD_F32) {                         // FP32: dense arrays in GPU memory, one kernel per half step; nothing to plan
      if (flags & AFDTD_FORCE_HOST) fail(AFDTD_EINVAL, "AFDTD_FP32 plans keep the domain in GPU memory (AFDTD_FORCE_HOST is for BF16 and FP16)");
      if (nx < 2 * ngpus) fail(AFDTD_EINVAL, "nx = %lld on %d GPUs: FP32 plans cut the domain along x, at least 2 planes per GPU", (long long)nx, ngpus);
      const size_t freeb = min_free(p), need = f32_bytes(nx, ny, nz, ngpus) + reserve_bytes(ny, nz);
      if (need > freeb)
        fail(AFDTD_ENOMEM, "%lld x %lld x %lld in FP32 needs %.2f GiB per GPU (48 bytes per cell), %.2f GiB free on the fullest of %d GPU(s); FP32 plans "
             "need the domain in GPU memory (BF16 and FP16 plans need about a third as much, and can use host memory)", (long long)nx, (long long)ny,
             (long long)nz, gib(need), gib(freeb), ngpus);
      p.obj = &kF32Object; p.on_host = 0; p.rigor = rigor == AFDTD_WISDOM_ONLY ? AFDTD_ESTIMATE : rigor; p.gpu_bytes = f32_bytes(nx, ny, nz, ngpus);
      int mclk = 0, bus = 0; cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, p.devs[0]); cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, p.devs[0]);
      cudaGetLastError();
      p.predicted = mclk > 0 && bus > 0 ? 2.0 * mclk * 1e3 * bus / 8 / 96 * ngpus : 0;   // memory floor: 96 B per update (two half-step kernels)
      p.mem = p.obj->create(geometry(nx, ny, nz, false), p.devs.data(), ngpus, p.knobs);
      p.sched = "obj=f32";
      return pp;
    }
    // ---- memory mode
    const Geo G = geometry(nx, ny, nz, ngpus > 1);
    const size_t freeb = min_free(p), resv = reserve_bytes(ny, nz), budget = freeb > resv ? freeb - resv : 0;
    std::vector<Cand> est = candidates(p, G, AFDTD_ESTIMATE);
    p.gpu_bytes = mem_bytes(G, ngpus, est[0].k.mx);
    if (p.gpu_bytes > budget) {                       // the pools at their legal minimum (long z)
      p.small_pools = 1; est = candidates(p, G, AFDTD_ESTIMATE);
      if (mem_bytes(G, ngpus, est[0].k.mx) <= budget) p.gpu_bytes = mem_bytes(G, ngpus, est[0].k.mx);
      else { p.small_pools = 0; est = candidates(p, G, AFDTD_ESTIMATE); }
    }
    HostDomain& h = p.host; h.G = geometry(nx, ny, nz, true); h.devs = p.devs; host_layout(h);
    const size_t host_need = h.bytes + h.RSB;
    const bool y_ok = ngpus == 1 || ny > (int64_t)kS * (2 * ngpus - 1);
    if (!(flags & AFDTD_FORCE_HOST) && p.gpu_bytes <= budget && y_ok) p.on_host = 0;
    else {
      int spg = 0;
      while (ring_bytes(h.G, 1, spg + 1) <= freeb && (spg + 1) * ngpus <= h.G.NJ) ++spg;
      if (host_need > host_budget || spg < 1) {
        if (!y_ok && !(flags & AFDTD_FORCE_HOST) && host_budget == 0)
          fail(AFDTD_EINVAL, "ny = %lld on %d GPUs in GPU memory: each GPU needs >= 2 rows of 24 y-cells (ny > %d)", (long long)ny, ngpus, kS * (2 * ngpus - 1));
        fail(AFDTD_ENOMEM, "%lld x %lld x %lld does not fit: GPU memory needs %.2f GiB per GPU (%.2f GiB free on the fullest of %d GPU(s)); host memory needs "
             "host_bytes >= %zu (%.2f GiB, given %.2f GiB) and one ring slot of %.2f GiB per GPU", (long long)nx, (long long)ny, (long long)nz, gib(p.gpu_bytes),
             gib(budget), ngpus, host_need, gib(host_need), gib(host_budget), gib(ring_bytes(h.G, 1, 1)));
      }
      p.on_host = 1;
      const double lam = ngpus == 1 ? 3.0 : 16.0;
      for (; spg >= 1; --spg) { const int q = max_Q(spg * ngpus, h.G.NI, h.G.NJ, ngpus, lam, p.nsm); if (q >= 1) { h.ring.qmax = q; break; } }
      if (spg < 1) fail(AFDTD_ENOMEM, "no legal host-memory ring for %lld x %lld x %lld on %d GPU(s)", (long long)nx, (long long)ny, (long long)nz, ngpus);
      h.ring.S = 1; h.ring.nslot = spg * ngpus; h.ring.lam = lam;
      p.gpu_bytes = ring_bytes(h.G, 1, spg); p.host_bytes = host_need;
    }
    // ---- schedule: wisdom, else rules (ESTIMATE) or timing (MEASURE, PATIENT)
    const std::string key = wisdom_key(p);
    std::string ws; int wr = 0; double wrate = 0;
    if (wisdom_find(key, rigor == AFDTD_WISDOM_ONLY ? 0 : rigor, ws, wr, wrate) && parse_sched(p, ws, G, budget)) { p.from_wisdom = 1; p.rigor = wr; p.measured = wrate; }
    else if (rigor == AFDTD_WISDOM_ONLY) fail(AFDTD_ENOPLAN, "AFDTD_WISDOM_ONLY: no usable wisdom record for %s", key.c_str());
    else if (p.on_host) { p.obj = object_for(p.cc, p.cc < 90 ? "zs7_cpasync_l2h0" : nullptr, p.prec); p.rigor = rigor; }
    else { p.obj = est[0].obj; p.knobs = est[0].k; p.rigor = rigor; }
    p.predicted = predict(p.devs[0], p.cc, ngpus, p.on_host);
    for (int d : p.devs) self_check(p.obj, d, p.on_host ? est[0].k : p.knobs);
    if (p.on_host) { host_alloc(h); p.host_bytes = h.bytes; }
    else {
      p.mem = p.obj->create(G, p.devs.data(), ngpus, p.knobs);
      if (!p.from_wisdom && rigor >= AFDTD_MEASURE) {     // time the candidates on this state, interleaved A B ... B A
        std::vector<Cand> cs = candidates(p, G, rigor);
        std::vector<double> sec(cs.size(), 0); std::vector<int> ok(cs.size(), 0);
        std::vector<std::pair<const Object*, void*>> st = {{p.obj, p.mem}};      // one state per kernel object (A100: two texts)
        const double cells = (double)nx * ny * nz;
        const int P = (int)std::min(256.0, std::max(4.0, std::ceil(1.0 * p.predicted / (cells * kK))));     // about 1 s per timing
        for (size_t i = 0; i < cs.size(); ++i) ok[i] = mem_bytes(G, ngpus, cs[i].k.mx) * (cs[i].obj == p.obj ? 1 : 2) <= budget;
        auto timed = [&](size_t i) -> double {
          void* m = nullptr;
          for (auto& x : st) if (x.first == cs[i].obj) m = x.second;
          if (!m) { m = cs[i].obj->create(G, p.devs.data(), ngpus, cs[i].k); st.push_back({cs[i].obj, m}); }
          cs[i].obj->set_knobs(m, cs[i].k);
          return cs[i].obj->advance(m, P, 0, nullptr);
        };
        const int rounds = rigor == AFDTD_PATIENT ? 3 : 2;
        try {
          for (size_t i = 0; i < cs.size(); ++i) if (ok[i]) for (int d : p.devs) self_check(cs[i].obj, d, cs[i].k);
          timed(0);                                         // warm-up
          for (int r = 0; r < rounds; ++r)
            for (size_t j = 0; j < cs.size(); ++j) { const size_t i = r % 2 ? cs.size() - 1 - j : j; if (ok[i]) sec[i] += timed(i); }
        } catch (...) { for (size_t i = 1; i < st.size(); ++i) st[i].first->destroy(st[i].second); throw; }
        size_t best = 0; for (size_t i = 0; i < cs.size(); ++i) if (ok[i] && sec[i] < sec[best]) best = i;
        for (size_t i = 1; i < st.size(); ++i) {
          if (st[i].first == cs[best].obj) { p.obj->destroy(p.mem); p.obj = st[i].first; p.mem = st[i].second; }
          else st[i].first->destroy(st[i].second);
        }
        p.knobs = cs[best].k; p.obj->set_knobs(p.mem, p.knobs); p.obj->zero(p.mem);
        p.measured = cells * kK * P * rounds / sec[best];
      }
    }
    p.sched = sched_text(p);
    if (!p.from_wisdom && rigor >= AFDTD_MEASURE) wisdom_add(key, p.sched, rigor, p.measured);
    return pp;
  } catch (...) { plan_destroy(pp); throw; }
}

void plan_destroy(Plan* p)
{
  if (!p) return;
  record_close_all(*p);
  inject_close_all(*p);
  try { if (p->mem) p->obj->destroy(p->mem); } catch (...) {}
  host_free(p->host);
  delete p;
}

void plan_execute(Plan& p, int64_t steps)
{
  if (steps < 0) fail(AFDTD_EINVAL, "steps = %lld < 0", (long long)steps);
  if (!steps) return;
  const int64_t passes = steps / kK; const int r = (int)(steps % kK);
  auto wall = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  const double w0 = wall();
  if (!p.on_host && (record_active(p) || inject_active(p))) {
    // with recorders or injectors: their work is at multiples of 7, so the passes start at a multiple of 7: a head of tail steps
    // first (the result is the same for any split of the steps), the work due at the start, the passes with the work inside the
    // launches, the remaining tail steps.  Injections first at a step, then frames.  While an injection is due in the call, the
    // persistent grid leaves one SM free for the injection kernels (they run beside it).
    const AgentHooks rh = record_hooks(p), jh = inject_hooks(p);
    AgentHooks h;
    h.want = [rh, jh](int64_t t) { return jh.want(t) || rh.want(t); };
    h.regions = [rh, jh](int64_t t, std::vector<Region>& o) { jh.regions(t, o); rh.regions(t, o); };
    h.begin = [rh](int64_t t) { rh.begin(t); };
    h.work = [rh, jh](int64_t t, const AgentDev& d, int q0, int q1) { if (q0 < 0) jh.work(t, d, q0, q1); else rh.work(t, d, q0, q1); };
    h.end = [rh, jh](int64_t t) { jh.end(t); rh.end(t); };
    const Knobs k0 = p.knobs;
    const bool spare = inject_due_between(p, p.steps_done + 1, p.steps_done + steps);
    if (spare) { Knobs k = k0; k.grid = std::max(1, (k0.grid ? k0.grid : p.nsm) - 1); p.obj->set_knobs(p.mem, k); }
    struct Restore { Plan& p; Knobs k; bool on; ~Restore() { if (on) try { p.obj->set_knobs(p.mem, k); } catch (...) {} } } restore{p, k0, spare};
    int64_t t = p.steps_done, left = steps;
    auto now_if = [&]() { if (h.want(t)) p.obj->agents_now(p.mem, t, h); };
    now_if();
    const int64_t head = std::min<int64_t>(left, (kK - t % kK) % kK);
    if (head) { p.obj->tail(p.mem, (int)head); t += head; left -= head; now_if(); }
    if (left / kK) { p.obj->advance(p.mem, left / kK, t, &h); t += left / kK * kK; }
    if (left % kK) p.obj->tail(p.mem, (int)(left % kK));
    p.tail_steps += head + left % kK - r;            // r is added below
  } else if (!p.on_host) {
    if (passes) p.obj->advance(p.mem, passes, p.steps_done, nullptr);
    if (r) p.obj->tail(p.mem, r);
  } else {
    HostDomain& h = p.host;
    // each run: whole rounds of balanced depth (every round moves the domain over the host link once); tables at most ~1 GiB per run
    const int64_t per_run = std::max<int64_t>(h.ring.qmax, (int64_t)16000000 / ((int64_t)h.G.NI * h.G.NJ) / h.ring.qmax * h.ring.qmax);
    for (int64_t done = 0; done < passes;) {
      const int pr = (int)std::min(per_run, passes - done);
      const int nr = (pr + h.ring.qmax - 1) / h.ring.qmax;
      int Q = (pr + nr - 1) / nr;
      while (Q > 1 && ring_bad(Q, h.ring.nslot * h.ring.S, h.G.NI, h.G.NJ, (int)h.devs.size(), pr, h.ring.lam, p.nsm)) --Q;
      p.obj->host_run(h, pr, Q);
      done += pr;
    }
    if (r) p.obj->host_tail(h, r);
  }
  const double w = wall() - w0;
  if (p.prec == AFDTD_F32) p.tail_steps = 0;          // one kernel per half step for every step: no tail path
  p.steps_done += steps; if (p.prec != AFDTD_F32) p.tail_steps += r;
  p.last_rate = w > 0 ? (double)p.n[0] * p.n[1] * p.n[2] * steps / w : 0;
  const std::string e = record_take_error(p);
  if (!e.empty()) fail(AFDTD_EIO, "the steps ran, but a recorder failed (frames from then on are missing): %s", e.c_str());
}

void plan_checksums(Plan& p, uint64_t out[6])
{
  if (p.on_host) host_checksums(p.host, out); else p.obj->checksums(p.mem, out);
}

std::string plan_describe(const Plan& p)
{
  static const char* rn[] = {"estimate", "measure", "patient", "wisdom-only"};
  char b[1024];
  char hint[160] = "";
  if (p.on_host) snprintf(hint, sizeof hint, "; host memory: predicted rate for execute calls of >= %d steps (each call streams the domain over "
                          "the host link once per %d steps)", kK * p.host.ring.qmax, kK * p.host.ring.qmax);
  snprintf(b, sizeof b, "%lld x %lld x %lld %s in %s memory on %d x %s; rigor %s%s; predicted %.3g, measured %.3g, last %.3g cell updates/s; "
           "%lld steps, %lld on the tail path; %.2f GiB per GPU, %.2f GiB host%s | schedule (opaque, as in wisdom): %s",
           (long long)p.n[0], (long long)p.n[1], (long long)p.n[2], p.prec == AFDTD_F32 ? "FP32" : p.prec == AFDTD_F16 ? "FP16" : "BF16",
           p.on_host ? "host" : "GPU", (int)p.devs.size(), p.gpu_name.c_str(), rn[p.rigor],
           p.from_wisdom ? " (from wisdom)" : "", p.predicted, p.measured, p.last_rate, (long long)p.steps_done, (long long)p.tail_steps,
           gib(p.gpu_bytes), gib(p.host_bytes), hint, p.sched.c_str());
  return b;
}

}  // namespace afdtd
