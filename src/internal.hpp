// SPDX-License-Identifier: MIT
// internal.hpp -- what the host code of libalphafdtd and the kernel objects share.  Not installed.
//
// State layout (identical for every committed kernel text; object.cu checks it against the kernel's macros):
//   the domain is cut along y into rows of kS = 24 cells.  Fields: one record of 6 words (Ex Ey Ez Hx Hy Hz) per (z plane, x pair,
//   y lane); word c holds x = 2 gp' (low half) and 2 gp' + 1 (high half), gp' = gp - kGOFF.  Record (zi, gp, l) of row J at
//   J ROWREC + zi PLREC + gp kS + l, zi = z + kSOFF.  Coefficients (Cax Cay Caz Cbx Cby Cbz, x pairs per word): row J at J crow,
//   plane zi at zi PS, then (gp 30 + l) kCREC bytes; lanes 24..29 of row J repeat lanes 0..5 of row J + 1.  Every value outside the
//   owned domain is +0.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <cuda.h>

namespace afdtd {

enum { kS = 24, kK = 7, kSOFF = 20, kGOFF = 8, kCREC = 24, kSSH = 3, kSW0 = -2, kNCOPY = 2, kXGS = 2176, kYGS = 512, kPMAX = 256 };
constexpr size_t kGR = size_t(2) << 20;                       // VMM granule
inline size_t rup(size_t x, size_t m) { return (x + m - 1) / m * m; }

// An error crossing from the internals to the API: an AFDTD_E* code and a one-line message.
struct Error : std::runtime_error { int code; Error(int c, const std::string& m) : std::runtime_error(m), code(c) {} };
[[noreturn]] void fail(int code, const char* fmt, ...);
[[noreturn]] void cuda_fail(int err, const char* what, int line);
#define AF_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) ::afdtd::cuda_fail((int)e_, #x, __LINE__); } while (0)
#define AF_CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) ::afdtd::cuda_fail(-(int)r_, #x, __LINE__); } while (0)

// The driver API entry points used (virtual memory, tensor maps), resolved at run time so that the library loads without libcuda.
struct Driver {
  decltype(&::cuMemCreate) MemCreate; decltype(&::cuMemRelease) MemRelease; decltype(&::cuMemMap) MemMap;
  decltype(&::cuMemUnmap) MemUnmap; decltype(&::cuMemSetAccess) MemSetAccess;
  decltype(&::cuMemAddressReserve) MemAddressReserve; decltype(&::cuMemAddressFree) MemAddressFree;
  decltype(&::cuTensorMapEncodeTiled) TensorMapEncodeTiled;
  decltype(&::cuStreamWaitValue32) StreamWaitValue32;     // may be NULL (the host-memory executor then serialises its launches)
};
const Driver& drv();

struct Geo {                     // engine geometry of an nx * ny * nz domain
  int nx, ny, nz, NI, NJ, NB, NZP, NPAIR;
  long PLREC, ROWREC;            // field records per plane of a row, per row
  size_t PS, XBUF, YBUF;         // coefficient plane stride; pool slab sizes
};
inline Geo geometry(int64_t nx, int64_t ny, int64_t nz, bool padded_rows)
{
  Geo g;
  g.nx = (int)nx; g.ny = (int)ny; g.nz = (int)nz;
  g.NI = (int)((nx + 2 * kSSH + 15) / 16); g.NJ = (int)((ny + kS - 1) / kS);
  g.NB = ((int)nz + 2 * kK + 2 - kSW0 + kNCOPY - 1) / kNCOPY * kNCOPY;
  g.NZP = g.NB + 2 * kK + 8; g.NPAIR = 8 * g.NI + kGOFF + 1; g.PLREC = (long)g.NPAIR * kS;
  g.ROWREC = padded_rows ? (long)rup((size_t)g.NZP * g.PLREC, (size_t)1 << 18) : (long)g.NZP * g.PLREC;
  g.PS = rup((size_t)g.NPAIR * 720, 128);
  g.XBUF = (size_t)(g.NB + 2) * kXGS; g.YBUF = (size_t)(g.NB + 1) * kYGS;
  return g;
}

struct Knobs { int grid = 0, tx = 12, gate = 1, mx = 1100; };   // in-GPU-memory executor (grid 0 = SM count)
struct Ring { int S = 1, nslot = 0, qmax = 0; double lam = 3; }; // host-memory executor: rows per slot, slots, deepest Q, lag

struct HostDomain {              // the domain in pinned host memory, one record row + its coefficients per row
  Geo G;
  std::vector<int> devs;
  Ring ring;
  char* rows = nullptr;          // row J at rows + J HRS: fields RSB bytes, then coefficients CSB bytes
  char* zero = nullptr;          // RSB bytes of +0 (the ghost row above the domain)
  size_t RSB = 0, CSB = 0, CROW = 0, HRS = 0, bytes = 0;
};

// One kernel text compiled for one architecture (object.cu).
struct Object {
  const char* name;              // kernel text, e.g. "zs7_tma"
  const char* arch;              // e.g. "sm_90a"
  int cc;                        // compute capability it runs on, 10 * major + minor
  // domain in GPU memory: devs[0..n) hold y stripes of one virtual window when n > 1
  void* (*create)(const Geo& G, const int* devs, int n, const Knobs& k);
  void (*destroy)(void* st);
  void (*set_knobs)(void* st, const Knobs& k);     // reallocates the pools when k.mx changes
  void (*box)(void* st, const int64_t off[3], const int64_t shape[3], uint16_t* const comp[12], int dir);  // 0 in, 1 out; dense BF16
  double (*advance)(void* st, int64_t passes);     // passes * 7 steps; returns seconds
  void (*tail)(void* st, int steps);               // single reference steps
  void (*checksums)(void* st, uint64_t out[6]);
  void (*zero)(void* st);                          // every value +0
  // domain in host memory
  double (*host_run)(HostDomain& h, int passes, int Q);
  void (*host_tail)(HostDomain& h, int steps);
};

const std::vector<const Object*>& objects();       // every object linked into the library

// Host-memory ring rule (the executor's): ring_bad > 0 means the ring of NR rows is too small for depth Q.
long ring_bad(int Q, int NR, int NI, int NJ, int ngpus, int passes, double lam, int grid);

}  // namespace afdtd

// ================================================================================================ the plan (host code)
#include <mutex>
namespace afdtd {
struct Plan {
  int64_t n[3] = {0, 0, 0};
  std::vector<int> devs;
  unsigned flags = 0;
  int rigor = 0, from_wisdom = 0, on_host = 0, cc = 0, nsm = 0, small_pools = 0;
  std::string gpu_name, sched;   // device name; schedule as in its wisdom record
  const Object* obj = nullptr;
  Knobs knobs; void* mem = nullptr;      // domain in GPU memory
  HostDomain host;                       // domain in host memory
  size_t gpu_bytes = 0, host_bytes = 0;
  double predicted = 0, measured = 0, last_rate = 0;
  int64_t steps_done = 0, tail_steps = 0;
  uint64_t wz_coef = 0, wz_field = 0;
  bool broken = false;
  std::mutex mu;
};
Plan* plan_create(int64_t nx, int64_t ny, int64_t nz, int ngpus, const int* devices, size_t host_budget, int rigor, unsigned flags);
void plan_destroy(Plan* p);
void plan_execute(Plan& p, int64_t steps);
void plan_checksums(Plan& p, uint64_t out[6]);
void bytes_needed(int64_t nx, int64_t ny, int64_t nz, int ngpus, size_t* gpu, size_t* host);
std::string plan_describe(const Plan& p);
// boxio.cpp
struct BoxSpec { int64_t off[3], shape[3]; unsigned flags; struct { void* data; int dtype; int64_t stride[3]; } a[12]; };
void box_io(Plan& p, const BoxSpec& b, int dir);           // set (0, lower-wall rule applied) or get (1); checks first
void host_alloc(HostDomain& h);                            // allocate and pin h.bytes, every value +0
void host_free(HostDomain& h);
void host_checksums(const HostDomain& h, uint64_t out[6]);
void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& f);   // [begin, end) chunks on the CPU threads
// api.cpp: wisdom (one library-wide lock)
bool wisdom_find(const std::string& key, int min_rigor, std::string& sched, int& rigor, double& rate);
void wisdom_add(const std::string& key, const std::string& sched, int rigor, double rate);
}  // namespace afdtd
