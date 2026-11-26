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
#include <algorithm>
#include <cstddef>
#include <mutex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <cuda.h>
#include <cuda_runtime_api.h>

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
// Plans that share a GPU take turns: each API call holds its GPUs' locks (api.cpp), taken in ascending device order.  While a
// launch with recorders or injectors runs, the calling thread must keep enqueuing work; a call that frees memory or loads a module
// on that GPU from another thread blocks those enqueues until the GPU is idle, which it never is.
std::recursive_mutex& device_lock(int dev);
struct DeviceLocks {
  std::vector<int> d;
  explicit DeviceLocks(std::vector<int> devs) : d(std::move(devs)) { std::sort(d.begin(), d.end()); for (int x : d) device_lock(x).lock(); }
  ~DeviceLocks() { for (auto it = d.rbegin(); it != d.rend(); ++it) device_lock(*it).unlock(); }
  DeviceLocks(const DeviceLocks&) = delete; DeviceLocks& operator=(const DeviceLocks&) = delete;
};

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

// Agent work: host-enqueued GPU work on the state at 7-step boundaries inside a persistent launch (recorders' copies, injectors'
// adds), on one non-blocking stream per GPU.  The engine waits for the pass that produces the state, runs the work, and holds back
// the blocks of the next pass that would overwrite (or, for writes, read) the region.  Regions are in rows of kS y cells and x pairs.
struct Region { int J0, J1, q0, q1; bool write; };        // rows [J0, J1), pairs [q0, q1) (pair q: x = 2q, 2q + 1), all z
struct AgentDev {                       // device g owns rows [Ja, Jb) (16-bit engine) or x planes [Xa, Xb) (FP32: dense arrays at fld)
  int g, dev, Ja, Jb; char* fld; long ROWREC, PLREC; cudaStream_t st;
  int prec = 0; int64_t nx = 0, ny = 0, nz = 0; int Xa = 0, Xb = 0;
};
struct AgentHooks {
  std::function<bool(int64_t t)> want;                                 // work on the state at plan step t (t % 7 == 0)?
  std::function<void(int64_t t, std::vector<Region>& out)> regions;    // what that work touches (no side effects)
  std::function<void(int64_t t)> begin;                                // before the work at t is enqueued (may block: back-pressure)
  std::function<void(int64_t t, const AgentDev& d, int q0, int q1)> work;   // enqueue device d.g's part (its rows only, copies of x
                                                                       // pairs [q0, q1) only; writes all) on d.st
  std::function<void(int64_t t)> end;                                  // after every device's part at t is enqueued
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
  double (*advance)(void* st, int64_t passes, int64_t step0, const AgentHooks* h);   // passes * 7 steps from plan step step0; seconds
  void (*agents_now)(void* st, int64_t t, const AgentHooks& h);                      // the work at t on the state as it is (synchronous)
  void (*tail)(void* st, int steps);               // single reference steps
  void (*checksums)(void* st, uint64_t out[6]);
  void (*zero)(void* st);                          // every value +0
  // domain in host memory
  double (*host_run)(HostDomain& h, int passes, int Q);
  void (*host_tail)(HostDomain& h, int steps);
  int prec;                                        // the state's element type: AFDTD_BF16, AFDTD_F16 or AFDTD_F32
};

const std::vector<const Object*>& objects();       // every object linked into the library (the 16-bit engines)
extern const Object kF32Object;                    // f32.cu: the FP32 executor, every architecture
size_t f32_bytes(int64_t nx, int64_t ny, int64_t nz, int ngpus);   // its GPU memory, busiest GPU

// Host-memory ring rule (the executor's): ring_bad > 0 means the ring of NR rows is too small for depth Q.
long ring_bad(int Q, int NR, int NI, int NJ, int ngpus, int passes, double lam, int grid);

}  // namespace afdtd

// ================================================================================================ the plan (host code)
#include <memory>
#include <mutex>
namespace afdtd {
struct Recorder;                         // record.cpp
struct Injector;                         // inject.cu
struct Plan {
  int64_t n[3] = {0, 0, 0};
  std::vector<int> devs;
  unsigned flags = 0;
  int rigor = 0, from_wisdom = 0, on_host = 0, cc = 0, nsm = 0, small_pools = 0;
  int prec = 1;                          // the state's element type: AFDTD_BF16 (1), AFDTD_F32 (2) or AFDTD_F16 (3)
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
  std::vector<std::shared_ptr<Recorder>> rec;   // recorders, by id (null once closed); shared_ptr: Recorder is opaque here
  std::vector<std::shared_ptr<Injector>> inj;   // injectors, by id (null once closed)
};
Plan* plan_create(int64_t nx, int64_t ny, int64_t nz, int ngpus, const int* devices, size_t host_budget, int rigor, unsigned flags);
void plan_destroy(Plan* p);
void plan_execute(Plan& p, int64_t steps);
void plan_checksums(Plan& p, uint64_t out[6]);
void bytes_needed(int64_t nx, int64_t ny, int64_t nz, int ngpus, size_t* gpu, size_t* host, unsigned flags = 0);
std::string plan_describe(const Plan& p);
// record.cpp: recorders (frames of a box at every multiple of `every` steps, to a file or a callback)
struct RecOptions { int64_t off[3], shape[3], every; std::string path; int (*cb)(void*, int64_t, const void*, size_t); void* user; int slots; };
struct RecInfo { int64_t off[3], shape[3], every, row0, rows, z0, planes, pair0, pairs, first_step, frames; size_t frame_bytes, frame_stride, host_bytes;
                 double stall_s, sink_s, copy_s; int dtype; };
int record_open(Plan& p, const RecOptions& o);             // id
void record_info(Plan& p, int id, RecInfo& out);
void record_flush(Plan& p, int id);                        // every frame so far delivered to the sink
void record_close(Plan& p, int id);
void record_close_all(Plan& p) noexcept;
bool record_active(const Plan& p);
AgentHooks record_hooks(Plan& p);                          // the recorders' work at 7-step boundaries
std::string record_take_error(Plan& p);                    // the recorders' sink failures since the last call ("" if none)
bool in_recorder_thread();                                 // the calling thread runs recorder callbacks
// inject.cu: injectors (state += sum_j w[k][j] G_j at the steps first + 7 k)
struct InjOptions { int64_t off[3], shape[3], first, nrows; int M; const float* basis; const float* weights; };
int inject_open(Plan& p, const InjOptions& o);
void inject_close(Plan& p, int id);
void inject_close_all(Plan& p) noexcept;
bool inject_active(const Plan& p);
bool inject_due_between(const Plan& p, int64_t t0, int64_t t1);   // an injection at a step in [t0, t1]
AgentHooks inject_hooks(Plan& p);
// dft.cu: the DFT of a recording file (afdtd_dft)
void dft_file(const char* path, const double* omega, int nfreq, unsigned comps, int device, size_t gpu_bytes, float* out, const char* out_path);
// boxio.cpp
struct BoxSpec { int64_t off[3], shape[3]; unsigned flags; struct { void* data; int dtype; int64_t stride[3]; } a[12]; };
void box_io(Plan& p, const BoxSpec& b, int dir);           // set (0, lower-wall rule applied) or get (1); checks first
void host_alloc(HostDomain& h);                            // allocate and pin h.bytes, every value +0
uint16_t f32_to_f16(uint32_t f32bits);                   // IEEE half, round to nearest even (boxio.cpp)
uint32_t f16_to_f32(uint16_t h);                           // exact
void* pinned_alloc(size_t bytes);                          // pinned (portable), huge pages, zeroed; NULL on failure
void pinned_free(void* p) noexcept;
void host_free(HostDomain& h);
void host_checksums(const HostDomain& h, uint64_t out[6]);
void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& f);   // [begin, end) chunks on the CPU threads
// api.cpp: wisdom (one library-wide lock)
bool wisdom_find(const std::string& key, int min_rigor, std::string& sched, int& rigor, double& rate);
void wisdom_add(const std::string& key, const std::string& sched, int rigor, double rate);
}  // namespace afdtd
