// SPDX-License-Identifier: MIT
// api.cpp -- the C entry points of alpha_fdtd.h: argument checks, status codes and the thread-local error text, the per-plan lock,
// the exception firewall (no C++ exception crosses the ABI), the caller's current device restored, and wisdom.
#pragma GCC visibility push(default)
#include "alpha_fdtd.h"
#pragma GCC visibility pop
#include "internal.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

using namespace afdtd;

static thread_local std::string t_err;
#define AFDTD_EXPORT extern "C"

namespace {
struct DeviceGuard {             // restores the calling thread's current device
  int dev = -1;
  DeviceGuard() { if (cudaGetDevice(&dev) != cudaSuccess) { dev = -1; cudaGetLastError(); } }
  ~DeviceGuard() { if (dev >= 0) cudaSetDevice(dev); }
};
}

struct afdtd_plan_s { Plan* p; };

namespace {
// The text starts with the code's name ("AFDTD_ENOMEM: ..."): afdtd_plan_create returns only NULL, so this is how its caller tells the codes apart.
int set_error(int code, const char* msg)
{
  static const char* names[] = {"AFDTD_OK", "AFDTD_EINVAL", "AFDTD_ENOMEM", "AFDTD_ENOPLAN", "AFDTD_EUNSUPPORTED", "AFDTD_EDEVICE", "AFDTD_EWALLS", "AFDTD_EIO", "AFDTD_EINTERNAL"};
  t_err = std::string(code < 0 && code >= -8 ? names[-code] : "AFDTD_EINTERNAL") + ": " + msg;
  return code;
}
// Run f with the error text cleared and every exception turned into a status code; a CUDA failure inside a plan call makes it unusable.
template <class F> int guarded(afdtd_plan_s* h, F f)
{
  t_err.clear();
  DeviceGuard dg;
  try {
    if (!h) return set_error(AFDTD_EINVAL, "plan is NULL");
    // a recorder's callback runs while the plan's execute (or flush, close) waits for it: a plan call from there would deadlock
    if (in_recorder_thread()) return set_error(AFDTD_EINVAL, "plan functions cannot be called from a recorder callback");
    std::lock_guard<std::mutex> l(h->p->mu);
    DeviceLocks dl(h->p->devs);                       // plans sharing a GPU take turns (internal.hpp)
    if (h->p->broken) return set_error(AFDTD_EDEVICE, "the plan is unusable after an earlier CUDA failure; destroy it");
    try { return f(*h->p); }
    catch (const Error& e) { if (e.code == AFDTD_EDEVICE) h->p->broken = true; throw; }
  } catch (const Error& e) { return set_error(e.code, e.what()); }
  catch (const std::bad_alloc&) { return set_error(AFDTD_ENOMEM, "host memory allocation failed"); }
  catch (const std::exception& e) { return set_error(AFDTD_EINTERNAL, e.what()); }
  catch (...) { return set_error(AFDTD_EINTERNAL, "unknown internal error"); }
}
template <class F> int unguarded(F f)       // the same without a plan
{
  t_err.clear();
  DeviceGuard dg;
  try { return f(); }
  catch (const Error& e) { return set_error(e.code, e.what()); }
  catch (const std::bad_alloc&) { return set_error(AFDTD_ENOMEM, "host memory allocation failed"); }
  catch (const std::exception& e) { return set_error(AFDTD_EINTERNAL, e.what()); }
  catch (...) { return set_error(AFDTD_EINTERNAL, "unknown internal error"); }
}
}  // namespace

// ================================================================================================ plans
AFDTD_EXPORT afdtd_plan afdtd_plan_create(int64_t nx, int64_t ny, int64_t nz, const afdtd_options* o)
{
  afdtd_options z; memset(&z, 0, sizeof z);
  if (!o) o = &z;
  afdtd_plan h = nullptr;
  unguarded([&] {
    std::vector<int> dv; for (int i = 0; i < (o->ngpus > 0 && o->ngpus <= 8 ? o->ngpus : 1); ++i) dv.push_back(o->devices ? o->devices[i] : i);
    DeviceLocks dl(dv);
    Plan* p = plan_create(nx, ny, nz, o->ngpus, o->devices, o->host_bytes, o->rigor, o->flags);
    h = new afdtd_plan_s{p};
    return AFDTD_OK;
  });
  return h;
}
AFDTD_EXPORT void afdtd_plan_destroy(afdtd_plan h)
{
  t_err.clear();
  if (!h) return;
  if (in_recorder_thread()) { set_error(AFDTD_EINVAL, "afdtd_plan_destroy cannot be called from a recorder callback"); return; }
  { std::lock_guard<std::mutex> l(h->p->mu); }       // a call in flight on another thread finishes first
  unguarded([&] { DeviceLocks dl(h->p->devs); plan_destroy(h->p); return AFDTD_OK; });
  delete h;
}
AFDTD_EXPORT int afdtd_bytes_needed(int64_t nx, int64_t ny, int64_t nz, int ngpus, size_t* gpu, size_t* host)
{
  return unguarded([&] {
    for (int64_t v : {nx, ny, nz}) if (v < 16 || v > (1 << 20)) fail(AFDTD_EINVAL, "each axis needs 16 to 1048576 cells");
    if (ngpus < 0 || ngpus > 8) fail(AFDTD_EINVAL, "ngpus = %d: 1 to 8 (0 means 1)", ngpus);
    if ((double)nx * ny * nz > 4e15) fail(AFDTD_EINVAL, "size %lld x %lld x %lld: too many cells", (long long)nx, (long long)ny, (long long)nz);
    bytes_needed(nx, ny, nz, ngpus ? ngpus : 1, gpu, host, 0);
    return AFDTD_OK;
  });
}

// ================================================================================================ fields and stepping
static int box_call(afdtd_plan h, const afdtd_box* box, int dir)
{
  return guarded(h, [&](Plan& p) {
    if (!box) fail(AFDTD_EINVAL, "box is NULL");
    BoxSpec b;
    for (int a = 0; a < 3; ++a) { b.off[a] = box->offset[a]; b.shape[a] = box->shape[a]; }
    b.flags = box->flags;
    for (int c = 0; c < 12; ++c) { b.a[c].data = box->array[c].data; b.a[c].dtype = box->array[c].dtype; for (int a = 0; a < 3; ++a) b.a[c].stride[a] = box->array[c].stride[a]; }
    box_io(p, b, dir);
    return AFDTD_OK;
  });
}
AFDTD_EXPORT int afdtd_set(afdtd_plan h, const afdtd_box* box) { return box_call(h, box, 0); }
AFDTD_EXPORT int afdtd_get(afdtd_plan h, const afdtd_box* box) { return box_call(h, box, 1); }
AFDTD_EXPORT int afdtd_execute(afdtd_plan h, int64_t steps) { return guarded(h, [&](Plan& p) { plan_execute(p, steps); return AFDTD_OK; }); }
AFDTD_EXPORT int afdtd_checksums(afdtd_plan h, uint64_t out[6])
{
  return guarded(h, [&](Plan& p) { if (!out) fail(AFDTD_EINVAL, "out is NULL"); plan_checksums(p, out); return AFDTD_OK; });
}

AFDTD_EXPORT int afdtd_bytes_needed2(int64_t nx, int64_t ny, int64_t nz, int ngpus, unsigned flags, size_t* gpu, size_t* host)
{
  return unguarded([&] {
    if (ngpus < 0 || ngpus > 8) fail(AFDTD_EINVAL, "ngpus = %d: 1 to 8 (0 means 1)", ngpus);
    if (flags & ~(AFDTD_FORCE_HOST | AFDTD_STRICT_WALLS | AFDTD_FP16 | AFDTD_FP32)) fail(AFDTD_EINVAL, "unknown flag bits 0x%x", flags);
    for (int64_t v : {nx, ny, nz}) if (v < 16 || v > (1 << 20)) fail(AFDTD_EINVAL, "each axis needs 16 to 1048576 cells");
    if ((double)nx * ny * nz > 4e15) fail(AFDTD_EINVAL, "size %lld x %lld x %lld: too many cells", (long long)nx, (long long)ny, (long long)nz);
    bytes_needed(nx, ny, nz, ngpus ? ngpus : 1, gpu, host, flags);
    return AFDTD_OK;
  });
}
AFDTD_EXPORT int afdtd_plan_precision(afdtd_plan h) { return guarded(h, [&](Plan& p) { return p.prec; }); }

// ================================================================================================ recording (API 2)
AFDTD_EXPORT int afdtd_record(afdtd_plan h, const afdtd_recorder_options* o)
{
  return guarded(h, [&](Plan& p) {
    if (!o) fail(AFDTD_EINVAL, "options are NULL");
    if (o->flags) fail(AFDTD_EINVAL, "flags = %u: no flags are defined", o->flags);
    RecOptions r;
    for (int a = 0; a < 3; ++a) { r.off[a] = o->offset[a]; r.shape[a] = o->shape[a]; }
    r.every = o->every; r.path = o->path ? o->path : ""; r.cb = o->callback; r.user = o->user; r.slots = o->slots;
    if (o->path && !*o->path) fail(AFDTD_EINVAL, "path is empty");
    return record_open(p, r);
  });
}
AFDTD_EXPORT int afdtd_record_info(afdtd_plan h, int id, afdtd_recorder_info* o)
{
  return guarded(h, [&](Plan& p) {
    if (!o) fail(AFDTD_EINVAL, "info is NULL");
    RecInfo r; record_info(p, id, r);
    memset(o, 0, sizeof *o);
    for (int a = 0; a < 3; ++a) { o->offset[a] = r.off[a]; o->shape[a] = r.shape[a]; }
    o->every = r.every; o->row0 = r.row0; o->rows = r.rows; o->z0 = r.z0; o->planes = r.planes; o->pair0 = r.pair0; o->pairs = r.pairs;
    o->frame_bytes = r.frame_bytes; o->frame_stride = r.frame_stride; o->first_step = r.first_step; o->frames = r.frames;
    o->host_bytes = r.host_bytes; o->stall_s = r.stall_s; o->sink_s = r.sink_s; o->copy_s = r.copy_s; o->dtype = r.dtype;
    return AFDTD_OK;
  });
}
AFDTD_EXPORT int afdtd_record_flush(afdtd_plan h, int id) { return guarded(h, [&](Plan& p) { record_flush(p, id); return AFDTD_OK; }); }
AFDTD_EXPORT int afdtd_record_close(afdtd_plan h, int id) { return guarded(h, [&](Plan& p) { record_close(p, id); return AFDTD_OK; }); }
AFDTD_EXPORT int afdtd_inject(afdtd_plan h, const afdtd_injector_options* o)
{
  return guarded(h, [&](Plan& p) {
    if (!o) fail(AFDTD_EINVAL, "options are NULL");
    if (o->flags) fail(AFDTD_EINVAL, "flags = %u: no flags are defined", o->flags);
    InjOptions j;
    for (int a = 0; a < 3; ++a) { j.off[a] = o->offset[a]; j.shape[a] = o->shape[a]; }
    j.M = o->nbasis; j.basis = o->basis; j.first = o->first_step; j.nrows = o->rows; j.weights = o->weights;
    return inject_open(p, j);
  });
}
AFDTD_EXPORT int afdtd_inject_close(afdtd_plan h, int id) { return guarded(h, [&](Plan& p) { inject_close(p, id); return AFDTD_OK; }); }
AFDTD_EXPORT int afdtd_dft(const char* path, const double* omega, int nfreq, unsigned comps, int device, size_t gpu_bytes, float* out, const char* out_path)
{
  return unguarded([&] { DeviceLocks dl({device}); dft_file(path, omega, nfreq, comps, device, gpu_bytes, out, out_path); return AFDTD_OK; });
}

// ================================================================================================ introspection
AFDTD_EXPORT int afdtd_plan_info(afdtd_plan h, afdtd_info* o)
{
  return guarded(h, [&](Plan& p) {
    if (!o) fail(AFDTD_EINVAL, "info is NULL");
    memset(o, 0, sizeof *o);
    o->nx = p.n[0]; o->ny = p.n[1]; o->nz = p.n[2]; o->ngpus = (int)p.devs.size();
    for (size_t i = 0; i < p.devs.size(); ++i) o->devices[i] = p.devs[i];
    o->on_host = p.on_host; o->rigor = p.rigor; o->from_wisdom = p.from_wisdom; o->io_block_y = kS;
    o->gpu_bytes_used = p.gpu_bytes; o->host_bytes_used = p.host_bytes; o->steps_done = p.steps_done; o->tail_steps = p.tail_steps;
    o->walls_zeroed_coef = p.wz_coef; o->walls_zeroed_field = p.wz_field;
    o->predicted_rate = p.predicted; o->measured_rate = p.measured; o->last_rate = p.last_rate;
    return AFDTD_OK;
  });
}
AFDTD_EXPORT int afdtd_plan_describe(afdtd_plan h, char* buf, size_t n)
{
  return guarded(h, [&](Plan& p) {
    if (!buf && n) fail(AFDTD_EINVAL, "buf is NULL with n > 0");
    const std::string s = plan_describe(p);
    if (n) { const size_t k = std::min(n - 1, s.size()); memcpy(buf, s.data(), k); buf[k] = 0; }
    return (int)s.size();
  });
}

// ================================================================================================ wisdom
// Text, one record per line after the header "alpha-fdtd-wisdom 1":  <key fields> rigor=<r> rate=<r> sched="<schedule>".
// The key is everything before " rigor=".  Import merges: a record of higher rigor replaces one of lower.
namespace {
struct Rec { int rigor; double rate; std::string sched; };
std::mutex g_wm;
std::map<std::string, Rec> g_wisdom;
const char* kRigor[] = {"estimate", "measure", "patient"};

bool parse_line(const std::string& line, std::string& key, Rec& r)
{
  const size_t a = line.find(" rigor="), b = line.find(" rate=", a == std::string::npos ? 0 : a), c = line.find(" sched=\"", b == std::string::npos ? 0 : b);
  if (a == std::string::npos || b == std::string::npos || c == std::string::npos || line.back() != '"' || line.compare(0, 5, "kset=")) return false;
  key = line.substr(0, a);
  const std::string rg = line.substr(a + 7, b - a - 7);
  r.rigor = -1; for (int i = 0; i < 3; ++i) if (rg == kRigor[i]) r.rigor = i;
  char* end = nullptr; const std::string rt = line.substr(b + 6, c - b - 6); r.rate = strtod(rt.c_str(), &end);
  r.sched = line.substr(c + 8, line.size() - c - 9);
  return r.rigor >= 0 && end && !*end && !rt.empty() && !r.sched.empty() && r.sched.find('"') == std::string::npos;
}
int import_text(const std::string& text)
{
  std::istringstream in(text); std::string line; std::map<std::string, Rec> got; int n = 0, ln = 0;
  while (std::getline(in, line)) {
    ++ln;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (ln == 1) { if (line != "alpha-fdtd-wisdom 1") fail(AFDTD_EINVAL, "wisdom: the first line is not \"alpha-fdtd-wisdom 1\""); continue; }
    if (line.empty()) continue;
    std::string k; Rec r;
    if (!parse_line(line, k, r)) fail(AFDTD_EINVAL, "wisdom: line %d is not a record", ln);
    auto it = got.find(k); if (it == got.end() || r.rigor >= it->second.rigor) got[k] = r;
    ++n;
  }
  if (ln == 0) fail(AFDTD_EINVAL, "wisdom: empty text");
  std::lock_guard<std::mutex> l(g_wm);
  for (auto& kv : got) { auto it = g_wisdom.find(kv.first); if (it == g_wisdom.end() || kv.second.rigor >= it->second.rigor) g_wisdom[kv.first] = kv.second; }
  return n;
}
std::string export_text()
{
  std::lock_guard<std::mutex> l(g_wm);
  std::string s = "alpha-fdtd-wisdom 1\n";
  for (auto& kv : g_wisdom) {
    char b[64]; snprintf(b, sizeof b, " rigor=%s rate=%.6g sched=\"", kRigor[kv.second.rigor], kv.second.rate);
    s += kv.first + b + kv.second.sched + "\"\n";
  }
  return s;
}
}  // namespace

namespace afdtd {
bool wisdom_find(const std::string& key, int min_rigor, std::string& sched, int& rigor, double& rate)
{
  std::lock_guard<std::mutex> l(g_wm);
  auto it = g_wisdom.find(key);
  if (it == g_wisdom.end() || it->second.rigor < min_rigor) return false;
  sched = it->second.sched; rigor = it->second.rigor; rate = it->second.rate;
  return true;
}
void wisdom_add(const std::string& key, const std::string& sched, int rigor, double rate)
{
  std::lock_guard<std::mutex> l(g_wm);
  auto it = g_wisdom.find(key);
  if (it == g_wisdom.end() || rigor >= it->second.rigor) g_wisdom[key] = Rec{rigor, rate, sched};
}
}  // namespace afdtd

AFDTD_EXPORT char* afdtd_export_wisdom_to_string(void)
{
  char* out = nullptr;
  unguarded([&] { const std::string s = export_text(); out = (char*)malloc(s.size() + 1); if (!out) fail(AFDTD_ENOMEM, "out of host memory"); memcpy(out, s.c_str(), s.size() + 1); return AFDTD_OK; });
  return out;
}
AFDTD_EXPORT int afdtd_import_wisdom_from_string(const char* text)
{
  return unguarded([&] { if (!text) fail(AFDTD_EINVAL, "text is NULL"); return import_text(text); });
}
AFDTD_EXPORT int afdtd_export_wisdom_to_filename(const char* path)
{
  return unguarded([&] {
    if (!path) fail(AFDTD_EINVAL, "path is NULL");
    std::ofstream f(path, std::ios::binary); const std::string s = export_text(); f << s; f.close();
    if (!f) fail(AFDTD_EIO, "wisdom: cannot write %s", path);
    return AFDTD_OK;
  });
}
AFDTD_EXPORT int afdtd_import_wisdom_from_filename(const char* path)
{
  return unguarded([&] {
    if (!path) fail(AFDTD_EINVAL, "path is NULL");
    std::ifstream f(path, std::ios::binary);
    if (!f) fail(AFDTD_EIO, "wisdom: cannot read %s", path);
    std::stringstream ss; ss << f.rdbuf();
    return import_text(ss.str());
  });
}
AFDTD_EXPORT void afdtd_forget_wisdom(void) { t_err.clear(); std::lock_guard<std::mutex> l(g_wm); g_wisdom.clear(); }

// ================================================================================================ misc
AFDTD_EXPORT void afdtd_free(void* p) { free(p); }
AFDTD_EXPORT const char* afdtd_version(void) { return "0.2.0"; }
AFDTD_EXPORT const char* afdtd_last_error(void) { return t_err.c_str(); }

// ================================================================================================ the kernel objects built in
#define AFDTD_OBJECT(sym) extern const afdtd::Object sym;
#include "objects.inc"
#undef AFDTD_OBJECT
const std::vector<const afdtd::Object*>& afdtd::objects()
{
#define AFDTD_OBJECT(sym) &sym,
  static const std::vector<const afdtd::Object*> v = {
#include "objects.inc"
  };
#undef AFDTD_OBJECT
  return v;
}
