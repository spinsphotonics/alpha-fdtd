// SPDX-License-Identifier: MIT
// record.cpp -- recorders: frames of a box of the state (E and H', the native record layout) at every multiple of `every` steps,
// copied by the GPUs' copy engines during afdtd_execute (object.cu: agents) into pinned host slots and drained by one writer thread
// per recorder, to a file (O_DIRECT where the file system allows it, large parallel writes) or to the caller's callback.
//
// Frame layout: rows [row0, row0 + rows) of kS = 24 y cells, z planes [z0, z0 + planes), x pairs [pair0, pair0 + pairs); per (row, plane,
// pair) 24 lanes (y = 24 row + lane) of one record each, a record = 6 words Ex Ey Ez Hx Hy Hz, word = BF16 of x = 2 pair (low half) and
// 2 pair + 1 (high half).  The value of component c at cell (x, y, z) is uint16 element
//     ((((y / 24 - row0) * planes + (z - z0)) * pairs + (x / 2 - pair0)) * 24 + y % 24) * 12 + 2 c + (x & 1).
// Cells of the frame outside the box (or the domain) hold anything.  File: a 4096-byte JSON header (spaces after the text), then the
// frames, frame k at byte 4096 + k * frame_stride (frame_stride = frame_bytes rounded up to 4096), frame k at step first_step + k every.
#include "alpha_fdtd.h"
#include "internal.hpp"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <thread>

namespace afdtd {

static double rnow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static thread_local bool t_writer = false;     // this thread is a recorder's writer (its callbacks run here)
bool in_recorder_thread() { return t_writer; }

struct Recorder {
  RecOptions o;
  int prec = AFDTD_BF16;                                   // the plan's element type; FP32 frames are canonical floats
  int64_t n[3] = {0, 0, 0};
  int J0 = 0, J1 = 0, z0 = 0, z1 = 0, q0 = 0, q1 = 0;      // frame: rows, planes, pairs
  size_t row_bytes = 0, frame_bytes = 0, stride = 0;
  int nslot = 3, ndev = 1; std::vector<int> devs;
  char* slots = nullptr;                                   // nslot * stride bytes, pinned, portable
  std::vector<cudaEvent_t> ev, ev0; std::vector<char> rec; // [slot * ndev + g]: copies of device g done / started; recorded for this frame
  std::mutex m; std::condition_variable cv;
  std::deque<std::pair<int, int64_t>> q;                   // (slot, step): enqueued, not yet delivered
  std::vector<char> busy;                                  // slot holds a frame not yet delivered
  std::vector<double> landed;                              // per slot: when its frame's copies were seen complete (writer)
  int cur = -1;                                            // slot of the frame being enqueued (begin .. end), -1: none
  bool stop = false, err_new = false;
  std::thread writer;
  int fd = -1; bool direct = false; char* hdr = nullptr;  // 4096 bytes, aligned (O_DIRECT)
  int64_t first_step = -1, last_step = -1, frames = 0, delivered = 0, written = 0;   // written: frames in the file (a prefix)
  double stall = 0, sink_s = 0, copy_s = 0;
  std::string err;                                         // first failure (sticky)

  bool due(int64_t t) const { return t % o.every == 0 && t > last_step; }
  void fail_sink(const std::string& e) { std::lock_guard<std::mutex> l(m); if (err.empty()) { err = e; err_new = true; } }

  void write_header(bool complete)
  {
    memset(hdr, ' ', 4096);
    const int k = snprintf(hdr, 4096,
      "{\"format\": \"alpha-fdtd-record\", \"version\": 1, \"domain\": [%lld, %lld, %lld], \"offset\": [%lld, %lld, %lld], "
      "\"shape\": [%lld, %lld, %lld], \"every\": %lld, \"first_step\": %lld, \"frames\": %lld, \"complete\": %s, "
      "\"row0\": %d, \"rows\": %d, \"z0\": %d, \"planes\": %d, \"pair0\": %d, \"pairs\": %d, \"frame_bytes\": %zu, \"frame_stride\": %zu, "
      "\"data_offset\": 4096, \"dtype\": \"%s\", \"layout\": \"%s\"}\n",
      (long long)n[0], (long long)n[1], (long long)n[2], (long long)o.off[0], (long long)o.off[1], (long long)o.off[2],
      (long long)o.shape[0], (long long)o.shape[1], (long long)o.shape[2], (long long)o.every, (long long)first_step,
      (long long)written, complete ? "true" : "false", J0, J1 - J0, z0, z1 - z0, q0, q1 - q0, frame_bytes, stride,
      prec == AFDTD_F32 ? "f32" : prec == AFDTD_F16 ? "f16" : "bf16",
      prec == AFDTD_F32 ? "canonical: component (Ex Ey Ez Hx Hy Hz), x, y, z over the box; float32"
                        : "native: row, plane, pair, lane (24), component (Ex Ey Ez Hx Hy Hz), x & 1; 16-bit");
    if (k > 0 && k < 4096) hdr[k] = ' ';
    hdr[4095] = '\n';
    if (pwrite(fd, hdr, 4096, 0) != 4096) fail_sink(std::string("cannot write the header of ") + o.path + ": " + strerror(errno));
  }
  // one frame to the file: large writes from the pinned slot, in parallel (queue depth for striped NVMe)
  void write_frame(int64_t k, const char* src)
  {
    const size_t chunk = (size_t)64 << 20, nchunk = (stride + chunk - 1) / chunk;
    const off_t base = (off_t)4096 + (off_t)k * (off_t)stride;
    std::atomic<bool> bad{false}; std::string why;
    auto one = [&](size_t c0, size_t c1) {
      for (size_t c = c0; c < c1 && !bad; ++c) {
        size_t o2 = c * chunk; const size_t e = std::min(stride, o2 + chunk);
        while (o2 < e) {
          const ssize_t w = pwrite(fd, src + o2, e - o2, base + (off_t)o2);
          if (w <= 0) { if (w < 0 && errno == EINTR) continue; bad = true; break; }
          o2 += (size_t)w;
        }
      }
    };
    const size_t nt = std::min<size_t>(4, nchunk);
    if (nt <= 1) one(0, nchunk);
    else { std::vector<std::thread> th; for (size_t i = 0; i < nt; ++i) th.emplace_back(one, i * nchunk / nt, (i + 1) * nchunk / nt); for (auto& t : th) t.join(); }
    if (bad) fail_sink(std::string("cannot write frame ") + std::to_string(k) + " to " + o.path + ": " + strerror(errno));
  }
  void run_writer()
  {
    t_writer = true;
    for (;;) {
      std::pair<int, int64_t> f;
      { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return !q.empty() || stop; }); if (q.empty()) return; f = q.front(); }
      const int s = f.first;
      for (int g = 0; g < ndev; ++g) if (rec[(size_t)s * ndev + g]) {
        cudaSetDevice(devs[g]); const cudaError_t e = cudaEventSynchronize(ev[(size_t)s * ndev + g]);
        if (e != cudaSuccess) fail_sink(std::string("CUDA: ") + cudaGetErrorString(e) + " while copying a frame");
        float ms = 0; if (cudaEventElapsedTime(&ms, ev0[(size_t)s * ndev + g], ev[(size_t)s * ndev + g]) == cudaSuccess) copy_s += ms * 1e-3; else cudaGetLastError();
      }
      { std::lock_guard<std::mutex> l(m); landed[(size_t)s] = rnow(); }
      bool ok; { std::lock_guard<std::mutex> l(m); ok = err.empty(); }
      if (ok) {
        const double t0 = rnow();
        if (fd >= 0) { write_frame(delivered, slots + (size_t)s * stride); std::lock_guard<std::mutex> l(m); if (err.empty()) ++written; }
        else if (o.cb(o.user, f.second, slots + (size_t)s * stride, frame_bytes) != 0) fail_sink("the frame callback returned an error at step " + std::to_string(f.second));
        sink_s += rnow() - t0;
      }
      std::lock_guard<std::mutex> l(m);
      q.pop_front(); busy[(size_t)s] = 0; ++delivered; cv.notify_all();
    }
  }
  void flush()
  {
    std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return q.empty(); });
    l.unlock();
    if (fd >= 0) write_header(false);
  }
  void close() noexcept
  {
    try { flush(); } catch (...) {}
    { std::lock_guard<std::mutex> l(m); stop = true; } cv.notify_all();
    if (writer.joinable()) writer.join();
    if (fd >= 0) { bool ok; { std::lock_guard<std::mutex> l(m); ok = err.empty(); } write_header(ok); fsync(fd); ::close(fd); fd = -1; }
    for (size_t i = 0; i < ev.size(); ++i) { cudaSetDevice(devs[i % ndev]); if (ev[i]) cudaEventDestroy(ev[i]); if (i < ev0.size() && ev0[i]) cudaEventDestroy(ev0[i]); }
    ev.clear(); ev0.clear();
    pinned_free(slots); slots = nullptr;
    free(hdr); hdr = nullptr;
  }
  ~Recorder() { close(); }
};

static Recorder& get(Plan& p, int id)
{
  if (id < 0 || id >= (int)p.rec.size() || !p.rec[(size_t)id]) fail(AFDTD_EINVAL, "no open recorder with id %d", id);
  return *p.rec[(size_t)id];
}

int record_open(Plan& p, const RecOptions& o)
{
  if (p.on_host) fail(AFDTD_EINVAL, "recording needs the domain in GPU memory (this plan keeps it in host memory)");
  for (int a = 0; a < 3; ++a)
    if (o.shape[a] < 1 || o.off[a] < 0 || o.off[a] + o.shape[a] > p.n[a])
      fail(AFDTD_EINVAL, "recorder box offset %lld shape %lld on axis %d is outside the %lld-cell domain", (long long)o.off[a], (long long)o.shape[a], a, (long long)p.n[a]);
  if (o.every < kK || o.every % kK) fail(AFDTD_EINVAL, "every = %lld: a multiple of 7 is required", (long long)o.every);
  if (o.path.empty() == !o.cb) fail(AFDTD_EINVAL, "give exactly one of path and callback");
  if (o.slots < 0 || o.slots > 64) fail(AFDTD_EINVAL, "slots = %d (0 = 3, at most 64)", o.slots);
  auto r = std::make_shared<Recorder>();
  r->o = o; r->prec = p.prec; for (int a = 0; a < 3; ++a) r->n[a] = p.n[a];
  r->z0 = (int)o.off[2]; r->z1 = (int)(o.off[2] + o.shape[2]);
  if (p.prec == AFDTD_F32) {                               // canonical frames: (component, x, y, z) floats over the box
    r->frame_bytes = (size_t)6 * o.shape[0] * o.shape[1] * o.shape[2] * 4;
  } else {
    r->J0 = (int)(o.off[1] / kS); r->J1 = (int)((o.off[1] + o.shape[1] - 1) / kS + 1);
    r->q0 = (int)(o.off[0] / 2); r->q1 = (int)((o.off[0] + o.shape[0] - 1) / 2 + 1);
    r->row_bytes = (size_t)(r->q1 - r->q0) * kS * 24;
    r->frame_bytes = (size_t)(r->J1 - r->J0) * (size_t)(r->z1 - r->z0) * r->row_bytes;
  }
  r->stride = rup(r->frame_bytes, 4096);
  r->nslot = o.slots ? o.slots : 3; r->devs = p.devs; r->ndev = (int)p.devs.size();
  r->busy.assign((size_t)r->nslot, 0); r->rec.assign((size_t)r->nslot * r->ndev, 0); r->landed.assign((size_t)r->nslot, 0.0);
  r->last_step = p.steps_done - 1;
  r->slots = (char*)pinned_alloc((size_t)r->nslot * r->stride);
  if (!r->slots) fail(AFDTD_ENOMEM, "cannot pin %zu bytes of host memory for %d frame slots of %zu bytes", (size_t)r->nslot * r->stride, r->nslot, r->stride);
  r->ev.assign((size_t)r->nslot * r->ndev, nullptr); r->ev0.assign(r->ev.size(), nullptr);
  for (size_t i = 0; i < r->ev.size(); ++i) { AF_CK(cudaSetDevice(r->devs[i % r->ndev])); AF_CK(cudaEventCreateWithFlags(&r->ev[i], cudaEventBlockingSync));
                                              AF_CK(cudaEventCreate(&r->ev0[i])); }
  if (!o.path.empty()) {
    if (posix_memalign((void**)&r->hdr, 4096, 4096)) { r->hdr = nullptr; fail(AFDTD_ENOMEM, "header buffer"); }
#ifdef O_DIRECT
    r->fd = ::open(o.path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644); r->direct = r->fd >= 0;
#endif
    if (r->fd < 0) r->fd = ::open(o.path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (r->fd < 0) fail(AFDTD_EIO, "cannot open %s for writing: %s", o.path.c_str(), strerror(errno));
    r->write_header(false);
    if (!r->err.empty()) fail(AFDTD_EIO, "%s", r->err.c_str());
  }
  Recorder* rp = r.get();
  r->writer = std::thread([rp] { rp->run_writer(); });
  p.rec.push_back(r);
  return (int)p.rec.size() - 1;
}

void record_info(Plan& p, int id, RecInfo& out)
{
  Recorder& r = get(p, id);
  for (int a = 0; a < 3; ++a) { out.off[a] = r.o.off[a]; out.shape[a] = r.o.shape[a]; }
  out.every = r.o.every; out.row0 = r.J0; out.rows = r.J1 - r.J0; out.z0 = r.z0; out.planes = r.z1 - r.z0; out.pair0 = r.q0; out.pairs = r.q1 - r.q0;
  std::lock_guard<std::mutex> l(r.m);
  out.first_step = r.first_step; out.frames = r.frames; out.frame_bytes = r.frame_bytes; out.frame_stride = r.stride;
  out.host_bytes = (size_t)r.nslot * r.stride; out.stall_s = r.stall; out.sink_s = r.sink_s; out.copy_s = r.copy_s; out.dtype = r.prec;
}
void record_flush(Plan& p, int id)
{
  Recorder& r = get(p, id); r.flush();
  std::lock_guard<std::mutex> l(r.m);
  if (!r.err.empty()) { r.err_new = false; fail(AFDTD_EIO, "%s", r.err.c_str()); }
}
void record_close(Plan& p, int id)
{
  Recorder& r = get(p, id); r.close();
  std::string e; { std::lock_guard<std::mutex> l(r.m); e = r.err; }
  p.rec[(size_t)id].reset();
  if (!e.empty()) fail(AFDTD_EIO, "%s", e.c_str());
}
void record_close_all(Plan& p) noexcept { for (auto& r : p.rec) if (r) { r->close(); r.reset(); } }
bool record_active(const Plan& p) { for (auto& r : p.rec) if (r) return true; return false; }
std::string record_take_error(Plan& p)
{
  std::string all;
  for (size_t i = 0; i < p.rec.size(); ++i) if (p.rec[i]) {
    std::lock_guard<std::mutex> l(p.rec[i]->m);
    if (p.rec[i]->err_new) { p.rec[i]->err_new = false; all += (all.empty() ? "" : "; ") + std::string("recorder ") + std::to_string(i) + ": " + p.rec[i]->err; }
  }
  return all;
}

AgentHooks record_hooks(Plan& p)
{
  AgentHooks h;
  Plan* pp = &p;
  h.want = [pp](int64_t t) { for (auto& r : pp->rec) if (r && r->due(t)) return true; return false; };
  h.regions = [pp](int64_t t, std::vector<Region>& out) { for (auto& r : pp->rec) if (r && r->due(t)) out.push_back(Region{r->J0, r->J1, r->q0, r->q1, false}); };
  h.begin = [pp](int64_t t) {
    for (auto& rr : pp->rec) {
      if (!rr || !rr->due(t)) continue;
      Recorder& r = *rr; std::unique_lock<std::mutex> l(r.m);
      r.cur = -1;
      if (!r.err.empty()) continue;                      // a failed sink: no copies (the steps still run)
      const int s = (int)(r.frames % r.nslot);
      // waiting for a slot is normal (the host runs ahead of the GPUs); only the wait after the slot's frame had landed in host
      // memory is the sink's fault (stall_s): that is what can hold the GPUs back
      if (r.busy[(size_t)s]) { const double t0 = rnow(); r.cv.wait(l, [&] { return !r.busy[(size_t)s]; });
                               const double t1 = rnow(), la = r.landed[(size_t)s]; if (la > 0) r.stall += std::max(0.0, t1 - std::max(t0, la)); }
      r.busy[(size_t)s] = 1; r.cur = s; r.landed[(size_t)s] = 0;            // the slot's new frame has not landed yet
      for (int g = 0; g < r.ndev; ++g) r.rec[(size_t)s * r.ndev + g] = 0;
    }
  };
  // called once per part of the frame (the x pairs [q0, q1) that one strip of the schedule writes), in pass order
  h.work = [pp](int64_t t, const AgentDev& d, int q0, int q1) {
    for (auto& rr : pp->rec) {
      if (!rr || !rr->due(t) || rr->cur < 0) continue;
      Recorder& r = *rr;
      char* slot = r.slots + (size_t)r.cur * r.stride;
      const size_t e = (size_t)r.cur * r.ndev + d.g;
      if (d.prec == AFDTD_F32) {                           // dense arrays: one 3-D copy per component of the box's part on this GPU
        const int64_t sx = r.o.shape[0], sy = r.o.shape[1], sz = r.o.shape[2], xa = std::max<int64_t>(r.o.off[0], d.Xa), xb = std::min<int64_t>(r.o.off[0] + sx, d.Xb);
        if (xa >= xb) continue;
        if (!r.rec[e]) AF_CK(cudaEventRecord(r.ev0[e], d.st));
        const size_t N = (size_t)d.nx * d.ny * d.nz;
        for (int c = 0; c < 6; ++c) {
          cudaMemcpy3DParms m; memset(&m, 0, sizeof m);
          m.srcPtr = make_cudaPitchedPtr(d.fld + (size_t)c * N * 4, (size_t)d.nz * 4, (size_t)d.nz, (size_t)d.ny);
          m.srcPos = make_cudaPos((size_t)r.o.off[2] * 4, (size_t)r.o.off[1], (size_t)xa);
          m.dstPtr = make_cudaPitchedPtr(slot + (size_t)c * sx * sy * sz * 4, (size_t)sz * 4, (size_t)sz, (size_t)sy);
          m.dstPos = make_cudaPos(0, 0, (size_t)(xa - r.o.off[0]));
          m.extent = make_cudaExtent((size_t)sz * 4, (size_t)sy, (size_t)(xb - xa)); m.kind = cudaMemcpyDeviceToHost;
          AF_CK(cudaMemcpy3DAsync(&m, d.st));
        }
        AF_CK(cudaEventRecord(r.ev[e], d.st)); r.rec[e] = 1;
        continue;
      }
      const int a = std::max(r.J0, d.Ja), b = std::min(r.J1, d.Jb), qa = std::max(r.q0, q0), qb = std::min(r.q1, q1);
      if (a >= b || qa >= qb) continue;
      if (!r.rec[e]) AF_CK(cudaEventRecord(r.ev0[e], d.st));             // the first part of this frame on this GPU
      for (int J = a; J < b; ++J) {
        const char* src = d.fld + ((size_t)J * d.ROWREC + (size_t)(r.z0 + kSOFF) * d.PLREC + (size_t)(qa + kGOFF) * kS) * 24;
        char* dst = slot + (size_t)(J - r.J0) * (size_t)(r.z1 - r.z0) * r.row_bytes + (size_t)(qa - r.q0) * kS * 24;
        AF_CK(cudaMemcpy2DAsync(dst, r.row_bytes, src, (size_t)d.PLREC * 24, (size_t)(qb - qa) * kS * 24, (size_t)(r.z1 - r.z0), cudaMemcpyDeviceToHost, d.st));
      }
      AF_CK(cudaEventRecord(r.ev[e], d.st)); r.rec[e] = 1;                // re-recorded by later parts: the last one counts
    }
  };
  h.end = [pp](int64_t t) {
    for (auto& rr : pp->rec) {
      if (!rr || !rr->due(t)) continue;
      Recorder& r = *rr; std::lock_guard<std::mutex> l(r.m);
      if (r.cur >= 0) { r.q.emplace_back(r.cur, t); ++r.frames; if (r.first_step < 0) r.first_step = t; r.cv.notify_all(); }
      r.cur = -1; r.last_step = t;
    }
  };
  return h;
}

}  // namespace afdtd
