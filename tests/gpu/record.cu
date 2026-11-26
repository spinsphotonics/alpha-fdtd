// SPDX-License-Identifier: MIT
// record.cu -- recorders (alpha_fdtd.h section 7) against the oracle tests/oracle/reference.cu, bit for bit.
//
//   afdtd_record_test NX NY NZ SPLITS EVERY BOX [BOX ...] [gpus=N] [file=PATH] [sched=KNOBS] [seed=S] [neg=late] [slots=K] [prec=P]
//
// BOX is x0,y0,z0,sx,sy,sz (one recorder each).  Random fields and random lossy per-cell Ca, Cb (as exact.cu).  Every frame of every
// recorder is compared with the oracle's state at its step over the whole box (all six components), the frame count with the
// expected one (steps 0, EVERY, 2 EVERY, ... up to the total), and the final state with the oracle's (recording must not change it).
//   file=PATH   the first recorder writes a file (read back and compared too); the others use callbacks
//   sched=K     force schedule knobs through wisdom (for example "tx=4 gate=0"): a MEASURE plan's record, edited, re-imported
//   neg=late    negative control: compare each frame with the oracle one step later; passes only if mismatches are found
// Prints one JSON line.  Exit 0 = passed, 1 = failed, 77 = skipped (fewer GPUs than requested).
#include <alpha_fdtd.h>
#include "reference.cu"
#include "prec.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"passed\": false, \"error\": \"CUDA %s line %d\"}\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
static bool on_wall(int c, int64_t x, int64_t y, int64_t z) { const int a = c % 3; return (x == 0 && a != 0) || (y == 0 && a != 1) || (z == 0 && a != 2); }

struct Rec { int64_t b[6]; int id = -1; afdtd_recorder_info info; std::map<int64_t, std::vector<uint16_t>> frames; std::mutex m; };
static int on_frame(void* user, int64_t step, const void* data, size_t bytes)
{
  Rec* r = (Rec*)user; std::lock_guard<std::mutex> l(r->m);
  if (r->frames.count(step)) return 1;                     // a step twice: an error
  r->frames[step].assign((const uint16_t*)data, (const uint16_t*)data + bytes / 2);
  return 0;
}
static size_t elem(const afdtd_recorder_info& L, int c, int64_t x, int64_t y, int64_t z)     // native (16-bit) frames: uint16 element
{
  return (size_t)(((((y / 24 - L.row0) * L.planes + (z - L.z0)) * L.pairs + (x / 2 - L.pair0)) * 24 + y % 24) * 12 + 2 * c + (x & 1));
}
static uint32_t frame_value(const afdtd_recorder_info& L, const std::vector<uint16_t>& fr, int c, int64_t x, int64_t y, int64_t z)
{
  if (L.dtype != AFDTD_F32) return fr[elem(L, c, x, y, z)];
  const size_t i = (((size_t)c * L.shape[0] + (x - L.offset[0])) * L.shape[1] + (y - L.offset[1])) * L.shape[2] + (z - L.offset[2]);    // canonical floats
  uint32_t v; memcpy(&v, reinterpret_cast<const char*>(fr.data()) + 4 * i, 4); return v;
}

template <class U, class T> int run(int argc, char** argv, const Prec& P);
int main(int argc, char** argv)
{
  std::string prec = "bf16";
  for (int i = 6; i < argc; ++i) if (!strncmp(argv[i], "prec=", 5)) prec = argv[i] + 5;
  const Prec P = prec_of(prec);
  if (P.bits == 32) return run<uint32_t, float>(argc, argv, P);
  if (P.dtype == AFDTD_F16) return run<uint16_t, __half>(argc, argv, P);
  return run<uint16_t, __nv_bfloat16>(argc, argv, P);
}
template <class U, class T> int run(int argc, char** argv, const Prec& P)
{
  if (argc < 7) { fprintf(stderr, "usage: afdtd_record_test NX NY NZ SPLITS EVERY BOX [BOX ...] [gpus=N] [file=PATH] [sched=K] [seed=S] [neg=late] [slots=K]\n"); return 2; }
  const int64_t nx = atoll(argv[1]), ny = atoll(argv[2]), nz = atoll(argv[3]), every = atoll(argv[5]);
  const std::string splits = argv[4];
  int ngpus = 1, slots = 0; uint64_t seed = 7; std::string file, sched, neg; std::vector<Rec*> recs;
  for (int i = 6; i < argc; ++i) {
    const std::string a = argv[i];
    if (!a.compare(0, 5, "gpus=")) ngpus = atoi(a.c_str() + 5);
    else if (!a.compare(0, 5, "file=")) file = a.substr(5);
    else if (!a.compare(0, 6, "sched=")) sched = a.substr(6);
    else if (!a.compare(0, 5, "seed=")) seed = strtoull(a.c_str() + 5, nullptr, 10);
    else if (!a.compare(0, 4, "neg=")) neg = a.substr(4);
    else if (!a.compare(0, 6, "slots=")) slots = atoi(a.c_str() + 6);
    else if (!a.compare(0, 5, "prec=")) continue;
    else { Rec* r = new Rec; if (sscanf(a.c_str(), "%lld,%lld,%lld,%lld,%lld,%lld", (long long*)&r->b[0], (long long*)&r->b[1], (long long*)&r->b[2],
                                         (long long*)&r->b[3], (long long*)&r->b[4], (long long*)&r->b[5]) != 6) { fprintf(stderr, "bad argument %s\n", argv[i]); return 2; }
           recs.push_back(r); }
  }
  std::string name = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + " steps " + splits + " every " + std::to_string(every) +
                     " boxes " + std::to_string(recs.size()) + " gpus " + std::to_string(ngpus) + (file.empty() ? "" : " file") + (sched.empty() ? "" : " sched " + sched) +
                     (neg.empty() ? "" : " neg=" + neg) + (P.name != "bf16" ? " " + P.name : "");
  int have = 0;
  if (cudaGetDeviceCount(&have) != cudaSuccess || have < ngpus) { printf("{\"test\": \"%s\", \"skipped\": \"%d GPU(s) visible\"}\n", name.c_str(), have); return 77; }
  std::vector<int64_t> sp; int64_t total = 0;
  for (size_t p = 0; p < splits.size();) { size_t q = splits.find(',', p); if (q == std::string::npos) q = splits.size(); sp.push_back(atoll(splits.substr(p, q - p).c_str())); total += sp.back(); p = q + 1; }

  const size_t n = (size_t)nx * ny * nz;
  std::vector<U> in(12 * n), ref, got(12 * n);
  std::mt19937_64 rng(seed);
  auto u01 = [&] { return (double)(rng() >> 11) / 9007199254740992.0; };
  for (int c = 0; c < 12; ++c) for (size_t i = 0; i < n; ++i)
    in[c * n + i] = (U)quantize(P, c < 6 ? (float)(2 * u01() - 1) : c < 9 ? (float)(0.6 + 0.4 * u01()) : (float)(0.3 * u01()));
  ref = in;
  for (int c = 0; c < 12; ++c) {
    if (c >= 3 && c < 6) continue;
    for (size_t i = 0; i < n; ++i) if (on_wall(c, (int64_t)(i / ((size_t)ny * nz)), (int64_t)(i / nz % ny), (int64_t)(i % nz))) ref[c * n + i] = 0;
  }

  afdtd_options o; memset(&o, 0, sizeof o); o.ngpus = ngpus; o.flags = P.flag;
  std::string err, used_sched;
  if (!sched.empty()) {                    // a MEASURE record with its knobs replaced, then a wisdom_only plan
    o.rigor = AFDTD_MEASURE;
    afdtd_plan m = afdtd_plan_create(nx, ny, nz, &o);
    if (!m) { printf("{\"test\": \"%s\", \"passed\": false, \"error\": \"measure plan: %s\"}\n", name.c_str(), afdtd_last_error()); return 1; }
    afdtd_plan_destroy(m);
    char* w = afdtd_export_wisdom_to_string(); std::string t = w ? w : ""; afdtd_free(w);
    for (size_t p = 0; p < sched.size();) {           // each "key=value" replaces that key's value in the sched text
      size_t q = sched.find(' ', p); if (q == std::string::npos) q = sched.size();
      const std::string kv = sched.substr(p, q - p); const size_t e = kv.find('=');
      const std::string key = " " + kv.substr(0, e + 1);
      const size_t at = t.rfind(key);
      if (at == std::string::npos) { printf("{\"test\": \"%s\", \"skipped\": \"no knob %s in %s\"}\n", name.c_str(), kv.c_str(), t.c_str()); return 77; }
      size_t end = t.find_first_of(" \"", at + key.size());
      t = t.substr(0, at + key.size()) + kv.substr(e + 1) + t.substr(end);
      p = q + 1;
    }
    afdtd_forget_wisdom();
    if (afdtd_import_wisdom_from_string(t.c_str()) < 1) { printf("{\"test\": \"%s\", \"passed\": false, \"error\": \"import: %s\"}\n", name.c_str(), afdtd_last_error()); return 1; }
    o.rigor = AFDTD_WISDOM_ONLY;
  }
  afdtd_plan p = afdtd_plan_create(nx, ny, nz, &o);
  if (!p) { printf("{\"test\": \"%s\", \"passed\": false, \"error\": \"%s\"}\n", name.c_str(), afdtd_last_error()); return 1; }
  afdtd_box b; memset(&b, 0, sizeof b);
  b.shape[0] = nx; b.shape[1] = ny; b.shape[2] = nz;
  for (int c = 0; c < 12; ++c) { b.array[c].data = in.data() + c * n; b.array[c].dtype = P.dtype; }
  if (afdtd_set(p, &b)) err = std::string("set: ") + afdtd_last_error();
  for (size_t r = 0; r < recs.size() && err.empty(); ++r) {
    afdtd_recorder_options ro; memset(&ro, 0, sizeof ro);
    for (int a = 0; a < 3; ++a) { ro.offset[a] = recs[r]->b[a]; ro.shape[a] = recs[r]->b[3 + a]; }
    ro.every = every; ro.slots = slots;
    if (r == 0 && !file.empty()) ro.path = file.c_str(); else { ro.callback = on_frame; ro.user = recs[r]; }
    recs[r]->id = afdtd_record(p, &ro);
    if (recs[r]->id < 0) err = std::string("record: ") + afdtd_last_error();
  }
  const double t1 = now();
  for (int64_t s : sp) if (err.empty() && afdtd_execute(p, s)) err = std::string("execute: ") + afdtd_last_error();
  const double trun = now() - t1;
  for (auto* r : recs) if (err.empty() && (afdtd_record_flush(p, r->id) || afdtd_record_info(p, r->id, &r->info))) err = std::string("flush/info: ") + afdtd_last_error();
  double stall = 0; for (auto* r : recs) stall += r->info.stall_s;
  for (auto* r : recs) if (err.empty() && afdtd_record_close(p, r->id)) err = std::string("close: ") + afdtd_last_error();
  for (int c = 0; c < 12; ++c) { b.array[c].data = got.data() + c * n; b.array[c].dtype = P.dtype; }
  if (err.empty() && afdtd_get(p, &b)) err = std::string("get: ") + afdtd_last_error();
  char desc[512] = ""; afdtd_plan_describe(p, desc, sizeof desc);
  for (char* q = desc; *q; ++q) if (*q == '"' || *q == '\\') *q = '\'';
  afdtd_plan_destroy(p);

  // the file of recorder 0: frames read back into its map
  if (err.empty() && !file.empty()) {
    FILE* f = fopen(file.c_str(), "rb");
    if (!f) err = "cannot read back the file";
    else {
      char h[4097] = {0}; if (fread(h, 1, 4096, f) != 4096) err = "short header";
      const char* q = strstr(h, "\"frames\": "); const long long nf = q ? atoll(q + 10) : -1;
      q = strstr(h, "\"first_step\": "); const long long fs = q ? atoll(q + 14) : -1;
      if (nf < 0 || fs < 0 || !strstr(h, "\"complete\": true")) err = "header fields";
      const afdtd_recorder_info& L = recs[0]->info;
      std::vector<uint16_t> buf(L.frame_stride / 2);
      for (long long k = 0; err.empty() && k < nf; ++k) {
        if (fread(buf.data(), 1, L.frame_stride, f) != L.frame_stride) { err = "short frame"; break; }
        recs[0]->frames[fs + k * every].assign(buf.begin(), buf.begin() + L.frame_bytes / 2);
      }
      fclose(f); remove(file.c_str());
    }
  }

  // oracle: step by step; at every multiple of `every` compare every recorder's frame over its box
  RefFieldsT<T> F; RefGrid g = {(int)nx, (int)ny, (int)nz};
  CK(cudaSetDevice(0));
  for (int c = 0; c < 12; ++c) { CK(cudaMalloc(&F.f[c], n * sizeof(U))); CK(cudaMemcpy(F.f[c], ref.data() + c * n, n * sizeof(U), cudaMemcpyHostToDevice)); }
  dim3 gr, bl; reference_dims(g, gr, bl);
  const int lag = neg == "late" ? 1 : 0;
  uint64_t mism = 0, compared = 0, missing = 0, extra = 0, frames_seen = 0;
  std::vector<U> st(6 * n);
  for (int64_t t = 0; t <= total + lag; ++t) {
    if (t > 0) { reference_step_h_t<T><<<gr, bl>>>(F, g); reference_step_e_t<T><<<gr, bl>>>(F, g); }
    const int64_t ft = t - lag;                                 // the frame step compared with the oracle at t
    if (ft < 0 || ft % every || ft > total) continue;
    CK(cudaDeviceSynchronize());
    for (int c = 0; c < 6; ++c) CK(cudaMemcpy(st.data() + c * n, F.f[c], n * sizeof(U), cudaMemcpyDeviceToHost));
    for (auto* r : recs) {
      auto it = r->frames.find(ft);
      if (it == r->frames.end()) { ++missing; continue; }
      ++frames_seen;
      const std::vector<uint16_t>& fr = it->second;
      for (int c = 0; c < 6; ++c)
        for (int64_t x = r->b[0]; x < r->b[0] + r->b[3]; ++x) for (int64_t y = r->b[1]; y < r->b[1] + r->b[4]; ++y) for (int64_t z = r->b[2]; z < r->b[2] + r->b[5]; ++z) {
          const uint32_t want = st[c * n + ((size_t)x * ny + y) * nz + z], have = frame_value(r->info, fr, c, x, y, z);
          ++compared; mism += want != have;
        }
    }
  }
  for (auto* r : recs) for (auto& kv : r->frames) if (kv.first % every || kv.first > total) ++extra;
  // the final state (one more step was taken for neg=late: recompute from scratch is not needed, compare only without the lag)
  uint64_t fmism = 0;
  if (!lag) { for (int c = 0; c < 6; ++c) CK(cudaMemcpy(st.data() + c * n, F.f[c], n * sizeof(U), cudaMemcpyDeviceToHost));
              for (int c = 0; c < 6; ++c) for (size_t i = 0; i < n; ++i) fmism += got[c * n + i] != st[c * n + i]; }
  for (int c = 0; c < 12; ++c) CK(cudaFree(F.f[c]));
  const int64_t expected = (int64_t)recs.size() * (total / every + 1);
  bool passed = err.empty() && !missing && !extra && frames_seen == (uint64_t)expected && !fmism;
  passed = lag ? (err.empty() && mism > 0) : (passed && !mism);
  for (char& q : err) if (q == '"' || q == '\\') q = '\'';
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("{\"test\": \"%s\", \"passed\": %s, \"frames\": %llu, \"expected_frames\": %lld, \"missing\": %llu, \"extra\": %llu, \"compared\": %llu, "
         "\"mismatches\": %llu, \"final_mismatches\": %llu, \"run_s\": %.3f, \"stall_s\": %.3f, \"gpu\": \"%s\", \"error\": \"%s\", \"describe\": \"%s\"}\n",
         name.c_str(), passed ? "true" : "false", (unsigned long long)frames_seen, (long long)expected, (unsigned long long)missing, (unsigned long long)extra,
         (unsigned long long)compared, (unsigned long long)mism, (unsigned long long)fmism, trun, stall, pr.name, err.c_str(), desc);
  for (auto* r : recs) delete r;
  return passed ? 0 : 1;
}
