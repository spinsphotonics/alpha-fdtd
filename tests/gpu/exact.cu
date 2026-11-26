// SPDX-License-Identifier: MIT
// exact.cu -- alpha-fdtd through its public C API against the oracle tests/oracle/reference.cu, bit for bit.
//
//   afdtd_exact NX NY NZ SPLITS [gpus=N] [host] [rigor=R] [seed=S] [nonfinite] [neg=flip|steps] [prec=bf16|fp16|fp32]
//
// Random fields and random lossy per-cell Ca, Cb, with random values on the lower walls too (the oracle runs on the
// wall-zeroed data).  E goes in as F32 (exact for BF16 values), H and the coefficients as BF16 bit patterns, then a
// strided sub-box of Hy is set again with the same values.  SPLITS is the list of execute calls, for example 3,3,1,6.
// Compared: every value of the 6 fields, the 6 coefficient arrays read back (= the wall-zeroed input), the checksums,
// the wall counts, steps_done and tail_steps.
//   host        the domain in host memory (AFDTD_FORCE_HOST)
//   nonfinite   also put infinities, NaN and near-overflow values into H, and infinities into E on and off the walls
//   neg=flip    negative control: flip one bit of the result; passes only if exactly one mismatch is found
//   neg=steps   negative control: run the oracle one step longer; passes only if mismatches are found
//   prec=P      the plan's precision (alpha_fdtd.h section 1) and the oracle's
// Prints one JSON line.  Exit 0 = passed, 1 = failed, 77 = skipped (fewer GPUs than requested).
#include <alpha_fdtd.h>
#include "reference.cu"
#include "prec.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"passed\": false, \"error\": \"CUDA %s line %d\"}\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
static uint64_t mix64(uint64_t x) { x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31); }
static bool on_wall(int c, int64_t x, int64_t y, int64_t z) { const int a = c % 3; return (x == 0 && a != 0) || (y == 0 && a != 1) || (z == 0 && a != 2); }

template <class U, class T> int run(int64_t nx, int64_t ny, int64_t nz, const std::string& splits, int ngpus, int host, int rigor, uint64_t seed, int nonfinite,
                                     const std::string& neg, const Prec& P);
int main(int argc, char** argv)
{
  if (argc < 5) { fprintf(stderr, "usage: afdtd_exact NX NY NZ SPLITS [gpus=N] [host] [rigor=R] [seed=S] [nonfinite] [neg=flip|steps] [prec=P]\n"); return 2; }
  const int64_t nx = atoll(argv[1]), ny = atoll(argv[2]), nz = atoll(argv[3]);
  const std::string splits = argv[4];
  int ngpus = 1, host = 0, rigor = AFDTD_ESTIMATE, nonfinite = 0; uint64_t seed = 1; std::string neg, prec = "bf16";
  for (int i = 5; i < argc; ++i) {
    const std::string a = argv[i];
    if (!a.compare(0, 5, "gpus=")) ngpus = atoi(a.c_str() + 5);
    else if (a == "host") host = 1;
    else if (!a.compare(0, 6, "rigor=")) rigor = atoi(a.c_str() + 6);
    else if (!a.compare(0, 5, "seed=")) seed = strtoull(a.c_str() + 5, nullptr, 10);
    else if (a == "nonfinite") nonfinite = 1;
    else if (!a.compare(0, 4, "neg=")) neg = a.substr(4);
    else if (!a.compare(0, 5, "prec=")) prec = a.substr(5);
    else { fprintf(stderr, "afdtd_exact: unknown argument %s\n", argv[i]); return 2; }
  }
  const Prec P = prec_of(prec);
  if (P.bits == 32) return run<uint32_t, float>(nx, ny, nz, splits, ngpus, host, rigor, seed, nonfinite, neg, P);
  if (P.dtype == AFDTD_F16) return run<uint16_t, __half>(nx, ny, nz, splits, ngpus, host, rigor, seed, nonfinite, neg, P);
  return run<uint16_t, __nv_bfloat16>(nx, ny, nz, splits, ngpus, host, rigor, seed, nonfinite, neg, P);
}
template <class U, class T> int run(int64_t nx, int64_t ny, int64_t nz, const std::string& splits, int ngpus, int host, int rigor, uint64_t seed, int nonfinite,
                                     const std::string& neg, const Prec& P)
{
  std::string name = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + " steps " + splits + " gpus " + std::to_string(ngpus) +
                     (host ? " host" : "") + (rigor ? " rigor " + std::to_string(rigor) : "") + (nonfinite ? " nonfinite" : "") + (neg.empty() ? "" : " neg=" + neg) +
                     (P.name != "bf16" ? " " + P.name : "");
  int have = 0;
  if (cudaGetDeviceCount(&have) != cudaSuccess || have < ngpus) { printf("{\"test\": \"%s\", \"skipped\": \"%d GPU(s) visible\"}\n", name.c_str(), have); return 77; }
  std::vector<int64_t> sp; int64_t total = 0, tail = 0;
  for (size_t p = 0; p < splits.size();) { size_t q = splits.find(',', p); if (q == std::string::npos) q = splits.size(); sp.push_back(atoll(splits.substr(p, q - p).c_str())); total += sp.back(); tail += sp.back() % 7; p = q + 1; }

  // ---- input, and the oracle's wall-zeroed copy
  const size_t n = (size_t)nx * ny * nz;
  std::vector<U> in(12 * n), ref, got(12 * n);
  std::mt19937_64 rng(seed);
  auto u01 = [&] { return (double)(rng() >> 11) / 9007199254740992.0; };
  for (int c = 0; c < 12; ++c) for (size_t i = 0; i < n; ++i)
    in[c * n + i] = (U)quantize(P, c < 6 ? (float)(2 * u01() - 1) : c < 9 ? (float)(0.6 + 0.4 * u01()) : (float)(0.3 * u01()));
  if (nonfinite) {
    const U hv[5] = {(U)P.inf, (U)P.ninf, (U)P.nan, (U)P.big, (U)P.nbig};
    for (int t = 0; t < 64; ++t) in[(3 + t % 3) * n + rng() % n] = hv[t % 5];                 // H: inf, NaN, near overflow
    for (int t = 0; t < 16; ++t) in[(t % 3) * n + rng() % n] = t & 1 ? (U)P.inf : (U)P.ninf;   // E: inf anywhere
    in[0 * n + (size_t)(rng() % ny) * nz + rng() % nz] = (U)P.inf;                             // Ex on the plane i = 0 (normal: kept)
    in[1 * n + (size_t)(rng() % ny) * nz + rng() % nz] = (U)P.ninf;                            // Ey on the plane i = 0 (tangential: zeroed)
  }
  ref = in;
  uint64_t wall_coef = 0, wall_field = 0;
  for (int c = 0; c < 12; ++c) {
    if (c >= 3 && c < 6) continue;
    for (size_t i = 0; i < n; ++i)
      if (on_wall(c, (int64_t)(i / ((size_t)ny * nz)), (int64_t)(i / nz % ny), (int64_t)(i % nz))) {
        if (c >= 6) wall_coef += ref[c * n + i] != 0; else wall_field += (ref[c * n + i] & P.nosign) != 0;
        ref[c * n + i] = 0;
      }
  }

  // ---- alpha-fdtd
  afdtd_options o; memset(&o, 0, sizeof o);
  o.ngpus = ngpus; o.rigor = rigor; o.flags = P.flag;
  if (host) { o.flags |= AFDTD_FORCE_HOST; afdtd_bytes_needed2(nx, ny, nz, ngpus, P.flag, nullptr, &o.host_bytes); }
  const double t0 = now();
  afdtd_plan p = afdtd_plan_create(nx, ny, nz, &o);
  if (!p) { printf("{\"test\": \"%s\", \"passed\": false, \"error\": \"%s\"}\n", name.c_str(), afdtd_last_error()); return 1; }
  const double tplan = now() - t0;
  std::string err;
  afdtd_box b; memset(&b, 0, sizeof b);
  b.shape[0] = nx; b.shape[1] = ny; b.shape[2] = nz;
  std::vector<float> ef(3 * n);                       // E goes in as float (exact for the state's values), the rest in the state's own type
  for (size_t i = 0; i < 3 * n; ++i) ef[i] = widen(P, in[i]);
  for (int c = 0; c < 12; ++c) { b.array[c].data = c < 3 ? (void*)(ef.data() + c * n) : (void*)(in.data() + c * n); b.array[c].dtype = c < 3 ? AFDTD_F32 : P.dtype; }
  if (afdtd_set(p, &b)) err = std::string("set: ") + afdtd_last_error();
  afdtd_info inf; afdtd_plan_info(p, &inf);
  const uint64_t wzc = inf.walls_zeroed_coef, wzf = inf.walls_zeroed_field;
  {
    afdtd_box s; memset(&s, 0, sizeof s);
    s.offset[0] = nx / 3; s.offset[1] = ny / 4; s.offset[2] = 1; s.shape[0] = nx / 2; s.shape[1] = ny / 2; s.shape[2] = nz - 2; s.flags = AFDTD_BOX_STRIDED;
    s.array[AFDTD_HY].data = in.data() + 4 * n + ((size_t)s.offset[0] * ny + s.offset[1]) * nz + s.offset[2];
    s.array[AFDTD_HY].dtype = P.dtype;
    s.array[AFDTD_HY].stride[0] = ny * nz; s.array[AFDTD_HY].stride[1] = nz; s.array[AFDTD_HY].stride[2] = 1;
    if (err.empty() && afdtd_set(p, &s)) err = std::string("set sub-box: ") + afdtd_last_error();
  }
  const double t1 = now();
  for (int64_t s : sp) if (err.empty() && afdtd_execute(p, s)) err = std::string("execute: ") + afdtd_last_error();
  const double trun = now() - t1;
  for (int c = 0; c < 12; ++c) { b.array[c].data = got.data() + c * n; b.array[c].dtype = P.dtype; }
  if (err.empty() && afdtd_get(p, &b)) err = std::string("get: ") + afdtd_last_error();
  uint64_t cs[6] = {0};
  if (err.empty() && afdtd_checksums(p, cs)) err = std::string("checksums: ") + afdtd_last_error();
  afdtd_plan_info(p, &inf);
  char desc[512]; afdtd_plan_describe(p, desc, sizeof desc);
  for (char* q = desc; *q; ++q) if (*q == '"' || *q == '\\') *q = '\'';
  afdtd_plan_destroy(p);

  // ---- oracle
  const int64_t ref_steps = total + (neg == "steps" ? 1 : 0);
  RefFieldsT<T> F; RefGrid g = {(int)nx, (int)ny, (int)nz};
  CK(cudaSetDevice(0));
  for (int c = 0; c < 12; ++c) { CK(cudaMalloc(&F.f[c], n * sizeof(U))); CK(cudaMemcpy(F.f[c], ref.data() + c * n, n * sizeof(U), cudaMemcpyHostToDevice)); }
  dim3 gr, bl; reference_dims(g, gr, bl);
  for (int64_t t = 0; t < ref_steps; ++t) { reference_step_h_t<T><<<gr, bl>>>(F, g); reference_step_e_t<T><<<gr, bl>>>(F, g); }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  for (int c = 0; c < 6; ++c) CK(cudaMemcpy(ref.data() + c * n, F.f[c], n * sizeof(U), cudaMemcpyDeviceToHost));
  for (int c = 0; c < 12; ++c) CK(cudaFree(F.f[c]));
  if (neg == "flip") got[(size_t)(rng() % 6) * n + rng() % n] ^= 1;

  // ---- compare
  uint64_t mism = 0, cmism = 0, nonfin = 0, rcs[6] = {0};
  for (int c = 0; c < 12; ++c) for (size_t i = 0; i < n; ++i) {
    if (got[c * n + i] != ref[c * n + i]) ++(c < 6 ? mism : cmism);
    if (c < 6) { rcs[c] += mix64((((uint64_t)(c % 3) * n + i) << 32) + ref[c * n + i]); nonfin += ((uint32_t)ref[c * n + i] & P.inf) == P.inf; }
  }
  bool cs_equal = true;
  for (int c = 0; c < 6; ++c) cs_equal &= cs[c] == rcs[c];
  const bool counts = wzc == wall_coef && wzf == wall_field && inf.steps_done == total && inf.tail_steps == (P.bits == 32 ? 0 : tail);   // FP32: no tail path
  bool passed = err.empty() && counts && !cmism;
  if (neg == "flip") passed = passed && mism == 1;
  else if (neg == "steps") passed = passed && mism > 0;
  else passed = passed && !mism && cs_equal;
  for (char& q : err) if (q == '"' || q == '\\') q = '\'';
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("{\"test\": \"%s\", \"passed\": %s, \"values\": %zu, \"mismatches\": %llu, \"coef_mismatches\": %llu, \"checksums_equal\": %s, "
         "\"walls_coef\": [%llu, %llu], \"walls_field\": [%llu, %llu], \"tail_steps\": %lld, \"nonfinite\": %llu, \"plan_s\": %.2f, \"run_s\": %.3f, "
         "\"gpu\": \"%s\", \"error\": \"%s\", \"describe\": \"%s\"}\n",
         name.c_str(), passed ? "true" : "false", 12 * n, (unsigned long long)mism, (unsigned long long)cmism, cs_equal ? "true" : "false",
         (unsigned long long)wzc, (unsigned long long)wall_coef, (unsigned long long)wzf, (unsigned long long)wall_field, (long long)inf.tail_steps,
         (unsigned long long)nonfin, tplan, trun, pr.name, err.c_str(), desc);
  return passed ? 0 : 1;
}
