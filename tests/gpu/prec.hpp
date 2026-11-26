// SPDX-License-Identifier: MIT
// prec.hpp -- the GPU tests' view of alpha-fdtd's three precisions (alpha_fdtd.h section 1): element size, bit patterns of special
// values, rounding a float to the state's type and widening it back, the plan flag and the caller element type.  The roundings are
// written here independently of the library's.
#pragma once
#include <alpha_fdtd.h>
#include <cstdint>
#include <cstring>
#include <string>

struct Prec {
  std::string name;          // "bf16", "fp16", "fp32"
  int dtype;                 // AFDTD_BF16, AFDTD_F16, AFDTD_F32: the state's own element type
  unsigned flag;             // plan flag
  int bits;                  // 16 or 32
  uint32_t nosign, inf, ninf, nan, big, nbig;    // |x| mask; +inf, -inf, a quiet NaN, the largest finite, its negative
};
static inline Prec prec_of(const std::string& n)
{
  if (n == "fp16") return {n, AFDTD_F16, AFDTD_FP16, 16, 0x7FFFu, 0x7C00u, 0xFC00u, 0x7E00u, 0x7BFFu, 0xFBFFu};
  if (n == "fp32") return {n, AFDTD_F32, AFDTD_FP32, 32, 0x7FFFFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F7FFFFFu, 0xFF7FFFFFu};
  return {"bf16", AFDTD_BF16, 0u, 16, 0x7FFFu, 0x7F80u, 0xFF80u, 0x7FC0u, 0x7F7Fu, 0xFF7Fu};
}
static inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
// float -> the state's bit pattern, round to nearest even (finite inputs; the tests do not round NaN)
static inline uint32_t quantize(const Prec& p, float f)
{
  const uint32_t u = fbits(f);
  if (p.bits == 32) return u;
  if (p.dtype == AFDTD_BF16) return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
  // IEEE half by exact arithmetic on the significand: |f| = m 2^e with m < 2^24
  const uint32_t s = (u >> 16) & 0x8000u; const uint32_t a = u & 0x7FFFFFFFu;
  const int e = (int)(a >> 23); uint64_t m = (a & 0x7FFFFFu) | (e ? 0x800000u : 0u); const int ee = e ? e : 1;
  // value = m * 2^(ee - 150); in units of 2^-24 (the half subnormal step) at exponent ee: shift by (ee - 150 + 24)
  int sh = ee - 126;                                    // value / 2^-24 = m * 2^sh
  int hexp = 0; uint64_t q;
  // find the half exponent: normal halves have value in [2^-14, 65520); spacing 2^(E - 10) for value in [2^E, 2^(E+1))
  const int E = ee - 127;                               // unbiased float exponent (normal inputs)
  if (!e || E < -14) { // subnormal half: spacing 2^-24
    if (sh >= 0) q = m << sh; else { const int r = -sh; if (r >= 40) q = 0; else { const uint64_t lo = m & ((1ull << r) - 1), h = 1ull << (r - 1); q = m >> r; if (lo > h || (lo == h && (q & 1))) ++q; } }
    return s | (uint32_t)q;                             // q = 1024 is the smallest normal: the same bit pattern
  }
  if (E > 15) return s | 0x7C00u;
  // normal: significand of 11 bits: m has 24 bits, drop 13 with rounding
  const uint64_t lo = m & 0x1FFFu; q = m >> 13; if (lo > 0x1000u || (lo == 0x1000u && (q & 1))) ++q;
  hexp = E + 15;
  if (q == 0x800u) { q = 0x400u; ++hexp; }
  if (hexp >= 31) return s | 0x7C00u;
  return s | ((uint32_t)hexp << 10) | (uint32_t)(q & 0x3FFu);
}
static inline float widen(const Prec& p, uint32_t b)
{
  if (p.bits == 32) return bitsf(b);
  if (p.dtype == AFDTD_BF16) return bitsf(b << 16);
  const uint32_t s = (b & 0x8000u) << 16, e = (b >> 10) & 0x1Fu, m = b & 0x3FFu;
  if (e == 31) return bitsf(s | 0x7F800000u | (m << 13));
  float v = e ? (float)(1024 + m) * (float)(1.0 / 16777216.0) * (float)(1u << (e - 1)) : (float)m * (float)(1.0 / 16777216.0);
  return s ? -v : v;
}
