// SPDX-License-Identifier: MIT
// reference.cu -- the correctness oracle of the tests: the update of alpha_fdtd.h section 1, written as plainly as possible.
// One thread per cell, global memory, one array per component, every operation one correctly rounded BF16 operation in the
// header's order.  One step = reference_step_h over every cell, then reference_step_e over every cell.
//
// Layout: 12 arrays F[0..11] = Ex Ey Ez Hx Hy Hz Cax Cay Caz Cbx Cby Cbz, each nx*ny*nz elements, element (i,j,k) at
// ((i*ny)+j)*nz+k.  Closed box: neighbours outside the domain read as +0; the lower PEC walls are in the data (tangential E,
// Ca, Cb = +0 on the planes i = 0, j = 0, k = 0, set by the caller).
#ifndef AFDTD_REFERENCE_CU
#define AFDTD_REFERENCE_CU
#include <cuda_bf16.h>

// The oracle in each precision of alpha-fdtd (alpha_fdtd.h section 1): BF16, FP16 (__half) and FP32 (float), every operation one
// correctly rounded operation of that type.  reference_step_h / _e and RT are the BF16 instances.
#include <cuda_fp16.h>
template <class T> struct RefOps;
template <> struct RefOps<__nv_bfloat16> {
  static __device__ __forceinline__ __nv_bfloat16 sub(__nv_bfloat16 a, __nv_bfloat16 b) { return __hsub(a, b); }
  static __device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
  static __device__ __forceinline__ __nv_bfloat16 fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) { return __hfma(a, b, c); }
  static __device__ __forceinline__ __nv_bfloat16 zero() { unsigned short z = 0; return *reinterpret_cast<__nv_bfloat16*>(&z); }
};
template <> struct RefOps<__half> {
  static __device__ __forceinline__ __half sub(__half a, __half b) { return __hsub(a, b); }
  static __device__ __forceinline__ __half mul(__half a, __half b) { return __hmul(a, b); }
  static __device__ __forceinline__ __half fma(__half a, __half b, __half c) { return __hfma(a, b, c); }
  static __device__ __forceinline__ __half zero() { unsigned short z = 0; return *reinterpret_cast<__half*>(&z); }
};
template <> struct RefOps<float> {
  static __device__ __forceinline__ float sub(float a, float b) { return __fsub_rn(a, b); }
  static __device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
  static __device__ __forceinline__ float fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
  static __device__ __forceinline__ float zero() { return 0.f; }
};
struct RefGrid { int nx, ny, nz; };
template <class T> struct RefFieldsT { T* f[12]; };
enum { R_EX, R_EY, R_EZ, R_HX, R_HY, R_HZ, R_CAX, R_CAY, R_CAZ, R_CBX, R_CBY, R_CBZ };

template <class T> __device__ __forceinline__ T r_at(const T* p, int i, int j, int k, const RefGrid& g)
{
  if (i < 0 || i >= g.nx || j < 0 || j >= g.ny || k < 0 || k >= g.nz) return RefOps<T>::zero();
  return p[((size_t)i * g.ny + j) * g.nz + k];
}

// thread x -> k, thread y -> j, grid z -> i; launch with reference_dims().
#define REF_CELL \
  const int k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z; \
  if (i >= g.nx || j >= g.ny || k >= g.nz) return; \
  const size_t idx = ((size_t)i * g.ny + j) * g.nz + k;

static inline void reference_dims(const RefGrid& g, dim3& grid, dim3& block)
{
  block = dim3(256, 1, 1);
  grid = dim3((g.nz + 255) / 256, g.ny, g.nx);
}

// H half step, absorbed form:  H_a <- H_a - [ (E_c[+b] - E_c) - (E_b[+c] - E_b) ]
template <class T> __global__ void reference_step_h_t(RefFieldsT<T> F, RefGrid g)
{
  REF_CELL
  using O = RefOps<T>;
  const T ex = F.f[R_EX][idx], ey = F.f[R_EY][idx], ez = F.f[R_EZ][idx];
  const T cx = O::sub(O::sub(r_at(F.f[R_EZ], i, j + 1, k, g), ez), O::sub(r_at(F.f[R_EY], i, j, k + 1, g), ey));
  const T cy = O::sub(O::sub(r_at(F.f[R_EX], i, j, k + 1, g), ex), O::sub(r_at(F.f[R_EZ], i + 1, j, k, g), ez));
  const T cz = O::sub(O::sub(r_at(F.f[R_EY], i + 1, j, k, g), ey), O::sub(r_at(F.f[R_EX], i, j + 1, k, g), ex));
  F.f[R_HX][idx] = O::sub(F.f[R_HX][idx], cx);
  F.f[R_HY][idx] = O::sub(F.f[R_HY][idx], cy);
  F.f[R_HZ][idx] = O::sub(F.f[R_HZ][idx], cz);
}

// E half step:  E_a <- fma( Cb_a , [ (H_c - H_c[-b]) - (H_b - H_b[-c]) ] , Ca_a * E_a )
template <class T> __global__ void reference_step_e_t(RefFieldsT<T> F, RefGrid g)
{
  REF_CELL
  using O = RefOps<T>;
  const T hx = F.f[R_HX][idx], hy = F.f[R_HY][idx], hz = F.f[R_HZ][idx];
  const T cx = O::sub(O::sub(hz, r_at(F.f[R_HZ], i, j - 1, k, g)), O::sub(hy, r_at(F.f[R_HY], i, j, k - 1, g)));
  const T cy = O::sub(O::sub(hx, r_at(F.f[R_HX], i, j, k - 1, g)), O::sub(hz, r_at(F.f[R_HZ], i - 1, j, k, g)));
  const T cz = O::sub(O::sub(hy, r_at(F.f[R_HY], i - 1, j, k, g)), O::sub(hx, r_at(F.f[R_HX], i, j - 1, k, g)));
  F.f[R_EX][idx] = O::fma(F.f[R_CBX][idx], cx, O::mul(F.f[R_CAX][idx], F.f[R_EX][idx]));
  F.f[R_EY][idx] = O::fma(F.f[R_CBY][idx], cy, O::mul(F.f[R_CAY][idx], F.f[R_EY][idx]));
  F.f[R_EZ][idx] = O::fma(F.f[R_CBZ][idx], cz, O::mul(F.f[R_CAZ][idx], F.f[R_EZ][idx]));
}

// the BF16 names
typedef __nv_bfloat16 RT;
typedef RefFieldsT<RT> RefFields;
#define reference_step_h reference_step_h_t<__nv_bfloat16>
#define reference_step_e reference_step_e_t<__nv_bfloat16>
#endif
