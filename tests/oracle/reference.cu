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

typedef __nv_bfloat16 RT;
struct RefGrid { int nx, ny, nz; };
struct RefFields { RT* f[12]; };
enum { R_EX, R_EY, R_EZ, R_HX, R_HY, R_HZ, R_CAX, R_CAY, R_CAZ, R_CBX, R_CBY, R_CBZ };

__device__ __forceinline__ RT r_sub(RT a, RT b) { return __hsub(a, b); }
__device__ __forceinline__ RT r_mul(RT a, RT b) { return __hmul(a, b); }
__device__ __forceinline__ RT r_fma(RT a, RT b, RT c) { return __hfma(a, b, c); }
__device__ __forceinline__ RT r_zero() { unsigned short z = 0; return *reinterpret_cast<RT*>(&z); }

__device__ __forceinline__ RT r_at(const RT* p, int i, int j, int k, const RefGrid& g)
{
  if (i < 0 || i >= g.nx || j < 0 || j >= g.ny || k < 0 || k >= g.nz) return r_zero();
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
__global__ void reference_step_h(RefFields F, RefGrid g)
{
  REF_CELL
  const RT ex = F.f[R_EX][idx], ey = F.f[R_EY][idx], ez = F.f[R_EZ][idx];
  const RT cx = r_sub(r_sub(r_at(F.f[R_EZ], i, j + 1, k, g), ez), r_sub(r_at(F.f[R_EY], i, j, k + 1, g), ey));
  const RT cy = r_sub(r_sub(r_at(F.f[R_EX], i, j, k + 1, g), ex), r_sub(r_at(F.f[R_EZ], i + 1, j, k, g), ez));
  const RT cz = r_sub(r_sub(r_at(F.f[R_EY], i + 1, j, k, g), ey), r_sub(r_at(F.f[R_EX], i, j + 1, k, g), ex));
  F.f[R_HX][idx] = r_sub(F.f[R_HX][idx], cx);
  F.f[R_HY][idx] = r_sub(F.f[R_HY][idx], cy);
  F.f[R_HZ][idx] = r_sub(F.f[R_HZ][idx], cz);
}

// E half step:  E_a <- fma( Cb_a , [ (H_c - H_c[-b]) - (H_b - H_b[-c]) ] , Ca_a * E_a )
__global__ void reference_step_e(RefFields F, RefGrid g)
{
  REF_CELL
  const RT hx = F.f[R_HX][idx], hy = F.f[R_HY][idx], hz = F.f[R_HZ][idx];
  const RT cx = r_sub(r_sub(hz, r_at(F.f[R_HZ], i, j - 1, k, g)), r_sub(hy, r_at(F.f[R_HY], i, j, k - 1, g)));
  const RT cy = r_sub(r_sub(hx, r_at(F.f[R_HX], i, j, k - 1, g)), r_sub(hz, r_at(F.f[R_HZ], i - 1, j, k, g)));
  const RT cz = r_sub(r_sub(hy, r_at(F.f[R_HY], i - 1, j, k, g)), r_sub(hx, r_at(F.f[R_HX], i, j - 1, k, g)));
  F.f[R_EX][idx] = r_fma(F.f[R_CBX][idx], cx, r_mul(F.f[R_CAX][idx], F.f[R_EX][idx]));
  F.f[R_EY][idx] = r_fma(F.f[R_CBY][idx], cy, r_mul(F.f[R_CAY][idx], F.f[R_EY][idx]));
  F.f[R_EZ][idx] = r_fma(F.f[R_CBZ][idx], cz, r_mul(F.f[R_CAZ][idx], F.f[R_EZ][idx]));
}
#endif
