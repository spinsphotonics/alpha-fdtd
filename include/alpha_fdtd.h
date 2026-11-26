/* alpha_fdtd.h -- public C API of alpha-fdtd, a BF16 FDTD solver for large domains on 1 to 8 NVIDIA GPUs.
 *
 * SPDX-License-Identifier: MIT
 *
 * API v1 (FROZEN).  The functions, structs, constants and rules in this file do not change within API v1.  A later
 * API adds new functions; it never changes the meaning or layout of what is here.
 *
 * C99.  No CUDA types and no CUDA headers: a caller needs only this file and libalphafdtd.so.
 *
 * ======================================================================================================================
 * 1. WHAT IT COMPUTES
 * ======================================================================================================================
 *
 * One update, exactly: the lossy Yee scheme in BF16 with per-cell, per-axis coefficients Ca and Cb for E, in a closed
 * PEC box.  For (a, b, c) a cyclic permutation of (x, y, z), one step is
 *
 *     H half step (every cell):  H_a <- H_a - [ (E_c[+b] - E_c) - (E_b[+c] - E_b) ]
 *     E half step (every cell):  E_a <- fma( Cb_a , [ (H_c - H_c[-b]) - (H_b - H_b[-c]) ] , Ca_a * E_a )
 *
 * where F[+b] is the neighbour at index +1 along axis b and F[-b] the one at -1.  Every operation is one correctly
 * rounded BF16 operation (subtract, multiply, fused multiply-add), in exactly this order.  The H half step of a step
 * runs over the whole domain before its E half step.  Results are bit-identical to the reference kernel in the source
 * distribution's tests for every plan, every step count, every split of the steps into execute calls, in GPU memory
 * and in host memory, on 1 to 8 GPUs.  NaN and infinity follow the same arithmetic; nothing is clamped.
 *
 * THE H UPDATE IS THE ABSORBED FORM.  There is no H coefficient.  The textbook update
 *     H <- H - C_H curl E,        E <- Ca E + Cb curl H
 * is obtained by storing H' = H / C_H and Cb' = Cb * C_H (E and Ca unchanged) and reading back H = C_H * H'.
 * Every "Cb" and every "H" in this API is the absorbed one (Cb', H').  The two forms are equal in exact arithmetic.
 * When C_H is a power of two (2^-m) they are also bit-identical in BF16, because scaling by a power of two commutes
 * with rounding (except for results in the subnormal range or at overflow).  For other C_H the trajectory differs
 * from the textbook form by rounding.  Passing a textbook Cb is NOT an error alpha-fdtd can detect: in a dielectric it
 * gives a stable but wrong trajectory; in vacuum it violates the Courant limit (Cb' > 1/3), yet smooth fields can look
 * fine for thousands of steps before they grow to infinity and NaN.
 *
 * UNITS.  Lengths are in cells, time in steps.  With cell size 1, c = 1 in vacuum and mu = 1, C_H = dt, and for a
 * cell of relative permittivity eps and conductivity s:
 *     Ca  = (1 - s dt / (2 eps)) / (1 + s dt / (2 eps))
 *     Cb' = (dt * dt / eps)      / (1 + s dt / (2 eps))        (textbook Cb = dt / eps, times C_H = dt)
 * The 3D Courant condition dt <= 1/sqrt(3) becomes Cb' <= 1/3 in vacuum.  dt = 1/2 (exact, a power of two) gives
 * Cb' = 1/4 in vacuum.  Stability and the physical meaning of Ca and Cb are the caller's: they are not checked.
 *
 * ======================================================================================================================
 * 2. GRID, STAGGERING, TIME, BOUNDARY
 * ======================================================================================================================
 *
 * The domain has nx * ny * nz cells, (i, j, k) with 0 <= i < nx, 0 <= j < ny, 0 <= k < nz.  In units of the cell size,
 * the Yee positions of the values stored at cell (i, j, k) are
 *
 *     Ex (i+1/2, j,     k    )      Hx (i,     j+1/2, k+1/2)
 *     Ey (i,     j+1/2, k    )      Hy (i+1/2, j,     k+1/2)
 *     Ez (i,     j,     k+1/2)      Hz (i+1/2, j+1/2, k    )
 *
 * Ca_a and Cb_a sit with E_a.  A new plan holds Ca = Cb = +0, which sets E to +0 at every step: always set both
 * (Ca = 1 and Cb' = dt * dt / eps for lossless cells).  Time: a new plan holds E = +0 at time 0 and H' = +0 at time -1/2.  Each step advances
 * H' from t - 1/2 to t + 1/2, then E from t to t + 1.  So after n steps in all, E holds time n and H' holds time
 * n - 1/2 (physical H = C_H * H').
 *
 * The box [0, nx] x [0, ny] x [0, nz] is closed and its six faces are perfect electric conductors:
 *   - UPPER faces (x = nx, y = ny, z = nz): every neighbour outside the domain reads as +0.  Nothing to do.
 *   - LOWER faces (x = 0, y = 0, z = 0): the tangential E there are stored cells.  THE LOWER-WALL RULE: these values,
 *     and the coefficients that go with them, are +0 (bits 0x0000), always.
 *
 *         plane i = 0:  Ey Ez   Cay Caz   Cby Cbz
 *         plane j = 0:  Ex Ez   Cax Caz   Cbx Cbz
 *         plane k = 0:  Ex Ey   Cax Cay   Cbx Cby
 *
 *     Cells with i = 0, j = 0 or k = 0 hold the wall: put sources and initial fields at index >= 1.
 *     afdtd_set() writes +0 there (any other bit pattern, -0 and NaN included, is replaced) and counts what it
 *     replaced, separately for coefficients and for E (afdtd_info.walls_zeroed_coef, .walls_zeroed_field; the field
 *     count leaves out -0, so it counts only real field values that were removed).  Coefficients there are natural
 *     input (Ca = 1 everywhere touches them) and are always replaced silently.  With the plan flag
 *     AFDTD_STRICT_WALLS, a box whose E has a value other than +0 or -0 there is refused instead (AFDTD_EWALLS,
 *     nothing written).  The caller's arrays are never modified.  Normal components (Ex on i = 0, ...), every H value
 *     and every other cell are never changed.
 *
 * There is no periodic boundary, no PML, no source term and no other material model.
 *
 * ======================================================================================================================
 * 3. ARRAY LAYOUT
 * ======================================================================================================================
 *
 * Twelve components, in this order (AFDTD_EX ... AFDTD_CBZ):
 *     Ex Ey Ez Hx Hy Hz Cax Cay Caz Cbx Cby Cbz
 * An afdtd_box is a box of the domain, shape (sx, sy, sz) at offset (x0, y0, z0), and one afdtd_array per component.
 * Element (i, j, k) of the box is cell (x0 + i, y0 + j, z0 + k).  For component c:
 *   - array[c].data == NULL: the component is skipped (afdtd_set leaves it unchanged, afdtd_get does not write it).
 *   - box.flags without AFDTD_BOX_STRIDED: every array is dense C order of shape (sx, sy, sz), element (i, j, k) at
 *     data[(i * sy + j) * sz + k]; z is the fastest axis.  stride is ignored.
 *   - box.flags with AFDTD_BOX_STRIDED: element (i, j, k) is at data[i * stride[0] + j * stride[1] + k * stride[2]],
 *     strides in ELEMENTS.  A zero stride broadcasts along that axis (NumPy semantics; a single value with all three
 *     strides zero fills the whole box).  Zero strides are allowed for afdtd_set and refused by afdtd_get (the writes
 *     would overlap).  Negative strides are refused.
 * A NumPy array of shape (3, nx, ny, nz) holds E (or H, Ca, Cb) with component a at [a].
 *
 * Element type (afdtd_array.dtype), per component; 0 is not a type, so a forgotten dtype is refused:
 *   AFDTD_BF16  uint16_t bit patterns of bfloat16 (1 sign, 8 exponent, 7 mantissa bits; the top half of a float32).
 *               The state is stored in this format: values round-trip exactly.
 *   AFDTD_F32   float.  On set, rounded to BF16 to nearest, ties to even (NaN stays NaN, quiet; +-Inf stays).
 *               On get, widened exactly.
 *
 * ======================================================================================================================
 * 4. LIFE CYCLE (the FFTW pattern)
 * ======================================================================================================================
 *
 *     afdtd_import_wisdom_from_filename(path);            optional
 *     p = afdtd_plan_create(nx, ny, nz, &options);         options NULL: 1 GPU, GPU memory only, AFDTD_ESTIMATE
 *     afdtd_set(p, &box);                                 any number of boxes, any components
 *     afdtd_execute(p, steps);                            any number of calls, any step counts
 *     afdtd_get(p, &box);
 *     afdtd_export_wisdom_to_filename(path);              optional
 *     afdtd_plan_destroy(p);
 *
 * A plan IS the simulation: it owns the state E, H', Ca, Cb.  afdtd_plan_create allocates it (on the GPUs, or in
 * pinned host memory) and sets every value of every array to +0.  The planner decides where the domain lives: on the
 * GPUs when it fits in their free memory, otherwise in host memory up to options.host_bytes, streamed through the
 * GPUs.  The caller sees no difference except speed and memory use.
 *
 * ======================================================================================================================
 * 5. ERRORS
 * ======================================================================================================================
 *
 * Functions returning int return AFDTD_OK (0) or a negative AFDTD_E* code; afdtd_import_wisdom_* and
 * afdtd_plan_describe return a non-negative count on success instead of AFDTD_OK.  Functions returning a pointer
 * return NULL on failure.  Every function except afdtd_last_error, afdtd_version and afdtd_free clears the error text
 * on entry; after a failure, afdtd_last_error() gives a one-line reason (thread-local), and after a success it is "".
 * The library never prints, never exits, never aborts, and never lets a C++ exception escape.
 *
 * What is checked: sizes (>= 16 cells per axis; on n > 1 GPUs with the domain in GPU memory, ny > 24 * (2n - 1)),
 * overflow of sizes, options (rigor value, unknown flag bits, ngpus, device ordinals: exist, distinct, supported GPU,
 * peer access between them), memory (GPU free memory and the host budget, with the bytes needed in the message),
 * boxes (inside the domain, non-empty, at least one non-NULL array, a valid dtype for each, strides >= 0 and none zero
 * for get), step counts (>= 0), the lower-wall rule under AFDTD_STRICT_WALLS (section 2), wisdom syntax.
 * What is not checked: stability, the physical meaning of Ca and Cb, overlapping or aliased caller buffers.
 *
 * A refused call changes nothing: afdtd_set checks the whole box, every component, before it writes anything.  A CUDA
 * failure during a call (AFDTD_EDEVICE) leaves the plan unusable: every later call on it returns AFDTD_EDEVICE;
 * afdtd_plan_destroy is still required and safe.  A host-memory afdtd_execute whose GPU work stops making progress
 * for 120 s returns AFDTD_EDEVICE (with the stalled launch in the message) instead of hanging.  Using a plan after
 * afdtd_plan_destroy, or destroying it twice, is undefined behaviour.
 *
 * ======================================================================================================================
 * 6. THREADS AND CUDA
 * ======================================================================================================================
 *
 * - Different plans may be used from different threads at the same time.
 * - Calls on the same plan are serialised by the library (a second call waits for the first); afdtd_plan_destroy also
 *   waits for a call in flight on another thread before it frees the plan.
 * - Wisdom functions and the planner's use of wisdom are serialised by one library-wide lock.
 * - Planning with AFDTD_MEASURE or AFDTD_PATIENT times candidates on the plan's GPUs; other work on those GPUs at the same
 *   time distorts the timings (not the results).
 * - The library uses each device's primary CUDA context and restores the calling thread's current device on return.
 *   Other CUDA users in the same process (for example JAX or PyTorch) are fine, but memory they reserve is not available
 *   to alpha-fdtd: planning uses the free memory reported at afdtd_plan_create time, so a framework that pre-allocates
 *   most of the GPU (JAX does by default) silently moves the domain to host memory or makes the plan fail.
 * - All calls are synchronous: when a call returns, its GPU work is complete and the caller's buffers may be reused.
 * - Not fork-safe after the first call (as CUDA).
 *
 * Supported GPUs: compute capability 8.0 (A100), 9.0 (H100, H200), 10.0 (B200), 10.3 (B300).  Others are refused with
 * AFDTD_EUNSUPPORTED and a reason.  Requires an NVIDIA driver recent enough for the CUDA version the library was built
 * with; no CUDA toolkit is needed at run time.
 */
#ifndef ALPHA_FDTD_H
#define ALPHA_FDTD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AFDTD_API_VERSION   1
#define AFDTD_VERSION_MAJOR 0
#define AFDTD_VERSION_MINOR 1
#define AFDTD_VERSION_PATCH 0

/* ---- status codes ------------------------------------------------------------------------------------------------ */
enum {
  AFDTD_OK           =  0,
  AFDTD_EINVAL       = -1,  /* bad argument: size, options, box, dtype, step count, wisdom text; nothing changed    */
  AFDTD_ENOMEM       = -2,  /* the domain does not fit: the message gives the bytes needed in GPU memory and in
                               host memory, and the options.host_bytes that would work; no plan created            */
  AFDTD_ENOPLAN      = -3,  /* AFDTD_WISDOM_ONLY and no wisdom record for this problem; no plan created            */
  AFDTD_EUNSUPPORTED = -4,  /* no GPU, an unsupported GPU, or no peer access between the GPUs; no plan created     */
  AFDTD_EDEVICE      = -5,  /* a CUDA failure during a call: the plan is unusable (destroy it)                      */
  AFDTD_EWALLS       = -6,  /* AFDTD_STRICT_WALLS: E breaks the lower-wall rule; nothing was written              */
  AFDTD_EIO          = -7,  /* a wisdom file could not be read or written                                         */
  AFDTD_EINTERNAL    = -8   /* a self-check failed or an internal invariant broke: please report it               */
};

/* ---- planning rigor (afdtd_options.rigor; one value, not flags) ---------------------------------------------------
 *   AFDTD_ESTIMATE     pick by the built-in cost model; no timing.  Fast; may be slower at run time than MEASURE.
 *   AFDTD_MEASURE      time the usual candidates on the plan's GPUs and keep the fastest (seconds to minutes).
 *   AFDTD_PATIENT      time a wider set of candidates (minutes).
 *   AFDTD_WISDOM_ONLY  use a wisdom record or fail with AFDTD_ENOPLAN; never time, never estimate.
 * Any rigor uses a wisdom record of equal or higher rigor when one exists for the same problem, and every plan that was
 * timed is added to the in-memory wisdom.                                                                              */
enum { AFDTD_ESTIMATE = 0, AFDTD_MEASURE = 1, AFDTD_PATIENT = 2, AFDTD_WISDOM_ONLY = 3 };

/* ---- plan flags (afdtd_options.flags; OR-able).  Any other bit is refused.  ------------------------------------- */
#define AFDTD_FORCE_HOST    (1u << 8)   /* keep the domain in host memory even if it fits on the GPUs
                                           (leaves GPU memory free for other work; testing)                          */
#define AFDTD_STRICT_WALLS  (1u << 9)   /* afdtd_set refuses E that breaks the lower-wall rule (section 2)           */

/* ---- components and element types ------------------------------------------------------------------------------- */
enum {
  AFDTD_EX = 0, AFDTD_EY, AFDTD_EZ,
  AFDTD_HX, AFDTD_HY, AFDTD_HZ,
  AFDTD_CAX, AFDTD_CAY, AFDTD_CAZ,
  AFDTD_CBX, AFDTD_CBY, AFDTD_CBZ,
  AFDTD_NCOMP                         /* 12 */
};
enum { AFDTD_BF16 = 1, AFDTD_F32 = 2 };   /* 0 is not a type: a forgotten dtype is refused */

/* ---- boxes (section 3) ------------------------------------------------------------------------------------------ */
#define AFDTD_BOX_STRIDED   (1u << 0)   /* afdtd_box.flags: read each array's stride (0 = broadcast); else dense     */

typedef struct {
  void* data;               /* the caller's array for this component; NULL = skip.  afdtd_set never writes through it */
  int dtype;                /* AFDTD_BF16 or AFDTD_F32                                                               */
  int64_t stride[3];        /* element strides along x, y, z; read only with AFDTD_BOX_STRIDED                       */
} afdtd_array;

typedef struct {
  int64_t offset[3];        /* x0, y0, z0: first cell of the box in the domain                                       */
  int64_t shape[3];         /* sx, sy, sz: cells of the box along x, y, z (each >= 1)                                */
  unsigned flags;           /* 0 or AFDTD_BOX_STRIDED                                                                */
  afdtd_array array[AFDTD_NCOMP];   /* indexed by AFDTD_EX ... AFDTD_CBZ                                             */
} afdtd_box;

/* ---- plans -------------------------------------------------------------------------------------------------------
 * Plan options.  All-zero (or a NULL pointer) means: 1 GPU (device 0), GPU memory only, AFDTD_ESTIMATE, no flags.
 *   ngpus       GPUs to use, 1..8; 0 = 1.  On several GPUs the domain is cut along y.
 *   devices     CUDA device ordinals, ngpus of them, distinct; NULL = 0, 1, ..., ngpus - 1 (CUDA_VISIBLE_DEVICES
 *               applies).
 *   host_bytes  host memory the plan may allocate and pin for the domain when it does not fit on the GPUs; 0 = GPU
 *               memory only.  (Small staging buffers, at most 256 MiB, are always allowed and not counted.)
 *   rigor       AFDTD_ESTIMATE, AFDTD_MEASURE, AFDTD_PATIENT or AFDTD_WISDOM_ONLY.
 *   flags       0 or AFDTD_FORCE_HOST | AFDTD_STRICT_WALLS.                                                         */
typedef struct {
  int ngpus;
  const int* devices;
  size_t host_bytes;
  int rigor;
  unsigned flags;
} afdtd_options;

typedef struct afdtd_plan_s* afdtd_plan;

/* Plan an nx * ny * nz domain and allocate its state, all +0.  Returns NULL on failure (afdtd_last_error).  May take
 * minutes for MEASURE / PATIENT, and for the first touch and pinning of a large host domain.                          */
afdtd_plan afdtd_plan_create(int64_t nx, int64_t ny, int64_t nz, const afdtd_options* options);

/* Free the plan and its state, after any call in flight on another thread returns.  NULL is a no-op.                 */
void afdtd_plan_destroy(afdtd_plan p);

/* Memory an nx * ny * nz domain needs, without a GPU and without planning: *gpu_bytes, the free memory each of ngpus
 * GPUs needs for the domain in GPU memory (the state, the block pools at their smallest legal size, and the reserve the
 * library keeps free); *host_bytes, the options.host_bytes a host-memory plan needs.  Either pointer may be NULL.     */
int afdtd_bytes_needed(int64_t nx, int64_t ny, int64_t nz, int ngpus, size_t* gpu_bytes, size_t* host_bytes);

/* ---- fields ------------------------------------------------------------------------------------------------------
 * Copy the non-NULL arrays of `box` into the plan's state (lower-wall rule applied, section 2) or out of it.
 * One afdtd_set is atomic: it is refused as a whole or written as a whole.  Any box inside the domain, in either
 * memory mode; for a domain in host memory, boxes that span whole blocks of afdtd_info.io_block_y cells along y are
 * the fastest.  Synchronous.                                                                                          */
int afdtd_set(afdtd_plan p, const afdtd_box* box);
int afdtd_get(afdtd_plan p, const afdtd_box* box);

/* Advance the state by `steps` steps (>= 0; 0 is a no-op).  Bit-identical to `steps` steps of section 1 for every
 * step count and every split into calls.  SPEED: each call runs steps / 7 rounds of 7 steps on the fast path and
 * steps % 7 single steps on a much slower step-tail path (afdtd_info.tail_steps counts them).  Call with multiples
 * of 7, and between calls set sources every 7k steps where you can; execute(1) per step is the slowest way to run.     */
int afdtd_execute(afdtd_plan p, int64_t steps);

/* Six 64-bit checksums of Ex Ey Ez Hx Hy Hz (E and H only) over the whole domain, independent of the GPU count and
 * memory mode, equal to the reference checksum in the source distribution's tests:
 *     out[c] = sum over cells idx of mix64( ( ((c % 3) * n + idx) << 32 ) + bits_c(idx) )      (mod 2^64)
 * with n = nx * ny * nz, idx = (i * ny + j) * nz + k, bits_c the BF16 bit pattern of component c, and
 * mix64(x) = the SplitMix64 finaliser of x + 0x9E3779B97F4A7C15.  For comparing runs without copying the domain out.  */
int afdtd_checksums(afdtd_plan p, uint64_t out[6]);

/* ---- introspection ---------------------------------------------------------------------------------------------- */
typedef struct {
  int64_t nx, ny, nz;
  int ngpus;
  int devices[8];
  int on_host;              /* 1: the domain lives in pinned host memory and streams through the GPUs; 0: on the GPUs */
  int rigor;                /* rigor the schedule was chosen with (AFDTD_ESTIMATE, AFDTD_MEASURE or AFDTD_PATIENT)   */
  int from_wisdom;          /* 1: the schedule came from a wisdom record                                             */
  int64_t io_block_y;       /* boxes spanning whole blocks of this many y cells are the fastest in host memory       */
  size_t gpu_bytes_used;    /* device memory allocated on the busiest GPU                                            */
  size_t host_bytes_used;   /* host memory allocated and pinned for the domain                                       */
  int64_t steps_done;       /* steps executed since the plan was created                                             */
  int64_t tail_steps;       /* of those, steps run on the slow step-tail path (afdtd_execute)                        */
  uint64_t walls_zeroed_coef;   /* Ca, Cb values set to +0 by the last successful afdtd_set (section 2)              */
  uint64_t walls_zeroed_field;  /* E values other than +-0 set to +0 by the last successful afdtd_set (section 2)    */
  double predicted_rate;    /* cost model, cell updates per second (nx * ny * nz * steps / seconds); in host memory
                               for execute calls of at least 7 qmax steps (afdtd_plan_describe gives the count)      */
  double measured_rate;     /* timed while planning, or 0                                                            */
  double last_rate;         /* of the last afdtd_execute with steps > 0, or 0                                        */
} afdtd_info;

int afdtd_plan_info(afdtd_plan p, afdtd_info* out);

/* One line describing the plan, human-readable part first (memory mode, GPUs, rigor, rates, tail share), then the
 * schedule as in its wisdom record.  snprintf convention: writes at most n bytes including the NUL into buf (buf may be
 * NULL when n is 0) and returns the full length without the NUL, or < 0.                                              */
int afdtd_plan_describe(afdtd_plan p, char* buf, size_t n);

/* ---- wisdom (FFTW names) -----------------------------------------------------------------------------------------
 * Wisdom is text.  A record is one line: the problem key (library kernel-set id, GPU model, GPU count, nx ny nz,
 * memory mode) and the schedule chosen for it, with its rigor and measured rate.  Records from another kernel set or
 * another GPU model are kept but never used.  Import merges (a record of higher rigor replaces one of lower).          */
char* afdtd_export_wisdom_to_string(void);               /* free with afdtd_free; NULL on failure                    */
int afdtd_import_wisdom_from_string(const char* text);   /* number of records read (>= 0), or < 0; nothing is merged
                                                            from a text that has an error                            */
int afdtd_export_wisdom_to_filename(const char* path);   /* AFDTD_OK or < 0                                          */
int afdtd_import_wisdom_from_filename(const char* path); /* as afdtd_import_wisdom_from_string                       */
void afdtd_forget_wisdom(void);

/* ---- misc -------------------------------------------------------------------------------------------------------- */
void afdtd_free(void* ptr);                  /* frees memory returned by the library                               */
const char* afdtd_version(void);             /* "0.1.0"; static                                                    */
const char* afdtd_last_error(void);          /* reason for the last failure on this thread; "" after a success;
                                                never NULL                                                         */

#ifdef __cplusplus
}
#endif
#endif /* ALPHA_FDTD_H */
