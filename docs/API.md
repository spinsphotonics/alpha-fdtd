# API reference (v1)

This page is written from `include/alpha_fdtd.h`, which is normative; where they differ, the header wins. API v1 is
frozen: functions, structs and constants keep their meaning and layout, and a later API only adds functions.

Each C function has one Python counterpart in `alpha_fdtd`, and each Python method makes one C call (`Plan.set` also
reads `afdtd_plan_info` once, to raise `WallWarning`).

## Conventions

- **Status.** Functions that return `int` return `AFDTD_OK` (0) or a negative `AFDTD_E*` code. The exceptions are
  `afdtd_import_wisdom_*` and `afdtd_plan_describe`, which return a non-negative count on success. Functions that
  return a pointer return NULL on failure.
- **Error text.** Every function except `afdtd_last_error`, `afdtd_version` and `afdtd_free` clears the thread-local
  error text on entry. After a failure, `afdtd_last_error()` gives a one-line reason that starts with the code name
  (`AFDTD_ENOMEM: ...`); after a success it is `""`.
- **No side effects on failure.** A refused call changes nothing. The library never prints, exits or aborts, and no C++
  exception crosses the API.
- **Python.** A failed call raises the `Error` subclass for its code (table below), and the message is the error text.

| code | meaning | Python exception |
|---|---|---|
| `AFDTD_EINVAL` (-1) | bad argument: size, options, box, dtype, step count, wisdom text | `InvalidArgument` (also a `ValueError`) |
| `AFDTD_ENOMEM` (-2) | the domain does not fit; the message gives the bytes needed in GPU and host memory and the `host_bytes` that would work | `OutOfMemory` (also a `MemoryError`) |
| `AFDTD_ENOPLAN` (-3) | `AFDTD_WISDOM_ONLY` and no usable wisdom record | `NoPlan` |
| `AFDTD_EUNSUPPORTED` (-4) | no GPU, an unsupported GPU, or no peer access between the GPUs | `Unsupported` |
| `AFDTD_EDEVICE` (-5) | a CUDA failure, or a host-memory launch that made no progress for 120 s; the plan is unusable, destroy it | `DeviceError` |
| `AFDTD_EWALLS` (-6) | `AFDTD_STRICT_WALLS` and E breaks the lower-wall rule; nothing written | `WallError` (also a `ValueError`) |
| `AFDTD_EIO` (-7) | a wisdom file could not be read or written | `WisdomIOError` (also an `OSError`) |
| `AFDTD_EINTERNAL` (-8) | a self-check failed or an internal invariant broke; please report it | `InternalError` |

Python also raises `TypeError` for an unsupported element type or a non-integer argument, and `ValueError` for shapes
that disagree, negative strides, or a closed plan. These are refused before any C call.

## Plans

### `afdtd_plan_create`

```c
typedef struct { int ngpus; const int* devices; size_t host_bytes; int rigor; unsigned flags; } afdtd_options;
afdtd_plan afdtd_plan_create(int64_t nx, int64_t ny, int64_t nz, const afdtd_options* options);
```

```
Plan(shape, *, rigor="estimate", ngpus=None, devices=None, host_bytes=0, force_host=False, strict_walls=False)
```

This plans an `nx * ny * nz` domain and allocates its state, with every value +0. A NULL or all-zero `options` means
1 GPU (device 0), GPU memory only, `AFDTD_ESTIMATE` and no flags. On failure it returns NULL.

| field | Python | meaning |
|---|---|---|
| `ngpus` | `ngpus=` | 1 to 8; 0 means 1. The domain is cut along y. |
| `devices` | `devices=` | `ngpus` distinct CUDA ordinals, of one GPU model with peer access between them; NULL means `0 .. ngpus-1`. |
| `host_bytes` | `host_bytes=` | host memory the plan may pin when the domain does not fit on the GPUs; 0 means GPU memory only. |
| `rigor` | `rigor=` | `AFDTD_ESTIMATE` (0), `AFDTD_MEASURE` (1), `AFDTD_PATIENT` (2), `AFDTD_WISDOM_ONLY` (3); in Python `"estimate"`, `"measure"`, `"patient"`, `"wisdom_only"`. |
| `flags` | `force_host=`, `strict_walls=` | `AFDTD_FORCE_HOST` keeps the domain in host memory; `AFDTD_STRICT_WALLS` makes `set` refuse E on the lower wall. Other bits are refused. |

Requirements and timing:

- Each axis needs at least 16 cells.
- On n > 1 GPUs in GPU memory, `ny > 24 (2n - 1)`.
- `MEASURE` and `PATIENT` take seconds to minutes, and so does the first touch of a large host domain.
- Any rigor uses a wisdom record of equal or higher rigor for the same problem.

The Python `Plan` is a context manager. It is also closed when it is garbage-collected.

### `afdtd_plan_destroy`

```c
void afdtd_plan_destroy(afdtd_plan p);
```

`Plan.close()`, or leaving the `with` block, calls this. It frees the plan once any call in flight on another thread
has returned. NULL is a no-op. Using a plan after destroying it is undefined in C. In Python, `close()` can be called
more than once, and any later method raises `ValueError("plan is closed")`.

### `afdtd_bytes_needed`

```c
int afdtd_bytes_needed(int64_t nx, int64_t ny, int64_t nz, int ngpus, size_t* gpu_bytes, size_t* host_bytes);
```

```
bytes_needed(shape, ngpus=1) -> (gpu_bytes, host_bytes)
```

This works without a GPU and without planning. It returns two numbers:

- `gpu_bytes`: the free memory each GPU needs for the domain in GPU memory. That covers the state, the block pools at
  their smallest legal size, and the library's reserve.
- `host_bytes`: the `host_bytes` a host-memory plan needs.

Either pointer may be NULL.

## Fields

### `afdtd_set`, `afdtd_get`

```c
typedef struct { void* data; int dtype; int64_t stride[3]; } afdtd_array;
typedef struct { int64_t offset[3]; int64_t shape[3]; unsigned flags; afdtd_array array[AFDTD_NCOMP]; } afdtd_box;
int afdtd_set(afdtd_plan p, const afdtd_box* box);
int afdtd_get(afdtd_plan p, const afdtd_box* box);
```

```
Plan.set(*, offset=None, E=None, H=None, Ca=None, Cb=None, Ex=None, ..., Cbz=None)
Plan.get(*names, offset=(0, 0, 0), shape=None, dtype=np.float32, out=None)
```

These copy the non-NULL arrays of a box into the state, or out of it.

**The box.** A box has a shape `(sx, sy, sz)` at an offset `(x0, y0, z0)`, and one array slot per component, indexed
`AFDTD_EX ... AFDTD_CBZ`. The order is Ex Ey Ez Hx Hy Hz Cax Cay Caz Cbx Cby Cbz. A slot whose `data` is NULL is
skipped.

**Element types.** Each component has its own:

- `AFDTD_BF16` (1) is `uint16_t` BF16 bit patterns, the stored format.
- `AFDTD_F32` (2) is `float`. It is rounded to nearest even on set and widened exactly on get.
- 0 is refused.

**Layout.**

- Without `AFDTD_BOX_STRIDED`, arrays are dense C order: element `(i, j, k)` is at `(i * sy + j) * sz + k`.
- With it, element `(i, j, k)` is at `i*stride[0] + j*stride[1] + k*stride[2]`, with strides in elements.
- A zero stride broadcasts on `set` and is refused on `get`. Negative strides are refused.

**Atomicity and walls.**

- `afdtd_set` is atomic: it checks the whole box, every component, before it writes anything.
- It applies the lower-wall rule (header section 2): tangential E, Ca and Cb on the planes `i = 0`, `j = 0` and
  `k = 0` become +0.
- It records how many values it replaced in `walls_zeroed_coef` and `walls_zeroed_field`.
- The caller's arrays are never modified.

**Python, `set`.**

- `E`, `H`, `Ca` and `Cb` are arrays of shape `(3, sx, sy, sz)`; single components such as `Ez=` are `(sx, sy, sz)`.
- One call is one atomic `afdtd_set`, and all its arrays cover the same box.
- Without `offset=`, every array must cover the whole domain.
- Arrays go in by pointer and element strides, with no copies, so `np.broadcast_to` becomes zero strides.
- Accepted types are `float32`, `uint16` (bit patterns) and `ml_dtypes.bfloat16`.
- E on the lower wall raises `WallWarning`, or `WallError` under `strict_walls=True`.

**Python, `get`.**

- It returns one array for one name, or a tuple in the order of `names`.
- `shape=None` reads to the end of the domain. It is refused above `GET_IMPLICIT_LIMIT` (2^31) values, so read large
  domains in pieces.
- `out=` writes into your arrays instead of new ones.

**Speed.** For a domain in host memory, boxes that span whole blocks of `info.io_block_y` y cells are the fastest.

### `afdtd_execute`

```c
int afdtd_execute(afdtd_plan p, int64_t steps);
```

```
Plan.execute(steps)
```

This advances the state by `steps` steps; `steps` must be at least 0, and 0 is a no-op. The result is bit-identical
for every step count and every split into calls. `steps // 7` rounds run on the fast path, and `steps % 7` single steps
run on the slow tail path, so use multiples of 7.

The call is synchronous and cannot be interrupted. A host-memory execute that stops making progress for 120 s returns
`AFDTD_EDEVICE` instead of hanging. Python warns once with `CoefficientWarning` if Ca or Cb was never set.

### `afdtd_checksums`

```c
int afdtd_checksums(afdtd_plan p, uint64_t out[6]);
```

```
Plan.checksums() -> (6 ints)
```

These are six 64-bit checksums of Ex Ey Ez Hx Hy Hz over the whole domain. They do not depend on the GPU count or the
memory mode, so they compare runs without copying the domain out. The header gives the exact definition, a sum of
SplitMix64-mixed (component, index, bits) values.

## Introspection

### `afdtd_plan_info`

```c
int afdtd_plan_info(afdtd_plan p, afdtd_info* out);
```

```
Plan.info -> Info
```

| field | meaning |
|---|---|
| `nx, ny, nz` (Python `shape`), `ngpus`, `devices` | the problem and the GPUs used |
| `on_host` | 1 if the domain lives in pinned host memory and streams through the GPUs |
| `rigor`, `from_wisdom` | how the schedule was chosen |
| `io_block_y` | y extent of the fastest boxes in host memory |
| `gpu_bytes_used`, `host_bytes_used` | device memory on the busiest GPU; host memory pinned for the domain |
| `steps_done`, `tail_steps` | steps since creation; of those, steps on the slow tail path |
| `walls_zeroed_coef`, `walls_zeroed_field` | values set to +0 by the last successful `set` (the field count leaves out -0) |
| `predicted_rate`, `measured_rate`, `last_rate` | cell updates per second: the cost model, the planner's timing (or 0), the last `execute` (or 0) |

### `afdtd_plan_describe`

```c
int afdtd_plan_describe(afdtd_plan p, char* buf, size_t n);
```

```
Plan.describe() -> str
```

This returns one line. It starts with a human-readable part: memory mode, GPUs, rigor, rates and tail share. The
schedule follows, as it appears in the wisdom record; the schedule text is opaque. The C function follows `snprintf`:
it writes at most `n` bytes including the NUL, and returns the full length.

## Wisdom

```c
char* afdtd_export_wisdom_to_string(void);               /* free with afdtd_free */
int   afdtd_import_wisdom_from_string(const char* text); /* records read, or < 0 */
int   afdtd_export_wisdom_to_filename(const char* path);
int   afdtd_import_wisdom_from_filename(const char* path);
void  afdtd_forget_wisdom(void);
```

```
export_wisdom() -> str;  import_wisdom(text) -> int
export_wisdom_file(path);  import_wisdom_file(path) -> int
forget_wisdom()
```

Wisdom is text, one record per line. Each record holds a key (kernel-set id, GPU model, GPU count, size, memory mode)
and the schedule chosen for it, with its rigor and measured rate.

- Import merges: a record of higher rigor replaces one of lower rigor.
- A text that contains an error merges nothing.
- Records from another kernel set or GPU model are kept but never used.
- A record whose schedule is illegal for the problem is ignored.

The wisdom functions are serialised by one library-wide lock.

## Miscellaneous

| C | Python | |
|---|---|---|
| `void afdtd_free(void*)` | (internal) | frees memory returned by the library |
| `const char* afdtd_version(void)` | `version()`, `__version__` | `"0.1.0"`; the Python import checks that the two agree |
| `const char* afdtd_last_error(void)` | the exception message | this thread's reason for the last failure; `""` after a success; never NULL |
| `AFDTD_API_VERSION`, `AFDTD_VERSION_*` | | 1; 0.1.0 |

## Threads and CUDA

- Different plans may be used from different threads at the same time.
- Calls on the same plan are serialised.
- Every call is synchronous: when it returns, its GPU work is complete and your buffers may be reused. Python releases
  the GIL during each call.
- The library uses each device's primary context and restores the calling thread's current device.
- Other CUDA users in the process are fine, but planning uses only the memory that is free at `afdtd_plan_create`. If
  JAX runs in the same process, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- The library is not fork-safe after the first call, as with CUDA.
