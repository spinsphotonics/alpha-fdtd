# alpha-fdtd

A finite-difference time-domain (FDTD) solver for large 3-D domains on 1 to 8 NVIDIA GPUs. The domain lives in GPU
memory, or, when it does not fit, in host memory streamed through the GPUs. It does one thing: the lossy Yee update in
BF16 with per-cell, per-axis coefficients, in a closed PEC box. The result is bit-identical to a plain reference kernel
for every plan, step count, memory mode and GPU count.

It is a C library (`libalphafdtd.so`, header `include/alpha_fdtd.h`) with a thin Python wrapper (`alpha_fdtd`, NumPy in
and out). Each Python method is one C call.

**A plan is the simulation.** You create a plan for a domain size. It owns the state (E, H, Ca, Cb, all +0 at the
start). You `set` arrays into it, `execute` steps, and `get` arrays out. The planner chooses how to run (kernel,
schedule, GPU or host memory) the way FFTW does: by a cost model (`estimate`), by timing candidates (`measure`,
`patient`), or from saved wisdom.

## Install

```
pip install .                    # builds the library with CMake (CUDA toolkit 12.9 or later, CMake 3.24 or later)
```

Tested with CUDA toolkit 13.3.1 and GCC 13 on Ubuntu 24.04. Some standalone Python builds export `CC=clang`; if `CC` or
`CXX` names a compiler that is not installed, the build warns and uses the default compiler. To choose one, name it:
`CC=gcc CXX=g++ pip install .`.

Building needs the CUDA toolkit but no GPU. Running needs a supported GPU and an NVIDIA driver recent enough for the
toolkit used to build, but no toolkit. Linux x86_64 only. The only Python dependency is NumPy. There is no CPU fallback:
`import alpha_fdtd` works anywhere, and `alpha_fdtd.Plan(...)` raises `Unsupported` without a supported GPU.

For C, build and install the library with CMake:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cmake --install build --prefix <prefix>
cc -std=c99 examples/quickstart.c -I<prefix>/include -L<prefix>/lib -Wl,-rpath,<prefix>/lib -lalphafdtd -lm
```

Without `-Wl,-rpath,<prefix>/lib`, set `LD_LIBRARY_PATH=<prefix>/lib` when you run the program. On some systems the
directory is `lib64`.

## Quick start

```python
import numpy as np, alpha_fdtd as af

n, dt = 128, 0.5                                     # 128^3 cells; C_H = dt, a power of two: exact
x = np.arange(n, dtype=np.float32) - n / 2
E = np.zeros((3, n, n, n), np.float32)
E[2] = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 32)[:, :, None]   # Ez pulse, uniform in z: div E = 0
eps = np.ones((n, n), np.float32)
eps[40:88, 40:88] = 4.0                              # a dielectric rod along z
Ca = np.broadcast_to(np.float32(1), (3, n, n, n))    # lossless; a broadcast costs one value
Cb = np.broadcast_to((dt * dt / eps)[None, :, :, None], (3, n, n, n))  # absorbed form: Cb = dt * dt / eps
with af.Plan((n, n, n)) as plan:                     # rigor="estimate": no timing
    plan.set(E=E, Ca=Ca, Cb=Cb)
    plan.execute(700)                                # multiples of 7 run on the fast path
    Ez, H = plan.get("Ez"), dt * plan.get("H")       # E at step 700, physical H at 700 - 1/2
    print(float(np.abs(Ez).max()), plan.describe())
```

Always set both `Ca` and `Cb`. A new plan holds `Ca = Cb = 0`, which zeroes E at every step. Python warns with
`CoefficientWarning` on the first `execute` if either was never set. To get Ca and Cb from permittivity and
conductivity, copy `yee_coefficients` from `examples/coefficients.py`.

More in `examples/`:

- `coefficients.py`: Ca and Cb from permittivity, conductivity and dt, and a graded absorber.
- `from_fdtdz.py`: an fdtdz-style run with a source plane and output steps.
- `offload.py`: a domain in host memory.
- `multi_gpu.py`: two GPUs, bit-identical to one.
- `wisdom.py`: plan once with `measure`, reuse the result.
- `quickstart.c`: the C API.

## Coming from fdtdz

[fdtdz](https://github.com/spinsphotonics/fdtdz) is one function call. You pass permittivity, a source, a PML and output
steps, and it returns E at those steps. alpha-fdtd instead keeps a state that you step and read. The main differences:

- `Cb = dt * dt / eps` (the absorbed form, not `dt / eps`), and `Ca = 1` for lossless cells. Set both.
- There is no source term: set fields between `execute` calls, every 7 steps or a multiple of 7.
- There is no PML: use a graded-loss layer through Ca and Cb.
- The box is PEC on all six faces, and index 0 is the wall.
- The state is BF16. Fields go in and out as float32.

`docs/FROM_FDTDZ.md` has the full mapping and a worked example. That mapping has not yet been cross-checked by running
both codes on the same problem; see Limitations.

## The update and conventions

The normative statement is `include/alpha_fdtd.h`, sections 1 to 3. In short:

**Units.** Lengths are in cells and time is in steps. With cell size 1, c = 1 in vacuum and mu = 1, the H coefficient is
`C_H = dt`. For a cell of relative permittivity `eps` and conductivity `s`:

```
Ca = (1 - s dt / (2 eps)) / (1 + s dt / (2 eps))
Cb = (dt * dt / eps)      / (1 + s dt / (2 eps))        (the textbook dt / eps, times C_H = dt)
```

The 3-D Courant limit `dt <= 1/sqrt(3)` is `Cb <= 1/3` in vacuum. `dt = 0.5` gives `Cb = 0.25` in vacuum. Stability and
the physical meaning of Ca and Cb are yours; the library does not check them.

**The absorbed H form.** There is no H coefficient. The update is `H <- H - curl E`, then `E <- Ca E + Cb curl H`. The
textbook form maps onto it by storing `H' = H / dt` and `Cb' = Cb * dt`:

| you have | you pass | you read back |
|---|---|---|
| textbook `Cb = dt / eps` | `Cb = dt * dt / eps` | `get("Cb")` is `dt * dt / eps` |
| physical `H` | `H = H / dt` | physical `H = dt * plan.get("H")` |

With `dt` a power of two this is bit-identical to the textbook form. The library cannot detect a textbook `Cb` passed
by mistake. In a dielectric it gives a stable but wrong result. In vacuum it violates the Courant limit (`Cb > 1/3`),
but smooth fields may look fine for thousands of steps before they grow to inf and NaN.

**Time.** A new plan holds E at time 0 and H at time -1/2. One step advances H, then E. After `n` steps in all, `get`
returns E at time `n` and H at time `n - 1/2`.

**The lower wall: index 0 is the wall.** The box is closed and all six faces are perfect electric conductors. The upper
faces need nothing. On the lower faces the tangential E is a stored cell, and it and its coefficients are always +0:

```
plane i = 0:  Ey Ez  (and Cay Caz Cby Cbz)        k
plane j = 0:  Ex Ez  (and Cax Caz Cbx Cbz)        ^   PEC     . . . .
plane k = 0:  Ex Ey  (and Cax Cay Cbx Cby)        |   wall    . . . .   cells 1, 2, ... : your fields
                                                  |   +0 -->  0 . . .
                                                  +-----------------> i
```

Put sources and initial fields at index 1 or more. `set` writes +0 on the wall:

- for Ca and Cb, silently, because Ca = 1 everywhere is normal input;
- for E, with a `WallWarning`, because a value there is almost always a misplaced source;
- with `strict_walls=True`, it raises `WallError` and writes nothing.

Normal components (Ex on i = 0, and so on) and H are never changed.

**Arrays.** E, H, Ca and Cb are NumPy arrays of shape `(3, sx, sy, sz)`: the x, y, z component first, z the fastest
axis. Single components (`Ex` ... `Cbz`) are `(sx, sy, sz)`. Accepted element types:

- `float32`, rounded to BF16 on `set` (to nearest, ties to even);
- `uint16`, BF16 bit patterns;
- `ml_dtypes.bfloat16`.

Anything else raises `TypeError`; for `float64`, use `.astype(np.float32)`. Views, strides and broadcasts are passed
without copies, so `np.broadcast_to(np.float32(1), (3, nx, ny, nz))` costs one value. `set` without `offset=` must
cover the whole domain. `set(offset=..., Ez=block)` writes a sub-box, and `get` takes any box.

**Multiples of 7.** Each `execute(steps)` runs `steps // 7` rounds of 7 steps on the fast path and `steps % 7` single
steps on a much slower tail path; `plan.info.tail_steps` counts them. Call `execute` with multiples of 7, and change
fields between calls every 7k steps. `execute(1)` per step is the slowest way to run. The result is the same for any
split of the steps.

## Planner and wisdom

`rigor` picks how the schedule is chosen:

- `"estimate"` (the default) picks by rules and a cost model, without timing.
- `"measure"` times 4 to 12 candidates on your GPUs, twice each, for about one second of work per timing. On a large
  domain that takes tens of seconds.
- `"patient"` times more candidates, three times each.
- `"wisdom_only"` never plans: it uses a saved record or raises `NoPlan`.

In host memory, every rigor uses the same rule-based ring, and nothing is timed.

Every plan made with `measure` or `patient` is added to the in-memory wisdom. Save it, and later plans of the same problem are instant:

```python
import os, alpha_fdtd as af

if os.path.exists("my.wisdom"):
    af.import_wisdom_file("my.wisdom")
with af.Plan((256, 256, 256), rigor="measure") as plan:   # times candidates once; instant with a record
    print(plan.info.from_wisdom, plan.describe())
af.export_wisdom_file("my.wisdom")
```

Records are kept per kernel set, GPU model, GPU count, size and memory mode. A record from another build or another GPU
model is kept on export but never used. Wisdom is plain text, one line per record; the format is in
`docs/ARCHITECTURE.md`. `estimate` and a wisdom hit take well under a second, plus allocation.

## Memory and host-memory offload

`af.bytes_needed(shape, ngpus)` returns `(gpu_bytes, host_bytes)`, and works without a GPU:

- `gpu_bytes` is the free memory each GPU needs for the domain in GPU memory;
- `host_bytes` is the `host_bytes=` a host-memory plan needs.

For large cubes that is about 32 bytes per cell on the GPU (24 of state, the rest block pools and a small reserve) and
about 29 bytes per cell in host memory. Thin or short axes cost more per cell. x is padded to 16-cell groups, z by about
22 planes, and the block pools hold whole z columns. A domain with nx or nz below about 256 can need several times more,
so check `bytes_needed` rather than multiplying by 32. Your own float32 arrays come on top: 12 bytes per cell for each of
E, H, Ca and Cb that you build in full. Broadcast coefficients cost nothing.

```python
import alpha_fdtd as af

for n in (512, 1024, 2048):
    g, h = af.bytes_needed((n, n, n))
    print(f"{n}^3: {g / 2**30:7.1f} GiB on one GPU, or host_bytes >= {h / 2**30:7.1f} GiB")
```

A plan uses GPU memory when the domain fits in the GPUs' free memory. Otherwise it uses host memory, up to
`host_bytes=`; the default, 0, means GPU memory only. `force_host=True` keeps the domain in host memory even when it
fits. `plan.info.on_host` tells you which mode was chosen. Results are bit-identical in both modes.

In host memory, each `execute` call streams the whole domain over the host link once per round of up to `7 qmax` steps
(`plan.describe()` gives `qmax`). Calls that long run close to the GPU-memory rate; shorter calls are bound by the host
link. Setting a whole host-memory domain from NumPy is slow, so set coefficients once and set fields in boxes.

## Several GPUs

Use `Plan(shape, ngpus=4)`, or `devices=[...]` to name the CUDA devices. The GPUs must be one model with peer access
between them. The domain is cut along y. In GPU memory, each GPU needs at least two 24-cell rows, so
`ny > 24 (2n - 1)` for n GPUs. Host memory works on several GPUs too. Results are bit-identical to one GPU.

**Other GPU users in the process.** alpha-fdtd plans with the GPU memory that is free when the plan is created. JAX
reserves most of the GPU by default, which silently moves the domain to host memory or makes the plan fail. Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX starts.

**Long runs and Ctrl-C.** A call cannot be interrupted until it returns. Run long simulations as a loop of shorter
`execute` calls; that loop is also where progress reports go.

## Supported GPUs

Supported: compute capability 8.0 (A100), 9.0 (H100, H200), 10.0 (B200) and 10.3 (B300), in one library. Other GPUs are
refused with `Unsupported`. To time a plan on your own GPUs, build with `-DAFDTD_BUILD_TOOLS=ON` and run `afdtd_bench`
(`tools/bench/bench.c`).

## Limitations

- One update only: lossy Yee in BF16 (8 bits of precision) with per-cell Ca and Cb, the absorbed H form, a closed PEC
  box. No PML, no source term, no dispersion, no periodic boundary, no magnetic materials.
- At least 16 cells per axis. On several GPUs in GPU memory, `ny > 24 (2n - 1)`.
- `steps % 7` of each call run on a slow tail path: use multiples of 7.
- Linux x86_64 and NVIDIA GPUs of compute capability 8.0, 9.0, 10.0 and 10.3 only; no CPU fallback.
- A call cannot be interrupted with Ctrl-C until it returns.
- Thin or long-z domains need much more memory per cell (see Memory and host-memory offload).
- The fdtdz mapping in `docs/FROM_FDTDZ.md` has not been cross-checked by running both codes.

## How it was verified

The oracle is `tests/oracle/reference.cu`, a plain one-step-at-a-time kernel of the formula in `alpha_fdtd.h`. A second,
independent oracle is `tests/oracle/bf16ref.py`, a NumPy model of the BF16 arithmetic. The GPU tests compare every value
of all 12 arrays against both, bit for bit. Each test uses random fields, random lossy per-cell Ca and Cb, and random
data on the walls (the oracle runs on the wall-zeroed data).

The tests cover:

- step counts from 1 to 701, and splits such as 3+3+1+6 and 13 x 1;
- GPU and host memory, on 1 and 2 GPUs;
- `estimate`, `measure` and `patient` plans;
- non-finite data;
- strided, transposed and broadcast arrays, and sub-boxes;
- the wall rule and its counts, error paths, and wisdom round trips.

Two negative controls check the harness itself: one flipped bit must give exactly one mismatch, and an oracle run one
step longer must mismatch.

Before release, the full suite passed with 0 mismatches on H100 x1, A100-SXM4-80GB x1 and H100 x2. An earlier release
candidate passed on B300 x1 as well. Large cases were checked separately against the reference checksums over the whole
domain: 1338 x 1344 x 1336 on H100 and B300, 1380^3 on A100, and 2746 x 2880 x 2600 on 8 x H100. The 8-GPU, A100 and
B300 checks ran on the release candidate.

To run the tests yourself:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAFDTD_BUILD_TESTS=ON     # needs the CUDA toolkit, no GPU
cmake --build build -j
ctest --test-dir build -L cpu                                             # no GPU
ctest --test-dir build -L gpu                                             # needs a supported GPU; gpu2 tests need 2
pip install .[test] && pytest tests/python                                # GPU tests are marked gpu; skipped without one
```

`kernels_generated` (part of `-L cpu`) needs Python 3, and is left out when CMake finds none. Each ctest prints one JSON
line. Tests that need more GPUs than are visible are reported as skipped, never as passed.

## API reference

The normative reference is `include/alpha_fdtd.h`. It covers the update, the array layout, the lower-wall rule, errors
and threads. `docs/API.md` lists every C function with its Python counterpart. In brief:

| Python | C |
|---|---|
| `Plan(shape, rigor=, ngpus=, devices=, host_bytes=, force_host=, strict_walls=)`, `Plan.close()`, `with` | `afdtd_plan_create`, `afdtd_plan_destroy` |
| `Plan.set(offset=, E=, H=, Ca=, Cb=, Ex=, ...)`, `Plan.get(*names, offset=, shape=, dtype=, out=)` | `afdtd_set`, `afdtd_get` |
| `Plan.execute(steps)`, `Plan.checksums()` | `afdtd_execute`, `afdtd_checksums` |
| `Plan.info`, `Plan.describe()`, `Plan.shape` | `afdtd_plan_info`, `afdtd_plan_describe` |
| `import_wisdom`, `import_wisdom_file`, `export_wisdom`, `export_wisdom_file`, `forget_wisdom` | `afdtd_*_wisdom_*` |
| `bytes_needed(shape, ngpus)`, `version()` | `afdtd_bytes_needed`, `afdtd_version` |
| `InvalidArgument`, `OutOfMemory`, `NoPlan`, `Unsupported`, `DeviceError`, `WallError`, `WisdomIOError`, `InternalError` (all `Error`) | `AFDTD_E*` codes, `afdtd_last_error` |
| `WallWarning`, `CoefficientWarning` | (Python only) |

`ALPHA_FDTD_LIBRARY=/path/to/libalphafdtd.so` makes `import alpha_fdtd` load that library instead of the one inside the
package, for development builds. It is the only environment variable alpha-fdtd reads.

Design notes are in `docs/ARCHITECTURE.md`, and what sets the speed is in `docs/PERFORMANCE.md`.

## Contributing

See `CONTRIBUTING.md`. In short: every change keeps the state bit-identical to `tests/oracle/reference.cu`, and the
kernel texts in `src/kernels/` change only through `tools/gen`.

## Acknowledgements

alpha-fdtd is developed by SPINS Photonics Inc. It was inspired by [fdtdz](https://github.com/spinsphotonics/fdtdz), SPINS Photonics'
systolic FDTD solver for one GPU called from JAX. `docs/FROM_FDTDZ.md` maps its inputs onto alpha-fdtd.

## License

MIT; see `LICENSE`. Copyright (c) 2026 SPINS Photonics Inc.
