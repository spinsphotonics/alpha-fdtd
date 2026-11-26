# Changelog

## 0.2.0 (2026-11-26)

API 2: everything of API v1 unchanged, plus a choice of precision and pulsed runs (`docs/RECORDING.md`, header sections 1,
7 and 8).

- Precision (`AFDTD_FP16`, `AFDTD_FP32`; Python `precision=`):
  - FP16 is the same engine and speed in IEEE half precision, about 8 times closer to float64 than BF16 in our tests,
    but with the range of half (below 65504).
  - FP32 is a new executor: dense float arrays, one kernel per half step, 1 to 8 GPUs cut along x, GPU memory only.
    It is about 10,000 times closer than BF16, and about an order of magnitude slower.
  - Every precision is bit-identical to the reference kernel in that precision. `afdtd_bytes_needed2`,
    `afdtd_plan_precision` and the `AFDTD_F16` element type (`numpy.float16`) are new.

- Recorders (`afdtd_record`, `Plan.record`): frames of a box (E and H) every `every` steps (a multiple of 7), copied by
  the GPUs' copy engines inside `execute` (no SM time, no extra pass), to a file or a callback, through pinned slots.
  `Recording` reads a file without a GPU.
- The DFT of a recording (`afdtd_dft`, `dft`): any frequencies, on one GPU, reading the file once, to an array or a
  `.npy` file; E and H at their own sample times.
- Injectors (`afdtd_inject`, `Plan.inject`): fixed fields added with per-boundary weights after every 7th step, by an
  exact rule. `soft_source` builds them for a source that acts at every step, exactly in exact arithmetic, on the fast path.
- `execute` with a recorder or injector runs its first `(7 - steps_done % 7) % 7` steps on the tail path so that its
  passes start at a multiple of 7.
- Plans that share a GPU take turns, call by call (before, closing one plan's recorder while another plan on the same
  GPU stepped with a recorder could hang both). Header section 6 gives the rule for other CUDA code in the process
  during steps with recorders or injectors.
- Docs: measured rates removed from README.md and docs/PERFORMANCE.md; `afdtd_bench` (`tools/bench`) times a plan
  on your own GPUs.
- Example: `examples/pulse.py`.

## 0.1.0 (2026-11-25)

First release. C library `libalphafdtd` (API v1, `include/alpha_fdtd.h`) and the Python package `alpha_fdtd`.

- One update, bit-identical to the reference kernel in `tests/oracle`: lossy Yee in BF16, per-cell per-axis Ca and Cb,
  absorbed H form, closed PEC box.
- 1 to 8 GPUs; the domain in GPU memory, or in host memory streamed through the GPUs.
- FFTW-style planning: `estimate`, `measure`, `patient`, `wisdom_only`; wisdom as text.
- Field set and get on any box, with per-component element type (float32 or BF16) and strides (zero strides broadcast).
- GPUs: A100, H100, H200, B200, B300 (compute capability 8.0, 9.0, 10.0, 10.3) in one library.

Fixes before the first release (review of the release candidate):

- Host memory: `execute` could deadlock for some step counts (a launch's blocks could occupy every SM while the launch
  they wait for had none resident). Each launch now starts only once the previous launch has a resident block, and a
  watchdog turns a stalled launch into `AFDTD_EDEVICE` instead of a hang.
- `afdtd_plan_destroy` waits for a call in flight on another thread; Python `Plan.close()` likewise.
- Python: integer arguments are range-checked (no silent truncation to 64 bits) and name the parameter when they are
  not integers; `get` with an offset outside the domain says so; `CoefficientWarning` when `execute` runs with Ca or Cb
  never set.
- GPU memory: the fixed 1.5 GiB reserve is now 256 MiB plus the staging buffer; long-z domains fall back to smaller
  block pools; `bytes_needed` includes the reserve and refuses sizes that overflow.
- Wisdom records with an illegal schedule are ignored instead of failing at `execute`.
- MEASURE times each candidate for about one second (was about 0.3 s).
- `CC`/`CXX` naming a missing compiler falls back to the default; the developer benchmark is off by default.
