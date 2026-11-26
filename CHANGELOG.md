# Changelog

## Unreleased

- Docs: measured rates removed from README.md and docs/PERFORMANCE.md; `afdtd_bench` (`tools/bench`) times a plan
  on your own GPUs.

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
