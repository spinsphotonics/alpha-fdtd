# Performance

Rates are owned cell updates per second: nx * ny * nz * steps / wall seconds of the `execute` call. To measure them on
your own GPUs, build with `-DAFDTD_BUILD_TOOLS=ON` and run `afdtd_bench` (`tools/bench/bench.c`), which prints one JSON
line per run.

## The cost model

The model gives `afdtd_info.predicted_rate` and sizes MEASURE's timing runs. ESTIMATE picks a schedule by rules, not
by the model. The model is a fit to measured rates, not a measurement. Per GPU, the rate is the smaller of two limits:

- **SM limit:** SMs x clock / cycles per owned cell update per SM, fitted per kernel family.
- **Memory limit:** HBM bandwidth / bytes of HBM traffic per update.

Several GPUs scale the one-GPU rate by slightly less than the GPU count. Host memory scales it by a factor below one
for `execute` calls of at least `7 qmax` steps. Shorter calls are bound by the host link, which the model ignores.

## What sets the cost

- **State traffic.** The state is 24 bytes per cell (12 BF16 values). The fast path streams register pillars along z
  and advances them 7 steps per pass over the domain. That moves a fraction of the HBM traffic per update of a
  one-step-at-a-time kernel.
- **The tail.** `steps % 7` steps per call run on the one-step tail kernels. These are memory-bound and several times
  slower per step than the fast path. In host memory the tail also costs one extra pass over the host link.
- **Host link.** In host memory, each round of up to `7 qmax` steps moves the domain to the GPUs and back once. A call
  long enough to keep the ring full runs close to the GPU-memory rate; a shorter call is bound by the link.
- **Planning.** `measure` times each candidate for about one second, twice, so planning a large domain takes tens of
  seconds. A wisdom record makes later plans of the same problem instant.
- **Recording and sources.** A recorder's frames are copied by the copy engines inside the launches, so they cost no SM
  time, and a healthy host link barely slows the kernel. The steps wait only when a frame takes longer than about one
  pass to cross the host link, or when the sink is slower than the frames. A source keeps one SM out of the persistent
  grid while it is active. `docs/RECORDING.md` has the limits.
