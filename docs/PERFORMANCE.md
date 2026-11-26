# Performance

Apart from the model in the first section, measured numbers only; each row gives its date and GPU. Rates are owned
cell updates per second (nx * ny * nz * steps / wall seconds of the `execute` call), CUDA 13.3.1,
`rigor="measure"` unless stated.

## Performance model in brief

The model gives `afdtd_info.predicted_rate` and sizes MEASURE's timing runs (ESTIMATE picks a schedule by rules, not
by the model). It is a fit to measured rates, not a measurement. Per GPU, the rate is the smaller of two limits:

- **SM limit:** SMs x clock / cycles per owned cell update per SM, fitted per kernel family on the committed kernels:
  1.06 cycles (sm_80, A100), 0.734 (sm_90, from H200, where the kernel is SM-bound), 0.64 (sm_100 and sm_103, B200 and
  B300).
- **Memory limit:** HBM bandwidth / 10.15 bytes per update (fitted on H100, where the kernel is memory-bound).

n GPUs multiply the one-GPU rate by n x 0.965. Host memory multiplies it by 0.87, for `execute` calls of at least
`7 qmax` steps; shorter calls are bound by the host link, which the model ignores.

What sets the cost:

- **State traffic.** The state is 24 bytes per cell (12 BF16 values). The fast path streams register pillars along z
  and advances them 7 steps per pass over the domain, so it moves about 10 bytes of HBM traffic per update (the fitted
  10.15) instead of the roughly 48 that a one-step-at-a-time kernel moves.
- **The tail.** `steps % 7` steps per call run on the one-step tail kernels, which are memory-bound. The same tail
  kernels ran 4.5 to 6.9 times slower than a fast-path step when they were measured before being moved into this
  library; that ratio has not been re-measured here. In host memory the tail is also one extra pass over the host link.
- **Host link.** In host memory each round of up to `7 qmax` steps moves the domain to the GPUs and back once; the
  rate is the GPU-memory rate x 0.87 when the call is long enough to keep the ring full, and link-bound otherwise.

## One GPU, domain in GPU memory (25 Nov 2026)

448 steps per call; median of 3 timed calls per block after a 7-step warm-up, two or more blocks per point. Every
block's end state was checked against the reference kernel's checksums over the whole domain.

| GPU | domain | rate | schedule picked by MEASURE |
|---|---|---|---|
| H100 80GB HBM3 (SXM) | 1338 x 1344 x 1336 | 3.19e11 | tx 6, gate 0 |
| A100 80GB PCIe | 1380 x 1380 x 1380 | 1.31e11 | l2h0, tx 6, gate 0 |
| B300 | 1338 x 1344 x 1336 | 4.66e11 | tx 4 |

The H100 row is from the release build. The A100 and B300 rows were measured the same day on the release candidate
before the last fixes, which did not change the GPU-memory executor or the kernel code (the kernel texts changed only
in comments).

## Several GPUs and host memory (25 Nov 2026, release candidate)

| setup | domain | steps per call | rate |
|---|---|---|---|
| 8 x H100 SXM, GPU memory | 2746 x 2880 x 2600 | 448 | 2.45e12 (MEASURE picked tx 6; tx 12 was 1.2 % faster in a separate timing) |
| H100, host memory (272 GiB of state, 3.4x the GPU) | 1338 x 5760 x 1336 | 385 | 2.74e11 |
| H100, host memory, release build, zero state | 1338 x 5760 x 1336 | 448 (2 rounds of 32) | 3.1e11 (one call, 14.8 s) |

In host memory each call streams the domain over the host link once per round of at most `7 qmax` steps; shorter
calls are bound by the link (on H100, `execute(70)` on a 1408 x 1472 x 1344 domain ran at 5.6e10).

## Planning time (H100, 1338 x 1344 x 1336, release build)

`measure` from no wisdom: 18.2 s (8 candidates, about one second of timing each, twice). With a wisdom record: 0.15 s.

Not measured: H200, B200, A100-SXM rates, and 2 to 7 GPUs.
