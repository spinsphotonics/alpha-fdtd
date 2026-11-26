# Coming from fdtdz

[fdtdz](https://github.com/spinsphotonics/fdtdz) is a single-GPU FDTD solver called from JAX: one call takes a permittivity array of shape `(3, xx, yy, zz)`, a
time step, a source (field pattern, waveform, position), an absorption mask, z-PML parameters and a list of output
steps, and returns E at those steps. alpha-fdtd is a different kind of tool. It keeps a state that you step and
read, it targets domains fdtdz cannot hold (fdtdz's z extent is fixed by its warp layout, and it runs on one GPU),
and it computes one update only: lossy Yee in BF16, per-cell Ca and Cb, closed PEC box. There is no PML and no
dispersion; sources are injectors (`soft_source`), and output steps can be a recorder. A full worked example is in
`examples/from_fdtdz.py`.

> **Not yet cross-checked.** The fdtdz side of this page comes from reading fdtdz's source at a 2023 commit. Nobody
> has yet run both codes on the same small vacuum problem. Until someone does, three points are open: whether
> fdtdz's `dt` maps to `C_H` as below, whether its `epsilon` is a permittivity or its inverse, and fdtdz's half-cell
> staggering. Before you compare fields cell by cell, check your arrays against the staggering in `alpha_fdtd.h`
> section 2. A different offset convention shows up as a one-cell shift.

| in fdtdz | in alpha-fdtd |
|---|---|
| `epsilon`, shape `(3, xx, yy, zz)` | `Cb` of the same shape, `Cb = dt * dt / eps`, AND `Ca = 1` for lossless cells: set both. A new plan holds `Ca = Cb = 0`, which zeroes E every step (Python warns with `CoefficientWarning` on the first `execute` if either was never set). This is the absorbed form with `C_H = dt`. The textbook `dt / eps` is wrong here, and nothing reports it |
| `dt` | `C_H`. Use `dt = 0.5`: it is below the 3-D Courant limit of 1/sqrt(3), and as a power of two it makes the absorbed form bit-identical to the textbook one |
| absorption mask near the x-y edges | Per-cell loss through Ca and Cb. With conductivity `s`, `Ca = (1 - s dt / 2 eps) / (1 + s dt / 2 eps)` and `Cb = (dt * dt / eps) / (1 + s dt / 2 eps)` (`examples/coefficients.py`). Grade `s` towards the walls to make an absorber |
| PML (z only) | None. Use a graded-loss layer, as above, thick enough for your reflection target |
| source field, waveform, position | `af.soft_source(plan, offset, E=pattern, waveform=s, stop=...)`: the pattern times s(n) is added to E at every step n, on the fast path (`docs/RECORDING.md`). Alternatively set initial fields, or step in chunks and add the source between them: `get` the plane, add to it, then `set` it back with `offset=` |
| per-step soft source | `soft_source` (exact pass-level timing). Adding between `execute(7)` calls is a 7-step comb: its images alias onto recorded spectra |
| `output_steps` and E snapshots | `for ...: plan.execute(interval); plan.get("E")`. `get` takes any box, so reading a slice moves only that slice's bytes |
| fp16 or fp32 state | BF16 state by default (8 bits of precision, float32 range); `precision="fp16"` or `"fp32"` for more accuracy (README, Precision). Fields go in and out as float32 and the library converts them |
| H not exposed | H is part of the state, at time `n - 1/2` after `n` steps: `H = dt * plan.get("H")` |
| one GPU, z fixed by the warp layout | 1 to 8 GPUs (`ngpus=`, `devices=`), at least 16 cells per axis, and host memory beyond the GPUs (`host_bytes=`, sized with `af.bytes_needed`) |
| PML in z, absorber in x-y | A closed PEC box on all six faces. Index 0 is the wall, so put sources at index 1 or more |
| JAX arrays on the GPU | NumPy arrays in host memory: `np.asarray(jax_array)`. If the process also runs JAX, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX starts, because alpha-fdtd plans with the GPU memory that is free |

The conversion, in a few lines:

```python
import numpy as np, alpha_fdtd as af

xx, yy, zz, dt = 96, 96, 64, 0.5
epsilon = np.ones((3, xx, yy, zz), np.float32)       # what you would pass to fdtdz
epsilon[:, 30:66, 30:66, 20:44] = 2.25
Ca = np.broadcast_to(np.float32(1), epsilon.shape)  # lossless
Cb = (dt * dt / epsilon).astype(np.float32)          # absorbed form: NOT dt / epsilon
src = np.zeros((xx, yy, 1), np.float32)
src[40:56, 40:56] = 1.0                              # an Ex patch on the plane z = 10 (index >= 1: not the wall)
with af.Plan((xx, yy, zz)) as plan:
    plan.set(Ca=Ca, Cb=Cb)
    for k in range(20):                              # inject every 7 steps
        ex = plan.get("Ex", offset=(0, 0, 10), shape=(xx, yy, 1))
        plan.set(offset=(0, 0, 10), Ex=ex + np.float32(np.sin(0.3 * k)) * src)
        plan.execute(7)
    E = plan.get("E")                                # E at step 140, shape (3, xx, yy, zz)
    print(float(np.abs(E).max()))
```
