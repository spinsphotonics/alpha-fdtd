# Pulsed runs: sources, recording and the DFT

A pulsed run excites the structure with a short broadband pulse and gets the response at many frequencies from one run:
record the fields of a region in time while the steps run, then Fourier-transform the record at the end. alpha-fdtd does
this in three pieces:

- `soft_source` (Python) or `afdtd_inject` (C): a source that acts at every step (`alpha_fdtd.h` section 8).
- `Plan.record` / `afdtd_record`: frames of a box (E and H) every `every` steps, to a file or a callback (section 7).
- `dft` / `afdtd_dft`: the DFT of a recording file on a GPU, at any set of frequencies.

`examples/pulse.py` puts them together.

## Recording

```python
rec = plan.record(offset, shape, every=7, path="run.afr")   # E and H of the box every 7 steps
plan.execute(7000)                                          # frames are taken while the steps run
rec.close()
X = af.dft("run.afr", omega)                                # (nfreq, 6, sx, sy, sz) complex64
```

**How frames are taken without slowing the steps.** Inside one `execute`, the engine runs up to 256 passes of 7 steps as
one wavefront: different parts of the domain are at different times, so there is never a moment to stop and copy. A
box's state at a pass boundary stays in GPU memory from the moment the pass that writes it finishes until the next pass
overwrites it, about one pass later. The GPUs' copy engines copy the box to pinned host memory in that window. A small
stream of GPU-side waits orders the copy after the pass, and holds back only the blocks of the next pass that would
overwrite the box. The copy uses no SM time and needs no extra pass over the domain. Its costs are the copy's share of
HBM bandwidth, and a stall if the host link or the sink cannot keep up.

**Data rate.** A frame holds 12 bytes per cell (6 BF16 values), padded to whole rows of 24 y cells and whole x pairs. Let
R be the engine's rate in cell updates per second and N the number of cells in the domain. A recorder then needs
`frame_bytes * R / (N * every)` bytes per second on the host link (`afdtd_record_info` gives `frame_bytes`). Two limits
apply:

- **The copy window.** The copy is split by strips of the schedule. Each part must finish before the next pass reaches
  its strip, which is about one pass (7 steps) later, whatever `every` is. So one frame must cross the host link in about
  the time of one pass, or the steps wait for it.
- **The sink.** A file sink needs the same average rate from the disk. When the sink is slower, `execute` waits
  (`info.stall_s`).

`info.copy_s` gives the time from each frame's first copy to its last. Raise `every`, shrink the box, or put the file on
a faster disk when either limit binds.

**Choosing `every`.** Sampling every `every` steps resolves angular frequencies `|w dt| < pi / every` (radians per step).
With dt = 0.5, a vacuum wavelength of L cells is `w dt = pi / L` rad per step, so `every = 7` resolves wavelengths down to
7 cells in vacuum. A longer `every` cuts the data rate in proportion; use the largest that is above the Nyquist rate of
your band. Content above the Nyquist frequency aliases. The pulse's spectrum should be negligible there, and so should
grid noise, which a smooth pulse keeps small.

**Sizing the disk.** A recording is `frames * frame_stride` bytes plus a 4096-byte header. For example, a 400^3 box (about
0.8 GB per frame) every 7 steps over 7000 steps is 1000 frames, about 0.8 TB. The file is written with large
parallel writes, with O_DIRECT where the file system supports it, so the page cache does not fill up.

**Frames in host memory.** Without `path`, frames go to `Recorder.frames` (step, array) in Python, or to your callback
(`callback(step, frame)`, where `frame.get("E")` decodes them). The callback runs on a library thread, in step order,
while `execute` may be waiting for it. It must not call the plan (that raises), nor make CUDA calls that wait for the
plan's GPUs, such as `cudaDeviceSynchronize`, freeing GPU memory or a kernel's first launch: with every slot full, that
waits for itself. Copy the frame, or hand it to another thread. Other threads of the process have a similar rule while
`execute` runs with a recorder or injector open (`docs/API.md`, Threads and CUDA).

## The DFT

`af.dft(path, omega, components)` computes `X[f, c] = sum_k v_c(k) exp(-i omega_f tau_c(k))` on one GPU. Here `tau` is the
frame's step for E and the step minus 1/2 for H (H of a frame is at t - 1/2), and `omega` is in radians per step. Any number
of frequencies, at any spacing.

- It reads the file once. When the result does not fit on the GPU, it splits the box into chunks of frame rows, and each
  chunk is one contiguous range of every frame.
- The result goes to an array, or with `out_path=` to a `.npy` file for results larger than host memory
  (`np.load(out_path, mmap_mode="r")`).
- The sums are in float32 from a float64 phase. The test suite compares them with a float64 NumPy DFT.
- H is the stored `H / dt`. For the time integral `∫ v e^{-iwt} dt`, multiply by `every * dt`.

## Sources

Adding a source to E between `execute(7)` calls (get, add, set) injects a comb: impulses every 7 steps. Its spectrum
repeats with period `2 pi / (7 dt)`. The images radiate, and a recorder sampling every 7k steps aliases them back onto the
band being measured.

`soft_source` instead acts at every step, without leaving the fast path. Over one pass, a per-step source `s(n) g`
contributes the sum of `s(n)` times g propagated from step n to the end of the pass. That propagated field reaches at most
7 cells from g, and is a fixed field for each of the pass's 7 steps. `soft_source` computes the 7 fields once, in float64,
with the plan's Ca and Cb. The injector then adds their combination, weighted by the waveform, after each pass. In exact
arithmetic this is the per-step source. In BF16 it differs from a per-step BF16 injection by rounding only.

```python
src = af.soft_source(plan, (64, 64, 64), E=dipole, waveform=pulse, stop=300)   # E += pulse(n) * dipole at every step n
```

- `E=` and/or `H=` give the profiles, and `waveform` / `waveform_h` the amplitudes per step; H is in the absorbed form.
- The injector's box is the profile's box plus 8 cells.
- While an injection is due, the engine leaves one SM to the injection kernels.
- Weight rows that are all zero, such as after the pulse, cost nothing.

`Plan.inject` / `afdtd_inject` is the primitive under `soft_source`: any fixed fields, any weights per 7-step boundary,
with an exact rule that the tests check bit for bit.

## Limits

- Recorders and injectors need the domain in GPU memory, not host memory.
- Frames contain all six components, E and H, in the plan's precision.
- BF16 and FP16 frames are in the engine's native layout (section 7); FP32 frames are canonical (component, x, y, z).
  `Recording.get` and `dft` decode both. An FP32 frame is twice as large, and its copy runs between the steps (FP32 plans
  have no persistent launch), so it pauses them for the copy's duration.
