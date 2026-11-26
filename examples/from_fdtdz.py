# SPDX-License-Identifier: MIT
"""An fdtdz-style run in alpha-fdtd: permittivity array, dt, a source plane with a waveform, output steps.

fdtdz takes these as arguments of one call and returns E at the output steps; alpha-fdtd keeps a state you step and
read.  The mapping (docs/FROM_FDTDZ.md): Cb = dt * dt / eps (absorbed form), Ca = 1 for lossless cells, loss instead
of PML (graded conductivity near the walls), the source added to E between execute calls, every 7 steps.
"""
import os
import sys
import numpy as np
import alpha_fdtd as af

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coefficients import graded_absorber, yee_coefficients   # noqa: E402

shape = xx, yy, zz = 128, 128, 96
dt = 0.5                                             # <= 1/sqrt(3); a power of two keeps the absorbed form exact
epsilon = np.ones((3,) + shape, np.float32)          # fdtdz-style (3, xx, yy, zz) permittivity
epsilon[:, 40:88, 40:88, 30:66] = 2.25               # a dielectric block

# Coefficients: the absorber layer (graded loss, vacuum there) and the block (lossless).
ca, cb = graded_absorber(shape, thickness=16, sigma_max=0.5, dt=dt)
ca, cb = ca.copy(), cb.copy()
inside = epsilon != 1.0
ca_in, cb_in = yee_coefficients(epsilon[inside], 0.0, dt)
ca[inside], cb[inside] = ca_in, cb_in

# Source: an Ex pattern on the plane z = 20 with a modulated Gaussian waveform, added to E every 7 steps.
z0, period, every = 20, 7, 7
x = np.arange(xx, dtype=np.float32) - xx / 2
pattern = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 200).astype(np.float32)[:, :, None]   # (xx, yy, 1)
pattern[0] = pattern[:, 0] = 0                       # index 0 is the PEC wall
t = np.arange(0, 700, every)
waveform = np.sin(2 * np.pi * t / (8 * period)) * np.exp(-((t - 140) / 50.0) ** 2)

output_steps = range(140, 701, 140)
snapshots = {}
with af.Plan(shape) as plan:
    plan.set(Ca=ca, Cb=cb)
    for step, w in zip(t, waveform):
        ex = plan.get("Ex", offset=(0, 0, z0), shape=(xx, yy, 1))
        plan.set(offset=(0, 0, z0), Ex=ex + np.float32(w) * pattern)
        plan.execute(every)
        if step + every in output_steps:
            snapshots[step + every] = plan.get("E")
            print(f"step {step + every}: max |E| = {float(np.abs(snapshots[step + every]).max()):.4g}")
    print(plan.describe())
