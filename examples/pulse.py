# SPDX-License-Identifier: MIT
"""A pulsed run: a broadband soft dipole source, a recorder on a sub-box, and the fields at 200 frequencies at the end.

The source acts at every step (soft_source: exact pass-level timing, no 7-step comb).  The recorder copies the box (E and H)
every 7 steps to a file while the steps run; af.dft transforms the file on the GPU afterwards, for any set of frequencies.
Frequencies are in radians per step (w dt); sampling every 7 steps resolves |w dt| < pi / 7.
"""
import os
import sys
import tempfile

import numpy as np
import alpha_fdtd as af

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coefficients import graded_absorber   # noqa: E402

n, dt = 128, 0.5
shape = (n, n, n)
ca, cb = graded_absorber(shape, thickness=16, sigma_max=0.5, dt=dt)      # vacuum with an absorbing layer
w0 = 2 * np.pi / 16                                                      # centre: vacuum wavelength 16 cells
def pulse(k):                                                           # E injection amplitude at step k
    return np.sin(w0 * dt * (k - 150)) * np.exp(-((k - 150) / 50.0) ** 2)

dipole = np.zeros((3, 1, 1, 1), np.float32)
dipole[2] = 1.0                                                          # Ez at one cell
path = os.path.join(tempfile.mkdtemp(), "pulse.afr")
with af.Plan(shape) as plan:
    plan.set(Ca=ca, Cb=cb)
    src = af.soft_source(plan, (64, 64, 64), E=dipole, waveform=pulse, stop=300)
    rec = plan.record((80, 32, 32), (16, 64, 64), every=7, path=path)
    plan.execute(2100)                                                   # 300 frames, taken while the steps run
    print(rec.info.frames, "frames,", rec.info.stall_s, "s waiting for the disk")
    rec.close()
    src.close()

omega = np.linspace(0.5, 1.5, 200) * w0 * dt                             # radians per step
X = af.dft(path, omega, "E")                                             # (200, 3, 16, 64, 64) complex64
print(X.shape, "peak |Ez| at", omega[np.abs(X[:, 2]).max(axis=(1, 2, 3)).argmax()] / dt, "rad per unit time")
os.remove(path)
