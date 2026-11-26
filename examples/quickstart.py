# SPDX-License-Identifier: MIT
"""Quick start: an Ez pulse in a dielectric rod, 128^3 cells, 700 steps, on one GPU (a few seconds)."""
import numpy as np
import alpha_fdtd as af

n, dt = 128, 0.5                                     # 128^3 cells; dt = C_H, a power of two: exact
x = np.arange(n, dtype=np.float32) - n / 2
E = np.zeros((3, n, n, n), np.float32)
E[2] = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 32)[:, :, None]   # Ez pulse, uniform in z: div E = 0
eps = np.ones((n, n), np.float32)
eps[40:88, 40:88] = 4.0                              # a dielectric rod along z
Cb = np.broadcast_to((dt * dt / eps)[None, :, :, None], (3, n, n, n))  # absorbed: Cb = dt * dt / eps
with af.Plan((n, n, n)) as plan:                     # rigor="estimate": no timing
    plan.set(E=E, Ca=np.broadcast_to(np.float32(1), (3, n, n, n)), Cb=Cb)
    for frame in range(10):
        plan.execute(70)                             # multiples of 7 run on the fast path
        print(70 * (frame + 1), float(np.abs(plan.get("Ez")).max()))
    H = dt * plan.get("H")                           # physical H at time 700 - 1/2
    print(plan.describe())
