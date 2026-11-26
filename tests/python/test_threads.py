# SPDX-License-Identifier: MIT
"""Two plans with recorders on one GPU from two threads (alpha_fdtd.h section 6): each gives what it gives alone.  One closes
its recorder and plan while the other may still be stepping with its own recorder; before plans sharing a GPU took turns,
that close could block the other plan's copies for ever."""
import threading
import warnings

import numpy as np
import pytest

import alpha_fdtd as af

pytestmark = pytest.mark.gpu
SHAPE = (120, 97, 80)


def run(seed, steps, out):
    rng = np.random.default_rng(seed)
    s = (3,) + SHAPE
    E, H = rng.uniform(-1, 1, s).astype(np.float32), rng.uniform(-1, 1, s).astype(np.float32)
    Ca, Cb = rng.uniform(0.6, 1.0, s).astype(np.float32), rng.uniform(0.0, 0.3, s).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with af.Plan(SHAPE) as p:
            p.set(E=E, H=H, Ca=Ca, Cb=Cb)
            sums = []
            r = p.record((10, 20, 30), (50, 40, 30), 7, callback=lambda step, f: sums.append(float(f.get("E").sum())))
            p.execute(steps)
            r.close()
            out[seed] = (p.checksums(), sums)


def test_two_plans_one_gpu():
    alone, together = {}, {}
    run(1, 700, alone)
    run(2, 1400, alone)
    th = [threading.Thread(target=run, args=(s, n, together), daemon=True) for s, n in ((1, 700), (2, 1400))]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=300)
    assert not any(t.is_alive() for t in th), "a plan did not finish"
    assert together == alone
