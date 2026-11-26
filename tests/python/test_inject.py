# SPDX-License-Identifier: MIT
"""Injectors and soft sources through the Python API (alpha_fdtd.h section 8): Plan.inject bit for bit against the NumPy oracle
with the section 8 rule in NumPy float32 (each operation correctly rounded); soft_source against a float64 FDTD with the per-step
source (equal up to BF16 rounding: a timing error of one step would show as an error of about w dt)."""
import warnings

import numpy as np
import pytest

import alpha_fdtd as af
import bf16ref as ref

pytestmark = pytest.mark.gpu
SHAPE = (45, 53, 38)


def problem(shape, seed=0):
    rng = np.random.default_rng(seed)
    s = (3,) + tuple(shape)
    return (ref.quantize(rng.uniform(-1, 1, s)), ref.quantize(rng.uniform(-1, 1, s)),
            ref.quantize(rng.uniform(0.6, 1.0, s)), ref.quantize(rng.uniform(0.0, 0.3, s)))


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def oracle_inject(E, H, off, G, w):
    """the section 8 rule on (3, nx, ny, nz) bit patterns, in place"""
    sx, sy, sz = G.shape[2:]
    for c in range(6):
        F = E if c < 3 else H
        a = c % 3
        v = f32(F[a, off[0]:off[0] + sx, off[1]:off[1] + sy, off[2]:off[2] + sz])
        acc = np.zeros(v.shape, np.float32)
        for j in range(G.shape[0]):
            acc = (acc + (np.float32(w[j]) * G[j, c]).astype(np.float32)).astype(np.float32)
        r = ref.round_bits((v + acc).astype(np.float32).astype(np.float64))
        if c < 3:                                         # tangential E on the lower walls stays +0
            X, Y, Z = np.meshgrid(*(np.arange(o, o + s) for o, s in zip(off, (sx, sy, sz))), indexing="ij")
            wall = ((X == 0) & (a != 0)) | ((Y == 0) & (a != 1)) | ((Z == 0) & (a != 2))
            r = np.where(wall, F[a, off[0]:off[0] + sx, off[1]:off[1] + sy, off[2]:off[2] + sz], r)
        F[a, off[0]:off[0] + sx, off[1]:off[1] + sy, off[2]:off[2] + sz] = r


@pytest.mark.parametrize("off,shp,first,splits", [((0, 0, 0), (9, 30, 7), 7, (70,)), ((10, 20, 5), (12, 25, 20), 14, (3, 10, 1, 56))])
def test_inject_exact(off, shp, first, splits):
    E, H, Ca, Cb = problem(SHAPE, 4)
    rng = np.random.default_rng(9)
    M, rows = 5, 8
    G = rng.uniform(-0.1, 0.1, (M, 6) + shp).astype(np.float32)
    W = rng.uniform(-1, 1, (rows, M)).astype(np.float32)
    W[2] = 0                                              # skipped
    with af.Plan(SHAPE) as p:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", af.WallWarning)
            p.set(E=E, H=H, Ca=Ca, Cb=Cb)
        inj = p.inject(off, G, first, W)
        for s in splits:
            p.execute(s)
        inj.close()
        gotE, gotH = p.get("E", "H", dtype=np.uint16)
    total = sum(splits)
    Ez, Caz, Cbz = ref.wall_zero(E, Ca, Cb)
    e, h = Ez.copy(), H.copy()
    t = 0
    while t < total:
        n = min(7 - t % 7, total - t)
        e, h = ref.run(e, h, Caz, Cbz, n)
        t += n
        k = (t - first) // 7
        if t >= first and (t - first) % 7 == 0 and k < rows and W[k].any():
            oracle_inject(e, h, off, G, W[k])
    np.testing.assert_array_equal(gotE, e)
    np.testing.assert_array_equal(gotH, h)


def test_soft_source_matches_float64():
    """A modulated Gaussian Ez dipole: soft_source against the float64 per-step source (time domain, a small box)."""
    n, dt, steps = 40, 0.5, 210
    shape = (n, n, n)
    c = (20, 20, 20)
    w0 = 2 * np.pi / 10.0

    def s(k):
        return float(np.sin(w0 * dt * (k - 50)) * np.exp(-((k - 50) / 15.0) ** 2))

    ca = np.broadcast_to(np.float32(1), (3,) + shape)
    cb = np.broadcast_to(np.float32(dt * dt), (3,) + shape)
    gE = np.zeros((3, 1, 1, 1)); gE[2] = 1.0
    with af.Plan(shape) as p:
        p.set(Ca=ca, Cb=cb)
        inj = af.soft_source(p, c, E=gE, waveform=s, stop=100)
        p.execute(steps)
        inj.close()
        got = p.get("Ez").astype(np.float64)
        Ca, Cb = (a.astype(np.float64) for a in p.get("Ca", "Cb"))
    E = np.zeros((3,) + shape); H = np.zeros((3,) + shape)
    for k in range(1, steps + 1):
        H = H - af._curl_e(E)
        E = Ca * E + Cb * af._curl_h(H)
        if k <= 100:
            E[2][c] += s(k)
    err = np.linalg.norm(got - E[2]) / np.linalg.norm(E[2])
    assert err < 0.03, err


def test_inject_errors():
    with af.Plan(SHAPE) as p:
        G = np.zeros((2, 6, 4, 4, 4), np.float32)
        with pytest.raises(af.InvalidArgument):
            p.inject((0, 0, 0), G, 5, np.ones((3, 2), np.float32))       # first_step not a multiple of 7
        with pytest.raises(af.InvalidArgument):
            p.inject((43, 0, 0), G, 7, np.ones((3, 2), np.float32))      # outside the domain
        with pytest.raises(ValueError):
            p.inject((0, 0, 0), G, 7, np.ones((3, 3), np.float32))       # weights / basis mismatch
        p.set(Ca=np.ones((3,) + SHAPE, np.float32), Cb=np.ones((3,) + SHAPE, np.float32) * 0.25)
        p.execute(14)
        with pytest.raises(ValueError):
            af.soft_source(p, (5, 5, 5), E=np.ones((3, 1, 1, 1)), waveform=lambda k: 1.0, stop=10)   # stop before the plan's step
