# SPDX-License-Identifier: MIT
"""The three precisions through the Python API (alpha_fdtd.h section 1): round trips in each state's own type, conversions,
FP16 overflow, FP32 refusals, the accuracy order (FP16 closer to FP32 than BF16 on a smooth problem), and recorders, the DFT and
soft sources in FP16 and FP32.  Bit-exactness against the oracle is in the C tests (tests/gpu, prec=fp16 / fp32)."""
import warnings

import numpy as np
import pytest

import alpha_fdtd as af

pytestmark = pytest.mark.gpu
SHAPE = (48, 61, 40)


def smooth(shape, seed=0):
    """A smooth Ez pulse, uniform lossless coefficients (Cb = 0.25: dt = 0.5 in vacuum)."""
    x, y, z = (np.arange(n, dtype=np.float64) - n / 2 for n in shape)
    E = np.zeros((3,) + shape, np.float32)
    E[2] = np.exp(-(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2) / 30)
    return E, np.broadcast_to(np.float32(1), (3,) + shape), np.broadcast_to(np.float32(0.25), (3,) + shape)


@pytest.mark.parametrize("prec, dt", [("fp16", np.float16), ("fp32", np.float32)])
def test_round_trip_and_info(prec, dt):
    rng = np.random.default_rng(1)
    with af.Plan(SHAPE, precision=prec) as p:
        assert p.precision == prec and prec.upper() in p.describe()
        H = rng.uniform(-2, 2, (3,) + SHAPE).astype(dt)
        p.set(H=H)
        np.testing.assert_array_equal(p.get("H", dtype=dt), H)                    # the state's own type: exact
        Hf = rng.uniform(-2, 2, (3,) + SHAPE).astype(np.float32)
        p.set(H=Hf)
        want = Hf.astype(np.float16).astype(np.float32) if prec == "fp16" else Hf  # rounded to nearest even
        np.testing.assert_array_equal(p.get("H"), want)
        if prec == "fp16":
            p.set(offset=(3, 3, 3), Hx=np.full((2, 2, 2), 7e4, np.float32))         # beyond 65504: infinity
            assert np.isinf(p.get("Hx", offset=(3, 3, 3), shape=(2, 2, 2))).all()


def test_fp32_refusals():
    with pytest.raises(af.InvalidArgument):
        af.Plan(SHAPE, precision="fp32", force_host=True, host_bytes=1 << 34)
    with pytest.raises(ValueError):
        af.Plan(SHAPE, precision="fp64")


def test_accuracy_order():
    """The same smooth problem in BF16, FP16 and FP32: FP16 is well closer to FP32 than BF16 is."""
    E, Ca, Cb = smooth(SHAPE)
    out = {}
    for prec in ("bf16", "fp16", "fp32"):
        with af.Plan(SHAPE, precision=prec) as p:
            p.set(E=E, Ca=Ca, Cb=Cb)
            p.execute(280)
            out[prec] = p.get("E").astype(np.float64)
    ref = np.linalg.norm(out["fp32"])
    e_bf, e_f16 = np.linalg.norm(out["bf16"] - out["fp32"]) / ref, np.linalg.norm(out["fp16"] - out["fp32"]) / ref
    assert e_f16 < e_bf / 3, (e_bf, e_f16)


@pytest.mark.parametrize("prec", ["fp16", "fp32"])
def test_recorder_and_dft(prec, tmp_path):
    E, Ca, Cb = smooth(SHAPE)
    box = ((5, 23, 3), (30, 30, 30))
    with af.Plan(SHAPE, precision=prec) as a, af.Plan(SHAPE, precision=prec) as b:
        for p in (a, b):
            p.set(E=E, Ca=Ca, Cb=Cb)
        r = a.record(*box, 14, tmp_path / "r.afr")
        assert r.info.dtype == prec
        a.execute(140)
        r.close()
        rec = af.Recording(tmp_path / "r.afr")
        assert len(rec) == 11
        for k in range(11):
            want = np.concatenate(b.get("E", "H", offset=box[0], shape=box[1]))
            np.testing.assert_array_equal(rec.get(k), want)
            if k < 10:
                b.execute(14)
        assert a.checksums() == b.checksums()
    om = np.array([0.1, 0.25, 0.4])
    got = af.dft(tmp_path / "r.afr", om)
    X = np.stack([rec.get(k).astype(np.float64) for k in range(len(rec))])
    tau = rec.steps.astype(np.float64)
    for c in range(6):
        ph = np.exp(-1j * om[:, None] * (tau[None, :] - (0.5 if c >= 3 else 0.0)))
        want = np.tensordot(ph, X[:, c], axes=(1, 0))
        np.testing.assert_allclose(got[:, c], want, rtol=0, atol=3e-6 * max(np.abs(want).max(), 1e-30))
    part = af.dft(tmp_path / "r.afr", om, "Hy", gpu_bytes=32 << 20)               # chunked
    np.testing.assert_allclose(part[:, 0], got[:, 4], rtol=0, atol=1e-6 * np.abs(got[:, 4]).max())


def test_soft_source_fp32():
    """FP32 soft source against the float64 per-step source: equal to float32 rounding."""
    n, dt, steps = 40, 0.5, 140
    shape = (n, n, n); c = (20, 20, 20); w0 = 2 * np.pi / 10.0
    s = lambda k: float(np.sin(w0 * dt * (k - 40)) * np.exp(-((k - 40) / 12.0) ** 2))
    gE = np.zeros((3, 1, 1, 1)); gE[2] = 1.0
    with af.Plan(shape, precision="fp32") as p:
        p.set(Ca=np.broadcast_to(np.float32(1), (3,) + shape), Cb=np.broadcast_to(np.float32(dt * dt), (3,) + shape))
        inj = af.soft_source(p, c, E=gE, waveform=s, stop=80)
        p.execute(steps)
        inj.close()
        got = p.get("Ez").astype(np.float64)
        Ca, Cb = (a.astype(np.float64) for a in p.get("Ca", "Cb"))
    E = np.zeros((3,) + shape); H = np.zeros((3,) + shape)
    for k in range(1, steps + 1):
        H = H - af._curl_e(E)
        E = Ca * E + Cb * af._curl_h(H)
        if k <= 80:
            E[2][c] += s(k)
    assert np.linalg.norm(got - E[2]) / np.linalg.norm(E[2]) < 1e-5
