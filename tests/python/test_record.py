# SPDX-License-Identifier: MIT
"""Recorders and the DFT through the Python API (alpha_fdtd.h section 7): frames equal ``get`` at their steps (bit for bit),
file and memory sinks agree, the GPU DFT agrees with a float64 NumPy DFT of the frames, errors.  The frame decoder is also
checked without a GPU on a synthetic file built from the layout formula of the header."""
import gc
import json
import os
import warnings

import numpy as np
import pytest

import alpha_fdtd as af
import bf16ref as ref

SHAPE = (61, 77, 45)
BOX = ((5, 23, 3), (40, 30, 37))          # crosses a row boundary in y (24), odd offsets


def problem(shape, seed=0):
    rng = np.random.default_rng(seed)
    s = (3,) + tuple(shape)
    return (ref.quantize(rng.uniform(-1, 1, s)), ref.quantize(rng.uniform(-1, 1, s)),
            ref.quantize(rng.uniform(0.6, 1.0, s)), ref.quantize(rng.uniform(0.0, 0.3, s)))


def setup(plan, seed=0):
    E, H, Ca, Cb = problem(plan.shape, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", af.WallWarning)
        plan.set(E=E, H=H, Ca=Ca, Cb=Cb)


def box_now(plan, box):
    off, shp = box
    E, H = plan.get("E", "H", offset=off, shape=shp)
    return np.concatenate([E, H])


@pytest.mark.gpu
def test_frames_equal_get(tmp_path):
    """One execute(140) with a memory recorder and a file recorder == execute(14) x 10 with get() after each."""
    with af.Plan(SHAPE) as a, af.Plan(SHAPE) as b:
        setup(a); setup(b)
        mem = a.record(*BOX, 14)
        fil = a.record(*BOX, 14, tmp_path / "r.afr")
        a.execute(140)
        mem.flush(); fil.close()
        want = [box_now(b, BOX)]
        for _ in range(10):
            b.execute(14)
            want.append(box_now(b, BOX))
        assert [s for s, _ in mem.frames] == list(range(0, 141, 14))
        rec = af.Recording(tmp_path / "r.afr")
        assert len(rec) == 11 and list(rec.steps) == list(range(0, 141, 14))
        for k, w in enumerate(want):
            np.testing.assert_array_equal(mem.get(k), w)
            np.testing.assert_array_equal(rec.get(k), w)
        np.testing.assert_array_equal(rec.get(3, "Hy"), want[3][4:5])
        i = mem.info
        assert i.frames == 11 and i.first_step == 0 and i.every == 14
        assert a.checksums() == b.checksums()                 # recording does not change the state
        mem.close()


@pytest.mark.gpu
def test_unaligned_and_callback():
    """Steps not starting at a multiple of 7: the head runs on the tail path; frames still at multiples of every."""
    seen = []
    with af.Plan(SHAPE) as a, af.Plan(SHAPE) as b:
        setup(a, 3); setup(b, 3)
        a.execute(3); b.execute(3)
        r = a.record(*BOX, 7, callback=lambda step, f: seen.append((step, f.get("E").copy())))
        a.execute(4 + 21 + 2)                                  # steps 3 .. 30: frames at 7, 14, 21, 28
        r.close()
        assert [s for s, _ in seen] == [7, 14, 21, 28]
        for s, e in seen:
            b.execute(s - b.info.steps_done)
            np.testing.assert_array_equal(e, b.get("E", offset=BOX[0], shape=BOX[1]))
        b.execute(30 - b.info.steps_done)
        assert a.checksums() == b.checksums()


@pytest.mark.gpu
def test_dft_against_numpy(tmp_path):
    with af.Plan(SHAPE) as a:
        setup(a, 5)
        r = a.record(*BOX, 7, tmp_path / "d.afr")
        a.execute(7 * 40)
        r.close()
    rec = af.Recording(tmp_path / "d.afr")
    om = np.array([0.0, 0.05, 0.3, 1.1, -0.7])
    got = af.dft(tmp_path / "d.afr", om)
    X = np.stack([rec.get(k).astype(np.float64) for k in range(len(rec))])        # (t, 6, sx, sy, sz)
    tau = rec.steps.astype(np.float64)
    want = np.empty((om.size, 6) + BOX[1], np.complex128)
    for c in range(6):
        ph = np.exp(-1j * om[:, None] * (tau[None, :] - (0.5 if c >= 3 else 0.0)))   # (f, t)
        want[:, c] = np.tensordot(ph, X[:, c], axes=(1, 0))
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err < 2e-6, err
    part = af.dft(tmp_path / "d.afr", om[1:3], ("Ey", "Hz"), gpu_bytes=64 << 20)    # chunked over rows
    np.testing.assert_allclose(part, got[1:3][:, [1, 5]], rtol=0, atol=2e-6 * np.abs(want).max())
    af.dft(tmp_path / "d.afr", om, "H", out_path=tmp_path / "h.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "h.npy"), got[:, 3:])


@pytest.mark.gpu
def test_record_errors(tmp_path):
    with af.Plan(SHAPE) as a:
        setup(a)
        with pytest.raises(af.InvalidArgument):
            a.record(*BOX, 10)
        with pytest.raises(af.InvalidArgument):
            a.record((50, 0, 0), (20, 4, 4), 7)
        with pytest.raises(af.FileIOError):
            a.record(*BOX, 7, tmp_path / "no" / "such" / "dir.afr")

        def bad(step, f):
            raise RuntimeError("sink")
        r = a.record(*BOX, 7, callback=bad)
        with pytest.raises(af.FileIOError):                    # reported by execute, or by flush (frames reach the sink later)
            a.execute(14)
            r.flush()
        assert a.info.steps_done == 14                         # the steps ran
        with pytest.raises(af.FileIOError):
            r.close()
    shape = (37, 240, 29)
    with af.Plan(shape, force_host=True, host_bytes=af.bytes_needed(shape)[1]) as h:
        with pytest.raises(af.InvalidArgument):
            h.record((0, 0, 0), (4, 4, 4), 7)


@pytest.mark.gpu
def test_recorder_lifetime_and_reentry():
    """A recorder whose object is dropped keeps delivering (the plan holds it); a callback that calls the plan fails cleanly
    (no deadlock); close after the plan's close does nothing."""
    seen = []
    with af.Plan(SHAPE) as a:
        setup(a)
        a.record(*BOX, 7, callback=lambda step, f: seen.append(step))       # not kept by the caller
        gc.collect()
        _ = [af._FRAME_FN(lambda *x: 0) for _ in range(50)]                  # reuse freed callback memory, if any
        a.execute(140)
        for r in list(a._kids.values()):
            r.flush()
        assert seen == list(range(0, 141, 7))
        bad = a.record(*BOX, 7, callback=lambda step, f: a.info)             # a plan call from the callback
        with pytest.raises(af.FileIOError):
            a.execute(14)
            bad.flush()
        kept = a.record(*BOX, 7)
    kept.close()                                                             # the plan is closed: a no-op
    bad.close()


def test_decoder_synthetic(tmp_path):
    """No GPU: a file written from the layout formula decodes to the values put in."""
    off, shp, every = (3, 21, 2), (7, 30, 5), 14
    row0, rows, pair0, pairs, z0, planes = 0, 3, 1, 5, 2, 5
    fb = rows * planes * pairs * 576
    stride = -(-fb // 4096) * 4096
    rng = np.random.default_rng(1)
    vals = rng.integers(0, 65535, (2, 6) + shp, dtype=np.uint16)
    frames = np.zeros((2, stride // 2), np.uint16)
    for k in range(2):
        for c in range(6):
            for x in range(shp[0]):
                for y in range(shp[1]):
                    for z in range(shp[2]):
                        X, Y, Z = off[0] + x, off[1] + y, off[2] + z
                        e = ((((Y // 24 - row0) * planes + (Z - z0)) * pairs + (X // 2 - pair0)) * 24 + Y % 24) * 12 + 2 * c + (X & 1)
                        frames[k, e] = vals[k, c, x, y, z]
    hdr = {"format": "alpha-fdtd-record", "version": 1, "domain": [20, 60, 10], "offset": list(off), "shape": list(shp),
           "every": every, "first_step": 7, "frames": 2, "complete": True, "row0": row0, "rows": rows, "z0": z0,
           "planes": planes, "pair0": pair0, "pairs": pairs, "frame_bytes": fb, "frame_stride": stride, "data_offset": 4096}
    p = tmp_path / "s.afr"
    with open(p, "wb") as f:
        f.write(json.dumps(hdr).encode().ljust(4096, b" "))
        f.write(frames.tobytes())
    rec = af.Recording(p)
    assert list(rec.steps) == [7, 21]
    for k in range(2):
        got = rec.get(k).view(np.uint32) >> 16
        np.testing.assert_array_equal(got, vals[k])
