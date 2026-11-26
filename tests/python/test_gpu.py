# SPDX-License-Identifier: MIT
"""GPU tests through the Python API, against the NumPy oracle tests/oracle/bf16ref.py (bit for bit): in GPU and host
memory, any step count and split, sub-boxes, strides, broadcast, element types, walls, wisdom, errors, several GPUs."""
import re
import warnings

import numpy as np
import pytest

import alpha_fdtd as af
import bf16ref as ref

pytestmark = pytest.mark.gpu
SMALL = (37, 50, 29)            # off every block grid
HOSTS = (37, 240, 29)           # the smallest host-memory domains need about 10 rows of 24 y-cells


def problem(shape, seed=0):
    """Random lossy problem as bit patterns, random values on the walls too."""
    rng = np.random.default_rng(seed)
    s = (3,) + tuple(shape)
    return (ref.quantize(rng.uniform(-1, 1, s)), ref.quantize(rng.uniform(-1, 1, s)),
            ref.quantize(rng.uniform(0.6, 1.0, s)), ref.quantize(rng.uniform(0.0, 0.3, s)))


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def plan_for(shape, host=False, **kw):
    if host:
        kw.update(force_host=True, host_bytes=af.bytes_needed(shape, kw.get("ngpus") or 1)[1])
    return af.Plan(shape, **kw)


def check_exact(plan, E, H, Ca, Cb, splits):
    """set (E as float32, the rest as bit patterns), execute each split, compare everything with the oracle."""
    Ez, Caz, Cbz = ref.wall_zero(E, Ca, Cb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", af.WallWarning)
        plan.set(E=f32(E), H=H, Ca=Ca, Cb=Cb)
    for s in splits:
        plan.execute(s)
    rE, rH = ref.run(Ez, H, Caz, Cbz, sum(splits))
    gE, gH, gCa, gCb = plan.get("E", "H", "Ca", "Cb", dtype=np.uint16)
    assert (gE != rE).sum() == 0 and (gH != rH).sum() == 0
    assert np.array_equal(gCa, Caz) and np.array_equal(gCb, Cbz)
    assert plan.checksums() == ref.checksums(rE, rH)
    i = plan.info
    assert i.steps_done == sum(splits) and i.tail_steps == sum(s % 7 for s in splits)


@pytest.mark.parametrize("host", [False, True], ids=["gpu_memory", "host_memory"])
@pytest.mark.parametrize("splits", [[1], [7], [13], [1, 6, 7], [3, 3, 1, 6], [1] * 9 + [14], [0, 21]])
def test_exact(host, splits):
    shape = HOSTS if host else SMALL
    with plan_for(shape, host) as p:
        assert p.info.on_host == host
        check_exact(p, *problem(shape, len(splits)), splits)


def test_exact_nonfinite():
    E, H, Ca, Cb = problem(SMALL, 7)
    H.reshape(-1)[::997] = 0x7F80
    H.reshape(-1)[5::1499] = 0x7FC0
    H.reshape(-1)[9::701] = 0x7F7F
    E[0, 0, 5, 5] = 0x7F80                       # Ex on the plane i = 0: normal to it, kept
    with af.Plan(SMALL) as p:
        check_exact(p, E, H, Ca, Cb, [3, 7, 1])


def test_subbox_strided_and_out():
    E, H, Ca, Cb = problem(SMALL, 3)
    E2, H2, _, _ = problem(SMALL, 4)
    o, s = (5, 7, 2), (11, 20, 9)
    box = tuple(slice(a, a + b) for a, b in zip(o, s))
    with af.Plan(SMALL) as p:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", af.WallWarning)
            p.set(E=f32(E), H=H, Ca=Ca, Cb=Cb)
        big = np.zeros((2 * s[0], s[1], 3 * s[2]), np.uint16)            # a strided view of a larger array
        big[::2, :, ::3] = H2[1][box]
        p.set(offset=o, Hy=big[::2, :, ::3], Ex=np.ascontiguousarray(f32(E2[0][box]).T).T)   # transposed memory
        H[1][box], E[0][box] = H2[1][box], E2[0][box]
        out = np.zeros((3, s[0], 2 * s[1], s[2]), np.uint16)[:, :, ::2]
        assert p.get("H", offset=o, shape=s, dtype=np.uint16, out=out) is out
        assert np.array_equal(out, H[(slice(None),) + box])
        assert np.array_equal(p.get("Ex", offset=o, shape=s, dtype=np.uint16), E[0][box])
        check_exact(p, E, H, Ca, Cb, [7])


@pytest.mark.parametrize("host", [False, True], ids=["gpu_memory", "host_memory"])
def test_broadcast_equals_dense(host):
    shape = HOSTS if host else SMALL
    E, H, _, _ = problem(shape, 5)
    E, _, _ = ref.wall_zero(E, E, E)
    cb_plane = np.linspace(0.1, 0.3, shape[1], dtype=np.float32)[None, None, :, None]
    Ca = np.broadcast_to(np.float32(0.875), (3,) + shape)
    Cb = np.broadcast_to(cb_plane, (3,) + shape)
    sums = []
    for coef in [(Ca, Cb), (np.ascontiguousarray(Ca), np.ascontiguousarray(Cb))]:
        with plan_for(shape, host) as p:
            p.set(E=f32(E), H=H, Ca=coef[0], Cb=coef[1])
            p.execute(8)
            sums.append((p.checksums(), p.get("Ca", "Cb", dtype=np.uint16)))
    assert sums[0][0] == sums[1][0]
    assert all(np.array_equal(a, b) for a, b in zip(sums[0][1], sums[1][1]))


def test_dtypes_round_trip():
    E, H, Ca, Cb = problem(SMALL, 6)
    with af.Plan(SMALL) as p:
        p.set(offset=(1, 1, 1), H=H[:, 1:, 1:, 1:])
        assert np.array_equal(p.get("H", offset=(1, 1, 1), dtype=np.uint16), H[:, 1:, 1:, 1:])
        assert np.array_equal(p.get("Hx", offset=(1, 1, 1)).view(np.uint32), H[0, 1:, 1:, 1:].astype(np.uint32) << 16)
        x = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8, -0.0, np.inf, 3.0e38], np.float32)   # ties to even; overflow
        p.set(offset=(1, 1, 1), Hz=x.reshape(5, 1, 1))
        assert p.get("Hz", offset=(1, 1, 1), shape=(5, 1, 1), dtype=np.uint16).ravel().tolist() == \
            ref.quantize(x).tolist()
        try:
            import ml_dtypes
        except ImportError:
            return
        b = p.get("H", offset=(1, 1, 1), dtype=ml_dtypes.bfloat16)
        p.set(offset=(1, 1, 1), H=b)
        assert np.array_equal(p.get("H", offset=(1, 1, 1), dtype=np.uint16), b.view(np.uint16))


def test_walls():
    with af.Plan(SMALL) as p:
        E = np.zeros((3,) + SMALL, np.float32)
        E[1, 0, 3, 4] = 1.0                      # Ey on i = 0: tangential, zeroed
        E[0, 0, 3, 4] = 2.0                      # Ex on i = 0: normal, kept
        E[0, 1:, 1:, 0] = -0.0                   # -0 for Ex on k = 0 (tangential): replaced, not counted
        with pytest.warns(af.WallWarning, match="1 E value"):
            p.set(E=E)
        assert p.info.walls_zeroed_field == 1
        assert p.get("Ey")[0, 3, 4] == 0 and p.get("Ex")[0, 3, 4] == 2.0
        assert p.get("Ex", dtype=np.uint16)[:, :, 0].max() == 0         # +0 bits
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p.set(Ca=np.ones((3,) + SMALL, np.float32))                  # coefficients on the wall: silent
        nx, ny, nz = SMALL
        assert p.info.walls_zeroed_coef == 2 * (ny * nz + nx * nz + nx * ny) - (nx + ny + nz)
        assert p.info.walls_zeroed_field == 0
    with af.Plan(SMALL, strict_walls=True) as p:
        before = p.checksums()
        with pytest.raises(af.WallError, match="Ey"):
            p.set(E=E, H=np.ones((3,) + SMALL, np.float32))
        assert p.checksums() == before                                   # nothing was written
        E[1, 0] = 0
        p.set(E=E, Ca=np.ones((3,) + SMALL, np.float32))                 # coefficients are never refused
        p.execute(7)


def test_errors():
    with af.Plan(SMALL) as p:
        before = p.checksums()
        with pytest.raises(af.InvalidArgument, match="not inside"):
            p.set(offset=(30, 0, 0), Ex=np.zeros((10, 5, 5), np.float32))
        w = np.lib.stride_tricks.as_strided(np.zeros(1, np.float32), (4, 4, 4), (0, 0, 0), writeable=True)
        with pytest.raises(af.InvalidArgument, match="zero stride"):
            p.get("Ex", shape=(4, 4, 4), out=w)
        with pytest.raises(af.InvalidArgument):
            p.execute(-1)
        assert p.checksums() == before and p.info.steps_done == 0
    with pytest.raises(af.InvalidArgument, match="does not exist"):
        af.Plan(SMALL, devices=[99])
    with pytest.raises(af.OutOfMemory, match="host_bytes"):
        af.Plan((8192, 8192, 8192))
    af.forget_wisdom()
    with pytest.raises(af.NoPlan):
        af.Plan(SMALL, rigor="wisdom_only")


def test_wisdom_round_trip(tmp_path):
    shape = (64, 72, 48)
    af.forget_wisdom()
    E, H, Ca, Cb = problem(shape, 9)
    with af.Plan(shape, rigor="measure") as p:
        i = p.info
        assert i.rigor == "measure" and not i.from_wisdom and i.measured_rate > 0
        check_exact(p, E, H, Ca, Cb, [14])
        assert p.info.last_rate > 0
        desc = p.describe()
    text = af.export_wisdom()
    assert text.startswith("alpha-fdtd-wisdom 1\n") and text.count("rigor=measure") == 1
    af.export_wisdom_file(tmp_path / "w")
    af.forget_wisdom()
    assert af.import_wisdom_file(tmp_path / "w") == 1
    with af.Plan(shape, rigor="wisdom_only") as p:
        i = p.info
        assert i.from_wisdom and i.rigor == "measure"
        assert desc.split(" | ")[-1] == p.describe().split(" | ")[-1]          # the same schedule
        check_exact(p, E, H, Ca, Cb, [14])
    with af.Plan(shape) as p:                    # estimate uses a record of higher rigor
        assert p.info.from_wisdom
    af.forget_wisdom()
    with af.Plan(shape) as p:
        assert not p.info.from_wisdom and p.info.rigor == "estimate"


def test_illegal_wisdom_is_ignored():
    """A record with a pool too small for its grid is not used (planned again; wisdom_only: NoPlan), never EINTERNAL."""
    shape = (64, 72, 48)
    af.forget_wisdom()
    with af.Plan(shape, rigor="measure"):
        pass
    text = af.export_wisdom()
    bad = re.sub(r"mx=\d+", "mx=64", text)
    assert bad != text
    af.forget_wisdom()
    af.import_wisdom(bad)
    with af.Plan(shape) as p:
        assert not p.info.from_wisdom
        p.execute(7)
    with pytest.raises(af.NoPlan):
        af.Plan(shape, rigor="wisdom_only")
    af.forget_wisdom()


@pytest.mark.gpu2
@pytest.mark.parametrize("host", [False, True], ids=["gpu_memory", "host_memory"])
def test_two_gpus(host):
    shape = HOSTS if host else (37, 97, 29)      # in GPU memory on 2 GPUs: ny > 72
    with plan_for(shape, host, ngpus=2) as p:
        assert len(p.info.devices) == 2 and p.info.on_host == host
        check_exact(p, *problem(shape, 11), [3, 7, 1])
    with plan_for(shape, host, devices=[1, 0]) as p:
        assert p.info.devices == (1, 0)
        check_exact(p, *problem(shape, 12), [14])
