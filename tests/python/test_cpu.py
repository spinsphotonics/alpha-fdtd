# SPDX-License-Identifier: MIT
"""CPU tests (no GPU): the wrapper's marshalling against a recording stand-in for the C library, the real library's
refusals and wisdom, the example coefficient formulas, and the release tree's hygiene."""
import ctypes as C
import gc
import os
import re
import warnings

import numpy as np
import pytest

import alpha_fdtd as af
from conftest import ROOT


# -- a recording stand-in for libalphafdtd ----------------------------------------------------------------------------

class FakeLib:
    """Records every call; plan handle 1234; returns the status in ``self.rc`` and fakes plan_info."""

    def __init__(self):
        self.calls, self.boxes, self.rc, self.err, self.wf, self.wc, self.destroyed = [], [], 0, b"", 0, 0, 0
        self.create_result = 1234

    def _rec(self, name, *a):
        self.calls.append(name)

    def afdtd_plan_create(self, nx, ny, nz, opt):
        self._rec("create")
        o = opt._obj
        self.opts = dict(ngpus=o.ngpus, rigor=o.rigor, flags=o.flags, host_bytes=o.host_bytes,
                         devices=[o.devices[i] for i in range(o.ngpus)] if o.devices else None)
        return self.create_result

    def afdtd_plan_destroy(self, p):
        self._rec("destroy")
        self.destroyed += 1

    def _box(self, name, p, box):
        self._rec(name)
        b = box._obj
        self.boxes.append(dict(offset=tuple(b.offset), shape=tuple(b.shape), flags=b.flags,
                               arrays={c: (b.array[c].data, b.array[c].dtype, tuple(b.array[c].stride))
                                       for c in range(12) if b.array[c].data}))
        return self.rc

    def afdtd_set(self, p, box):
        return self._box("set", p, box)

    def afdtd_get(self, p, box):
        return self._box("get", p, box)

    def afdtd_execute(self, p, steps):
        self._rec("execute")
        self.steps = steps
        return self.rc

    def afdtd_plan_info(self, p, info):
        self._rec("info")
        i = info._obj
        i.nx, i.ny, i.nz, i.ngpus, i.rigor, i.on_host = 32, 33, 34, 2, 1, 1
        i.devices[0], i.devices[1] = 3, 5
        i.walls_zeroed_field, i.walls_zeroed_coef, i.tail_steps = self.wf, self.wc, 4
        return 0

    def afdtd_plan_describe(self, p, buf, n):
        self._rec("describe")
        s = b"x" * 5000
        if n:
            C.memmove(buf, s[:n - 1] + b"\0", n)
        return len(s)

    def afdtd_checksums(self, p, out):
        self._rec("checksums")
        for c in range(6):
            out[c] = c + 2 ** 63
        return 0

    def afdtd_last_error(self):
        return self.err

    def afdtd_version(self):
        return b"0.2.0"


@pytest.fixture
def fake(monkeypatch):
    f = FakeLib()
    monkeypatch.setattr(af, "_lib", f)
    return f


@pytest.fixture
def plan(fake):
    p = af.Plan((32, 33, 34))
    fake.calls.clear()
    return p


def strides(a):
    return tuple(s // a.itemsize for s in a.strides)


# -- marshalling --------------------------------------------------------------------------------------------------------

def test_options(fake):
    af.Plan((32, 33, 34), rigor="patient", devices=[3, 5], host_bytes=2 ** 40, force_host=True, strict_walls=True)
    assert fake.opts == dict(ngpus=2, rigor=2, flags=(1 << 8) | (1 << 9), host_bytes=2 ** 40, devices=[3, 5])
    af.Plan((32, 33, 34))
    assert fake.opts == dict(ngpus=1, rigor=0, flags=0, host_bytes=0, devices=None)
    af.Plan((32, 33, 34), ngpus=4, rigor="wisdom_only")
    assert fake.opts["ngpus"] == 4 and fake.opts["rigor"] == 3
    with pytest.raises(ValueError):
        af.Plan((32, 33, 34), ngpus=1, devices=[0, 1])
    with pytest.raises(ValueError):
        af.Plan((32, 33, 34), rigor="fast")
    with pytest.raises(ValueError):
        af.Plan((32, 33))


def test_set_dense_broadcast_and_strided(plan, fake):
    E = np.zeros((3, 32, 33, 34), np.float32)
    Ca = np.broadcast_to(np.float32(1), (3, 32, 33, 34))
    Hz = np.zeros((34, 33, 32), np.uint16).transpose(2, 1, 0)           # strided view
    plan.set(E=E, Ca=Ca, Hz=Hz)
    assert fake.calls == ["set", "info"]                                  # one atomic C call, then the wall count
    b = fake.boxes[0]
    assert b["offset"] == (0, 0, 0) and b["shape"] == (32, 33, 34) and b["flags"] == 1
    assert sorted(b["arrays"]) == [0, 1, 2, 5, 6, 7, 8]
    for a in range(3):
        assert b["arrays"][a] == (E[a].ctypes.data, 2, (33 * 34, 34, 1))
        assert b["arrays"][6 + a] == (Ca.ctypes.data, 2, (0, 0, 0))     # broadcast: zero strides, one value
    assert b["arrays"][5] == (Hz.ctypes.data, 1, (1, 32, 32 * 33))


def test_set_subbox_needs_offset(plan, fake):
    with pytest.raises(ValueError, match="offset"):
        plan.set(Ez=np.zeros((4, 4, 4), np.float32))
    plan.set(offset=(1, 2, 3), Ez=np.zeros((4, 4, 4), np.float32))
    assert fake.boxes[-1]["offset"] == (1, 2, 3) and fake.boxes[-1]["shape"] == (4, 4, 4)


@pytest.mark.parametrize("bad, exc", [
    (dict(E=np.zeros((3, 32, 33, 34))), TypeError),                       # float64
    (dict(E=np.zeros((3, 32, 33, 34), np.complex64)), TypeError),
    (dict(Ex=np.zeros((32, 33, 34), ">f4")), TypeError),                  # byte-swapped
    (dict(Ex=[[[0.0]]]), TypeError),
    (dict(Ex=np.zeros((32, 33, 34), np.float32)[::-1]), ValueError),     # negative stride
    (dict(E=np.zeros((32, 33, 34), np.float32)), ValueError),            # group without its axis of 3
    (dict(E=np.zeros((3, 32, 33, 34), np.float32), Ex=np.zeros((32, 33, 34), np.float32)), ValueError),  # twice
    (dict(Ex=np.zeros((32, 33, 34), np.float32), Ey=np.zeros((32, 33, 33), np.float32)), ValueError),    # two boxes
    (dict(Ew=np.zeros((32, 33, 34), np.float32)), TypeError),
    (dict(), ValueError),
])
def test_set_refusals(plan, fake, bad, exc):
    with pytest.raises(exc):
        plan.set(**bad)
    assert fake.calls == []                                               # refused before the library is called


def test_float64_message_names_the_fix(plan):
    with pytest.raises(TypeError, match=re.escape(".astype(np.float32)")):
        plan.set(E=np.zeros((3, 32, 33, 34)))


def test_ml_dtypes_bfloat16(plan, fake):
    ml = pytest.importorskip("ml_dtypes")
    plan.set(Ex=np.zeros((32, 33, 34), ml.bfloat16))
    assert fake.boxes[-1]["arrays"][0][1] == 1
    a = plan.get("Ex", dtype=ml.bfloat16)
    assert a.dtype == ml.bfloat16 and fake.boxes[-1]["arrays"][0][1] == 1


def test_wall_warning_only_for_fields(plan, fake):
    E = np.zeros((3, 32, 33, 34), np.float32)
    fake.wc = 10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan.set(Ca=E)                                                    # coefficients on the wall: silent
    fake.wf = 3
    with pytest.warns(af.WallWarning, match="3 E value"):
        plan.set(E=E)


def test_get(plan, fake):
    E, ez = plan.get("E", "Hz", offset=(1, 2, 3))
    assert fake.calls == ["get"]
    assert E.shape == (3, 31, 31, 31) and E.dtype == np.float32 and ez.shape == (31, 31, 31)
    b = fake.boxes[0]
    assert b["offset"] == (1, 2, 3) and b["shape"] == (31, 31, 31)
    assert b["arrays"][0] == (E[0].ctypes.data, 2, strides(E[0])) and b["arrays"][5][0] == ez.ctypes.data
    h = plan.get("Hy", shape=(2, 3, 4), dtype=np.uint16)
    assert h.shape == (2, 3, 4) and fake.boxes[-1]["arrays"][4][1] == 1
    out = np.zeros((3, 40, 40, 40), np.float32)[:, :5, 1:7, ::2]
    assert plan.get("Cb", out=out) is out
    assert fake.boxes[-1]["shape"] == (5, 6, 20) and fake.boxes[-1]["arrays"][9][2] == strides(out[0])


def test_get_refusals(plan, fake):
    for kw in [dict(dtype=np.float64), dict(out=np.zeros((32, 33, 34), np.float32)[None].repeat(2, 0)),
               dict(out=np.broadcast_to(np.float32(0), (32, 33, 34)))]:
        with pytest.raises((TypeError, ValueError)):
            plan.get("Ex", **kw)
    with pytest.raises(ValueError):
        plan.get()
    with pytest.raises(ValueError):
        plan.get("Ex", "Ex")
    assert fake.calls == []


def test_get_implicit_limit(fake):
    p = af.Plan((2048, 2048, 1024))
    with pytest.raises(ValueError, match="shape="):
        p.get("Ex")                                                       # 2^32 values: refused, nothing allocated
    assert p.get("Ex", shape=(1, 1, 1)).shape == (1, 1, 1)


def test_execute_checksums_info_describe(plan, fake):
    plan.execute(np.int64(14))
    assert fake.steps == 14
    with pytest.raises(TypeError):
        plan.execute(1.5)
    assert plan.checksums() == tuple(c + 2 ** 63 for c in range(6))
    i = plan.info
    assert i.shape == (32, 33, 34) and i.devices == (3, 5) and i.on_host is True and i.rigor == "measure"
    assert i.tail_steps == 4
    assert plan.describe() == "x" * 5000                                 # longer than the first buffer
    assert plan.shape == (32, 33, 34)


@pytest.mark.parametrize("code, cls, base", [(-1, af.InvalidArgument, ValueError), (-2, af.OutOfMemory, MemoryError),
                                             (-3, af.NoPlan, af.Error), (-4, af.Unsupported, af.Error),
                                             (-5, af.DeviceError, af.Error), (-6, af.WallError, ValueError),
                                             (-7, af.WisdomIOError, OSError), (-8, af.InternalError, af.Error)])
def test_errors(plan, fake, code, cls, base):
    fake.rc, fake.err = code, b"AFDTD_X: reason"
    with pytest.raises(cls) as e:
        plan.execute(1)
    assert isinstance(e.value, base) and isinstance(e.value, af.Error) and e.value.code == code
    assert e.value.message == "AFDTD_X: reason"
    fake.create_result, fake.err = None, ("AFDTD_" + cls.name + ": no").encode()
    with pytest.raises(cls):
        af.Plan((32, 33, 34))


def test_close(fake):
    with af.Plan((32, 33, 34)) as p:
        pass
    assert fake.destroyed == 1
    p.close()
    assert fake.destroyed == 1
    for f in (lambda: p.execute(7), lambda: p.get("Ex"), lambda: p.info, lambda: p.checksums()):
        with pytest.raises(ValueError, match="plan is closed"):
            f()
    af.Plan((32, 33, 34))
    gc.collect()
    assert fake.destroyed == 2                                            # garbage-collected plans are destroyed


def test_integer_arguments(plan, fake):
    for bad in (2 ** 70, 2 ** 64 + 7, -1):                                # ctypes would keep the low 64 bits
        with pytest.raises(ValueError, match="steps"):
            plan.execute(bad)
    with pytest.raises(TypeError, match="steps must be an int"):
        plan.execute(7.0)
    with pytest.raises(TypeError, match="host_bytes must be an int"):
        af.Plan((32, 33, 34), host_bytes=1.5 * 2 ** 30)
    with pytest.raises(ValueError, match="shape"):
        af.Plan((2 ** 64 + 64, 72, 48))
    with pytest.raises(ValueError, match="offset"):
        plan.set(offset=(2 ** 64, 0, 0), Ex=np.zeros((1, 1, 1), np.float32))
    with pytest.raises(ValueError, match="not inside"):
        plan.get("Ez", offset=(40, 0, 0))
    assert "execute" not in fake.calls and "get" not in fake.calls and "set" not in fake.calls


def test_coefficient_warning(plan, fake):
    with pytest.warns(af.CoefficientWarning, match="Cax"):
        plan.execute(7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan.execute(7)                                                   # once per plan
        p = af.Plan((32, 33, 34))
        one = np.ones((3, 32, 33, 34), np.float32)
        p.set(Ca=one, Cb=one * 0.25)
        p.execute(7)


def test_close_waits_for_a_call_in_flight(fake):
    import threading, time
    p = af.Plan((32, 33, 34))
    started, order = threading.Event(), []
    def slow(h, steps):
        started.set(); time.sleep(0.2); order.append("execute"); return 0
    def destroy(h):
        order.append("destroy")
    fake.afdtd_execute, fake.afdtd_plan_destroy = slow, destroy
    p._fin.detach()
    p._fin = lambda: destroy(None)
    t = threading.Thread(target=lambda: p.execute(7))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t.start(); started.wait(); p.close(); t.join()
    assert order == ["execute", "destroy"]


# -- the real library, no GPU needed -----------------------------------------------------------------------------------

def test_version():
    assert af.version() == af.__version__ == "0.2.0"


def test_bytes_needed():
    g1, h1 = af.bytes_needed((1000, 1000, 1000))
    g2, _ = af.bytes_needed((1000, 1000, 1000), ngpus=2)
    assert g1 >= 24e9 and h1 >= 24e9 and g2 < g1
    with pytest.raises(af.InvalidArgument):
        af.bytes_needed((15, 16, 16))
    with pytest.raises(af.InvalidArgument, match="too many cells"):
        af.bytes_needed((2 ** 20,) * 3)                                   # would overflow size_t
    g3, h3 = af.bytes_needed((1000, 1000, 1000), precision="fp32")
    g4, _ = af.bytes_needed((1000, 1000, 1000), precision="fp16")
    assert g3 >= 48e9 and h3 == 0 and g4 == g1                           # FP32: 48 bytes per cell, GPU memory only
    with pytest.raises(ValueError):
        af.bytes_needed((100, 100, 100), precision="fp64")


def test_plan_refusals_without_gpu():
    for kw in [dict(shape=(15, 16, 16)), dict(shape=(16, 16, 16), ngpus=9), dict(shape=(16, 16, 16), ngpus=-1)]:
        with pytest.raises(af.InvalidArgument):
            af.Plan(**kw)


def test_wisdom_text_and_files(tmp_path):
    rec = 'kset=000000000000 gpu=X ngpus=1 n=16x16x16 mode=gpu rigor=measure rate=2 sched="b"'
    af.forget_wisdom()
    assert af.export_wisdom() == "alpha-fdtd-wisdom 1\n"
    with pytest.raises(af.InvalidArgument):
        af.import_wisdom("alpha-fdtd-wisdom 1\n" + rec + "\ngarbage\n")
    assert af.export_wisdom() == "alpha-fdtd-wisdom 1\n"                 # nothing merged from a bad text
    assert af.import_wisdom("alpha-fdtd-wisdom 1\n" + rec + "\n") == 1
    af.export_wisdom_file(tmp_path / "w")
    af.forget_wisdom()
    assert af.import_wisdom_file(str(tmp_path / "w")) == 1
    assert af.export_wisdom() == "alpha-fdtd-wisdom 1\n" + rec + "\n"
    with pytest.raises(af.WisdomIOError):
        af.import_wisdom_file(tmp_path / "missing")
    with pytest.raises(af.WisdomIOError):
        af.export_wisdom_file(tmp_path / "no-such-dir" / "w")
    af.forget_wisdom()


# -- documentation and examples ----------------------------------------------------------------------------------------

def test_coefficients_example_matches_header():
    from coefficients import yee_coefficients
    header = open(os.path.join(ROOT, "include", "alpha_fdtd.h")).read()
    assert "Ca  = (1 - s dt / (2 eps)) / (1 + s dt / (2 eps))" in header
    assert "Cb' = (dt * dt / eps)      / (1 + s dt / (2 eps))" in header
    assert "dt * dt / eps" in af.__doc__ and "H = dt * plan.get(\"H\")" in af.__doc__
    for eps, s, dt in [(1.0, 0.0, 0.5), (4.0, 0.3, 0.5), (2.25, 0.05, 0.4)]:
        ca, cb = yee_coefficients(eps, s, dt)
        q = s * dt / (2 * eps)
        assert ca == np.float32((1 - q) / (1 + q)) and cb == np.float32((dt * dt / eps) / (1 + q))
    assert yee_coefficients(1.0)[1] == np.float32(0.25)


def test_release_tree_hygiene():
    """Every source file has the SPDX line; no absolute home paths anywhere in the tree."""
    missing, paths = [], []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if not x.startswith((".", "build", "_skbuild", "dist")) and x != "__pycache__"]
        for f in files:
            p = os.path.join(d, f)
            if f.endswith((".c", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".py", ".txt", ".toml")) and "data" not in d:
                head = open(p, errors="replace").read(400)
                if "SPDX-License-Identifier: MIT" not in head and not f.startswith("zs7_"):
                    missing.append(os.path.relpath(p, ROOT))
            if f.endswith((".c", ".cpp", ".cu", ".h", ".hpp", ".py", ".md", ".txt", ".toml", ".map", ".cuh")):
                if re.search(r"/(Users|home)/[A-Za-z]", open(p, errors="replace").read()):
                    paths.append(os.path.relpath(p, ROOT))
    assert not missing, f"no SPDX line: {missing}"
    assert not paths, f"absolute home paths: {paths}"
