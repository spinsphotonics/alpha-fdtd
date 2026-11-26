# SPDX-License-Identifier: MIT
"""alpha_fdtd -- Python bindings for alpha-fdtd, a BF16 FDTD solver for large domains on 1 to 8 NVIDIA GPUs.

A thin ctypes wrapper over ``libalphafdtd.so`` (``alpha_fdtd.h``, API v1): each method is one C call, NumPy arrays are
passed by pointer and stride (no copies), and a failed call raises the :class:`Error` subclass of its status code.  The
header is the normative statement of the update, the array layout and the lower-wall rule; the two rules that silently
change results if missed (the absorbed H form and the lower wall) are restated here.

``libalphafdtd.so`` is loaded from this package's directory, or from the path in the environment variable
``ALPHA_FDTD_LIBRARY`` (development builds; the one environment variable alpha-fdtd reads).  ``import`` works without a
GPU; :class:`Plan` then raises :class:`Unsupported`.  There is no CPU fallback.

ctypes releases the GIL during every library call, so a long ``execute`` does not block other Python threads.  It also
cannot be interrupted with Ctrl-C until it returns: run long simulations as a loop of shorter ``execute`` calls.

A plan is the simulation: it holds E, H, Ca and Cb, starting at +0 (E at time 0, H at time -1/2).

The absorbed H form (read this)
  There is no H coefficient.  With ``C_H = dt`` (cell size 1, c = 1):
    * set ``Cb = Cb_textbook * dt``, i.e. ``dt * dt / eps`` for a lossless cell (0.25 in vacuum at dt = 0.5);
    * set H as ``H / dt``; ``get("H")`` returns ``H / dt``, so physical ``H = dt * plan.get("H")``.
  A textbook ``Cb = dt / eps`` is not detected: it gives a wrong trajectory in a dielectric and, in vacuum, breaks the
  Courant limit ``Cb <= 1/3`` (smooth fields can look fine for thousands of steps before they blow up).
  A new plan holds ``Ca = Cb = 0``, which zeroes E every step: set both (:class:`CoefficientWarning` otherwise).  Lossy cells: see ``examples/coefficients.py`` (Ca and Cb from eps, sigma, dt).

The lower wall (read this)
  Cells with ``i == 0``, ``j == 0`` or ``k == 0`` hold the PEC wall: the tangential E and its Ca, Cb there are always
  +0 (``alpha_fdtd.h`` section 2).  ``set`` writes +0 there.  For Ca and Cb that is silent (``Ca = 1`` everywhere is
  normal input).  For E it removes a real value, so ``set`` warns (:class:`WallWarning`) or, with
  ``strict_walls=True``, raises :class:`WallError` and writes nothing.  Put sources and initial fields at index >= 1.

Arrays
  E, H, Ca, Cb are NumPy arrays of shape ``(3, sx, sy, sz)`` (component x, y, z first, then the box; z fastest), as in
  fdtdz.  Single components (``Ex`` ... ``Cbz``) are arrays of shape ``(sx, sy, sz)``.  Element types, the same for
  ``set`` and ``get``:
    * ``numpy.float32``: rounded to BF16 on set (nearest, ties to even), widened exactly on get;
    * ``numpy.uint16``: BF16 bit patterns, stored as given (exact round trip);
    * ``ml_dtypes.bfloat16`` (if the caller uses it; never imported here): BF16 values, exact round trip.
  Any other type raises TypeError: ``float64`` (NumPy's default) must be converted with ``.astype(np.float32)``.
  Strided and broadcast views are passed without copying (``np.broadcast_to(np.float32(1), (3, nx, ny, nz))`` fills
  Ca with one value of memory); negative strides raise ValueError.

Example::

    import numpy as np, alpha_fdtd as af
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
"""
from __future__ import annotations

import ctypes as C
import dataclasses
import operator
import os
import threading
import warnings
import weakref
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Plan", "Info", "bytes_needed",
    "Error", "InvalidArgument", "OutOfMemory", "NoPlan", "Unsupported", "DeviceError", "WallError",
    "WisdomIOError", "InternalError", "WallWarning", "CoefficientWarning",
    "import_wisdom", "import_wisdom_file", "export_wisdom", "export_wisdom_file", "forget_wisdom",
    "version", "__version__",
]

__version__ = "0.1.0"

#: Component names in library order (AFDTD_EX ... AFDTD_CBZ) and the four groups of three.
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Cax", "Cay", "Caz", "Cbx", "Cby", "Cbz")
GROUPS = {"E": (0, 1, 2), "H": (3, 4, 5), "Ca": (6, 7, 8), "Cb": (9, 10, 11)}

#: Values of ``rigor`` (AFDTD_ESTIMATE ... AFDTD_WISDOM_ONLY).
RIGORS = ("estimate", "measure", "patient", "wisdom_only")

#: ``get`` without ``shape=`` or ``out=`` refuses to allocate more values than this (a whole host-memory domain can be
#: terabytes); pass ``offset=`` and ``shape=`` to read it in pieces.
GET_IMPLICIT_LIMIT = 2 ** 31

_BF16, _F32 = 1, 2                       # AFDTD_BF16, AFDTD_F32
_FORCE_HOST, _STRICT_WALLS = 1 << 8, 1 << 9
_BOX_STRIDED = 1

Shape3 = Tuple[int, int, int]


# -- errors: one class per AFDTD_E* code ------------------------------------------------------------------------------

class Error(RuntimeError):
    """A failed library call.  Attributes: ``code`` (the AFDTD_E* status), ``name`` (``"EWALLS"``, ...), ``message``
    (``afdtd_last_error()``)."""

    code = 0
    name = ""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidArgument(Error, ValueError):
    """AFDTD_EINVAL: a bad size, option, box, dtype, step count or wisdom text.  Nothing changed."""
    code, name = -1, "EINVAL"


class OutOfMemory(Error, MemoryError):
    """AFDTD_ENOMEM: the domain does not fit; the message gives the bytes needed and the ``host_bytes`` that works."""
    code, name = -2, "ENOMEM"


class NoPlan(Error):
    """AFDTD_ENOPLAN: ``rigor="wisdom_only"`` and no wisdom record for this problem."""
    code, name = -3, "ENOPLAN"


class Unsupported(Error):
    """AFDTD_EUNSUPPORTED: no GPU, an unsupported GPU, or no peer access between the GPUs.  No plan was created."""
    code, name = -4, "EUNSUPPORTED"


class DeviceError(Error):
    """AFDTD_EDEVICE: a CUDA failure during a call.  The plan is unusable; close it."""
    code, name = -5, "EDEVICE"


class WallError(Error, ValueError):
    """AFDTD_EWALLS: ``strict_walls=True`` and E has values on the lower wall.  Nothing was written."""
    code, name = -6, "EWALLS"


class WisdomIOError(Error, OSError):
    """AFDTD_EIO: a wisdom file could not be read or written."""
    code, name = -7, "EIO"


class InternalError(Error):
    """AFDTD_EINTERNAL: a self-check failed or an internal invariant broke.  Please report it."""
    code, name = -8, "EINTERNAL"


class WallWarning(UserWarning):
    """``set`` replaced non-zero E values on the lower wall (i, j or k == 0) by +0; see the module docstring."""


class CoefficientWarning(UserWarning):
    """``execute`` on a plan whose Ca or Cb was never set: a new plan holds Ca = Cb = 0, which zeroes E every step.
    Set both (``Ca = 1`` and ``Cb = dt * dt / eps`` for lossless cells), even if only eps matters to you."""


_ERRORS = {cls.code: cls for cls in (InvalidArgument, OutOfMemory, NoPlan, Unsupported, DeviceError, WallError,
                                     WisdomIOError, InternalError)}
_BY_NAME = {"AFDTD_" + cls.name: cls for cls in _ERRORS.values()}


# -- the C library ----------------------------------------------------------------------------------------------------

class _Array(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dtype", C.c_int), ("stride", C.c_int64 * 3)]


class _Box(C.Structure):
    _fields_ = [("offset", C.c_int64 * 3), ("shape", C.c_int64 * 3), ("flags", C.c_uint), ("array", _Array * 12)]


class _Options(C.Structure):
    _fields_ = [("ngpus", C.c_int), ("devices", C.POINTER(C.c_int)), ("host_bytes", C.c_size_t), ("rigor", C.c_int),
                ("flags", C.c_uint)]


class _Info(C.Structure):
    _fields_ = [("nx", C.c_int64), ("ny", C.c_int64), ("nz", C.c_int64), ("ngpus", C.c_int), ("devices", C.c_int * 8),
                ("on_host", C.c_int), ("rigor", C.c_int), ("from_wisdom", C.c_int), ("io_block_y", C.c_int64),
                ("gpu_bytes_used", C.c_size_t), ("host_bytes_used", C.c_size_t), ("steps_done", C.c_int64),
                ("tail_steps", C.c_int64), ("walls_zeroed_coef", C.c_uint64), ("walls_zeroed_field", C.c_uint64),
                ("predicted_rate", C.c_double), ("measured_rate", C.c_double), ("last_rate", C.c_double)]


def _load():
    path = os.environ.get("ALPHA_FDTD_LIBRARY") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "libalphafdtd.so")
    try:
        lib = C.CDLL(path)
    except OSError as e:
        raise ImportError(f"alpha_fdtd: cannot load the library {path}: {e}") from None
    P, I, I64, SZ = C.c_void_p, C.c_int, C.c_int64, C.c_size_t
    sigs = {
        "afdtd_plan_create": (P, [I64, I64, I64, C.POINTER(_Options)]),
        "afdtd_plan_destroy": (None, [P]),
        "afdtd_bytes_needed": (I, [I64, I64, I64, I, C.POINTER(SZ), C.POINTER(SZ)]),
        "afdtd_set": (I, [P, C.POINTER(_Box)]),
        "afdtd_get": (I, [P, C.POINTER(_Box)]),
        "afdtd_execute": (I, [P, I64]),
        "afdtd_checksums": (I, [P, C.POINTER(C.c_uint64)]),
        "afdtd_plan_info": (I, [P, C.POINTER(_Info)]),
        "afdtd_plan_describe": (I, [P, C.c_char_p, SZ]),
        "afdtd_export_wisdom_to_string": (P, []),
        "afdtd_import_wisdom_from_string": (I, [C.c_char_p]),
        "afdtd_export_wisdom_to_filename": (I, [C.c_char_p]),
        "afdtd_import_wisdom_from_filename": (I, [C.c_char_p]),
        "afdtd_forget_wisdom": (None, []),
        "afdtd_free": (None, [P]),
        "afdtd_version": (C.c_char_p, []),
        "afdtd_last_error": (C.c_char_p, []),
    }
    for name, (res, args) in sigs.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = res, args
    v = lib.afdtd_version().decode()
    if v != __version__:
        raise ImportError(f"alpha_fdtd {__version__}: the library {path} is version {v}")
    return lib


_lib = _load()


def _error(code: Optional[int] = None) -> Error:
    msg = _lib.afdtd_last_error().decode(errors="replace")
    if code is None:                    # afdtd_plan_create returns NULL; the text starts with the code's name
        code = _BY_NAME.get(msg.split(":", 1)[0], InternalError).code
    return _ERRORS.get(code, InternalError)(msg or f"alpha-fdtd status {code}")


def _check(rc: int) -> int:
    if rc < 0:
        raise _error(rc)
    return rc


def _path(p) -> bytes:
    return os.fsencode(os.fspath(p))


# -- array marshalling (pointers and strides only) --------------------------------------------------------------------

def _dtype_code(dt: np.dtype) -> int:
    if dt == np.float32 and dt.isnative:
        return _F32
    if (dt == np.uint16 or dt.name == "bfloat16") and dt.isnative:
        return _BF16
    raise TypeError(f"alpha_fdtd: unsupported element type {dt}; use numpy.float32 (for example .astype(np.float32)), "
                    f"numpy.uint16 (BF16 bit patterns) or ml_dtypes.bfloat16")


def _fill(slot: _Array, a: np.ndarray) -> None:
    slot.dtype = _dtype_code(a.dtype)
    for ax, s in enumerate(a.strides):
        if s < 0:
            raise ValueError("alpha_fdtd: negative strides are not supported (copy the array first)")
        if s % a.itemsize:
            raise ValueError("alpha_fdtd: array strides are not a multiple of the element size (copy the array first)")
        slot.stride[ax] = s // a.itemsize
    slot.data = a.ctypes.data


def _components(groups: dict, singles: dict) -> dict:
    """{component index: 3-D array} from group arrays (3, sx, sy, sz) and single components (sx, sy, sz)."""
    out = {}
    for name, arr in list(groups.items()) + list(singles.items()):
        if arr is None:
            continue
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"alpha_fdtd: {name} must be a numpy array, not {type(arr).__name__}")
        if name in GROUPS:
            if arr.ndim != 4 or arr.shape[0] != 3:
                raise ValueError(f"alpha_fdtd: {name} must have shape (3, sx, sy, sz), not {arr.shape}")
            parts = zip(GROUPS[name], arr)
        elif name in COMPONENTS:
            if arr.ndim != 3:
                raise ValueError(f"alpha_fdtd: {name} must have shape (sx, sy, sz), not {arr.shape}")
            parts = [(COMPONENTS.index(name), arr)]
        else:
            raise TypeError(f"alpha_fdtd: unknown component {name!r}; use E, H, Ca, Cb or one of {', '.join(COMPONENTS)}")
        for c, a in parts:
            if c in out:
                raise ValueError(f"alpha_fdtd: {COMPONENTS[c]} is given twice")
            out[c] = a
    return out


def _box(offset, shape, comps: dict) -> _Box:
    b = _Box()
    b.offset[:] = offset
    b.shape[:] = shape
    b.flags = _BOX_STRIDED
    for c, a in comps.items():
        _fill(b.array[c], a)
    return b


def _int(v, what: str, hi: int = 2 ** 63 - 1, example: str = "") -> int:
    """An integer argument for a C int64 / size_t: a TypeError naming the parameter for floats and other types,
    InvalidArgument (a ValueError) outside [0, hi] (ctypes would silently keep only the low 64 bits)."""
    try:
        i = operator.index(v)
    except TypeError:
        raise TypeError(f"alpha_fdtd: {what} must be an int, not {type(v).__name__}{example}") from None
    if not 0 <= i <= hi:
        raise InvalidArgument(f"alpha_fdtd: {what} = {i} is out of range (0 to {hi})")
    return i


def _shape3(s, what: str) -> Shape3:
    t = tuple(_int(v, what) for v in s)
    if len(t) != 3:
        raise ValueError(f"alpha_fdtd: {what} must have 3 entries, not {len(t)}")
    return t


# -- plans ------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Info:
    """``afdtd_info``, field for field (see alpha_fdtd.h)."""

    shape: Shape3
    devices: Tuple[int, ...]
    on_host: bool
    rigor: str
    from_wisdom: bool
    io_block_y: int
    gpu_bytes_used: int
    host_bytes_used: int
    steps_done: int
    tail_steps: int
    walls_zeroed_coef: int
    walls_zeroed_field: int
    predicted_rate: float
    measured_rate: float
    last_rate: float


def bytes_needed(shape: Shape3, ngpus: int = 1) -> Tuple[int, int]:
    """``afdtd_bytes_needed``: ``(gpu_bytes, host_bytes)``, the state's bytes on the busiest GPU with the domain in GPU
    memory, and the ``host_bytes`` a host-memory plan needs.  No GPU needed.  Your own float32 arrays come on top:
    12 bytes per cell for each of E, H, Ca, Cb you build in full."""
    g, h = C.c_size_t(), C.c_size_t()
    _check(_lib.afdtd_bytes_needed(*_shape3(shape, "shape"), _int(ngpus, "ngpus", 8), C.byref(g), C.byref(h)))
    return g.value, h.value


class Plan:
    """A planned domain and its state (``afdtd_plan``): the simulation.  Use as a context manager, or call
    :meth:`close`; a plan that is garbage-collected is closed then."""

    def __init__(
        self,
        shape: Shape3,
        *,
        rigor: str = "estimate",
        ngpus: Optional[int] = None,
        devices: Optional[Sequence[int]] = None,
        host_bytes: int = 0,
        force_host: bool = False,
        strict_walls: bool = False,
    ) -> None:
        """Plan an ``(nx, ny, nz)`` domain and allocate its state, every value +0 (``afdtd_plan_create``).

        Args:
            shape: cells along x, y, z; at least 16 each.
            rigor: ``"estimate"`` (cost model, no timing), ``"measure"`` (times candidates: seconds to minutes),
                ``"patient"`` (more candidates: minutes) or ``"wisdom_only"``.  Wisdom of equal or higher rigor is
                used when present.
            ngpus: number of GPUs, 1 to 8; default ``len(devices)``, or 1.
            devices: CUDA device ordinals; default ``0 .. ngpus-1``.  ValueError if it disagrees with ``ngpus``.
            host_bytes: bytes of host memory the plan may pin for the domain when it does not fit on the GPUs
                (0 = GPU memory only); see :func:`bytes_needed`.  An int, for example ``int(1.5 * 2**40)``.
            force_host: keep the domain in host memory even if it fits on the GPUs.
            strict_walls: raise :class:`WallError` instead of warning when E has values on the lower wall.

        In a process that also runs JAX, set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` before JAX starts: alpha-fdtd
        plans with the GPU memory that is free, and JAX's default pre-allocation silently moves the domain to host
        memory (about 10x slower) or makes the plan fail.

        Raises:
            OutOfMemory, NoPlan, Unsupported, InvalidArgument, DeviceError, InternalError.
        """
        self._p = None
        self._cv = threading.Condition()      # close() waits for calls in flight on other threads
        self._busy = 0
        self._coef = set()                    # coefficient components set so far (CoefficientWarning)
        self._shape = _shape3(shape, "shape")
        if rigor not in RIGORS:
            raise ValueError(f"alpha_fdtd: rigor must be one of {RIGORS}, not {rigor!r}")
        o = _Options(rigor=RIGORS.index(rigor), host_bytes=_int(host_bytes, "host_bytes", 2 ** 64 - 1,
                                                                ", e.g. int(1.5 * 2**30) bytes"),
                     flags=(_FORCE_HOST if force_host else 0) | (_STRICT_WALLS if strict_walls else 0))
        if devices is not None:
            devs = [_int(d, "devices", 2 ** 31 - 1) for d in devices]
            if ngpus is not None and ngpus != len(devs):
                raise ValueError(f"alpha_fdtd: ngpus={ngpus} but {len(devs)} devices given")
            self._devs = (C.c_int * len(devs))(*devs)          # kept alive for the call
            o.ngpus, o.devices = len(devs), C.cast(self._devs, C.POINTER(C.c_int))
        else:
            o.ngpus = 1 if ngpus is None else _int(ngpus, "ngpus", 8)
        p = _lib.afdtd_plan_create(*self._shape, C.byref(o))
        if not p:
            raise _error()
        self._p = p
        self._fin = weakref.finalize(self, _lib.afdtd_plan_destroy, C.c_void_p(p))

    def _call(self, f, *args) -> int:
        """f(handle, *args) with the plan held open: close() on another thread waits until it returns."""
        with self._cv:
            if self._p is None:
                raise ValueError("plan is closed")
            self._busy += 1
            h = self._p
        try:
            return _check(f(h, *args))
        finally:
            with self._cv:
                self._busy -= 1
                self._cv.notify_all()

    # -- fields ------------------------------------------------------------------------------------------------------

    def set(self, *, offset: Optional[Shape3] = None, E: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None,
            Ca: Optional[np.ndarray] = None, Cb: Optional[np.ndarray] = None, **components: np.ndarray) -> None:
        """Copy arrays into the state (``afdtd_set``, one atomic call).  Arrays not given are left unchanged.

        ``E``, ``H``, ``Ca``, ``Cb`` have shape ``(3, sx, sy, sz)``; single components (``Ez=...``, ``Cbx=...``) have
        shape ``(sx, sy, sz)``.  Every array in one call covers the same box.  The arrays are not modified.

        ``Cb`` and ``H`` are in the absorbed form: ``Cb = Cb_textbook * dt`` (``dt * dt / eps`` lossless) and
        ``H = H_physical / dt`` (module docstring).  E on the lower wall is replaced by +0 with a :class:`WallWarning`
        (``strict_walls``: :class:`WallError`, nothing written); Ca and Cb there silently.  Counts: ``Plan.info``.

        Args:
            offset: first cell of the box.  Without it every array must cover the whole domain (ValueError
                otherwise, so a wrong-shaped array is never written as a sub-box at the origin).

        Raises:
            TypeError: an unsupported element type (for float64: ``.astype(np.float32)``).
            ValueError: shapes that disagree with each other or with the domain, a component named twice, a
                negative stride.
            WallError, InvalidArgument, DeviceError.
        """
        comps = _components({"E": E, "H": H, "Ca": Ca, "Cb": Cb}, components)
        if not comps:
            raise ValueError("alpha_fdtd: set needs at least one array")
        shapes = {a.shape for a in comps.values()}
        if len(shapes) != 1:
            raise ValueError(f"alpha_fdtd: the arrays of one set must cover the same box, got shapes {sorted(shapes)}")
        shape = shapes.pop()
        if offset is None:
            if shape != self._shape:
                raise ValueError(f"alpha_fdtd: arrays of shape {shape} do not cover the {self._shape} domain; "
                                 f"pass offset= to set a sub-box")
            offset = (0, 0, 0)
        self._call(_lib.afdtd_set, C.byref(_box(_shape3(offset, "offset"), shape, comps)))
        self._coef.update(c for c in comps if c >= 6)
        info = _Info()
        self._call(_lib.afdtd_plan_info, C.byref(info))
        if info.walls_zeroed_field:
            warnings.warn(f"alpha_fdtd: {info.walls_zeroed_field} E value(s) on the lower walls (index 0) were "
                          f"replaced by 0; put fields at index >= 1", WallWarning, stacklevel=2)

    def get(self, *names: str, offset: Shape3 = (0, 0, 0), shape: Optional[Shape3] = None, dtype: type = np.float32,
            out: Union[None, np.ndarray, Sequence[np.ndarray]] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Copy arrays out of the state (``afdtd_get``).

        After n steps in all, E is at time n and H at time n - 1/2, and H is the stored ``H / dt``: physical
        ``H = dt * plan.get("H")``.

        Args:
            names: ``"E"``, ``"H"``, ``"Ca"``, ``"Cb"`` (each ``(3, sx, sy, sz)``) or single components (``"Ez"``,
                each ``(sx, sy, sz)``).  At least one.
            offset, shape: the box; ``shape=None`` means up to the end of the domain, refused (ValueError) if the
                result would exceed ``GET_IMPLICIT_LIMIT`` values: pass ``shape=`` and read in pieces.
            dtype: ``numpy.float32``, ``numpy.uint16`` (bit patterns) or ``ml_dtypes.bfloat16``.
            out: arrays to write into (one per name; any non-negative, non-zero strides), instead of new arrays.

        Returns:
            One array for one name, else a tuple in the order of ``names``.
        """
        if not names:
            raise ValueError("alpha_fdtd: get needs at least one name (E, H, Ca, Cb, Ex, ...)")
        if len(set(names)) != len(names):
            raise ValueError(f"alpha_fdtd: a name is given twice in {names}")
        offset = _shape3(offset, "offset")
        if any(o >= n for o, n in zip(offset, self._shape)):
            raise ValueError(f"alpha_fdtd: box offset {offset} is not inside the {self._shape} domain")
        if out is not None:
            outs = [out] if isinstance(out, np.ndarray) else list(out)
            if len(outs) != len(names):
                raise ValueError(f"alpha_fdtd: {len(names)} names but {len(outs)} out arrays")
            for o in outs:
                if not o.flags.writeable:
                    raise ValueError("alpha_fdtd: an out array is read-only")
            box = outs[0].shape[-3:]
            if shape is not None and _shape3(shape, "shape") != box:
                raise ValueError(f"alpha_fdtd: shape={shape} disagrees with the out arrays' box {box}")
        else:
            if shape is None:
                box = tuple(n - o for n, o in zip(self._shape, offset))
                if sum((3 if n in GROUPS else 1) for n in names) * int(np.prod(box)) > GET_IMPLICIT_LIMIT:
                    raise ValueError(f"alpha_fdtd: get of {names} over {box} without shape= exceeds "
                                     f"GET_IMPLICIT_LIMIT values; pass shape= (and offset=) to read it in pieces")
            else:
                box = _shape3(shape, "shape")
            dt = np.dtype(dtype)
            _dtype_code(dt)
            outs = [np.empty(((3,) if n in GROUPS else ()) + tuple(box), dt) for n in names]
        comps = _components({}, dict(zip(names, outs)))
        if any(a.shape != tuple(box) for a in comps.values()):
            raise ValueError("alpha_fdtd: the out arrays must cover the same box")
        self._call(_lib.afdtd_get, C.byref(_box(offset, box, comps)))
        return outs[0] if len(outs) == 1 else tuple(outs)

    # -- time stepping ------------------------------------------------------------------------------------------------

    def execute(self, steps: int) -> None:
        """Advance the state by ``steps`` steps (>= 0) (``afdtd_execute``).  Any count, any split, same bits.

        Speed: ``steps // 7`` rounds run on the fast path, ``steps % 7`` single steps on a much slower path.  Use
        multiples of 7; ``execute(1)`` per step (per-step source injection) is the slowest way to run.
        """
        steps = _int(steps, "steps")
        if steps and len(self._coef) < 6:
            miss = [COMPONENTS[c] for c in range(6, 12) if c not in self._coef]
            warnings.warn(f"alpha_fdtd: execute with {', '.join(miss)} never set (a new plan holds 0 there, which zeroes E); "
                          f"set both Ca and Cb", CoefficientWarning, stacklevel=2)
            self._coef.update(range(6, 12))                   # warn once
        self._call(_lib.afdtd_execute, steps)

    def checksums(self) -> Tuple[int, int, int, int, int, int]:
        """Six 64-bit checksums of Ex Ey Ez Hx Hy Hz (E and H only; not Ca, Cb), independent of the GPU count and the
        memory mode (``afdtd_checksums``; definition in alpha_fdtd.h).  For comparing runs without copying out."""
        out = (C.c_uint64 * 6)()
        self._call(_lib.afdtd_checksums, out)
        return tuple(out)

    # -- introspection ------------------------------------------------------------------------------------------------

    @property
    def shape(self) -> Shape3:
        """``(nx, ny, nz)``."""
        return self._shape

    @property
    def info(self) -> Info:
        """``afdtd_plan_info``, as an :class:`Info`."""
        i = _Info()
        self._call(_lib.afdtd_plan_info, C.byref(i))
        return Info((i.nx, i.ny, i.nz), tuple(i.devices[:i.ngpus]), bool(i.on_host), RIGORS[i.rigor],
                    bool(i.from_wisdom), i.io_block_y, i.gpu_bytes_used, i.host_bytes_used, i.steps_done, i.tail_steps,
                    i.walls_zeroed_coef, i.walls_zeroed_field, i.predicted_rate, i.measured_rate, i.last_rate)

    def describe(self) -> str:
        """``afdtd_plan_describe``: memory mode, GPUs, rigor, rates and tail share, then the schedule; one line."""
        buf = C.create_string_buffer(4096)
        n = self._call(_lib.afdtd_plan_describe, buf, len(buf))
        if n >= len(buf):
            buf = C.create_string_buffer(n + 1)
            self._call(_lib.afdtd_plan_describe, buf, len(buf))
        return buf.value.decode()

    # -- life cycle ---------------------------------------------------------------------------------------------------

    def close(self) -> None:
        """Free the plan and its state (``afdtd_plan_destroy``).  Idempotent; any later method raises
        ``ValueError("plan is closed")``.  A call running on another thread (``execute``) finishes first."""
        with self._cv:
            if self._p is None:
                return
            self._p = None
            while self._busy:
                self._cv.wait()
        self._fin()

    def __enter__(self) -> "Plan":
        return self

    def __exit__(self, *exc) -> None:
        """Calls :meth:`close`."""
        self.close()

    def __repr__(self) -> str:
        return f"<alpha_fdtd.Plan {self._shape}{' closed' if self._p is None else ''}>"


# -- wisdom (FFTW names, one function per C function) -----------------------------------------------------------------

def import_wisdom(text: str) -> int:
    """Merge wisdom text (``afdtd_import_wisdom_from_string``).  Returns the number of records read."""
    return _check(_lib.afdtd_import_wisdom_from_string(text.encode()))


def import_wisdom_file(path) -> int:
    """Merge a wisdom file (``afdtd_import_wisdom_from_filename``).  Returns the number of records read."""
    return _check(_lib.afdtd_import_wisdom_from_filename(_path(path)))


def export_wisdom() -> str:
    """All wisdom as text (``afdtd_export_wisdom_to_string``)."""
    p = _lib.afdtd_export_wisdom_to_string()
    if not p:
        raise _error(OutOfMemory.code)
    try:
        return C.string_at(p).decode()
    finally:
        _lib.afdtd_free(p)


def export_wisdom_file(path) -> None:
    """Write all wisdom to a file (``afdtd_export_wisdom_to_filename``)."""
    _check(_lib.afdtd_export_wisdom_to_filename(_path(path)))


def forget_wisdom() -> None:
    """Drop all wisdom held by the library (``afdtd_forget_wisdom``)."""
    _lib.afdtd_forget_wisdom()


def version() -> str:
    """The library's version (``afdtd_version``); equal to ``__version__`` (checked at import)."""
    return _lib.afdtd_version().decode()
