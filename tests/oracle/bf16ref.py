# SPDX-License-Identifier: MIT
"""bf16ref -- a NumPy oracle for the update of alpha_fdtd.h section 1, on BF16 bit patterns (uint16).

Every BF16 value is exactly a binary64 value.  Products of two BF16 values are exact in binary64.  A sum (of two values,
or of an exact product and a value) is computed in binary64 rounded to odd (TwoSum decides inexactness), then rounded to
BF16 through binary32 rounded to odd and an integer round-half-even of the low 16 bits.  Rounding a round-to-odd value
of q >= p + 2 bits to p bits gives the single correct rounding (Boldo and Melquiond), so each operation is one correctly
rounded BF16 operation, with signed zeros, subnormals, overflow to infinity, and NaN as the canonical quiet 0x7FFF.
Independent of the library: it shares no code with it and runs on the CPU.
"""
import numpy as np

CYC = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
_CANON_NAN = 0x7FFF


def to_f64(bits):
    """BF16 bit patterns (uint16) -> exact binary64 values."""
    with np.errstate(invalid="ignore"):
        return (np.asarray(bits).astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)


def _f32_odd_bits(x):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        t = x.astype(np.float32)
    back = t.astype(np.float64)
    b = t.view(np.uint32).copy()
    inexact = back != x
    b[inexact & (np.abs(back) > np.abs(x))] -= np.uint32(1)       # one binary32 towards zero
    b[inexact] |= np.uint32(1)                                       # sticky
    return b


def round_bits(x):
    """binary64 values (exact, or rounded to odd) -> BF16 bit patterns, round to nearest even."""
    x = np.asarray(x, np.float64)
    shape = x.shape
    x = x.ravel()
    nan = np.isnan(x)
    b = _f32_odd_bits(np.where(nan, 0.0, x)).astype(np.uint64)
    r = ((b + np.uint64(0x7FFF) + ((b >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)).astype(np.uint16)
    r[nan] = _CANON_NAN
    return r.reshape(shape)


def quantize(x):
    """Any float array -> BF16 bit patterns (one rounding)."""
    return round_bits(np.asarray(x, np.float64))


def _sum_odd(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        s = a + b
        bb = s - a
        err = (a - (s - bb)) + (b - bb)
    fix = np.isfinite(s) & (err != 0.0) & ((s.view(np.int64) & 1) == 0)
    if fix.any():
        s = s.copy()
        s[fix] = np.nextafter(s[fix], np.where(err[fix] > 0, np.inf, -np.inf))
    return s


def _rnd(x):
    return to_f64(round_bits(x))


def _sub(a, b):
    return _rnd(_sum_odd(a, -b))


def _mul(a, b):
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        return _rnd(a * b)


def _fma(a, b, c):
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        return _rnd(_sum_odd(a * b, c))


def _shift(arr, axis, d):
    """arr[index + d] along axis; outside the domain reads +0."""
    out = np.zeros_like(arr)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    src[axis], dst[axis] = (slice(d, None), slice(None, -d)) if d > 0 else (slice(None, d), slice(-d, None))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def wall_zero(E, Ca, Cb):
    """Copies of E, Ca, Cb ((3, nx, ny, nz) bit patterns) with the lower-wall rule of alpha_fdtd.h section 2 applied."""
    E, Ca, Cb = E.copy(), Ca.copy(), Cb.copy()
    for a in range(3):
        for w in range(3):
            if w != a:
                sl = (a,) + tuple(0 if q == w else slice(None) for q in range(3))
                E[sl] = Ca[sl] = Cb[sl] = 0
    return E, Ca, Cb


def run(E, H, Ca, Cb, steps):
    """`steps` steps of the update on (3, nx, ny, nz) uint16 bit patterns (walls already applied); returns (E, H) bits."""
    Ev = [to_f64(E[a]) for a in range(3)]
    Hv = [to_f64(H[a]) for a in range(3)]
    Cav = [to_f64(Ca[a]) for a in range(3)]
    Cbv = [to_f64(Cb[a]) for a in range(3)]
    for _ in range(steps):
        for a in range(3):
            b, c = CYC[a]
            curl = _sub(_sub(_shift(Ev[c], b, +1), Ev[c]), _sub(_shift(Ev[b], c, +1), Ev[b]))
            Hv[a] = _sub(Hv[a], curl)
        for a in range(3):
            b, c = CYC[a]
            curl = _sub(_sub(Hv[c], _shift(Hv[c], b, -1)), _sub(Hv[b], _shift(Hv[b], c, -1)))
            Ev[a] = _fma(Cbv[a], curl, _mul(Cav[a], Ev[a]))
    return np.stack([round_bits(v) for v in Ev]), np.stack([round_bits(v) for v in Hv])


def mix64(x):
    x = np.asarray(x, np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def checksums(E, H):
    """The six checksums of afdtd_checksums (alpha_fdtd.h) of (3, nx, ny, nz) E and H bit patterns."""
    n = E[0].size
    idx = np.arange(n, dtype=np.uint64)
    out = []
    for c in range(6):
        bits = (E if c < 3 else H)[c % 3].ravel().astype(np.uint64)
        with np.errstate(over="ignore"):
            k = ((np.uint64(c % 3) * np.uint64(n) + idx) << np.uint64(32)) + bits
            out.append(int(np.add.reduce(mix64(k), dtype=np.uint64)))
    return tuple(out)
