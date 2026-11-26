# SPDX-License-Identifier: MIT
"""Ca and Cb for alpha-fdtd from permittivity, conductivity and dt (alpha_fdtd.h section 1, UNITS).

Cell size 1, c = 1 in vacuum, mu = 1, C_H = dt.  For a cell of relative permittivity eps and conductivity sigma:
    Ca = (1 - sigma dt / (2 eps)) / (1 + sigma dt / (2 eps))
    Cb = (dt * dt / eps)          / (1 + sigma dt / (2 eps))      (absorbed: the textbook dt / eps, times C_H = dt)
Stability in vacuum needs dt <= 1/sqrt(3), i.e. Cb <= 1/3.  dt = 1/2 (a power of two) gives Cb = 1/4 in vacuum and
makes the absorbed form bit-identical to the textbook form.  Copy this function into your code; it is not part of the
alpha_fdtd package (the package does no numerics).
"""
import numpy as np


def yee_coefficients(eps, sigma=0.0, dt=0.5):
    """(Ca, Cb) as float32 arrays of the broadcast shape of eps and sigma, Cb in the absorbed form."""
    eps = np.asarray(eps, np.float64)
    s = np.asarray(sigma, np.float64) * dt / (2.0 * eps)
    ca = (1.0 - s) / (1.0 + s)
    cb = (dt * dt / eps) / (1.0 + s)
    return ca.astype(np.float32), cb.astype(np.float32)


def graded_absorber(shape, thickness, sigma_max, dt=0.5, eps=1.0):
    """(Ca, Cb) of shape (3, nx, ny, nz): conductivity rising as ((distance into the layer) / thickness)^2 towards
    all six faces, sigma_max at the face.  A simple absorber for the closed PEC box; its reflection depends on the
    thickness and the grading, so check it for your problem."""
    d = np.zeros(shape)
    for ax, n in enumerate(shape):
        i = np.arange(n, dtype=np.float64)
        depth = np.maximum(np.maximum(thickness - i, i - (n - 1 - thickness)), 0.0) / thickness
        d = np.maximum(d, depth.reshape([-1 if a == ax else 1 for a in range(3)]))
    ca, cb = yee_coefficients(eps, sigma_max * d ** 2, dt)
    return np.broadcast_to(ca, (3,) + tuple(shape)), np.broadcast_to(cb, (3,) + tuple(shape))


if __name__ == "__main__":
    for eps, sigma in [(1.0, 0.0), (4.0, 0.0), (2.25, 0.1)]:
        ca, cb = yee_coefficients(eps, sigma)
        print(f"eps {eps:5.2f}  sigma {sigma:4.2f}  ->  Ca {float(ca):.6f}  Cb {float(cb):.6f}")
