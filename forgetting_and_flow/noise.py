"""Smooth noise and the divergence-free velocity field of The Flow.

The TouchDesigner original uses GLSL simplex noise. Here a 3-D (x, y, t)
lattice value-noise is computed at low resolution and bilinearly upsampled —
cheap in NumPy, smooth, and continuous in time. The velocity field is the
*curl* of that scalar field (its gradient rotated 90 degrees), which makes it
divergence-free: the churn swirls without any net drift and without sinks that
would drain the canvas. See ``tests/test_noise.py`` for the divergence check.
"""
from __future__ import annotations

import numpy as np

from .ops import gradient, resize


def _hash3(ix: np.ndarray, iy: np.ndarray, it: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic integer hash of lattice coordinates -> float in [0, 1)."""
    h = (ix.astype(np.int64) * 374761393
         + iy.astype(np.int64) * 668265263
         + it.astype(np.int64) * 2147483647
         + np.int64(seed) * 1013904223)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return ((h & 0xFFFFFF).astype(np.float32)) / float(0x1000000)


def _smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def value_noise_3d(y: np.ndarray, x: np.ndarray, t: float, seed: int) -> np.ndarray:
    """Trilinearly interpolated lattice noise, values in [0, 1]."""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    t0 = int(np.floor(t))
    fx = _smooth((x - x0).astype(np.float32))
    fy = _smooth((y - y0).astype(np.float32))
    ft = _smooth(np.float32(t - t0))
    t0a = np.full_like(x0, t0)
    t1a = t0a + 1

    def corner(dx, dy, ta):
        return _hash3(x0 + dx, y0 + dy, ta, seed)

    def plane(ta):
        a = corner(0, 0, ta) * (1 - fx) + corner(1, 0, ta) * fx
        b = corner(0, 1, ta) * (1 - fx) + corner(1, 1, ta) * fx
        return a * (1 - fy) + b * fy

    return (plane(t0a) * (1 - ft) + plane(t1a) * ft).astype(np.float32)


def fbm_3d(y, x, t, seed, octaves: int = 3, lacunarity: float = 2.0, gain: float = 0.5):
    """Fractional Brownian motion: a few octaves of value noise, centred on 0."""
    out = np.zeros_like(x, dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for i in range(octaves):
        out += amp * (value_noise_3d(y * freq, x * freq, t * freq, seed + 97 * i) - 0.5)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return out / norm


class CurlField:
    """Time-varying divergence-free velocity field on an (h, w) canvas.

    ``scale`` is the noise feature size in pixels; ``speed`` is how fast the
    field evolves per frame; ``strength`` is the displacement in pixels per
    frame. The scalar field is evaluated on a grid ``down``× smaller than the
    canvas and upsampled.
    """

    def __init__(self, h: int, w: int, *, seed: int, scale: float = 160.0,
                 speed: float = 0.004, strength: float = 1.2, down: int = 8,
                 octaves: int = 3):
        self.h, self.w = h, w
        self.seed = int(seed)
        self.scale = float(scale)
        self.speed = float(speed)
        self.strength = float(strength)
        self.octaves = octaves
        lh, lw = max(4, h // down), max(4, w // down)
        ly, lx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        self._ly = ly * (h / lh) / self.scale
        self._lx = lx * (w / lw) / self.scale
        self._down = (h / lh)

    def potential(self, frame: int) -> np.ndarray:
        """Scalar stream-function psi at full resolution."""
        t = frame * self.speed
        psi = fbm_3d(self._ly, self._lx, t, self.seed, octaves=self.octaves)
        return resize(psi, self.h, self.w)

    def velocity(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """(vy, vx) in pixels/frame: the curl of psi, i.e. grad(psi) rotated 90°."""
        psi = self.potential(frame)
        gy, gx = gradient(psi)
        # curl of a 2-D scalar potential: v = (d psi/dx, -d psi/dy) -> rotated gradient
        vy, vx = gx, -gy
        # normalise so that ``strength`` is the typical displacement in pixels
        mag = float(np.sqrt(np.mean(vy * vy + vx * vx)) + 1e-8)
        s = self.strength / mag
        return (vy * s).astype(np.float32), (vx * s).astype(np.float32)


def divergence(vy: np.ndarray, vx: np.ndarray) -> np.ndarray:
    dvy_dy, _ = gradient(vy)
    _, dvx_dx = gradient(vx)
    return dvy_dy + dvx_dx
