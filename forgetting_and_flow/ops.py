"""Image operators, pure NumPy.

Everything the two systems do to an image is built from the handful of
functions in this file. None of them is clever; all of them are lossy in the
exact ways the technical appendix describes:

* ``sample``       — bilinear resampling. Information below one pixel is
                     averaged away and cannot be recovered.
* ``box_blur``     — neighbourhood averaging (the "mild diffusion").
* ``zoom``         — sub-pixel zoom about the centre, via ``sample``.
* ``hue_rotate``   — rotation of the chroma plane in YIQ space.
* ``unsharp``      — luminance-only unsharp mask (display grade, never in-loop).
* ``contrast``     — contrast expansion about mid-grey (display grade).

Arrays are float32 in [0, 1], shape (H, W) for single-channel fields and
(H, W, 3) for colour images.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

_RGB2YIQ = np.array(
    [[0.299, 0.587, 0.114],
     [0.596, -0.274, -0.322],
     [0.211, -0.523, 0.312]], dtype=np.float32)
_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an (H, W, 3) image -> (H, W)."""
    return rgb @ _LUMA


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return luminance(rgb) if rgb.ndim == 3 else rgb


def to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=-1) if gray.ndim == 2 else gray


def hue_rotate(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate hue by ``angle`` radians (YIQ chroma-plane rotation)."""
    if angle == 0.0:
        return rgb
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    m = _YIQ2RGB @ rot @ _RGB2YIQ
    return (rgb @ m.T).astype(np.float32)


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------

def grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinate grids (Y, X), float32, shape (H, W)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return y, x


def sample(img: np.ndarray, y: np.ndarray, x: np.ndarray, *, wrap: bool = True) -> np.ndarray:
    """Bilinear resample ``img`` at float coordinates (y, x).

    This is the one irreducibly lossy primitive of The Flow: every call
    averages four neighbours and discards whatever lived between them.
    """
    h, w = img.shape[:2]
    if wrap:
        x = np.mod(x, w)
        y = np.mod(y, h)
    else:
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    if wrap:
        x1 = (x0 + 1) % w
        y1 = (y0 + 1) % h
        x0 %= w
        y0 %= h
    else:
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
    # gather through flat indices: ~2x faster than 2-D fancy indexing in NumPy
    shape = y.shape + ((img.shape[2],) if img.ndim == 3 else ())
    flat = img.reshape(h * w, -1)
    i00 = (y0 * w + x0).ravel()
    i01 = (y0 * w + x1).ravel()
    i10 = (y1 * w + x0).ravel()
    i11 = (y1 * w + x1).ravel()
    a = np.take(flat, i00, axis=0)
    b = np.take(flat, i01, axis=0)
    c = np.take(flat, i10, axis=0)
    d = np.take(flat, i11, axis=0)
    fx = fx.reshape(-1, 1)
    fy = fy.reshape(-1, 1)
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    out = top + (bot - top) * fy
    return out.reshape(shape).astype(np.float32, copy=False)


def zoom(img: np.ndarray, factor: float) -> np.ndarray:
    """Zoom about the centre by ``factor`` (1.0 = identity). Sub-pixel values
    such as 1.002 are the intended use: each frame resamples a hair closer."""
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    y, x = grid(h, w)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    return sample(img, (y - cy) / factor + cy, (x - cx) / factor + cx, wrap=False)


def resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize to (h, w)."""
    sh, sw = img.shape[:2]
    if (sh, sw) == (h, w):
        return img
    y = (np.arange(h, dtype=np.float32) + 0.5) * (sh / h) - 0.5
    x = (np.arange(w, dtype=np.float32) + 0.5) * (sw / w) - 0.5
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return sample(img, yy, xx, wrap=False)


def resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
    sh, sw = img.shape[:2]
    yi = np.minimum((np.arange(h) * sh // h), sh - 1)
    xi = np.minimum((np.arange(w) * sw // w), sw - 1)
    return img[yi][:, xi]


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------

def box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur of radius ``radius`` (window 2r+1), edge-replicated."""
    if radius <= 0:
        return img
    out = img.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float32)
        zero = np.zeros_like(np.take(c, [0], axis=axis))
        c = np.concatenate([zero, c], axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = (hi - lo) / (2 * radius + 1)
    return out.astype(np.float32)


def gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient of a (H, W) field -> (d/dy, d/dx)."""
    gy = np.zeros_like(field)
    gx = np.zeros_like(field)
    gy[1:-1] = (field[2:] - field[:-2]) * 0.5
    gx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) * 0.5
    return gy, gx


def unsharp(gray: np.ndarray, radius: int, amount: float) -> np.ndarray:
    """Unsharp mask on a single channel."""
    if amount == 0.0:
        return gray
    return gray + amount * (gray - box_blur(gray, radius))


def contrast(img: np.ndarray, gain: float, pivot: float = 0.5) -> np.ndarray:
    """Expand contrast about ``pivot`` by ``gain`` and clamp."""
    return np.clip((img - pivot) * gain + pivot, 0.0, 1.0).astype(np.float32)


def mix(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """GLSL-style mix: (1-k)*a + k*b."""
    return ((1.0 - k) * a + k * b).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def from_uint8(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0)
