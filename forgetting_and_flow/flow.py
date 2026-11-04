"""The Flow: forgetting as lossy recursion.

A video feedback loop. Each frame's output is written back as the next
frame's input. The loop body applies, in order:

1. a sub-pixel zoom;
2. advection by a curl-noise vector field;
3. self-refraction (displacement by the gradient of the image's own blurred
   luminance);
4. a mild diffusion;
5. a low-rate injection of the source image, ``out = mix(loop, source, k)``;
6. a slow hue rotation;
7. a clamp.

No buffer anywhere stores the original image; it exists only as a diminishing
contribution inside the recursion. With injection rate ``k`` the share of the
original left after ``N`` frames is

    contribution(N) = 1 - (1 - k) ** N

so with k = 0.06 an image is half-replaced in 11 frames and 99 % replaced in
75 (a second and a quarter at 60 fps). What survives is its statistics —
tones, energy, palette — kept in circulation indefinitely.

The display grade (luminance unsharp mask, contrast expansion) is applied
*outside* the loop, exactly once per frame on the way to the screen. Inside
the recursion it would compound and destabilise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .noise import CurlField
from .ops import (box_blur, contrast, gradient, grid, hue_rotate, luminance,
                  mix, sample, to_rgb, unsharp, zoom)


@dataclass
class FlowParams:
    k: float = 0.06                 # injection rate of the source per frame
    zoom: float = 1.002             # sub-pixel zoom factor per frame
    advect_strength: float = 1.2    # pixels/frame of curl-noise displacement
    noise_scale: float = 160.0      # feature size of the noise field, pixels
    noise_speed: float = 0.004      # evolution of the noise field per frame
    refract_strength: float = 3.0   # pixels of self-refraction per unit luma gradient
    refract_blur: int = 6           # blur radius of the luminance used to refract
    diffusion: float = 0.15         # weight of a 3x3 box blur blended in
    hue_speed: float = 0.002        # radians of hue rotation per frame
    display_unsharp_radius: int = 3
    display_unsharp_amount: float = 0.8
    display_contrast: float = 1.6

    @staticmethod
    def transport_off() -> "FlowParams":
        """Everything but the injection disabled — used to verify the decay formula."""
        return FlowParams(zoom=1.0, advect_strength=0.0, refract_strength=0.0,
                          diffusion=0.0, hue_speed=0.0)


def contribution(n: int, k: float) -> float:
    """Share of the original image remaining in the loop after n frames."""
    return 1.0 - (1.0 - k) ** n


def frames_to_replace(fraction: float, k: float) -> int:
    """Smallest N with contribution(N, k) >= fraction."""
    return int(np.ceil(np.log(1.0 - fraction) / np.log(1.0 - k)))


class Flow:
    def __init__(self, h: int, w: int, params: FlowParams, seed: int):
        self.h, self.w = h, w
        self.p = params
        self.seed = int(seed)
        self.buf = np.zeros((h, w, 3), np.float32)
        self.frame = 0
        self._Y, self._X = grid(h, w)
        self.field = CurlField(h, w, seed=self.seed, scale=params.noise_scale,
                               speed=params.noise_speed, strength=params.advect_strength)

    # -- the loop body -----------------------------------------------------
    def step(self, source: np.ndarray) -> np.ndarray:
        p = self.p
        buf = self.buf
        # 1 + 2: sub-pixel zoom and advection are composed into a single
        # backward-mapped resample (one lossy pass instead of two)
        if p.zoom != 1.0 or p.advect_strength != 0.0:
            y, x = self._Y, self._X
            if p.zoom != 1.0:
                cy, cx = (self.h - 1) / 2.0, (self.w - 1) / 2.0
                y = (y - cy) / p.zoom + cy
                x = (x - cx) / p.zoom + cx
            if p.advect_strength != 0.0:
                vy, vx = self.field.velocity(self.frame)
                y = y - vy                                           # semi-Lagrangian advection
                x = x - vx
            buf = sample(buf, y, x, wrap=False)
        if p.refract_strength != 0.0:
            lum = box_blur(luminance(buf), p.refract_blur)
            gy, gx = gradient(lum)
            buf = sample(buf, self._Y + gy * p.refract_strength,
                         self._X + gx * p.refract_strength)          # self-refraction
        if p.diffusion != 0.0:
            buf = mix(buf, box_blur(buf, 1), p.diffusion)
        buf = mix(buf, to_rgb(source), p.k)                          # injection
        if p.hue_speed != 0.0:
            buf = hue_rotate(buf, p.hue_speed)
        self.buf = np.clip(buf, 0.0, 1.0).astype(np.float32)
        self.frame += 1
        return self.buf

    # -- display grade, outside the recursion ------------------------------
    def display(self) -> np.ndarray:
        p = self.p
        lum = luminance(self.buf)
        sharp = unsharp(lum, p.display_unsharp_radius, p.display_unsharp_amount)
        out = self.buf + (sharp - lum)[..., None]
        return contrast(out, p.display_contrast)

    # -- characterisation --------------------------------------------------
    def metrics(self, source: np.ndarray, frames_since_enter: int) -> dict:
        a = luminance(self.buf).ravel()
        b = luminance(to_rgb(source)).ravel()
        sa, sb = a.std(), b.std()
        corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 1e-6 and sb > 1e-6 else 0.0
        return {
            "frame": self.frame,
            "corr_to_source": corr,
            "contribution_theory": contribution(frames_since_enter, self.p.k),
            "buffer_mean": float(self.buf.mean()),
            "buffer_std": float(self.buf.std()),
            "buffer_entropy_bits": _entropy_bits(a),
        }

    def describe(self) -> dict:
        return {"system": "flow", "size": [self.h, self.w], "params": asdict(self.p)}


def _entropy_bits(x: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(np.clip(x, 0, 1), bins=bins, range=(0.0, 1.0))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
