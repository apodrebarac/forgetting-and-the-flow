"""The Forgetting: forgetting as irreversible state.

A single-channel *memory mask* — white means remembered — updated each frame
by a rule that can only subtract:

    mask(t+1) = clamp( mask(t) - bites(t) - edge_creep(t), 0, 1 )

The mask is an accumulator in a one-way state machine. ``MemoryMask.step``
contains no operation that can increase it, and asserts as much after every
update. The image stream is multiplied by the mask; lost regions render as
void.

Two processes subtract from the mask:

* **bites** — sparse random blobs, reseeded for every image pair so the
  pattern never repeats;
* **edge creep** — erosion in proportion to how much void already surrounds a
  pixel, so loss compounds at its own margins.

The mask persists across image changes: damage incurred while one image is on
screen is inherited by every image that follows (path dependence).

Above the erosion a glitch layer keeps the surface unstable: horizontal slice
bands tear sideways in discrete stutters, a chunky ordered dither converts
tone to stipple, and a black-and-white contrast grade holds the material in a
harsh forensic register.

**Forgiveness is external.** ``MemoryMask.forgive`` is the only thing that
can restore the mask, and it is never called from inside the update rule. The
CLI exposes it as a gesture from outside (a signal file, a key). In the
terminal run it is withheld.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .ops import box_blur, contrast, resize_nearest, to_gray


class ForgivenessWithheld(RuntimeError):
    """Raised when forgiveness is requested during a run that has withheld it."""


#: measured front speed of edge creep, in pixels per frame per unit ``creep_rate``
#: (a straight void edge with creep_radius = 1; see tests/test_systems.py).
CREEP_FRONT_SPEED = 1.1


@dataclass
class ForgettingParams:
    # erosion, in human units (converted per canvas and frame rate by MemoryMask)
    bites_per_minute: float = 2.0      # new bites over the whole canvas, per minute
    creep_px_per_minute: float = 1.0   # how fast the edge of a void advances
    bite_radius: int = 3               # bite blob radius, pixels
    bite_depth: float = 1.0            # how much of the mask a bite removes (1 = to void)
    creep_radius: int = 1              # neighbourhood over which void is sensed
    # raw per-frame rates; when set they override the human-unit fields above
    bite_rate: Optional[float] = None  # per-pixel probability per frame of seeding a bite
    creep_rate: Optional[float] = None # mask lost per frame at a fully void-surrounded edge
    # glitch layer
    band_min: int = 3              # slice band heights, pixels
    band_max: int = 32
    stutter_prob: float = 0.06     # per-frame probability that a band re-tears
    tear_max: float = 0.08         # max tear offset, fraction of width
    tear_prob: float = 0.35        # fraction of bands that are torn at all
    dither_block: int = 3          # stipple grain, pixels
    dither_levels: int = 2         # grey levels after dither (2 = pure B&W)
    gamma: float = 1.0
    contrast_gain: float = 1.5

    def rates(self, h: int, w: int, fps: float) -> tuple[float, float]:
        """(bite_rate per pixel per frame, creep_rate per frame) for a canvas and timebase."""
        frames_per_minute = 60.0 * fps
        bite = (self.bite_rate if self.bite_rate is not None
                else self.bites_per_minute / frames_per_minute / (h * w))
        creep = (self.creep_rate if self.creep_rate is not None
                 else self.creep_px_per_minute / frames_per_minute / CREEP_FRONT_SPEED)
        return float(bite), float(creep)


def half_life_minutes(h: int, w: int, params: ForgettingParams, fps: float) -> float:
    """Rough expected time until half the canvas is void, from the rates alone.

    Bites arrive at ``b`` per frame and each wound grows at ``v`` px/frame from
    an initial radius ``r0``, so the void area after ``t`` frames is about
    ``b * pi * (r0^2 t + r0 v t^2 + v^2 t^3 / 3)`` (ignoring overlap, which
    only slows the last stretch). Solved for area = h*w/2. ``inf`` without bites.
    """
    bite_px, creep = params.rates(h, w, fps)
    b = bite_px * h * w
    if b <= 0.0:
        return float("inf")
    r0 = (2 * params.bite_radius + 1) / np.sqrt(np.pi)
    v = CREEP_FRONT_SPEED * creep
    target = 0.5 * h * w / (b * np.pi)
    roots = np.roots([v * v / 3.0, r0 * v, r0 * r0, -target])
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if not real:
        return float("inf")
    return min(real) / (60.0 * fps)


class MemoryMask:
    """The one-way accumulator. White (1.0) = remembered, 0.0 = void."""

    def __init__(self, h: int, w: int, params: ForgettingParams, seed: int, fps: float = 60.0):
        self.h, self.w = h, w
        self.p = params
        self.fps = fps
        self.bite_rate, self.creep_rate = params.rates(h, w, fps)
        self.mask = np.ones((h, w), np.float32)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.forgiven = 0
        self.withhold_forgiveness = False

    def reseed(self, seed: int) -> None:
        """New bite pattern (called for every image pair)."""
        self.rng = np.random.default_rng(seed)

    def _bites(self) -> np.ndarray:
        p = self.p
        if self.bite_rate <= 0.0:
            return np.zeros_like(self.mask)
        seeds = (self.rng.random(self.mask.shape) < self.bite_rate).astype(np.float32)
        if p.bite_radius > 0:
            blob = (box_blur(seeds, p.bite_radius) > 1e-6).astype(np.float32)
        else:
            blob = seeds
        return blob * p.bite_depth

    def _edge_creep(self) -> np.ndarray:
        p = self.p
        if self.creep_rate <= 0.0:
            return np.zeros_like(self.mask)
        void = 1.0 - self.mask
        neighbour_void = box_blur(void, p.creep_radius)
        return (self.creep_rate * neighbour_void * (self.mask > 0.0)).astype(np.float32)

    def step(self) -> np.ndarray:
        """mask(t+1) = clamp(mask(t) - bites - edge_creep, 0, 1). Monotone."""
        before = self.mask
        after = np.clip(before - self._bites() - self._edge_creep(), 0.0, 1.0).astype(np.float32)
        # Irreversibility by construction, and checked: nothing here may add.
        assert np.all(after <= before + 1e-7), "memory mask increased inside the update rule"
        self.mask = after
        self.updates += 1
        return self.mask

    # -- the one external gesture -------------------------------------------
    def forgive(self) -> None:
        """Restore the mask completely. Not part of the dynamics; an act from outside."""
        if self.withhold_forgiveness:
            raise ForgivenessWithheld("forgiveness is withheld for this run")
        self.mask = np.ones_like(self.mask)
        self.forgiven += 1

    # -- characterisation ----------------------------------------------------
    def remaining(self) -> float:
        return float(self.mask.mean())

    def void_fraction(self) -> float:
        return float((self.mask <= 0.0).mean())

    def edge_length(self) -> int:
        """Number of remembered pixels adjacent to void (the active front)."""
        m = self.mask > 0.5
        v = ~m
        nb = np.zeros_like(v)
        nb[1:] |= v[:-1]
        nb[:-1] |= v[1:]
        nb[:, 1:] |= v[:, :-1]
        nb[:, :-1] |= v[:, 1:]
        return int((m & nb).sum())

    def metrics(self) -> dict:
        return {
            "updates": self.updates,
            "remaining": self.remaining(),
            "void_fraction": self.void_fraction(),
            "edge_length": self.edge_length(),
            "forgiven": self.forgiven,
        }

    def describe(self) -> dict:
        return {
            "bite_rate_per_pixel_per_frame": self.bite_rate,
            "creep_rate_per_frame": self.creep_rate,
            "bites_per_minute": self.bite_rate * self.h * self.w * 60.0 * self.fps,
            "creep_px_per_minute": self.creep_rate * CREEP_FRONT_SPEED * 60.0 * self.fps,
            "half_life_minutes_estimate": half_life_minutes(self.h, self.w, self.p, self.fps),
        }


# ---------------------------------------------------------------------------
# surface degradation (glitch layer)
# ---------------------------------------------------------------------------

_BAYER4 = (np.array([[0, 8, 2, 10],
                     [12, 4, 14, 6],
                     [3, 11, 1, 9],
                     [15, 7, 13, 5]], np.float32) + 0.5) / 16.0


class Glitch:
    def __init__(self, h: int, w: int, params: ForgettingParams, seed: int):
        self.h, self.w = h, w
        self.p = params
        self.rng = np.random.default_rng(seed)
        self._bands: list[tuple[int, int]] = []
        self._offsets: np.ndarray = np.zeros(0, np.int32)
        self.retear()
        bh = (h + params.dither_block - 1) // params.dither_block
        bw = (w + params.dither_block - 1) // params.dither_block
        self._bayer = np.tile(_BAYER4, (bh // 4 + 1, bw // 4 + 1))[:bh, :bw]

    def retear(self) -> None:
        """Re-partition the image into slice bands (done on each image change)."""
        p = self.p
        bands, y = [], 0
        while y < self.h:
            hgt = int(self.rng.integers(p.band_min, p.band_max + 1))
            bands.append((y, min(self.h, y + hgt)))
            y += hgt
        self._bands = bands
        self._offsets = np.zeros(len(bands), np.int32)
        self._stutter(all_bands=True)

    def _stutter(self, all_bands: bool = False) -> None:
        p = self.p
        n = len(self._bands)
        hit = np.ones(n, bool) if all_bands else (self.rng.random(n) < p.stutter_prob)
        torn = self.rng.random(n) < p.tear_prob
        amp = int(p.tear_max * self.w)
        new = np.where(torn, self.rng.integers(-amp, amp + 1, n), 0).astype(np.int32)
        self._offsets = np.where(hit, new, self._offsets)

    def apply(self, gray: np.ndarray) -> np.ndarray:
        p = self.p
        self._stutter()
        out = gray.copy()
        for (y0, y1), off in zip(self._bands, self._offsets):
            if off:
                out[y0:y1] = np.roll(gray[y0:y1], int(off), axis=1)
        # tone grade
        out = np.power(np.clip(out, 0, 1), p.gamma)
        out = contrast(out, p.contrast_gain)
        # chunky ordered dither; the threshold tile is shifted every frame so
        # the stipple crawls instead of sitting still
        b = p.dither_block
        if b > 1:
            ph, pw = (-self.h) % b, (-self.w) % b
            padded = np.pad(out, ((0, ph), (0, pw)), mode="edge")
            small = padded.reshape(padded.shape[0] // b, b, padded.shape[1] // b, b).mean(axis=(1, 3))
        else:
            small = out
        levels = max(2, p.dither_levels)
        dy, dx = self.rng.integers(0, 4, 2)
        thr = np.roll(self._bayer[:small.shape[0], :small.shape[1]], (int(dy), int(dx)), axis=(0, 1))
        q = np.floor(small * (levels - 1) + thr) / (levels - 1)
        q = np.clip(q, 0, 1).astype(np.float32)
        if b > 1:
            q = resize_nearest(q, self.h, self.w)
        return q


class Forgetting:
    """The whole second system: mask + glitch, fed by the image stream."""

    def __init__(self, h: int, w: int, params: ForgettingParams, seed: int, fps: float = 60.0):
        self.h, self.w = h, w
        self.p = params
        self.seed = int(seed)
        self.memory = MemoryMask(h, w, params, seed, fps)
        self.glitch = Glitch(h, w, params, seed ^ 0x5F3759DF)
        self._pair_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
        self.frame = 0
        self.out = np.zeros((h, w), np.float32)

    def on_image_change(self) -> None:
        """Called for every new image pair: reseed bites, re-partition bands."""
        self.memory.reseed(int(self._pair_rng.integers(0, 2**63 - 1)))
        self.glitch.retear()

    def step(self, source: np.ndarray) -> np.ndarray:
        mask = self.memory.step()
        gray = to_gray(source)
        self.out = self.glitch.apply(gray * mask)   # lost regions render as void
        self.frame += 1
        return self.out

    def display(self) -> np.ndarray:
        return self.out

    def metrics(self) -> dict:
        m = self.memory.metrics()
        m["frame"] = self.frame
        return m

    def describe(self) -> dict:
        return {"system": "forgetting", "size": [self.h, self.w], "params": asdict(self.p),
                "rates": self.memory.describe()}
