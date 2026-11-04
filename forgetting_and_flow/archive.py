"""The archive, the deck, and the two loader slots.

A finite folder of images is the only input either system has. It is dealt as
a shuffled deck with no immediate repeats, advanced by a timer whose
crossfades ping-pong between two loader slots so the visible image never
reloads mid-view.

The archive is never bundled with this code: it is private, and in the final
form it is consumed. ``make_demo_archive`` writes synthetic images so the
pipeline can be exercised without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image

from .ops import from_uint8, resize

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(root: Path, exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"archive directory not found: {root}")
    exts = {e.lower() for e in exts}
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"no images in {root}")
    return paths


def load_rgb(path: Path) -> np.ndarray:
    """Load any image as float32 RGB in [0, 1]."""
    with Image.open(path) as im:
        return from_uint8(np.asarray(im.convert("RGB")))


def cover_fit(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize to cover (h, w) preserving aspect, then centre-crop."""
    sh, sw = img.shape[:2]
    scale = max(h / sh, w / sw)
    nh, nw = max(h, int(round(sh * scale))), max(w, int(round(sw * scale)))
    out = resize(img, nh, nw)
    y0 = (nh - h) // 2
    x0 = (nw - w) // 2
    return np.ascontiguousarray(out[y0:y0 + h, x0:x0 + w])


class Deck:
    """A shuffled deck over ``n`` cards, reshuffled when exhausted, with the
    guarantee that the same card is never dealt twice in a row."""

    def __init__(self, n: int, rng: np.random.Generator):
        if n < 1:
            raise ValueError("deck needs at least one card")
        self.n = n
        self.rng = rng
        self._order: list[int] = []
        self._last: Optional[int] = None
        self.dealt = 0

    def _reshuffle(self) -> None:
        order = list(self.rng.permutation(self.n))
        if self.n > 1 and self._last is not None and order[0] == self._last:
            # move the offending first card somewhere else in the deck
            j = int(self.rng.integers(1, self.n))
            order[0], order[j] = order[j], order[0]
        self._order = order

    def next(self) -> int:
        if not self._order:
            self._reshuffle()
        card = int(self._order.pop(0))
        self._last = card
        self.dealt += 1
        return card


@dataclass
class PlayerEvent:
    kind: str          # "enter" | "surfaced" | "exit"
    index: int
    path: Path
    frame: int


@dataclass
class Player:
    """Timer-driven crossfader between two loader slots.

    ``hold_frames`` is how long an image is held fully visible; ``fade_frames``
    is the crossfade length. ``source()`` returns the current blended image.
    Listeners receive ``PlayerEvent``s: *enter* when an image starts fading
    in, *surfaced* when it is fully visible, *exit* when it is fully gone.
    """

    paths: list[Path]
    h: int
    w: int
    rng: np.random.Generator
    hold_frames: int = 240
    fade_frames: int = 90
    loader: Callable[[Path], np.ndarray] = load_rgb
    listeners: list[Callable[[PlayerEvent], None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.deck = Deck(len(self.paths), self.rng)
        self.frame = 0
        self._phase = "hold"  # "hold" | "fade"
        self._phase_frame = 0
        self.cur = self.deck.next()
        self.nxt: Optional[int] = None
        self._slots = [self._load(self.cur), None]  # ping-pong slots
        self._cur_slot = 0
        self._emit("enter", self.cur)
        self._emit("surfaced", self.cur)

    # -- loading -----------------------------------------------------------
    def _load(self, index: int) -> np.ndarray:
        return cover_fit(self.loader(self.paths[index]), self.h, self.w)

    def reload_current(self) -> None:
        """Re-read the current image from disk (used by the terminal run after
        a write-back, so the next appearance is of the degraded file)."""
        self._slots[self._cur_slot] = self._load(self.cur)

    # -- events ------------------------------------------------------------
    def _emit(self, kind: str, index: int) -> None:
        ev = PlayerEvent(kind, index, self.paths[index], self.frame)
        for fn in self.listeners:
            fn(ev)

    # -- time --------------------------------------------------------------
    @property
    def fade_alpha(self) -> float:
        if self._phase != "fade":
            return 0.0
        return min(1.0, self._phase_frame / max(1, self.fade_frames))

    def source(self) -> np.ndarray:
        a = self._slots[self._cur_slot]
        if self._phase == "fade" and self.nxt is not None:
            b = self._slots[1 - self._cur_slot]
            t = self.fade_alpha
            return ((1.0 - t) * a + t * b).astype(np.float32)
        return a

    def step(self) -> np.ndarray:
        """Advance one frame and return the source image for this frame."""
        self.frame += 1
        self._phase_frame += 1
        if self._phase == "hold" and self._phase_frame >= self.hold_frames:
            self.nxt = self.deck.next()
            self._slots[1 - self._cur_slot] = self._load(self.nxt)
            self._phase = "fade"
            self._phase_frame = 0
            self._emit("enter", self.nxt)
        elif self._phase == "fade" and self._phase_frame >= self.fade_frames:
            old = self.cur
            self.cur = self.nxt  # type: ignore[assignment]
            self.nxt = None
            self._cur_slot = 1 - self._cur_slot
            self._slots[1 - self._cur_slot] = None
            self._phase = "hold"
            self._phase_frame = 0
            self._emit("exit", old)
            self._emit("surfaced", self.cur)
        return self.source()


# ---------------------------------------------------------------------------
# synthetic stand-in archive
# ---------------------------------------------------------------------------

def make_demo_archive(out: Path, n: int = 12, size: int = 384, seed: int = 0) -> list[Path]:
    """Write ``n`` synthetic images (soft shapes, gradients, text-like bars).

    These stand in for the private archive during development and tests. They
    are not the work's material and carry no meaning.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    paths = []
    for i in range(n):
        img = np.zeros((size, size, 3), np.float32)
        base = rng.random(3) * 0.5
        img += base
        for _ in range(int(rng.integers(3, 8))):
            cy, cx, r = rng.random(), rng.random(), rng.uniform(0.05, 0.35)
            col = rng.random(3)
            blob = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r)))
            img += blob[..., None] * (col - base) * rng.uniform(0.4, 1.0)
        if rng.random() < 0.6:  # horizontal "lines of text"
            rows = (np.sin(y * size / rng.uniform(4, 9)) > 0.3).astype(np.float32)
            cols = (rng.random((1, size)) < 0.7).astype(np.float32)
            img *= 1.0 - 0.6 * (rows * cols)[..., None]
        img = np.clip(img, 0, 1)
        p = out / f"demo_{i:03d}.png"
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(p)
        paths.append(p)
    return paths
