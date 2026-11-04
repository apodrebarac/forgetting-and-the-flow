"""The terminal run: the archive consumes itself.

Adds one gesture to everything else: each image, every time it surfaces, is
written back to disk more degraded than it was loaded. A copy of a copy of a
copy. Generation by generation the files hold less, until they hold noise,
and then nothing.

What this module does and does not do, precisely:

* It **overwrites** image files in the archive directory you give it, in
  place, keeping their names and formats. Each write-back blends the file as
  loaded with what the running system(s) made of it, at ``generation_mix``;
  JPEG files are additionally re-encoded at a quality that falls with each
  generation. The next time the deck deals that image, the degraded file is
  what gets loaded.
* It **never deletes** anything. Not the images, not backups, not the log.
  The paper reserves the deletion of the last backup for the artist's own
  hand, before the run begins, because that responsibility cannot be
  automated. This code keeps that boundary.
* It **refuses to run** unless told explicitly, with ``--consume``, that the
  directory it is pointed at is meant to be consumed.
* Forgiveness is withheld: the signal file and the key do nothing but log.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .archive import PlayerEvent, load_rgb
from .eventlog import sha256_file
from .ops import mix, resize, to_gray, to_rgb, to_uint8


@dataclass
class ConsumerParams:
    generation_mix: float = 0.5     # share of the system's output written into the file per generation
    jpeg_quality_start: int = 75
    jpeg_quality_step: int = 3      # quality lost per generation (floor 1)


class Consumer:
    """Listener on the run's player: writes each image back as it leaves."""

    def __init__(self, params: ConsumerParams):
        self.p = params
        self.generations: dict[str, int] = {}

    def attach(self, run) -> None:
        run.listeners.append(self.on_event)
        run.log.write("consumer", params=self.p.__dict__,
                      note="files in the archive will be overwritten in place; nothing is deleted")

    # The outgoing image is still fully visible at the moment the next one
    # *enters* (the crossfade starts after this event), so that is when its
    # degraded appearance is captured and written home.
    def on_event(self, ev: PlayerEvent, run) -> None:
        if ev.kind != "enter" or ev.index == run.player.cur:
            return
        idx = run.player.cur
        path = run.paths[idx]
        gen = self.generations.get(str(path), 0) + 1
        self.generations[str(path)] = gen
        stats = self.write_back(path, run.composite_for_file(), gen)
        run.player.reload_current()  # the slot now holds what the file now holds
        run.log.write("generation", file=path.name, generation=gen, **stats)

    def write_back(self, path: Path, degraded: np.ndarray, generation: int) -> dict:
        with Image.open(path) as im:
            fmt = im.format or "PNG"
            size = im.size  # (w, h)
            mode = im.mode
        loaded = load_rgb(path)
        deg = resize(to_rgb(degraded), size[1], size[0])
        out = mix(loaded, deg, self.p.generation_mix)
        arr = to_uint8(out)
        if mode in ("L", "1", "I;16", "I"):
            img = Image.fromarray(to_uint8(to_gray(out)), "L")
        else:
            img = Image.fromarray(arr, "RGB")
        save_kwargs = {}
        if fmt.upper() in ("JPEG", "JPG"):
            q = max(1, self.p.jpeg_quality_start - self.p.jpeg_quality_step * (generation - 1))
            save_kwargs = {"quality": int(q)}
            fmt = "JPEG"
        # atomic replace: the file is never half-written
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".gen_", suffix=path.suffix)
        os.close(fd)
        try:
            img.save(tmp, format=fmt, **save_kwargs)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        g = to_gray(out)
        return {
            "mean": float(g.mean()),
            "std": float(g.std()),
            "entropy_bits": _entropy_bits(g),
            "sha256": sha256_file(path),
            **({"jpeg_quality": save_kwargs["quality"]} if save_kwargs else {}),
        }


def _entropy_bits(x: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(np.clip(x, 0, 1), bins=bins, range=(0.0, 1.0))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
