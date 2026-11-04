"""The run loop: deck -> loader slots -> system(s) -> display, logged.

A ``Run`` owns one random generator (seeded once, per run), a ``Player`` over
the archive, one or both systems, and the event log. It advances them frame by
frame, writes metrics to the log at intervals, optionally saves frames to
disk, optionally shows a live window, and watches for the one external gesture
— forgiveness — which arrives as a signal file ``FORGIVE`` in the run
directory (or the ``f`` key in the live window). The run loop itself never
calls ``forgive``; it only relays a gesture that came from outside.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .archive import Player, PlayerEvent, list_images
from .eventlog import EventLog, choose_seed, new_run_id, sha256_file
from .flow import Flow, FlowParams
from .forgetting import Forgetting, ForgettingParams, ForgivenessWithheld
from .ops import to_rgb, to_uint8

FORGIVE_SIGNAL = "FORGIVE"


@dataclass
class RunConfig:
    archive: Path
    out_dir: Path
    systems: tuple[str, ...] = ("forgetting",)       # any of "flow", "forgetting"
    size: int = 512                                    # canvas height; width follows aspect
    aspect: float = 1.0                                # width / height
    fps: float = 60.0                                  # nominal frame rate (timebase)
    hold_seconds: float = 4.0                          # time an image is held fully visible
    fade_seconds: float = 1.5                          # crossfade length
    mode: str = "performance"                          # "performance" | "study"
    seed: Optional[int] = None                         # study mode only
    frames: Optional[int] = None                       # stop after N frames (None = run until seconds/stop)
    seconds: Optional[float] = None                    # stop after wall-clock seconds
    log_every: int = 30                                # frames between metric records
    save_every: int = 0                                # frames between saved PNG frames (0 = never)
    display: bool = False                              # live tkinter window
    pace: bool = False                                 # sleep to hold nominal fps when running ahead
    rebirth_seconds: Optional[float] = None            # optional long-installation rebirth cycle
    withhold_forgiveness: bool = False                 # the terminal run
    flow: FlowParams = field(default_factory=FlowParams)
    forgetting: ForgettingParams = field(default_factory=ForgettingParams)
    label: str = ""


class Run:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.run_id = new_run_id()
        seed_used, seed_recorded = choose_seed(cfg.mode, cfg.seed)
        self.rng = np.random.default_rng(seed_used)
        self.h = int(cfg.size)
        self.w = int(round(cfg.size * cfg.aspect))
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = EventLog(self.out_dir / "events.jsonl", self.run_id)

        self.paths = list_images(cfg.archive)
        self.player = Player(self.paths, self.h, self.w, self.rng,
                             hold_frames=max(1, int(cfg.hold_seconds * cfg.fps)),
                             fade_frames=max(1, int(cfg.fade_seconds * cfg.fps)))
        self.player.listeners.append(self._on_player_event)

        self.flow: Optional[Flow] = None
        self.forgetting: Optional[Forgetting] = None
        if "flow" in cfg.systems:
            self.flow = Flow(self.h, self.w, cfg.flow, int(self.rng.integers(0, 2**31 - 1)))
        if "forgetting" in cfg.systems:
            self.forgetting = Forgetting(self.h, self.w, cfg.forgetting,
                                         int(self.rng.integers(0, 2**31 - 1)), fps=cfg.fps)
            self.forgetting.memory.withhold_forgiveness = cfg.withhold_forgiveness

        self.frame = 0
        self._image_enter_frame = 0
        self._stop = False
        self._window = None
        self.listeners: list[Callable[[PlayerEvent, "Run"], None]] = []
        self._t0 = time.monotonic()
        self._rebirth_frames = (int(cfg.rebirth_seconds * cfg.fps)
                                if cfg.rebirth_seconds else None)

        self.log.write("run_start",
                       label=cfg.label,
                       mode=cfg.mode,
                       seed=("withheld (performance)" if seed_recorded is None else seed_recorded),
                       archive=str(cfg.archive), archive_images=len(self.paths),
                       canvas=[self.h, self.w], fps=cfg.fps,
                       hold_frames=self.player.hold_frames, fade_frames=self.player.fade_frames,
                       systems=[s.describe() for s in self._systems()],
                       forgiveness=("withheld" if cfg.withhold_forgiveness else "external gesture"),
                       environment=EventLog.environment())
        # replay the initial events the player emitted before we were listening
        self.log.write("image", kind="surfaced", index=self.player.cur,
                       file=self.paths[self.player.cur].name, frame=0)

    # -- plumbing ------------------------------------------------------------
    def _systems(self):
        return [s for s in (self.flow, self.forgetting) if s is not None]

    def _on_player_event(self, ev: PlayerEvent) -> None:
        if self.frame == 0:
            return  # construction-time events; logged in run_start
        if ev.kind == "enter":
            if self.forgetting is not None:
                self.forgetting.on_image_change()
        if ev.kind == "surfaced":
            self._image_enter_frame = self.frame
        self.log.write("image", kind=ev.kind, index=ev.index, file=ev.path.name, frame=self.frame)
        for fn in self.listeners:
            fn(ev, self)

    # -- the one external gesture ----------------------------------------------
    def forgive(self, origin: str) -> bool:
        """Relay a forgiveness gesture that came from outside the dynamics."""
        if self.forgetting is None:
            self.log.write("forgive", origin=origin, result="no memory mask in this run")
            return False
        try:
            self.forgetting.memory.forgive()
        except ForgivenessWithheld as e:
            self.log.write("forgive", origin=origin, result="withheld", detail=str(e))
            return False
        self.log.write("forgive", origin=origin, result="restored", frame=self.frame)
        return True

    def _check_signal(self) -> None:
        sig = self.out_dir / FORGIVE_SIGNAL
        if sig.exists():
            try:
                sig.unlink()
            except OSError:
                pass
            self.forgive("signal-file")

    # -- output ----------------------------------------------------------------
    def composite(self) -> np.ndarray:
        views = [to_rgb(s.display()) for s in self._systems()]
        return views[0] if len(views) == 1 else np.concatenate(views, axis=1)

    def composite_for_file(self) -> np.ndarray:
        """What the terminal run writes home: the system's view of the image.
        With both systems running, the two views are averaged."""
        views = [to_rgb(s.display()) for s in self._systems()]
        return views[0] if len(views) == 1 else np.mean(views, axis=0).astype(np.float32)

    def save_frame(self, tag: str = "") -> Path:
        p = self.out_dir / f"frame_{self.frame:08d}{tag}.png"
        Image.fromarray(to_uint8(self.composite())).save(p)
        return p

    def metrics(self, source: np.ndarray) -> dict:
        m: dict = {"frame": self.frame, "elapsed_s": round(time.monotonic() - self._t0, 3),
                   "image": self.paths[self.player.cur].name}
        if self.flow is not None:
            m["flow"] = self.flow.metrics(source, self.frame - self._image_enter_frame)
        if self.forgetting is not None:
            m["forgetting"] = self.forgetting.metrics()
        return m

    # -- stepping ---------------------------------------------------------------
    def step(self) -> np.ndarray:
        self.frame += 1
        source = self.player.step()
        for s in self._systems():
            s.step(source)
        if self._rebirth_frames and self.frame % self._rebirth_frames == 0:
            self.forgive("rebirth-cycle")
        self._check_signal()
        if self.cfg.log_every and self.frame % self.cfg.log_every == 0:
            self.log.write("metrics", **self.metrics(source))
        if self.cfg.save_every and self.frame % self.cfg.save_every == 0:
            self.save_frame()
        return source

    def stop(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        if self._stop:
            return True
        if self.cfg.frames is not None and self.frame >= self.cfg.frames:
            return True
        if self.cfg.seconds is not None and (time.monotonic() - self._t0) >= self.cfg.seconds:
            return True
        return False

    def run(self) -> dict:
        cfg = self.cfg
        if cfg.display:
            from .display import Window
            self._window = Window(self, title=" + ".join(cfg.systems))
        frame_time = 1.0 / cfg.fps if cfg.fps > 0 else 0.0
        reason = "complete"
        try:
            while not self._should_stop():
                t = time.monotonic()
                source = self.step()
                if self._window is not None:
                    if not self._window.show(self.composite()):
                        reason = "window closed"
                        break
                if cfg.pace or cfg.display:
                    dt = time.monotonic() - t
                    if dt < frame_time:
                        time.sleep(frame_time - dt)
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            final = self.metrics(self.player.source())
            self.log.write("run_end", reason=reason, frames=self.frame,
                           elapsed_s=round(time.monotonic() - self._t0, 3), final=final)
            self.log.close()
            if self._window is not None:
                self._window.close()
        return final
