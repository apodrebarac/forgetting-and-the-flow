"""The event log is the artefact.

Neither system keeps frames. What a run leaves behind is this append-only
JSON-lines file: when the run began, what it was given, what it did to it, and
when it stopped. In *performance* mode the random seed is drawn from the
operating system and deliberately **not** recorded, so the run cannot be
replayed — the system itself has no memory of its former appearance. In
*study* mode a seed is given and recorded so that the dynamics can be
characterised.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import secrets
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    return secrets.token_hex(8)


def choose_seed(mode: str, seed: Optional[int]) -> tuple[int, Optional[int]]:
    """Return (seed_used, seed_to_record).

    performance: fresh entropy, not recorded.
    study:       the given seed (or a fresh one), recorded.
    """
    if mode == "performance":
        return secrets.randbits(63), None
    s = secrets.randbits(63) if seed is None else int(seed)
    return s, s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EventLog:
    """Append-only JSONL log. One line per event. Flushed on every write."""

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> dict:
        rec = {"t": utcnow(), "run": self.run_id, "event": event}
        rec.update(_jsonable(fields))
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        return rec

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    @staticmethod
    def environment() -> dict:
        return {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def read_log(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
