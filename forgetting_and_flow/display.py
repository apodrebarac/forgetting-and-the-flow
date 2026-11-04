"""Optional live window (tkinter, standard library). Keys: ``f`` forgive, ``q`` quit."""
from __future__ import annotations

import numpy as np

from .ops import to_uint8


class Window:
    def __init__(self, run, title: str = "The Forgetting and the Flow"):
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except Exception as e:  # pragma: no cover
            raise RuntimeError("live display needs tkinter and Pillow's ImageTk; "
                               "run without --display or install python-tk") from e
        self._tk, self._Image, self._ImageTk = tk, Image, ImageTk
        self.run = run
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        self.label = tk.Label(self.root, bg="black", bd=0)
        self.label.pack()
        self._photo = None
        self._open = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key>", self._on_key)

    def _on_close(self) -> None:
        self._open = False

    def _on_key(self, ev) -> None:
        if ev.char == "q":
            self._open = False
        elif ev.char == "f":
            self.run.forgive("key-f")

    def show(self, rgb: np.ndarray) -> bool:
        if not self._open:
            return False
        img = self._Image.fromarray(to_uint8(rgb))
        self._photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:
            self._open = False
        return self._open

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
