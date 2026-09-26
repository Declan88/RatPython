"""
The hit marker: four short diagonal ticks around the screen centre that pop up
when a shot lands on another player and fade out fast (Call of Duty style).

The ticks are one small texture drawn once (see _build_texture - the UI's quads
are axis-aligned, so diagonal lines have to come from a texture), so a hit marker
costs a single quad per frame and nothing at all while it isn't showing.
"""

import time

import numpy as np
import pygame

from .widgets import Anchor, Widget, _color

_TEXTURE_SIZE = 128


def _build_texture():
    """A white X with a gap in the middle - four diagonal ticks - edged in
    black for contrast on any background, anti-aliased by distance."""
    n = _TEXTURE_SIZE
    half = n / 2.0
    axis = np.arange(n, dtype="f4") + 0.5 - half
    x, y = np.meshgrid(axis, axis)
    inner, outer, radius = 0.20 * half, 0.92 * half, 0.075 * half   # tick start/end distance from the centre, half thickness
    line = np.zeros((n, n), "f4")
    outline = np.zeros((n, n), "f4")
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        d = np.array((sx, sy), "f4") / np.sqrt(2.0)
        along = x * d[0] + y * d[1]
        across = np.abs(x * -d[1] + y * d[0])
        # Distance to the segment [inner, outer] along d (0 while inside it).
        past = np.maximum(np.maximum(inner - along, along - outer), 0.0)
        dist = np.hypot(across, past)
        line = np.maximum(line, np.clip(radius + 0.5 - dist, 0.0, 1.0))
        edge = np.clip(radius + 2.5 - dist, 0.0, 1.0)
        outline = np.maximum(outline, edge)
    rgba = np.zeros((n, n, 4), "u1")
    rgba[..., :3] = (line[..., None] * 255).astype("u1")            # white ticks on a black edge
    rgba[..., 3] = (np.maximum(line, outline * 0.6) * 255).astype("u1")
    return pygame.image.frombuffer(rgba.tobytes(), (n, n), "RGBA")


class Hitmarker(Widget):
    def __init__(self, duration=0.28, pop=1.25, color=(255, 255, 255, 255), **kw):
        kw.setdefault("anchor", Anchor.CENTER)
        kw.setdefault("size", (46, 46))
        super().__init__(**kw)
        self.duration = duration    # seconds it stays up
        self.pop = pop              # how much bigger than normal it starts (shrinks to 1)
        self.color = color
        self._start = None
        self._tex = None

    def trigger(self):
        """Shows it (restarting the fade if it's already up)."""
        self._start = time.perf_counter()

    def _release_resources(self):
        super()._release_resources()
        if self._tex is not None:
            self._tex.release()
            self._tex = None

    def draw(self, out):
        if self._start is not None:
            t = (time.perf_counter() - self._start) / self.duration
            if t >= 1.0:
                self._start = None
            elif self.manager is not None:
                if self._tex is None:
                    self._tex = self.manager.renderer.texture_from_surface(_build_texture())
                x, y, w, h = self.rect
                scale = 1.0 + (self.pop - 1.0) * (1.0 - t) ** 2      # settles quickly
                sw, sh = w * scale, h * scale
                r, g, b, a = self.color
                alpha = int(a * min(1.0, (1.0 - t) * 2.0))           # holds, then fades over the second half
                out.texture_logical(self._tex, (x + (w - sw) / 2, y + (h - sh) / 2, sw, sh), _color((r, g, b, alpha)))
        super().draw(out)
