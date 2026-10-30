"""
Damage numbers: when one of our hits lands on another player, the damage pops up at the hit point, drifts
upward and fades. Headshots use the hitmarker's headshot red.

Like NameTags, each number is an ordinary Label re-projected from its world position every frame, so it reuses the
UI renderer's text cache and batching. Labels are pooled (at most MAX_LIVE on screen; a new hit reuses the oldest),
so rapid fire never allocates widgets or grows without bound.
"""

import random
import time

import glm

from .hitmarker import Hitmarker
from .widgets import Anchor, Label, Panel

LIFETIME = 0.8            # seconds a number stays up
FADE = 0.35               # ...the last this many of them fading out
RISE = 45.0               # logical px it drifts up over its lifetime
JITTER = 22.0             # logical px of random sideways spread, so rapid hits don't stack on one spot
MAX_LIVE = 24
FONT_SIZE = 30
COLOR = (255, 255, 255)
HEADSHOT_COLOR = Hitmarker.HEADSHOT_COLOR[:3]


def _format(amount):
    rounded = round(amount, 1)
    return f"{rounded:.0f}" if rounded == int(rounded) else f"{rounded:.1f}"


class DamageNumbers:
    def __init__(self, ui):
        self.ui = ui
        self.container = ui.root.add(Panel(size_frac=(1, 1), name="damage_numbers"))
        self._entries = []    # [label, world position, start time, sideways jitter, rgb], oldest first
        self._free = []       # pooled labels not on screen

    @property
    def visible(self):
        return self.container.visible

    @visible.setter
    def visible(self, value):
        self.container.visible = value

    def add(self, position, amount, headshot=False):
        """A number for `amount` damage at world `position` (where the hit landed)."""
        if len(self._entries) >= MAX_LIVE:
            label = self._entries.pop(0)[0]
        elif self._free:
            label = self._free.pop()
        else:
            label = self.container.add(Label("", font_size=FONT_SIZE, anchor=Anchor.TOP_LEFT, pivot=(0.5, 1.0),
                                             visible=False))
        label.text = _format(amount)
        rgb = HEADSHOT_COLOR if headshot else COLOR
        self._entries.append([label, glm.vec3(position), time.perf_counter(), random.uniform(-JITTER, JITTER), rgb])

    def clear(self):
        for entry in self._entries:
            entry[0].visible = False
            self._free.append(entry[0])
        self._entries.clear()

    def update(self, camera, window_size):
        """Call once a frame: moves each number to its hit point's current place on screen, drifting and fading."""
        if not self._entries:
            return
        now = time.perf_counter()
        while self._entries and now - self._entries[0][2] >= LIFETIME:
            label = self._entries.pop(0)[0]
            label.visible = False
            self._free.append(label)
        w, h = window_size
        scale = self.ui.scale
        view_proj = camera.get_projection_matrix() * camera.get_view_matrix()
        for label, position, start, jitter, rgb in self._entries:
            clip = view_proj * glm.vec4(position, 1.0)
            if clip.w <= 0.1:          # behind the camera
                label.visible = False
                continue
            t = (now - start) / LIFETIME
            sx = (clip.x / clip.w * 0.5 + 0.5) * w / scale + jitter
            sy = (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * h / scale - RISE * (1.0 - (1.0 - t) ** 2)   # eases out
            alpha = min(1.0, (LIFETIME - (now - start)) / FADE)
            label.offset = (sx, sy)
            label.color = (*rgb, round(255 * alpha))
            label.visible = True
