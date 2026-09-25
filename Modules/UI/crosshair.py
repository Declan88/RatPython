"""
The aiming crosshair: a small centre dot and four short lines around it whose gap is
the weapon's real spread, drawn so a shot can land no further from the centre
than the gap's inner edge.

Shots deviate from the aim by up to `spread` degrees (WeaponsBase.
spread_degrees). A direction `spread` degrees off the view axis projects to
    tan(spread) / tan(fov / 2) * (screen height / 2)
pixels from the centre for a perspective camera with vertical field of view
`fov` - exactly what set_spread converts to, in the UI's logical units (its
reference screen is always 1080 high, see UIManager.scale_mode), so the
crosshair matches the true bullet cone at any resolution.
"""

import math

from .widgets import Anchor, Widget, _color

_REFERENCE_HEIGHT = 1080.0   # logical screen height (UIManager.reference_height)


class Crosshair(Widget):
    def __init__(self, length=9.0, thickness=2.0, dot=2.0, color=(255, 255, 255, 235),
                 outline=(0, 0, 0, 190), **kw):
        kw.setdefault("anchor", Anchor.CENTER)
        super().__init__(**kw)
        self.length = length
        self.thickness = thickness
        self.dot = dot          # side of the centre dot, logical px (0 = none)
        self.color = color
        self.outline = outline
        self.gap = 0.0    # logical px from the centre to the lines' inner ends

    def set_spread(self, spread_degrees, fov_degrees):
        """Sets the gap from the weapon's current spread (half-angle of the
        cone its shots land in, degrees) and the camera's vertical FOV."""
        half_height = _REFERENCE_HEIGHT / 2.0
        self.gap = half_height * math.tan(math.radians(spread_degrees)) / math.tan(math.radians(fov_degrees) / 2.0)

    def draw(self, out):
        x, y, w, h = self.rect
        cx, cy = x + w / 2.0, y + h / 2.0
        gap, length, t = self.gap, self.length, self.thickness
        o = 1.0   # outline width around each line, for contrast on any background
        # (left, top, width, height) of each of the four lines.
        segments = (
            (cx - t / 2, cy - gap - length, t, length),   # up
            (cx - t / 2, cy + gap, t, length),            # down
            (cx - gap - length, cy - t / 2, length, t),   # left
            (cx + gap, cy - t / 2, length, t),            # right
        )
        for sx, sy, sw, sh in segments:
            out.rect((sx - o, sy - o, sw + 2 * o, sh + 2 * o), _color(self.outline))
        for segment in segments:
            out.rect(segment, _color(self.color))
        if self.dot > 0.0:
            d = self.dot
            out.rect((cx - d / 2 - o, cy - d / 2 - o, d + 2 * o, d + 2 * o), _color(self.outline))
            out.rect((cx - d / 2, cy - d / 2, d, d), _color(self.color))
        super().draw(out)
