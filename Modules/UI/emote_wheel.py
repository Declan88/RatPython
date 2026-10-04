"""
The radial emote wheel (hold B - see app.py's own wiring and PlayerModel.
play_emote). Text-only: no icon assets exist for this, so each slot is just
the clip's own name, arranged in a circle around screen center. The pages/
slots aren't hardcoded anywhere - set_pages() is handed whatever Assets/
Animations/Emotes/Emotes.glb's own clips turned out to be named (see app.py's
load_additional_animations call site), chunked into pages of up to
_SLOTS_PER_PAGE each.

No pie-slice/wedge graphics: Modules/UI/renderer.py's DrawList has no arc/
polygon primitive, only axis-aligned and uniformly-rotated quads - a ring of
plain labels over a procedurally-generated circular backdrop (same numpy/
pygame "build a soft gradient onto a Surface once, upload as a texture"
pattern damage_indicator.py's own _build_texture already uses) is the zero-
new-art approach, not a corner cut.

Open and close both animate through the SAME single progress value (see
_current_scale/_start_anim) - closing is just opening played backwards from
wherever the progress currently sits, not a separate animation, so flicking B
open/closed quickly still eases smoothly instead of popping.
"""

import math
import time

import numpy as np
import pygame

from . import theme
from .widgets import Anchor, Label, Panel, _color

_SLOTS_PER_PAGE = 5
_RADIUS = 170.0           # logical px from screen center to each slot's label
_DEADZONE_RADIUS = 40.0   # logical px - inside this, nothing counts as "hovered"
_FONT_SIZE = 30

_RING_TEXTURE_SIZE = 256
_RING_DIAMETER = _RADIUS * 2.0 + 110.0   # logical px - the backdrop disc's own size
_RING_FILL = theme.CARD[:3]
_RING_RIM = theme.ACCENT[:3]
_RING_FILL_ALPHA = 215
# Where (0 center .. 1 edge) the accent rim sits, and how wide its soft band is -
# a thin bright ring just inside the disc's own edge, not a hard outline.
_RIM_AT = 0.86
_RIM_WIDTH = 0.05
# How far in from the true edge the fill itself starts fading to fully
# transparent, so the disc's own boundary reads as a soft glow, not a hard-
# edged circle (same reasoning as damage_indicator.py's own _PEAK_T comment).
_EDGE_SOFTEN = 0.12

# Full-screen dim behind the ring itself, at full progress - lighter than
# PauseMenu's 150 since this isn't a hard-blocking modal.
_BACKDROP_PEAK_ALPHA = 60

_ANIM_SECONDS = 0.16          # open/close duration, either direction
_HOVER_PUSH = 1.1             # hovered slot's label sits this much farther out


def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _build_ring_texture():
    """A soft circular disc (dark fill, fading to transparent at its own edge,
    with a thin brighter accent rim just inside that edge) - the wheel's own
    backdrop, standing in for the pie-slice/wedge art this project doesn't
    have (see this module's own docstring)."""
    n = _RING_TEXTURE_SIZE
    yy, xx = np.mgrid[0:n, 0:n].astype("f4")
    c = (n - 1) / 2.0
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c   # 0 at center, 1 at the texture's edge
    fill = np.clip((1.0 - dist) / _EDGE_SOFTEN, 0.0, 1.0)
    rim = np.clip(1.0 - np.abs(dist - _RIM_AT) / _RIM_WIDTH, 0.0, 1.0)
    base = np.array(_RING_FILL, dtype="f4")
    rim_color = np.array(_RING_RIM, dtype="f4")
    rgb = base[None, None, :] * (1.0 - rim[..., None]) + rim_color[None, None, :] * rim[..., None]
    rgba = np.zeros((n, n, 4), "u1")
    rgba[..., 0:3] = np.clip(rgb, 0, 255).astype("u1")
    rgba[..., 3] = (fill * _RING_FILL_ALPHA).astype("u1")
    return pygame.image.frombuffer(rgba.tobytes(), (n, n), "RGBA")


class EmoteWheel(Panel):
    """Hold B to open: up to 5 emote names in a circle, the one under the
    mouse's current angle (from screen center) highlighted. Releasing B (see
    app.py) reads confirmed_emote_name() and closes it - closing PLAYS THE
    OPEN ANIMATION IN REVERSE (see _current_scale) rather than vanishing
    instantly. More than 5 emotes -> more pages, changed by scrolling the
    mouse wheel while open - see scroll_by, reached automatically through the
    existing UIManager MOUSEWHEEL -> pick_scroll -> scroll_by path
    (manager.py), no extra input wiring needed for paging."""

    scrollable = True

    def __init__(self, **kw):
        kw.setdefault("size_frac", (1.0, 1.0))
        # color stays fully transparent here - the backdrop's own CURRENT
        # (animated) alpha is computed fresh in draw() instead (see
        # _current_scale), so there's nothing meaningful to set once at
        # construction time the way a static Panel.color normally would be.
        super().__init__(color=(0, 0, 0, 0), visible=False, **kw)
        self._ring_tex = None   # built lazily on first draw - see draw()

        # Single bidirectional progress value (0 = fully closed/invisible, 1 =
        # fully open) - see _current_scale/_start_anim. Starts at a plain 0.0,
        # already-at-rest value (no animation running) rather than mid-
        # transition, matching the wheel's own starting visible=False.
        self._anim_start_scale = 0.0
        self._anim_start_time = time.perf_counter()
        self._anim_target = 0.0

        # 5 Labels created ONCE, here, and never again - their angle (which
        # direction from center) is fixed forever; only their CURRENT radius
        # (animated - see _apply_layout) and color/text change after this.
        # Never touches font_size - that would re-rasterize the label's
        # texture AND change its measured size on every hover/animation frame
        # (Label._ensure_texture/measure both key off it) - color and offset
        # alone are both free (Label.draw tints an already-cached texture;
        # offset is a plain position, not a re-render).
        self._label_angles = [i * 360.0 / _SLOTS_PER_PAGE for i in range(_SLOTS_PER_PAGE)]
        self._labels = [
            self.add(Label("", font_size=_FONT_SIZE, color=theme.TEXT, align="center",
                            anchor=Anchor.CENTER))
            for _ in range(_SLOTS_PER_PAGE)
        ]

        self._pages = [[]]
        self._page = 0
        self._hovered_index = None
        self._apply_layout(0.0)

    def _current_scale(self):
        """The animation's current 0..1 progress, computed fresh from elapsed
        real time rather than stored/ticked - cheap (a few float ops) and
        never drifts out of sync with wall-clock time the way accumulating a
        per-frame dt could. Shared by open(), close(), update() and draw()
        so every consumer agrees on exactly the same value within a frame."""
        elapsed = time.perf_counter() - self._anim_start_time
        if elapsed >= _ANIM_SECONDS:
            return self._anim_target
        eased = _smoothstep(elapsed / _ANIM_SECONDS)
        return self._anim_start_scale + (self._anim_target - self._anim_start_scale) * eased

    def _start_anim(self, target):
        """Begins easing toward `target` (1.0 to open, 0.0 to close) from
        wherever the progress actually is RIGHT NOW - not from 0 or 1 - so
        reversing mid-animation (flicking B open/closed quickly) eases
        smoothly from that in-between point instead of popping to the start
        of a fresh animation first."""
        self._anim_start_scale = self._current_scale()
        self._anim_start_time = time.perf_counter()
        self._anim_target = target

    def _apply_layout(self, scale):
        """Sets every label's text/color/position, and is also what draw()
        reads the ring/backdrop's own current scale from indirectly (see
        _current_scale - draw() calls that itself rather than caching a value
        here, so a single source of truth drives both). Called on a page
        change, a hover change, and once per frame while the open/close
        animation is actually running (see update())."""
        page = self._pages[self._page]
        for i, (label, angle_deg) in enumerate(zip(self._labels, self._label_angles)):
            label.text = page[i] if i < len(page) else ""
            hovered = i == self._hovered_index
            push = _HOVER_PUSH if hovered else 1.0
            radius = _RADIUS * scale * push
            angle = math.radians(angle_deg)
            label.offset = (radius * math.sin(angle), -radius * math.cos(angle))
            base_color = theme.ACCENT if hovered else theme.TEXT
            label.color = (base_color[0], base_color[1], base_color[2], round(base_color[3] * scale))

    def set_pages(self, names):
        """Chunks `names` (the full, auto-discovered emote catalogue) into
        pages of up to _SLOTS_PER_PAGE each. Called once, right after that
        catalogue is known (see app.py) - not per-frame, not per-open."""
        names = list(names)
        self._pages = [
            names[i:i + _SLOTS_PER_PAGE] for i in range(0, len(names), _SLOTS_PER_PAGE)
        ] or [[]]
        self._page = 0
        self._hovered_index = None
        self._apply_layout(self._current_scale())

    def open(self):
        self._hovered_index = None
        self._start_anim(1.0)
        self.visible = True
        self._apply_layout(self._current_scale())

    def close(self):
        """Starts easing OUT (see _start_anim) rather than hiding instantly -
        update() flips self.visible back to False itself once that reverse
        animation actually finishes, so draw() keeps rendering the shrinking/
        fading wheel for the rest of this transition. A no-op if already
        closed/closing (self.visible already False, or already mid-close)."""
        if not self.visible:
            return
        self._start_anim(0.0)

    def is_closing(self):
        """True once close() has been called and the reverse animation hasn't
        finished yet - app.py uses this to stop feeding mouse-hover updates
        into a wheel that's on its way out (picking a slot while it shrinks
        away would be a confusing final flicker right before it disappears)."""
        return self._anim_target <= 0.0 and self.visible

    def update(self):
        """Call once per frame while self.visible (see app.py's main loop) -
        advances the open/close animation and, once a close reaches 0, flips
        self.visible back off for real. Cheap regardless: _current_scale is a
        handful of float ops, and _apply_layout only touches 5 labels'
        offset/color (no texture work - see its own docstring)."""
        scale = self._current_scale()
        self._apply_layout(scale)
        if self._anim_target <= 0.0 and scale <= 0.0:
            self.visible = False

    def scroll_by(self, dy):
        if self.is_closing() or len(self._pages) <= 1:
            return
        self._page = (self._page + (1 if dy < 0 else -1)) % len(self._pages)
        self._hovered_index = None
        self._apply_layout(self._current_scale())

    def pick_scroll(self, x, y):
        # Same shape as ScrollBox.pick_scroll (controls.py) - the full-screen
        # backdrop means this always resolves to self while visible, which is
        # exactly what lets the mouse wheel page through emotes with no new
        # input-routing code in app.py/count_click.
        if not self._contains(x, y):
            return None
        return super().pick_scroll(x, y) or self

    def update_hover(self, mouse_logical_x, mouse_logical_y):
        """Call once per frame, only while self.visible and NOT is_closing()
        (see app.py's main loop) - O(1) regardless of total catalogue size:
        maps the mouse's angle from screen center straight to a slot index on
        the CURRENT page via one division, never loops over slots or the
        full catalogue."""
        cx = self.rect[0] + self.rect[2] / 2.0
        cy = self.rect[1] + self.rect[3] / 2.0
        dx = mouse_logical_x - cx
        dy = mouse_logical_y - cy
        page = self._pages[self._page]
        n = len(page)
        if n == 0 or dx * dx + dy * dy < _DEADZONE_RADIUS * _DEADZONE_RADIUS:
            index = None
        else:
            angle_deg = math.degrees(math.atan2(dx, -dy)) % 360.0
            index = int(round(angle_deg / (360.0 / n))) % n
        if index != self._hovered_index:
            self._hovered_index = index
            self._apply_layout(self._current_scale())

    def confirmed_emote_name(self):
        """The name at the currently hovered slot, or None if nothing's
        hovered - app.py reads this at the moment B is released."""
        if self._hovered_index is None:
            return None
        page = self._pages[self._page]
        if self._hovered_index >= len(page):
            return None
        return page[self._hovered_index]

    def draw(self, out):
        scale = self._current_scale()
        if _BACKDROP_PEAK_ALPHA > 0:
            out.rect(self.rect, _color((0, 0, 0, round(_BACKDROP_PEAK_ALPHA * scale))))
        if self.manager is not None:
            if self._ring_tex is None:
                self.manager._ensure_renderer()
                self._ring_tex = self.manager.renderer.texture_from_surface(_build_ring_texture())
            cx = self.rect[0] + self.rect[2] / 2.0
            cy = self.rect[1] + self.rect[3] / 2.0
            d = _RING_DIAMETER * scale
            rect = (cx - d / 2.0, cy - d / 2.0, d, d)
            out.texture_logical(self._ring_tex, rect, _color((255, 255, 255, round(255 * scale))))
        for c in self.children:
            if c.visible:
                c.draw(out)

    def _release_resources(self):
        super()._release_resources()
        if self._ring_tex is not None:
            self._ring_tex.release()
            self._ring_tex = None
