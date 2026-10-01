"""
A CS/CoD-style damage direction indicator: when the local player takes damage,
a red glow appears on a ring around the crosshair, oriented toward wherever
the shot actually came from in the WORLD (not the screen) - it keeps pointing
at that same real-world direction even as the camera turns, fading out after a
couple of seconds. Several can be up at once (a burst of hits from different
directions each get their own), same as a real match's own version of this.

The glow itself is a thin, soft red rectangle - brightest a short way in from
the tip (the end pointing away from the crosshair, toward the actual damage
source), fading smoothly to fully transparent at both that tip and its base
(the end nearest the crosshair), rather than a tapered arrow/wedge shape.

Needs an arbitrarily-rotated quad (it has to point in any of 360 degrees, not
just the UI's usual axis-aligned rects/textures) - see DrawList.
texture_logical_oriented in renderer.py, added for exactly this.
"""

import math
import time

import numpy as np
import pygame

from .widgets import Anchor, Widget, _color

_TEXTURE_SIZE = 64
RING_RADIUS = 100.0            # logical px from screen centre the indicator's own centre sits at
INDICATOR_SIZE = (16.0, 150.0)  # logical px (width, height) - height is its "pointing" axis
DISPLAY_SECONDS = 1.4
FADE_SECONDS = 0.5
_RED = (230, 25, 25)
# Where along the texture's own length (0 = tip, 1 = base) the gradient peaks - NOT 0 itself,
# so the tip fades IN from fully transparent too (see _build_texture's own docstring on why
# pinning the peak at the very edge read as a hard line instead of a soft glow).
_PEAK_T = 0.14
# Caps the texture's own brightest alpha well under 255 - "more transparent... less
# intrusive" per direct request, independent of DamageIndicator.color's own alpha (which
# multiplies this again) and each entry's own fade-out.
_MAX_ALPHA = 150


def _build_texture():
    """A plain, full-width column (no taper, unlike an arrow/wedge shape) filled with a
    red glow along its own length - fully transparent at BOTH the tip (texture y=0, this
    widget's "forward"/pointing end) and the base (y=n-1), brightest at _PEAK_T in between.
    Peaking inside the texture rather than right at the tip's own edge matters: a gradient
    pinned to full opacity AT the boundary has nothing to fade into there, so the tip reads
    as a flat, hard-edged line of solid color sitting right at the end rather than a soft
    glow - fading in before that edge is reached removes that line entirely. "Thin" comes
    entirely from INDICATOR_SIZE's own width, not this texture - every column across the
    texture's width is identical, so bilinear sampling (see texture_from_surface's own
    LINEAR filter) already softens its left/right edges with no extra masking needed here."""
    n = _TEXTURE_SIZE
    yy = np.mgrid[0:n, 0:n][0].astype("f4")
    t = yy / (n - 1.0)
    rising = np.clip(t / _PEAK_T, 0.0, 1.0)              # 0 at the tip -> 1 at the peak
    falling = np.clip((1.0 - t) / (1.0 - _PEAK_T), 0.0, 1.0)   # 1 at the peak -> 0 at the base
    gradient = np.minimum(rising, falling) ** 0.8
    rgba = np.zeros((n, n, 4), "u1")
    rgba[..., 0] = _RED[0]
    rgba[..., 1] = _RED[1]
    rgba[..., 2] = _RED[2]
    rgba[..., 3] = (gradient * _MAX_ALPHA).astype("u1")
    return pygame.image.frombuffer(rgba.tobytes(), (n, n), "RGBA")


class DamageIndicator(Widget):
    def __init__(self, color=(255, 255, 255, 170), **kw):
        kw.setdefault("anchor", Anchor.CENTER)
        kw.setdefault("size", (0, 0))
        super().__init__(**kw)
        # White, not red - the gradient's own red is already baked into the texture (see
        # _build_texture); this only ever multiplies it, so it controls overall opacity
        # (the .a below, combined with each entry's own fade - see draw()) without
        # retinting the hue. Left as a real color (not a bare float) in case a future
        # caller ever wants a differently-colored variant (e.g. a distinct color for a
        # damage-over-time/burn effect) the same way Hitmarker's own color param works.
        self.color = color
        self._tex = None
        self._entries = []   # [{"start": t, "dir": (dx, dz) unit, world-space XZ}]
        self._live = []      # this frame's [(center_logical, screen_dir, alpha)] - see update()

    def prime(self):
        """Builds its texture now (needs the UI's renderer) instead of at the first hit -
        same reasoning as Hitmarker.prime()."""
        if self._tex is None and self.manager is not None:
            self.manager._ensure_renderer()
            self._tex = self.manager.renderer.texture_from_surface(_build_texture())

    def _release_resources(self):
        super()._release_resources()
        if self._tex is not None:
            self._tex.release()
            self._tex = None

    def trigger(self, direction_xz):
        """Shows a new arrow for a hit that came from `direction_xz` - (dx, dz), the
        horizontal (XZ - this project's Y is up, see physics_world.py's own module
        docstring) world-space vector from the victim TOWARD the attacker at the moment of
        the hit. Any nonzero length (normalized here); a near-zero vector (the rare case of
        an attack landing essentially on top of the victim, e.g. a point-blank explosion)
        is dropped rather than showing an arrow pointing in an arbitrary/undefined
        direction."""
        length = math.hypot(*direction_xz)
        if length < 1e-6:
            return
        self._entries.append({"start": time.perf_counter(), "dir": (direction_xz[0] / length, direction_xz[1] / length)})

    def update(self, camera):
        """Call once a frame (app.py's main loop does, unconditionally - same shape as
        Hitmarker/KillFeed): drops expired arrows and recomputes every live one's CURRENT
        screen position/orientation from the camera's CURRENT facing - done here, fresh
        every frame, rather than once at trigger() time, so an arrow keeps pointing at the
        same fixed real-world direction even as the player spins around while it's still
        showing (exactly how a real game's own version of this behaves) - only the world
        direction itself (see trigger()) is fixed at hit time, never the on-screen angle."""
        if not self._entries:
            self._live = []
            return
        now = time.perf_counter()
        total = DISPLAY_SECONDS + FADE_SECONDS
        self._entries = [e for e in self._entries if now - e["start"] < total]
        x, y, w, h = self.rect
        cx, cy = x + w / 2.0, y + h / 2.0
        # Both already flattened to the horizontal plane (ignore pitch) - looking up/down
        # shouldn't swing the arrow around, only yaw should.
        fwd, right = camera.get_flat_forward(), camera.get_flat_right()
        live = []
        for e in self._entries:
            dx, dz = e["dir"]
            # Signed bearing off the camera's own forward, +right = clockwise on screen -
            # atan2(right-component, forward-component) rather than comparing raw yaw
            # angles avoids ever having to reason about camera.yaw's own +-180 wrap-around.
            bearing = math.atan2(dx * right.x + dz * right.z, dx * fwd.x + dz * fwd.z)
            screen_dir = (math.sin(bearing), -math.cos(bearing))   # bearing=0 (dead ahead) -> straight up
            center = (cx + screen_dir[0] * RING_RADIUS, cy + screen_dir[1] * RING_RADIUS)
            elapsed = now - e["start"]
            alpha = 255
            if elapsed > DISPLAY_SECONDS:
                alpha = round(255 * (1.0 - (elapsed - DISPLAY_SECONDS) / FADE_SECONDS))
            live.append((center, screen_dir, alpha))
        self._live = live

    def draw(self, out):
        if self._live and self.manager is not None:
            if self._tex is None:
                self._tex = self.manager.renderer.texture_from_surface(_build_texture())
            r, g, b, a = self.color
            for center, screen_dir, alpha in self._live:
                out.texture_logical_oriented(
                    self._tex, center, screen_dir, INDICATOR_SIZE, _color((r, g, b, round(a * alpha / 255)))
                )
        super().draw(out)
