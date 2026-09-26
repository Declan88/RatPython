"""
The "YOU DIED" screen: the view darkens to red, the title fades in and a countdown (with a
bar) runs down to the respawn.

Nothing is drawn while hidden, and the countdown label only re-renders its text once
a second (a Label caches its texture by text), so it costs a handful of quads and one
text rasterisation per second while it's up.
"""

import math
import time

from . import theme
from .controls import ProgressBar
from .widgets import Anchor, Label, Panel


class DeathScreen(Panel):
    def __init__(self, respawn_seconds=3.0, **kw):
        kw.setdefault("size_frac", (1.0, 1.0))
        super().__init__(color=(0, 0, 0, 0), visible=False, **kw)
        self.respawn_seconds = respawn_seconds
        self.fade_seconds = 0.6
        self._shown_at = None
        self._backdrop_alpha = 175
        self.title = self.add(Label(
            "YOU DIED", font_size=150, color=(200, 24, 24, 0), align="center",
            font=theme.title_font_path(), anchor=Anchor.CENTER, offset=(0, -70)))
        self.subtitle = self.add(Label(
            "", font_size=44, color=(235, 225, 225, 0), align="center",
            anchor=Anchor.CENTER, offset=(0, 70)))
        self.bar = self.add(ProgressBar(
            value=0.0, max_value=respawn_seconds, color=(0, 0, 0, 0), fill_color=(200, 40, 40, 0),
            anchor=Anchor.CENTER, offset=(0, 130), size=(360, 6)))

    def show(self):
        """Starts the screen and its countdown."""
        self._shown_at = time.perf_counter()
        self.visible = True
        self.update()

    def hide(self):
        self._shown_at = None
        self.visible = False

    @property
    def elapsed(self):
        return 0.0 if self._shown_at is None else time.perf_counter() - self._shown_at

    @property
    def finished(self):
        """True once the countdown has run out (time to respawn)."""
        return self._shown_at is not None and self.elapsed >= self.respawn_seconds

    def update(self):
        """Call once a frame while shown: fades things in and updates the countdown."""
        if self._shown_at is None:
            return
        t = self.elapsed
        fade = min(1.0, t / self.fade_seconds)
        eased = fade * fade * (3.0 - 2.0 * fade)
        self.color = (30, 0, 0, int(self._backdrop_alpha * eased))
        # The title lands with a slight settle: starts a little bigger-feeling via its alpha
        # only (Label sizes re-rasterise, so no scaling), then holds.
        self.title.color = (200, 24, 24, int(255 * eased))
        remaining = max(0.0, self.respawn_seconds - t)
        self.subtitle.text = f"Respawning in {math.ceil(remaining)}" if remaining > 0.0 else "Respawning..."
        self.subtitle.color = (235, 225, 225, int(235 * eased))
        self.bar.value = t
        self.bar.color = (0, 0, 0, int(140 * eased))
        self.bar.fill_color = (200, 40, 40, int(255 * eased))
