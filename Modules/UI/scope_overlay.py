"""
The scope sight picture: Reticle.png, fit to the screen WITHOUT distorting its own
1920x1080 aspect ratio - unlike a plain Image stretched via size_frac (what this
replaced), which squashes/stretches the vignette and reticle lines non-uniformly on
any window whose aspect isn't exactly the image's own 16:9. Instead this fits the
image undistorted against whichever axis of the screen actually constrains it, and
EXTENDS the image's own outermost border pixels to cover the other axis's leftover
space (see DrawList.texture_logical_cover) - seamless for a vignette-style overlay
whose border reads as a solid/near-solid colour, rather than showing gaps (a plain
"contain" fit) or cropping the reticle itself off-centre (a plain "cover" fit).
"""

import pygame

from .widgets import Anchor, Widget, _color


class ScopeOverlay(Widget):
    def __init__(self, path, tint=(255, 255, 255, 255), **kw):
        kw.setdefault("anchor", Anchor.TOP_LEFT)
        kw.setdefault("size", (0, 0))
        kw.setdefault("size_frac", (1, 1))
        super().__init__(**kw)
        self.path = path
        self.tint = tint
        self._tex = None
        self._image_size = (1, 1)

    def _load(self):
        if self._tex is not None or self.manager is None:
            return
        surf = pygame.image.load(self.path).convert_alpha()
        self._image_size = surf.get_size()
        self.manager._ensure_renderer()
        self._tex = self.manager.renderer.texture_from_surface(surf)
        # CLAMP_TO_EDGE, not the default REPEAT - texture_logical_cover deliberately
        # samples UVs outside [0, 1] on whichever axis needs extending, relying on this
        # to repeat the outermost pixel row/column there instead of wrapping back
        # around to the image's own opposite edge.
        self._tex.repeat_x = False
        self._tex.repeat_y = False

    def _release_resources(self):
        super()._release_resources()
        if self._tex is not None:
            self._tex.release()
            self._tex = None

    def draw(self, out):
        self._load()
        if self._tex is not None:
            out.texture_logical_cover(self._tex, self.rect, self._image_size, _color(self.tint))
