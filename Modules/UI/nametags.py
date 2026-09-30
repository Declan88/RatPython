"""
Floating Steam-name tags over other players' heads. Tags are ordinary UI
widgets (a small dark pill with a Label) whose position is re-projected from
the player's world position each frame - so they reuse the UI renderer's text
cache and batching instead of anything world-space of their own.

Occlusion, though, IS a real 3D-space check: a tag's head point is the anchor
a physics line trace runs against every frame (see update's own scene
parameter), same as this project's per-light shadow approximation
(PhysicsWorld.line_of_sight) - so a wall between the camera and that point
hides the tag exactly like it hides the player wearing it, not just a flat
screen-space overlay that shows through geometry.
"""

import glm

from . import theme
from .widgets import Anchor, Label, Panel

HEAD_HEIGHT = 1.85      # metres above the feet - clears the rat and a tall hat
MAX_DISTANCE = 40.0     # tags fade out entirely beyond this


class NameTags:
    def __init__(self, ui):
        self.ui = ui
        self.container = ui.root.add(Panel(size_frac=(1, 1), visible=False, name="nametags"))
        self._tags = {}    # steam id -> (pill, label)

    @property
    def visible(self):
        return self.container.visible

    @visible.setter
    def visible(self, value):
        self.container.visible = value

    def _make(self):
        pill = self.container.add(Panel(color=(8, 10, 18, 170), anchor=Anchor.TOP_LEFT,
                                        pivot=(0.5, 1.0), layout="horizontal", padding=6,
                                        fit_content=True, visible=False))
        label = pill.add(Label("", font_size=22, color=theme.TEXT))
        return pill, label

    def update(self, players, camera, window_size, scene=None):
        """players: {steam id: RemotePlayer} (anything with .name and a
        .model.obj["position"] feet position). scene: used for the wall-
        occlusion check (see this module's own docstring) - None skips it
        (a tag then behaves as before, always showing through geometry)."""
        for steam_id in [i for i in self._tags if i not in players]:
            self.container.remove(self._tags.pop(steam_id)[0])

        w, h = window_size
        scale = self.ui.scale
        view_proj = camera.get_projection_matrix() * camera.get_view_matrix()
        for steam_id, player in players.items():
            tag = self._tags.get(steam_id)
            if tag is None:
                tag = self._tags[steam_id] = self._make()
            pill, label = tag

            obj = player.model.obj
            if obj is None or getattr(player, "dead", False):
                pill.visible = False
                continue
            head = obj["position"] + glm.vec3(0.0, HEAD_HEIGHT, 0.0)
            clip = view_proj * glm.vec4(head, 1.0)
            if clip.w <= 0.1 or glm.distance(head, camera.position) > MAX_DISTANCE:
                pill.visible = False
                continue
            sx = (clip.x / clip.w * 0.5 + 0.5) * w
            sy = (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * h
            if not (0 <= sx <= w and 0 <= sy <= h):
                pill.visible = False
                continue
            # Cheapest checks (behind camera, too far, off-screen) first - only a tag that's
            # already passed all of those is worth a physics trace at all.
            if scene is not None and not scene.physics.line_of_sight(camera.position, head):
                pill.visible = False
                continue

            text = player.name or f"Player {steam_id % 10000}"
            if label.text != text:
                label.text = text
            pill.offset = (sx / scale, sy / scale)
            pill.visible = True
