import math

import glm

from Modules.Scenes.scene_base import Scene
from Modules.Player.player_model import PlayerModel
from Modules.Player.rat_colors import RAT_TINT_MASK_PATH
from Modules.UI import theme
from Modules.UI.lobby_menu import LobbyMenu

_RAT_ANIM_TIME_SCALE = 24.0 / 30.0


class MainMenuScene(Scene):
    """Title screen: a slowly turning skybox behind the Host/Join menu. It
    owns no geometry, physics bodies or sounds - app.py runs it instead of
    the gameplay loop while active, and calls build_ui() for its widgets."""

    # Where the preview rat stands: this far in front of the camera, at this
    # fraction of the way from screen centre to the right edge (the left is
    # the menu sidebar), feet this far below the camera.
    PREVIEW_DISTANCE = 3.7
    PREVIEW_FEET_DROP = 0.9

    def __init__(self, ctx):
        super().__init__(ctx, recalculate_shadows=False)
        self.add_equirect_skybox("Assets/Textures/Skybox/borealis.png", exposure=1.0)
        self.time = 0.0
        self.menu = None
        # Soft, frontal studio-style light for the preview rat: strong fill
        # ambient, a gentler key from the camera side and no shadow casting
        # (its own shadow was the harsh black patches on its body).
        self.set_ambient_intensity(8.0)
        self.light_intensity = 1.3
        self.light_dir = glm.vec3(0.3, 0.8, 1.0)
        self._preview_pos = glm.vec3(0.0, 0.0, -self.PREVIEW_DISTANCE)

        # The preview: a standing, idling copy of the player's rat that wears
        # whatever hat the wardrobe dropdown has picked.
        self.preview = PlayerModel(
            self, "Assets/Models/rat.glb", visible_in_color=True, cast_shadow=False,
            states=(("idle", "rifle_idle", 0.0),), time_scale=_RAT_ANIM_TIME_SCALE,
            tint_mask_path=RAT_TINT_MASK_PATH,
        )
        if self.preview.obj is not None:
            self.preview.obj["specular_strength"] = 0
            self.load_additional_animations(
                self.preview.obj, "Assets/Animations/Poses/Rifle/rifleidle.glb",
                rename={"New": "rifle_idle"}, time_scale=_RAT_ANIM_TIME_SCALE,
            )

    def set_preview_hat(self, name):
        self.preview.set_hat(name)

    def set_preview_color(self, rgb):
        self.preview.set_tint(rgb)

    def update(self, dt):
        super().update(dt)
        self.time += dt
        # Faces the camera (yaw 90 = model forward is +Z) with a slow sway so
        # it feels alive.
        yaw = 72.0 + 14.0 * math.sin(self.time * 0.6)
        self.preview.update(dt, self._preview_pos, yaw, 0.0)
        if self.menu is not None:
            self.menu.update(dt)

    def apply_camera(self, camera):
        camera.position = glm.vec3(0.0, 0.9, 0.0)
        camera.yaw = -90.0 + 1.5 * math.sin(self.time * 0.25)
        camera.pitch = 0.0
        yaw, pitch = glm.radians(camera.yaw), glm.radians(camera.pitch)
        camera.front = glm.normalize(glm.vec3(
            glm.cos(yaw) * glm.cos(pitch), glm.sin(pitch), glm.sin(yaw) * glm.cos(pitch),
        ))
        # Centre the rat in the space right of the sidebar at any window shape:
        # that region's centre sits at ndc x = sidebar / window width (both in
        # the UI's 1080-high logical pixels), and the view's half-width at the
        # rat's distance is tan(fov/2) * aspect * distance.
        ndc_x = theme.SIDEBAR_WIDTH / (1080.0 * camera.aspect)
        half_width = self.PREVIEW_DISTANCE * math.tan(math.radians(camera.fov) / 2.0) * camera.aspect
        self._preview_pos = glm.vec3(
            half_width * ndc_x, camera.position.y - self.PREVIEW_FEET_DROP,
            -self.PREVIEW_DISTANCE,
        )

    def build_ui(self, ui, net, maps, on_start):
        """Adds the menu to ui.root and returns its full-screen container
        (toggle .visible to show/hide it). maps: [(display name, scene
        key)]; on_start(scene_key) is called once a lobby is hosted/joined."""
        self.menu = LobbyMenu(ui, net, maps, on_start, on_hat_change=self.set_preview_hat,
                              on_color_change=self.set_preview_color)
        self.set_preview_hat(net.local_hat)
        self.set_preview_color(net.local_color)
        ui.root.add(self.menu.root)
        return self.menu.root
