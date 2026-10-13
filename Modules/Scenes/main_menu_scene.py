import math

import glm

from Modules.Scenes.scene_base import Scene
from Modules.Player.player_model import PlayerModel
from Modules.Player.rat_colors import RAT_TINT_MASK_PATH
from Modules.UI import theme
from Modules.UI.lobby_menu import LobbyMenu
from Modules.UI.lobby_room import LobbyRoom

_RAT_ANIM_TIME_SCALE = 24.0 / 30.0
SKYBOX_PATH = "Assets/Textures/Skybox/menu.png"
SKYBOX_SPIN_DEGREES_PER_SECOND = 10  # one full turn of the backdrop takes 3 minutes
# Where the backdrop starts, in degrees of skybox_yaw. Negative looks more to the RIGHT of the panorama
# (a positive yaw slides the scenery right on screen, i.e. looks left). The spin then carries on from here.
SKYBOX_START_YAW = -40.0
# The camera itself keeps the game's 45 degrees, so the preview rat looks the way it always has (a wide
# lens exaggerates anything near the camera and made it huge). menu.png is a crisp landscape, though, and
# through a 45 degree lens it shows only ~73 degrees of the panorama magnified ~4.6x, which reads as zoomed
# in - and any wider perspective lens stretches the edges. So the sky is not drawn through a lens at all:
# it is laid FLAT on the screen exactly as the image is authored (see Scene.skybox_flat_span), which shows as
# much of it as you like with no stretching. This is how many degrees of the image's HEIGHT are visible:
# 180 = all of it (and, on a 16:9 window, about 320 of its 360 degrees across); lower zooms in.
# It has no effect on the rat or the UI.
MENU_FOV = 45.0
# How strongly the skybox lights the preview rat: 1.0 = exactly the panorama's own light. menu.png is a dark
# scene, so at 1.0 the rat is dim like everything in it; raise this to brighten the rat alone (the sky
# itself is not affected).
ENV_LIGHT_INTENSITY = 3.0
SKYBOX_VERTICAL_SPAN = 180.0


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
        self.add_equirect_skybox(SKYBOX_PATH, exposure=1.0)
        self.skybox_yaw = SKYBOX_START_YAW % 360.0
        self.skybox_flat_span = SKYBOX_VERTICAL_SPAN
        self.time = 0.0
        self.menu = None
        self.lobby_room = None
        # The preview rat is lit by the skybox itself: the panorama's own diffuse light (spherical harmonics -
        # see Scene.use_environment_sh), so the glows in menu.png light it from where they are and the
        # dark ground lights it from below. No sun and no studio fill; ENV_LIGHT_INTENSITY scales it.
        self.use_environment_sh = True
        self.set_ambient_intensity(ENV_LIGHT_INTENSITY)
        self.light_intensity = 0.0
        self._preview_pos = glm.vec3(0.0, 0.0, -self.PREVIEW_DISTANCE)

        # The preview: a standing, idling copy of the player's rat that wears
        # whatever hat the wardrobe dropdown has picked.
        self.preview = PlayerModel(
            self,
            "Assets/Models/rat.glb",
            visible_in_color=True,
            cast_shadow=False,
            states=(("idle", "rifle_idle", 0.0),),
            time_scale=_RAT_ANIM_TIME_SCALE,
            tint_mask_path=RAT_TINT_MASK_PATH,
        )
        if self.preview.obj is not None:
            self.preview.obj["specular_strength"] = 0
            self.load_additional_animations(
                self.preview.obj,
                "Assets/Animations/Poses/Rifle/rifleidle.glb",
                rename={"New": "rifle_idle"},
                time_scale=_RAT_ANIM_TIME_SCALE,
            )

    def set_preview_hat(self, name):
        self.preview.set_hat(name)

    def set_preview_color(self, rgb):
        self.preview.set_tint(rgb)

    def update(self, dt):
        super().update(dt)
        self.time += dt
        self.skybox_yaw = (
            self.skybox_yaw + SKYBOX_SPIN_DEGREES_PER_SECOND * dt
        ) % 360.0
        # Faces the camera (yaw 90 = model forward is +Z) with a slow sway so
        # it feels alive.
        yaw = 72.0 + 14.0 * math.sin(self.time * 0.6)
        self.preview.update(dt, self._preview_pos, yaw, 0.0)
        if self.menu is not None:
            self.menu.update(dt)
        if self.lobby_room is not None:
            self.lobby_room.update(dt)

    def apply_camera(self, camera):
        camera.fov = MENU_FOV
        camera.position = glm.vec3(0.0, 0.9, 0.0)
        camera.yaw = -90.0 + 1.5 * math.sin(self.time * 0.25)
        camera.pitch = 0.0
        yaw, pitch = glm.radians(camera.yaw), glm.radians(camera.pitch)
        camera.front = glm.normalize(
            glm.vec3(
                glm.cos(yaw) * glm.cos(pitch),
                glm.sin(pitch),
                glm.sin(yaw) * glm.cos(pitch),
            )
        )
        # Centre the rat in the space right of the sidebar at any window shape:
        # that region's centre sits at ndc x = sidebar / window width (both in
        # the UI's 1080-high logical pixels), and the view's half-width at the
        # rat's distance is tan(fov/2) * aspect * distance.
        ndc_x = theme.SIDEBAR_WIDTH / (1080.0 * camera.aspect)
        half_width = (
            self.PREVIEW_DISTANCE
            * math.tan(math.radians(camera.fov) / 2.0)
            * camera.aspect
        )
        self._preview_pos = glm.vec3(
            half_width * ndc_x,
            camera.position.y - self.PREVIEW_FEET_DROP,
            -self.PREVIEW_DISTANCE,
        )

    def build_ui(self, ui, net, maps, modes, on_start, on_lobby_enter, on_lobby_leave):
        """Adds the menu AND the lobby waiting room (Modules/UI/
        lobby_room.py) to ui.root, and returns the menu's own full-screen
        container (toggle .visible to show/hide it, same as before this
        existed). maps: [(display name, scene key)]; modes: [(display
        name, game mode id)] - see app.py's own MODES, built from
        Modules.GameModes.registry.

        on_start(scene_key, mode_id): fires once the host ACTUALLY
        presses Start in the lobby room (or immediately, for single-
        player, which never shows one at all) - this is the real
        "begin gameplay" signal app.py already had; it now runs through
        the lobby room instead of right after hosting/joining.

        on_lobby_enter()/on_lobby_leave(): app.py's own hooks for its
        in_lobby flag (a new third main-loop state alongside in_menu/
        gameplay) - fired when hosting/joining succeeds (showing the
        lobby room) and when Back is pressed there (hiding it, showing
        the main menu again) respectively. Purely bookkeeping on app.py's
        side; this scene's own menu/lobby_room visibility is managed
        right here regardless."""
        self.lobby_room = LobbyRoom(
            ui,
            net,
            maps,
            modes,
            on_game_start=on_start,
            on_leave=lambda: (
                setattr(self.menu.root, "visible", True),
                on_lobby_leave(),
            ),
        )
        ui.root.add(self.lobby_room.root)

        def on_lobby_ready(map_key, mode_key):
            self.menu.root.visible = False
            self.lobby_room.show(map_key, mode_key)
            on_lobby_enter()

        self.menu = LobbyMenu(
            ui,
            net,
            maps,
            modes,
            on_start,
            on_lobby_ready=on_lobby_ready,
            on_hat_change=self.set_preview_hat,
            on_color_change=self.set_preview_color,
        )
        self.set_preview_hat(net.local_hat)
        self.set_preview_color(net.local_color)
        ui.root.add(self.menu.root)
        return self.menu.root
