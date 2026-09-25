import glm

from Modules.Scenes.scene_base import Scene
from Modules.UI.lobby_menu import LobbyMenu


class MainMenuScene(Scene):
    """Title screen: a slowly turning skybox behind the Host/Join menu. It
    owns no geometry, physics bodies or sounds - app.py runs it instead of
    the gameplay loop while active, and calls build_ui() for its widgets."""

    def __init__(self, ctx):
        super().__init__(ctx, recalculate_shadows=False)
        self.add_equirect_skybox("Assets/Textures/Skybox/borealis.png", exposure=1.0)
        self.yaw = -90.0
        self.pitch = 8.0
        self.menu = None

    def update(self, dt):
        super().update(dt)
        self.yaw += dt * 4.0
        if self.menu is not None:
            self.menu.update(dt)

    def apply_camera(self, camera):
        camera.position = glm.vec3(0.0, 1.6, 0.0)
        camera.yaw = self.yaw
        camera.pitch = self.pitch
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)
        camera.front = glm.normalize(glm.vec3(
            glm.cos(yaw) * glm.cos(pitch), glm.sin(pitch), glm.sin(yaw) * glm.cos(pitch),
        ))

    def build_ui(self, ui, net, maps, on_start):
        """Adds the menu to ui.root and returns its full-screen container
        (toggle .visible to show/hide it). maps: [(display name, scene
        key)]; on_start(scene_key) is called once a lobby is hosted/joined."""
        self.menu = LobbyMenu(ui, net, maps, on_start)
        ui.root.add(self.menu.root)
        return self.menu.root
