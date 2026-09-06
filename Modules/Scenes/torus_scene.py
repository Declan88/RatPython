import glm
from Modules.Scenes.scene_base import Scene


class TorusScene(Scene):
    def __init__(self, ctx):
        super().__init__(ctx)

        # Natural overhead sun angle (shining down from above, slightly right and forward)
        self.light_dir = glm.vec3(0.5, 1.0, 0.8)

        self.add_static(
            "Assets/Models/plane.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
        )

        self.add_static(
            "Assets/Models/monkey.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1, 0.0)),
        )

        self.add_dynamic(
            "Assets/Models/torus.glb", position=glm.vec3(0.0, 3.0, 0.0), rot_speed=1
        )

        self.add_lights_from_glb("Assets/Models/plane.glb", cast_shadows=True,radius_multiplier=3,intensity_multiplier=.8)
