import glm
from Modules.Scenes.scene_base import Scene


class TorusScene(Scene):
    def __init__(self, ctx):
        super().__init__(ctx, recalculate_shadows=True)

        # Natural overhead sun angle (shining down from above, slightly right and forward)
        self.light_dir = glm.vec3(0.5, 1.0, 0.8)

        # Skybox Setup
        self.add_equirect_skybox("Assets/Textures/Skybox/borealis.png", exposure=1.0)
        # Scene Static Elements

        self.add_static(
            "Assets/Models/Testmap/plane.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
            collision=True,
        )

        self.add_static(
            "Assets/Models/Testmap/plane2.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
            collision=True,
        )

        self.add_static(
                    "Assets/Models/Testmap/floorbase.glb",
                    transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, 2, 0.0)),
                    collision=True,
        )

        self.add_static(
            "Assets/Models/Testmap/sphere.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
            collision=True,
        )

        self.add_static(
            "Assets/Models/monkey.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1, 0.0)),
            collision=True,
        )

        # Scene Dynamic Elements

        self.add_dynamic(
            "Assets/Models/torus.glb", position=glm.vec3(0.0, 3.0, 0.0), rot_speed=1
        )

        # Skeletal Meshes
        character = self.add_skeletal(
            "Assets/Models/rat.glb", position=(0, -1, -3), animation="funnyrat_ARMAction"
        )
        character["specular_strength"] = 0
        print("Animations available:", list(character["skeleton"].animations.keys()))

        # Lights

        self.add_lights_from_glb(
            "Assets/Models/Testmap/plane.glb",
            cast_shadows=True,
            radius_multiplier=3,
            intensity_multiplier=0.8,
        )

        self.add_lights_from_glb(
            "Assets/Models/Testmap/plane2.glb",
            cast_shadows=True,
            radius_multiplier=3,
            intensity_multiplier=0.8,
        )

        # Sound

        self.sound_manager.add_sound(
            "Assets/Audio/crucial_urban_rooftop_ambloop01.wav",
            position=(0.0, 3.0, 0.0),
            volume=0.2,
            min_distance=0,
            max_distance=100000,
            loop=True,
            universal=True,
        )

        self.sound_manager.add_sound(
            "Assets/Audio/barkfar.wav",
            position=(3.0, 3.0, 0.0),
            volume=0.8,
            min_distance=1.0,
            max_distance=15.0,
            loop=True,
        )

        # Bake Lights

        self.bake_static_lighting(
            lightmap_resolution=4096, point_shadow_resolution=4096
        )