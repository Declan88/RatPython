import glm
from Modules.Scenes.scene_base import Scene


class TorusScene(Scene):
    def __init__(self, ctx):
        super().__init__(ctx, recalculate_shadows=True)

        # Natural overhead sun angle (shining down from above, slightly right and forward)
        self.light_dir = glm.vec3(0.5, 1.0, 0.8)

        # Skybox Setup
        self.add_skybox(
            [
                "Assets/Textures/Skybox/southsidert.png",  # +X
                "Assets/Textures/Skybox/southsidelf.png",  # -X
                "Assets/Textures/Skybox/southsideup.png",  # +Y
                "Assets/Textures/Skybox/southsidedn.png",  # -Y
                "Assets/Textures/Skybox/southsidebk.png",  # +Z
                "Assets/Textures/Skybox/southsideft.png",  # -Z
            ],
            # tint=(0.05, 0.3, 0.35),
            rotations=[180, 180, 0, 180, 180, 180],
            edge_fade=0.2,
        )

        # Scene Static Elements

        self.add_static(
            "Assets/Models/Testmap/plane.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
        )

        self.add_static(
            "Assets/Models/Testmap/plane2.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
        )

        self.add_static(
            "Assets/Models/Testmap/sphere.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
        )

        self.add_static(
            "Assets/Models/monkey.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1, 0.0)),
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
            "Assets/Audio/testsound.wav",
            position=(0.0, 3.0, 0.0),
            volume=0.8,
            min_distance=1.0,
            max_distance=15.0,
            loop=True,
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