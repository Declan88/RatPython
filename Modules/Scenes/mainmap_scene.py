import glm
from Modules.Scenes.scene_base import Scene


class MainMapScene(Scene):
    def __init__(self, ctx):
        super().__init__(ctx, recalculate_shadows=True)

        # Same sun/skybox lighting setup as TorusScene (torus_scene.py) -
        # kept identical rather than re-tuned, since this scene's only
        # purpose right now is showing mainmap.glb under the same
        # lighting conditions already dialed in there.
        self.light_dir = glm.vec3(0.5, 1.0, 0.8)
        self.light_color = glm.vec3(0.6, 0.8, 1)
        self.light_intensity = 0.1

        self.add_equirect_skybox("Assets/Textures/Skybox/borealis.png", exposure=1.0)

        self.add_static(
            "Assets/Models/Map/mainmap.glb",
            collision=True,
            # Make tree material use opacity Mask instead of Blend
            alpha_mode_overrides={"Treemat": "MASK"},
        )

        self.add_lights_from_glb(
            "Assets/Models/Map/mainmap.glb",
            cast_shadows=True,
            radius_multiplier=3,
            intensity_multiplier=0.8,
        )

        self.bake_static_lighting(lightmap_resolution=1024)
