import glm
from Modules.Scenes.scene_base import Scene

SEWER_MODEL = "Assets/Models/Map/Sewer/Sewer.glb"


class SewerScene(Scene):
    """The Sewer map (Deathmatch only - see Modules/GameModes/registry.py's MAP_MODES). Sky and
    lighting are built from Assets/Textures/Skybox/Sky03.png; see __init__."""

    def __init__(self, ctx):
        super().__init__(ctx, recalculate_shadows=True)

        # Sewer.glb has a catch-all floor plane at y=-7.45 well below the walkable level.
        self.min_height = -10.0

        # Everything below is taken from Sky03.png, a sunset: the sun is the bright blob at about
        # u=0.853, 4 degrees above the horizon (equirect: u = atan2(z, x) / 2pi + 0.5), so the light
        # comes from that direction - raised to 45 degrees, since a sun at 4 degrees would throw
        # shadows many times longer than the map. Its colour is the warm horizon glow. The ambient
        # (the sky's own colours, see set_ambient_intensity) and the fog colour (the horizon band's
        # average) come from the same image. Sun intensity and ambient are tuned for visible shadows, not measured.
        self.light_dir = glm.vec3(-0.425, 0.707, 0.565)
        self.light_color = glm.vec3(1.0, 0.80, 0.62)
        self.light_intensity = 0.4
        self.add_equirect_skybox("Assets/Textures/Skybox/Sky03.png", exposure=1.0)
        self.set_ambient_intensity(1.5)
        self.set_fog(
            color=(126, 102, 79), start=40.0, end=380.0, max_density=1.0, sky_density=0.35
        )

        # Sewer.glb's materials are all exported doubleSided (Blender's default when Backface
        # Culling isn't ticked per material); ignore that so opaque materials cull back faces.
        self.add_static(
            SEWER_MODEL, collision=True, ignore_source_double_sided=True,
            # Per bloom_ object (by node name): dimmer than the default, spread wider.
            bloom_overrides={"bloom_sphere": {"intensity": 0.5, "radius": 2.5}},
        )

        self.add_lights_from_glb(SEWER_MODEL, cast_shadows=True, radius_multiplier=3, intensity_multiplier=1)
        self.add_actors_from_glb(SEWER_MODEL)
        self.add_ladders_from_glb(SEWER_MODEL)
        # "character_spawn" Empties become player spawn points (see Scene.add_spawns_from_glb).
        self.add_spawns_from_glb(SEWER_MODEL)

        self.bake_static_lighting(lightmap_resolution=2048, auto_lightmap_resolution=True)
