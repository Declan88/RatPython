import glm
from Modules.Scenes.scene_base import Scene

SEWER_MODEL = "Assets/Models/Map/Sewer/Sewer.glb"


class SewerScene(Scene):
    """The Sewer map (Deathmatch only - see Modules/GameModes/registry.py's MODE_MAPS). Sky and
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
        # Distance fog (see Scene.set_fog). Boardwalk's 40 m to 380 m is for a much bigger world; this map is
        # only about 90 m across and dark, so the fog starts almost at once and is nearly solid by 70 m. The
        # colour is close to the sky's horizon average (126, 102, 79), a little darker, so distant walls
        # fade into the sunset haze.
        self.set_fog(
            color=(110, 90, 72), start=3.0, end=70.0, max_density=0.92, sky_density=0.35
        )

        # Sewer.glb's materials are all exported doubleSided (Blender's default when Backface
        # Culling isn't ticked per material); ignore that so opaque materials cull back faces.
        self.add_static(
            SEWER_MODEL, collision=True, ignore_source_double_sided=True,
            # Per bloom_ object (by node name): dimmer than the default, spread wider.
            bloom_overrides={"bloom_sphere": {"intensity": 0.5, "radius": 2.5}},
            # The water object is "water_floor", with the same "Watermat" material as Boardwalk's water. These
            # are Boardwalk's water settings (see mainmap_scene.py for what each does), matched by MATERIAL
            # name - except the collision one just below, which is by object name.
            # Boardwalk's water object has no collider at all (see mainmap_scene.py's "Water": None): you
            # fall or wade through it instead of standing on its surface. Same here, by OBJECT name.
            collision_object_overrides={"water_floor": None},
            alpha_mode_overrides={"Watermat": "BLEND"},
            base_alpha_overrides={"Watermat": 0.8},
            specular_strength_overrides={"Watermat": 5},
            double_sided_overrides={"Watermat": True},
            water_overrides={
                "Watermat": {
                    "pan1_speed": (0.03, 0.015),
                    "pan2_speed": (-0.02, 0.025),
                    "uv_scale": 0.4,
                    "uv2_scale": 2.3,
                    "normal_scale": 2,
                    "reflection_mode": "ssr",
                }
            },
        )

        # What "reflection_mode": "ssr" above needs (see Scene.enable_screen_space_reflections).
        self.enable_screen_space_reflections()

        self.add_lights_from_glb(SEWER_MODEL, cast_shadows=True, radius_multiplier=3, intensity_multiplier=1)
        self.add_actors_from_glb(SEWER_MODEL)
        self.add_ladders_from_glb(SEWER_MODEL)
        # "character_spawn" Empties become player spawn points (see Scene.add_spawns_from_glb).
        self.add_spawns_from_glb(SEWER_MODEL)

        self.bake_static_lighting(lightmap_resolution=2048, auto_lightmap_resolution=True)
