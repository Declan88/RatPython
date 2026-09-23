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
        self.light_intensity = 0.12
        self.set_ambient_intensity(0.4)

        self.add_equirect_skybox("Assets/Textures/Skybox/borealis.png", exposure=1.0)
        # Boosts just the skybox-derived hemisphere ambient (see Scene.
        # set_ambient_intensity's own docstring) without also brightening
        # the visible sky backdrop itself the way exposure above would -
        # this scene's own light_intensity is a deliberately dim 0.05
        # (see below), so surfaces facing away from the sun were reading
        # too dark/flat without a stronger ambient term to fill them in.
        # Tune to taste in-engine.
        self.set_ambient_intensity(2.5)

        self.add_static(
            "Assets/Models/Map/mainmap.glb",
            collision=True,
            # Make tree material use opacity Mask instead of Blend.
            # Watermat -> BLEND is what actually MAKES transparency
            # possible at all - an OPAQUE material (what Watermat is
            # authored as) always outputs full alpha regardless of any
            # factor, and the opaque pass never even enables GL_BLEND -
            # see base_alpha_overrides below for the actual transparency
            # amount, and add_static's own alpha_mode_overrides/
            # base_alpha_overrides docstrings for the full reasoning.
            alpha_mode_overrides={"Treemat": "MASK", "Watermat": "BLEND"},
            # 1.0 = fully opaque, 0.0 = fully invisible - only has any
            # visible effect because Watermat is BLEND now (see above).
            # Tune to taste in-engine.
            base_alpha_overrides={"Watermat": .8},
            specular_strength_overrides={"Watermat": 5},
            water_overrides={"Watermat": {
                "pan1_speed": (0.03, 0.015),
                "pan2_speed": (-0.02, 0.025),
                # < 1.0 tiles the ripple normal map FEWER times across
                # the water's surface, so each ripple reads bigger/more
                # spread out (see add_static's own water_overrides
                # docstring for why this is the inverse of what "scale"
                # might suggest at a glance - it scales UV coordinates,
                # not the visual pattern size directly).
                "uv_scale": 0.4,
                "uv2_scale": 2.3,
                # Overrides the source glTF material's own authored 0.3
                # (see add_static's own water_overrides docstring) for a
                # more pronounced/bumpier ripple look - tune to taste.
                "normal_scale": 2,
                "reflection_mode": "ssr",
            }},
            # mainmap.glb's own re-export set doubleSided=true on nearly
            # every material (confirmed via a raw glTF check - a Blender
            # export default when "Backface Culling" isn't explicitly
            # enabled per-material, not deliberate authoring), which
            # Scene._render_scene now faithfully respects - meaning
            # everything started rendering double-sided, not just
            # Treemat (already double-sided anyway via its own MASK
            # alpha_mode above, independent of this). This makes every
            # OPAQUE material cull normally regardless of what the file
            # itself says - EXCEPT Watermat, which genuinely does need to
            # stay double-sided (visible from both above and below the
            # surface) and would otherwise silently start culling too,
            # now that it's no longer riding on the file's own (here,
            # unreliable) flag for that. double_sided_overrides is there
            # for any OTHER material (a fence, grate, thin leaf card not
            # already using MASK, ...) that turns out to genuinely need
            # double-sided rendering later.
            double_sided_overrides={"Watermat": True},
            ignore_source_double_sided=True,
            # Per-OBJECT collision behavior (see add_static's own
            # collision_object_overrides docstring - matched by the
            # mesh's own NODE/Object name, confirmed directly against
            # mainmap.glb's node list, NOT its material name - "Water"
            # and "CenterFloor" are both real objects there, distinct
            # from the "Watermat" MATERIAL several other objects also
            # happen to use). Water=None means no collider at all for
            # that one object's geometry (fall/swim through it instead
            # of standing on it, matching how it visually reads as water
            # rather than a solid floor) - every OTHER object using
            # Watermat (if any) still collides normally, unlike the
            # material-keyed collision_overrides this replaced, which
            # would have zeroed out collision for anything sharing that
            # material. CenterFloor="wade" gives just that one floor
            # object its own water-knee footstep sound, independent of
            # the blanket physical_material everything else uses.
            collision_object_overrides={"Water": None, "CenterFloor": "wade"},
        )

        # Only meaningful because "reflection_mode": "ssr" is actually
        # set above - see enable_screen_space_reflections' own docstring.
        # Unlike the earlier planar-reflection approach, no water_height/
        # per-material plane assumption is needed at all: SSR ray-marches
        # real per-pixel depth, so it works for any reflective surface's
        # actual shape, not just a flat plane at a known height.
        self.enable_screen_space_reflections()

        self.add_lights_from_glb(
            "Assets/Models/Map/mainmap.glb",
            cast_shadows=True,
            radius_multiplier=3,
            intensity_multiplier=1,
        )

        self.sound_manager.add_sound(
            "Assets/Audio/crucial_urban_rooftop_ambloop01.wav",
            position=(0.0, 3.0, 0.0),
            volume=0.2,
            min_distance=0,
            max_distance=100000,
            loop=True,
            universal=True,
        )



        self.bake_static_lighting(lightmap_resolution=2048)
