import glm
from Modules.Scenes.scene_base import Scene


class TorusScene(Scene):
    def __init__(self, ctx):
        super().__init__(ctx, recalculate_shadows=True)

        # Natural overhead sun angle (shining down from above, slightly right and forward)
        self.light_dir = glm.vec3(0.5, 1.0, 0.8)
        self.light_color = glm.vec3(0.6, 0.8, 1)  # slightly warm sunlight
        self.light_intensity = 0.1  # scene_base.py's default - raise/lower to taste

        # Skybox Setup
        self.add_equirect_skybox("Assets/Textures/Skybox/borealis.png", exposure=1.0)
        # Scene Static Elements

        # plane.glb renders as one mesh but is actually 3 unrelated
        # shapes (a flat floor quad, a small box, and a tall pillar)
        # baked together - as exported, adjacent triangles didn't share
        # vertex indices (glTF flat-shading commonly duplicates a
        # vertex's position once per face so each triangle can carry its
        # own normal), so Bullet had no adjacency info and treated every
        # shared edge as an "internal edge" a sweep could catch on
        # (confirmed as the cause of a walking jitter, worse on non-
        # axis-aligned faces since a diagonal slide crosses those seams
        # far more often than a slide parallel to them). That used to be
        # worked around here with 3 precise box colliders standing in
        # for the mesh - boxes have no internal edges - but PhysicsWorld.
        # add_static_mesh now welds coincident vertex positions back
        # together when building the Bullet triangle mesh (see its
        # setWeldingDistance/remove_duplicate_vertices=True), which
        # restores the adjacency info directly, so exact per-triangle
        # mesh collision can be used here instead of the box
        # approximation.
        # plane.glb also contains a staircase (local bounds roughly
        # x:[4.17,6.45], y:[-0.06,3.67], z:[0.22,4.13]) whose individual
        # treads are only ~0.165m deep - narrower than the player hull's
        # own 0.8m (2*radius) footprint, so the hull can never rest on a
        # single tread without its leading face already overlapping the
        # next riser. Every step-up settle sweep in CharacterController.
        # _step_slide_move then lands on that tread/riser EDGE instead of
        # a clean flat top (a blended, non-walkable normal), so no
        # step_height fixes this - confirmed by an offline physics-only
        # simulation reproducing the same stuck-at-the-base result at
        # step_height 0.4 and 10 alike. Source has the exact same
        # limitation for its own (near-identical-sized) player hull, and
        # the standard fix in Source-family mapping is a plain invisible
        # clip brush - a smooth sloped collider laid over the visual
        # stair geometry so the player walks a ramp while the detailed
        # steps stay purely decorative underneath.
        #
        # This box is that clip brush - its endpoints/rotation/length
        # were fit to the actual tread-nosing line (front-top corner of
        # each of the 21 treads, not the mesh's overall bounding-box
        # corners - those include the base plinth below the first tread
        # and are measurably off the real nosing line), which comes out
        # to a ~46.3-degree incline. CharacterController.max_slope_degrees
        # is raised to 47 in app.py to keep that walkable (Source's
        # 45.57-degree default doesn't quite cover it).
        #
        # collision_exclude_local_bounds carves the whole stairwell
        # volume out of plane.glb's own per-triangle mesh collision so
        # this box is the ONLY collider there - leaving the fine stair
        # mesh solid underneath/around it too (which it visually still
        # is) was confirmed to cause erratic movement while descending:
        # a step-settle sweep would land on whichever of the two
        # overlapping surfaces (this ramp, or a tread/riser edge just
        # under it) happened to be closest that tick, flipping the
        # surface normal and travel direction mid-slide.
        self.add_static(
            "Assets/Models/Testmap/plane.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
            collision=True,
            physical_material="concrete",
        )
        # The staircase's own flat top landing (past the last tread, local
        # z:[3.689,4.134] at a constant y=3.665) - also inside the excluded
        # bounds above, so it needs its own simple flat box to stay solid;
        # unrotated, it just butts up against the ramp's top end.

        self.add_static(
            "Assets/Models/Testmap/floorbase.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, 2, 0.0)),
            collision=True,
            # floorbase.glb is a flat quad triangulated as 2 triangles
            # split diagonally through the origin - the player's
            # default spawn (0,2,0) sits exactly on that shared edge.
            # Standing on a mesh-collider's internal triangle seam is a
            # known trigger for BulletCharacterControllerNode jitter
            # (the ground sweep alternates between the two triangles'
            # contact manifolds each substep). A box collider has no
            # internal edges, so it sidesteps the issue entirely - and
            # is the right tool anyway for a flat rectangular slab.
            collision_shape="box",
            physical_material="wood",
        )

        self.add_static(
            "Assets/Models/monkey.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1, 0.0)),
            collision=True,
            physical_material="metal",
        )

        # Scene Dynamic Elements

        self.add_dynamic(
            "Assets/Models/torus.glb",
            position=glm.vec3(0.0, 6.0, 0.0),
            collision=True,
            collision_shape="mesh",
            kinematic=True,
            rot_speed=1,
        )

        # Skeletal Meshes
        character = self.add_skeletal(
            "Assets/Models/rat.glb",
            position=(0, -1, -3),
            animation="funnyrat_ARMAction",
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
