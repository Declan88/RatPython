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
        # baked together with no vertex welding between faces - walking
        # a BulletCharacterControllerNode across any multi-triangle
        # surface like that floor quad catches on Bullet's "internal
        # edge" problem (the sweep test doesn't know adjacent triangles
        # are coplanar, so crossing their shared edge produces a tiny
        # catch/normal glitch each time - confirmed as the cause of a
        # walking jitter). Panda3D's Bullet bindings don't expose
        # Bullet's real fix for this (btGenerateInternalEdgeInfo), so
        # instead this renders the mesh as-is but gives it 3 precise
        # box colliders (extents pulled from the mesh's own connected
        # components, offset by this same (0,-1,0) transform) rather
        # than one triangulated collider - boxes have no internal edges
        # to catch on.
        self.add_static(
            "Assets/Models/Testmap/plane.glb",
            transform=glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, 0.0)),
            collision=False,
        )
        self.physics.add_static_box((4.0, 0.05, 4.0), position=(0.0, -1.0, 0.0))
        self.physics.add_static_box(
            (0.4776, 2.2299, 1.0404), position=(-2.2964, 1.2299, 0.0)
        )
        self.physics.add_static_box(
            (0.5929, 0.277, 0.5929), position=(0.0, -0.723, 0.0)
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
            "Assets/Models/torus.glb",
            position=glm.vec3(0.0, 3.0, 0.0),
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
