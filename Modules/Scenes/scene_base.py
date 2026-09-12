from __future__ import annotations

from pathlib import Path

import moderngl
import glm
import numpy as np

from Modules.Audio.sound_manager import SoundManager
from Modules.Audio.footstep_materials import get_footstep_sound
from Modules.Physics.physics_world import PhysicsWorld, CollisionGroup, to_physics_vec
from Modules.Graphics.pbr_shader import (
    create_program,
    bind_material,
    bind_point_lights,
    bind_environment
)
from Modules.Graphics.shadow_module import CascadedShadowMap
from Modules.Graphics.point_shadow_module import PointShadowMap
from Modules.Graphics.gltf_lights import extract_punctual_lights
from Modules.Graphics.lightmap_baker import (
    create_bake_program,
    create_lightmap,
    bake_point_light
)
from Modules.Graphics.model_loader import load_glb
from Modules.Graphics import lightmap_cache_io
from Modules.Graphics.skeletal_loader import load_skinned_glb, create_skeletal_vao
from Modules.Graphics.skeletal_shader import (
    create_skeletal_program,
    create_skeletal_shadow_program,
    bind_bone_matrices,
)
from Modules.Graphics.skybox import (
    create_skybox_program,
    load_skybox_textures,
    create_skybox_vao,
    render_skybox,
    create_equirect_skybox_program,
    load_equirect_texture,
    create_equirect_skybox_vao,
    render_equirect_skybox,
)


def _release(resource):
    """Best-effort GL resource release - swallows errors since this is
    always called during cleanup/replacement, never on a hot path where
    a failure should be visible."""
    if resource is not None:
        try:
            resource.release()
        except Exception:
            pass


class Scene:
    def __init__(self, ctx, recalculate_shadows=True):
        self.ctx = ctx

        # If True, bake_static_lighting() recomputes lighting from scratch
        # and saves it to disk. If False, it loads the previously-saved
        # lightmaps instead of re-baking, which is much faster - useful
        # once you're happy with the lighting and don't want to pay the
        # bake cost on every launch. Falls back to baking automatically if
        # the cached files aren't there yet (e.g. first run).
        self.recalculate_shadows = recalculate_shadows
        self.lightmap_dir = Path("Assets/Lightmaps") / self.__class__.__name__

        self.static_objects = []
        self.dynamic_objects = []
        self.skeletal_objects = []
        self.point_lights = []
        self.sound_manager = SoundManager()
        self.physics = PhysicsWorld()

        self.light_dir = glm.vec3(0.5, 1.0, 0.8)

        # The directional (sun) light's own color and brightness -
        # separate from light_dir, which only ever controlled its
        # direction (its vector's magnitude does nothing; the shader
        # normalizes it). Defaults match the shader's old hardcoded
        # behavior exactly (implicitly white at a fixed 2.0 multiplier)
        # so existing scenes look unchanged unless a scene overrides
        # these, same pattern as light_dir itself.
        self.light_color = glm.vec3(1.0, 1.0, 1.0)
        self.light_intensity = 2.0

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        self.pbr_program = create_program(self.ctx)

        self.shadow_program = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 u_light_mvp;

                in vec3 in_position;

                void main()
                {
                    gl_Position = u_light_mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                void main()
                {
                }
            """
        )

        self.shadow_manager = CascadedShadowMap(self.ctx)
        self.bake_program = create_bake_program(self.ctx)
        self.skeletal_program = create_skeletal_program(self.ctx)
        self.skeletal_shadow_program = create_skeletal_shadow_program(self.ctx)

        self.skybox_program = create_skybox_program(self.ctx)
        self.skybox_textures = None
        self.skybox_average_colors = None
        self.skybox_vao = None
        self.skybox_vbo = None
        self.skybox_edge_fade = 0.05

        self.equirect_skybox_program = create_equirect_skybox_program(self.ctx)
        self.equirect_skybox_texture = None
        self.equirect_skybox_vao = None
        self.equirect_skybox_vbo = None
        self.equirect_exposure = 1.0
        self.equirect_is_hdr = True

        # Hemisphere ("skylight"-style) ambient term derived from
        # whichever skybox is loaded (see add_skybox/add_equirect_skybox
        # and pbr_shader.py's u_sky_color/u_ground_color) - dim neutral
        # gray until a skybox actually sets these, close to the flat
        # ambient constant this replaced.
        self.environment_sky_color = (0.025, 0.025, 0.025)
        self.environment_ground_color = (0.025, 0.025, 0.025)

    # =============================================================
    # OBJECT LOADING
    # =============================================================

    def _load_object(self, model_path):
        model = load_glb(model_path, self.ctx, self.pbr_program)
        if model is None:
            return None

        shadow_model = load_glb(model_path, self.ctx, self.shadow_program)

        # Only bother with a bake-program VAO if the glb actually has a
        # second UV channel to bake into - otherwise this is wasted work.
        lightmap_model = (
            load_glb(model_path, self.ctx, self.bake_program)
            if model.get("has_lightmap_uv") else None
        )

        if shadow_model is None:
            _release(model.get("vao"))
            if lightmap_model is not None:
                _release(lightmap_model.get("vao"))
            _release(model.get("texture"))
            _release(model.get("metallic_roughness_texture"))
            return None

        return {
            "name": Path(model_path).stem,
            "vao": model["vao"],
            "shadow_vao": shadow_model["vao"],
            "lightmap_vao": lightmap_model["vao"] if lightmap_model else None,
            "has_lightmap_uv": model.get("has_lightmap_uv", False),
            "lightmap_texture": None,
            "texture": model.get("texture"),
            "metallic_roughness_texture": model.get("metallic_roughness_texture"),
            "metallic": model.get("metallic", 0.1),
            "roughness": model.get("roughness", 0.5),
            "emissive": model.get("emissive", [0.0, 0.0, 0.0]),
            "has_texture": model.get("has_texture", 0),
            "has_metallic_roughness_texture": model.get("has_metallic_roughness_texture", 0),
        }

    # =============================================================
    # STATIC OBJECTS
    # =============================================================

    def add_static(self, model_path, position=None, rotation=None, scale=None,
                    transform=None, metallic=None, roughness=None,
                    collision=False, collision_shape="mesh", collision_mask=CollisionGroup.ALL,
                    collision_exclude_local_bounds=None, physical_material=None):
        """collision=True registers a collider for this object in
        self.physics, so a CharacterController (or a dynamic object
        with its own collision=True) can stand/collide on it.
        collision_shape: "mesh" (default) is exact per-triangle
        collision built from model_path's own geometry - the right
        choice for level geometry (floors, walls, ramps). "box" is a
        cheaper axis-aligned-in-local-space box sized from the mesh's
        bounds - fine for simple blocking volumes. collision_mask: see
        CollisionGroup / physics_world.py's module docstring for the
        collision-filtering model - the default (ALL) collides with
        everything. collision_exclude_local_bounds: only for
        collision_shape="mesh" - see PhysicsWorld.add_static_mesh's
        exclude_local_bounds docstring; carves a region out of the mesh
        collision (e.g. one being replaced by a separate simplified
        collider added alongside this call). physical_material: a name
        like "dirt"/"concrete"/"wood" (see Modules/Audio/
        footstep_materials.py for the full set, derived from Assets/
        Audio/footsteps' filenames) - tags this object's collider so
        CharacterController's footstep sounds pick the right sample set
        while standing on it. None falls back to
        footstep_materials.DEFAULT_FOOTSTEP_MATERIAL. Only meaningful
        alongside collision=True."""
        model = self._load_object(model_path)
        if model is None:
            return None

        if metallic is not None:
            model["metallic"] = metallic
        if roughness is not None:
            model["roughness"] = roughness

        if transform is not None:
            model["transform"] = glm.mat4(transform)
        else:
            model["position"] = glm.vec3(position if position is not None else glm.vec3(0.0))
            model["rotation"] = glm.vec3(rotation if rotation is not None else glm.vec3(0.0))
            model["scale"] = glm.vec3(scale if scale is not None else glm.vec3(1.0))

        self.static_objects.append(model)

        # New static geometry invalidates any cached point-light shadow
        # bakes, since they only cover static objects.
        self.mark_static_dirty()

        if collision:
            self._add_static_collision(
                model_path, model, collision_shape, collision_mask,
                collision_exclude_local_bounds, physical_material,
            )

        return model

    def _add_static_collision(self, model_path, model, collision_shape, collision_mask,
                               exclude_local_bounds=None, physical_material=None):
        pos, rot, scl = self._collision_transform_args(model)
        if collision_shape == "mesh":
            self.physics.add_static_mesh(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                exclude_local_bounds=exclude_local_bounds, material=physical_material,
            )
        elif collision_shape == "box":
            self.physics.add_static_box_from_bounds(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                material=physical_material,
            )
        else:
            raise ValueError(f"Unknown collision_shape {collision_shape!r} for add_static - use 'mesh' or 'box'.")

    def _collision_transform_args(self, obj):
        """Returns (position, rotation, scale) in the form PhysicsWorld's
        add_* methods expect, whether obj uses a decomposed position/
        rotation/scale or a single transform-matrix override."""
        if "transform" in obj:
            scale, rot_quat, translation = glm.vec3(), glm.quat(), glm.vec3()
            skew, persp = glm.vec3(), glm.vec4()
            glm.decompose(obj["transform"], scale, rot_quat, translation, skew, persp)
            return translation, glm.eulerAngles(rot_quat), scale
        return (
            obj.get("position", glm.vec3(0.0)),
            obj.get("rotation", glm.vec3(0.0)),
            obj.get("scale", glm.vec3(1.0)),
        )

    # =============================================================
    # DYNAMIC OBJECTS
    # =============================================================

    def add_dynamic(self, model_path, position=glm.vec3(0.0), rotation=glm.vec3(0.0),
                     scale=glm.vec3(1.0), rot_speed=0.0, transform=None,
                     metallic=None, roughness=None,
                     collision=False, collision_shape="box", mass=1.0,
                     collision_mask=CollisionGroup.ALL, gravity=True, kinematic=False,
                     physical_material=None):
        """collision=True hands this object over to self.physics as a
        rigid body (mass, in kg-equivalent units) - from then on its
        position/rotation are driven by the physics simulation every
        frame (see Scene.update). collision_shape: "box" (default) or
        "sphere" are cheap primitives sized from the mesh's own
        bounds; "mesh" is a convex hull built from the mesh's actual
        vertices (Bullet requires a convex shape for anything that
        moves, so this is as exact as a dynamic body can get - use it
        for oddly-shaped props where a box/sphere would clip visibly).

        gravity=False (only meaningful alongside collision=True,
        ignored if kinematic=True) keeps full collision response -
        other bodies can still push it and it can still push them -
        but exempts just this body from falling, via a per-body
        gravity override rather than the world's own gravity. Still a
        normal DYNAMIC body otherwise, so any collision impulse (the
        player walking into it, another prop landing on it) is free to
        knock it around - fine for a floating-but-shovable obstacle,
        wrong for something that should hold a fixed path (see
        kinematic below for that case).

        kinematic=True (also only meaningful alongside collision=True)
        makes this body immovable by any physical force or collision
        impulse whatsoever - other things still collide against it
        solidly, but nothing can ever push, knock, or otherwise budge
        IT. rot_speed keeps working normally for a kinematic object
        (driving its actual transform every frame, same as with no
        collision at all) rather than being replaced by a real angular
        velocity the way it is for a plain dynamic body - use this for
        a spinning/moving obstacle that must follow an exact path
        regardless of what bumps into it. gravity is meaningless here
        (a kinematic body is never affected by it either way).

        See CollisionGroup / physics_world.py's module docstring for
        collision_mask."""
        model = self._load_object(model_path)
        if model is None:
            return None

        if metallic is not None:
            model["metallic"] = metallic
        if roughness is not None:
            model["roughness"] = roughness

        if transform is not None:
            model["transform"] = glm.mat4(transform)
        else:
            model["position"] = glm.vec3(position)
            model["rotation"] = glm.vec3(rotation)
            model["scale"] = glm.vec3(scale)

        model["rot_speed"] = float(rot_speed)
        self.dynamic_objects.append(model)

        if collision:
            self._add_dynamic_collision(
                model_path, model, collision_shape, mass, collision_mask, gravity, kinematic, physical_material,
            )

        return model

    def _add_dynamic_collision(self, model_path, model, collision_shape, mass, collision_mask,
                                gravity=True, kinematic=False, physical_material=None):
        pos, rot, scl = self._collision_transform_args(model)
        rot_speed = model.get("rot_speed", 0.0)

        # Physics (or, for a kinematic body, THIS method's own caller -
        # see Scene.update) owns position/rotation from here on -
        # replace any transform-matrix override with the equivalent
        # decomposed fields so _get_model_matrix keeps following it
        # every frame. For a plain dynamic body, rot_speed itself is
        # zeroed here (Scene.update() ignores it entirely for anything
        # with a _physics_body that isn't _kinematic - it'd otherwise
        # fight the physics rotation applied each frame) but its VALUE
        # is kept above and re-applied below as a real angular velocity
        # on the rigid body instead, so a spinning prop keeps spinning
        # through actual physics rather than silently stopping the
        # moment collision is turned on. A kinematic body keeps
        # rot_speed working exactly as it always did (Scene.update
        # applies it directly to model["rotation"] every frame, same
        # as a non-colliding object) since IT drives the transform,
        # not the other way around.
        model.pop("transform", None)
        model["position"] = pos
        model["rotation"] = rot
        model["scale"] = scl
        if not kinematic:
            model["rot_speed"] = 0.0
        model["_kinematic"] = kinematic

        if collision_shape == "box":
            body = self.physics.add_dynamic_box_from_bounds(
                model_path, position=pos, rotation=rot, scale=scl, mass=mass,
                collision_mask=collision_mask, gravity=gravity, kinematic=kinematic,
                material=physical_material,
            )
        elif collision_shape == "sphere":
            body = self.physics.add_dynamic_sphere_from_bounds(
                model_path, position=pos, scale=scl, mass=mass,
                collision_mask=collision_mask, gravity=gravity, kinematic=kinematic,
                material=physical_material,
            )
        elif collision_shape == "mesh":
            body = self.physics.add_dynamic_mesh(
                model_path, position=pos, rotation=rot, scale=scl, mass=mass,
                collision_mask=collision_mask, gravity=gravity, kinematic=kinematic,
                material=physical_material,
            )
        else:
            raise ValueError(f"Unknown collision_shape {collision_shape!r} for add_dynamic - use 'box', 'sphere', or 'mesh'.")

        model["_physics_body"] = body

        if rot_speed != 0.0 and not kinematic:
            # Real angular velocity (render Y axis - vertical spin,
            # the same axis rot_speed always meant) rather than the
            # dead rot_speed field, so this keeps spinning through
            # actual physics and still fully participates in collision
            # (something bumping into it interacts with its real,
            # continuously-changing orientation). Not applicable to a
            # kinematic body - it ignores velocity-based integration
            # entirely by design, which is exactly what makes it
            # immovable; rot_speed drives it directly instead, pushed
            # into the physics transform each frame by Scene.update.
            body.node().setAngularVelocity(to_physics_vec(glm.vec3(0.0, rot_speed, 0.0)))

    # =============================================================
    # SKELETAL (ANIMATED) OBJECTS
    # =============================================================

    def add_skeletal(self, model_path, position=None, rotation=None, scale=None,
                      transform=None, animation=None, metallic=None, roughness=None,
                      emissive=None, texture_path=None,
                      visible_in_color=True, cast_shadow=True,
                      upper_body_root_joints=None, upper_animation=None):
        """Loads a skinned/animated glb - see skeletal_loader.py for
        format constraints (one skin, one mesh primitive, LINEAR/STEP
        interpolation only).

        The glb's own material (base color texture, metallic/roughness
        factors, emissive, metallic-roughness texture) is used
        automatically. Pass metallic/roughness/emissive/texture_path
        explicitly only if you want to override what's actually authored
        in the file.

        animation: name of the clip to start playing immediately (loops
        automatically once it reaches the end). Pass None to start in the
        bind/rest pose with nothing playing yet - use
        set_skeletal_animation() later to start one.

        visible_in_color/cast_shadow: both default True (existing
        behavior, unchanged for every current caller). Set
        visible_in_color=False to still cast a shadow every frame (see
        _render_shadows) without ever being drawn in the normal color
        pass - built for a first-person player's own body model, which
        should shadow the ground but never actually be seen by its own
        camera. cast_shadow=False is the inverse (visible, no shadow).
        Bone matrices are still recomputed every frame in Scene.update()
        regardless of either flag, since a shadow-only object still
        needs correct bones for its shadow draw.

        upper_body_root_joints: optional list of joint names (see
        Skeleton.compute_joint_mask - a list, not a single pivot joint,
        since some rigs split into more than one subtree there). When
        given, `animation`/set_skeletal_animation drive every OTHER
        joint (the "lower body") while a second, independently-timed
        clip - set via upper_animation here or set_skeletal_upper_
        animation() later - drives just the masked joints, composited
        into one skeleton each frame (Skeleton.
        compute_blended_bone_matrices). Leave this None (the default)
        for the existing single-clip behavior - untouched for every
        current caller.

        Skeletal objects are always real-time only, never lightmap-baked
        - they're inherently dynamic (animated), so
        bake_static_lighting() never looks at this list at all, same as
        dynamic_objects already doesn't."""
        data = load_skinned_glb(model_path, ctx=self.ctx)
        if data is None:
            return None

        render_vao_info = create_skeletal_vao(self.ctx, self.skeletal_program, data)
        shadow_vao_info = create_skeletal_vao(self.ctx, self.skeletal_shadow_program, data)

        texture = data.get("texture")
        if texture_path is not None:
            # Explicit override - release whatever the glb's own material
            # provided (if anything) and use this instead.
            _release(texture)
            from PIL import Image
            img = Image.open(texture_path).convert("RGB")
            texture = self.ctx.texture(img.size, 3, img.tobytes())
            texture.build_mipmaps()
            texture.repeat_x = texture.repeat_y = True

        mr_texture = data.get("metallic_roughness_texture")
        final_metallic = metallic if metallic is not None else data.get("metallic", 0.1)
        final_roughness = roughness if roughness is not None else data.get("roughness", 0.5)
        final_emissive = emissive if emissive is not None else data.get("emissive", [0.0, 0.0, 0.0])

        skeleton = data["skeleton"]
        upper_joint_mask = (
            skeleton.compute_joint_mask(upper_body_root_joints)
            if upper_body_root_joints is not None else None
        )

        if upper_joint_mask is not None:
            initial_bones = skeleton.compute_blended_bone_matrices(
                animation, 0.0, upper_animation, 0.0, upper_joint_mask
            )
        elif animation is not None:
            initial_bones = skeleton.compute_bone_matrices(animation, 0.0)
        else:
            initial_bones = [glm.mat4(1.0) for _ in skeleton.joints]

        obj = {
            "vao": render_vao_info["vao"],
            "shadow_vao": shadow_vao_info["vao"],
            "_render_vbos": render_vao_info["vbos"],
            "_render_ibo": render_vao_info["ibo"],
            "_shadow_vbos": shadow_vao_info["vbos"],
            "_shadow_ibo": shadow_vao_info["ibo"],
            "skeleton": skeleton,
            "animation": animation,
            "anim_time": 0.0,
            "upper_joint_mask": upper_joint_mask,
            "upper_animation": upper_animation,
            "upper_anim_time": 0.0,
            "bone_matrices": initial_bones,
            "texture": texture,
            "metallic_roughness_texture": mr_texture,
            "metallic": float(final_metallic),
            "roughness": float(final_roughness),
            "emissive": list(final_emissive),
            "has_texture": 1 if texture else 0,
            "has_metallic_roughness_texture": 1 if mr_texture else 0,
            "has_lightmap_uv": False,
            "lightmap_texture": None,
            "visible_in_color": bool(visible_in_color),
            "cast_shadow": bool(cast_shadow),
        }

        if transform is not None:
            obj["transform"] = glm.mat4(transform)
        else:
            obj["position"] = glm.vec3(position) if position is not None else glm.vec3(0.0)
            obj["rotation"] = glm.vec3(rotation) if rotation is not None else glm.vec3(0.0)
            obj["scale"] = glm.vec3(scale) if scale is not None else glm.vec3(1.0)

        self.skeletal_objects.append(obj)
        return obj

    def set_skeletal_animation(self, obj, animation_name):
        """Switches obj to a different animation clip, restarting from
        time 0. animation_name must exist in obj["skeleton"].animations
        (print obj["skeleton"].animations.keys() to see what a loaded
        glb actually has)."""
        obj["animation"] = animation_name
        obj["anim_time"] = 0.0

    def set_skeletal_upper_animation(self, obj, animation_name):
        """The upper-body equivalent of set_skeletal_animation - only
        meaningful for an obj created with upper_body_root_joints set
        (see add_skeletal); switches just the masked joints' clip,
        restarting THEIR time from 0 independently of the lower-body
        anim_time. Calling this on an obj without a mask configured is
        harmless (the fields get set but nothing ever reads them, since
        Scene.update()'s bone recompute only takes the blended path when
        obj["upper_joint_mask"] is not None)."""
        obj["upper_animation"] = animation_name
        obj["upper_anim_time"] = 0.0

    # =============================================================
    # SKYBOX
    # =============================================================

    def add_skybox(self, face_paths, tint=None, rotations=None, top_height=1.0,
                    bottom_height=-1.0, half_extent=1.0, padding=0, edge_fade=0.05):
        """face_paths: sequence of exactly 6 image file paths, in order
        +X, -X, +Y, -Y, +Z, -Z (same face-order convention already used
        by PointShadowMap elsewhere in this project). Each face is its
        own plain 2D texture at its native resolution/aspect - not
        forced square or matched in size, unlike an earlier version of
        this built on a hardware cubemap (see skybox.py's module
        docstring for why that was the wrong approach here).

        tint: optional (r, g, b) color correction applied to all 6 faces
        (or a list of 6, one per face).

        rotations: optional list of 6 degree values (0/90/180/270), one
        per face - for a face whose source material rotates its texture.

        top_height/bottom_height/half_extent: box shape in world units.
        Defaults make a full symmetric cube (every face, including the
        top/bottom caps, exactly 1:1) - right when all 6 textures are the
        same square size. Since the skybox's view matrix has translation
        stripped, the camera is always effectively at local origin
        (0,0,0) inside this box - bottom_height MUST stay strictly
        negative (floor below the camera), never 0.0 or positive, or
        most viewing directions won't intersect the box's geometry at
        all (see skybox.py's create_skybox_vao docstring).

        padding: pixels of edge-replication padding added to each face
        (see skybox.py's load_skybox_textures docstring - with the
        current per-face UV mapping and clamp-to-edge already active,
        this doesn't change what's visible; it won't fix the harsh seam
        between different faces either, since that's caused by zero
        blending between separate textures, not edge sampling).

        edge_fade: UV-space margin each face fades toward black over,
        softening the seam where two different face textures meet - see
        skybox.py's render_skybox docstring. Default 0.05; set to 0.0
        for hard, unfaded edges (the original behavior).

        Only one skybox per scene - calling this again replaces the
        previous one (releasing its GPU resources first)."""
        if self.skybox_textures is not None:
            for tex in self.skybox_textures:
                tex.release()
        if self.skybox_vao is not None:
            self.skybox_vao.release()
        if self.skybox_vbo is not None:
            self.skybox_vbo.release()

        self.skybox_textures, self.skybox_average_colors = load_skybox_textures(
            self.ctx, face_paths, tint=tint, padding=padding
        )
        self.skybox_vao, self.skybox_vbo = create_skybox_vao(
            self.ctx, self.skybox_program, top_height=top_height,
            bottom_height=bottom_height, half_extent=half_extent, rotations=rotations
        )
        self.skybox_edge_fade = edge_fade

        # Approximate hemisphere ambient from the +Y/-Y face averages
        # (face order is +X,-X,+Y,-Y,+Z,-Z - see load_skybox_textures)
        # - see environment_sky_color's definition in __init__. These
        # average_colors are gamma-encoded display values (they're only
        # otherwise used for on-screen edge-fade blending), not
        # gamma-decoded to linear the way add_equirect_skybox's are, so
        # this is a cruder approximation than that path - acceptable
        # for a subtle ambient term on a shader that's already not
        # claiming photometric accuracy (see pbr_shader.py's docstring).
        self.environment_sky_color = self.skybox_average_colors[2]
        self.environment_ground_color = self.skybox_average_colors[3]

    def add_equirect_skybox(self, path, exposure=1.0):
        """Loads a single equirectangular panorama as the skybox, sampled
        directly by direction vector - no discrete faces, so none of
        add_skybox()'s seam-blending machinery (edge_fade, average-color
        neighbor blending) applies or is needed here.

        Accepts either HDR (.exr, requires the OpenEXR package) or LDR
        (.png/.jpg/etc, via PIL, no extra dependency) - auto-detected
        from the file extension. See skybox.py's load_equirect_texture
        docstring for the tonemapping difference between the two.

        exposure: brightness multiplier applied before tonemapping - the
        standard HDRI exposure control, since raw radiance values don't
        have one inherently "correct" display brightness. Still applies
        for LDR sources too (plain brightness multiplier).

        Don't call this alongside add_skybox() in the same scene - both
        would render, wastefully (whichever draws wouldn't be dangerous,
        just redundant with the other)."""
        if self.equirect_skybox_texture is not None:
            self.equirect_skybox_texture.release()
        if self.equirect_skybox_vao is not None:
            self.equirect_skybox_vao.release()
        if self.equirect_skybox_vbo is not None:
            self.equirect_skybox_vbo.release()

        self.equirect_skybox_texture, self.equirect_is_hdr, sky_color, ground_color = load_equirect_texture(
            self.ctx, path
        )
        self.equirect_skybox_vao, self.equirect_skybox_vbo = create_equirect_skybox_vao(
            self.ctx, self.equirect_skybox_program
        )
        self.equirect_exposure = exposure

        # See environment_sky_color's definition in __init__ - already
        # linear (load_equirect_texture handles LDR gamma-decoding), so
        # just apply the same exposure multiplier the skybox itself
        # renders with, for a consistent look between the visible sky
        # and its ambient contribution.
        self.environment_sky_color = tuple(c * exposure for c in sky_color)
        self.environment_ground_color = tuple(c * exposure for c in ground_color)

    # =============================================================
    # POINT LIGHTS
    # =============================================================

    def add_point_light(self, position, color=(1.0, 1.0, 1.0), intensity=1.0,
                         radius=10.0, cast_shadows=False):
        """cast_shadows now means "shadow-test this light against static
        geometry during bake_static_lighting()", not "give it a real-time
        cubemap" - point lights are always unshadowed in real time (see
        pbr_shader.py's docstring). No cap on how many lights can request
        this: baking processes lights one at a time, so it doesn't hit
        the shader register-limit issue a large real-time array would."""
        light = {
            "position": glm.vec3(position),
            "color": glm.vec3(color),
            "intensity": float(intensity),
            "radius": float(radius),
            "bake_shadows": bool(cast_shadows),
        }
        self.point_lights.append(light)
        return light

    def add_lights_from_glb(self, model_path, cast_shadows=False, default_radius=8.0,
                             intensity_multiplier=1.0, radius_multiplier=1.0):
        """Reads KHR_lights_punctual lights out of a glb and adds any
        point lights found as real point lights in the scene.

        intensity_multiplier: scales the converted intensity (see the
        unit-conversion note in gltf_lights.extract_punctual_lights) -
        useful since the candela->linear conversion is only approximate
        and glTF-authored lights often end up too dim/bright as-is.

        radius_multiplier: scales the light's falloff radius (from the
        glTF "range" field, or default_radius if range wasn't authored).
        Useful because "range" is just a culling hint in the glTF spec,
        not a value tuned for your shader's specific falloff curve."""
        added = []

        for light in extract_punctual_lights(model_path):
            if light["type"] != "point":
                print(
                    f"[Scene] Skipping '{light['type']}' light from "
                    f"{model_path} - only 'point' lights are supported "
                    f"(no spot cone / directional handling)."
                )
                continue

            added.append(self.add_point_light(
                position=light["position"],
                color=light["color"],
                intensity=(light["intensity"] or 1.0) * intensity_multiplier,
                radius=(light["range"] or default_radius) * radius_multiplier,
                cast_shadows=cast_shadows,
            ))

        return added

    def mark_static_dirty(self):
        """No-op. Kept only so existing call sites like add_static() don't
        break. There's no runtime point-light shadow cache to invalidate
        anymore - point lights are always unshadowed in real time, and
        baked shadows are recomputed by explicitly calling
        bake_static_lighting() again, not by a dirty flag."""
        pass

    # =============================================================
    # 3D SOUND
    # =============================================================

    def update_audio(self, camera):
        """Call once per frame from your main loop. Sound emitters
        themselves live on self.sound_manager - use
        scene.sound_manager.add_sound(...) to add one, not a method on
        Scene (see Modules/Audio/sound_manager.py)."""
        self.sound_manager.update(camera)

    def play_footstep_sound(self, material, position, volume=1.0):
        """One-shot footstep sample for `material` (see Modules/Audio/
        footstep_materials.py - falls back to
        footstep_materials.DEFAULT_FOOTSTEP_MATERIAL if material is
        None or unrecognized), played as a normal positional emitter
        through self.sound_manager - same distance falloff/panning
        every other 3D sound in the scene gets (see SoundManager.update),
        so footsteps attenuate with distance exactly like everything
        else rather than needing separate logic here.

        Intended to be driven by CharacterController.pop_footstep() -
        see app.py's main loop - once per footstep, not once per frame.
        loop=False means SoundManager.update automatically drops the
        emitter once playback finishes, so nothing here needs to track
        or manually destroy the sound afterward."""
        sound_path = get_footstep_sound(material)
        if sound_path is None:
            return None
        return self.sound_manager.add_sound(
            sound_path, position, volume=volume,
            min_distance=1.0, max_distance=12.0, loop=False,
        )

    # =============================================================
    # LIGHTMAP BAKING
    # =============================================================

    def bake_static_lighting(self, lightmap_resolution=256, point_shadow_resolution=1024):
        """Call this once, after adding all static objects and point
        lights, to bake shadow-tested point light contributions (from
        lights added with cast_shadows=True) into each static object's
        lightmap. Lights are baked one at a time (additively blended) -
        see lightmap_baker.py's docstring for why that matters.

        The directional light is deliberately NEVER baked here - it
        stays fully real-time via CascadedShadowMap for every object,
        static or dynamic (see _render_shadows). Baking it too would
        double-count it: the runtime shader already adds a real-time,
        correctly-shadowed directional term for every object regardless
        of whether it has a lightmap, so an object with both a baked
        directional contribution AND the real-time one would show it
        twice, at roughly double brightness.

        Point lights added with cast_shadows=False are not baked at all;
        they stay real-time-unshadowed only (see add_point_light)."""
        eligible = [
            obj for obj in self.static_objects
            if obj.get("has_lightmap_uv") and obj.get("lightmap_vao") is not None
        ]

        # This runs once, typically from a Scene subclass's __init__ -
        # i.e. before the app's main loop has necessarily set a real
        # window viewport. Capturing/restoring self.ctx.viewport like the
        # per-frame render passes do isn't safe here, since that snapshot
        # could just be moderngl's early default rather than the actual
        # window size - restore to the real framebuffer size explicitly
        # instead.
        restore_viewport = (0, 0, *self.ctx.screen.size)

        if not eligible:
            print(
                f"[Scene] bake_static_lighting: no static objects have lightmap UVs - "
                f"nothing to bake ({len(self.static_objects)} static object(s) checked)."
            )
            for i, obj in enumerate(self.static_objects):
                print(f"  static_objects[{i}]: has_lightmap_uv={obj.get('has_lightmap_uv')}")
            return

        self.lightmap_dir.mkdir(parents=True, exist_ok=True)

        # Named after the object (its model file's stem - see
        # _load_object) rather than a bare index, so cache files are
        # identifiable on disk (e.g. floorbase.exr instead of
        # lightmap_0.exr) - disambiguated with a _2, _3, ... suffix on
        # any repeat, since two static objects loaded from the same
        # model (or coincidentally sharing a stem) would otherwise
        # collide on the same cache file.
        seen_name_counts = {}
        cache_names = []
        for obj in eligible:
            name = obj["name"]
            seen_name_counts[name] = seen_name_counts.get(name, 0) + 1
            count = seen_name_counts[name]
            cache_names.append(name if count == 1 else f"{name}_{count}")
        cache_paths = [
            lightmap_cache_io.lightmap_cache_path(self.lightmap_dir, name)
            for name in cache_names
        ]

        def _load_cache():
            """Returns the loaded arrays if every cache file exists AND
            matches the resolutions currently being requested, else None.
            This check is what stops the cache from silently reusing
            stale data when lightmap_resolution or point_shadow_resolution
            change between runs."""
            loaded = []
            for path in cache_paths:
                array = lightmap_cache_io.load_lightmap_cache(
                    path, lightmap_resolution, point_shadow_resolution
                )
                if array is None:
                    return None
                loaded.append(array)
            return loaded

        if not self.recalculate_shadows:
            cached_arrays = _load_cache()
            if cached_arrays is not None:
                for obj, array in zip(eligible, cached_arrays):
                    _release(obj.get("lightmap_texture"))

                    # .astype/.tobytes() rather than relying on the loaded
                    # array's dtype directly - guards against a cache file
                    # ever ending up float32 (e.g. from a different numpy
                    # version) not matching the f2 (half-float) texture
                    # format below.
                    #
                    # 3 components (RGB), not 4: the RGBA requirement only
                    # applied during baking, because GL_RGB16F isn't a
                    # guaranteed-renderable framebuffer format. This
                    # texture is only ever sampled at runtime, never
                    # rendered into again, so that constraint doesn't
                    # apply here - no reason to carry a wasted alpha
                    # channel through disk storage and back.
                    array = array.astype(np.float16)
                    texture = self.ctx.texture(
                        (array.shape[1], array.shape[0]), 3, array.tobytes(), dtype="f2"
                    )
                    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    obj["lightmap_texture"] = texture

                return

            print(
                f"[Scene] recalculate_shadows is False but no valid cached "
                f"lightmaps found in {self.lightmap_dir} - baking instead."
            )

        print(f"[Scene] Baking lighting for {len(eligible)} static object(s)...")

        for obj in eligible:
            _release(obj.get("lightmap_texture"))
            obj["lightmap_texture"] = create_lightmap(self.ctx, lightmap_resolution)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE

        # Point lights flagged for baked shadows, one at a time. Each
        # gets its own temporary 6-face shadow cube (static geometry
        # only), used just for this bake, then destroyed.
        for light in self.point_lights:
            if not light.get("bake_shadows"):
                continue

            temp_shadow = PointShadowMap(
                self.ctx, resolution=point_shadow_resolution,
                near=0.05, far=max(light["radius"] * 2.0, 1.0)
            )
            temp_shadow.set_position(light["position"])

            # No culling for this depth pass. Front-face culling only
            # gives correct results for closed/watertight geometry (it
            # relies on a "back" face still being there to record depth
            # once the front is culled) - for thin, single-sided walls
            # (a flat quad with no backside), culling the front face on
            # certain sides of the light can remove the wall from this
            # pass ENTIRELY, leaving zero depth data recorded there. No
            # depth data means the shadow test can never find an
            # occluder, so light passes straight through as if the wall
            # weren't there - which is exactly "bleeding through walls"
            # rather than a subtler bias/precision artifact. Rendering
            # both sides is the safe choice for arbitrary/open geometry;
            # the existing depth bias in the shadow test should still
            # handle ordinary acne fine without front-face culling's help.
            self.ctx.disable(moderngl.CULL_FACE)

            for face in range(6):
                fbo = temp_shadow.live_fbos[face]
                fbo.use()
                self.ctx.viewport = (0, 0, temp_shadow.resolution, temp_shadow.resolution)
                fbo.clear(depth=1.0)
                light_vp = temp_shadow.light_mvps[face]
                for obj in self.static_objects:
                    light_mvp = light_vp * self._get_model_matrix(obj)
                    self.shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                    obj["shadow_vao"].render()

            for obj in eligible:
                bake_point_light(
                    self.ctx, self.bake_program, obj, self._get_model_matrix(obj),
                    light, temp_shadow
                )

            temp_shadow.destroy()

        for obj, path in zip(eligible, cache_paths):
            texture = obj["lightmap_texture"]
            width, height = texture.size
            array = np.frombuffer(texture.read(), dtype=np.float16).reshape(height, width, 4)
            # Drop the alpha channel before persisting - it's unused dead
            # weight here (see the load path above for why).
            lightmap_cache_io.save_lightmap_cache(
                path, array[:, :, :3], lightmap_resolution, point_shadow_resolution
            )

        print(f"[Scene] Baked and saved {len(eligible)} lightmap(s) to {self.lightmap_dir}")

        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.screen.use()
        self.ctx.viewport = restore_viewport
        self.ctx.depth_func = "<"
        self.ctx.cull_face = "back"

    # =============================================================
    # MODEL MATRIX
    # =============================================================

    def _get_model_matrix(self, obj):
        if "transform" in obj:
            return glm.mat4(obj["transform"])

        model = glm.mat4(1.0)
        model = glm.translate(model, obj["position"])

        rotation = obj["rotation"]
        model = glm.rotate(model, rotation.x, glm.vec3(1.0, 0.0, 0.0))
        model = glm.rotate(model, rotation.y, glm.vec3(0.0, 1.0, 0.0))
        model = glm.rotate(model, rotation.z, glm.vec3(0.0, 0.0, 1.0))

        model = glm.scale(model, obj["scale"])

        return model

    # =============================================================
    # UPDATE
    # =============================================================

    def update(self, dt):
        # Step collision/physics first so this frame's dynamic-object
        # sync below (and any CharacterController.get_position() calls
        # the caller makes after this) reflect where things just moved.
        self.physics.step(dt)

        for obj in self.dynamic_objects:
            physics_body = obj.get("_physics_body")
            if physics_body is not None:
                if obj.get("_kinematic"):
                    # WE drive a kinematic body's transform, not the
                    # other way around (see add_dynamic's docstring) -
                    # apply rot_speed exactly like a non-colliding
                    # object below, then push the result onto the
                    # physics node so Bullet uses it for collision.
                    rot_speed = obj.get("rot_speed", 0.0)
                    if rot_speed != 0.0 and "rotation" in obj:
                        obj["rotation"].y += rot_speed * dt
                    self.physics.set_transform(physics_body, obj["position"], obj["rotation"])
                else:
                    obj["position"], obj["rotation"] = self.physics.get_transform(physics_body)
                continue

            rot_speed = obj.get("rot_speed", 0.0)
            if rot_speed != 0.0 and "rotation" in obj:
                obj["rotation"].y += rot_speed * dt

        for obj in self.skeletal_objects:
            skeleton = obj["skeleton"]
            upper_mask = obj.get("upper_joint_mask")

            # Lower-body (or, with no upper_joint_mask, the object's only)
            # clip time advance - unchanged from before per-object.
            if obj["animation"] is not None:
                clip = skeleton.animations.get(obj["animation"])
                obj["anim_time"] = (
                    (obj["anim_time"] + dt) % clip.duration if clip is not None and clip.duration > 0.0 else 0.0
                )

            if upper_mask is not None:
                # Upper-body clip advances independently of the lower
                # one - see add_skeletal's upper_body_root_joints and
                # Skeleton.compute_blended_bone_matrices. Runs even if
                # obj["animation"] is None (lower body just sits in bind
                # pose while the upper body still plays) and even if
                # obj["upper_animation"] is None (compute_blended_bone_
                # matrices then falls back to the lower clip for every
                # joint, matching the single-clip behavior exactly).
                if obj["upper_animation"] is not None:
                    upper_clip = skeleton.animations.get(obj["upper_animation"])
                    obj["upper_anim_time"] = (
                        (obj["upper_anim_time"] + dt) % upper_clip.duration
                        if upper_clip is not None and upper_clip.duration > 0.0 else 0.0
                    )
                obj["bone_matrices"] = skeleton.compute_blended_bone_matrices(
                    obj["animation"], obj["anim_time"],
                    obj["upper_animation"], obj["upper_anim_time"],
                    upper_mask,
                )
                continue

            if obj["animation"] is None:
                continue
            obj["bone_matrices"] = skeleton.compute_bone_matrices(obj["animation"], obj["anim_time"])

    # =============================================================
    # DIRECTIONAL SHADOW PASS
    # =============================================================

    def _render_shadows(self, camera):
        self.shadow_manager.update(camera, self.light_dir)
        resolution = self.shadow_manager.resolution

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "front"

        old_viewport = self.ctx.viewport

        for cascade in range(self.shadow_manager.num_cascades):
            framebuffer = self.shadow_manager.framebuffers[cascade]
            light_vp = self.shadow_manager.light_mvps[cascade]

            framebuffer.use()
            self.ctx.viewport = (0, 0, resolution, resolution)
            framebuffer.clear(depth=1.0)

            # Static and dynamic objects share the same (non-skinned)
            # shadow program, so they're drawn the same way here.
            for obj in (*self.static_objects, *self.dynamic_objects):
                light_mvp = light_vp * self._get_model_matrix(obj)
                self.shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                obj["shadow_vao"].render()

            for obj in self.skeletal_objects:
                if not obj.get("cast_shadow", True):
                    continue
                light_mvp = light_vp * self._get_model_matrix(obj)
                self.skeletal_shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                bind_bone_matrices(self.skeletal_shadow_program, obj["bone_matrices"])
                obj["shadow_vao"].render()

        self.ctx.screen.use()
        self.ctx.viewport = old_viewport

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

    # =============================================================
    # PBR PASS
    # =============================================================

    def _render_scene(self, camera):
        self.ctx.screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        # Point-light data and shadow textures are identical for every
        # object this frame, so bind them once here rather than inside
        # the per-object loop below. Done separately for pbr_program and
        # skeletal_program - they're two distinct compiled GL programs,
        # each with its own uniform locations, so binding one doesn't
        # affect the other even though the uniform names match.
        bind_point_lights(self.pbr_program, self.point_lights)
        bind_point_lights(self.skeletal_program, self.point_lights)
        bind_environment(self.pbr_program, self.environment_sky_color, self.environment_ground_color)
        bind_environment(self.skeletal_program, self.environment_sky_color, self.environment_ground_color)

        for obj in (*self.static_objects, *self.dynamic_objects):
            model_matrix = self._get_model_matrix(obj)
            bind_material(
                self.pbr_program, obj, model_matrix, camera,
                self.light_dir, self.shadow_manager,
                light_color=self.light_color, light_intensity=self.light_intensity
            )
            obj["vao"].render()

        for obj in self.skeletal_objects:
            if not obj.get("visible_in_color", True):
                continue
            model_matrix = self._get_model_matrix(obj)

            # bind_material is fully generic on prog - reused as-is here
            # rather than duplicating a "bind_skeletal_material":
            # skeletal_program declares the exact same uniform names (it
            # reuses pbr_shader's fragment shader verbatim, see
            # skeletal_shader.py's docstring), so this works unmodified.
            bind_material(
                self.skeletal_program, obj, model_matrix, camera,
                self.light_dir, self.shadow_manager,
                light_color=self.light_color, light_intensity=self.light_intensity
            )
            bind_bone_matrices(self.skeletal_program, obj["bone_matrices"])
            obj["vao"].render()

    # =============================================================
    # RENDER
    # =============================================================

    def render(self, camera, prog=None):
        self._render_shadows(camera)
        self._render_scene(camera)

        if self.skybox_textures is not None:
            render_skybox(
                self.ctx, self.skybox_program, self.skybox_vao, self.skybox_textures,
                self.skybox_average_colors, camera, edge_fade=self.skybox_edge_fade
            )

        if self.equirect_skybox_texture is not None:
            render_equirect_skybox(
                self.ctx, self.equirect_skybox_program, self.equirect_skybox_vao,
                self.equirect_skybox_texture, camera, exposure=self.equirect_exposure,
                apply_tonemap=self.equirect_is_hdr
            )

    # =============================================================
    # DESTROY
    # =============================================================

    def destroy(self):
        for obj in self.static_objects:
            self._release_object(obj)
        for obj in self.dynamic_objects:
            self._release_object(obj)
        for obj in self.skeletal_objects:
            self._release_skeletal_object(obj)

        self.point_lights.clear()
        self.sound_manager.destroy()
        self.physics.destroy()

        if self.shadow_manager is not None:
            try:
                self.shadow_manager.destroy()
            except Exception:
                pass
            self.shadow_manager = None

        # Every other GL program/VAO/VBO/texture the Scene itself owns
        # (as opposed to per-object resources, released above) follows
        # the same release-then-clear pattern, so it's just a loop.
        program_and_buffer_attrs = (
            "pbr_program", "shadow_program", "bake_program",
            "skeletal_program", "skeletal_shadow_program",
            "skybox_program", "skybox_vao", "skybox_vbo",
            "equirect_skybox_program", "equirect_skybox_texture",
            "equirect_skybox_vao", "equirect_skybox_vbo",
        )
        for attr in program_and_buffer_attrs:
            _release(getattr(self, attr))
            setattr(self, attr, None)

        if self.skybox_textures is not None:
            for tex in self.skybox_textures:
                _release(tex)
            self.skybox_textures = None
            self.skybox_average_colors = None

        self.static_objects.clear()
        self.dynamic_objects.clear()
        self.skeletal_objects.clear()

    def _release_skeletal_object(self, obj):
        for key in ("vao", "shadow_vao", "_render_ibo", "_shadow_ibo", "texture"):
            _release(obj.get(key))

        for vbo_dict_key in ("_render_vbos", "_shadow_vbos"):
            for vbo in obj.get(vbo_dict_key, {}).values():
                _release(vbo)

    def _release_object(self, obj):
        for key in (
            "vao", "shadow_vao", "lightmap_vao", "lightmap_texture",
            "texture", "metallic_roughness_texture",
        ):
            _release(obj.get(key))
