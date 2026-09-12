"""Generic skeletal-model attachment for a player (local or remote) -
composes Scene.add_skeletal's own generic-glb loading with animation-
state switching, so any skinned model can be hung off a player without
hardcoding a specific asset anywhere in here. See app.py (local player,
visible_in_color=False so only its shadow shows) and
Modules/Networking/remote_player.py (remote players, fully visible plus
a hitbox) for the two current call sites.
"""

import glm


class PlayerModel:
    def __init__(self, scene, model_path, position=None, forward_offset_degrees=0.0,
                 feet_offset=0.0, visible_in_color=True, cast_shadow=True,
                 idle_animation=None, walk_animation=None, run_animation=None,
                 walk_speed_threshold=90.0 * 0.0254, run_speed_threshold=220.0 * 0.0254,
                 upper_body_root_joints=None, upper_animation=None,
                 scale=None, metallic=None, roughness=None, emissive=None, texture_path=None):
        """scene: the Scene this model is added to (needs scene.add_skeletal
        and scene.set_skeletal_animation).

        forward_offset_degrees/feet_offset: visual-correction knobs, same
        spirit as skybox.py's own `rotations` param - a glb's authored
        rest-pose forward axis and origin height can't be known without
        actually looking at it rendered, so these exist to tune a
        specific model after checking in-engine rather than guessing
        silently. feet_offset shifts the model vertically relative to
        whatever position update() is given (see update()'s own
        docstring for the feet-vs-hull-center contract that motivates
        this); forward_offset_degrees rotates it - if a model faces
        backward/sideways at 0, try 180 first.

        idle_animation/walk_animation/run_animation: each optional -
        every state that isn't given falls back to whichever OTHER
        state's clip IS given (see _resolve_clip). Passing only one
        (e.g. idle_animation="foo") plays that single clip for every
        movement state, matching a model with no walk/run clips at all.

        walk_speed_threshold/run_speed_threshold (m/s): default to
        Source's own velwalk/velrun (90/220 Source units/s * 0.0254) -
        the same constants character_controller.py's footstep-sound
        code already uses (_STEP_MIN_SPEED_STAND/_STEP_RUN_SPEED_STAND).
        Duplicated here as plain float literals rather than imported,
        since this class must stay usable for a remote player, which has
        no CharacterController at all - update() only ever takes a plain
        speed float from whatever the caller derived it from.

        upper_body_root_joints: optional list of joint names splitting
        this model into a lower body (idle/walk/run above - locomotion)
        and an upper body driven independently via set_upper_animation()
        (e.g. a gun-holding pose) - see Scene.add_skeletal's own
        upper_body_root_joints/Skeleton.compute_joint_mask docstrings
        for exactly how the split works (a LIST of root names, not one
        pivot joint, since some rigs - rat.glb's Source/Valve Biped rig
        included - split the arm/neck/head chain off as a sibling
        subtree rather than a strict descendant of one spine bone).
        upper_animation seeds the initial upper-body clip (None = upper
        joints start in bind pose, matching the lower-body `animation`
        param's own None-means-bind-pose convention); leave both this
        and upper_body_root_joints at their defaults (None) for a plain
        single-clip model with no body split at all - update()'s speed-
        driven idle/walk/run switching is completely unaffected either
        way, since it only ever touches the LOWER-body clip.

        This class deliberately has no aim/fire/reload state machine of
        its own for the upper body - nothing in this project (yet) knows
        what a weapon or crouch even means at the animation layer, so
        set_upper_animation() is a plain external hook: call it whenever
        a future weapon/aim system decides the upper-body pose should
        change, same as it would for anything else driving the model.

        scale/metallic/roughness/emissive/texture_path: passed straight
        through to scene.add_skeletal for full override parity with that
        method's own knobs."""
        self._scene = scene
        self._forward_offset_radians = glm.radians(forward_offset_degrees)
        self._feet_offset = float(feet_offset)
        self._idle_animation = idle_animation
        self._walk_animation = walk_animation
        self._run_animation = run_animation
        self._walk_speed_threshold = float(walk_speed_threshold)
        self._run_speed_threshold = float(run_speed_threshold)
        self._current_state = None  # "idle"/"walk"/"run", None until a clip is actually resolved

        initial_pos = glm.vec3(position) if position is not None else glm.vec3(0.0)
        initial_pos.y += self._feet_offset

        self.obj = scene.add_skeletal(
            model_path, position=initial_pos, rotation=glm.vec3(0.0),
            scale=scale, animation=self._resolve_clip("idle"),
            metallic=metallic, roughness=roughness, emissive=emissive,
            texture_path=texture_path,
            visible_in_color=visible_in_color, cast_shadow=cast_shadow,
            upper_body_root_joints=upper_body_root_joints, upper_animation=upper_animation,
        )
        # add_skeletal returns None on load failure (bad path, wrong
        # format, etc - see its own docstring) - every other method
        # below no-ops in that case rather than raising, so a missing/
        # broken model asset degrades to "no visible body" instead of
        # crashing whatever owns this PlayerModel.
        if self.obj is not None and self._resolve_clip("idle") is not None:
            self._current_state = "idle"

    def _resolve_clip(self, state):
        """Returns the best available animation clip name for `state`
        ("idle"/"walk"/"run"), preferring that state's own clip first and
        falling back through the others - so passing only one animation
        (to whichever of the three params) makes every state resolve to
        that same clip, and passing none leaves every state at None
        (rest pose)."""
        preference = {
            "idle": (self._idle_animation, self._walk_animation, self._run_animation),
            "walk": (self._walk_animation, self._run_animation, self._idle_animation),
            "run": (self._run_animation, self._walk_animation, self._idle_animation),
        }[state]
        for name in preference:
            if name is not None:
                return name
        return None

    def update(self, dt, position, yaw_degrees, speed, is_crouched=False):
        """Call once per frame (or per network update, for a remote
        player) to sync this model's transform and animation state.

        position: an already FEET-LEVEL (ground-contact) position, glm.
        vec3-able - converting from whatever the caller actually tracks
        (a CharacterController's hull-CENTER for the local player, an
        assumed eye-to-feet offset off a synced camera position for a
        remote player) is the CALLER's job, not this class's - the two
        conversions differ per caller and PlayerModel has no opinion on
        which kind of player it's attached to. Only this class's own
        fixed feet_offset (an authoring correction, not a gameplay one)
        is applied on top here.

        yaw_degrees: camera-convention yaw in degrees (camera.py's
        front.x=cos(yaw)cos(pitch), front.z=sin(yaw)cos(pitch)) - NOT
        already-converted model-space radians.

        speed: horizontal movement speed in m/s, used only to pick an
        idle/walk/run animation state - the caller computes this however
        makes sense for it (CharacterController.velocity's horizontal
        magnitude for the local player, a position-delta/elapsed-time
        estimate for a remote player driven by network updates).

        is_crouched: accepted for forward API compatibility with a
        future crouch pose/animation, but unused this iteration - no
        crouch clip concept exists yet since only one clip exists across
        this project's one skeletal asset. dt is similarly accepted (for
        symmetry with this codebase's update(dt, ...) convention) but
        unused - animation time advancement already happens inside
        Scene.update()'s own per-frame bone recompute, not here; this
        method only ever touches position/rotation/which-clip-is-active."""
        if self.obj is None:
            return

        feet_pos = glm.vec3(position)
        feet_pos.y += self._feet_offset
        self.obj["position"] = feet_pos

        # Camera yaw (degrees) -> model rotation.y (radians about +Y,
        # scene_base.py's _get_model_matrix convention). Derived, not
        # guessed: camera.py's front vector is
        # (cos(yaw), sin(pitch), sin(yaw)) i.e. camera-forward sweeps
        # +X -> +Z as yaw increases. _get_model_matrix's
        # glm.rotate(model, rotation.y, (0,1,0)) maps a model's local +Z
        # (glTF's standard rest-forward axis) to world
        # (sin(rotation.y), 0, cos(rotation.y)). Solving so the model's
        # local +Z tracks the camera's flat-forward direction
        # (cos(yaw), 0, sin(yaw)) gives rotation.y = radians(90 - yaw).
        # If a specific model still visually faces the wrong way (or is
        # mirrored front-to-back) at forward_offset_degrees=0, that
        # model's rest pose isn't the standard +Z-forward convention -
        # fix it via forward_offset_degrees (try 180 first), not by
        # changing this formula.
        self.obj["rotation"].y = glm.radians(90.0) - glm.radians(yaw_degrees) + self._forward_offset_radians

        if speed >= self._run_speed_threshold:
            new_state = "run"
        elif speed >= self._walk_speed_threshold:
            new_state = "walk"
        else:
            new_state = "idle"

        if new_state != self._current_state:
            clip = self._resolve_clip(new_state)
            # set_skeletal_animation restarts anim_time at 0 - only call
            # it when the resolved clip is actually DIFFERENT from what's
            # already playing, so the common single-clip case (every
            # state resolves to the same name) never restarts the
            # animation just because the movement state label changed.
            if clip is not None and clip != self.obj["animation"]:
                self._scene.set_skeletal_animation(self.obj, clip)
            self._current_state = new_state

    def set_upper_animation(self, animation_name):
        """Switches the upper-body clip (see __init__'s
        upper_body_root_joints) to animation_name, restarting its time
        from 0 - independent of update()'s own lower-body idle/walk/run
        switching, which never touches this. Only meaningful if this
        PlayerModel was constructed with upper_body_root_joints; a no-op
        (not an error) otherwise, or if the underlying model failed to
        load (self.obj is None)."""
        if self.obj is None:
            return
        self._scene.set_skeletal_upper_animation(self.obj, animation_name)

    def destroy(self):
        """Removes this model from the scene. No Scene.remove_skeletal
        helper exists (nothing in this codebase currently removes a
        skeletal object once added), so this is the minimal safe
        teardown - drop it from the list the render/update loops
        iterate. GPU resource release (VAOs/VBOs/textures) is a
        pre-existing gap in Scene's own API this doesn't attempt to
        fix."""
        if self.obj is not None and self.obj in self._scene.skeletal_objects:
            self._scene.skeletal_objects.remove(self.obj)
        self.obj = None
