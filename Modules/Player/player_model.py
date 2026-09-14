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
                 states=(("idle", None, 0.0),), upper_states=None,
                 state_hysteresis=0.8, min_state_dwell=0.15, animation_blend_duration=None,
                 upper_body_root_joints=None, upper_animation=None,
                 jump_animation=None, crouch_animation=None, crouch_blend_duration=0.1,
                 scale=None, metallic=None, roughness=None, emissive=None, texture_path=None,
                 time_scale=1.0):
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

        states: a speed-driven blend-state table - a sequence of
        (name, clip, min_speed) triples, e.g.:
            [("idle", "rifle_idle", 0.0),
             ("walk", "rifle_walk", 90.0 * 0.0254),
             ("run",  "rifle_run",  220.0 * 0.0254)]
        Sorted internally by min_speed ascending, so passing order
        doesn't matter. update()'s `speed` picks the highest state whose
        min_speed it has reached (see _compute_movement_state for the
        hysteresis rule), then plays that state's clip on the LOWER
        body. This is deliberately an open-ended list, not three fixed
        named params - appending a new blend state (crouch-walk, sprint,
        whatever) later is just adding another (name, clip, min_speed)
        triple here, no other code in this class needs to change. A
        state's clip may be None (e.g. a placeholder state with no
        animation yet) - see _resolve_clip for the fallback this
        triggers. The default, a single "idle" state pinned at
        min_speed=0.0 with clip=None, matches a model with no
        speed-driven animation at all (every speed resolves to that one
        state, whose clip resolves to None -> bind pose).

        walk_speed_threshold/run_speed_threshold no longer exist as
        separate params - Source's own velwalk/velrun (90/220 Source
        units/s * 0.0254, the same constants character_controller.py's
        footstep-sound code uses) are just the min_speed values a caller
        puts directly into its own "walk"/"run" state triples above;
        PlayerModel itself has no opinion on how many states or what
        their thresholds are.

        state_hysteresis (0..1): falling back DOWN to a lower state
        needs speed to drop below that lower state's min_speed*
        state_hysteresis, not just below the current state's min_speed
        itself - a plain single threshold per boundary would let speed
        hovering right at it (completely normal during accelerate/
        decelerate, or just tapping a movement key) flip the state back
        and forth every tick. Each flip calls set_skeletal_animation/
        set_skeletal_upper_animation, which RESETS anim_time to 0 - with
        every state resolving to the same clip that reset is invisible,
        but once they're genuinely different clips it means the
        animation keeps snapping back to frame 0 and never gets
        anywhere, reading as "it doesn't loop" even though the
        underlying per-frame time advance in Scene.update() loops it via
        modulo just fine on its own. Rising to a HIGHER state always
        uses the plain (non-scaled) threshold, so speeding up always
        responds immediately - only slowing back down is deliberately
        "sticky".

        min_state_dwell (seconds): a state change is only accepted if at
        least this long has passed since the LAST accepted change -
        independent of, and in addition to, state_hysteresis above.
        Hysteresis alone only looks at the current tick's speed against
        a scaled-down exit threshold; it doesn't protect against a brief
        transient dip that swings past that margin for just a tick or
        two and then recovers (e.g. real per-tick horizontal-speed noise
        from Source-style ground accelerate/friction is measurably
        noisier while turning/strafing diagonally than moving in a fixed
        straight line - confirmed by simulating CharacterController's own
        _accelerate/_apply_friction with a rotating wishdir - or a
        one-frame gap in a diagonal two-key input combo). Without this,
        such a blip still flips the state (walk<->run, say), which
        RESETS anim_time to 0 via set_skeletal_animation, then flips
        right back a few frames later and resets it again - looking like
        the animation randomly restarts mid-stride even though the
        underlying speed only dipped for an instant. 0.15s default is
        short enough that a genuine, sustained speed change (actually
        slowing to a walk, actually stopping) is still barely
        noticeable, while filtering out flicker on the order of a few
        physics ticks. Applies to every transition uniformly (not just
        falling, unlike state_hysteresis) since a blip can just as
        easily be a brief spike as a brief dip.

        animation_blend_duration: seconds to crossfade over on each
        state transition (see Scene.set_skeletal_animation/
        set_skeletal_upper_animation) - None (the default) uses the
        Scene's own default (currently 0.25s). Pass 0.0 for an instant
        hard cut instead (the original behavior, before crossfading
        existed) if a snap is ever actually wanted.

        upper_body_root_joints: optional list of joint names splitting
        this model into a lower body (locomotion, driven by `states`
        above) and an upper body (e.g. a gun-holding pose) - see
        Scene.add_skeletal's own upper_body_root_joints/Skeleton.
        compute_joint_mask docstrings for exactly how the split works (a
        LIST of root names, not one pivot joint, since some rigs -
        rat.glb's Source/Valve Biped rig included - split the arm/neck/
        head chain off as a sibling subtree rather than a strict
        descendant of one spine bone). upper_animation seeds the initial
        upper-body clip when upper_states is NOT given (None = upper
        joints start in bind pose, matching the lower-body's own
        None-clip-means-bind-pose convention); ignored if upper_states IS
        given (that table decides the initial clip instead, via its own
        "idle"-equivalent lowest-min_speed entry - see upper_states
        below). Leave both this and upper_body_root_joints at their
        defaults (None) for a plain single-clip model with no body split
        at all - `states`' own speed-driven switching is completely
        unaffected either way, since it only ever touches the LOWER-body
        clip unless upper_states is also given.

        This class deliberately has no aim/fire/reload state machine of
        its own for the upper body - nothing in this project (yet) knows
        what a weapon or crouch even means at the animation layer, so
        set_upper_animation() is a plain external hook: call it whenever
        a future weapon/aim system decides the upper-body pose should
        change, same as it would for anything else driving the model.

        upper_states: the upper-body counterpart to `states` above - same
        (name, clip, min_speed) shape, but its min_speed values are
        IGNORED; only its name->clip mapping is used. update() computes
        ONE movement-state name per frame (from `states`' own
        thresholds) and looks that name up in BOTH tables, so the two
        stay in lockstep by construction - there's no way for the lower
        and upper body to disagree about which state they're in, only
        about whether that state has a clip on each side. A name present
        in `states` but missing (or with clip=None) here falls back
        through the same nearest-by-name-distance search _resolve_clip
        uses (see its docstring) restricted to upper_states' own
        entries, so giving upper_states just one entry (e.g. only a
        rifle-idle pose) plays that same pose for every movement state,
        same as passing a single animation used to. None (the default)
        keeps the upper body purely externally driven via
        set_upper_animation() instead (e.g. once a real aim/fire state
        machine exists and wants finer control than "mirrors
        locomotion") - the two modes are mutually exclusive in practice
        (auto-driving will overwrite a manual set_upper_animation() call
        the next time the movement state changes) but nothing stops
        calling set_upper_animation() for a one-off override in between.

        jump_animation: optional one-shot clip that overrides `states`/
        `upper_states` entirely for as long as update()'s is_grounded
        argument reads False - played once with loop=False (see Scene.
        add_skeletal/set_skeletal_animation's own loop param) so it
        freezes on its last frame rather than looping the takeoff motion
        while still airborne, on BOTH the lower body and, if this
        PlayerModel auto-drives its upper body (upper_states given), the
        upper body too - a jump pose is inherently full-body, unlike the
        upper/lower locomotion split. The instant is_grounded reads True
        again, normal `states`-driven switching resumes immediately
        (re-picked fresh from the current speed, not whatever state was
        cached before the jump - see update()'s own comment). None (the
        default) disables this entirely - is_grounded is then accepted
        but ignored, matching every caller before this param existed
        (nothing currently passes is_grounded=False without also passing
        jump_animation).

        crouch_animation: optional clip that overrides `states`/
        `upper_states` entirely for as long as update()'s is_crouched
        argument reads True - a plain held pose (looping, unlike
        jump_animation - RifleCrouch.glb is a static 2-keyframe pose
        with identical start/end values, so looping it is a no-op
        anyway, but a future genuinely-animated crouch clip would still
        loop correctly here), applied to BOTH the lower body and, if
        this PlayerModel auto-drives its upper body, the upper body too
        - same full-body reasoning as jump_animation, since there's only
        one crouch pose covering the whole rig, not a separate upper/
        lower split. Checked AFTER jump_animation/is_grounded (crouching
        has no effect while airborne - is_grounded=False's own override
        wins) but BEFORE `states`' normal speed/sprint-driven selection,
        so landing while still holding crouch goes straight to the
        crouch pose rather than flashing a locomotion clip first. The
        instant is_crouched reads False again, normal `states`-driven
        switching resumes immediately, same resync mechanism as
        jump_animation's own landing case. None (the default) disables
        this entirely - is_crouched is then accepted but ignored,
        matching every caller before this param existed.

        crouch_blend_duration: crossfade seconds used specifically for
        crouch_animation transitions (both entering and leaving crouch),
        SEPARATE from animation_blend_duration above - defaults to 0.5s,
        noticeably longer than the 0.25s Scene default every other
        transition uses. Standing to crouched (and back) is a much
        bigger, slower full-body pose change than idle/walk/run ever are
        with each other, so the same short blend that works for those
        looks like a snap here - a longer, more visible ease between the
        two poses reads as an actual crouch motion instead. Only
        meaningful alongside crouch_animation; ignored otherwise.

        scale/metallic/roughness/emissive/texture_path/time_scale:
        passed straight through to scene.add_skeletal for full override
        parity with that method's own knobs - time_scale specifically is
        an opt-in per-keyframe-time correction for a model file known to
        be baked at the wrong frame rate (see skeletal_loader.
        load_skinned_glb's own docstring)."""
        self._scene = scene
        self._forward_offset_radians = glm.radians(forward_offset_degrees)
        self._feet_offset = float(feet_offset)

        if not states:
            raise ValueError("PlayerModel requires at least one state in `states`")
        self._states = sorted(states, key=lambda s: s[2])
        self._state_names = [s[0] for s in self._states]
        self._state_clips = {s[0]: s[1] for s in self._states}
        self._state_thresholds = [float(s[2]) for s in self._states]
        self._state_hysteresis = float(state_hysteresis)
        self._min_state_dwell = float(min_state_dwell)
        # Already "ready" so the very first real transition (e.g. idle
        # -> walk the instant the player starts moving) isn't delayed by
        # this - only SUBSEQUENT changes within min_state_dwell of the
        # last one get held back.
        self._state_dwell_elapsed = self._min_state_dwell
        self._animation_blend_kwargs = (
            {} if animation_blend_duration is None else {"blend_duration": float(animation_blend_duration)}
        )
        self._current_state = None  # name from _state_names, None until a clip is actually resolved

        self._upper_state_clips = {name: clip for name, clip, _min_speed in upper_states} if upper_states else None
        self._upper_auto_driven = self._upper_state_clips is not None
        self._current_upper_state = None

        self._jump_animation = jump_animation
        self._airborne = False  # tracks the last is_grounded seen by update(), for edge detection

        self._crouch_animation = crouch_animation
        self._crouching = False  # tracks the last is_crouched seen by update(), for edge detection
        self._crouch_blend_kwargs = {"blend_duration": float(crouch_blend_duration)}
        # Set only by the "stood back up" edge above, consumed (and
        # cleared) by the very next lower-body clip switch in update() -
        # see that edge's own comment for why.
        self._pending_blend_kwargs = None

        initial_pos = glm.vec3(position) if position is not None else glm.vec3(0.0)
        initial_pos.y += self._feet_offset

        initial_state = self._state_names[0]
        initial_lower_clip = self._resolve_clip(initial_state)
        initial_upper_animation = (
            self._resolve_upper_clip(initial_state) if self._upper_auto_driven else upper_animation
        )

        self.obj = scene.add_skeletal(
            model_path, position=initial_pos, rotation=glm.vec3(0.0),
            scale=scale, animation=initial_lower_clip,
            metallic=metallic, roughness=roughness, emissive=emissive,
            texture_path=texture_path,
            visible_in_color=visible_in_color, cast_shadow=cast_shadow,
            upper_body_root_joints=upper_body_root_joints, upper_animation=initial_upper_animation,
            time_scale=time_scale,
        )
        # add_skeletal returns None on load failure (bad path, wrong
        # format, etc - see its own docstring) - every other method
        # below no-ops in that case rather than raising, so a missing/
        # broken model asset degrades to "no visible body" instead of
        # crashing whatever owns this PlayerModel.
        if self.obj is not None:
            self._current_state = initial_state
            if self._upper_auto_driven:
                self._current_upper_state = initial_state

    def _compute_movement_state(self, speed, dt, is_sprinting=None):
        """Speed (and, if given, is_sprinting) -> a name from
        self._state_names, with a minimum dwell time (see __init__'s
        min_state_dwell docstring) so a brief transient blip can't flip
        the state for only a tick or two - resetting the animation via
        set_skeletal_animation each time, which is what actually made
        this read as "the animation randomly restarts" rather than just
        an instant pop. dt accumulates into self._state_dwell_elapsed on
        every call (whether or not a change ends up happening this
        tick); a change is only computed/returned once that reaches
        min_state_dwell, otherwise this returns self._current_state
        unchanged outright, without even looking at speed/is_sprinting.

        Two selection modes, once dwell allows a change at all:

        is_sprinting is None (the default - used by callers with no
        notion of a discrete sprint input, e.g. a remote player only
        ever given a position stream to estimate speed from): the
        original purely speed-driven hysteresis chain, generalized to
        any number of states - two passes in state-index order
        (ascending min_speed), starting from whichever state is
        currently active: (1) falling - step DOWN one state at a time
        while speed is below the state being stepped INTO's
        min_speed*state_hysteresis (sticky, on purpose - see __init__);
        (2) rising - step UP one state at a time while speed has reached
        the NEXT state's plain min_speed (immediate, on purpose).

        is_sprinting is True/False: speed only decides MOVING vs IDLE
        (states[0] is idle; states[1]'s min_speed, scaled by
        state_hysteresis while already moving, is the moving/idle
        boundary - same hysteresis idea, just for one boundary instead
        of a whole chain). Once moving, is_sprinting alone - not speed -
        picks between states[1] (the base moving state, e.g. "walk") and
        states[-1] (the fastest state, e.g. "run"): a discrete key-held
        input has no momentum/physics-noise to flicker on the way a
        continuously-measured speed does, which is exactly the point
        (see PlayerModel.update()'s own is_sprinting param docstring).
        Only meaningful with at least 2 states; with exactly 2 (no
        distinct "run" beyond "walk"), states[-1] and states[1] are the
        same entry and is_sprinting has no effect."""
        self._state_dwell_elapsed += dt
        if self._state_dwell_elapsed < self._min_state_dwell:
            return self._current_state

        try:
            idx = self._state_names.index(self._current_state)
        except ValueError:
            idx = 0
        start_idx = idx

        if is_sprinting is None or len(self._state_names) < 2:
            while idx > 0 and speed < self._state_thresholds[idx] * self._state_hysteresis:
                idx -= 1
            while idx < len(self._state_names) - 1 and speed >= self._state_thresholds[idx + 1]:
                idx += 1
        else:
            moving_threshold = self._state_thresholds[1]
            is_moving = (
                speed >= moving_threshold * self._state_hysteresis if idx > 0
                else speed >= moving_threshold
            )
            idx = (len(self._state_names) - 1 if is_sprinting else 1) if is_moving else 0

        if idx != start_idx:
            self._state_dwell_elapsed = 0.0
        return self._state_names[idx]

    def _resolve_clip(self, state_name):
        """Returns the best available LOWER-body clip for state_name -
        preferring that state's own clip first, then falling back
        through the other states nearest-by-index first (ties broken
        toward the higher/faster state), so a state left with clip=None
        still plays something reasonable instead of snapping to bind
        pose. Returns None only if every state in `states` has clip=None."""
        return self._resolve_from(state_name, self._state_names, self._state_clips)

    def _resolve_upper_clip(self, state_name):
        """The upper-body equivalent of _resolve_clip, searched over
        upper_states' own name->clip table instead - see __init__'s
        upper_states docstring. state_name still comes from the LOWER
        body's `states` list (the two tables are indexed by the same
        movement-state names), so a name upper_states doesn't have at
        all is treated as a hole, same as clip=None."""
        return self._resolve_from(state_name, self._state_names, self._upper_state_clips)

    @staticmethod
    def _resolve_from(state_name, state_names, clip_map):
        try:
            idx = state_names.index(state_name)
        except ValueError:
            idx = 0
        # Nearest-by-index first, ties broken toward the higher/faster
        # state (matches the original hand-written idle/walk/run
        # preference tables exactly: idle->[idle,walk,run],
        # walk->[walk,run,idle], run->[run,walk,idle]).
        order = sorted(range(len(state_names)), key=lambda i: (abs(i - idx), -i))
        for i in order:
            clip = clip_map.get(state_names[i])
            if clip is not None:
                return clip
        return None

    def update(self, dt, position, yaw_degrees, speed, is_crouched=False, is_grounded=True,
               is_sprinting=None):
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

        speed: horizontal movement speed in m/s, used to pick a blend
        state from `states` - the caller computes this however makes
        sense for it (CharacterController.velocity's horizontal
        magnitude for the local player, a position-delta/elapsed-time
        estimate for a remote player driven by network updates). Still
        the ONLY signal used to decide moving-vs-idle even when
        is_sprinting is given below; is_sprinting only takes over
        deciding walk-vs-run once already moving.

        is_sprinting: None (the default) keeps the original purely
        speed-driven selection among ALL of `states` (matches every
        caller before this param existed, and still what a remote player
        uses - a synced position stream carries no discrete sprint-key
        bit to read). Pass True/False instead to pick walk-vs-run by
        actual input intent - whatever the caller's own "is the sprint
        key held" state is (CharacterController.is_sprinting() for the
        local player) - rather than by how fast the character happens to
        currently be moving. This matters because physical speed is
        noisy: Source-style ground accelerate/friction genuinely
        produces small tick-to-tick speed fluctuations, more so while
        turning/strafing diagonally than moving in a straight line, and
        those fluctuations can cross a speed threshold's hysteresis
        margin for just a frame or two even though the player never
        stopped sprinting - each crossing flips the animation state
        (resetting anim_time to 0) and reads as the run animation
        randomly restarting mid-stride. A discrete key-held boolean has
        no such momentum to flicker on. See _compute_movement_state's
        own docstring for exactly how is_sprinting changes selection.

        is_crouched: accepted for forward API compatibility with a
        future crouch state, but unused this iteration - a crouch state
        can be added purely via `states`/`upper_states` entries once a
        crouch clip exists; this parameter would then gate which state
        table (or an extra crouch flag folded into _compute_movement_
        state) applies.

        dt: NOT used for animation time advancement (that already
        happens inside Scene.update()'s own per-frame bone recompute,
        not here) - this method only ever touches position/rotation/
        which-clip-is-active. It IS used, though, to accumulate
        _compute_movement_state's min_state_dwell timer (see __init__'s
        own docstring) - pass the real per-frame delta time, not a
        placeholder, or dwell-gating will be measured in the wrong units.

        is_grounded: only meaningful if this PlayerModel was constructed
        with jump_animation - see its own docstring. True (the default)
        matches every caller before this param existed. The caller
        decides what "grounded" means for it (CharacterController.
        is_on_ground() for the local player; nothing currently supplies
        it for a remote player, since a synced position stream has no
        direct ground-contact signal to read - jump_animation is simply
        left unset there for now)."""
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

        if self._jump_animation is not None:
            if not is_grounded and not self._airborne:
                # Just left the ground - override both bodies with the
                # jump clip, loop=False so Scene.update() clamps it at
                # its own duration and holds the last frame instead of
                # restarting the takeoff motion while still airborne.
                self._airborne = True
                if self._jump_animation != self.obj["animation"]:
                    self._scene.set_skeletal_animation(
                        self.obj, self._jump_animation, loop=False, **self._animation_blend_kwargs
                    )
                if self._upper_auto_driven and self._jump_animation != self.obj["upper_animation"]:
                    self._scene.set_skeletal_upper_animation(
                        self.obj, self._jump_animation, loop=False, **self._animation_blend_kwargs
                    )
            elif is_grounded and self._airborne:
                # Landed - invalidate the cached state names so the
                # movement-state block below is forced to resync to a
                # real locomotion clip even if the resolved state name
                # (e.g. "idle") is unchanged from before the jump; without
                # this the "new_state != self._current_state" guard would
                # skip switching away from the jump clip entirely, since
                # the STATE never actually changed, only what's playing
                # underneath it while airborne did.
                self._airborne = False
                self._current_state = None
                if self._upper_auto_driven:
                    self._current_upper_state = None
                # Also force _compute_movement_state's dwell gate open
                # immediately - otherwise, with self._current_state now
                # None, a still-accumulating dwell timer would return
                # None right back out unchanged (None != None is False),
                # silently skipping the resync this was just set up for.
                self._state_dwell_elapsed = self._min_state_dwell

        if self._airborne:
            # Held on the jump clip's last frame - skip normal
            # speed-driven switching entirely until landed (handled
            # above), so a mid-air speed change never overwrites it.
            return

        if self._crouch_animation is not None:
            if is_crouched and not self._crouching:
                # Just crouched - override both bodies with the crouch
                # pose, same full-body reasoning as jump_animation above.
                self._crouching = True
                if self._crouch_animation != self.obj["animation"]:
                    self._scene.set_skeletal_animation(
                        self.obj, self._crouch_animation, **self._crouch_blend_kwargs
                    )
                if self._upper_auto_driven and self._crouch_animation != self.obj["upper_animation"]:
                    self._scene.set_skeletal_upper_animation(
                        self.obj, self._crouch_animation, **self._crouch_blend_kwargs
                    )
            elif not is_crouched and self._crouching:
                # Stood back up - same cached-state invalidation trick as
                # jump_animation's own landing case, and for the same
                # reason (the movement-state block below must be forced
                # to resync even if the resolved state name is unchanged
                # from before crouching). The resync itself happens in
                # the generic clip-switch block below, using
                # _animation_blend_kwargs by default - _pending_blend_
                # kwargs overrides that ONE switch to use the same longer
                # crouch_blend_duration this whole pose change deserves
                # symmetrically (standing up is just as big a pose change
                # as crouching down was), without changing the default
                # for every OTHER, unrelated state switch that might
                # happen to land on this same tick.
                self._crouching = False
                self._current_state = None
                if self._upper_auto_driven:
                    self._current_upper_state = None
                self._state_dwell_elapsed = self._min_state_dwell
                self._pending_blend_kwargs = self._crouch_blend_kwargs

        if self._crouching:
            # Held on the crouch pose - skip normal speed/sprint-driven
            # switching entirely until standing back up (handled above).
            return

        new_state = self._compute_movement_state(speed, dt, is_sprinting=is_sprinting)

        # _pending_blend_kwargs (see the "stood back up" edge above) is a
        # one-shot override for whichever locomotion switch(es) happen on
        # THIS call - consumed and cleared immediately so it can only
        # ever affect this one resync, shared identically by both the
        # lower and upper body since they're resyncing to the same
        # new_state for the same reason (a genuinely animated future
        # crouch clip could someday split into a lower_crouch_blend_
        # kwargs/upper_crouch_blend_kwargs pair if lower/upper ever
        # needed different durations here, but nothing currently does).
        blend_kwargs = (
            self._pending_blend_kwargs if self._pending_blend_kwargs is not None
            else self._animation_blend_kwargs
        )
        self._pending_blend_kwargs = None

        if new_state != self._current_state:
            clip = self._resolve_clip(new_state)
            # set_skeletal_animation restarts anim_time at 0 - only call
            # it when the resolved clip is actually DIFFERENT from what's
            # already playing, so the common single-clip case (every
            # state resolves to the same name) never restarts the
            # animation just because the state label changed.
            if clip is not None and clip != self.obj["animation"]:
                self._scene.set_skeletal_animation(self.obj, clip, **blend_kwargs)
            self._current_state = new_state

        if self._upper_auto_driven and new_state != self._current_upper_state:
            # Mirrors the lower body's own movement state exactly (same
            # new_state, same transition-only guard) - see __init__'s
            # upper_states docstring for why this is opt-in rather than
            # always tracking the lower body.
            upper_clip = self._resolve_upper_clip(new_state)
            if upper_clip is not None and upper_clip != self.obj["upper_animation"]:
                self._scene.set_skeletal_upper_animation(self.obj, upper_clip, **blend_kwargs)
            self._current_upper_state = new_state

    def set_upper_animation(self, animation_name):
        """Switches the upper-body clip (see __init__'s
        upper_body_root_joints) to animation_name, restarting its time
        from 0 - independent of update()'s own blend-state switching,
        which never touches this. Only meaningful if this PlayerModel
        was constructed with upper_body_root_joints; a no-op (not an
        error) otherwise, or if the underlying model failed to load
        (self.obj is None)."""
        if self.obj is None:
            return
        self._scene.set_skeletal_upper_animation(self.obj, animation_name, **self._animation_blend_kwargs)

    def set_visible_in_color(self, visible):
        """Switches whether this model actually draws in the normal
        color pass (see add_skeletal's own visible_in_color) - e.g.
        toggling third-person mode on/off, where the local player's body
        should suddenly be visible instead of shadow-only, or vice versa
        going back to first person. cast_shadow is untouched either way
        (this model still casts a shadow in both modes) - a no-op if the
        underlying model failed to load (self.obj is None)."""
        if self.obj is None:
            return
        self.obj["visible_in_color"] = bool(visible)

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
