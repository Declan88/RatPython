"""
The local player's first-person arms and gun: skinned models glued to the
camera.

Efficiency: the parts live in Scene.viewmodel_objects (see Scene.add_
viewmodel), so shadow/SSR/culling/physics passes never touch them, and their
whole per-frame cost is one matrix write per part here plus one draw call each
at the end of the frame (Scene._render_viewmodels). When hidden (third person)
that pass returns before touching any GL state, and hidden parts skip their
animation update too.

A weapon (Modules/Weapons) names the glb holding its arms and gun (one file,
one rig each - see WeaponsBase.viewmodel_*) and the idle/etc. animations;
set_weapon() loads them the FIRST time a given weapon is shown, and just
switches visibility after that - see set_weapon's own docstring for why
(app.py's own weapon switching relies on this to be near-instant instead of
re-loading a weapon's whole viewmodel every time the player switches back to
it).
"""

import glm

from Modules.Graphics import pose_batch
from Modules.Player.rat_colors import RAT_TINT_MASK_PATH
from Modules.Weapons.weapons_base import _load_state_clip, _parse_clip_spec


# The rigs are authored around a reference camera at the model origin (0, 0, 0)
# looking down +Z, so the viewmodel is simply the camera's own transform turned
# around to face the way the camera looks (-Z) - no offset. _OFFSET is a
# camera-space nudge (metres; +Y up) for tuning only.
_OFFSET = glm.vec3(0.0, 0.0, 0.0)
_SCALE = 1.0

# Mouse-look sway: the viewmodel lags a little behind a fast camera turn, like the gun's own
# inertia, then settles back to dead-center - the classic FPS "weapon sway" feel. Driven by
# camera.yaw/pitch's own frame-to-frame DELTA (already tracked for other reasons - see
# NetworkManager.set_local_state's own yaw/pitch args - not a second mouse-delta source of
# truth), not raw mouse input, so it stays correct through anything else that can move yaw/
# pitch (recoil, a cutscene-style camera snap) exactly the same as an actual mouse turn would.
# An initial 0.05/8.0 pass was reported "too fast" - settling in a few frames read as a snap,
# not a drift - so DECAY stays slow (3.0, unchanged since that fix). GAIN (how MUCH lag a
# given turn produces, independent of how fast it then settles) was then raised back up from
# that same pass's shared 0.012 to a 0.03 shared value after a request for more intensity,
# and now split into separate yaw/pitch gains (and clamps) at a further request for pitch
# (looking up/down) to read more intense than yaw (side to side) - pitch at 1.8x yaw's own
# values, both still well under the original "too fast" pass's numbers.
_SWAY_GAIN_YAW = 0.03     # degrees of lag added per degree of YAW turned this frame
_SWAY_GAIN_PITCH = 0.054  # same, for PITCH - 1.8x yaw's own gain, the actual "more intense" ask
_SWAY_DECAY = 3.0         # how fast lag settles back toward zero, per second (shared - this
                          # governs SPEED, not amount, no reason for the two axes to differ)
_SWAY_MAX_YAW_DEG = 4.0   # clamp - keeps a fast flick-turn from swinging the gun too far
_SWAY_MAX_PITCH_DEG = 7.0  # same, for pitch - raised to match its own bigger gain above
# Below this (degrees), skip building/multiplying the sway rotation matrices at all this
# frame - the overwhelmingly common case (mouse not moving, or lag already fully settled)
# costs nothing beyond the two-float decay update below, not two glm.rotate calls + two 4x4
# multiplies on top of the transform this function was already building regardless.
_SWAY_EPSILON_DEG = 1e-3
# The rotation's own pivot, in camera space (same convention as _OFFSET - +Y up, -Z forward).
# Literally rotating around camera-space (0,0,0) - the eye itself, since the rig is authored
# with the camera AT the model origin (see _OFFSET's own comment) - swings the WHOLE gun
# through a wide arc from a point well behind/above where it actually sits, which read as
# pivoting from the wrong place entirely. Real weapon sway pivots from roughly where the gun
# itself is held, not the eye socket - this nudges the rotation's own pivot forward and
# slightly down to approximate that, via a translate/rotate/translate-back sandwich (see
# update()) instead of rotating around the raw origin.
_SWAY_PIVOT = glm.vec3(0.0, -0.08, -0.35)


def _unique_parts(parts):
    """A weapon's arms and gun objects without repeats (arms and gun can be the same object)."""
    seen = []
    for obj in [parts["arms"]] + parts["gun_parts"]:
        if obj is not None and not any(obj is other for other in seen):
            seen.append(obj)
    return seen


class ViewModel:
    def __init__(self, scene, tint_mask_path=RAT_TINT_MASK_PATH):
        self._scene = scene
        self._tint_mask_path = tint_mask_path
        self._local = (
            glm.translate(glm.mat4(1.0), _OFFSET)
            * glm.rotate(glm.mat4(1.0), glm.pi(), glm.vec3(0.0, 1.0, 0.0))
            * glm.scale(glm.mat4(1.0), glm.vec3(_SCALE))
        )
        self._tint = None
        self._weapon = None    # the ACTIVE (visible) weapon, or None
        # weapon -> {"arms": obj, "gun": obj, "clips": {(part, state): clip}}
        # - every weapon that's been shown at least once, PRIMED (loaded,
        # animations merged) whether or not it's the active one right now.
        self._primed = {}
        self._one_shot = None     # part whose one-shot animation is playing
        self._one_shot_state = None  # which STATE that is (see is_one_shot_active)
        self.arms = None          # always the ACTIVE weapon's parts - None if none is active
        self.gun = None           # the gun's first (main) part - see gun_parts
        self.gun_parts = []       # every gun object (one per material for a multi-material gun)
        # Mouse-look sway state (see _SWAY_GAIN's own comment) - _last_yaw/_pitch is None
        # until the first real update() call, so the very first frame never sees a huge
        # bogus "delta" from whatever camera.yaw/pitch happened to already be.
        self._sway_yaw = 0.0
        self._sway_pitch = 0.0
        self._last_yaw = None
        self._last_pitch = None

    def set_weapon(self, weapon):
        """Shows `weapon`, priming it first (loading its arms/gun and
        merging its animations) ONLY if this is the first time it's been
        shown on this ViewModel - a weapon already primed just gets shown/
        hidden (see _set_visible), no re-loading. The PREVIOUS active
        weapon (if any) is hidden, not unloaded, so switching back to it
        later is equally instant. Call clear_weapon() instead for an actual
        full teardown (e.g. the viewmodel itself going away)."""
        if self._weapon is weapon:
            return
        if self._weapon is not None:
            self._set_visible(self._weapon, False)
        if weapon not in self._primed:
            self._prime(weapon)
        self._weapon = weapon
        parts = self._primed[weapon]
        self.arms, self.gun, self.gun_parts = parts["arms"], parts["gun"], parts["gun_parts"]
        self._set_visible(weapon, True)
        # Just "idle" here - WeaponsBase.set_active (called right after this
        # by equip_player, same weapon switch) is what actually triggers
        # "draw", with its own fallback for a weapon with no draw clip. Not
        # duplicated here too, or a weapon WITH one would restart it twice.
        self.play("idle")

    def prime_weapon(self, weapon):
        """Loads weapon's arms/gun/animations if not already primed, WITHOUT
        showing it or touching whichever weapon is currently active - the
        pre-load half of set_weapon, for a weapon the player isn't holding
        yet (see WeaponsBase.equip_viewmodel's own activate=False and
        app.py's setup_game)."""
        if weapon not in self._primed:
            self._prime(weapon)

    def _prime(self, weapon):
        """One-time load of `weapon`'s arms/gun models and animation clips -
        see set_weapon's own docstring for why this only ever runs once per
        weapon per ViewModel."""
        path = weapon.viewmodel_model
        gun_path = weapon.viewmodel_gun_model or path
        arms = self._scene.add_viewmodel(
            path, tint_mask_path=self._tint_mask_path, skin_index=weapon.viewmodel_arms_skin)
        if gun_path == path and weapon.viewmodel_gun_skin == weapon.viewmodel_arms_skin:
            # Same file, same skin - arms and gun are already one combined
            # mesh/rig, unlike pistol.glb's separate arms/gun skins.
            # Loading it a second time would render the whole model twice
            # on top of itself. Share the one object instead; every loop
            # below already treats "arms" and "gun" as independent slots
            # that may just happen to be the same object (clip names are
            # per-STATE, not per-part, so the second _load_state_clip call
            # below no-ops - see its own "already merged" early-return).
            gun = arms
            gun_parts = [gun]
        elif weapon.viewmodel_gun_nodes:
            # One skinned object keeps only ONE material, so a gun with
            # several (body/cheese/glass) loads as one object per mesh node.
            gun_parts = []
            for node in weapon.viewmodel_gun_nodes:
                node_modes = weapon.viewmodel_node_alpha_mode_overrides
                ignore = node in node_modes     # this node opts out of the by-material overrides
                obj = self._scene.add_viewmodel(
                    gun_path, skin_index=weapon.viewmodel_gun_skin, node_names=(node,),
                    alpha_mode_overrides=None if ignore else weapon.viewmodel_alpha_mode_overrides,
                    roughness_overrides=weapon.viewmodel_roughness_overrides)
                if obj is not None:
                    if node_modes.get(node):
                        obj["alpha_mode"] = node_modes[node]
                    gun_parts.append(obj)
            gun = gun_parts[0] if gun_parts else None
        else:
            gun = self._scene.add_viewmodel(
                gun_path, skin_index=weapon.viewmodel_gun_skin,
                alpha_mode_overrides=weapon.viewmodel_alpha_mode_overrides,
                roughness_overrides=weapon.viewmodel_roughness_overrides)
            gun_parts = [gun]
        if arms is not None:
            self._scene.set_skeletal_tint(arms, self._tint)
        clips = {}
        # The gun rig's clips (see WeaponsBase.viewmodel_animations) play on
        # both parts: it's the arms rig plus the weapon bones. See _parse_
        # clip_spec's own docstring for the full set of forms a value can
        # take - a bare path, (path, skin_index) for a file whose gun rig
        # isn't at the weapon's usual viewmodel_gun_skin (USP's "reload"
        # entry needs this), or (path, source_clip_name) for a file exported
        # with several named actions sharing one skin instead of one clip
        # per file (pencil.py's Pencil.glb needs this).
        for state, clip_spec in weapon.viewmodel_animations.items():
            clip_path, skin_index, source_clip = _parse_clip_spec(clip_spec, weapon.viewmodel_gun_skin)
            # "draw" is the only state with a speed multiplier right now
            # (WeaponsBase.draw_speed) - baked into the clip's own timeline
            # at load time (see _load_state_clip's own time_scale docstring),
            # so the animation actually plays faster, not just gameplay
            # unlocking early while the full-length clip keeps going.
            time_scale = (1.0 / weapon.draw_speed) if state == "draw" else 1.0
            for part, obj in [("arms", arms)] + [("gun", g) for g in gun_parts]:
                clip = _load_state_clip(
                    self._scene, obj, clip_path, f"{weapon.player_clip(state)}_viewmodel",
                    skin_index=skin_index, time_scale=time_scale, source_clip=source_clip)
                if clip is not None:
                    clips[(part, state)] = clip
        self._primed[weapon] = {"arms": arms, "gun": gun, "gun_parts": gun_parts, "clips": clips}
        for obj in [arms] + gun_parts:
            if obj is not None:
                obj["viewmodel_visible"] = False

    def warm_animations(self):
        """Bakes every primed weapon's clips for the fast pose path (see
        pose_batch.baked_clip) now, so the first time each one plays - a draw,
        a reload, the other slot's idle - it doesn't stall on the bake. Returns
        how many clips were baked."""
        baked = 0
        for parts in self._primed.values():
            names = {name for name in parts["clips"].values() if name}
            for obj in _unique_parts(parts):
                for name in names:
                    if pose_batch.baked_clip(obj["skeleton"], name) is not None:
                        baked += 1
        return baked

    def _set_visible(self, weapon, visible):
        parts = self._primed.get(weapon)
        if parts is None:
            return
        for obj in [parts["arms"]] + parts["gun_parts"]:
            if obj is not None:
                obj["viewmodel_visible"] = visible

    def clear_weapon(self):
        """Full teardown of EVERY primed weapon, active or not - only for
        when the viewmodel itself goes away entirely (see WeaponsBase.
        unequip). An ordinary weapon switch never calls this - see
        set_weapon's own docstring."""
        for parts in self._primed.values():
            for obj in _unique_parts(parts):
                if obj is not None:
                    self._scene.remove_viewmodel(obj)
        self._primed.clear()
        self.arms = self.gun = None
        self.gun_parts = []
        self._one_shot = None
        self._one_shot_state = None
        self._weapon = None

    def play(self, state):
        """Switches the ACTIVE weapon's arms and gun to `state`'s animation
        (a part with no clip for that state is left as it is). A one-shot
        state (see WeaponsBase.viewmodel_one_shot_states) plays once, from
        the start, and update() drops back to idle when it ends. Playing it
        again while it's still going restarts it from the beginning."""
        if self._weapon is None:
            return
        clips = self._primed[self._weapon]["clips"]
        one_shot = state in self._weapon.viewmodel_one_shot_states
        self._one_shot = None
        self._one_shot_state = None
        for part, obj in [("arms", self.arms)] + [("gun", g) for g in self.gun_parts]:
            clip = clips.get((part, state))
            if obj is None or clip is None:
                continue
            if one_shot:
                self._scene.set_skeletal_animation(obj, clip, blend_duration=0.0, loop=False)
                # set_skeletal_animation ignores a clip that's already
                # playing, but a shot fired mid-recoil must restart it.
                obj["anim_time"] = 0.0
                self._one_shot = obj
                self._one_shot_state = state
            else:
                self._scene.set_skeletal_animation(obj, clip)

    def is_one_shot_active(self, state):
        """True while STATE's one-shot animation is the current one AND
        still actually playing (see _one_shot_finished) - False once it's
        finished, if some other state has been played since, or if STATE
        never had a clip to play in the first place. WeaponsBase.drawing/
        reloading are bound to this instead of a guessed duration constant -
        see their own docstrings for why (a fixed-timer version of this kept
        drifting out of sync with the real clip's actual length)."""
        return self._one_shot_state == state and not self._one_shot_finished()

    @property
    def one_shot_state(self):
        """The one-shot state currently playing, or None."""
        return self._one_shot_state

    def one_shot_time(self, state):
        """Seconds into STATE's one-shot animation (in the clip's own timeline,
        i.e. after any time_scale baked in at load), or None if STATE isn't the
        one-shot currently playing."""
        if self._one_shot_state != state or self._one_shot is None:
            return None
        return self._one_shot["anim_time"]

    def _one_shot_finished(self):
        obj = self._one_shot
        if obj is None:
            return False
        clip = obj["skeleton"].animations.get(obj["animation"])
        return clip is not None and obj["anim_time"] >= clip.effective_duration

    def set_tint(self, rgb):
        """Fur color on the arms, same as PlayerModel.set_tint - 0-1 floats
        or None. Applied to EVERY primed weapon's arms, active or not, so
        switching to one primed earlier doesn't show it in the wrong
        color."""
        self._tint = None if rgb is None else tuple(rgb)
        for parts in self._primed.values():
            if parts["arms"] is not None:
                self._scene.set_skeletal_tint(parts["arms"], self._tint)

    def _update_sway(self, camera, dt):
        """Advances the lag-then-settle sway state by one frame (see _SWAY_GAIN_YAW's own
        comment) - a handful of scalar float ops regardless of how many viewmodel parts
        exist, since the resulting angles get applied to one shared transform below, not
        recomputed per part."""
        if self._last_yaw is None:
            self._last_yaw, self._last_pitch = camera.yaw, camera.pitch
            return
        dyaw = camera.yaw - self._last_yaw
        dpitch = camera.pitch - self._last_pitch
        self._last_yaw, self._last_pitch = camera.yaw, camera.pitch
        # +dyaw, not -dyaw (pitch below is still -dpitch) - reported swinging the wrong way
        # (right turn swayed the gun right instead of lagging left behind it) at the original
        # sign; flipped once confirmed, rather than re-deriving the whole pivot/rotation
        # chain for what was really just this one axis's convention being backwards.
        self._sway_yaw = max(-_SWAY_MAX_YAW_DEG, min(_SWAY_MAX_YAW_DEG, self._sway_yaw + dyaw * _SWAY_GAIN_YAW))
        self._sway_pitch = max(-_SWAY_MAX_PITCH_DEG, min(_SWAY_MAX_PITCH_DEG, self._sway_pitch - dpitch * _SWAY_GAIN_PITCH))
        # A plain per-frame multiplicative decay (not a real exp()) - cheap, and at any
        # sane frame rate indistinguishable from one: settles from full clamp to
        # imperceptible in well under half a second at the default _SWAY_DECAY.
        decay = max(0.0, 1.0 - _SWAY_DECAY * dt)
        self._sway_yaw *= decay
        self._sway_pitch *= decay

    def update(self, camera, visible, dt, scope_blend=0.0):
        """Call once per frame with the camera as it will render; visible
        False (third person, dead, menu...) hides the ACTIVE weapon (any
        other primed weapon is already hidden regardless - see
        _set_visible). dt drives the mouse-look sway's own settle speed
        (_SWAY_DECAY) - see _update_sway.

        scope_blend: the active weapon's own eased 0..1 ADS fraction (see WeaponsBase.
        update_scope - app.py calls that BEFORE this, same frame, and passes the result
        straight through). Shifts the whole viewmodel's own LOCAL (camera-space)
        transform so its scope_point() moves toward the camera's view origin - NOT a
        camera.position move: the viewmodel is rendered entirely in camera space
        already (transform = inverse(camera.get_view_matrix()) * local, see this
        method's own tail end), so moving camera.position has literally no visible
        effect on it at all - it would just be screen-locked in exactly the same spot
        regardless of where the "world" camera sits. Moving the GUN's own local
        transform toward the camera's fixed view axis instead is what actually reads as
        the view pushing into the gun/scope."""
        parts = []
        for o in [self.arms] + self.gun_parts:
            if o is not None and o not in parts:
                parts.append(o)
        for obj in parts:
            obj["viewmodel_visible"] = bool(visible)
        if not visible or not parts:
            # Still track yaw/pitch while hidden, so re-showing the viewmodel later doesn't
            # see one huge accumulated "delta" from however far the camera moved in the
            # meantime and yank the sway to its clamp on the very next visible frame.
            self._last_yaw, self._last_pitch = camera.yaw, camera.pitch
            return
        if self._one_shot_finished():
            self.play("idle")
        self._update_sway(camera, dt)
        local = self._local
        if abs(self._sway_yaw) > _SWAY_EPSILON_DEG or abs(self._sway_pitch) > _SWAY_EPSILON_DEG:
            # translate(+pivot) * rotate * translate(-pivot): the standard "rotate around an
            # arbitrary point" sandwich - shifts _SWAY_PIVOT to the origin, applies the sway
            # rotation THERE instead of at camera-space (0,0,0), then shifts back. Without
            # this, rotating around the raw origin swings the whole gun through a wide arc
            # from the eye - see _SWAY_PIVOT's own comment.
            local = (
                glm.translate(glm.mat4(1.0), _SWAY_PIVOT)
                * glm.rotate(glm.mat4(1.0), glm.radians(self._sway_pitch), glm.vec3(1.0, 0.0, 0.0))
                * glm.rotate(glm.mat4(1.0), glm.radians(self._sway_yaw), glm.vec3(0.0, 1.0, 0.0))
                * glm.translate(glm.mat4(1.0), -_SWAY_PIVOT)
                * local
            )
        view = camera.get_view_matrix()
        weapon = self._weapon
        if (scope_blend > 0.0 and weapon is not None and weapon.has_scope
                and weapon.viewmodel_scope_bone and self.gun is not None):
            # Read the scope bone's CURRENT camera-space position - set obj["transform"]
            # to the pre-ADS `local` FIRST (joint_world_position reads it straight off
            # obj["transform"] via Scene._get_model_matrix), get its WORLD position back,
            # then re-project through `view` to land back in camera space: world = inverse
            # (view) * local * bone_local, so view * world = local * bone_local exactly -
            # the bone's position in `local`'s own space, with the full skin/bone
            # hierarchy already accounted for (view/inverse(view) cancel out completely;
            # going via world space is just the easiest way to reuse scene.joint_world_
            # position's existing skinning math rather than re-deriving it here).
            self.gun["transform"] = glm.inverse(view) * local
            world_point = self._scene.joint_world_position(
                self.gun, weapon.viewmodel_scope_bone, local_offset=weapon.viewmodel_scope_offset)
            if world_point is not None:
                cam_space_point = glm.vec3(view * glm.vec4(world_point, 1.0))
                # Shifts the WHOLE viewmodel by the negative of that point, scaled by how
                # far into the scope we are - at scope_blend=1.0 the bone sits exactly at
                # camera-space (0,0,0), i.e. the eye itself. By the time that's reached,
                # `scoped` is True and hidden_while_scoped below has already taken the
                # model off screen, so this extreme convergence is never actually seen
                # clipping through the near plane - the scope overlay has fully taken
                # over by then.
                local = glm.translate(glm.mat4(1.0), -cam_space_point * scope_blend) * local
        transform = glm.inverse(view) * local
        # Once fully eased into the scope (see WeaponsBase.scoped) the gun is doing
        # nothing useful on screen anyway (it's converged to sitting right at the eye -
        # see the block above) and the scope overlay (app.py's own reticle widget) is
        # meant to read as actually looking THROUGH the sight, not still seeing the gun
        # in the way - see _render_viewmodels' own hidden_while_scoped filter for why
        # this is a separate flag from viewmodel_visible rather than just hiding it the
        # normal way (animation/muzzle tracking need to keep running live underneath).
        hidden_while_scoped = bool(weapon is not None and weapon.has_scope and weapon.scoped)
        for obj in parts:
            obj["transform"] = transform
            obj["position"] = camera.position
            obj["hidden_while_scoped"] = hidden_while_scoped
