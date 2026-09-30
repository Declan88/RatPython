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
from Modules.Weapons.weapons_base import _load_state_clip


# The rigs are authored around a reference camera at the model origin (0, 0, 0)
# looking down +Z, so the viewmodel is simply the camera's own transform turned
# around to face the way the camera looks (-Z) - no offset. _OFFSET is a
# camera-space nudge (metres; +Y up) for tuning only.
_OFFSET = glm.vec3(0.0, 0.0, 0.0)
_SCALE = 1.0


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
        # both parts: it's the arms rig plus the weapon bones. A value can be
        # (path, skin_index) instead of a bare path, for a file whose gun rig
        # isn't at the weapon's usual viewmodel_gun_skin (see that field's
        # own docstring - USP's "reload" entry needs this).
        for state, clip_spec in weapon.viewmodel_animations.items():
            if isinstance(clip_spec, tuple):
                clip_path, skin_index = clip_spec
            else:
                clip_path, skin_index = clip_spec, weapon.viewmodel_gun_skin
            # "draw" is the only state with a speed multiplier right now
            # (WeaponsBase.draw_speed) - baked into the clip's own timeline
            # at load time (see _load_state_clip's own time_scale docstring),
            # so the animation actually plays faster, not just gameplay
            # unlocking early while the full-length clip keeps going.
            time_scale = (1.0 / weapon.draw_speed) if state == "draw" else 1.0
            for part, obj in [("arms", arms)] + [("gun", g) for g in gun_parts]:
                clip = _load_state_clip(
                    self._scene, obj, clip_path, f"{weapon.player_clip(state)}_viewmodel",
                    skin_index=skin_index, time_scale=time_scale)
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

    def update(self, camera, visible):
        """Call once per frame with the camera as it will render; visible
        False (third person, dead, menu...) hides the ACTIVE weapon (any
        other primed weapon is already hidden regardless - see
        _set_visible)."""
        parts = []
        for o in [self.arms] + self.gun_parts:
            if o is not None and o not in parts:
                parts.append(o)
        for obj in parts:
            obj["viewmodel_visible"] = bool(visible)
        if not visible or not parts:
            return
        if self._one_shot_finished():
            self.play("idle")
        transform = glm.inverse(camera.get_view_matrix()) * self._local
        for obj in parts:
            obj["transform"] = transform
            obj["position"] = camera.position
