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

import os
import time

import glm

from Modules.Player.rat_colors import RAT_TINT_MASK_PATH
from Modules.Weapons.weapons_base import _load_state_clip


def _debug_active():  # TEMP DEBUG (RATWAR_AUTOTEST)
    until = os.environ.get("RATWAR_DEBUG_UNTIL")
    return until is not None and time.perf_counter() < float(until)


# The rigs are authored around a reference camera at the model origin (0, 0, 0)
# looking down +Z, so the viewmodel is simply the camera's own transform turned
# around to face the way the camera looks (-Z) - no offset. _OFFSET is a
# camera-space nudge (metres; +Y up) for tuning only.
_OFFSET = glm.vec3(0.0, 0.0, 0.0)
_SCALE = 1.0


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
        self.gun = None

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
        self.arms, self.gun = parts["arms"], parts["gun"]
        if _debug_active():
            print(f"[VM DEBUG] set_weapon({weapon.name}) arms={id(self.arms)} gun={id(self.gun)} same_obj={self.arms is self.gun}")
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
        else:
            gun = self._scene.add_viewmodel(gun_path, skin_index=weapon.viewmodel_gun_skin)
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
            for part, obj in (("arms", arms), ("gun", gun)):
                clip = _load_state_clip(
                    self._scene, obj, clip_path, f"{weapon.player_clip(state)}_viewmodel",
                    skin_index=skin_index, time_scale=time_scale)
                if clip is not None:
                    clips[(part, state)] = clip
        self._primed[weapon] = {"arms": arms, "gun": gun, "clips": clips}
        for obj in (arms, gun):
            if obj is not None:
                obj["viewmodel_visible"] = False

    def _set_visible(self, weapon, visible):
        parts = self._primed.get(weapon)
        if parts is None:
            return
        for obj in (parts["arms"], parts["gun"]):
            if obj is not None:
                obj["viewmodel_visible"] = visible

    def clear_weapon(self):
        """Full teardown of EVERY primed weapon, active or not - only for
        when the viewmodel itself goes away entirely (see WeaponsBase.
        unequip). An ordinary weapon switch never calls this - see
        set_weapon's own docstring."""
        for parts in self._primed.values():
            for obj in (parts["arms"], parts["gun"]):
                if obj is not None:
                    self._scene.remove_viewmodel(obj)
        self._primed.clear()
        self.arms = self.gun = None
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
        if _debug_active():
            print(f"[VM DEBUG] play({state!r}) weapon={self._weapon.name} one_shot={one_shot}")
        for part, obj in (("arms", self.arms), ("gun", self.gun)):
            clip = clips.get((part, state))
            if obj is None or clip is None:
                if _debug_active():
                    print(f"  [VM DEBUG] part={part} SKIPPED (obj is None: {obj is None}, clip is None: {clip is None})")
                continue
            if _debug_active():
                print(f"  [VM DEBUG] part={part} obj={id(obj)} clip={clip!r} prev_animation={obj['animation']!r} prev_anim_time={obj['anim_time']:.4f} anim_blend_duration={obj['anim_blend_duration']:.4f} anim_blend_elapsed={obj['anim_blend_elapsed']:.4f} pose_snap_is_none={obj.get('pose_snap') is None}")
            if one_shot:
                self._scene.set_skeletal_animation(obj, clip, blend_duration=0.0, loop=False)
                # set_skeletal_animation ignores a clip that's already
                # playing, but a shot fired mid-recoil must restart it.
                obj["anim_time"] = 0.0
                self._one_shot = obj
                self._one_shot_state = state
            else:
                self._scene.set_skeletal_animation(obj, clip)
            if _debug_active():
                print(f"  [VM DEBUG] part={part} AFTER: animation={obj['animation']!r} anim_time={obj['anim_time']:.4f} anim_blend_duration={obj['anim_blend_duration']:.4f} pose_snap_is_none={obj.get('pose_snap') is None}")

    def is_one_shot_active(self, state):
        """True while STATE's one-shot animation is the current one AND
        still actually playing (see _one_shot_finished) - False once it's
        finished, if some other state has been played since, or if STATE
        never had a clip to play in the first place. WeaponsBase.drawing/
        reloading are bound to this instead of a guessed duration constant -
        see their own docstrings for why (a fixed-timer version of this kept
        drifting out of sync with the real clip's actual length)."""
        return self._one_shot_state == state and not self._one_shot_finished()

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
        parts = [o for o in (self.arms, self.gun) if o is not None]
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
