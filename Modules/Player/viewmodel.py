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
set_weapon() loads them.
"""

import glm

from Modules.Player.rat_colors import RAT_TINT_MASK_PATH
from Modules.Weapons.weapons_base import _load_state_clip

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
        self._weapon = None
        self._clips = {}          # (part name, state) -> clip name
        self._one_shot = None     # part whose one-shot animation is playing
        self.arms = None
        self.gun = None

    def set_weapon(self, weapon):
        """Loads `weapon`'s arms and gun and gives each its animations."""
        self.clear_weapon()
        self._weapon = weapon
        path = weapon.viewmodel_model
        self.arms = self._scene.add_viewmodel(
            path, tint_mask_path=self._tint_mask_path, skin_index=weapon.viewmodel_arms_skin)
        self.gun = self._scene.add_viewmodel(path, skin_index=weapon.viewmodel_gun_skin)
        if self.arms is not None:
            self._scene.set_skeletal_tint(self.arms, self._tint)
        # The gun rig's clips (see WeaponsBase.viewmodel_animations) play on
        # both parts: it's the arms rig plus the weapon bones.
        for state, clip_path in weapon.viewmodel_animations.items():
            for part, obj in (("arms", self.arms), ("gun", self.gun)):
                clip = _load_state_clip(
                    self._scene, obj, clip_path, f"{weapon.player_clip(state)}_viewmodel",
                    skin_index=weapon.viewmodel_gun_skin)
                if clip is not None:
                    self._clips[(part, state)] = clip
        self.play("idle")

    def clear_weapon(self):
        for obj in (self.arms, self.gun):
            if obj is not None:
                self._scene.remove_viewmodel(obj)
        self.arms = self.gun = None
        self._clips.clear()
        self._weapon = None

    def play(self, state):
        """Switches the arms and gun to `state`'s animation (a part with no
        clip for that state is left as it is). A one-shot state (see
        WeaponsBase.viewmodel_one_shot_states) plays once, from the start, and
        update() drops back to idle when it ends. Playing it again while it's
        still going restarts it from the beginning."""
        one_shot = self._weapon is not None and state in self._weapon.viewmodel_one_shot_states
        self._one_shot = None
        for part, obj in (("arms", self.arms), ("gun", self.gun)):
            clip = self._clips.get((part, state))
            if obj is None or clip is None:
                continue
            if one_shot:
                self._scene.set_skeletal_animation(obj, clip, blend_duration=0.0, loop=False)
                # set_skeletal_animation ignores a clip that's already
                # playing, but a shot fired mid-recoil must restart it.
                obj["anim_time"] = 0.0
                self._one_shot = obj
            else:
                self._scene.set_skeletal_animation(obj, clip)

    def _one_shot_finished(self):
        obj = self._one_shot
        if obj is None:
            return False
        clip = obj["skeleton"].animations.get(obj["animation"])
        return clip is not None and obj["anim_time"] >= max(ch.times[-1] for ch in clip.channels)

    def set_tint(self, rgb):
        """Fur color on the arms, same as PlayerModel.set_tint - 0-1 floats or
        None."""
        self._tint = None if rgb is None else tuple(rgb)
        if self.arms is not None:
            self._scene.set_skeletal_tint(self.arms, self._tint)

    def update(self, camera, visible):
        """Call once per frame with the camera as it will render; visible
        False (third person, dead, menu...) hides everything."""
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
