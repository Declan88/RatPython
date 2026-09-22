"""Visual + physical representation of one other connected peer.

Rewritten from scratch - the previous version was leftover Panda3D code
(self.app.loader.loadModel / self.app.render) that doesn't exist
anywhere in this moderngl/pygame engine and would have crashed the
instant a peer packet arrived (see network_manager.py's own docstring
for the wider "ported from a Panda3D-based version" context). This
version builds a generic PlayerModel (Modules/Player/player_model.py -
the same class the LOCAL player's shadow-only body uses, just fully
visible here) plus a query-only hitbox, driven entirely by whatever
transform network_manager.py's handle_data() forwards from the wire.
"""

import time

import glm

from Modules.Player.player_model import PlayerModel

# A remote peer only ever broadcasts its CAMERA (eye) position over the
# network (see network_manager.py's _broadcast_transform - it sends
# camera.position/camera.yaw/camera.pitch verbatim), never a feet
# position or crouch state, so there's no way to derive an exact feet
# height from the wire data alone while crouched - this is a plain,
# documented approximation for a standing player instead. Not imported
# from CharacterController (which has no notion of "remote players" at
# all) - kept as its own constant here, set to the rat.glb model's own
# measured eye level (1.39225m above its feet) rather than a generic
# ratio, since that's exactly what app.py's own player_height is now
# chosen to reproduce too (see its comment) - a future reader can keep
# the two in sync by eye rather than an import dependency PlayerModel/
# RemotePlayer must not have.
_ASSUMED_EYE_TO_FEET = 1.39225

_DEFAULT_MODEL_PATH = "Assets/Models/rat.glb"
# Blend-state table: (name, clip, min_speed), min_speed in m/s, same
# Source velwalk/velrun thresholds (90/220 Source units/s * 0.0254)
# app.py's local player uses. rat.glb's own baked-in "New" clip is
# actually a dancing animation, not an idle one - "rifle_idle"/
# "rifle_walk"/"rifle_run" (loaded from the separate pose-only files
# below, sharing rat.glb's armature - see Scene.load_additional_
# animations) are the real idle/walk/run poses, same swap app.py's local
# player and torus_scene.py's decorative prop make. All three are
# full-body clips (legs included, not just arms), so there's no separate
# upper-body split here (unlike the local player) - a remote player's
# whole body just plays whichever of these its speed resolves to.
# RifleRunN.glb, unlike rifleidle.glb but like RifleWalkN.glb, was
# confirmed already baked at a correct 30fps (same raw-keyframe-spacing
# check as the other pose files), so it needs no time_scale correction
# either - see _POSE_FILES below.
_DEFAULT_ANIMATION_STATES = (
    ("idle", "rifle_idle", 0.0),
    ("walk", "rifle_walk", 90.0 * 0.0254),
    ("run", "rifle_run", 220.0 * 0.0254),
)
# rat.glb's own baked-in clip and rifleidle.glb were both baked with
# Blender's factory-default scene frame rate (24fps) left unchanged,
# even though actually authored/intended for 30fps - confirmed by their
# raw keyframe spacing (1/24s). RifleWalkN.glb, unlike the other two,
# has ALREADY been re-exported at a correct 30fps (confirmed: its own
# keyframes are spaced at exactly 1/30s) - giving it the same
# correction would double-correct an already-fixed file and play it 20%
# too fast, hence the per-file time_scale below rather than one shared
# constant. See skeletal_loader.load_skinned_glb/load_animation_clips'
# own time_scale docstring for the correction mechanism, and app.py/
# torus_scene.py for the same per-file split applied to the local
# player and the decorative prop.
_RAT_ANIM_TIME_SCALE = 24.0 / 30.0
_POSE_FILES = {
    "Assets/Animations/Poses/Rifle/rifleidle.glb": ({"New": "rifle_idle"}, _RAT_ANIM_TIME_SCALE),
    "Assets/Animations/Poses/Rifle/RifleWalkN.glb": ({"New": "rifle_walk"}, 1.0),
    "Assets/Animations/Poses/Rifle/RifleRunN.glb": ({"New": "rifle_run"}, 1.0),
}
# A static box approximation of the player capsule - placeholder
# "groundwork" per this feature's scope (no damage/weapon system exists
# yet to actually query it). A tighter capsule-shaped hitbox could be
# added to PhysicsWorld later if a real damage system wants one.
_DEFAULT_HITBOX_HALF_EXTENTS = (0.4, 0.9, 0.4)


class RemotePlayer:
    def __init__(self, scene, steam_id, model_path=_DEFAULT_MODEL_PATH,
                 animation_states=_DEFAULT_ANIMATION_STATES,
                 forward_offset_degrees=0.0, scale=None,
                 hitbox_half_extents=_DEFAULT_HITBOX_HALF_EXTENTS):
        self.scene = scene
        self.steam_id = steam_id

        # Fully visible (visible_in_color=True) - the opposite of the
        # local player's shadow-only body (see app.py) - other players
        # need to actually be seen.
        self.model = PlayerModel(
            scene, model_path,
            visible_in_color=True, cast_shadow=True,
            states=animation_states, forward_offset_degrees=forward_offset_degrees,
            scale=scale, time_scale=_RAT_ANIM_TIME_SCALE,
        )
        # Matches torus_scene.py's own decorative rat.glb character's
        # shading exactly (same flat "character["specular_strength"] = 0"
        # mutation there, and app.py's local player) - PlayerModel has no
        # constructor knob for this (bind_material's own default is 1.0,
        # a normal specular highlight), so it's set directly on the obj
        # dict here too.
        if self.model.obj is not None:
            self.model.obj["specular_strength"] = 0
        # Only meaningful when model_path/animation_states are left at
        # their defaults (rat.glb + the rifle_idle/rifle_walk/rifle_run
        # table above) - loading these unconditionally is harmless even
        # if a caller overrides either to something else entirely, the
        # clips just sit unused on the skeleton in that case.
        if self.model.obj is not None:
            for path, (rename, time_scale) in _POSE_FILES.items():
                scene.load_additional_animations(
                    self.model.obj, path, rename=rename, time_scale=time_scale,
                )

        self._hitbox = scene.physics.add_hitbox(hitbox_half_extents, owner=steam_id)

        self._last_feet_pos = None   # glm.vec3, for speed estimation between updates
        self._last_update_time = None  # time.monotonic()

    def update_transform(self, pos, hpr):
        """pos/hpr: exactly what network_manager.py's handle_data()
        forwards straight from the decoded JSON packet - pos is
        (x, y, z) camera/eye position, hpr is (yaw, pitch, 0.0) degrees
        (see _broadcast_transform - hpr[0] is always yaw, matching this
        project's wire format, not a real full heading/pitch/roll)."""
        eye_pos = glm.vec3(pos[0], pos[1], pos[2])
        yaw_degrees = float(hpr[0])
        feet_pos = glm.vec3(eye_pos.x, eye_pos.y - _ASSUMED_EYE_TO_FEET, eye_pos.z)

        now = time.monotonic()
        if self._last_feet_pos is not None and self._last_update_time is not None:
            elapsed = max(now - self._last_update_time, 1e-4)
            speed = glm.length(feet_pos - self._last_feet_pos) / elapsed
        else:
            # First update since construction - no prior sample to
            # diff against, so default to idle rather than a
            # divide-by-zero or a garbage first-frame speed spike. No
            # elapsed time either, for the same reason - see below.
            elapsed = 0.0
            speed = 0.0
        self._last_feet_pos = feet_pos
        self._last_update_time = now

        # dt=elapsed (real wall time since the last packet) - PlayerModel.
        # update() doesn't currently use dt for anything itself (its
        # locomotion blend space is re-evaluated fresh from speed/
        # direction every call, with no dwell timer to accumulate -
        # actual animation time advance stays entirely inside Scene.
        # update()'s own per-frame loop), but is passed through anyway to
        # match the local player's own call and in case a future need
        # arises here.
        self.model.update(elapsed, feet_pos, yaw_degrees, speed)

        # Hitbox centered vertically on the body (feet to eye), not at
        # floor level, and follows facing so a future directional query
        # (e.g. a cone/box in front of the shooter) lines up.
        self.scene.physics.update_hitbox(
            self._hitbox,
            feet_pos + glm.vec3(0.0, _ASSUMED_EYE_TO_FEET / 2.0, 0.0),
            glm.vec3(0.0, glm.radians(yaw_degrees), 0.0),
        )

    def destroy(self):
        """Not called anywhere yet - network_manager.py has no lobby-
        member-left/disconnect handling today (a pre-existing gap this
        feature doesn't add to), so a disconnected peer's model/hitbox
        currently just stays put. Exposed now so wiring that up later is
        a one-line call, not another rewrite."""
        self.model.destroy()
        if self._hitbox is not None:
            self.scene.physics.remove_hitbox(self._hitbox)
            self._hitbox = None
