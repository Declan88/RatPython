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
# height from the wire data alone - this is a plain, documented
# approximation instead. Not imported from CharacterController (which
# has no notion of "remote players" at all, and whose own height can
# change with crouch, which isn't synced) - kept as its own constant
# here, deliberately close to this project's default standing eye
# height: app.py's CharacterController(height=1.5) with
# character_controller.py's default _DEFAULT_EYE_RATIO (0.7/1.8) puts
# the local player's own eye at height/2 + height*0.7/1.8 =~ 1.33m above
# its feet, which is exactly this constant - a future reader can keep
# the two in sync by eye rather than an import dependency PlayerModel/
# RemotePlayer must not have.
_ASSUMED_EYE_TO_FEET = 1.3333

_DEFAULT_MODEL_PATH = "Assets/Models/rat.glb"
_DEFAULT_IDLE_ANIMATION = "funnyrat_ARMAction"
# A static box approximation of the player capsule - placeholder
# "groundwork" per this feature's scope (no damage/weapon system exists
# yet to actually query it). A tighter capsule-shaped hitbox could be
# added to PhysicsWorld later if a real damage system wants one.
_DEFAULT_HITBOX_HALF_EXTENTS = (0.4, 0.9, 0.4)


class RemotePlayer:
    def __init__(self, scene, steam_id, model_path=_DEFAULT_MODEL_PATH,
                 idle_animation=_DEFAULT_IDLE_ANIMATION,
                 walk_animation=None, run_animation=None,
                 forward_offset_degrees=0.0,
                 hitbox_half_extents=_DEFAULT_HITBOX_HALF_EXTENTS):
        self.scene = scene
        self.steam_id = steam_id

        # Fully visible (visible_in_color=True) - the opposite of the
        # local player's shadow-only body (see app.py) - other players
        # need to actually be seen.
        self.model = PlayerModel(
            scene, model_path,
            visible_in_color=True, cast_shadow=True,
            idle_animation=idle_animation, walk_animation=walk_animation,
            run_animation=run_animation, forward_offset_degrees=forward_offset_degrees,
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
            # divide-by-zero or a garbage first-frame speed spike.
            speed = 0.0
        self._last_feet_pos = feet_pos
        self._last_update_time = now

        self.model.update(0.0, feet_pos, yaw_degrees, speed)

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
