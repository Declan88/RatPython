"""
Camera recoil: each shot kicks the view up (and a touch sideways) and it then
settles back down by itself - no pulling down needed.

There is ONE Recoil per player (Inventory.recoil), not one per gun. Recoil is a
state of the player's camera, and the camera does not stop being kicked because
a different weapon is now in hand: a gun only supplies the NUMBERS for the kick
it fires (its recoil_* settings - see kick), and the player's recoil carries on
draining and being applied every frame whichever weapon is out. (It used to live
on each weapon and only the active one was advanced, so switching away mid-kick
froze the camera off-centre until that gun was drawn again.)

Two moving parts. `target` is how far the recoil currently wants the camera
displaced; a shot pushes it further out and it always drains back toward zero
at the firing weapon's recoil_recovery (degrees/second). `offset` is what's
actually applied to the camera: it chases `target` at recoil_kick_speed, which
is fast, so the view snaps up on the shot and then follows the target back down.
Shooting faster than the target drains stacks kicks up to recoil_max.

The offset is applied as a DELTA on top of the camera's own yaw/pitch (see
apply), so mouse look keeps working underneath it and the recoil can never
leave the camera permanently rotated.
"""

import random


def _toward(value, goal, step):
    if value < goal:
        return min(goal, value + step)
    return max(goal, value - step)


class Recoil:
    def __init__(self):
        self._target_pitch = 0.0
        self._target_yaw = 0.0
        self.pitch = 0.0        # degrees currently applied to the camera (+ = up)
        self.yaw = 0.0
        self._applied_pitch = 0.0
        self._applied_yaw = 0.0
        # How fast the kick drains and the view chases it: whatever the weapon that last fired
        # asked for. Kept here (not read off the active weapon) so a kick keeps settling at the
        # rate it was fired with after the player has switched to something else.
        self._recovery = 6.0
        self._kick_speed = 120.0

    def kick(self, weapon):
        """One shot's worth of recoil, from `weapon`'s recoil_* settings."""
        self._recovery = weapon.recoil_recovery
        self._kick_speed = weapon.recoil_kick_speed
        limit = weapon.recoil_max
        self._target_pitch = min(limit, self._target_pitch + weapon.recoil_pitch)
        if weapon.recoil_yaw:
            kick = random.uniform(-weapon.recoil_yaw, weapon.recoil_yaw)
            self._target_yaw = max(-limit, min(limit, self._target_yaw + kick))

    def update(self, dt):
        drain = self._recovery * dt
        self._target_pitch = _toward(self._target_pitch, 0.0, drain)
        self._target_yaw = _toward(self._target_yaw, 0.0, drain)
        chase = self._kick_speed * dt
        self.pitch = _toward(self.pitch, self._target_pitch, chase)
        self.yaw = _toward(self.yaw, self._target_yaw, chase)

    def apply(self, camera, dt):
        """Advances the recoil by dt and adds the change since last frame to
        camera's yaw/pitch (recomputing its front vector). Call once a frame,
        after mouse look has moved the camera - every frame the player is alive,
        whichever weapon (or none) is in hand."""
        self.update(dt)
        d_pitch = self.pitch - self._applied_pitch
        d_yaw = self.yaw - self._applied_yaw
        if d_pitch == 0.0 and d_yaw == 0.0:
            return
        before = camera.pitch
        camera.yaw += d_yaw
        camera.pitch = max(-89.0, min(89.0, camera.pitch + d_pitch))
        camera.update_vectors()
        # What actually got applied (pitch may have hit the clamp), so the
        # next delta is measured from the real thing and nothing sticks.
        self._applied_pitch += camera.pitch - before
        self._applied_yaw = self.yaw

    def reset(self):
        """Forgets all recoil, including what is already applied to the camera - for a new
        life or match, where the camera is being put somewhere fresh anyway (a respawn sets
        its yaw/pitch outright). NOT for a weapon switch: that must let the kick keep settling."""
        self._target_pitch = self._target_yaw = 0.0
        self.pitch = self.yaw = 0.0
        self._applied_pitch = self._applied_yaw = 0.0
