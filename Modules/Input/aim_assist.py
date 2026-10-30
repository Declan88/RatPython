"""
Cheap controller aim assist - the game only runs it while the controller is the device in use.

Two parts, both only for a target near the crosshair (inside CONE degrees, within MAX_RANGE, in line of sight):
  * slowdown: right-stick turning is scaled down as the crosshair nears the target, so it's easier to stay on;
  * pull: while the player is turning or moving, the view eases toward the target at up to PULL degrees/second.
    With no input at all it does nothing - it never aims for a player who isn't playing.

Cost: an angle per candidate target per frame, plus one raycast (line of sight to the chosen target) at most every
LOS_INTERVAL seconds.
"""

import math
import time

import glm

CONE = 8.0                # degrees from the crosshair a target can be helped from
TARGET_RADIUS = 0.45      # metres: a target is a sphere this wide, so up close the cone widens to its body
MAX_RANGE = 60.0          # metres
SLOWDOWN = 0.45           # stick speed multiplier with the crosshair right on a target (1 = none)
PULL = 10.0               # degrees/second toward a target dead centre in the cone, while there's input
LOS_INTERVAL = 0.1        # seconds a line-of-sight answer is reused


def _yaw_pitch(direction):
    """Camera yaw/pitch (degrees) that look along `direction` (Camera.update_vectors' convention)."""
    return (math.degrees(math.atan2(direction.z, direction.x)),
            math.degrees(math.asin(max(-1.0, min(1.0, direction.y)))))


class AimAssist:
    def __init__(self):
        self._los = {}        # target key -> (time checked, visible)

    def update(self, camera, targets, has_input, dt, line_of_sight):
        """targets: [(key, world position)] - every player it may help aim at. has_input: the player is turning or
        moving. line_of_sight(a, b) -> bool. Returns (look_scale, yaw, pitch): multiply this frame's stick look by
        look_scale, then add yaw/pitch degrees."""
        eye = glm.vec3(camera.position)
        front = glm.normalize(camera.front)
        best = None
        for key, position in targets:
            to_target = glm.vec3(position) - eye
            distance = glm.length(to_target)
            if distance < 1e-3 or distance > MAX_RANGE:
                continue
            direction = to_target / distance
            angle = math.degrees(math.acos(max(-1.0, min(1.0, glm.dot(front, direction)))))
            angle -= math.degrees(math.atan(TARGET_RADIUS / distance))      # to the edge of its body
            if angle < CONE and (best is None or angle < best[0]):
                best = (max(0.0, angle), key, direction, glm.vec3(position))
        if best is None:
            return 1.0, 0.0, 0.0
        angle, key, direction, position = best
        now = time.perf_counter()
        checked = self._los.get(key)
        if checked is None or now - checked[0] > LOS_INTERVAL:
            visible = line_of_sight(eye, position) if line_of_sight is not None else True
            checked = self._los[key] = (now, bool(visible))
        if not checked[1]:
            return 1.0, 0.0, 0.0
        closeness = 1.0 - angle / CONE                                      # 1 on target, 0 at the cone's edge
        look_scale = 1.0 - (1.0 - SLOWDOWN) * closeness
        if not has_input:
            return look_scale, 0.0, 0.0
        target_yaw, target_pitch = _yaw_pitch(direction)
        yaw_error = (target_yaw - camera.yaw + 180.0) % 360.0 - 180.0
        pitch_error = target_pitch - camera.pitch
        step = PULL * closeness * dt
        return (look_scale,
                max(-abs(yaw_error), min(abs(yaw_error), math.copysign(step, yaw_error))),
                max(-abs(pitch_error), min(abs(pitch_error), math.copysign(step * 0.5, pitch_error))))
