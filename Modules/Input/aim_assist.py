"""
Cheap controller aim assist - the game only runs it while the controller is the device in use.

Two parts, both only for a target near the crosshair (inside CONE degrees, within MAX_RANGE, in line of sight):
  * slowdown: right-stick turning is scaled down as the crosshair nears the target, so it's easier to stay on;
  * pull: the view eases toward the target at up to PULL degrees/second, with or without stick input, stopping once
    the crosshair is PULL_DEPTH inside its body - it never drags the aim on to the centre.

Line of sight is checked to a few points on the body (_PARTS), not just its centre, so a target only partly in view
- a head peeking over cover, a shoulder round a corner - still counts. The assist then works on the part that's
visible (its own, smaller sphere), so it pulls on to the head rather than into the wall in front of the body.

Cost: an angle per candidate target per frame, plus up to len(_PARTS) raycasts (line of sight to the chosen target,
stopping at the first visible part) at most every LOS_INTERVAL seconds.
"""

import math
import time

import glm

CONE = 4.0                # degrees from the edge of a target's body it can be helped from
TARGET_RADIUS = 0.45      # metres: a target is a sphere this wide, so up close the cone widens to its body
PULL_DEPTH = 0.15         # metres inside the body's edge the pull carries the crosshair before it stops
MAX_RANGE = 60.0          # metres
SLOWDOWN = 0.45           # stick speed multiplier with the crosshair right on a target (1 = none)
PULL = 10.0               # degrees/second toward a target at the body's edge, easing off toward the cone's edge
LOS_INTERVAL = 0.1        # seconds a line-of-sight answer is reused

# The points line of sight is tested to, in order, as (up, sideways, radius) metres from a target's position (a
# remote player's body_center, about 1.1 m above the feet); sideways is across the line of sight. The first is the
# whole body; the rest are the parts that show on their own when the rest is behind cover.
_PARTS = (
    (0.0, 0.0, TARGET_RADIUS),   # whole body
    (0.31, 0.0, 0.22),           # head
    (-0.45, 0.0, 0.3),           # lower body and legs
    (0.0, -0.3, 0.25),           # left side
    (0.0, 0.3, 0.25),            # right side
)
_UP = glm.vec3(0.0, 1.0, 0.0)


def _yaw_pitch(direction):
    """Camera yaw/pitch (degrees) that look along `direction` (Camera.update_vectors' convention)."""
    return (math.degrees(math.atan2(direction.z, direction.x)),
            math.degrees(math.asin(max(-1.0, min(1.0, direction.y)))))


def _part_centre(eye, position, part):
    up, side, _ = _PARTS[part]
    centre = position + _UP * up
    if side:
        across = glm.cross(position - eye, _UP)
        length = glm.length(across)
        if length > 1e-6:
            centre += across / length * side
    return centre


def _angles(eye, front, centre, radius):
    """(degrees from the crosshair to the centre, to the sphere's edge, to the point PULL_DEPTH inside the edge,
    direction to the centre) for a sphere of `radius` at `centre`, or None if it's out of range."""
    to_target = centre - eye
    distance = glm.length(to_target)
    if distance < 1e-3 or distance > MAX_RANGE:
        return None
    direction = to_target / distance
    to_centre = math.degrees(math.acos(max(-1.0, min(1.0, glm.dot(front, direction)))))
    edge = to_centre - math.degrees(math.atan(radius / distance))
    depth = min(PULL_DEPTH, radius * 0.5)
    stop = to_centre - math.degrees(math.atan((radius - depth) / distance))
    return to_centre, edge, stop, direction


class AimAssist:
    def __init__(self):
        self._los = {}        # target key -> (time checked, index into _PARTS of the visible part, or None)

    def _visible_part(self, key, eye, position, line_of_sight):
        """Index into _PARTS of the first part of the target in view (None: none is), rechecked every
        LOS_INTERVAL."""
        now = time.perf_counter()
        checked = self._los.get(key)
        if checked is None or now - checked[0] > LOS_INTERVAL:
            part = 0
            if line_of_sight is not None:
                part = next((i for i in range(len(_PARTS))
                             if line_of_sight(eye, _part_centre(eye, position, i))), None)
            checked = self._los[key] = (now, part)
        return checked[1]

    def update(self, camera, targets, dt, line_of_sight):
        """targets: [(key, world position)] - every player it may help aim at. line_of_sight(a, b) -> bool.
        Returns (look_scale, yaw, pitch): multiply this frame's stick look by look_scale, then add yaw/pitch degrees."""
        eye = glm.vec3(camera.position)
        front = glm.normalize(camera.front)
        best = None
        for key, position in targets:
            position = glm.vec3(position)
            found = _angles(eye, front, position, TARGET_RADIUS)
            if found is not None and found[1] < CONE and (best is None or found[1] < best[0]):
                best = (found[1], key, position)
        if best is None:
            return 1.0, 0.0, 0.0
        _, key, position = best
        part = self._visible_part(key, eye, position, line_of_sight)
        if part is None:
            return 1.0, 0.0, 0.0
        found = _angles(eye, front, _part_centre(eye, position, part), _PARTS[part][2])
        if found is None or found[1] >= CONE:
            return 1.0, 0.0, 0.0
        to_centre, edge, stop, direction = found
        closeness = 1.0 - max(0.0, edge) / CONE                            # 1 on target, 0 at the cone's edge
        look_scale = 1.0 - (1.0 - SLOWDOWN) * closeness
        if stop <= 0.0:                            # already far enough on to it: no pull toward its centre
            return look_scale, 0.0, 0.0
        # Pull only PULL_DEPTH past its edge, not on to its centre.
        share = stop / to_centre
        target_yaw, target_pitch = _yaw_pitch(direction)
        yaw_error = ((target_yaw - camera.yaw + 180.0) % 360.0 - 180.0) * share
        pitch_error = (target_pitch - camera.pitch) * share
        step = PULL * closeness * dt
        return (look_scale,
                max(-abs(yaw_error), min(abs(yaw_error), math.copysign(step, yaw_error))),
                max(-abs(pitch_error), min(abs(pitch_error), math.copysign(step * 0.5, pitch_error))))
