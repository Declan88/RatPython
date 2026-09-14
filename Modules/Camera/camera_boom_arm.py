"""Third-person camera "boom arm" (a.k.a. spring arm) - computes an
actual camera position a fixed distance behind/above a pivot point
(typically the player's own eye position), pulled in via a physics
sweep test whenever a wall or other obstacle would otherwise put the
camera inside solid geometry - the standard "spring arm" collision
behavior most third-person games use, so the camera never clips through
a wall the player backs up against.

Deliberately independent of Camera itself - it only ever computes a
position, never owns or mutates a Camera object. The caller (app.py)
just feeds the result into camera.position each frame while in third-
person mode, exactly the way it already computes camera.position
directly (eye_offset above the player) for first-person - third person
is just a different position SOURCE, not a different kind of camera.
camera.front (and therefore the actual look direction/view matrix,
movement's own get_flat_forward(), etc.) is completely untouched by
this - the player still looks and moves via the same mouse-driven yaw/
pitch either way.
"""

import glm
from panda3d.core import TransformState
from panda3d.bullet import BulletSphereShape

from Modules.Physics.physics_world import CollisionGroup, to_physics_pos


class CameraBoomArm:
    def __init__(self, physics_world, length=4.0, height_offset=0.3,
                 collision_radius=0.2, collision_mask=CollisionGroup.ALL):
        """physics_world: the scene's PhysicsWorld (needs its own
        .world for the collision sweep - same BulletWorld everything
        else in the scene collides against).

        length: meters behind the pivot the camera sits when nothing is
        in the way. height_offset: meters ABOVE the pivot the arm
        extends from first, before going backward - without this the
        arm's collision sphere sits exactly at head height, which reads
        as "the camera is embedded in the back of the player's own
        head" the moment anything (even the player's own model) is
        nearby; lifting it a bit gives the classic slightly-elevated
        third-person view instead.

        collision_radius: the arm's own collision probe size - a small
        sphere (spheres being the standard, simplest choice for this,
        same reasoning most engines use it) swept from the pivot to the
        desired camera position; using a real radius rather than a bare
        ray keeps the camera a small distance clear of a wall's surface
        rather than resting exactly on it (which would let the near
        clip plane poke through)."""
        self._physics_world = physics_world
        self.length = float(length)
        self.height_offset = float(height_offset)
        self._shape = BulletSphereShape(float(collision_radius))
        # Excludes PLAYER so the arm's own sweep never collides with the
        # very player capsule it's orbiting - same exclusion (and same
        # underlying phantom-hit reasoning) as CharacterController's own
        # _sweep_mask for its self-collision sweeps.
        self._sweep_mask = collision_mask & ~CollisionGroup.PLAYER

    def _sweep(self, from_pos, to_pos):
        from_ts = TransformState.makePos(to_physics_pos(from_pos))
        to_ts = TransformState.makePos(to_physics_pos(to_pos))
        return self._physics_world.world.sweepTestClosest(self._shape, from_ts, to_ts, self._sweep_mask, 0.0)

    def get_camera_position(self, pivot, yaw_degrees, pitch_degrees):
        """pivot: glm.vec3, the point the arm extends backward from -
        typically the player's own eye position (the same one first-
        person already uses), so switching modes doesn't otherwise
        change where the player appears to be standing.

        yaw_degrees/pitch_degrees: same convention Camera itself uses
        (camera.yaw/camera.pitch, degrees) - the arm points the camera
        back along the same direction the player is actually looking,
        so looking up swings the camera up and back over the player's
        shoulder rather than orbiting on some independent axis."""
        forward = glm.vec3(
            glm.cos(glm.radians(yaw_degrees)) * glm.cos(glm.radians(pitch_degrees)),
            glm.sin(glm.radians(pitch_degrees)),
            glm.sin(glm.radians(yaw_degrees)) * glm.cos(glm.radians(pitch_degrees)),
        )
        raised_pivot = glm.vec3(pivot) + glm.vec3(0.0, self.height_offset, 0.0)
        desired = raised_pivot - forward * self.length

        result = self._sweep(raised_pivot, desired)
        if result.hasHit():
            # Pull in along the same line to just short of whatever was
            # hit, rather than snapping exactly onto the surface.
            fraction = max(result.getHitFraction() - 0.05, 0.0)
            return raised_pivot + (desired - raised_pivot) * fraction
        return desired
