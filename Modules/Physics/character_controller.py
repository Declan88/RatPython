"""
Player capsule: gravity, walking, sprinting, crouching, jumping, with
Source-engine-style (Half-Life 2 / GMod) movement feel, built on
Panda3D's BulletCharacterControllerNode for collision/step/slope
resolution (see physics_world.py's module docstring for the Z-up/Y-up
coordinate bridge and why a hand-rolled character controller isn't
needed just for that part).

The horizontal movement model below is a direct port of the algorithm
in Source's public SDK (game/shared/gamemovement.cpp - CGameMovement::
WalkMove/AirMove/Friction/Accelerate/AirAccelerate), not Bullet's own
walk handling: BulletCharacterControllerNode's setLinearMovement()
snaps straight to a target speed every tick, which feels nothing like
Source/Quake movement. Ground movement here instead:

  1. Applies FRICTION to the player's own persistent velocity (decayed
     by sv_friction-equivalent, with a stop_speed floor so slow motion
     doesn't asymptote forever - same shape as Source's Friction()).
  2. ACCELERATES that velocity toward the wish direction/speed, capped
     by how much speed is still missing (sv_accelerate-equivalent) -
     this is why starting to move ramps up instead of snapping to max
     speed, and why turning while moving carries momentum instead of
     instantly redirecting.

Airborne movement uses Source's AirAccelerate instead: the ADD-SPEED
check is capped at ~30 Source units/s (0.762m, the exact unit
conversion - 1 Source unit = 1 inch), but the accel-speed applied per
tick still uses the UNCAPPED wishspeed. That asymmetry is deliberate
and is exactly the mechanism behind Source/Quake air-strafing and
bunny-hop acceleration - reproduced faithfully here, not accidentally.

CROUCHING (Source/GMod-style, default bind CTRL): two things happen,
mostly independently:
  - The collision capsule swaps to a shorter one INSTANTLY the moment
    crouch is pressed, so the player can duck under a low obstacle
    right away rather than waiting on an animation - this is genuinely
    how Source's hitbox works too (the view height is what's animated,
    not the hitbox). Bullet's character controller node has no runtime
    "resize", so this rebuilds the node with a second pre-built capsule
    shape, preserving position (feet planted - see _swap_to_shape) and
    velocity.
  - Standing back up is refused if a BulletGhostNode positioned where
    the standing capsule would go detects any overlap (a low ceiling,
    a shelf) - same "don't pop through geometry" rule Source enforces -
    so releasing crouch under something low just keeps you crouched
    until you move clear, checked every tick.
  - The camera's eye height (get_eye_offset()) eases smoothly toward
    the crouched value over crouch_transition_time instead of popping,
    computed from the FEET position (invariant across the instant hull
    swap) so the transition reads as smooth despite the hull itself
    changing in one step.
  - Move speed is scaled by crouch_speed_multiplier while crouched
    (Source/CS-style ducking is slow, not just shorter).

Because all of this needs to run at a fixed tick rate (friction/
acceleration math, and the crouch/eye lerp, are only frame-rate
independent if dt is constant - Source itself runs movement on the
server's fixed tick, not the render frame), it's registered as a
PhysicsWorld pre-substep callback rather than being driven from
Scene/app.py's per-render-frame update.
"""

import glm
from panda3d.core import Point3
from panda3d.bullet import BulletCapsuleShape, BulletBoxShape, BulletCharacterControllerNode, BulletGhostNode, ZUp

from Modules.Physics.physics_world import CollisionGroup, to_physics_pos, to_physics_vec, to_physics_extent, to_render_pos

# Source's AirAccelerate add-speed cap is 30 units/s; 1 Source unit is
# 1 inch (0.0254m), so 30 * 0.0254 = 0.762 m/s - the exact conversion,
# not a rounded guess.
_AIR_SPEED_CAP = 30.0 * 0.0254

# Ratio of eye_height/crouch_eye_height to height/crouch_height when
# not given explicitly - matches this project's original hardcoded
# PLAYER_EYE_HEIGHT=0.7 default for height=1.8 (eyes a bit below the
# top of the capsule), kept as a ratio so crouch_eye_height scales
# sensibly with whatever height/crouch_height_ratio are passed.
_DEFAULT_EYE_RATIO = 0.7 / 1.8


class CharacterController:
    def __init__(self, physics_world, position=(0.0, 2.0, 0.0), radius=0.4,
                 height=1.8, step_height=0.4, ground_speed=4.0, sprint_speed=7.0,
                 ground_accel=10.0, air_accel=10.0, friction=4.0, stop_speed=1.0,
                 jump_height=1.2, jump_speed=6.0, collision_mask=CollisionGroup.ALL,
                 crouch_height_ratio=0.5, crouch_speed_multiplier=0.34,
                 crouch_transition_time=0.25, eye_height=None, crouch_eye_height=None):
        """ground_speed/sprint_speed: target ground move speed (m/s),
        walk vs. sprint (see set_sprinting). ground_accel/air_accel:
        sv_accelerate/sv_airaccelerate-equivalent - higher snaps to
        target speed faster. friction/stop_speed: sv_friction/
        sv_stopspeed-equivalent ground deceleration. height is the
        player's total STANDING capsule height including both
        hemispherical caps; crouch_height_ratio scales that down for
        the crouched capsule (0.5 matches Source's 36/72 duck-hull
        ratio). crouch_speed_multiplier is approximate - Source-family
        games commonly use ~0.34, but the exact HL2 constant isn't
        something this port claims to reproduce byte-for-byte; tune to
        taste. eye_height/crouch_eye_height default to a fixed ratio of
        height/crouch height if not given (see get_eye_offset)."""
        self.radius = float(radius)
        self.height = float(height)
        self.ground_speed = float(ground_speed)
        self.sprint_speed = float(sprint_speed)
        self.ground_accel = float(ground_accel)
        self.air_accel = float(air_accel)
        self.friction = float(friction)
        self.stop_speed = float(stop_speed)
        self.crouch_speed_multiplier = float(crouch_speed_multiplier)
        self.crouch_transition_time = float(crouch_transition_time)
        self.eye_height = float(eye_height) if eye_height is not None else self.height * _DEFAULT_EYE_RATIO

        crouch_height = self.height * float(crouch_height_ratio)
        self.crouch_eye_height = (
            float(crouch_eye_height) if crouch_eye_height is not None else crouch_height * _DEFAULT_EYE_RATIO
        )

        self.step_height = float(step_height)
        self.jump_height = float(jump_height)
        self.jump_speed = float(jump_speed)
        self.collision_mask = collision_mask
        self._physics_world = physics_world

        self._standing_cylinder_height = max(self.height - 2.0 * self.radius, 0.01)
        self._crouch_cylinder_height = max(crouch_height - 2.0 * self.radius, 0.01)
        self._standing_shape = BulletCapsuleShape(self.radius, self._standing_cylinder_height, ZUp)
        self._crouch_shape = BulletCapsuleShape(self.radius, self._crouch_cylinder_height, ZUp)
        self._current_cylinder_height = self._standing_cylinder_height

        self.node = self._make_node(self._standing_shape)
        self.node_path = physics_world._root.attachNewNode(self.node)
        self.node_path.setPos(to_physics_pos(position))
        physics_world.world.attachCharacter(self.node)

        # "Can I stand up" probe: a flat-topped/bottomed box covering
        # only the HEADROOM slice standing would newly occupy (crouch-
        # top to stand-top), not the full standing capsule re-tested
        # from the floor up. Re-testing the full capsule from the floor
        # would always register an overlap with the floor itself (a
        # grounded body always rests with a hair of contact
        # penetration), permanently refusing to stand up anywhere -
        # confirmed by hitting exactly that bug during testing. A probe
        # that only spans the newly-needed headroom sidesteps the floor
        # entirely by construction, no epsilon-tuning required.
        headroom = (
            self._capsule_total_height(self._standing_cylinder_height)
            - self._capsule_total_height(self._crouch_cylinder_height)
        )
        self._stand_check_shape = BulletBoxShape(to_physics_extent((self.radius, headroom / 2.0, self.radius)))

        # Kept alive for the controller's whole lifetime and just
        # repositioned each crouched tick, rather than attaching/
        # detaching a temporary node every check.
        self._stand_check_ghost = BulletGhostNode("stand_check")
        self._stand_check_ghost.addShape(self._stand_check_shape)
        self._stand_check_ghost.setIntoCollideMask(collision_mask)
        self._stand_check_ghost_np = physics_world._root.attachNewNode(self._stand_check_ghost)
        physics_world.world.attachGhost(self._stand_check_ghost)

        # Persistent ground-plane velocity (render-space, Y always 0) -
        # this is what makes movement feel like Source instead of
        # snapping to speed: it's built up/decayed tick over tick by
        # _on_pre_substep, never reset to a target value directly.
        self.velocity = glm.vec3(0.0)
        self._move_direction = glm.vec3(0.0)
        self._sprinting = False
        self._jump_requested = False
        self._crouch_input = False
        self._is_crouched = False
        self._crouch_amount = 0.0  # 0 = standing, 1 = fully crouched (eased, see get_eye_offset)

        physics_world.add_pre_substep_callback(self._on_pre_substep)

    def _make_node(self, shape):
        node = BulletCharacterControllerNode(shape, self.step_height, "player")
        node.setIntoCollideMask(self.collision_mask)
        node.setMaxJumpHeight(self.jump_height)
        node.setJumpSpeed(self.jump_speed)
        return node

    def _capsule_total_height(self, cylinder_height):
        return cylinder_height + 2.0 * self.radius

    def set_move_direction(self, direction):
        """direction: glm.vec3 (or anything glm.vec3() accepts) on the
        render XZ ground plane - Y is ignored. Only the direction
        matters (this is normalized internally to get wishdir); pass a
        zero vector when no movement key is held."""
        self._move_direction = glm.vec3(direction)
        self._move_direction.y = 0.0

    def set_sprinting(self, sprinting):
        """sprinting=True uses sprint_speed as the wish speed instead
        of ground_speed - call every frame with the sprint key's
        current held state (see app.py), not just on press. Ignored
        while crouched, same as Source - ducking is always slow."""
        self._sprinting = bool(sprinting)

    def set_crouching(self, crouching):
        """crouching=True shrinks the collision capsule immediately and
        starts easing the camera toward crouch_eye_height.
        crouching=False requests standing back up, but is refused (and
        silently retried every tick) while something overhead blocks
        it - see the module docstring. Call every frame with the crouch
        key's current held state, not just on press."""
        self._crouch_input = bool(crouching)

    def jump(self):
        """Queues a jump for the next physics tick this capsule is on
        the ground - safe to call every frame while the jump key is
        held (matches Source: holding jump auto-hops on landing rather
        than requiring a fresh press each time)."""
        self._jump_requested = True

    def is_on_ground(self):
        return self.node.isOnGround()

    def is_crouched(self):
        return self._is_crouched

    def _wish(self):
        length = glm.length(self._move_direction)
        if length <= 1e-6:
            return glm.vec3(0.0), 0.0
        wishdir = self._move_direction / length
        wishspeed = self.sprint_speed if self._sprinting else self.ground_speed
        if self._is_crouched:
            wishspeed *= self.crouch_speed_multiplier
        return wishdir, wishspeed

    def _apply_friction(self, dt):
        speed = glm.length(self.velocity)
        if speed < 1e-6:
            self.velocity = glm.vec3(0.0)
            return
        control = max(speed, self.stop_speed)
        drop = control * self.friction * dt
        new_speed = max(speed - drop, 0.0)
        self.velocity *= new_speed / speed

    def _accelerate(self, wishdir, wishspeed, accel, dt):
        current_speed = glm.dot(self.velocity, wishdir)
        add_speed = wishspeed - current_speed
        if add_speed <= 0.0:
            return
        accel_speed = min(accel * dt * wishspeed, add_speed)
        self.velocity += accel_speed * wishdir

    def _air_accelerate(self, wishdir, wishspeed, accel, dt):
        # The add-speed check uses the CAPPED wishspeed (limits how much
        # a single tick's air control can redirect existing momentum),
        # but accel_speed's magnitude still comes from the UNCAPPED
        # wishspeed - see the module docstring for why this asymmetry
        # is intentional, not a typo.
        capped_wishspeed = min(wishspeed, _AIR_SPEED_CAP)
        current_speed = glm.dot(self.velocity, wishdir)
        add_speed = capped_wishspeed - current_speed
        if add_speed <= 0.0:
            return
        accel_speed = min(accel * dt * wishspeed, add_speed)
        self.velocity += accel_speed * wishdir

    def _can_stand(self):
        # Ignore self.node itself - the ghost is positioned right where
        # the player's own capsule roughly is, so it would otherwise
        # always "detect" the player as blocking its own stand-up.
        for node in self._stand_check_ghost.getOverlappingNodes():
            if node != self.node:
                return False
        return True

    def _position_stand_check_ghost(self):
        # Centers the headroom-slice probe (see __init__) directly
        # above the player's current top, spanning up to where the
        # standing capsule's top would be - never reaching down toward
        # the floor at all, by construction. current_top is the current
        # (crouched) capsule's TOP, not its center - the probe's center
        # is current_top plus half the headroom slice above it.
        current_pos = self.node_path.getPos()
        standing_total = self._capsule_total_height(self._standing_cylinder_height)
        current_total = self._capsule_total_height(self._current_cylinder_height)
        headroom = standing_total - current_total
        current_top = current_pos.z + current_total / 2.0
        candidate_pos = Point3(current_pos.x, current_pos.y, current_top + headroom / 2.0)
        self._stand_check_ghost_np.setPos(candidate_pos)

    def _swap_to_shape(self, shape, cylinder_height):
        # Keeps the capsule's BOTTOM (feet) fixed across the swap by
        # shifting the center by half the total-height difference,
        # rather than leaving the center in place - crouching should
        # lower the head, not lift the feet off the ground.
        old_total = self._capsule_total_height(self._current_cylinder_height)
        new_total = self._capsule_total_height(cylinder_height)
        old_pos = self.node_path.getPos()
        new_pos = Point3(old_pos.x, old_pos.y, old_pos.z + (new_total - old_total) / 2.0)

        self._physics_world.world.removeCharacter(self.node)
        self.node_path.removeNode()

        new_node = self._make_node(shape)
        new_node_path = self._physics_world._root.attachNewNode(new_node)
        new_node_path.setPos(new_pos)
        self._physics_world.world.attachCharacter(new_node)

        self.node = new_node
        self.node_path = new_node_path
        self._current_cylinder_height = cylinder_height

    def _update_crouch(self, dt):
        if self._crouch_input:
            if not self._is_crouched:
                self._swap_to_shape(self._crouch_shape, self._crouch_cylinder_height)
                self._is_crouched = True
        elif self._is_crouched and self._can_stand():
            self._swap_to_shape(self._standing_shape, self._standing_cylinder_height)
            self._is_crouched = False

        if self._is_crouched:
            # Positions the ghost for the NEXT tick's _can_stand() read -
            # BulletGhostNode overlap results only refresh after a
            # doPhysics() call, so this tick's query above reflects
            # wherever the ghost was left last tick, a ~1/120s-old
            # result. Imperceptible at that rate.
            self._position_stand_check_ghost()

        target = 1.0 if self._is_crouched else 0.0
        rate = dt / max(self.crouch_transition_time, 1e-6)
        if self._crouch_amount < target:
            self._crouch_amount = min(self._crouch_amount + rate, target)
        else:
            self._crouch_amount = max(self._crouch_amount - rate, target)

    def _on_pre_substep(self, dt):
        self._update_crouch(dt)

        grounded = self.is_on_ground()
        wishdir, wishspeed = self._wish()

        if grounded:
            self._apply_friction(dt)
            self._accelerate(wishdir, wishspeed, self.ground_accel, dt)
        else:
            self._air_accelerate(wishdir, wishspeed, self.air_accel, dt)

        if self._jump_requested and grounded:
            self.node.doJump()
        self._jump_requested = False

        self.node.setLinearMovement(to_physics_vec(self.velocity), True)

    def get_position(self):
        """Render-space glm.vec3 - the capsule's current center
        position (its height above this changes when crouched - see
        get_eye_offset for a camera offset that accounts for that)."""
        return to_render_pos(self.node_path.getPos())

    def get_eye_offset(self):
        """Render-space Y offset from get_position() a first-person
        camera should sit at. Computed from the FEET position (which
        _swap_to_shape keeps invariant across the standing/crouch hull
        swap) rather than get_position() directly, so this eases
        smoothly across crouch_transition_time even though the
        collision hull itself resizes in one instant - see the module
        docstring's CROUCHING section."""
        standing_total = self._capsule_total_height(self._standing_cylinder_height)
        crouch_total = self._capsule_total_height(self._crouch_cylinder_height)
        current_total = self._capsule_total_height(self._current_cylinder_height)

        standing_feet_to_eye = standing_total / 2.0 + self.eye_height
        crouch_feet_to_eye = crouch_total / 2.0 + self.crouch_eye_height
        feet_to_eye = standing_feet_to_eye + (crouch_feet_to_eye - standing_feet_to_eye) * self._crouch_amount

        feet_offset_from_center = -current_total / 2.0
        return feet_offset_from_center + feet_to_eye

    def destroy(self):
        self._physics_world.remove_pre_substep_callback(self._on_pre_substep)
        self._physics_world.world.removeGhost(self._stand_check_ghost)
        self._stand_check_ghost_np.removeNode()
        self._physics_world.world.removeCharacter(self.node)
        self.node_path.removeNode()
