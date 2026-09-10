"""
Player hull: gravity, walking, sprinting, crouching, jumping, and
collision resolution, all built as a direct, from-scratch port of
Source's public SDK player movement (game/shared/gamemovement.cpp -
CGameMovement) rather than on top of any vendor "kinematic character
controller" black box.

WHY NOT BulletCharacterControllerNode: an earlier version of this file
used Panda3D's BulletCharacterControllerNode, which handles gravity,
jumping, ground detection and step/slope resolution internally. That
works, but two things about it fundamentally conflict with "as close to
Source as possible":

  1. Its internal vertical fall-speed is completely opaque - Panda3D's
     bindings expose no getter or setter for it at all. Since Bullet's
     kinematic controller also has no runtime shape-resize (needed for
     crouching), the old code had to destroy and recreate the whole
     node every time the player crouched or stood up, which silently
     reset that hidden fall-speed to zero - a fall in progress would
     slow to a crawl, a jump's rise would flatten out. Real Source
     ducking never touches velocity at all, because Source's movement
     was never tied to the hull object to begin with (see below). A
     workable patch existed (re-priming the fresh node's fall-speed via
     a deliberately tiny extra physics step with temporarily-boosted
     gravity), but it was still working around a black box rather than
     actually owning the state.
  2. setLinearMovement() snaps straight to a target speed every tick,
     which is nothing like Source/Quake's momentum-carrying movement -
     the previous version already had to bolt Source's own Friction/
     Accelerate/AirAccelerate math on top of it for ground movement.

Real Source movement has neither of these problems because it was
never built on a packaged character controller in the first place:
CGameMovement stores the player's full velocity itself (mv->
m_vecVelocity, one vector, gravity and jumping and walking all just
adjusting the SAME vector), and figures out the ground and resolves
collisions with its own TracePlayerBBox sweep-and-slide, run against
the game's collision world. Crouching there just changes the hull's
mins/maxs; the velocity vector was never part of the hull object, so
there's nothing to lose.

This file follows that shape directly, using Bullet only for its
raw collision primitives (a BulletGhostNode's box shape for the hull,
and BulletWorld.sweepTestClosest for the actual trace) rather than its
higher-level character controller:

  - self.velocity is the single source of truth for all motion,
    horizontal AND vertical (Source's mv->m_vecVelocity) - a plain
    Python attribute, never reset by anything else in this file.
  - _categorize_position() (Source's CGameMovement::CategorizePosition)
    sweeps the hull down a short fixed distance each tick to determine
    ground contact and the ground normal, independent of velocity.
  - _try_move()/_step_slide_move() (Source's TryPlayerMove/StepMove)
    sweep the hull along the velocity, clip against whatever it hits
    (sliding along walls, or along the crease where two hit planes
    meet, up to 4 bumps per tick) and - while grounded - separately try
    stepping up over the obstacle first, keeping whichever attempt
    covers more ground, same as Source's dual flat-vs-stepped attempt.
  - Friction/Accelerate/AirAccelerate are unchanged from the previous
    version (already a faithful port - see their docstrings).
  - Crouching swaps the hull's BulletShape via BulletGhostNode's own
    addShape/removeShape (confirmed to work on a live node, unlike
    BulletCharacterControllerNode) - the SAME ghost node the whole
    time, so self.velocity is simply never touched by it.

Because all of this needs to run at a fixed tick rate (friction/
acceleration math, and the crouch/eye lerp, are only frame-rate
independent if dt is constant - Source itself runs movement on the
server's fixed tick, not the render frame), it's registered as a
PhysicsWorld pre-substep callback rather than being driven from
Scene/app.py's per-render-frame update.
"""

import math

import glm
from panda3d.core import Point3, TransformState
from panda3d.bullet import BulletBoxShape, BulletGhostNode

from Modules.Physics.physics_world import (
    CollisionGroup, to_physics_pos, to_render_pos, to_physics_extent, to_render_vec,
)
from Modules.Audio.footstep_materials import get_footstep_volume

# Time constant (seconds) for the vertical-only low-pass filter in
# _on_pre_substep - see its comment. Short enough that stairs/slopes
# still feel immediate (reaches ~95% of a step change in about 3*tau =
# 0.15s), long enough to average out any residual per-tick sweep-test
# noise while walking.
_EYE_SMOOTH_TAU = 0.05

# Source's AirAccelerate add-speed cap is 30 units/s; 1 Source unit is
# 1 inch (0.0254m), so 30 * 0.0254 = 0.762 m/s - the exact conversion,
# not a rounded guess.
_AIR_SPEED_CAP = 30.0 * 0.0254

# Ratio of eye_height/crouch_eye_height to height/crouch_height when
# not given explicitly - matches this project's original hardcoded
# PLAYER_EYE_HEIGHT=0.7 default for height=1.8 (eyes a bit below the
# top of the hull), kept as a ratio so crouch_eye_height scales
# sensibly with whatever height/crouch_height_ratio are passed.
_DEFAULT_EYE_RATIO = 0.7 / 1.8

# Source's CategorizePosition ground trace is a fixed 2 (Source) units
# regardless of velocity - 2 * 0.0254 = 0.0508m, the exact conversion.
_GROUND_TRACE_DISTANCE = 2.0 * 0.0254

# A small pushback applied along a hit surface's normal after each
# sweep-test collision in _try_move, so the NEXT sweep in the same call
# (or next tick) starts already clear of the surface instead of
# exactly touching/embedded in it - without this, a sweep that starts
# exactly at a touching point can behave inconsistently right at the
# boundary. Small enough to be visually and physically meaningless.
_SURFACE_PUSHBACK = 0.001

# Up to 4 distinct planes per tick before giving up and treating the
# player as fully stuck (matches Source's TryPlayerMove numbumps).
_MAX_BUMPS = 4

# Direct port of the constants in Source SDK 2013's
# CBasePlayer::UpdateStepSound / GetStepSoundVelocities /
# SetStepSoundTime (game/shared/baseplayer_shared.cpp) - confirmed by
# reading that file directly, not guessed. Source's footstep cadence is
# a FIXED TIME interval between steps (a countdown timer, reset only
# when a step actually fires), NOT distance-based - counterintuitive
# for a "footstep" but that's genuinely how it works: speed only
# affects which of two fixed intervals applies (walk vs run), not a
# continuously-scaling rate. All speed values are Source's own
# (velwalk/velrun in Source units/s = inches/s) converted to meters via
# *0.0254, matching every other Source-unit conversion in this file.
_STEP_MIN_SPEED_STAND = 90.0 * 0.0254   # velwalk - below this, no footsteps at all
_STEP_RUN_SPEED_STAND = 220.0 * 0.0254  # velrun - bWalking = speed < velrun
_STEP_MIN_SPEED_CROUCH = 60.0 * 0.0254
_STEP_RUN_SPEED_CROUCH = 80.0 * 0.0254

_STEP_INTERVAL_WALK = 0.4   # STEPSOUNDTIME_NORMAL, bWalking ? 400 : 300 (ms -> s)
_STEP_INTERVAL_RUN = 0.3
_STEP_INTERVAL_CROUCH_EXTRA = 0.1  # SetStepSoundTime's "+= 100" while FL_DUCKING

# Per-surface walk/run volumes now come from footstep_materials.
# get_footstep_volume (Source's own psurface->game.material switch in
# UpdateStepSound), not a flat constant here. Ducking multiplies
# whatever that returns by 0.65, exactly like UpdateStepSound's own
# "if (FL_DUCKING) fvol *= 0.65".
_STEP_VOLUME_DUCK_MULT = 0.65


class CharacterController:
    def __init__(self, physics_world, position=(0.0, 2.0, 0.0), radius=0.4,
                 height=1.8, step_height=.5, ground_speed=4.0, sprint_speed=7.0,
                 ground_accel=10.0, air_accel=10.0, friction=4.0, stop_speed=1.0,
                 jump_speed=6.0, collision_mask=CollisionGroup.ALL,
                 crouch_height_ratio=0.5, crouch_speed_multiplier=0.34,
                 crouch_transition_time=0.25, eye_height=None, crouch_eye_height=None,
                 max_slope_degrees=75, gravity=None):
        """ground_speed/sprint_speed: target ground move speed (m/s),
        walk vs. sprint (see set_sprinting). ground_accel/air_accel:
        sv_accelerate/sv_airaccelerate-equivalent - higher snaps to
        target speed faster. friction/stop_speed: sv_friction/
        sv_stopspeed-equivalent ground deceleration. radius/height
        define an axis-aligned BOX hull (Source's player hull is
        literally an AABB, not a capsule - radius is half the box's
        X/Z footprint). crouch_height_ratio scales height down for the
        crouched hull (0.5 matches Source's 36/72 duck-hull ratio).
        crouch_speed_multiplier is approximate - Source-family games
        commonly use ~0.34, but the exact HL2 constant isn't something
        this port claims to reproduce byte-for-byte; tune to taste.
        eye_height/crouch_eye_height default to a fixed ratio of
        height/crouch height if not given (see get_eye_offset).
        max_slope_degrees: a ground contact steeper than this counts as
        a wall, not floor (Source's default is 45.57 degrees, i.e. a
        ground-normal.y cutoff of ~0.7). gravity defaults to matching
        physics_world's own gravity if not given."""
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

        self._crouch_height = self.height * float(crouch_height_ratio)
        self.crouch_eye_height = (
            float(crouch_eye_height) if crouch_eye_height is not None else self._crouch_height * _DEFAULT_EYE_RATIO
        )

        self.step_height = float(step_height)
        self.jump_speed = float(jump_speed)
        self.collision_mask = collision_mask
        self.gravity = float(gravity) if gravity is not None else physics_world.world.getGravity().length()
        self._max_slope_cos = math.cos(math.radians(max_slope_degrees))
        # Worst-case rise-per-run of any surface still classified as
        # walkable floor - see _step_slide_move's settle_distance, which
        # scales with this so a fast-moving tick's settle sweep can
        # still reach a steep-but-walkable ramp below.
        self._max_slope_tan = math.tan(math.radians(max_slope_degrees))
        self._physics_world = physics_world

        # sweepTestClosest() takes a bare shape + transform, not a body
        # reference, so Bullet has no way to know "don't count the body
        # this shape happens to belong to" - confirmed empirically that
        # it reports a same-position, zero-fraction hit against our OWN
        # ghost when nothing else is around to compete for "closest".
        # That phantom hit's near-zero-but-nonzero normal (floating-
        # point noise in the exactly-touching case) was getting run
        # through ClipVelocity every tick, which is exactly why gravity
        # alone was leaking into a slow horizontal drift with no input
        # and nothing else in the world at all. Fixed by giving the
        # player's own ghost a dedicated identity bit that every
        # outgoing sweep explicitly excludes from its query mask - the
        # ghost can still be hit by OTHER systems querying for ALL/
        # PLAYER (contactTest's exclusion-by-identity-check, for
        # instance, is unaffected), it just can never satisfy our own
        # movement sweeps' mask.
        self._sweep_mask = self.collision_mask & ~CollisionGroup.PLAYER

        self._standing_shape = BulletBoxShape(to_physics_extent((self.radius, self.height / 2.0, self.radius)))
        self._crouch_shape = BulletBoxShape(to_physics_extent((self.radius, self._crouch_height / 2.0, self.radius)))
        self._current_shape = self._standing_shape
        self._current_height = self.height

        # A single persistent BulletGhostNode for the whole lifetime of
        # the controller - crouching swaps its shape in place (see
        # _swap_to_shape) via addShape/removeShape, confirmed to work
        # on a live node, so unlike BulletCharacterControllerNode this
        # never needs to be destroyed/recreated and self.velocity is
        # simply never disturbed by crouching.
        self.node = BulletGhostNode("player")
        self.node.addShape(self._standing_shape)
        self.node.setIntoCollideMask(CollisionGroup.PLAYER)
        self.node_path = physics_world._root.attachNewNode(self.node)
        self.node_path.setPos(to_physics_pos(position))
        physics_world.world.attachGhost(self.node)

        # "Can I stand up" probe: a flat-topped/bottomed box covering
        # only the HEADROOM slice standing would newly occupy (crouch-
        # top to stand-top), not the full standing hull re-tested from
        # the floor up. Re-testing the full hull from the floor up
        # would always register an overlap with the floor itself,
        # permanently refusing to stand up anywhere. A probe that only
        # spans the newly-needed headroom sidesteps the floor entirely
        # by construction, no epsilon-tuning required.
        headroom = self.height - self._crouch_height
        self._stand_check_shape = BulletBoxShape(to_physics_extent((self.radius, headroom / 2.0, self.radius)))
        self._stand_check_ghost = BulletGhostNode("stand_check")
        self._stand_check_ghost.addShape(self._stand_check_shape)
        # Same dedicated PLAYER-only identity as self.node (see the
        # comment by _sweep_mask above) - this ghost sits at the
        # NodePath default position (physics origin) until the first
        # crouch ever repositions it, and with a normal ALL into-mask
        # it would silently block the player's own ground/movement
        # sweeps the moment they passed near that spot (confirmed:
        # this is exactly what stopped free-fall early in testing).
        # Bullet's own group/mask filtering only requires ONE bit to
        # match in each direction, so narrowing this to PLAYER doesn't
        # stop it from detecting real ceilings/obstacles (which keep
        # their normal ALL into-mask on the other side of the pairing,
        # and their default from-mask already covers ALL too) - it
        # only stops OUR OWN outgoing sweeps (whose mask explicitly
        # excludes PLAYER) from matching it.
        self._stand_check_ghost.setIntoCollideMask(CollisionGroup.PLAYER)
        self._stand_check_ghost_np = physics_world._root.attachNewNode(self._stand_check_ghost)
        physics_world.world.attachGhost(self._stand_check_ghost)

        # The single source of truth for all motion - horizontal AND
        # vertical, render-space (Source's mv->m_vecVelocity). Built
        # up/decayed tick over tick, never reset to a target value
        # directly, and never disturbed by crouching (see above).
        self.velocity = glm.vec3(0.0)
        self._move_direction = glm.vec3(0.0)
        self._sprinting = False
        self._jump_requested = False
        self._crouch_input = False
        self._is_crouched = False
        self._crouch_amount = 0.0  # 0 = standing, 1 = fully crouched (eased, see get_eye_offset)
        self._grounded = False
        self._ground_normal = glm.vec3(0.0, 1.0, 0.0)
        self._ground_material = None

        # Footstep cadence bookkeeping - see _update_footsteps and
        # pop_footstep(). Mirrors Source's m_flStepSoundTime: counts
        # down to 0, a step can only fire once it reaches 0, then it's
        # reset to the next interval.
        self._step_sound_timer = 0.0
        self._pending_footstep = None

        # Last completed substep's (smoothed - see _on_pre_substep)
        # render-space position, for get_position() to interpolate
        # from - see PhysicsWorld.get_interpolation_alpha()'s
        # docstring.
        self._prev_position = to_render_pos(self.node_path.getPos())
        self._smoothed_position = glm.vec3(self._prev_position)

        physics_world.add_pre_substep_callback(self._on_pre_substep)

    # -----------------------------------------------------------------
    # Low-level collision helpers (Source's trace-* equivalents)
    # -----------------------------------------------------------------

    def _sweep(self, shape, from_render_pos, to_render_pos_):
        """Sweeps shape from from_render_pos to to_render_pos_ (both
        render-space glm.vec3) against the world, returning Panda3D's
        BulletClosestHitSweepResult directly (see callers for what
        they read off it). Confirmed empirically that a sweep never
        reports a hit against our OWN ghost node, so no self-exclusion
        handling is needed here."""
        from_ts = TransformState.makePos(to_physics_pos(from_render_pos))
        to_ts = TransformState.makePos(to_physics_pos(to_render_pos_))
        return self._physics_world.world.sweepTestClosest(shape, from_ts, to_ts, self._sweep_mask, 0.0)

    def _categorize_position(self):
        """Source's CGameMovement::CategorizePosition: a short, fixed-
        distance downward trace (independent of velocity) to determine
        ground contact and the ground normal, run BEFORE movement each
        tick using last tick's settled position.

        Skips the trace entirely (unconditionally airborne) whenever
        velocity.y is positive - confirmed as a real, reproducible bug
        without this: a single tick's rise (jump_speed * dt) can be
        smaller than _GROUND_TRACE_DISTANCE, so the very next tick's
        trace re-detects the floor and sets grounded back to True
        immediately after a jump. Since the landing-velocity-reset in
        _on_pre_substep only clears NEGATIVE velocity, that leaves the
        hull re-grounded while still nominally carrying jump_speed -
        and because _step_slide_move's grounded branch re-settles the
        hull onto the floor every tick regardless of velocity's sign,
        the jump silently goes nowhere: velocity is genuinely
        jump_speed the whole time, but the hull never actually leaves
        the ground. Rising can only ever mean "airborne, most likely
        mid-jump" in this system in the first place - grounded movement
        always holds velocity.y at exactly 0 or lets gravity pull it
        negative, nothing here ever produces an ambiguous small
        positive value the way a real trace-based engine occasionally
        does from ramps/bumps - so there's no real ambiguity being
        papered over by skipping the trace here. This mirrors Source's
        own CategorizePosition, which skips ground detection outright
        above NON_JUMP_VELOCITY for exactly this reason."""
        if self.velocity.y > 0.0:
            self._grounded = False
            self._ground_normal = glm.vec3(0.0, 1.0, 0.0)
            self._ground_material = None
            return

        pos = to_render_pos(self.node_path.getPos())
        probe_to = glm.vec3(pos.x, pos.y - _GROUND_TRACE_DISTANCE, pos.z)
        result = self._sweep(self._current_shape, pos, probe_to)
        if result.hasHit():
            normal = to_render_vec(result.getHitNormal())
            if normal.y >= self._max_slope_cos:
                self._grounded = True
                self._ground_normal = normal
                # Read back whatever material the hit body was tagged
                # with (see PhysicsWorld._add_body's material param) so
                # footstep sounds can vary per surface - None (untagged,
                # or no hit body for some reason) falls back to
                # footstep_materials.DEFAULT_FOOTSTEP_MATERIAL.
                hit_node = result.getNode()
                self._ground_material = (
                    hit_node.getPythonTag("physical_material")
                    if hit_node is not None and hit_node.hasPythonTag("physical_material")
                    else None
                )
                return
        self._grounded = False
        self._ground_normal = glm.vec3(0.0, 1.0, 0.0)
        self._ground_material = None

    @staticmethod
    def _clip_velocity(vel, normal, overbounce=1.0):
        """Source's CGameMovement::ClipVelocity - projects vel onto the
        plane defined by normal, removing the component driving INTO
        the surface (a "slide along the wall" reflection, not a
        bounce)."""
        backoff = glm.dot(vel, normal) * overbounce
        return vel - normal * backoff

    def _try_move(self, shape, start_pos, start_vel, dt):
        """Source's CGameMovement::TryPlayerMove: sweeps shape along
        vel for up to _MAX_BUMPS planes, clipping velocity against
        whatever it hits each time (sliding along walls), and sliding
        along the CREASE where two hit planes meet this call rather
        than getting stuck if the clipped velocity would drive back
        into an earlier plane. Pure function - does not touch self.*,
        just returns the resulting (pos, vel) so callers (see
        _step_slide_move) can try more than one candidate move and
        keep whichever is better."""
        pos = glm.vec3(start_pos)
        vel = glm.vec3(start_vel)
        time_left = dt
        hit_normals = []

        for _ in range(_MAX_BUMPS):
            if time_left <= 1e-9 or glm.length(vel) < 1e-6:
                break

            target = pos + vel * time_left
            result = self._sweep(shape, pos, target)
            if not result.hasHit():
                pos = target
                break

            fraction = result.getHitFraction()
            pos = pos + vel * time_left * fraction
            time_left *= (1.0 - fraction)

            normal = to_render_vec(result.getHitNormal())
            pos = pos + normal * _SURFACE_PUSHBACK
            hit_normals.append(normal)

            new_vel = self._clip_velocity(vel, normal)
            for other in hit_normals[:-1]:
                if glm.dot(new_vel, other) < 0.0:
                    # The plane we just clipped against would send us
                    # back into an EARLIER plane this call already hit -
                    # slide along the crease (the line where the two
                    # planes meet) instead of stalling here, same as
                    # Source's multi-plane handling in TryPlayerMove.
                    crease = glm.cross(normal, other)
                    crease_len_sq = glm.dot(crease, crease)
                    if crease_len_sq > 1e-9:
                        new_vel = crease * (glm.dot(vel, crease) / crease_len_sq)
                    else:
                        new_vel = glm.vec3(0.0)
                    break
            vel = new_vel

        return pos, vel

    def _step_slide_move(self, dt):
        """Source's CGameMovement::StepMove: while grounded, tries the
        move both flat (_try_move directly) and "stepped" (raise by up
        to step_height, run the same horizontal move from there, then
        settle back down onto the ground - handles both stepping UP
        onto a small obstacle and conforming DOWN a slope or small
        ledge within step_height in one mechanism), and keeps whichever
        covers more horizontal ground - exactly Source's dual-attempt
        approach, which is what makes small steps/slopes invisible to
        the player while a genuine wall still stops them. Airborne,
        there's no stepping at all (matches Source - AirMove never
        steps), just the flat attempt."""
        start_pos = to_render_pos(self.node_path.getPos())
        start_vel = glm.vec3(self.velocity)

        flat_pos, flat_vel = self._try_move(self._current_shape, start_pos, start_vel, dt)

        if self._grounded:
            up_target = glm.vec3(start_pos.x, start_pos.y + self.step_height, start_pos.z)
            up_result = self._sweep(self._current_shape, start_pos, up_target)
            # Only a genuine ceiling overhead (a surface actually facing
            # DOWN into the sweep, normal.y meaningfully negative) should
            # clamp the step height - a normal near/above 0 means the
            # sweep grazed a wall the hull is already pressed against
            # sideways (e.g. a stair riser the flat move above just got
            # blocked by, leaving the hull touching it within Bullet's
            # collision margin). That's not something purely vertical
            # motion actually runs into, but Bullet's sweep test can
            # still report a spurious near-zero-fraction hit against it
            # from an already-touching start (the same class of phantom
            # contact __init__'s _sweep_mask comment describes for the
            # player's own ghost) - treating every such hit as a real
            # ceiling collapsed raised_pos back to start_pos every tick,
            # permanently refusing to step up anything the player was
            # already flush against, stairs included.
            if up_result.hasHit() and to_render_vec(up_result.getHitNormal()).y < -0.1:
                raised_pos = start_pos + glm.vec3(0.0, self.step_height, 0.0) * max(up_result.getHitFraction() - 0.01, 0.0)
            else:
                raised_pos = up_target

            step_pos, step_vel = self._try_move(self._current_shape, raised_pos, start_vel, dt)

            # A fixed settle_distance (just step_height + a small pad)
            # is enough to reach back down to a walkable ramp/stairs
            # surface at ordinary walking speed, but not necessarily at
            # sprint: the horizontal distance _try_move just covered
            # (raised_pos -> step_pos) needs a proportionally bigger
            # vertical reach to re-find a STEEP walkable surface the
            # faster that horizontal distance is, since a steep ramp
            # drops height fast per unit of forward travel. Undershoot
            # this and the down-sweep below finds nothing on a perfectly
            # fine descending ramp/staircase purely because the player
            # is moving fast enough that a single tick's forward stride
            # outruns the fixed reach - confirmed as exactly why running
            # down stairs/ramps (but not walking down them) fell off
            # instead of following the slope down. Scaling by the
            # steepest slope this controller still considers walkable
            # (_max_slope_tan) covers any legitimate floor, however
            # steep, regardless of how far a single tick's stride is.
            horiz_dx = step_pos.x - raised_pos.x
            horiz_dz = step_pos.z - raised_pos.z
            horiz_dist = math.sqrt(horiz_dx * horiz_dx + horiz_dz * horiz_dz)
            settle_distance = self.step_height + horiz_dist * self._max_slope_tan + 0.05
            down_target = glm.vec3(step_pos.x, step_pos.y - settle_distance, step_pos.z)
            down_result = self._sweep(self._current_shape, step_pos, down_target)
            # A step is only valid if it actually lands back on walkable
            # ground - without this, sliding past a wall above
            # step_height's reach (grazing a corner, or a wall shorter
            # than the player but taller than step_height) leaves
            # settled_pos floating at the raised height with no floor
            # under it at all, which still "covers more ground" than the
            # correctly-blocked flat attempt and gets picked - reading as
            # the player boosting up onto/over solid walls.
            if down_result.hasHit() and to_render_vec(down_result.getHitNormal()).y >= self._max_slope_cos:
                settled_pos = step_pos + glm.vec3(0.0, -settle_distance, 0.0) * down_result.getHitFraction()

                flat_dist_sq = (flat_pos.x - start_pos.x) ** 2 + (flat_pos.z - start_pos.z) ** 2
                step_dist_sq = (settled_pos.x - start_pos.x) ** 2 + (settled_pos.z - start_pos.z) ** 2

                # flat_pos comes from a pure-velocity sweep with no
                # vertical component (vel.y is 0 while grounded - see
                # _on_pre_substep's landed/gravity handling just before
                # this runs). Whenever NEITHER attempt is obstructed -
                # the overwhelmingly common case while walking - flat and
                # stepped cover essentially the SAME horizontal distance
                # (the horizontal half of _try_move doesn't care whether
                # it started raised or not), so step_dist_sq is a tie
                # with flat_dist_sq, not strictly greater. A strict ">"
                # comparison breaks that tie in flat's favor - which,
                # on any downslope steeper than one tick's horizontal
                # travel, means picking the candidate that stayed at the
                # OLD height and sailed clean through the open air past
                # the edge of the floor, unobstructed (nothing there to
                # clip _try_move's sweep against), instead of the one
                # that actually settled onto the real ground below. That
                # then only gets caught by luck (_categorize_position's
                # short ground probe still happening to graze something)
                # for as many ticks as the gap stays within its reach,
                # widening a little further each time flat wins the tie
                # again, until it finally misses outright and the player
                # drops like a stone - exactly the "doesn't stay on the
                # stairs going down" symptom. Since a settled_pos here
                # has already been confirmed to land on real walkable
                # ground (the hasHit + slope check above), it should win
                # every tie - flat only wins when it's UNAMBIGUOUSLY
                # better (e.g. stepping got blocked by something flat
                # could slide past).
                if flat_dist_sq > step_dist_sq + 1e-9:
                    final_pos, final_vel = flat_pos, flat_vel
                else:
                    final_pos, final_vel = settled_pos, step_vel
            else:
                final_pos, final_vel = flat_pos, flat_vel
        else:
            final_pos, final_vel = flat_pos, flat_vel

        self.node_path.setPos(to_physics_pos(final_pos))
        self.velocity = final_vel

    # -----------------------------------------------------------------
    # Public control surface (unchanged from the previous version)
    # -----------------------------------------------------------------

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
        """crouching=True shrinks the collision hull immediately and
        starts easing the camera toward crouch_eye_height.
        crouching=False requests standing back up, but is refused (and
        silently retried every tick) while something overhead blocks
        it - see the module docstring. Call every frame with the crouch
        key's current held state, not just on press."""
        self._crouch_input = bool(crouching)

    def jump(self):
        """Queues a jump for the next physics tick this hull is on the
        ground - safe to call every frame while the jump key is held
        (matches Source: holding jump auto-hops on landing rather than
        requiring a fresh press each time)."""
        self._jump_requested = True

    def is_on_ground(self):
        return self._grounded

    def is_crouched(self):
        return self._is_crouched

    # -----------------------------------------------------------------
    # Ground/air movement math (unchanged from the previous version -
    # a direct port of Source's Friction/Accelerate/AirAccelerate)
    # -----------------------------------------------------------------

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
        horiz = glm.vec3(self.velocity.x, 0.0, self.velocity.z)
        speed = glm.length(horiz)
        if speed < 1e-6:
            self.velocity.x = 0.0
            self.velocity.z = 0.0
            return
        control = max(speed, self.stop_speed)
        drop = control * self.friction * dt
        new_speed = max(speed - drop, 0.0)
        scale = new_speed / speed
        self.velocity.x *= scale
        self.velocity.z *= scale

    def _accelerate(self, wishdir, wishspeed, accel, dt):
        horiz = glm.vec3(self.velocity.x, 0.0, self.velocity.z)
        current_speed = glm.dot(horiz, wishdir)
        add_speed = wishspeed - current_speed
        if add_speed <= 0.0:
            return
        accel_speed = min(accel * dt * wishspeed, add_speed)
        self.velocity.x += accel_speed * wishdir.x
        self.velocity.z += accel_speed * wishdir.z

    def _air_accelerate(self, wishdir, wishspeed, accel, dt):
        # The add-speed check uses the CAPPED wishspeed (limits how much
        # a single tick's air control can redirect existing momentum),
        # but accel_speed's magnitude still comes from the UNCAPPED
        # wishspeed - see the module docstring for why this asymmetry
        # is intentional, not a typo.
        capped_wishspeed = min(wishspeed, _AIR_SPEED_CAP)
        horiz = glm.vec3(self.velocity.x, 0.0, self.velocity.z)
        current_speed = glm.dot(horiz, wishdir)
        add_speed = capped_wishspeed - current_speed
        if add_speed <= 0.0:
            return
        accel_speed = min(accel * dt * wishspeed, add_speed)
        self.velocity.x += accel_speed * wishdir.x
        self.velocity.z += accel_speed * wishdir.z

    # -----------------------------------------------------------------
    # Crouching (shape swap is now trivial - see the module docstring)
    # -----------------------------------------------------------------

    def _can_stand(self):
        # Ignore self.node itself - the ghost is positioned right where
        # the player's own hull roughly is, so it would otherwise
        # always "detect" the player as blocking its own stand-up.
        #
        # Uses BulletWorld.contactTest() - an immediate, on-demand
        # narrow-phase query against the ghost's CURRENT transform -
        # rather than BulletGhostNode.getOverlappingNodes(), which reads
        # Bullet's cached broadphase pair list and can stay stale
        # ("overlapping") for several ticks after the shapes have
        # actually separated (confirmed empirically during development).
        result = self._physics_world.world.contactTest(self._stand_check_ghost)
        for i in range(result.getNumContacts()):
            contact = result.getContact(i)
            other = contact.getNode1() if contact.getNode0() == self._stand_check_ghost else contact.getNode0()
            if other != self.node:
                return False
        return True

    def _position_stand_check_ghost(self):
        # Centers the headroom-slice probe (see __init__) directly
        # above the player's current top, spanning up to where the
        # standing hull's top would be - never reaching down toward the
        # floor at all, by construction. current_top is the current
        # (crouched) hull's TOP, not its center - the probe's center is
        # current_top plus half the headroom slice above it.
        current_pos = self.node_path.getPos()
        headroom = self.height - self._current_height
        current_top = current_pos.z + self._current_height / 2.0
        candidate_pos = Point3(current_pos.x, current_pos.y, current_top + headroom / 2.0)
        self._stand_check_ghost_np.setPos(candidate_pos)

    def _swap_to_shape(self, shape, total_height, target_crouch_amount):
        # Swaps the shape on the SAME persistent ghost node
        # (addShape/removeShape, confirmed to work on a live node)
        # rather than replacing the node itself, so self.velocity is
        # never touched by this at all - unlike
        # BulletCharacterControllerNode, there's no hidden internal
        # state here that recreating a node would lose.
        old_pos = self.node_path.getPos()

        if self._grounded:
            # Keeps the hull's BOTTOM (feet) fixed across the swap by
            # shifting the center by half the total-height difference,
            # rather than leaving the center in place - crouching
            # should lower the head, not lift the feet off (or sink
            # them into) the ground.
            new_pos = Point3(old_pos.x, old_pos.y, old_pos.z + (total_height - self._current_height) / 2.0)
        else:
            # Airborne, there's no floor to plant feet against, so
            # "feet stay fixed" is an arbitrary reference with nothing
            # to do with what the player is actually looking at -
            # anchoring on it mid-air just pops the CAMERA around for
            # no physical reason. Anchor on the EYE position instead:
            # shift the hull so the eye position ends up EXACTLY where
            # it was before the swap - the view doesn't move at all,
            # only the hitbox repositions around it. Uses
            # target_crouch_amount rather than self._crouch_amount
            # because _update_crouch's airborne eye-offset snap hasn't
            # run yet this tick - this computes what the eye offset
            # WILL be right after it does, so the two stay in sync.
            old_eye_offset = self._eye_offset_for(self._current_height, self._crouch_amount)
            new_eye_offset = self._eye_offset_for(total_height, target_crouch_amount)
            new_pos = Point3(old_pos.x, old_pos.y, old_pos.z + (old_eye_offset - new_eye_offset))

        self.node.removeShape(self._current_shape)
        self.node.addShape(shape)
        self.node_path.setPos(new_pos)

        self._current_shape = shape
        self._current_height = total_height

        # This recenter is an intentional, instant repositioning - not
        # per-tick sweep-test noise, so it must bypass get_position()'s
        # vertical smoothing filter rather than being caught by it.
        # Without this reset, the filter (see _on_pre_substep) treats
        # the jump in raw Y as a new target to ease toward over its
        # ~150ms window, which reads as the camera slowly drifting down
        # over several frames instead of popping to the new height
        # immediately - resetting both tracked points to the new
        # position now makes this swap pass through instantly.
        new_render_pos = to_render_pos(new_pos)
        self._prev_position = new_render_pos
        self._smoothed_position = glm.vec3(new_render_pos)

    def _update_crouch(self, dt):
        if self._crouch_input:
            if not self._is_crouched:
                self._swap_to_shape(self._crouch_shape, self._crouch_height, target_crouch_amount=1.0)
                self._is_crouched = True
        elif self._is_crouched:
            # Position the probe at today's position, THEN check it -
            # contactTest() (see _can_stand) queries fresh, so there's
            # no need to check against yesterday's position anymore.
            self._position_stand_check_ghost()
            if self._can_stand():
                self._swap_to_shape(self._standing_shape, self.height, target_crouch_amount=0.0)
                self._is_crouched = False

        target = 1.0 if self._is_crouched else 0.0
        if self._grounded:
            # Ease over crouch_transition_time, same as always.
            rate = dt / max(self.crouch_transition_time, 1e-6)
            if self._crouch_amount < target:
                self._crouch_amount = min(self._crouch_amount + rate, target)
            else:
                self._crouch_amount = max(self._crouch_amount - rate, target)
        else:
            # Airborne: snap the EYE offset straight to the target
            # instead of easing it. The hull itself already swaps
            # instantly regardless of ground state (matches Source's
            # duck-jump - crouching mid-air changes your hitbox right
            # away for tech like fitting through gaps), but easing the
            # CAMERA over crouch_transition_time on top of an
            # independently-falling capsule reads as the view doing its
            # own thing while also plummeting - confusing rather than
            # smooth. On the ground there's no such competing motion,
            # so the eased transition there still reads fine.
            self._crouch_amount = target

    # -----------------------------------------------------------------
    # Main per-tick update (Source's CGameMovement::PlayerMove, in the
    # same order: categorize ground, handle jump/gravity, apply
    # friction+accel, then resolve the actual move)
    # -----------------------------------------------------------------

    def _on_pre_substep(self, dt):
        raw = to_render_pos(self.node_path.getPos())

        # Only while grounded - airborne there's no ground-contact
        # sweep to be noisy about (free-fall is already perfectly
        # smooth with no filtering at all), and this filter's lag is
        # proportional to velocity * tau, which is negligible at
        # walking speed but becomes a real, visible trailing offset at
        # jump/fall speed - filtering that away would just make the
        # camera noticeably lag behind where the hull actually is for
        # the whole arc of every jump.
        if self._grounded:
            alpha_smooth = 1.0 - math.exp(-dt / _EYE_SMOOTH_TAU)
        else:
            alpha_smooth = 1.0
        smoothed_y = self._smoothed_position.y + (raw.y - self._smoothed_position.y) * alpha_smooth

        self._prev_position = self._smoothed_position
        self._smoothed_position = glm.vec3(raw.x, smoothed_y, raw.z)

        self._update_crouch(dt)

        # Ground state uses last tick's settled position, BEFORE this
        # tick's movement - matches Source's CategorizePosition, which
        # runs at the start of PlayerMove using the position left over
        # from the previous frame.
        self._categorize_position()

        if self._grounded and self.velocity.y < 0.0:
            # Landed (or resting) - clear any residual downward
            # velocity so it doesn't accumulate call after call; actual
            # ground-conforming (slopes, small ledges) is handled by
            # _step_slide_move's position-based settle, not by feeding
            # it a persistent downward speed.
            self.velocity.y = 0.0

        if self._jump_requested and self._grounded:
            self.velocity.y = self.jump_speed
            self._grounded = False
        self._jump_requested = False

        wishdir, wishspeed = self._wish()
        if self._grounded:
            self._apply_friction(dt)
            self._accelerate(wishdir, wishspeed, self.ground_accel, dt)
        else:
            self.velocity.y -= self.gravity * dt
            self._air_accelerate(wishdir, wishspeed, self.air_accel, dt)

        self._step_slide_move(dt)

        self._update_footsteps(dt)

    def _update_footsteps(self, dt):
        """Direct port of Source's CBasePlayer::UpdateStepSound (see the
        module-level constants above for exactly where these numbers
        come from). The timer counts down every tick regardless of
        movement state (matching UpdateStepSound running unconditionally
        at the top before any of its early-outs); a step can only fire
        once it reaches 0, is gated by a separate minimum-speed check
        (moving_fast_enough in the original), and resets the timer to
        the next interval - it does NOT reset just because the player
        stopped or went airborne, matching Source exactly (there's no
        such reset in UpdateStepSound either)."""
        if self._step_sound_timer > 0.0:
            self._step_sound_timer = max(self._step_sound_timer - dt, 0.0)

        if not self._grounded:
            return

        horiz_speed = math.hypot(self.velocity.x, self.velocity.z)
        min_speed = _STEP_MIN_SPEED_CROUCH if self._is_crouched else _STEP_MIN_SPEED_STAND
        run_speed = _STEP_RUN_SPEED_CROUCH if self._is_crouched else _STEP_RUN_SPEED_STAND

        if self._step_sound_timer > 0.0 or horiz_speed < min_speed:
            return

        walking = horiz_speed < run_speed
        interval = _STEP_INTERVAL_WALK if walking else _STEP_INTERVAL_RUN
        volume = get_footstep_volume(self._ground_material, walking)
        if self._is_crouched:
            interval += _STEP_INTERVAL_CROUCH_EXTRA
            volume *= _STEP_VOLUME_DUCK_MULT

        self._step_sound_timer = interval
        self._pending_footstep = (self._ground_material, volume)

    def pop_footstep(self):
        """Returns (material, volume) for a footstep that fired since
        the last call, or None if none did - call once per frame (see
        app.py) and forward to Scene.play_footstep_sound. Consumes the
        event (returns None until the next one fires) so the same
        footstep is never played twice even if this is polled more than
        once before the next physics tick runs."""
        event = self._pending_footstep
        self._pending_footstep = None
        return event

    def get_position(self):
        """Render-space glm.vec3 - the hull's current center position
        (its height above this changes when crouched - see
        get_eye_offset for a camera offset that accounts for that).

        Interpolated between the last two completed physics substeps
        (see PhysicsWorld.get_interpolation_alpha()) rather than
        returning the raw physics position directly - physics runs at a
        fixed 120Hz while rendering runs at whatever the display does,
        so without this a render frame that lands between two substeps
        would see the same position repeat and then jump, reading as
        jitter (worst at low speed, where each tick's actual movement
        is small next to that jump). The Y axis is additionally
        low-pass filtered while grounded (see _on_pre_substep) to
        smooth out any residual per-tick sweep-test noise while
        walking; X/Z are exact, unfiltered physics positions."""
        alpha = self._physics_world.get_interpolation_alpha()
        return glm.mix(self._prev_position, self._smoothed_position, alpha)

    def _eye_offset_for(self, current_height, crouch_amount):
        """Render-space Y offset from a hull center to where the camera
        should sit, for an arbitrary (current_height, crouch_amount)
        pair rather than necessarily the controller's own current
        state - factored out so _swap_to_shape can compute this for
        the state BEFORE and the state AFTER a hull swap (see its
        eye-anchored branch) without duplicating the formula."""
        standing_feet_to_eye = self.height / 2.0 + self.eye_height
        crouch_feet_to_eye = self._crouch_height / 2.0 + self.crouch_eye_height
        feet_to_eye = standing_feet_to_eye + (crouch_feet_to_eye - standing_feet_to_eye) * crouch_amount

        feet_offset_from_center = -current_height / 2.0
        return feet_offset_from_center + feet_to_eye

    def get_eye_offset(self):
        """Render-space Y offset from get_position() a first-person
        camera should sit at. When grounded, this eases smoothly across
        crouch_transition_time even though the collision hull itself
        resizes in one instant, because _swap_to_shape keeps the FEET
        position invariant across that resize - see the module
        docstring's CROUCHING section. Airborne, _swap_to_shape instead
        keeps the EYE position itself invariant (see its comment for
        why "feet planted" doesn't mean anything with no floor to plant
        against), so this doesn't need to smooth anything there - the
        camera was never moved by the swap in the first place."""
        return self._eye_offset_for(self._current_height, self._crouch_amount)

    def destroy(self):
        self._physics_world.remove_pre_substep_callback(self._on_pre_substep)
        self._physics_world.world.removeGhost(self._stand_check_ghost)
        self._stand_check_ghost_np.removeNode()
        self._physics_world.world.removeGhost(self.node)
        self.node_path.removeNode()
