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

# Squared-distance tolerance (meters^2) for _step_slide_move's own flat-
# vs-stepped tie-break - see its own comment for the full reasoning.
# Needs to be comfortably bigger than _SURFACE_PUSHBACK's own scale: both
# candidates apply that same ~1mm pushback independently (via their own
# separate _try_move call), so two outcomes that are "equally blocked" -
# pushed straight into a flat wall, say, with no stairs/slope involved at
# all - routinely differ from each other by roughly that much even though
# nothing is genuinely different between them. A too-tight tolerance here
# (1e-9 - three orders of magnitude below the pushback's own ~1e-6
# squared-distance scale) let final_pos flip between flat_pos and
# settled_pos essentially at random, every tick, purely from which
# candidate's own independent sweep noise happened to travel fractionally
# further that tick - visibly, the whole character's root position
# vibrated by that sub-centimeter amount continuously while pinned
# against any wall, reading as the legs/stride stuttering even though the
# animation blend weights themselves (see PlayerModel.update()'s own
# is_sprinting/move_direction docstrings) were completely stable. 1e-4
# (1cm, squared) sits comfortably above that pushback-scale noise floor
# while staying far below any genuine stair/slope horizontal travel this
# tie-break needs to keep detecting correctly.
_STEP_TIE_EPSILON_SQ = 1e-4

# How much narrower (per side, in X and Z) the probe used for _categorize_position's ground
# check and _step_slide_move's up/down sweeps is than the player's real hull. Those three
# sweeps are all purely VERTICAL and only ever care about what's directly below/above the
# hull's own footprint - but run with the FULL-width hull, right at the moment the player is
# pressed flush against a low obstacle (exactly the moment stepping up onto it matters most),
# they catch that obstacle's own front-top EDGE instead of the floor or the obstacle's flat
# top: the hull's leading face already touches the obstacle before its CENTER has crossed
# over it, so a straight-down sweep from there grazes the corner where the obstacle's front
# and top faces meet (a blended, non-floor-like normal, confirmed directly: a sweep from a
# hull still short of a 0.1m-tall box reported a hit with normal.y around 0.24-0.30, well
# under max_slope_cos) and gets rejected as "not walkable" - ground detection then flickers
# grounded/not-grounded tick to tick (disabling stepping on the "not grounded" ticks and,
# worse, fully clipping velocity to zero via the flat-only path each time), and even once
# genuinely grounded, the step-up's own down-settle sweep hits the same corner and rejects
# the climb outright - reproduced directly: a flat-topped obstacle well under step_height
# (0.1-0.49m against a 0.5m step_height) simply never got climbed, confirmed via a dedicated
# physics-only test, no matter how long forward input was held. Narrowing the probe by more
# than this inset clears the obstacle's edge entirely once the hull's CENTER is still short
# of it (no longer spanning both the open floor and the obstacle at once), letting the sweep
# find the real ground cleanly; 0.15m verified (same test) to climb every height from 0.1m up
# to the 0.5m step_height cleanly while still correctly refusing anything taller.
_VERTICAL_PROBE_INSET = 0.15

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

# Minimum downward speed (m/s) at the moment ground is found for a
# landing to count as a genuine fall worth a sound - see
# _on_pre_substep's _fall_speed_on_touch and _update_footsteps'
# just_landed. Comfortably above the sub-0.1 m/s noise a single tick (or
# few) of ground-sweep flicker produces (e.g. pressed against a wall),
# comfortably below any real jump/fall's landing speed.
_MIN_LANDING_FALL_SPEED = 1.0

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
        self._sweep_mask = self.collision_mask & ~(CollisionGroup.PLAYER | CollisionGroup.GIB)

        self._standing_shape = BulletBoxShape(to_physics_extent((self.radius, self.height / 2.0, self.radius)))
        self._crouch_shape = BulletBoxShape(to_physics_extent((self.radius, self._crouch_height / 2.0, self.radius)))
        self._current_shape = self._standing_shape
        self._current_height = self.height

        # A narrower stand-in for self._current_shape, used ONLY for _categorize_position's
        # ground probe and _step_slide_move's up/down vertical sweeps (never for the flat/
        # stepped horizontal moves, which need the real width to collide correctly) - see
        # _VERTICAL_PROBE_INSET's own docstring for why the full-width hull can't be used there.
        probe_radius = max(0.05, self.radius - _VERTICAL_PROBE_INSET)
        self._standing_vertical_probe_shape = BulletBoxShape(
            to_physics_extent((probe_radius, self.height / 2.0, probe_radius)))
        self._crouch_vertical_probe_shape = BulletBoxShape(
            to_physics_extent((probe_radius, self._crouch_height / 2.0, probe_radius)))
        self._current_vertical_probe_shape = self._standing_vertical_probe_shape

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
        # True from the instant a REAL jump executes (see _on_pre_substep) until its
        # rise is over (velocity.y drops back to <=0 - see _categorize_position's own
        # clear condition) - what _categorize_position actually skips the ground probe
        # for, instead of keying off raw velocity.y > 0 (see its own docstring for why
        # that was wrong: a slope-clip artifact can ALSO produce a transient positive
        # Y with no jump involved at all).
        self._jumping = False
        self._crouch_input = False
        self._is_crouched = False
        self._crouch_amount = 0.0  # 0 = standing, 1 = fully crouched (eased, see get_eye_offset)
        self._grounded = False
        self._ground_normal = glm.vec3(0.0, 1.0, 0.0)
        self._ground_material = None
        # _step_slide_move's flat-vs-stepped tie-break (see its own comment): which candidate
        # won LAST tick, reused as the default whenever this tick is a near-tie again, instead
        # of deciding fresh every time - see that comment for why a persistent near-tie needs
        # this to avoid a visible stutter.
        self._last_step_was_flat = False

        # Footstep cadence bookkeeping - see _update_footsteps and
        # pop_footstep(). Mirrors Source's m_flStepSoundTime: counts
        # down to 0, a step can only fire once it reaches 0, then it's
        # reset to the next interval.
        self._step_sound_timer = 0.0
        self._pending_footstep = None
        # Grounded state as of the END of the previous tick's footstep
        # update - used only to detect a landing edge (see
        # _update_footsteps' just_landed).
        self._was_grounded = False
        # Whether categorize_position found ground THIS tick, captured
        # before the jump-request handling in _on_pre_substep can clear
        # self._grounded again - see that capture site and
        # _update_footsteps' just_landed for why this exists separately
        # from self._grounded.
        self._touched_ground = False

        # Set the instant a queued jump actually executes (see
        # _on_pre_substep) - mirrors _pending_footstep/pop_footstep()
        # exactly (a physics-tick-rate event latched until the next
        # render-frame poll consumes it via pop_jumped()), so a caller
        # driving player animation (see app.py/PlayerModel.update()'s own
        # just_jumped param) can react to the ACTUAL jump input the same
        # frame it happens, rather than only inferring "airborne" from
        # is_on_ground() - which, pressed against a wall, can flicker
        # False for a tick or two with no jump involved at all (see
        # PlayerModel's own _JUMP_CONFIRM_SECONDS docstring), so waiting
        # on that debounce to also gate a REAL jump's own animation made
        # a genuine jump feel noticeably delayed/laggy.
        self._pending_jump_event = False
        self._fall_speed_on_touch = 0.0

        # One-off diagnostic toggle (see toggle_debug_log/app.py's own debug keybind) -
        # prints every physics tick's velocity/grounded/jumping/ground_normal straight to
        # the console while True, so a landing/slide bug can be inspected tick-by-tick
        # instead of guessed at from how it LOOKS on screen. Off by default - not meant to
        # stay on during normal play.
        self._debug_log = False
        self._debug_tick = 0
        self._debug_trace_distance = 0.0
        self._debug_probe_hit = False
        self._debug_probe_normal_y = None
        self._debug_probe_fraction = None

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

    def _categorize_position(self, dt):
        """Source's CGameMovement::CategorizePosition: a short downward trace to determine
        ground contact and the ground normal, run BEFORE movement each tick using last
        tick's settled position.

        The trace distance is normally the fixed _GROUND_TRACE_DISTANCE, but is extended
        to cover THIS tick's expected fall distance (-velocity.y * dt) whenever that's
        bigger - see the "sliding after any jump" bug this fixes, below.

        Why that extension is needed: without it, a fall fast enough that one tick's
        travel exceeds the fixed trace distance reaches the floor BEFORE this probe ever
        sees it coming - the probe (run on LAST tick's position) reports not-grounded,
        so the fall continues through the normal airborne path, and the floor is instead
        discovered mid-tick by _step_slide_move's own collision sweep (_try_move), which
        hits it and runs it through the SAME generic _clip_velocity used for walls. For a
        perfectly flat floor (normal exactly (0,1,0)) that clip only zeroes the vertical
        component, harmlessly - but for ANY other walkable-but-not-perfectly-flat surface
        (a ramp, a stair tread, a slightly uneven floor tile - i.e. nearly everything in a
        real level), clipping a large downward velocity against a tilted normal leaves a
        genuine horizontal remainder along the surface's own tangent (exactly like a ball
        deflecting off a ramp - physically correct for an object, wrong for a player who
        should just land and stop). That remainder is real self.velocity, grounded is
        still False THAT tick so neither the dedicated "landed -> zero velocity.y" line
        below nor ground friction ever touches it, and it only starts bleeding off via
        friction from the NEXT tick onward - reading exactly as "landing puts me in a
        sliding state," on any surface, not just steep ones, since a typical landing fall
        speed is comfortably past the fixed trace distance's reach (confirmed: jump_speed
        itself, 6.0 m/s, already travels _GROUND_TRACE_DISTANCE's 0.0508m in under 0.0085s
        - under one tick at 120Hz - so even a dead-level jump's landing speed alone is
        right at this threshold, and landing anywhere even slightly below launch height
        clears it easily). Extending the trace to always reach at least this tick's own
        fall distance guarantees the dedicated landing path (the explicit vel.y=0 zeroing
        a few lines down in _on_pre_substep) is what actually catches every landing,
        instead of the generic wall-sliding clip math catching it first by accident.

        Skips the trace entirely (unconditionally airborne) while self._jumping is
        true - confirmed as a real, reproducible bug without this: a single tick's
        rise (jump_speed * dt) can be smaller than _GROUND_TRACE_DISTANCE, so the
        very next tick's trace re-detects the floor and sets grounded back to True
        immediately after a jump. Since the landing-velocity-reset in
        _on_pre_substep only clears NEGATIVE velocity, that leaves the hull
        re-grounded while still nominally carrying jump_speed - and because
        _step_slide_move's grounded branch re-settles the hull onto the floor every
        tick regardless of velocity's sign, the jump silently goes nowhere: velocity
        is genuinely jump_speed the whole time, but the hull never actually leaves
        the ground.

        This used to key off "self.velocity.y > 0.0" directly (treating ANY rise as
        "must be mid-jump, skip the probe), on the theory that grounded movement
        never produces an ambiguous positive Y value the way a real trace-based
        engine occasionally does from ramps/bumps. That theory is WRONG - confirmed
        directly: _try_move's own ClipVelocity, airborne, against a steep-but-
        walkable slope, DOES produce a genuine positive clipped.y whenever the
        incoming horizontal velocity has enough of an UPHILL component relative to
        the current fall speed (sliding "up and along" the slope's own tangent
        plane, exactly like a ball thrown at a ramp can deflect upward off it) -
        reliably reproduced by holding movement input pointed away from a slope's
        own downhill direction while falling onto it (fast enough fall, enough
        opposing horizontal speed). Keying off raw velocity.y there meant a player
        who landed that way got stuck: self._grounded would never go True again
        (every tick's clip re-produces a positive Y off the slope, re-triggering
        the skip), sliding down the incline in what LOOKS like ordinary falling
        (gravity still visibly pulling them down, the hull still in contact with
        the slope) but with the ground probe never actually running - so jump()
        never fires either (it requires self._grounded) - until the hull finally
        slides clear of the incline onto flat ground where the artifact can't
        recur. self._jumping (see the jump-request handling and its own clear
        condition further down) instead tracks a REAL jump explicitly, so a
        slope-clip artifact's transient positive Y can never masquerade as one -
        the probe still runs on every such tick, correctly re-grounds as soon as
        the hit normal comes back within max_slope_cos, same as any other landing."""
        if self._jumping and self.velocity.y <= 0.0:
            # The jump's rise is over (gravity's caught up - falling now, same
            # threshold the old velocity.y-based check used) - clear it so the
            # probe resumes running normally for the actual fall/landing, exactly
            # like a genuine jump always has.
            self._jumping = False
        if self._jumping:
            self._grounded = False
            self._ground_normal = glm.vec3(0.0, 1.0, 0.0)
            self._ground_material = None
            return

        pos = to_render_pos(self.node_path.getPos())
        # See this method's own docstring on why the trace must reach at least this
        # tick's own expected fall distance, not just the fixed baseline, whenever
        # falling faster than that baseline would cover.
        trace_distance = max(_GROUND_TRACE_DISTANCE, -self.velocity.y * dt)
        probe_to = glm.vec3(pos.x, pos.y - trace_distance, pos.z)
        # Narrower than the real hull (see _VERTICAL_PROBE_INSET) - the full-width hull, right
        # when the player is pressed flush against a low obstacle, catches that obstacle's own
        # front-top edge here instead of the real floor beneath, flickering grounded on and off.
        result = self._sweep(self._current_vertical_probe_shape, pos, probe_to)
        if not result.hasHit() and self._grounded:
            # The narrower probe (see _VERTICAL_PROBE_INSET) can genuinely miss a surface
            # the player is still resting on - confirmed directly via this file's own
            # debug_log (F6): on some slopes, _try_move's own collision response (which
            # uses the REAL, full-width hull - self._current_shape) kept finding and
            # resolving contact against the floor every single tick, while this narrower,
            # straight-down-only probe reported no hit at all for hundreds of consecutive
            # ticks in a row, each one stomping self._grounded back to False moments after
            # _step_slide_move had just set it True from that same real contact (see its
            # own discovered_ground comment) - a flip-flop every tick that reset grounded
            # to False before anything (friction, jump eligibility) downstream of THIS
            # call ever got to see it True. Retried with the actual hull width ONLY when
            # already grounded (never while first approaching/climbing something, where
            # the narrower probe's edge-avoidance - see its own docstring - still matters)
            # since that's the exact shape _try_move keeps successfully colliding with;
            # only a genuine, real departure from the floor should fail THIS too.
            result = self._sweep(self._current_shape, pos, probe_to)
        if self._debug_log:
            # Raw probe result, independent of whatever _categorize_position's own verdict
            # ends up being below - see the print in _on_pre_substep, which needs this to
            # tell "probe found nothing at all" apart from "probe found something but
            # rejected it as too steep".
            self._debug_trace_distance = trace_distance
            self._debug_probe_hit = result.hasHit()
            self._debug_probe_normal_y = to_render_vec(result.getHitNormal()).y if result.hasHit() else None
            self._debug_probe_fraction = result.getHitFraction() if result.hasHit() else None
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
        just returns the resulting (pos, vel, ground_normal) so callers (see
        _step_slide_move) can try more than one candidate move and keep
        whichever is better. ground_normal is None unless this call's own sweep
        directly touched a walkable floor/ramp (see the landing branch below) -
        _step_slide_move uses that to set self._grounded immediately, rather
        than only ever waiting on _categorize_position's separate, narrower,
        straight-down-only probe to rediscover the same contact a tick later -
        see _step_slide_move's own comment on why that matters."""
        pos = glm.vec3(start_pos)
        vel = glm.vec3(start_vel)
        time_left = dt
        hit_normals = []
        ground_normal = None

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

            if normal.y >= self._max_slope_cos:
                # Touched a walkable surface THIS sweep, however this call ends up
                # resolving velocity below - worth reporting back regardless of whether
                # vel.y happened to be negative at this exact bump (a sliding-but-not-
                # currently-falling contact, e.g. one already leveled off by an earlier
                # bump this same call, still counts as real ground contact).
                ground_normal = normal

            if normal.y >= self._max_slope_cos and vel.y < 0.0:
                # Falling straight into a walkable floor/ramp - a LANDING, not a wall to
                # slide along. Clipping the FULL incoming vector (vel, including its big
                # negative Y) against the hit plane - what the generic _clip_velocity
                # below does, correct for a wall - converts part of the fall speed into a
                # genuine NEW horizontal velocity along the slope's downhill tangent:
                # physically accurate for a bouncing/skidding object, but not how a
                # Source-style player should land (confirmed as the earlier cause of
                # "landing anywhere slides downhill, picking up speed with every landing
                # until the bottom").
                #
                # The fix isn't simply discarding vel.y outright and keeping the
                # horizontal part dead level, either - confirmed via this file's own
                # debug_log (F6) that doing exactly that breaks ground adherence on any
                # DESCENDING slope: moving dead level from a point sitting right on a
                # downward-curving ramp immediately carries the hull up and away from it,
                # since the surface drops out from under a level path. A few ticks later
                # gravity pulls it back down into the ramp again, gets leveled off again,
                # drifts away again - a repeating bounce that never gives
                # _categorize_position's own short ground probe (see its own docstring)
                # a chance to ever latch onto "grounded", confirmed directly in the log as
                # grounded staying False for the rest of the session after one jump. Since
                # friction only ever runs while grounded, that's a permanent, undecaying
                # slide - matching "still sliding, now ALONG the slope" exactly.
                #
                # Clipping just the HORIZONTAL part of vel (vel.y already discarded, so
                # there's nothing left to convert into unwanted NEW speed) against the
                # same normal re-projects it onto the slope's own tangent - i.e. it keeps
                # following the ramp's downward direction, hugging the surface, instead of
                # going dead level and drifting off it. For an exactly flat floor
                # (normal.y == 1) this is a no-op (dot(horiz_vel, (0,1,0)) is already 0),
                # so flat-ground landings are completely unaffected.
                horiz_vel = glm.vec3(vel.x, 0.0, vel.z)
                new_vel = self._clip_velocity(horiz_vel, normal)
            else:
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

        return pos, vel, ground_normal

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
        steps), just the flat attempt.

        Also the only place self._grounded is ever set TRUE outside of
        _categorize_position's own dedicated probe - see discovered_ground's own
        comment below for why that probe alone isn't reliable enough to be the sole
        path back into "grounded" after becoming airborne."""
        start_pos = to_render_pos(self.node_path.getPos())
        start_vel = glm.vec3(self.velocity)

        flat_pos, flat_vel, flat_ground = self._try_move(self._current_shape, start_pos, start_vel, dt)

        if self._grounded:
            up_target = glm.vec3(start_pos.x, start_pos.y + self.step_height, start_pos.z)
            # Narrower than the real hull (see _VERTICAL_PROBE_INSET), same reasoning as
            # _categorize_position's own ground probe.
            up_result = self._sweep(self._current_vertical_probe_shape, start_pos, up_target)
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

            step_pos, step_vel, step_ground = self._try_move(self._current_shape, raised_pos, start_vel, dt)

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
            # Narrower than the real hull (see _VERTICAL_PROBE_INSET) - this is the sweep that
            # was confirmed (via a dedicated physics-only test) to catch a short obstacle's own
            # front-top edge and reject an otherwise-climbable step as "not walkable ground"
            # while the hull's center is still short of the obstacle, even though its own
            # leading face already touches it (the moment stepping up matters most).
            down_result = self._sweep(self._current_vertical_probe_shape, step_pos, down_target)
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
                # could slide past). See _STEP_TIE_EPSILON_SQ's own
                # docstring for why this comparison needs a real
                # tolerance rather than an exact ">" (a too-tight one
                # visibly vibrated the character against any wall).
                # Moving diagonally up a slope (forward+strafe together, not straight up the
                # fall line) is a genuine, PERSISTENT near-tie, not sweep noise: the flat
                # attempt's clip against the slope only removes the into-slope part of vel, so
                # the more of the player's input is sideways-along-the-slope rather than
                # straight up it, the closer flat_dist_sq sits to step_dist_sq every single
                # tick - unlike the wall-vibration case the epsilon above already covers, this
                # doesn't average out. Deciding fresh each tick inside that near-tie band still
                # flips the winner tick to tick on which candidate's own sweep noise happened to
                # land fractionally ahead, and flat_vel/step_vel genuinely differ (flat_vel is
                # slope-clipped and slower, step_vel carries the raw unclipped input speed) - so
                # flipping pulses the player's speed up and down while climbing, reading as a
                # stutter even though neither candidate is more "correct" than the other here.
                # Sticking with whichever one won last tick while inside the band (only crossing
                # when one candidate pulls unambiguously ahead by more than the epsilon) turns
                # that flicker into a single stable choice for as long as the tie persists.
                if flat_dist_sq > step_dist_sq + _STEP_TIE_EPSILON_SQ:
                    flat_wins = True
                elif step_dist_sq > flat_dist_sq + _STEP_TIE_EPSILON_SQ:
                    flat_wins = False
                else:
                    flat_wins = self._last_step_was_flat
                self._last_step_was_flat = flat_wins
                if flat_wins:
                    final_pos, final_vel, discovered_ground = flat_pos, flat_vel, flat_ground
                else:
                    final_pos, final_vel, discovered_ground = settled_pos, step_vel, step_ground
            else:
                final_pos, final_vel, discovered_ground = flat_pos, flat_vel, flat_ground
        else:
            final_pos, final_vel, discovered_ground = flat_pos, flat_vel, flat_ground

        self.node_path.setPos(to_physics_pos(final_pos))
        self.velocity = final_vel
        if discovered_ground is not None:
            # _try_move's own collision sweep just directly touched a walkable floor/ramp
            # THIS tick - trust that immediately rather than waiting on
            # _categorize_position's separate, narrower, straight-down-only probe to
            # independently rediscover the same contact (possibly not until next tick, if
            # ever): confirmed via this file's own debug_log (F6) that after landing on a
            # sloped surface, that probe can keep reporting hit=False indefinitely even
            # while the player is plainly resting on/sliding along the exact surface
            # _try_move keeps colliding with every tick - a straight vertical ray only
            # has to miss a tilted plane by a little to never reach it within its own
            # short fixed reach, especially from a contact point offset to the side of
            # where the hull center's own straight-down ray would land. Every tick that
            # keeps missing is a tick _grounded stays False, which means friction never
            # runs (see the grounded-gated branches in _on_pre_substep) - a slide that
            # never decays. Setting it here, from ground contact _try_move already
            # confirmed directly, closes that gap instead of depending on the probe ever
            # managing to independently confirm the same thing.
            self._grounded = True
            self._ground_normal = discovered_ground
        # The "flat" candidate's own ClipVelocity (inside _try_move) is what makes it climb a
        # slope in the first place - the clipped vector is tangent to the slope, so it carries a
        # genuine nonzero Y the moment this tick's input has any into-slope component. Left in
        # self.velocity for the NEXT tick to inherit, that positive Y satisfies _categorize_
        # position's OWN "velocity.y > 0 means airborne" test - which this file's own docstrings
        # assume can only ever happen from an actual jump - so the very next tick skips the
        # ground probe outright and reports not-grounded, despite the hull sitting right on the
        # slope's surface with real horizontal speed. That one false "not grounded" tick skips
        # stepping (disabling it for one tick) and, worse, lets THIS tick's flat-only path clip
        # velocity straight to near-zero against the slope - exactly a stutter, and more likely
        # the more of the player's movement is sideways-along-the-slope (see _STEP_TIE_EPSILON_
        # SQ's own comment on near-ties above) since that's when "flat" wins over "step" more
        # often. Harmless to clear here regardless of which candidate won: conforming to a
        # slope/small ledge is handled by this method's own position-based settle every tick,
        # never by carrying a persistent vertical speed between ticks (see _on_pre_substep's own
        # landed-velocity-reset, which already does exactly this for the negative/falling case -
        # this is that same invariant restored for the positive/climbing case it missed).
        if self._grounded:
            self.velocity.y = 0.0

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

    def teleport(self, position):
        """Puts the hull at `position` (render space, hull CENTER) with no velocity
        and no interpolation smear - for respawning."""
        self.node_path.setPos(to_physics_pos(position))
        self.velocity = glm.vec3(0.0)
        self._prev_position = to_render_pos(self.node_path.getPos())
        self._smoothed_position = glm.vec3(self._prev_position)
        self._grounded = False
        self._jumping = False

    def toggle_debug_log(self):
        """Flips the per-physics-tick console dump on/off (see self._debug_log's own
        comment) - returns the new state so a caller (app.py's debug keybind) can also
        reflect it somewhere on screen if useful."""
        self._debug_log = not self._debug_log
        self._debug_tick = 0
        return self._debug_log

    def is_on_ground(self):
        return self._grounded

    def is_sprinting(self):
        """Whether set_sprinting(True) is the current input state - the
        player's actual INTENT, not a derived speed measurement (see
        PlayerModel.update()'s own is_sprinting param, which uses this
        instead of a speed threshold to pick the run animation)."""
        return self._sprinting

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
        #
        # Only ever called while grounded (see _update_crouch) - the
        # headroom-only slice this checks (see __init__) relies on the
        # grounded stand-up's own feet-fixed/grow-upward-only assumption
        # to safely skip the floor; there's no equivalent airborne call
        # site anymore; standing up is simply never attempted mid-air at
        # all (_update_crouch locks "still crouched" as the only outcome
        # of crouching in the air, until landing).
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

    def _compute_swap_pos(self, total_height, target_crouch_amount):
        """Where _swap_to_shape would move the hull to for a swap to
        (total_height, target_crouch_amount) - factored out of
        _swap_to_shape only so the two don't duplicate this math."""
        old_pos = self.node_path.getPos()

        if self._grounded:
            # Keeps the hull's BOTTOM (feet) fixed across the swap by
            # shifting the center by half the total-height difference,
            # rather than leaving the center in place - crouching
            # should lower the head, not lift the feet off (or sink
            # them into) the ground.
            return Point3(old_pos.x, old_pos.y, old_pos.z + (total_height - self._current_height) / 2.0)
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
            #
            # This branch is only ever reached going DOWN (crouching)
            # now - _update_crouch never attempts to grow back to
            # standing height while airborne at all (see its own
            # comment), so there's no case here where this needs a
            # clearance check: shrinking always moves the feet UP, away
            # from any floor.
            old_eye_offset = self._eye_offset_for(self._current_height, self._crouch_amount)
            new_eye_offset = self._eye_offset_for(total_height, target_crouch_amount)
            return Point3(old_pos.x, old_pos.y, old_pos.z + (old_eye_offset - new_eye_offset))

    def _swap_to_shape(self, shape, total_height, target_crouch_amount):
        # Swaps the shape on the SAME persistent ghost node
        # (addShape/removeShape, confirmed to work on a live node)
        # rather than replacing the node itself, so self.velocity is
        # never touched by this at all - unlike
        # BulletCharacterControllerNode, there's no hidden internal
        # state here that recreating a node would lose.
        new_pos = self._compute_swap_pos(total_height, target_crouch_amount)

        self.node.removeShape(self._current_shape)
        self.node.addShape(shape)
        self.node_path.setPos(new_pos)

        self._current_shape = shape
        self._current_height = total_height
        self._current_vertical_probe_shape = (
            self._crouch_vertical_probe_shape if shape is self._crouch_shape
            else self._standing_vertical_probe_shape)

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
        elif self._is_crouched and self._grounded:
            # Standing back up is ONLY ever attempted while grounded -
            # crouch_input being released while AIRBORNE does nothing at
            # all here (falls through to neither branch), no matter how
            # many times it's toggled before landing. This used to also
            # attempt a (correctly refused, once the earlier clip-
            # through-the-floor bug was fixed) stand-up mid-air, which
            # meant repeatedly tapping crouch in the air could land you
            # in a state where the LAST toggle before touching down
            # happened to be "released" at a moment the check refused,
            # leaving you stuck crouched on the ground with no obvious
            # reason why (the very next successful release should have
            # stood you up, but nothing was still watching for it once
            # this call returned). Locking "still crouched" as the ONLY
            # possible outcome of crouching at all while airborne -
            # standing up strictly requires a fresh crouch_input=False
            # read while self._grounded is already True - removes that
            # ambiguity entirely: landing always re-evaluates crouch_
            # input fresh on the very next tick regardless of how it was
            # spammed in the air, and going down mid-air is still fully
            # allowed and instant (see _compute_swap_pos's airborne
            # branch - shrinking always moves the feet UP, away from any
            # floor, so it never needed a clearance check to begin with).
            #
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
        self._categorize_position(dt)
        if self._debug_log:
            # Snapshot of categorize_position's own verdict, before anything below
            # (jump handling, friction/gravity, step_slide_move) can change it - see the
            # print at the bottom of this method for why both this and the post-move
            # state matter.
            debug_pre_grounded = self._grounded
            debug_pre_jumping = self._jumping
            debug_pre_normal = glm.vec3(self._ground_normal)
            debug_pre_vel = glm.vec3(self.velocity)

        # Captured immediately after categorize_position, BEFORE the
        # jump-request handling below can clear self._grounded again -
        # holding jump (app.py calls player.jump() every frame the key
        # is held, for Source-style auto-bunnyhop-on-landing) means a
        # landing tick often immediately re-launches within the SAME
        # tick, so by the time _update_footsteps runs later this tick,
        # self._grounded already reads False again even though the hull
        # genuinely touched the floor a few lines up. _update_footsteps
        # needs to see that real, if momentary, ground contact to fire a
        # landing sound - see its just_landed.
        self._touched_ground = self._grounded
        # How fast the hull was actually falling at the moment ground
        # was found this tick (0 if it wasn't falling at all) - see
        # _update_footsteps' just_landed for why this gates the landing
        # sound instead of self._touched_ground alone: pressed against a
        # wall while otherwise resting on the floor, the ground sweep
        # can flicker no-hit/hit tick to tick (the same class of sweep-
        # margin noise noted elsewhere in this file) with velocity.y
        # sitting at ~0 throughout - not a real fall, just noise - and
        # firing a landing sound on every one of those flickers reads as
        # a machine-gun of footsteps while just standing still shoving a
        # wall. A genuine fall (even a small hop) carries real downward
        # speed by the time it's caught here; standing-still noise does
        # not.
        self._fall_speed_on_touch = -self.velocity.y if self.velocity.y < 0.0 else 0.0

        if self._grounded and self.velocity.y < 0.0:
            # Landed (or resting) - clear any residual downward
            # velocity so it doesn't accumulate call after call; actual
            # ground-conforming (slopes, small ledges) is handled by
            # _step_slide_move's position-based settle, not by feeding
            # it a persistent downward speed. Horizontal velocity is
            # deliberately left alone - fall momentum carries through
            # into the landing, same as every other velocity-driven move
            # here.
            self.velocity.y = 0.0

        if self._jump_requested and self._grounded:
            self.velocity.y = self.jump_speed
            self._grounded = False
            self._jumping = True
            self._pending_jump_event = True
        self._jump_requested = False

        wishdir, wishspeed = self._wish()
        if self._grounded:
            self._apply_friction(dt)
            self._accelerate(wishdir, wishspeed, self.ground_accel, dt)
        else:
            self.velocity.y -= self.gravity * dt
            self._air_accelerate(wishdir, wishspeed, self.air_accel, dt)

        self._step_slide_move(dt)

        if self._debug_log:
            # One line per physics tick: the ground verdict/velocity BEFORE this tick's
            # jump/gravity/move handling ran ("pre", i.e. exactly what categorize_position
            # saw) next to the velocity/position AFTER _step_slide_move settled this
            # tick's actual move ("post") - printed together so a landing/slide tick is
            # visible as one line showing both "what the probe thought" and "what the
            # move actually did to velocity", instead of needing to cross-reference two
            # separate prints.
            self._debug_tick += 1
            pos = to_render_pos(self.node_path.getPos())
            horiz_speed = math.hypot(self.velocity.x, self.velocity.z)
            print(
                f"[movedbg #{self._debug_tick}] "
                f"pre(grounded={debug_pre_grounded}, jumping={debug_pre_jumping}, "
                f"normal=({debug_pre_normal.x:.3f},{debug_pre_normal.y:.3f},{debug_pre_normal.z:.3f}), "
                f"vel=({debug_pre_vel.x:.3f},{debug_pre_vel.y:.3f},{debug_pre_vel.z:.3f})) "
                f"probe(trace_dist={self._debug_trace_distance:.4f}, hit={self._debug_probe_hit}, "
                f"normal_y={self._debug_probe_normal_y}, fraction={self._debug_probe_fraction}) "
                f"post(grounded={self._grounded}, "
                f"vel=({self.velocity.x:.3f},{self.velocity.y:.3f},{self.velocity.z:.3f}), "
                f"horiz_speed={horiz_speed:.3f}, "
                f"pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}))"
            )

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
        # Landing edge (was airborne last tick, touched ground THIS
        # tick) - deliberately keyed off self._touched_ground, captured
        # in _on_pre_substep right after categorize_position, rather
        # than self._grounded as it reads down here. With jump held
        # (app.py calls player.jump() every frame the key's down, for
        # Source-style auto-bunnyhop-on-landing), a landing tick almost
        # always re-launches within that SAME tick - by the time this
        # method runs, self._jump_requested handling in _on_pre_substep
        # has already flipped self._grounded back to False, even though
        # the hull genuinely touched the floor a few lines earlier that
        # tick. Using the stale self._grounded here would mean bunny-
        # hopping (airborne almost the entire time, "grounded" for
        # under one tick between hops) never registers a landing at all.
        just_landed = (
            self._touched_ground and not self._was_grounded
            and self._fall_speed_on_touch >= _MIN_LANDING_FALL_SPEED
        )
        self._was_grounded = self._touched_ground

        if self._step_sound_timer > 0.0:
            self._step_sound_timer = max(self._step_sound_timer - dt, 0.0)

        if just_landed:
            # A landing impact is footstep-worthy ground contact on its
            # own, independent of the walking cadence timer below (which
            # counts down even while airborne, matching Source's
            # UpdateStepSound exactly - see this method's docstring) -
            # a hop's brief mid-air time is rarely enough for that timer
            # to reach 0 by the next landing, and a step that never hits
            # 0 never fires at all. Uses running cadence/volume (the
            # landing IS the impact, not a gait to keep time with) and
            # still resets the timer afterward so an immediate walking
            # step right after landing doesn't double up.
            volume = get_footstep_volume(self._ground_material, walking=False)
            if self._is_crouched:
                volume *= _STEP_VOLUME_DUCK_MULT
            self._step_sound_timer = _STEP_INTERVAL_RUN
            self._pending_footstep = (self._ground_material, volume)
            return

        if not self._grounded:
            return

        horiz_speed = math.hypot(self.velocity.x, self.velocity.z)
        min_speed = _STEP_MIN_SPEED_CROUCH if self._is_crouched else _STEP_MIN_SPEED_STAND
        run_speed = _STEP_RUN_SPEED_CROUCH if self._is_crouched else _STEP_RUN_SPEED_STAND
        walking = horiz_speed < run_speed
        interval = _STEP_INTERVAL_WALK if walking else _STEP_INTERVAL_RUN

        if self._step_sound_timer > 0.0 or horiz_speed < min_speed:
            return

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

    def pop_jumped(self):
        """Returns whether a jump actually executed since the last call,
        consuming the event the same way pop_footstep() does (so it never
        double-fires even if this is polled more than once before the
        next physics tick runs) - call once per frame (see app.py) and
        forward to PlayerModel.update()'s own just_jumped param."""
        event = self._pending_jump_event
        self._pending_jump_event = False
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
