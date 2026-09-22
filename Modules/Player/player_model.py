"""Generic skeletal-model attachment for a player (local or remote) -
composes Scene.add_skeletal's own generic-glb loading with animation-
state switching, so any skinned model can be hung off a player without
hardcoding a specific asset anywhere in here. See app.py (local player,
visible_in_color=False so only its shadow shows) and
Modules/Networking/remote_player.py (remote players, fully visible plus
a hitbox) for the two current call sites.
"""

import math
import time

import glm

# Temporary debugging aid - set True (or flip via a debugger/console) to
# print (AND append to _DEBUG_LOCOMOTION_LOG_PATH, since a console isn't
# always convenient to copy from) a line every time the SET of blended
# animation clips actually changes for a locomoting PlayerModel, so a
# reported "animation glitch" can be pinned down to an exact composition
# change (which clips, what weights, at what speed/commit/angle) instead
# of guessed at. Safe to leave False permanently; remove entirely once no
# longer needed.
DEBUG_LOCOMOTION = False
_DEBUG_LOCOMOTION_LOG_PATH = "locomotion_debug.log"


def _debug_log(line):
    """Prints line AND appends it to _DEBUG_LOCOMOTION_LOG_PATH - shared
    by every DEBUG_LOCOMOTION callsite (the per-frame composition-change
    line, and the discrete top-state ENTER/RESET lines) so all of them
    land in the same file in the same order, regardless of whether a
    console is actually being watched live."""
    print(line)
    try:
        with open(_DEBUG_LOCOMOTION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# The 8 buckets a facing-relative direction can bracket, in angle order
# starting from straight-forward (0 degrees) and sweeping toward +right -
# see _facing_angle_degrees for the angle convention. Order here only
# matters for _LocomotionBlendSpace's own bracket/neighbor search, not
# for the angle math itself.
_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# Seconds for PlayerModel's own self._sprint_blend to fully ease from 0
# to 1 (or back) once is_sprinting flips - see update()'s own is_sprinting
# docstring. Short enough that a sprint key press still reads as
# responsive, long enough to actually look like an eased blend rather
# than an instant cut.
_SPRINT_EASE_SECONDS = 0.15

# Seconds for PlayerModel's own self._smoothed_direction_angle to catch
# up to a NEW target facing-relative angle - see update()'s own
# move_direction docstring for why this exists: WASD input only ever
# produces a small fixed set of exact directions (whatever combination of
# keys is held), each dead-on one of directional_clips' own 8 sample
# angles, relative to the SAME camera basis the samples themselves are
# defined in - so without this, the blend space's direction axis would
# only ever see a discrete key-driven JUMP between two exact samples
# (100% of one to 100% of the other in a single frame), never an
# in-between angle to actually blend across, even though the math is
# perfectly continuous. Same idea, and same magnitude, as _SPRINT_EASE_
# SECONDS above for the speed axis's own discrete is_sprinting input.
_DIRECTION_EASE_SECONDS = 0.15

# Time constant (seconds) for PlayerModel's own self._smoothed_speed low-
# pass filter on the RAW speed input - see update()'s own speed docstring.
# Physics collision response can make a single physics-tick's velocity
# reading swing wildly for a frame or two even while the player's real
# held input never changes - most visibly, walking straight into a wall,
# where each tick's resolved velocity can bounce between near-zero
# (blocked) and near-full speed (the very next tick's acceleration
# reasserting itself before being blocked again). Fed straight into the
# blend space, that reads as the animation "flickering" between an idle
# and a walk/run pose every single frame - a real, large swing, not the
# small imperceptible wobble a continuous blend space is otherwise immune
# to (see _LocomotionBlendSpace's own docstring). Small enough that a
# genuine, deliberate speed change (actually starting to move, actually
# stopping) still reads as prompt.
_SPEED_EASE_SECONDS = 0.1

# Seconds update()'s own is_grounded argument must read False CONTINUOUSLY
# before PlayerModel treats the player as genuinely airborne and enters
# the "jump" top-level state - see _confirmed_airborne. CharacterController.
# is_on_ground() is a direct, undebounced read of its own ground-sweep
# trace, which Modules/Physics/character_controller.py's own comments
# (see _MIN_LANDING_FALL_SPEED) explicitly document as flickering for a
# tick or two while the player is pressed into a wall - reacting to every
# single one of those readings, as a plain "not is_grounded" edge check
# would, briefly enters "jump" for that one instant (playing the takeoff
# pose, then immediately crossfading back out) purely from the ground
# trace glitching, with no actual jump ever happening - visible as the
# character's pose popping (and, because entering the jump state exits
# and later re-enters the locomotion blend space, its locomotion_phase
# resetting to 0 - see add_skeletal's own docstring) every time it's
# walked into a wall. 0.05s was the original value here and turned out to
# be too short: a debug log capturing this exact scenario (pinned against
# a wall, never jumping) showed is_grounded reading False for 0.05-0.067s
# at a stretch, repeatedly - right at or just past that threshold, so it
# was barely ever actually filtering anything. 0.2s gives real margin
# above that observed worst case while still comfortably shorter than any
# genuine jump's own airborne time (a jump_speed of a few m/s against
# normal gravity stays airborne for several tenths of a second at least,
# usually much longer).
_JUMP_CONFIRM_SECONDS = 0.2

# Seconds move_direction must read near-zero-length CONTINUOUSLY before
# PlayerModel treats the player as genuinely having released every
# movement key - see _confirmed_has_input. Only used for the DIRECTION
# axis now (see _compute_smoothed_direction_angle) - the speed axis went
# back to being purely velocity-driven (see _compute_effective_speed's
# own docstring for why). app.py's own move_dir is built fresh each frame
# from pygame.key.get_pressed(), summing whichever of WASD's own camera-
# relative vectors are currently held - rolling a finger from one strafe
# key to its opposite (A to D, say) can leave the OS reporting BOTH
# unpressed for a single poll even though the player never stopped trying
# to move, momentarily zeroing move_dir's length. Confirmed via a debug
# log capturing exactly this: self._smoothed_direction_angle snapping
# toward _facing_angle_degrees' own meaningless zero-vector fallback on a
# single-frame blip, briefly blending in a wrong-facing directional clip
# while the player was clearly still trying to walk. Small enough that
# genuinely releasing every key still reads as prompt.
_NO_INPUT_CONFIRM_SECONDS = 0.1


def _ease_angle_degrees(current, target, dt, ease_seconds):
    """Moves current (degrees) toward target (degrees) by the fraction
    dt/ease_seconds of their shortest signed angular distance (wrapping
    correctly through +-180, e.g. easing from 170 toward -170 moves
    through 180/-180, not all the way back down through 0) - the same
    "move toward a moving target and clamp at it" shape PlayerModel's own
    _sprint_blend easing uses, just for a circular quantity instead of a
    plain 0..1 scalar. Reaches target exactly once dt/ease_seconds >= 1."""
    if ease_seconds <= 0.0:
        return target
    diff = ((target - current + 180.0) % 360.0) - 180.0
    return current + diff * min(1.0, dt / ease_seconds)


def _ease_value(current, target, dt, ease_seconds):
    """Simple exponential low-pass filter: moves current toward target by
    the fraction dt/ease_seconds of their remaining distance every call -
    unlike _ease_angle_degrees/_sprint_blend's own linear ramp (which
    reaches its target in a fixed time regardless of distance, right for
    smoothing a discrete input's OWN transition), this asymptotically
    settles toward a continuously-varying target, damping a brief
    transient spike/dip proportional to its size rather than following it
    - what self._smoothed_speed needs (see _SPEED_EASE_SECONDS's own
    docstring)."""
    if ease_seconds <= 0.0:
        return target
    return current + (target - current) * min(1.0, dt / ease_seconds)


def _ease_toward_linear(current, target, dt, ease_seconds):
    """Moves current toward target at a constant rate, reaching it
    EXACTLY within ease_seconds regardless of how far apart they start -
    used by self._sprint_blend (see _compute_effective_speed) so a
    sprint key press/release still eases smoothly into the walk<->run
    split instead of the discrete input snapping the blend weight
    outright."""
    if ease_seconds <= 0.0:
        return target
    step = dt / ease_seconds
    if current < target:
        return min(target, current + step)
    return max(target, current - step)


def _facing_angle_degrees(move_direction, yaw_degrees):
    """Returns a world-space horizontal move_direction's angle, in
    degrees, relative to yaw_degrees' own facing - 0 at due "N" (moving
    the way the camera looks), +-180 at "S" (backpedal), +90 at "E"
    (strafe right), -90 at "W" (strafe left) - the standard third-person
    strafe scheme, NOT an absolute world compass direction. yaw_degrees
    uses the exact same camera-yaw convention as PlayerModel.update()'s
    own model-rotation math (see its comment for the forward/right vector
    derivation: forward=(cos(yaw),0,sin(yaw)), right=(-sin(yaw),0,cos(yaw))).

    Computed as the angle of move_direction in the (forward, right)
    basis via atan2, continuous (NOT snapped to a 45-degree bucket -
    unlike this project's old discrete direction-bucketing, the blend
    space wants the raw angle so it can interpolate between the two
    nearest directional clips instead of picking one). A near-zero
    move_direction (no real direction to read) returns 0.0 ("N") - only
    reached if a caller passes one while directional_clips still has an
    entry for the resolved state despite there being no real movement
    input (e.g. still coasting on decaying speed after releasing every
    movement key)."""
    move = glm.vec3(move_direction)
    move.y = 0.0
    if glm.length(move) < 1e-6:
        return 0.0
    move = glm.normalize(move)
    yaw = glm.radians(yaw_degrees)
    forward = glm.vec3(glm.cos(yaw), 0.0, glm.sin(yaw))
    right = glm.vec3(-glm.sin(yaw), 0.0, glm.cos(yaw))
    fwd_component = glm.dot(move, forward)
    right_component = glm.dot(move, right)
    return glm.degrees(glm.atan(right_component, fwd_component))


class _LocomotionBlendSpace:
    """A genuine 2-axis GRID blend space for ONE animation track (lower
    or upper body) - X = speed, Y = facing-relative direction angle,
    exactly the two parameters a UE "Blend Space" (rendered there as a 3D
    graph: X/Y the two input axes, Z the resulting per-sample weight)
    would use for an 8-directional locomotion setup. Built once from a
    `states`-shaped (name, clip, min_speed) table, which places one "ring"
    of samples per speed value along the X axis, and an optional
    directional_clips table, which - for whichever rings have an entry -
    spaces that ring's own samples around the Y axis at up to 8 points
    (45 degrees apart); a ring with no directional_clips entry is a
    single OMNI-DIRECTIONAL sample instead (uniform across the whole Y
    axis at that X, e.g. idle/run here, matching how a real blend space
    handles an axis a given sample doesn't vary along). See PlayerModel.
    __init__'s own docstrings for both tables' shapes/meaning, unchanged
    from before this existed.

    compute_weights() is evaluated fresh every single frame and returns a
    flat list of (clip_name, weight) pairs summing to ~1, fed straight
    into Scene's locomotion_weights (see Scene.set_skeletal_locomotion/
    Skeleton._sample_weighted) - every sample the current (speed,
    direction) point's enclosing cell touches is sampled and blended
    simultaneously, instead of snapping to one "nearest" clip and
    hard-restarting it on every threshold/direction crossing. The
    interpolation is bilinear across the grid (linear on the X/speed axis
    between the two bracketing rings, linear on the Y/direction axis
    within each of those rings between its two bracketing samples) -
    mathematically exact for this rectilinear ring layout, unlike a
    generic scattered-point blend space, which needs full Delaunay
    triangulation to get the same guarantee.

    This intentionally has no hysteresis or minimum-dwell-time mechanism
    the way the old discrete state switcher did - those existed purely to
    stop a discrete snap-and-restart from flickering near a threshold; a
    CONTINUOUS blend weight wobbling slightly near a threshold (e.g.
    48%/52% jittering to 52%/48% for a tick) is imperceptible, so there's
    nothing here left to protect against. Both axes' own input values
    (speed, direction) are expected to already be whatever continuous
    number the caller wants blended - see PlayerModel.update()'s own
    effective_speed for how a discrete is_sprinting intent gets turned
    into one instead of ever being handled inside this class."""

    def __init__(self, states, directional_clips):
        sorted_states = sorted(states, key=lambda s: s[2])
        self._names = [s[0] for s in sorted_states]
        self._clips = [s[1] for s in sorted_states]
        self._thresholds = [float(s[2]) for s in sorted_states]
        self._directional_clips = directional_clips or {}

    @property
    def thresholds(self):
        """This blend space's own sorted speed-axis (X) sample values -
        exposed so PlayerModel.update() can map a discrete is_sprinting
        intent onto a continuous point along the SAME axis (see its own
        effective_speed docstring) without duplicating `states`' table
        here."""
        return self._thresholds

    def compute_weights(self, speed, direction_angle):
        """Returns [(clip_name, weight), ...] summing to ~1 (an entry is
        omitted entirely if its ring's weight is 0, or its ring has
        neither a plain clip nor any directional clip to substitute).
        speed: the X-axis value to sample (see this class's own
        docstring for why the caller may not always pass real speed
        as-is). direction_angle: the Y-axis value in degrees (see
        _facing_angle_degrees for the convention), already resolved -
        and, for a caller like PlayerModel with a discrete WASD-driven
        move_direction, already SMOOTHED over time (see its own
        _smoothed_direction_angle) rather than the raw instantaneous
        angle, for the same reason effective_speed exists on the X axis -
        or None to disable the direction axis entirely for this call
        (every ring just uses its plain clip)."""
        result = []
        for ring_index, ring_weight in self._ring_weights(speed):
            if ring_weight <= 0.0:
                continue
            name = self._names[ring_index]
            direction_map = self._directional_clips.get(name)
            if direction_map and direction_angle is not None:
                for clip_name, sub_weight in self._directional_weights(direction_map, direction_angle):
                    result.append((clip_name, ring_weight * sub_weight))
            elif self._clips[ring_index] is not None:
                result.append((self._clips[ring_index], ring_weight))
        return result

    def _ring_weights(self, speed):
        """Returns [(ring_index, weight), ...] - at most 2 adjacent rings
        active at once (the X/speed axis of the grid, over `states`' own
        sorted min_speed values), weights summing to 1: below the lowest
        threshold is 100% that ring, above the highest is 100% the
        fastest, and speed anywhere between two adjacent thresholds
        blends linearly between them."""
        thresholds = self._thresholds
        n = len(thresholds)
        if n == 1:
            return [(0, 1.0)]
        if speed <= thresholds[0]:
            return [(0, 1.0)]
        for i in range(1, n):
            if speed < thresholds[i]:
                span = thresholds[i] - thresholds[i - 1]
                t = 1.0 if span <= 0.0 else (speed - thresholds[i - 1]) / span
                return [(i - 1, 1.0 - t), (i, t)]
        return [(n - 1, 1.0)]

    @staticmethod
    def _directional_weights(direction_map, angle):
        """Returns [(clip_name, weight), ...] (1 or 2 entries, weights
        summing to 1) - the two directional clips bracketing angle
        (degrees, see _facing_angle_degrees's own convention),
        continuously interpolated by the fractional angle between them
        instead of snapped to whichever is nearest. A bracket missing
        from direction_map (e.g. a table with only cardinals authored so
        far, no diagonals) falls back outward to its own two immediate
        45-degree neighbors, same search order as this project's old
        discrete direction-bucket fallback, so a partially-authored table
        still degrades gracefully instead of dropping that bracket
        silently."""
        raw = angle / 45.0
        lo_index = int(math.floor(raw)) % 8
        frac = raw - math.floor(raw)
        hi_index = (lo_index + 1) % 8

        lo_name = _LocomotionBlendSpace._nearest_available(direction_map, lo_index)
        hi_name = _LocomotionBlendSpace._nearest_available(direction_map, hi_index)
        if lo_name is None and hi_name is None:
            return []
        if lo_name is None:
            return [(hi_name, 1.0)]
        if hi_name is None:
            return [(lo_name, 1.0)]
        if lo_name == hi_name:
            return [(lo_name, 1.0)]
        return [(lo_name, 1.0 - frac), (hi_name, frac)]

    @staticmethod
    def _nearest_available(direction_map, index):
        """direction_map's own clip at _DIRECTIONS[index] if present,
        else its two immediate neighbors in cyclic order (counter-
        clockwise first, then clockwise), else None if none of the three
        are authored."""
        for i in (index, (index - 1) % 8, (index + 1) % 8):
            clip = direction_map.get(_DIRECTIONS[i])
            if clip is not None:
                return clip
        return None


def _as_blend_space(spec, directional_clips):
    """Normalizes a jump_animation/crouch_animation-style constructor
    param into a _LocomotionBlendSpace, so either kind of state can be
    driven by the exact same weighted-clip machinery locomotion itself
    already uses - the "conditional swapping between animations" a real
    UE state machine gets from putting a full blend space (or another
    state machine) inside a single state, instead of every state being
    forced to hold just one fixed pose.

    spec may be:
    - None: feature disabled - returns None.
    - a single clip NAME (str): wrapped as a trivial one-tier blend space
      that always resolves to that one clip regardless of speed/
      direction - matches this project's original single-clip jump_
      animation/crouch_animation behavior exactly, so every existing
      caller keeps working unchanged.
    - a `states`-shaped (name, clip, min_speed) tuple, exactly like
      PlayerModel's own `states` param: a real conditional blend,
      resolved by whatever speed/direction the caller passes to
      compute_weights - e.g. a crouch table with ("idle", "crouch_idle",
      0.0) and ("walk", "crouch_walk", 90*0.0254) blends smoothly between
      a held crouch pose and a crouch-walk cycle by speed, instead of
      crouch being stuck on one static pose the instant it's entered.
      Shares the SAME directional_clips table passed to PlayerModel.
      __init__ (looked up by whichever state names this tuple uses), so
      a crouch-walk tier can get its own 8-directional variants the exact
      same way locomotion's own "walk" tier does, with no separate table
      needed."""
    if spec is None:
        return None
    if isinstance(spec, str):
        return _LocomotionBlendSpace((("_single", spec, 0.0),), None)
    return _LocomotionBlendSpace(spec, directional_clips)


class PlayerModel:
    def __init__(self, scene, model_path, position=None, forward_offset_degrees=0.0,
                 feet_offset=0.0, visible_in_color=True, cast_shadow=True,
                 states=(("idle", None, 0.0),), upper_states=None,
                 animation_blend_duration=None,
                 upper_body_root_joints=None, upper_animation=None,
                 override_upper_body_root_joints=None,
                 upper_rotation_offset_degrees=(0.0, 0.0, 0.0),
                 jump_animation=None, crouch_animation=None, crouch_blend_duration=0.1,
                 directional_clips=None,
                 scale=None, metallic=None, roughness=None, emissive=None, texture_path=None,
                 time_scale=1.0):
        """scene: the Scene this model is added to (needs scene.add_skeletal
        and scene.set_skeletal_locomotion/set_skeletal_animation).

        forward_offset_degrees/feet_offset: visual-correction knobs, same
        spirit as skybox.py's own `rotations` param - a glb's authored
        rest-pose forward axis and origin height can't be known without
        actually looking at it rendered, so these exist to tune a
        specific model after checking in-engine rather than guessing
        silently. feet_offset shifts the model vertically relative to
        whatever position update() is given (see update()'s own
        docstring for the feet-vs-hull-center contract that motivates
        this); forward_offset_degrees rotates it - if a model faces
        backward/sideways at 0, try 180 first.

        This model's LOCOMOTION (idle/walk/run, and any facing-relative
        directional variants - see directional_clips below) is driven by
        a continuous speed/direction BLEND SPACE, not a discrete state
        switcher: every frame, `states`/directional_clips' own clips are
        sampled and blended together, weighted by the current speed and
        move_direction, the same way Unreal's AnimGraph blend spaces
        work - see _LocomotionBlendSpace's own docstring for exactly how.
        This is one of three top-level states this class tracks
        explicitly (self._top_state): "locomotion" (the blend space,
        described here), "jump", and "crouch" (jump_animation/
        crouch_animation below - held single poses that override
        locomotion entirely while active, with an ordinary crossfade back
        INTO the blend space on exit).

        states: a speed-driven blend-space axis - a sequence of
        (name, clip, min_speed) triples, e.g.:
            [("idle", "rifle_idle", 0.0),
             ("walk", "rifle_walk", 90.0 * 0.0254),
             ("run",  "rifle_run",  220.0 * 0.0254)]
        Sorted internally by min_speed ascending, so passing order
        doesn't matter. update()'s `speed` continuously blends between
        whichever two adjacent tiers it falls between (see
        _LocomotionBlendSpace._tier_weights) - below the lowest min_speed
        is 100% that tier, above the highest is 100% the fastest, speed
        between two tiers' thresholds blends linearly. This is
        deliberately an open-ended list, not three fixed named params -
        appending a new tier (crouch-walk, sprint, whatever) later is
        just adding another (name, clip, min_speed) triple here, no other
        code in this class needs to change. A tier's clip may be None
        (e.g. a placeholder tier with no animation yet), in which case it
        simply contributes no weighted entry of its own (see
        _LocomotionBlendSpace.compute_weights) - the default, a single
        "idle" tier pinned at min_speed=0.0 with clip=None, matches a
        model with no speed-driven animation at all.

        Source's own velwalk/velrun (90/220 Source units/s * 0.0254 - the
        same constants character_controller.py's footstep-sound code
        uses) are just the min_speed values a caller puts directly into
        its own "walk"/"run" tier triples above; PlayerModel itself has
        no opinion on how many tiers or what their thresholds are.

        animation_blend_duration: seconds to crossfade over whenever a
        top-level state actually changes (locomotion <-> jump <->
        crouch - see Scene.set_skeletal_animation/set_skeletal_
        locomotion) - None (the default) uses the Scene's own default
        (currently 0.25s). Pass 0.0 for an instant hard cut instead. Has
        no bearing on locomotion's own internal blend-space weighting,
        which is continuous by construction and needs no crossfade timer
        of its own.

        upper_body_root_joints: optional list of joint names splitting
        this model into a lower body (locomotion, driven by `states`
        above) and an upper body (e.g. a gun-holding pose) - see
        Scene.add_skeletal's own upper_body_root_joints/Skeleton.
        compute_joint_mask docstrings for exactly how the split works (a
        LIST of root names, not one pivot joint, since some rigs - rat.
        glb's Source/Valve Biped rig included - split the arm/neck/head
        chain off as a sibling subtree rather than a strict descendant of
        one spine bone). upper_animation seeds the initial upper-body
        clip when upper_states is NOT given; ignored if upper_states IS
        given (that table's own lowest-min_speed tier decides the initial
        clip instead). Leave both this and upper_body_root_joints at
        their defaults (None) for a plain single-clip model with no body
        split at all - `states`' own blend space is completely unaffected
        either way, since it only ever touches the LOWER-body track
        unless upper_states is also given.

        override_upper_body_root_joints: an optional, WIDER (or just
        different) root-joint list used ONLY while a set_upper_override
        lock is active - None (the default) means overrides use the
        exact same split as everything else (upper_body_root_joints).
        Locomotion-driven upper-body poses deliberately stay narrower
        (just the clavicles - see [[upper-body-joint-split]] memory),
        keeping spine/neck/head under the LOWER body's control, since a
        gun-holding pose blended continuously by speed/direction
        shouldn't fight the lower body for head look-direction/spine
        lean. A manually-forced override is a different situation - e.g.
        rat.glb's pistolidle.glb was authored with its own Spine4/neck
        rotation as PART of the pose, so splitting it off at the
        clavicles fed that clip's clavicle rotation the wrong PARENT
        transform. Passing ["...Spine4"] here (Spine4 is the shared
        parent of both clavicle chains AND Neck1 in this rig) lets such
        an override supply its OWN correct Spine4/neck pose instead.
        set_upper_override switches to this mask; clear_upper_override
        switches back to upper_body_root_joints.

        upper_rotation_offset_degrees: passed straight through to Scene.
        add_skeletal's own param of the same name - a COMPONENT-space
        correction applied to upper_body_root_joints' root joints only,
        every frame, regardless of which upper-body pose is currently
        playing. See set_upper_rotation_offset to change this later.

        This class deliberately has no aim/fire/reload state machine of
        its own for the upper body - nothing in this project (yet) knows
        what a weapon or crouch even means at the animation layer, so
        set_upper_animation() is a plain external hook: call it whenever
        a future weapon/aim system decides the upper-body pose should
        change, same as it would for anything else driving the model.

        upper_states: the upper-body counterpart to `states` above - same
        (name, clip, min_speed) shape, blended by its OWN
        _LocomotionBlendSpace every frame in lockstep with the lower
        body's (same effective_speed/move_direction point on the grid -
        see update()'s own effective_speed docstring - and the same
        directional_clips table - see below), so the two can never
        disagree about which tier(s) are active, only about whether a
        given tier has a clip on each side. None (the default) keeps the
        upper body purely externally driven via set_upper_animation()
        instead (e.g. once a real aim/fire state machine exists and wants
        finer control than "mirrors locomotion") - the two modes are
        mutually exclusive in practice (auto-driving overwrites a manual
        set_upper_animation() call on the very next frame) but nothing
        stops calling set_upper_animation() for a one-off override in
        between (or set_upper_override() for one that STAYS put - see its
        own docstring).

        jump_animation: optional clip (or, for conditional swapping
        between a few jump poses, a `states`-shaped (name, clip,
        min_speed) tuple - see _as_blend_space) that overrides locomotion
        entirely for as long as update()'s is_grounded argument reads
        False. Unlike locomotion/crouch_animation, this is resolved to a
        single clip ONCE, at the instant of leaving the ground (from the
        current speed/direction at that moment - e.g. a table with a
        "standing jump" tier at min_speed=0 and a "running jump" tier at
        some higher speed) rather than continuously reblended while
        airborne - the highest-weighted candidate from that one-time
        resolution is what actually plays, with loop=False (see Scene.
        add_skeletal/set_skeletal_animation's own loop param) so it
        freezes on its last frame rather than looping the takeoff motion
        while still airborne. This one-shot-pick (rather than a truly
        continuous blend, unlike crouch_animation below) is a deliberate
        engine-capability tradeoff, not a design preference: Scene's
        weighted/blend-space clip path (see set_skeletal_locomotion) has
        no equivalent of "clamp and hold the last frame" yet, which a
        held one-shot takeoff pose needs. Applies to BOTH the lower body
        and, if this PlayerModel auto-drives its upper body (upper_states
        given), the upper body too - a jump pose is inherently full-body,
        unlike the upper/lower locomotion split. The instant is_grounded
        reads True again, the locomotion blend space resumes immediately,
        crossfaded in via Scene.set_skeletal_locomotion (freshly
        re-evaluated from the current speed/direction, not whatever was
        blended before the jump). None (the default) disables this
        entirely - is_grounded is then accepted but ignored.

        crouch_animation: optional clip (or, for a genuine crouch-idle/
        crouch-walk blend space - directional variants included, via the
        SAME directional_clips table below - a `states`-shaped (name,
        clip, min_speed) tuple, see _as_blend_space) that overrides
        locomotion entirely for as long as update()'s is_crouched
        argument reads True. Unlike jump_animation above, this stays a
        REAL continuous blend space for as long as crouched - re-
        evaluated fresh every single frame from the current speed/
        direction, exactly like locomotion's own blend space (indeed
        driven through the very same Scene.set_skeletal_locomotion/
        locomotion_weights mechanism) - so a crouch table with more than
        one tier smoothly blends from a held crouch-idle pose into a
        crouch-walk cycle as the player starts moving, instead of
        freezing on whatever pose was showing the instant crouch was
        entered. A single-clip crouch_animation (the original behavior)
        is just the degenerate one-tier case of this same mechanism -
        every frame resolves to that one clip regardless of speed, which
        looks identical to a plain held pose. Applies to BOTH the lower
        body and, if this PlayerModel auto-drives its upper body, the
        upper body too (the SAME resolved weighted clip list is used for
        both tracks - there's no separate upper-body crouch table, since
        a crouch pose is inherently full-body like jump_animation's own,
        not split the way ordinary locomotion is). Checked AFTER
        jump_animation/is_grounded (crouching has no effect while
        airborne) but BEFORE locomotion's own blend space, so landing
        while still holding crouch goes straight to the crouch blend
        space rather than flashing a locomotion blend first. The instant
        is_crouched reads False again, the locomotion blend space resumes
        immediately, same crossfade-in mechanism as jump_animation's own
        landing case. None (the default) disables this entirely.

        crouch_blend_duration: crossfade seconds used specifically for
        crouch_animation transitions (both entering and leaving crouch),
        SEPARATE from animation_blend_duration above - defaults to 0.1s
        here but every existing caller passes its own value. Standing to
        crouched (and back) is a much bigger, slower full-body pose
        change than locomotion's own blend-space wobble ever is, so a
        longer, more visible ease between the two poses reads as an
        actual crouch motion instead of a snap. Only meaningful alongside
        crouch_animation; ignored otherwise.

        directional_clips: optional {state_name: {direction: clip}}
        table overriding a blend-space tier's own single clip with up to
        8 facing-relative directional variants - "N"/"NE"/"E"/"SE"/"S"/
        "SW"/"W"/"NW" (see _facing_angle_degrees - the standard
        third-person strafe scheme, NOT absolute world compass
        directions). A tier absent from this table (or with move_
        direction=None) keeps its plain clip exactly as if directional_
        clips didn't exist. A tier present here is blended continuously
        between whichever two directional entries bracket the current
        facing angle (see _LocomotionBlendSpace._directional_weights) -
        not snapped to the nearest one - so e.g. turning smoothly from
        moving forward into a rightward strafe fades N's weight down and
        E's up continuously, with no restart at any point along the way.
        A direction missing from a tier's own sub-dict falls back to its
        two immediate 45-degree neighbors before giving up on that
        bracket entirely (see _nearest_available) - so e.g. giving "walk"
        only the 4 cardinals (no diagonals yet) still blends a sensible
        pair for a diagonal input instead of skipping straight to the
        plain clip. None (the default) disables this entirely.

        scale/metallic/roughness/emissive/texture_path/time_scale:
        passed straight through to scene.add_skeletal for full override
        parity with that method's own knobs - time_scale specifically is
        an opt-in per-keyframe-time correction for a model file known to
        be baked at the wrong frame rate (see skeletal_loader.
        load_skinned_glb's own docstring)."""
        self._scene = scene
        self._forward_offset_radians = glm.radians(forward_offset_degrees)
        self._feet_offset = float(feet_offset)

        if not states:
            raise ValueError("PlayerModel requires at least one state in `states`")
        self._lower_blend_space = _LocomotionBlendSpace(states, directional_clips)

        self._upper_auto_driven = upper_states is not None
        self._upper_blend_space = (
            _LocomotionBlendSpace(upper_states, directional_clips) if self._upper_auto_driven else None
        )

        self._animation_blend_kwargs = (
            {} if animation_blend_duration is None else {"blend_duration": float(animation_blend_duration)}
        )

        self._upper_override = None
        self._upper_body_root_joints = upper_body_root_joints
        self._override_upper_body_root_joints = override_upper_body_root_joints

        self._jump_blend_space = _as_blend_space(jump_animation, directional_clips)
        self._crouch_blend_space = _as_blend_space(crouch_animation, directional_clips)
        self._crouch_blend_kwargs = {"blend_duration": float(crouch_blend_duration)}

        # One of "locomotion" / "jump" / "crouch" - see this class's own
        # docstring above. Starts at "locomotion" unconditionally: a
        # caller only ever passes is_grounded=False/is_crouched=True on a
        # LATER update() call, matching every caller before jump/crouch
        # existed (nothing constructs a PlayerModel already airborne or
        # already crouched).
        self._top_state = "locomotion"

        # 0.0 (walk intent) .. 1.0 (run intent) - see update()'s own
        # is_sprinting docstring for why this exists and how it's eased
        # toward its target every frame rather than snapped.
        self._sprint_blend = 0.0

        # Degrees, or None while there's no real move_direction to track -
        # see update()'s own move_direction docstring for why this eases
        # toward the raw facing-relative angle every frame instead of
        # using it directly. None also means "not yet initialized",
        # snapping straight to the first real target instead of easing
        # in from a meaningless default the moment movement starts.
        self._smoothed_direction_angle = None

        # Low-pass-filtered speed - see update()'s own speed docstring
        # and _SPEED_EASE_SECONDS for why the raw per-frame speed isn't
        # fed into the blend space directly.
        self._smoothed_speed = 0.0

        # How long is_grounded has read False CONTINUOUSLY, most recently
        # - see _confirmed_airborne/_JUMP_CONFIRM_SECONDS.
        self._airborne_confirm_elapsed = 0.0

        # How long move_direction has read near-zero-length CONTINUOUSLY,
        # most recently - see _confirmed_has_input/_NO_INPUT_CONFIRM_
        # SECONDS.
        self._no_input_elapsed = 0.0

        # See DEBUG_LOCOMOTION.
        self._debug_last_clip_names = None

        initial_pos = glm.vec3(position) if position is not None else glm.vec3(0.0)
        initial_pos.y += self._feet_offset

        # Speed=0, no move_direction - the initial pose is whatever the
        # lowest-min_speed tier resolves to (matching this class's
        # original idle-at-rest behavior), fully weighted since nothing
        # else has any speed-axis weight at speed 0.
        initial_lower_weights = self._lower_blend_space.compute_weights(0.0, None)
        initial_lower_clip = initial_lower_weights[0][0] if initial_lower_weights else None
        if self._upper_auto_driven:
            initial_upper_weights = self._upper_blend_space.compute_weights(0.0, None)
            initial_upper_animation = initial_upper_weights[0][0] if initial_upper_weights else None
        else:
            initial_upper_weights = None
            initial_upper_animation = upper_animation

        self.obj = scene.add_skeletal(
            model_path, position=initial_pos, rotation=glm.vec3(0.0),
            scale=scale, animation=initial_lower_clip,
            metallic=metallic, roughness=roughness, emissive=emissive,
            texture_path=texture_path,
            visible_in_color=visible_in_color, cast_shadow=cast_shadow,
            upper_body_root_joints=upper_body_root_joints, upper_animation=initial_upper_animation,
            upper_rotation_offset_degrees=upper_rotation_offset_degrees,
            time_scale=time_scale,
        )
        # add_skeletal returns None on load failure (bad path, wrong
        # format, etc - see its own docstring) - every other method
        # below no-ops in that case rather than raising, so a missing/
        # broken model asset degrades to "no visible body" instead of
        # crashing whatever owns this PlayerModel.
        if self.obj is not None:
            # Enters the blend-space state from frame 1 - blend_duration
            # =0.0 since add_skeletal above already seeded the exact same
            # initial clip(s), so there's nothing to actually crossfade
            # FROM here, just the bookkeeping to start driving locomotion_
            # weights instead of a single static clip.
            self._scene.set_skeletal_locomotion(
                self.obj, initial_lower_weights, initial_upper_weights, blend_duration=0.0
            )

    def _compute_effective_speed(self, speed, dt, is_sprinting):
        """Returns the speed-axis (X) value to actually feed into
        _LocomotionBlendSpace.compute_weights this frame - see update()'s
        own is_sprinting docstring for why this exists. speed here is
        already self._smoothed_speed (see update()'s own speed docstring
        and _SPEED_EASE_SECONDS), not the raw per-frame value - real
        measured velocity is what drives idle/walk/run, same as the
        blend space's own continuous design intends (a brief in-flight
        detour made this depend on move_direction/is_sprinting instead,
        specifically to dodge is_grounded flickering during a wall
        collision popping the jump pose in - now that the ACTUAL bug
        there is fixed at its source (see _JUMP_CONFIRM_SECONDS and
        update()'s own just_jumped param), speed is trustworthy again and
        this went back to depending on it).

        is_sprinting is None (no discrete sprint bit to read, e.g. a
        remote player only ever given a position stream to estimate
        speed from): every tier boundary interpolates continuously by
        speed alone - _LocomotionBlendSpace's own grid already does the
        right thing with no help needed here.

        is_sprinting is True/False: the idle<->moving boundary still
        comes from real speed directly; but once actually moving,
        self._sprint_blend eases toward 1.0 (sprinting) or 0.0 (not) at a
        constant rate over _SPRINT_EASE_SECONDS, and the returned value
        is a point on the speed axis interpolated between the walk
        tier's own threshold (at self._sprint_blend=0) and the fastest
        tier's own threshold (at self._sprint_blend=1) - so the SAME grid
        interpolation used for a continuously-measured speed also
        handles this discrete key-driven case, just fed a smoothly-
        moving synthetic input instead of the boolean directly. Only
        meaningful with at least 2 tiers - with exactly 2 (no distinct
        "run" beyond "walk"), the walk and fastest thresholds are the
        same value and is_sprinting has no effect, matching this
        project's own is_sprinting rationale."""
        if is_sprinting is None:
            return speed

        self._sprint_blend = _ease_toward_linear(self._sprint_blend, 1.0 if is_sprinting else 0.0, dt, _SPRINT_EASE_SECONDS)

        thresholds = self._lower_blend_space.thresholds
        walk_threshold = thresholds[1] if len(thresholds) >= 2 else None
        if walk_threshold is None or speed <= 0.0 or speed < walk_threshold:
            return speed
        fast_threshold = thresholds[-1]
        return walk_threshold + self._sprint_blend * (fast_threshold - walk_threshold)

    def _compute_smoothed_direction_angle(self, move_direction, yaw_degrees, dt, has_input):
        """Returns the direction-axis (Y) value to actually feed into
        _LocomotionBlendSpace.compute_weights this frame, or None to
        disable the direction axis entirely - see update()'s own
        move_direction docstring for why this eases toward the raw
        facing-relative angle instead of using it directly.

        move_direction=None resets self._smoothed_direction_angle to None
        and returns None - this is the "no signal at ALL" case (see
        _compute_effective_speed's own docstring, e.g. a remote player
        with no discrete input to read), NOT the same as has_input/a
        near-zero move_direction below.

        Whenever move_direction IS given but currently reads as "no real
        direction" - either has_input is False (a movement key was
        genuinely released - see _confirmed_has_input/_NO_INPUT_CONFIRM_
        SECONDS) or its raw length is momentarily ~0 (an input-polling
        gap mid-blip, not a real stop) - this HOLDS self._smoothed_
        direction_angle exactly where it already was, rather than
        resetting it to None or re-deriving a target from move_direction
        (whose own _facing_angle_degrees falls back to a MEANINGLESS 0.0
        for a near-zero vector). This matters even after a real release,
        not just mid-blip: _LocomotionBlendSpace's speed axis still
        blends the walk tier's weight continuously down toward idle as
        speed decays (see _ring_weights) - resetting direction to None
        the instant the key lifts made that tier fall back to its PLAIN
        (forward/N) clip while it still had real weight, popping visibly
        through a forward-walk pose before settling to idle even when
        the player had been strafing sideways the whole time. Since a
        stale held angle only ever gets used again once the player
        actually starts moving in some direction (at which point it
        eases toward the NEW real target below, same as always), holding
        it indefinitely instead of clearing to None costs nothing - the
        first-ever movement's own snap-vs-ease distinction this used to
        preserve was a much smaller concern than this pop."""
        if move_direction is None:
            self._smoothed_direction_angle = None
            return None

        if not has_input or glm.length(move_direction) <= 1e-4:
            return self._smoothed_direction_angle

        target = _facing_angle_degrees(move_direction, yaw_degrees)
        if self._smoothed_direction_angle is None:
            self._smoothed_direction_angle = target
        else:
            self._smoothed_direction_angle = _ease_angle_degrees(
                self._smoothed_direction_angle, target, dt, _DIRECTION_EASE_SECONDS
            )
        return self._smoothed_direction_angle

    def _confirmed_airborne(self, is_grounded, dt):
        """Returns whether the player should be treated as genuinely
        airborne for jump_animation's own top-level state edge - see
        _JUMP_CONFIRM_SECONDS for why this debounces is_grounded instead
        of reacting to it directly. Resets the debounce timer to 0.0 the
        instant is_grounded reads True (so a genuine LANDING is still
        recognized immediately, via update()'s own separate is_grounded-
        and-top_state=='jump' check - only a NOT-grounded reading needs
        to persist a moment before being believed)."""
        if is_grounded:
            self._airborne_confirm_elapsed = 0.0
            return False
        self._airborne_confirm_elapsed += dt
        return self._airborne_confirm_elapsed >= _JUMP_CONFIRM_SECONDS

    def _confirmed_has_input(self, move_direction, dt):
        """Returns whether the player should be treated as having real
        movement input this frame, for the locomotion blend space's own
        idle-vs-moving decision - see _NO_INPUT_CONFIRM_SECONDS for why
        this debounces move_direction instead of reacting to a single
        frame's raw length directly. move_direction=None (no signal
        available at all - see update()'s own docstring) is a completely
        separate case handled by the caller, not this method - resets the
        debounce timer to 0.0 and returns... this method is only ever
        called when move_direction is not None, so that distinction lives
        in update() itself. Resets the debounce timer to 0.0 the instant a
        real (non-near-zero) move_direction is read (so ACTUALLY starting
        to move is recognized immediately - only a near-zero reading
        needs to persist a moment before being believed)."""
        if glm.length(move_direction) > 1e-4:
            self._no_input_elapsed = 0.0
            return True
        self._no_input_elapsed += dt
        return self._no_input_elapsed < _NO_INPUT_CONFIRM_SECONDS

    def update(self, dt, position, yaw_degrees, speed, is_crouched=False, is_grounded=True,
               is_sprinting=None, move_direction=None, just_jumped=False):
        """Call once per frame (or per network update, for a remote
        player) to sync this model's transform and animation state.

        position: an already FEET-LEVEL (ground-contact) position, glm.
        vec3-able - converting from whatever the caller actually tracks
        (a CharacterController's hull-CENTER for the local player, an
        assumed eye-to-feet offset off a synced camera position for a
        remote player) is the CALLER's job, not this class's - the two
        conversions differ per caller and PlayerModel has no opinion on
        which kind of player it's attached to. Only this class's own
        fixed feet_offset (an authoring correction, not a gameplay one)
        is applied on top here.

        yaw_degrees: camera-convention yaw in degrees (camera.py's
        front.x=cos(yaw)cos(pitch), front.z=sin(yaw)cos(pitch)) - NOT
        already-converted model-space radians.

        speed: horizontal movement speed in m/s, the primary blend-space
        parameter (see _LocomotionBlendSpace) - the caller computes this
        however makes sense for it (CharacterController.velocity's
        horizontal magnitude for the local player, a position-delta/
        elapsed-time estimate for a remote player driven by network
        updates). This is what decides idle-vs-moving (and, continuously,
        walk-vs-run when is_sprinting is None below) - not fed into the
        blend space directly, though: first low-pass filtered into self.
        _smoothed_speed (see _SPEED_EASE_SECONDS's own docstring) to
        damp a brief physics-tick velocity spike/dip (e.g. collision
        response bouncing the resolved velocity while pushed into a
        wall) that has nothing to do with the player's actual intent. An
        earlier version of this went further and derived idle/walk/run
        from move_direction/is_sprinting instead of speed entirely, to
        dodge is_grounded flickering during exactly that same kind of
        wall collision popping the jump state in and resetting the whole
        blend space's phase (see set_skeletal_locomotion's own
        docstring) - but that was masking a real bug rather than fixing
        it; now that the actual cause is fixed at its source (see
        _JUMP_CONFIRM_SECONDS and this method's own just_jumped param),
        speed is trustworthy again and idle/walk/run went back to
        depending on it, same as any ordinary Unreal-style locomotion
        blend space.

        move_direction: optional world-space horizontal wish/move
        direction (glm.vec3-able, Y ignored, magnitude doesn't matter -
        only its direction is used) - required for directional_clips
        (see __init__) to have any effect; None (the default) disables
        directional blending entirely and every tier uses its plain clip
        exactly as before this param existed. The caller computes this
        however makes sense for it - the local player passes the same raw
        WASD-derived direction vector it already builds for
        CharacterController.set_move_direction (see app.py); a remote
        player with no discrete input to read simply omits it.

        This raw vector's own facing-relative angle is NOT fed straight
        into the blend space's direction axis - a WASD-driven vector only
        ever points at one of a small fixed set of exact directions
        (whatever key combination is currently held), each one landing
        dead-on one of directional_clips' own 8 sample angles (both are
        defined relative to the SAME camera yaw), so using it directly
        would only ever produce a hard 100%/0% jump between two exact
        samples the instant the held keys change - never an in-between
        angle to actually blend across. self._smoothed_direction_angle
        eases toward this raw angle instead (see _ease_angle_degrees),
        over _DIRECTION_EASE_SECONDS, the same fix _sprint_blend applies
        to the speed axis's own discrete is_sprinting input - so a snap
        turn (e.g. releasing W and pressing D) now actually sweeps the
        direction weights through the intermediate angles between the two
        samples instead of popping between them in one frame.

        is_sprinting: None (the default) keeps the original purely
        speed-driven blend across the whole grid (matches every caller
        before this param existed, and still what a remote player uses -
        a synced position stream carries no discrete sprint-key bit to
        read). Pass True/False instead to pick walk-vs-run by actual
        input intent rather than by how fast the character happens to
        currently be moving - done here, NOT inside _LocomotionBlendSpace
        itself (which only ever sees a plain continuous speed number - see
        its own docstring), by smoothly easing an internal self._sprint_
        blend value (0..1) toward the target over _SPRINT_EASE_SECONDS
        each time this is called, then mapping that eased value onto a
        synthetic point on the SAME speed axis (walk's own threshold at
        0, the fastest tier's own threshold at 1) once actually moving -
        so a sprint key press/release still eases smoothly into the new
        tier instead of the discrete input snapping the blend weight
        outright the way it would if True/False were fed straight in as
        0%/100%.

        is_crouched/is_grounded: see jump_animation/crouch_animation's
        own __init__ docstrings for exactly what these gate. dt: used for
        the is_sprinting easing above (needs a real per-frame delta, not
        a placeholder, or that ease will run at the wrong rate) - actual
        animation time advance happens inside Scene.update()'s own
        per-frame bone recompute, not here.

        just_jumped: True for exactly the one update() call covering the
        frame a queued jump actually executed (the local player's own
        CharacterController.pop_jumped(), a one-shot consumed event - see
        its own docstring - forwarded straight through by app.py; a
        remote player/caller with no such signal simply omits this,
        matching every caller before this param existed). Enters the
        "jump" top-level state IMMEDIATELY when True, bypassing
        _confirmed_airborne's own debounce entirely - that debounce
        exists to reject is_grounded flickering False for a tick or two
        with NO jump involved at all (see _JUMP_CONFIRM_SECONDS' own
        docstring - pressed against a wall), which is an important
        distinction from an ACTUAL jump input, which should play its
        takeoff pose the instant it happens, not up to _JUMP_CONFIRM_
        SECONDS later. is_grounded's own debounced reading is still used
        as a fallback for entering "jump" without this flag (e.g. walking
        off a ledge with no jump key involved at all needs SOME way to
        detect genuine airborne-ness), so the two together cover both a
        deliberate jump (instant) and an incidental fall (debounced)."""
        if self.obj is None:
            return

        feet_pos = glm.vec3(position)
        feet_pos.y += self._feet_offset
        self.obj["position"] = feet_pos

        # Camera yaw (degrees) -> model rotation.y (radians about +Y,
        # scene_base.py's _get_model_matrix convention). Derived, not
        # guessed: camera.py's front vector is
        # (cos(yaw), sin(pitch), sin(yaw)) i.e. camera-forward sweeps
        # +X -> +Z as yaw increases. _get_model_matrix's
        # glm.rotate(model, rotation.y, (0,1,0)) maps a model's local +Z
        # (glTF's standard rest-forward axis) to world
        # (sin(rotation.y), 0, cos(rotation.y)). Solving so the model's
        # local +Z tracks the camera's flat-forward direction
        # (cos(yaw), 0, sin(yaw)) gives rotation.y = radians(90 - yaw).
        # If a specific model still visually faces the wrong way (or is
        # mirrored front-to-back) at forward_offset_degrees=0, that
        # model's rest pose isn't the standard +Z-forward convention -
        # fix it via forward_offset_degrees (try 180 first), not by
        # changing this formula.
        self.obj["rotation"].y = glm.radians(90.0) - glm.radians(yaw_degrees) + self._forward_offset_radians

        # Computed unconditionally, every frame, regardless of top_state -
        # locomotion's own blend space needs these below, crouch_
        # animation's blend space needs the SAME two values (see its own
        # __init__ docstring - it stays continuously reweighted the whole
        # time crouched, not just resolved once on entry), and jump_
        # animation's one-shot pick uses the plain (non-eased, non-
        # smoothed) speed/move_direction directly instead (see below)
        # since a discrete one-time resolution has no "wobble over time"
        # to smooth in the first place. Computing both here also means
        # self._sprint_blend/self._smoothed_direction_angle/self.
        # _smoothed_speed keep easing continuously even while jump/crouch
        # briefly own the animation, instead of freezing mid-ease and
        # jumping the remaining distance the next time locomotion's blend
        # space reads them.
        self._smoothed_speed = _ease_value(self._smoothed_speed, speed, dt, _SPEED_EASE_SECONDS)
        effective_speed = self._compute_effective_speed(self._smoothed_speed, dt, is_sprinting)
        # Debounced so the direction axis doesn't snap toward a
        # meaningless fallback angle on a single-frame move_direction blip
        # (see _confirmed_has_input/_NO_INPUT_CONFIRM_SECONDS) - only
        # relevant to direction now; the speed axis went back to being
        # purely velocity-driven (see _compute_effective_speed's own
        # docstring).
        has_input = move_direction is not None and self._confirmed_has_input(move_direction, dt)
        direction_angle = self._compute_smoothed_direction_angle(move_direction, yaw_degrees, dt, has_input)

        entering_locomotion = False
        entry_blend_kwargs = self._animation_blend_kwargs

        if self._jump_blend_space is not None:
            # _confirmed_airborne (not a plain "not is_grounded" check) -
            # see _JUMP_CONFIRM_SECONDS for why: CharacterController's own
            # ground sweep can flicker is_grounded False for a tick or two
            # while pressed into a wall, and reacting to that directly
            # would briefly enter "jump" purely from the glitch, popping
            # to the takeoff pose and immediately crossfading back with no
            # actual jump happening. Called unconditionally, every frame,
            # so the debounce timer keeps advancing/resetting correctly
            # regardless of which top_state is currently active. just_
            # jumped (see update()'s own docstring) bypasses that debounce
            # entirely for an ACTUAL jump input - _confirmed_airborne
            # alone remains as the fallback for entering "jump" from an
            # incidental fall (walking off a ledge) with no jump key
            # involved, which has no equivalent one-shot event to read.
            confirmed_airborne = self._confirmed_airborne(is_grounded, dt)
            if (just_jumped or confirmed_airborne) and self._top_state != "jump":
                # Just left the ground - resolve this takeoff's jump pose
                # ONCE, from the plain current speed/direction (see
                # jump_animation's own __init__ docstring for why this is
                # a single pick rather than a continuous reblend), taking
                # whichever candidate _LocomotionBlendSpace weighted
                # highest rather than actually blending several
                # simultaneously - Scene's weighted clip path has no
                # equivalent of loop=False's "clamp and hold the last
                # frame" yet, which a held takeoff pose needs. Overrides
                # both bodies, loop=False so Scene.update() clamps it at
                # its own duration and holds the last frame instead of
                # looping the takeoff motion while still airborne.
                self._top_state = "jump"
                if DEBUG_LOCOMOTION:
                    reason = "just_jumped" if just_jumped else f"confirmed_airborne ({self._airborne_confirm_elapsed:.3f}s)"
                    _debug_log(f"[locomotion] t={time.monotonic():.3f} ENTER jump ({reason}, speed={speed:.3f})")
                raw_angle = _facing_angle_degrees(move_direction, yaw_degrees) if move_direction is not None else None
                jump_weights = self._jump_blend_space.compute_weights(speed, raw_angle)
                jump_clip = max(jump_weights, key=lambda cw: cw[1])[0] if jump_weights else None
                if jump_clip is not None and jump_clip != self.obj["animation"]:
                    self._scene.set_skeletal_animation(
                        self.obj, jump_clip, loop=False, **self._animation_blend_kwargs
                    )
                if (
                    jump_clip is not None
                    and self._upper_auto_driven and self._upper_override is None
                    and jump_clip != self.obj["upper_animation"]
                ):
                    self._scene.set_skeletal_upper_animation(
                        self.obj, jump_clip, loop=False, **self._animation_blend_kwargs
                    )
            elif is_grounded and self._top_state == "jump":
                # Landed - resume the locomotion blend space, crossfaded
                # in from the jump clip's held last frame.
                if DEBUG_LOCOMOTION:
                    _debug_log(f"[locomotion] t={time.monotonic():.3f} RESET (landed from jump) - locomotion_phase -> 0")
                self._top_state = "locomotion"
                entering_locomotion = True
                entry_blend_kwargs = self._animation_blend_kwargs

        if self._top_state == "jump":
            # Held on the jump clip's last frame - skip locomotion
            # entirely until landed (handled above), so a mid-air speed
            # change never overwrites it.
            return

        if self._crouch_blend_space is not None:
            if is_crouched and self._top_state != "crouch":
                # Just crouched - resolve THIS frame's crouch weights and
                # enter the blend space, same full-body reasoning as
                # jump_animation above (one weighted list drives both
                # bodies - see crouch_animation's own __init__ docstring
                # for why there's no separate upper-body crouch table).
                self._top_state = "crouch"
                if DEBUG_LOCOMOTION:
                    _debug_log(f"[locomotion] t={time.monotonic():.3f} ENTER crouch")
                crouch_weights = self._crouch_blend_space.compute_weights(effective_speed, direction_angle)
                crouch_upper_weights = (
                    crouch_weights if self._upper_auto_driven and self._upper_override is None else None
                )
                self._scene.set_skeletal_locomotion(
                    self.obj, crouch_weights, crouch_upper_weights, **self._crouch_blend_kwargs
                )
            elif not is_crouched and self._top_state == "crouch":
                # Stood back up - resume the locomotion blend space,
                # crossfaded in using crouch_blend_duration (standing up
                # is just as big a pose change as crouching down was, so
                # it deserves the same longer, more visible ease).
                if DEBUG_LOCOMOTION:
                    _debug_log(f"[locomotion] t={time.monotonic():.3f} RESET (stood up from crouch) - locomotion_phase -> 0")
                self._top_state = "locomotion"
                entering_locomotion = True
                entry_blend_kwargs = self._crouch_blend_kwargs

        if self._top_state == "crouch":
            # Still crouched - re-evaluate the crouch blend space fresh
            # every frame, exactly like locomotion's own below, so
            # crouch-walking (if crouch_animation has more than one tier)
            # actually blends instead of freezing on whatever pose was
            # showing the instant crouch was entered.
            crouch_weights = self._crouch_blend_space.compute_weights(effective_speed, direction_angle)
            self.obj["locomotion_weights"] = crouch_weights
            if self._upper_auto_driven and self._upper_override is None:
                self.obj["upper_locomotion_weights"] = crouch_weights
            return

        # top_state == "locomotion": re-evaluate the blend space fresh
        # every single frame (see _LocomotionBlendSpace.compute_weights) -
        # there is no cached "current state" to compare against and no
        # dwell timer to satisfy, since a continuously-reweighted blend
        # has nothing analogous to the pop a discrete state switch used
        # to risk.
        lower_weights = self._lower_blend_space.compute_weights(effective_speed, direction_angle)
        upper_weights = (
            self._upper_blend_space.compute_weights(effective_speed, direction_angle)
            if self._upper_auto_driven and self._upper_override is None else None
        )

        if DEBUG_LOCOMOTION:
            names = tuple(sorted(name for name, _ in lower_weights))
            if names != self._debug_last_clip_names:
                self._debug_last_clip_names = names
                phase = self.obj.get("locomotion_phase") if self.obj is not None else None
                phase_str = f"{phase:.3f}" if phase is not None else "n/a"
                line = (
                    f"[locomotion] t={time.monotonic():.3f} top_state={self._top_state} "
                    f"speed={speed:.3f} smoothed={self._smoothed_speed:.3f} "
                    f"effective={effective_speed:.3f} sprint_blend={self._sprint_blend:.3f} "
                    f"angle={direction_angle} phase={phase_str} "
                    f"weights={[(n, round(w, 3)) for n, w in lower_weights]}"
                )
                _debug_log(line)

        if entering_locomotion:
            self._scene.set_skeletal_locomotion(self.obj, lower_weights, upper_weights, **entry_blend_kwargs)
        else:
            self.obj["locomotion_weights"] = lower_weights
            if self._upper_auto_driven and self._upper_override is None:
                self.obj["upper_locomotion_weights"] = upper_weights

    def set_upper_animation(self, animation_name):
        """Switches the upper-body clip (see __init__'s
        upper_body_root_joints) to animation_name, restarting its time
        from 0 - independent of update()'s own locomotion blending, which
        never touches this once called (until the NEXT frame the
        locomotion blend space is auto-driving the upper body, if it is -
        see upper_states' own docstring: auto-driving overwrites a manual
        call like this on its very next update()). Only meaningful if
        this PlayerModel was constructed with upper_body_root_joints; a
        no-op (not an error) otherwise, or if the underlying model failed
        to load (self.obj is None)."""
        if self.obj is None:
            return
        self.obj["upper_locomotion_weights"] = None
        self._scene.set_skeletal_upper_animation(self.obj, animation_name, **self._animation_blend_kwargs)

    def set_upper_override(self, animation_name):
        """Forces the upper body to animation_name and LOCKS it there,
        completely bypassing update()'s own upper-body driving -
        locomotion (upper_states), jump_animation, and crouch_animation
        all normally touch the upper body too, but none of them will
        while an override is active - until clear_upper_override() is
        called. Unlike the plain set_upper_animation() hook above (which
        upper_states' own auto-driving would silently overwrite on the
        very next frame if upper_auto_driven), this actually stays
        exactly as set regardless of what the player does - meant for
        something like a manual weapon-idle pose swap that should hold no
        matter how the player moves/jumps/crouches in the meantime. A
        no-op if the model failed to load (self.obj is None).

        If this PlayerModel was constructed with
        override_upper_body_root_joints, the upper-body joint mask
        switches to that (wider) split for as long as the override is
        active - see its own __init__ docstring for why an override
        commonly needs a different split than locomotion does."""
        self._upper_override = animation_name
        if self.obj is not None:
            self.obj["upper_locomotion_weights"] = None
            if self._override_upper_body_root_joints is not None:
                self._scene.set_skeletal_upper_joint_mask(self.obj, self._override_upper_body_root_joints)
            self._scene.set_skeletal_upper_animation(self.obj, animation_name, **self._animation_blend_kwargs)

    def clear_upper_override(self):
        """Releases a set_upper_override() lock - the very next update()
        call resumes whatever would normally be driving the upper body
        (the locomotion blend space if upper_auto_driven - re-evaluated
        completely fresh every frame regardless, so there's no cached
        state to invalidate here the way the old discrete switcher
        needed - jump/crouch overrides if applicable, or nothing at all
        if the upper body was purely externally driven to begin with).
        Also restores the joint mask to upper_body_root_joints if
        set_upper_override had switched it to override_upper_body_root_
        joints."""
        self._upper_override = None
        if self.obj is not None and self._override_upper_body_root_joints is not None:
            self._scene.set_skeletal_upper_joint_mask(self.obj, self._upper_body_root_joints)

    def set_upper_rotation_offset(self, degrees, blend_duration=None):
        """Changes upper_rotation_offset_degrees (see __init__) at
        runtime - e.g. dialing in the right correction interactively for
        a pose authored on a rig whose bind orientation doesn't quite
        match this skeleton's own, rather than guessing a constant up
        front. degrees: either a single (x, y, z) Euler tuple (applied
        to every root joint) or a dict {joint_name: (x, y, z)} for an
        independent per-side correction; (0,0,0) removes the correction
        entirely. Applies regardless of which upper-body pose is
        currently playing (locomotion blend space, jump/crouch override,
        or a set_upper_override lock) - a no-op if the model failed to
        load or wasn't constructed with upper_body_root_joints set.

        Crossfades from whatever correction was active a moment ago
        (see Scene.set_skeletal_upper_rotation_offset's own docstring)
        rather than snapping to the new one - blend_duration controls
        how long that takes; None (the default) uses this PlayerModel's
        own animation_blend_duration (the same crossfade length every
        top-level state change here already uses), so a pose change that
        also changes the rotation offset eases both in together by
        default."""
        if self.obj is None:
            return
        kwargs = self._animation_blend_kwargs if blend_duration is None else {"blend_duration": float(blend_duration)}
        self._scene.set_skeletal_upper_rotation_offset(self.obj, degrees, **kwargs)

    def set_visible_in_color(self, visible):
        """Switches whether this model actually draws in the normal
        color pass (see add_skeletal's own visible_in_color) - e.g.
        toggling third-person mode on/off, where the local player's body
        should suddenly be visible instead of shadow-only, or vice versa
        going back to first person. cast_shadow is untouched either way
        (this model still casts a shadow in both modes) - a no-op if the
        underlying model failed to load (self.obj is None)."""
        if self.obj is None:
            return
        self.obj["visible_in_color"] = bool(visible)

    def destroy(self):
        """Removes this model from the scene. No Scene.remove_skeletal
        helper exists (nothing in this codebase currently removes a
        skeletal object once added), so this is the minimal safe
        teardown - drop it from the list the render/update loops
        iterate. GPU resource release (VAOs/VBOs/textures) is a
        pre-existing gap in Scene's own API this doesn't attempt to
        fix."""
        if self.obj is not None and self.obj in self._scene.skeletal_objects:
            self._scene.skeletal_objects.remove(self.obj)
        self.obj = None
