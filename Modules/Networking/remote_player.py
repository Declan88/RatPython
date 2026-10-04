"""Visual + physical representation of one other connected peer.

Drives a generic PlayerModel (Modules/Player/player_model.py - the same
class the LOCAL player's shadow-only body uses, just fully visible here)
plus a query-only hitbox from the state network_manager.py's handle_data()
forwards from the wire. The sender transmits its real FEET position and
locomotion state (see NetworkManager.set_local_state), so nothing here has
to guess a body position from a camera or estimate speed from position
deltas.

Motion is smoothed by snapshot interpolation: packets arrive ~30 times a
second and irregularly, so each is stored with its arrival time and the
model is drawn INTERP_DELAY seconds in the past, blending between the two
snapshots that straddle that moment. That costs a fixed ~100ms of visual
latency in exchange for smooth movement instead of a step per packet.
"""

import random
import time

import glm

from Modules.Physics.physics_world import CollisionGroup
from Modules.Player.player_model import PlayerModel
from Modules.Player.rat_colors import RAT_TINT_MASK_PATH, decode_color
from Modules.Weapons import create_weapon
from Modules.Weapons.damage_classes import DISSOLVE, get_damage_class

# How far behind real time remote players are drawn - see module docstring.
# Comfortably more than one packet interval (1/30s) so there are almost
# always two snapshots to blend between, even with a dropped packet.
INTERP_DELAY = 0.1
_MAX_SNAPSHOTS = 40
# A peer that goes quiet for this long is treated as standing still rather
# than freezing mid-stride in whatever locomotion pose its last packet had.
_STALE_SECONDS = 1.0
# How often a remote player's animation-state logic runs (see RemotePlayer.update).
_MODEL_INTERVAL = 1.0 / 100.0

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
    "Assets/Animations/Poses/Rifle/RifleJump.glb": ({"New": "rifle_jump"}, 1.0),
    "Assets/Animations/Poses/Rifle/RifleCrouch.glb": ({"New": "rifle_crouch"}, 1.0),
    "Assets/Animations/Poses/Rifle/RifleWalkS.glb": ({"New": "rifle_walk_s"}, 1.0),
    "Assets/Animations/Poses/Rifle/RifleWalkE.glb": ({"New": "rifle_walk_e"}, 1.0),
    "Assets/Animations/Poses/Rifle/RifleWalkW.glb": ({"New": "rifle_walk_w"}, _RAT_ANIM_TIME_SCALE),
    "Assets/Animations/Poses/Rifle/RifleWalkNE.glb": ({"New": "rifle_walk_ne"}, _RAT_ANIM_TIME_SCALE),
    "Assets/Animations/Poses/Rifle/RifleWalkNW.glb": ({"New": "rifle_walk_nw"}, _RAT_ANIM_TIME_SCALE),
    "Assets/Animations/Poses/Rifle/RifleWalkSE.glb": ({"New": "rifle_walk_se"}, _RAT_ANIM_TIME_SCALE),
    "Assets/Animations/Poses/Rifle/RifleWalkSW.glb": ({"New": "rifle_walk_sw"}, _RAT_ANIM_TIME_SCALE),
}
# Same facing-relative walk clips the local player uses (see app.py).
_DIRECTIONAL_CLIPS = {
    "walk": {
        "N": "rifle_walk", "S": "rifle_walk_s", "E": "rifle_walk_e", "W": "rifle_walk_w",
        "NE": "rifle_walk_ne", "NW": "rifle_walk_nw", "SE": "rifle_walk_se", "SW": "rifle_walk_sw",
    },
}
# A static box approximation of the player capsule - placeholder
# "groundwork" per this feature's scope (no damage/weapon system exists
# yet to actually query it). A tighter capsule-shaped hitbox could be
# added to PhysicsWorld later if a real damage system wants one.
_DEFAULT_HITBOX_HALF_EXTENTS = (0.4, 0.9, 0.4)
# rat.glb's measured eye level above its feet - the hitbox spans feet to eye.
_BODY_HEIGHT = 1.39225
# The head's box in the rat's own frame (metres from the feet; +z is the way it faces): where the
# head mesh sits (measured off Gibs.glb's head), a little generous so a shot that grazes it counts.
_HEAD_HALF_EXTENTS = (0.20, 0.22, 0.34)
_HEAD_CENTER = (0.0, 1.42, 0.09)


class RemotePlayer:
    # How long a dissolve (Modules/Weapons/damage_classes.py's own Zap) takes to fully clear
    # the body, seconds - see _set_dead/update's own draining of self._dissolving.
    DISSOLVE_SECONDS = 1.2
    # Ambient dust (spark/glow) fires on this fixed interval - Source's own equivalent
    # (C_EntityDissolve::DrawModel) actually runs every single rendered frame, rolling per
    # hitbox (roughly a dozen on a human ragdoll) whether to spawn one - a fixed, fairly quick
    # interval from a single point is this engine's simpler stand-in for that same "constant
    # fine crackle" density without per-hitbox machinery.
    DISSOLVE_SPARK_INTERVAL = 0.025
    # Fired from this many different random points on the body each interval (same "several
    # points at once, not one" reasoning DISSOLVE_ARC_BURST_COUNT below already uses) - a
    # single point every 0.025s reads as sparse, isolated pinpricks; several at once, that
    # often, is what actually reads as a dense drip of motes coming off the whole body.
    DISSOLVE_SPARK_BURST_COUNT = 4
    # The tesla-arc bursts (DISSOLVE_ARC_BURST_COUNT lgtning streaks at once, see update()'s
    # own _dissolve_arc_interval) are the actual "electricity jumping off them" read (see
    # dissolve_sparks.py's own docstring) - Source's own DoSparks fires 3 beams per call, on an
    # interval that itself SPEEDS UP over the effect's life: SimpleSplineRemapVal(dt, 0,
    # m_flFadeOutStart, 2*TICK_INTERVAL, 0.4) - slow (~0.4s) right after the effect starts,
    # ramping down to rapid (~0.03s) as it approaches the point the model finishes fading. The
    # two ends of that same ramp, reused below.
    DISSOLVE_ARC_INTERVAL_START = 0.4
    DISSOLVE_ARC_INTERVAL_END = 0.03
    DISSOLVE_ARC_BURST_COUNT = 3
    # How fast the body drifts upward while dissolving, m/s - stands in for Source's own
    # C_EntityDissolve::Simulate, which attaches a zero-gravity IPhysicsMotionController to a
    # dissolving ragdoll (linear.z -= -1.02 * GetCurrentGravity() cancels gravity outright),
    # so any residual death velocity just carries the body slowly upward instead of settling -
    # the familiar "corpse gently rises and drifts" look. This project's remote bodies aren't
    # real physics ragdolls, so a fixed gentle rise is a simpler stand-in for the same read.
    DISSOLVE_RISE_SPEED = 0.4

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
            scale=scale, time_scale=_RAT_ANIM_TIME_SCALE, tint_mask_path=RAT_TINT_MASK_PATH,
            directional_clips=_DIRECTIONAL_CLIPS,
            jump_animation="rifle_jump", crouch_animation="rifle_crouch",
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
            # Same emote catalogue file app.py merges onto the local player's own
            # model (see its own load_additional_animations call site) - needed
            # here too so play_emote (triggered by an incoming "em" packet - see
            # NetworkManager._receive_emote) can find the clip by name. The
            # return value isn't needed on this side - a remote player only ever
            # has to validate/play a name the network tells it, which play_emote's
            # own animations.get(name) guard already covers.
            scene.load_additional_animations(self.model.obj, "Assets/Animations/Emotes/Emotes.glb")

        # The weapon in their hand is whichever their packets name (see _apply_weapon);
        # each one they've shown is kept (loaded once) so switching back is instant.
        # The world model goes in the hand; the arm pose only applies to a model with
        # an upper-body split, which this one doesn't have - see WeaponsBase.equip_player.
        self._weapons = {}
        self._weapon_wanted = None   # weapon id from the latest packet
        self._weapon_id = None       # id of the weapon in their hand now
        self.weapon = None
        self._apply_weapon(self.DEFAULT_WEAPON)

        self._hitbox = scene.physics.add_hitbox(hitbox_half_extents, owner=steam_id)
        # Same owner, own group: see CollisionGroup.HEAD and WeaponsBase.fire.
        self._head_hitbox = scene.physics.add_hitbox(
            _HEAD_HALF_EXTENTS, owner=steam_id, collision_mask=CollisionGroup.HEAD)

        self._snapshots = []      # [(arrival_time, feet_pos, yaw_degrees)], oldest first
        self._state = {}          # latest non-interpolated locomotion state
        self._last_packet = 0.0
        self._jump_seen = None    # last jump counter value seen
        self._pending_jump = False
        self.name = ""            # Steam persona name (packets carry it)
        self._hat_wanted = None   # from the latest packet ('' = bare-headed)
        self._hat_applied = None
        self._shot_seen = None     # last shot counter value seen
        self._pending_shots = 0
        # Set alongside _pending_shots whenever the shot counter increases (see
        # receive_state) - consumed the same one-shot way _pending_jump is
        # (update() passes it into model.update() as just_shot, then clears it),
        # so a shot cancels an in-progress emote on THIS peer's own screen with
        # no extra network message - see PlayerModel.update()'s own just_shot.
        self._pending_shot_cancel = False
        self._pending_tracers = []  # end points of shots to draw a tracer for
        self.on_tracer = None       # callback(start, end), set by NetworkManager
        self._footstep_seen = None   # last footstep counter value seen
        self._pending_footsteps = 0
        self._footstep_material = None
        self._footstep_volume = 1.0
        self.on_footstep = None     # callback(material, position, volume), set by NetworkManager
        self._voice_position = glm.vec3(0.0)  # where this player's shots sound from
        self._color_wanted = None  # fur color from the latest packet (None = the rat's own)
        self._color_applied = None
        # Deaths travel as a counter like jumps and shots (so a lost packet can't drop
        # one) plus an alive flag: a counter increase = they died just now (their body
        # bursts into gibs, see on_death), and while the flag stays down their model and
        # hitbox are gone.
        self.dead = False
        self._model_debt = 0.0
        self._model_never_run = True
        self._hitbox_key = None
        self.on_death = None        # callback(feet_position, velocity, fur colour, damage_class), set by NetworkManager
        self._death_seen = None
        self._kill_seen = None
        self.kills = 0    # kills this player has been credited for (see NetworkManager.local_kills)
        self.deaths = 0   # times this player has died
        self._pending_death = False
        self._alive_wanted = True
        self._dissolving = False    # currently animating a Zap death's dissolve - see _set_dead/update
        self._dissolve_t = 0.0
        self._dissolve_spark_t = 0.0
        self._dissolve_arc_t = 0.0
        self._dissolve_rise = 0.0   # accumulated upward drift so far - see update()'s own DISSOLVE_RISE_SPEED
        self.on_dissolve_spark = None   # callback(position), set by NetworkManager - see update()
        self.on_dissolve_arc = None     # callback(position), set by NetworkManager - see update()

    def play_emote(self, name):
        """Called by NetworkManager._receive_emote the instant this peer's own
        "em" packet arrives - a trivial forwarder, since PlayerModel.play_emote
        is fully shared between local and remote use (the "switch to third
        person"/"notify_emote" parts are local-only, app.py-level concerns, not
        part of PlayerModel at all)."""
        if self.model.obj is None:
            return
        self.model.play_emote(name)

    def receive_state(self, state):
        """state: the decoded packet dict from NetworkManager._broadcast_
        transform - "p" feet position, "y" yaw degrees, "v" horizontal
        speed, "c" crouched, "g" grounded, "s" sprinting, "d" (x, z) move
        direction, "j" jump counter."""
        now = time.perf_counter()
        pos = glm.vec3(*state["p"])
        self._snapshots.append((now, pos, float(state["y"])))
        del self._snapshots[:-_MAX_SNAPSHOTS]
        self._state = state
        self._last_packet = now

        # A jump is a one-shot event but packets can be lost, so it travels
        # as a monotonically increasing counter: any increase is a jump.
        jumps = int(state.get("j", 0))
        if self._jump_seen is not None and jumps > self._jump_seen:
            self._pending_jump = True
        self._jump_seen = jumps
        self._hat_wanted = state.get("h") or None
        self._weapon_wanted = state.get("wp") or self._weapon_wanted
        self._color_wanted = decode_color(state.get("k"))
        # Shots travel as a counter like jumps: any increase is that many shots
        # (capped so a long stall can't replay a burst of gunfire at once).
        shots = int(state.get("f", 0))
        if self._shot_seen is not None and shots > self._shot_seen:
            self._pending_shots = min(self._pending_shots + shots - self._shot_seen, 3)
            self._pending_shot_cancel = True
            # The packet carries the end points of their last few shots; the
            # newest ones are the shots just fired.
            ends = state.get("e") or []
            fresh = min(shots - self._shot_seen, len(ends))
            if fresh > 0:
                self._pending_tracers.extend(ends[-fresh:])
                del self._pending_tracers[:-3]
        self._shot_seen = shots
        # Footsteps travel as a counter like jumps/shots (see NetworkManager.notify_footstep's
        # own docstring on why) - just the newest one's (material, volume), no history needed.
        footsteps = int(state.get("fs", 0))
        if self._footstep_seen is not None and footsteps > self._footstep_seen:
            self._pending_footsteps = min(self._pending_footsteps + footsteps - self._footstep_seen, 3)
            self._footstep_material = state.get("fm") or None
            self._footstep_volume = float(state.get("fv", 1.0))
        self._footstep_seen = footsteps
        deaths = int(state.get("x", 0))
        if self._death_seen is not None and deaths > self._death_seen:
            self._pending_death = True
        self._death_seen = deaths
        self.deaths = deaths
        kills = int(state.get("ki", 0))
        self._kill_seen = kills
        self.kills = kills
        self._alive_wanted = bool(state.get("a", 1))
        if state.get("n"):
            self.name = str(state["n"])[:32]

    @property
    def body_center(self):
        """Roughly the middle of the body, in world space."""
        return glm.vec3(self._voice_position)

    def _sample(self, render_time):
        """(feet_pos, yaw) at render_time, interpolated between the two
        snapshots around it (clamped to the oldest/newest)."""
        snaps = self._snapshots
        if render_time <= snaps[0][0]:
            return snaps[0][1], snaps[0][2]
        for i in range(len(snaps) - 1):
            t0, p0, y0 = snaps[i]
            t1, p1, y1 = snaps[i + 1]
            if t0 <= render_time <= t1:
                f = (render_time - t0) / max(t1 - t0, 1e-6)
                yaw = y0 + ((y1 - y0 + 180.0) % 360.0 - 180.0) * f  # shortest way round
                return glm.mix(p0, p1, f), yaw
        return snaps[-1][1], snaps[-1][2]

    def update(self, dt):
        """Call once per frame (NetworkManager.update does)."""
        if not self._snapshots:
            return
        now = time.perf_counter()
        feet_pos, yaw = self._sample(now - INTERP_DELAY)
        self._voice_position = feet_pos + glm.vec3(0.0, _BODY_HEIGHT * 0.8, 0.0)
        s = self._state
        stale = now - self._last_packet > _STALE_SECONDS
        move = s.get("d", (0.0, 0.0))
        # The body follows the interpolated position every frame, but the animation-state logic
        # (blend weights, facing ease, jump/crouch poses) only runs about 100 times a second, with the
        # time that passed since it last did - at hundreds of fps that's most of a player's cost.
        self._model_debt += dt
        if self._model_debt + 0.5 * dt >= _MODEL_INTERVAL or self._model_never_run:
            self._model_never_run = False
            self.model.update(
                self._model_debt, feet_pos, yaw, 0.0 if stale else float(s.get("v", 0.0)),
                is_crouched=bool(s.get("c", False)),
                is_grounded=bool(s.get("g", True)),
                is_sprinting=bool(s.get("s", False)),
                move_direction=None if stale else glm.vec3(move[0], 0.0, move[1]),
                just_jumped=self._pending_jump,
                just_shot=self._pending_shot_cancel,
            )
            self._model_debt = 0.0
            self._pending_jump = False
            self._pending_shot_cancel = False
        else:
            self.model.move_to(feet_pos)
        if self._pending_death:
            self._pending_death = False
            damage_class = str(s.get("dc", "bullet"))   # what killed them - see NetworkManager.notify_death
            self._set_dead(True, damage_class)
            if self.on_death is not None:
                move = s.get("d", (0.0, 0.0))
                speed = float(s.get("v", 0.0))
                velocity = glm.vec3(move[0] * speed, 0.0, move[1] * speed)
                push = s.get("b")   # an explosion's shove, sent with the death (see NetworkManager.notify_death)
                if push is not None and len(push) == 3:
                    # gibs.spawn carries 0.6 of a body's velocity, so scale up to give the push its full strength.
                    velocity += glm.vec3(*push) / 0.6
                self.on_death(feet_pos, velocity, self._color_wanted, damage_class)
        elif self.dead and self._alive_wanted:
            self._set_dead(False)
        if self._dissolving:
            self._dissolve_t += dt
            obj = self.model.obj
            if obj is not None:
                if self._dissolve_t >= self.DISSOLVE_SECONDS:
                    self._dissolving = False
                    obj["dissolve_amount"] = 0.0
                    obj["visible_in_color"] = False
                    obj["cast_shadow"] = False
                else:
                    obj["dissolve_amount"] = self._dissolve_t / self.DISSOLVE_SECONDS
                    # Gentle upward drift (see DISSOLVE_RISE_SPEED's own docstring on why) -
                    # obj["position"] was just re-pinned to the network feet position above
                    # (model.update/move_to), so the accumulated rise has to be re-added every
                    # frame rather than nudged once, or it'd be overwritten right back to 0.
                    self._dissolve_rise += self.DISSOLVE_RISE_SPEED * dt
                    obj["position"] = glm.vec3(
                        obj["position"].x, obj["position"].y + self._dissolve_rise, obj["position"].z)
                    # Ambient dust crackling around the body (Source's own C_EntityDissolve::
                    # DrawModel spawns these every frame it's active, scattered across the
                    # character's hitboxes - see dissolve_sparks.py's own docstring) - the
                    # shader itself only ever tints/discards (see skeletal_shader.py's own
                    # dissolve comment), all the crackle is this callback's doing.
                    pos = obj["position"]
                    def _random_body_point():
                        return glm.vec3(
                            pos.x + random.uniform(-0.3, 0.3),
                            pos.y + random.uniform(0.1, _BODY_HEIGHT - 0.1),
                            pos.z + random.uniform(-0.3, 0.3),
                        )
                    self._dissolve_spark_t -= dt
                    if self._dissolve_spark_t <= 0.0 and self.on_dissolve_spark is not None:
                        self._dissolve_spark_t = self.DISSOLVE_SPARK_INTERVAL
                        for _ in range(self.DISSOLVE_SPARK_BURST_COUNT):
                            self.on_dissolve_spark(_random_body_point())
                    # The tesla-arc bursts - the actual "electricity" read (see
                    # DISSOLVE_ARC_INTERVAL_START's own docstring on the interval ramp this
                    # mirrors) - DISSOLVE_ARC_BURST_COUNT arcs fired from DIFFERENT points at
                    # once each time, same as Source's own 3-beams-per-DoSparks-call, instead
                    # of one arc from a single point - a single point at a time reads as one
                    # faint crackle instead of the body being wreathed in electricity.
                    self._dissolve_arc_t -= dt
                    if self._dissolve_arc_t <= 0.0 and self.on_dissolve_arc is not None:
                        progress = self._dissolve_t / self.DISSOLVE_SECONDS
                        # SimpleSplineRemapVal's own smoothstep shape (3t^2 - 2t^3), not a
                        # straight lerp - matches how gently Source's own ramp starts and ends.
                        eased = progress * progress * (3.0 - 2.0 * progress)
                        self._dissolve_arc_t = (
                            self.DISSOLVE_ARC_INTERVAL_START
                            + (self.DISSOLVE_ARC_INTERVAL_END - self.DISSOLVE_ARC_INTERVAL_START) * eased)
                        for _ in range(self.DISSOLVE_ARC_BURST_COUNT):
                            self.on_dissolve_arc(_random_body_point())
        if self.dead:
            self._pending_shots = 0
            self._pending_tracers.clear()
            self._pending_footsteps = 0
        while self._pending_footsteps > 0:
            self._pending_footsteps -= 1
            if self.on_footstep is not None:
                # Feet-level (see PlayerModel.update's own position contract), not
                # _voice_position (roughly chest height, used for gunfire/tracers) -
                # a footstep should sound like it's coming from the ground.
                feet = self.model.obj["position"] if self.model.obj is not None else self._voice_position
                self.on_footstep(self._footstep_material, glm.vec3(feet), self._footstep_volume)
        while self._pending_shots > 0:
            self._pending_shots -= 1
            # From about chest height at where they're standing now.
            # From about chest height, following this player while it plays.
            self.weapon.play_fire_sound(self.scene, self._voice_position, follow=lambda: self._voice_position)
        for end in self._pending_tracers:
            if self.on_tracer is not None:
                end = glm.vec3(*end)
                aim = end - self._voice_position
                if glm.length(aim) > 1e-3:
                    # From the muzzle bone of the gun in their hand (or, if
                    # it isn't posed yet, a little ahead of the chest).
                    muzzle = self.weapon.muzzle_position(self.scene)
                    if muzzle is None:
                        muzzle = self._voice_position + glm.normalize(aim) * 0.6
                    # (the flash stays on the gun's muzzle as it moves)
                    self.on_tracer(muzzle, end, lambda: self.weapon.muzzle_position(self.scene),
                                   self.weapon)
        self._pending_tracers.clear()
        if self._weapon_wanted is not None and self._weapon_wanted != self._weapon_id:
            self._apply_weapon(self._weapon_wanted)
        if self._hat_wanted != self._hat_applied:
            self._hat_applied = self._hat_wanted   # even if unknown - don't retry every frame
            self.model.set_hat(self._hat_wanted)
        if self._color_wanted != self._color_applied:
            self._color_applied = self._color_wanted
            self.model.set_tint(self._color_wanted)

        # Hitbox centered vertically on the body (feet to eye), not at
        # floor level, and follows facing so a future directional query
        # (e.g. a cone/box in front of the shooter) lines up.
        # (Parked far below the map while they're dead so shots pass through.) Only touched when
        # they've actually moved or turned: a player standing still costs no physics writes.
        key = (round(feet_pos.x, 3), round(feet_pos.y, 3), round(feet_pos.z, 3), round(yaw, 1), self.dead)
        if key != self._hitbox_key:
            self._hitbox_key = key
            self.scene.physics.update_hitbox(
                self._hitbox,
                glm.vec3(0.0, -1000.0, 0.0) if self.dead else feet_pos + glm.vec3(0.0, _BODY_HEIGHT / 2.0, 0.0),
                glm.vec3(0.0, glm.radians(yaw), 0.0),
            )
            self._update_head_hitbox(feet_pos, yaw)

    def _update_head_hitbox(self, feet_pos, yaw):
        """Puts the head box on the rat's head, turned the way the model is (the model's local +z
        is the camera's flat forward - see PlayerModel.update's yaw formula)."""
        if self.dead:
            self.scene.physics.update_hitbox(self._head_hitbox, glm.vec3(0.0, -1000.0, 0.0), glm.vec3(0.0))
            return
        turn = glm.radians(90.0 - yaw)
        forward = glm.vec3(glm.sin(turn), 0.0, glm.cos(turn))
        center = feet_pos + glm.vec3(0.0, _HEAD_CENTER[1], 0.0) + forward * _HEAD_CENTER[2]
        self.scene.physics.update_hitbox(self._head_hitbox, center, glm.vec3(0.0, turn, 0.0))

    DEFAULT_WEAPON = "usp"      # until their packets say otherwise

    def _apply_weapon(self, weapon_id):
        """Puts the weapon with that id in their hand (loading it the first time it's
        shown), putting the previous one away. An unknown id is ignored."""
        weapon = self._weapons.get(weapon_id)
        if weapon is None:
            weapon = create_weapon(weapon_id)
            if weapon is None:
                self._weapon_wanted = self._weapon_id if self.weapon is not None else None
                return
            self._weapons[weapon_id] = weapon
        if self.weapon is not None:
            self.weapon.deactivate()
        self.weapon = weapon
        self._weapon_id = weapon_id
        weapon.equip_player(self.scene, self.model)
        # A dead player's new weapon stays hidden too - just the weapon, NOT the full
        # _set_dead (which would also touch the body/dissolve state - wrong here: this isn't
        # a death transition, it's an already-dead player's weapon simply changing, and
        # calling the full thing would incorrectly interrupt an in-progress dissolve).
        dead = getattr(self, "dead", False)
        for obj in weapon.worldmodel_objs:
            obj["visible_in_color"] = not dead
            obj["cast_shadow"] = not dead

    def _set_dead(self, dead, damage_class="bullet"):
        """Hides (or shows again) their gun, always instantly - and their BODY too, unless
        this death's damage_class dissolves instead of gibbing (see Modules/Weapons/
        damage_classes.py's own death_effect): a dissolve needs the body to stay visible and
        animate for DISSOLVE_SECONDS first (see update()'s own draining of self._dissolving),
        so hiding it here immediately would just make it vanish outright instead."""
        self.dead = dead
        for obj in getattr(self.weapon, "worldmodel_objs", ()):
            if obj is not None:
                obj["visible_in_color"] = not dead
                obj["cast_shadow"] = not dead
        obj = self.model.obj
        if obj is None:
            return
        dc = get_damage_class(damage_class) if dead else None
        if dead and dc.death_effect == DISSOLVE:
            self._dissolving = True
            self._dissolve_t = 0.0
            self._dissolve_spark_t = 0.0
            self._dissolve_arc_t = 0.0
            self._dissolve_rise = 0.0
            obj["dissolve_color"] = dc.dissolve_color
            # visible_in_color/cast_shadow left exactly as they were (alive) - update() ramps
            # dissolve_amount up over DISSOLVE_SECONDS, THEN hides the body, once it's fully gone.
        else:
            self._dissolving = False
            obj["dissolve_amount"] = 0.0
            obj["visible_in_color"] = not dead
            obj["cast_shadow"] = not dead

    def destroy(self):
        """Not called anywhere yet - network_manager.py has no lobby-
        member-left/disconnect handling today (a pre-existing gap this
        feature doesn't add to), so a disconnected peer's model/hitbox
        currently just stays put. Exposed now so wiring that up later is
        a one-line call, not another rewrite."""
        for weapon in self._weapons.values():
            weapon.unequip()
        self.model.destroy()
        if self._hitbox is not None:
            self.scene.physics.remove_hitbox(self._hitbox)
            self._hitbox = None
        if self._head_hitbox is not None:
            self.scene.physics.remove_hitbox(self._head_hitbox)
            self._head_hitbox = None
