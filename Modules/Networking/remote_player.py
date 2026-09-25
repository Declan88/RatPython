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

import time

import glm

from Modules.Player.player_model import PlayerModel

# How far behind real time remote players are drawn - see module docstring.
# Comfortably more than one packet interval (1/30s) so there are almost
# always two snapshots to blend between, even with a dropped packet.
INTERP_DELAY = 0.1
_MAX_SNAPSHOTS = 40
# A peer that goes quiet for this long is treated as standing still rather
# than freezing mid-stride in whatever locomotion pose its last packet had.
_STALE_SECONDS = 1.0

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


class RemotePlayer:
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
            scale=scale, time_scale=_RAT_ANIM_TIME_SCALE,
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

        self._hitbox = scene.physics.add_hitbox(hitbox_half_extents, owner=steam_id)

        self._snapshots = []      # [(arrival_time, feet_pos, yaw_degrees)], oldest first
        self._state = {}          # latest non-interpolated locomotion state
        self._last_packet = 0.0
        self._jump_seen = None    # last jump counter value seen
        self._pending_jump = False

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
        s = self._state
        stale = now - self._last_packet > _STALE_SECONDS
        move = s.get("d", (0.0, 0.0))
        self.model.update(
            dt, feet_pos, yaw, 0.0 if stale else float(s.get("v", 0.0)),
            is_crouched=bool(s.get("c", False)),
            is_grounded=bool(s.get("g", True)),
            is_sprinting=bool(s.get("s", False)),
            move_direction=None if stale else glm.vec3(move[0], 0.0, move[1]),
            just_jumped=self._pending_jump,
        )
        self._pending_jump = False

        # Hitbox centered vertically on the body (feet to eye), not at
        # floor level, and follows facing so a future directional query
        # (e.g. a cone/box in front of the shooter) lines up.
        self.scene.physics.update_hitbox(
            self._hitbox,
            feet_pos + glm.vec3(0.0, _BODY_HEIGHT / 2.0, 0.0),
            glm.vec3(0.0, glm.radians(yaw), 0.0),
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
