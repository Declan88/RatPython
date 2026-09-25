"""
Base class for weapons. A weapon is mostly DATA - which sound it makes, which
models show it in the first-person view and in a character's hand, which
animation files pose the owner's arms - plus the little behaviour every weapon
shares (a rate-limited fire() that plays the gunshot, equip/unequip, and
play(state) to switch every attached part to that state's animation).

Subclass it (see usp.py) and override the class attributes below; nothing
else needs touching. The defaults describe the pistol set, so a bare
WeaponsBase() is a working (USP-sounding) pistol.

One instance per OWNER: it remembers what it attached (the world model in a
hand, the first-person gun), so the local player and every remote player each
build their own.

Animation files: each state ("idle", later "fire", "reload"...) maps to a .glb
holding that state's animation for that rig (a file with several rigs, like
pistol.glb's arms + gun, has one clip per rig - the part's skin picks it), or None for "the
clip already inside the model's own file". The clip is renamed to
"<animation_prefix>_<state>" (plus "_arms"/"_gun"/"_wm" for the parts that
share a name with the player rig) when merged onto the target skeleton, so
e.g. the player's idle pose ends up as "pistol_idle".
"""

import time

import glm

from Modules.Graphics.skeletal_loader import _read_glb_json_and_blob

_POSE_DIR = "Assets/Models/Arms/New Folder/Pistol"
_SPINE4 = "ValveBiped.Bip01_Spine4"


def _first_clip_name(path, skin_index=0):
    """Name of the first animation in a glb that actually animates the given
    skin's joints (None if none does) - a file holding several rigs has one
    clip per rig."""
    result = _read_glb_json_and_blob(path)
    if result is None:
        return None
    gltf = result[0]
    skins = gltf.get("skins") or []
    if not 0 <= skin_index < len(skins):
        return None
    joints = set(skins[skin_index]["joints"])
    for animation in gltf.get("animations") or []:
        if any(channel["target"].get("node") in joints for channel in animation["channels"]):
            return animation.get("name")
    return None


def _load_state_clip(scene, obj, path, clip_name, skin_index=0):
    """Merges the clip of `path` that animates its skin `skin_index` onto obj's
    skeleton as `clip_name`. Returns the name to play (clip_name), or None when
    there was nothing to load. A path of None means the clip already lives in
    the model's own file: the skeleton's first clip is used as-is."""
    if obj is None or "skeleton" not in obj:
        return None
    animations = obj["skeleton"].animations
    if path is None:
        return next(iter(animations), None)
    if clip_name in animations:
        return clip_name          # already merged (e.g. another owner of the same rig)
    source = _first_clip_name(path, skin_index)
    if source is None:
        return None
    added = scene.load_additional_animations(obj, path, rename={source: clip_name}, skin_index=skin_index)
    return clip_name if clip_name in added else None


class Shot:
    """One trigger pull that went off (see WeaponsBase.fire). hit: the
    PhysicsWorld.RayHit of the line trace along the aim, or None if it hit
    nothing within range. victim: the owner tag of the hitbox it struck (a
    remote player's steam_id), or None if it hit level geometry/nothing.
    damage: what this shot does to a victim."""
    __slots__ = ("hit", "victim", "damage")

    def __init__(self, hit, damage):
        self.hit = hit
        self.victim = hit.owner if hit is not None else None
        self.damage = damage


class WeaponsBase:
    name = "weapon"
    # Prefix for clip names this weapon adds to a rig (see module docstring).
    animation_prefix = "pistol"

    # ---- gunfire sound -------------------------------------------------
    fire_sound = "Assets/Audio/Guns/USP/usp_unsil-1.wav"
    fire_volume = 1.0
    # SoundManager's falloff is flat-ish out to about half of max_distance and
    # then drops steeply to silence at max_distance: a gunshot is loud enough
    # to be clearly heard across a whole map (~100m) but not through it, and
    # full volume for anyone standing within a few metres.
    fire_min_distance = 5.0
    fire_max_distance = 160.0
    fire_interval = 0.0       # minimum seconds between shots (0 = as fast as the owner fires)
    automatic = False         # True: holding the trigger keeps firing; False: one shot per click

    # ---- damage --------------------------------------------------------
    damage = 10.0             # health taken from a player each shot hits
    max_range = 500.0         # metres a shot's line trace reaches

    # ---- models --------------------------------------------------------
    # First person: ONE glb holding both the arms and the gun, each with its own
    # rig (skin index) and its own baked animation.
    viewmodel_model = f"{_POSE_DIR}/pistol.glb"
    viewmodel_arms_skin = 0
    viewmodel_gun_skin = 1
    worldmodel = f"{_POSE_DIR}/pistolWM.glb"
    # The character joint the world model is held by, and its placement in
    # that joint's frame: (x, y, z) metres and (x, y, z) euler degrees, plus a
    # uniform scale. Tune these to seat the model in the hand.
    worldmodel_bone = "ValveBiped.Bip01_R_Hand"
    worldmodel_position = (0.07, 0.0, 0.02)
    worldmodel_rotation = (-90.0, 0.0, -90.0)
    worldmodel_scale = 1.0

    # ---- animations (state -> .glb, or None for the model's own clip) ----
    # One set for both parts: the arms and the gun share an armature (the gun
    # rig is the arms rig plus the weapon bones), so each file's GUN-rig clip
    # (viewmodel_gun_skin) plays on the arms too - joints the arms don't have
    # are simply skipped. States in viewmodel_one_shot_states play once and
    # then drop back to idle.
    viewmodel_animations = {
        "idle": f"{_POSE_DIR}/pistol_idle.glb",
        "shoot": f"{_POSE_DIR}/pistol_shoot.glb",
    }
    viewmodel_one_shot_states = ("shoot",)
    worldmodel_animations = {"idle": None}
    # The player rig's own pose for holding this weapon (applied to the upper
    # body - see equip_player), and a rotation correction for it per joint
    # ({joint name: (x, y, z) degrees}, component space - see
    # PlayerModel.set_upper_rotation_offset).
    player_animations = {"idle": "Assets/Animations/Poses/Pistol/pistolidle.glb"}
    player_upper_rotation_degrees = {}
    # While the owner is MOVING the pose reads as looking too far to one side:
    # an override pose is absolute, so it doesn't follow the lean/twist the
    # running lower body adds. This is an extra yaw on Spine4 (component space,
    # degrees; negative = turn right) blended in while horizontal speed is
    # above player_moving_speed (m/s) - see set_moving.
    player_moving_yaw_degrees = 0.0
    player_moving_speed = 1.0

    def __init__(self):
        self._last_fire = float("-inf")
        self._fire_channel = None   # the mixer channel the last shot played on
        self._scene = None
        self._player_model = None
        self.worldmodel_obj = None
        self._viewmodel = None
        self._clips = {}   # (part, state) -> clip name actually loaded
        self._moving = False

    # ---- firing --------------------------------------------------------

    def can_fire(self, now=None):
        now = time.perf_counter() if now is None else now
        return now - self._last_fire >= self.fire_interval

    def fire(self, scene, position, direction=None, now=None, follow=None):
        """Pulls the trigger: plays the gunshot at `position` (world space)
        with this weapon's distance falloff (`follow`, a callable returning the
        owner's current position, keeps it attached to the owner while it
        plays), starts the shoot animations, and - if `direction` (the aim, a
        unit vector) is given - traces a line from `position` along it, up to
        max_range, through scene.physics. Returns a Shot describing the trace
        (Shot.victim is a struck player's id), or None - doing nothing - if
        it fired less than fire_interval ago. What a hit DOES (health, the
        network message) is the caller's to apply."""
        now = time.perf_counter() if now is None else now
        if not self.can_fire(now):
            return None
        self._last_fire = now
        self.play_fire_sound(scene, position, follow)
        self.play("shoot")
        hit = None
        if direction is not None:
            origin = glm.vec3(position)
            hit = scene.physics.raycast(origin, origin + glm.normalize(glm.vec3(direction)) * self.max_range)
        return Shot(hit, self.damage)

    def play_fire_sound(self, scene, position, follow=None):
        """The gunshot alone, with no rate limit - for replaying a shot some
        other player fired (their own weapon already rate-limited it)."""
        if not self.fire_sound:
            return None
        # Every shot plays on the SAME channel as the last: a shot fired before
        # the previous one finished cuts it off instead of asking the mixer for
        # another free channel, so a fast trigger can't run it out of channels
        # and lose the sound.
        emitter = scene.sound_manager.add_sound(
            self.fire_sound, glm.vec3(position), volume=self.fire_volume,
            min_distance=self.fire_min_distance, max_distance=self.fire_max_distance,
            loop=False, follow=follow, channel=self._fire_channel,
        )
        self._fire_channel = emitter["channel"]
        return emitter

    # ---- equipping -----------------------------------------------------

    def player_clip(self, state="idle"):
        return f"{self.animation_prefix}_{state}"

    def equip_player(self, scene, player_model):
        """Puts the world model in the character's hand and poses its upper
        body with this weapon's player animation. player_model: a
        PlayerModel (Modules/Player/player_model.py). The pose needs one built
        with an override_upper_body_root_joints split (the local player's is);
        on one without, only the world model is attached."""
        self.unequip_player()
        self._scene = scene
        self._player_model = player_model
        player_obj = player_model.obj
        if player_obj is None:
            return

        for state, path in self.player_animations.items():
            clip = _load_state_clip(scene, player_obj, path, self.player_clip(state))
            if clip is not None:
                self._clips[("player", state)] = clip
        idle = self._clips.get(("player", "idle"))
        if idle is not None and getattr(player_model, "_override_upper_body_root_joints", None) is not None:
            player_model.set_upper_override(idle)
            offsets = self._upper_offsets()
            if offsets:
                player_model.set_upper_rotation_offset(offsets)

        self._attach_worldmodel(scene, player_obj)

    def _attach_worldmodel(self, scene, player_obj):
        if not self.worldmodel:
            return
        obj = scene.add_skeletal(self.worldmodel, visible_in_color=True, cast_shadow=True)
        if obj is None:
            return
        obj["specular_strength"] = 0
        rotation = glm.mat4(glm.quat(glm.radians(glm.vec3(*self.worldmodel_rotation))))
        local = (
            glm.translate(glm.mat4(1.0), glm.vec3(*self.worldmodel_position))
            * rotation * glm.scale(glm.mat4(1.0), glm.vec3(self.worldmodel_scale))
        )
        if scene.attach_skeletal(obj, player_obj, self.worldmodel_bone, local) is None:
            scene.remove_skeletal(obj)
            return
        self.worldmodel_obj = obj
        for state, path in self.worldmodel_animations.items():
            clip = _load_state_clip(scene, obj, path, f"{self.player_clip(state)}_wm")
            if clip is not None:
                self._clips[("worldmodel", state)] = clip
        idle = self._clips.get(("worldmodel", "idle"))
        if idle is not None:
            scene.set_skeletal_animation(obj, idle, blend_duration=0.0)

    def _upper_offsets(self):
        offsets = dict(self.player_upper_rotation_degrees)
        if self._moving and self.player_moving_yaw_degrees:
            x, y, z = offsets.get(_SPINE4, (0.0, 0.0, 0.0))
            offsets[_SPINE4] = (x, y + self.player_moving_yaw_degrees, z)
        return offsets

    def set_moving(self, horizontal_speed):
        """Call each frame with the owner's horizontal speed (m/s): applies or
        releases the moving-yaw correction (player_moving_yaw_degrees) on the
        pose set by equip_player, crossfading via the usual offset change."""
        moving = horizontal_speed > self.player_moving_speed
        if moving == self._moving:
            return
        self._moving = moving
        model = self._player_model
        if model is not None and getattr(model, "_override_upper_body_root_joints", None) is not None:
            offsets = self._upper_offsets()
            if offsets:
                model.set_upper_rotation_offset(offsets)

    def equip_viewmodel(self, viewmodel):
        """Shows this weapon's first-person gun on `viewmodel` (a ViewModel -
        Modules/Player/viewmodel.py) and gives its arms this weapon's arm
        animations."""
        self._viewmodel = viewmodel
        viewmodel.set_weapon(self)

    def unequip_player(self):
        if self.worldmodel_obj is not None and self._scene is not None:
            self._scene.remove_skeletal(self.worldmodel_obj)
        self.worldmodel_obj = None
        if self._player_model is not None:
            self._player_model.clear_upper_override()
        self._player_model = None

    def unequip(self):
        self.unequip_player()
        if self._viewmodel is not None:
            self._viewmodel.clear_weapon()
            self._viewmodel = None
        self._clips.clear()

    def play(self, state="idle"):
        """Switches every attached part to `state`'s animation (a part with
        no clip for that state is left as it is)."""
        scene = self._scene
        if scene is not None and self.worldmodel_obj is not None:
            clip = self._clips.get(("worldmodel", state))
            if clip is not None:
                scene.set_skeletal_animation(self.worldmodel_obj, clip)
        if self._player_model is not None:
            clip = self._clips.get(("player", state))
            if clip is not None and getattr(self._player_model, "_override_upper_body_root_joints", None) is not None:
                self._player_model.set_upper_override(clip)
        if self._viewmodel is not None:
            self._viewmodel.play(state)
