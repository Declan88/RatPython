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

import math
import random
import time

import glm

from Modules.Graphics.skeletal_loader import _read_glb_json_and_blob
from Modules.Physics.physics_world import CollisionGroup
from Modules.Weapons.recoil import Recoil

_POSE_DIR = "Assets/Models/Arms/New Folder/Pistol"
_SPINE4 = "ValveBiped.Bip01_Spine4"


def _debug_active():  # TEMP DEBUG (RATWAR_AUTOTEST)
    import os
    until = os.environ.get("RATWAR_DEBUG_UNTIL")
    return until is not None and time.perf_counter() < float(until)


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


def _load_state_clip(scene, obj, path, clip_name, skin_index=0, time_scale=1.0):
    """Merges the clip of `path` that animates its skin `skin_index` onto obj's
    skeleton as `clip_name`. Returns the name to play (clip_name), or None when
    there was nothing to load. A path of None means the clip already lives in
    the model's own file: the skeleton's first clip is used as-is.

    time_scale: scales every keyframe TIME (not the poses) by this factor as
    it's merged in, same mechanism this project already uses to correct a
    file baked at the wrong fps (see load_animation_clips' own docstring) -
    here it's what a speed multiplier like USP's own draw_speed actually
    does: 1/draw_speed shrinks the clip's timeline, so it plays through the
    same poses in less real time. Baked in at load time rather than adjusted
    per-frame, so it costs nothing beyond the one-time merge."""
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
    added = scene.load_additional_animations(
        obj, path, rename={source: clip_name}, skin_index=skin_index, time_scale=time_scale)
    return clip_name if clip_name in added else None


def _spread_direction(direction, spread_degrees):
    """`direction` (a unit vector) turned by a random angle within a cone of
    half-angle spread_degrees around it - uniform over the cone's cross
    section, so shots cluster no more at the centre than the edge."""
    direction = glm.normalize(glm.vec3(direction))
    if spread_degrees <= 0.0:
        return direction
    up = glm.vec3(0.0, 1.0, 0.0) if abs(direction.y) < 0.99 else glm.vec3(1.0, 0.0, 0.0)
    right = glm.normalize(glm.cross(direction, up))
    true_up = glm.cross(right, direction)
    polar = math.radians(spread_degrees) * math.sqrt(random.random())
    azimuth = random.random() * 2.0 * math.pi
    offset = right * math.cos(azimuth) + true_up * math.sin(azimuth)
    return glm.normalize(direction * math.cos(polar) + offset * math.sin(polar))


class Shot:
    """One trigger pull that went off (see WeaponsBase.fire). hit: the
    PhysicsWorld.RayHit of the line trace along the aim, or None if it hit
    nothing within range. victim: the owner tag of the hitbox it struck (a
    remote player's steam_id), or None if it hit level geometry/nothing.
    damage: what this shot does to a victim."""
    __slots__ = ("hit", "victim", "damage", "direction", "spread", "headshot")

    def __init__(self, hit, damage, direction=None, spread=0.0, headshot=False):
        self.hit = hit
        self.victim = hit.owner if hit is not None else None
        self.headshot = headshot     # the trace also passed through the victim's head hitbox
        self.damage = damage
        self.direction = direction   # the (spread-adjusted) direction the trace actually went
        self.spread = spread         # the cone half-angle (degrees) it was drawn from


class FireMode:
    """A weapon's fire_mode (below) - what shots_this_frame does with a
    frame's clicks/held state. SEMI and AUTO are built into shots_this_frame
    itself; CUSTOM is a marker meaning "this class overrides shots_this_frame
    with its own logic" (burst fire, etc.) - see that method's own docstring."""
    SEMI = "semi"
    AUTO = "auto"
    CUSTOM = "custom"


class WeaponsBase:
    name = "weapon"
    # Prefix for clip names this weapon adds to a rig (see module docstring).
    animation_prefix = "pistol"
    # A square icon file - the weapon select HUD (Modules/UI/weapon_hud.py)
    # and the kill feed (Modules/UI/killfeed.py's own WEAPON_ICONS, keyed by
    # `name` since a kill feed entry only ever has the weapon's name off the
    # wire, not the class) both show it. None draws no icon.
    icon = None

    # ---- gunfire sound -------------------------------------------------
    fire_sound = "Assets/Audio/Guns/USP/usp_unsil-1.wav"
    fire_volume = 1.0
    # Realistic-for-a-game falloff (SoundManager's "inverse" curve): full
    # volume within a few metres, then about -4 dB per doubling of distance,
    # fading to silence over the last quarter of max_distance - and shots that
    # start far away play a low-passed copy (muffle), so they sound like
    # distant gunfire rather than just a quieter close one.
    fire_min_distance = 5.0
    fire_max_distance = 160.0
    fire_interval = 0.0        # minimum seconds between shots (0 = as fast as the owner can trigger it)
    fire_mode = FireMode.SEMI  # see FireMode/shots_this_frame - was a bare `automatic` bool before

    # ---- ammo / reload --------------------------------------------------
    # 0 = no ammo tracking at all (unlimited, never needs a reload - the
    # behavior every weapon had before this existed). A positive value is
    # the magazine's capacity; fire() refuses to shoot at 0 rounds left
    # (see can_fire) until start_reload() finishes. Both reload and draw
    # (below) finish exactly when their OWN real animation does - see
    # _one_shot_still_playing - not a guessed duration constant (an earlier
    # version of this used reload_time/draw_time timers, which kept drifting
    # out of sync with the actual clip's real length - see git history).
    magazine_size = 12

    # ---- draw --------------------------------------------------------
    # Playback-speed multiplier for the "draw" state (see viewmodel_
    # animations' own "draw" entry) - 1.5 = 50% faster. Baked into the
    # clip's own timeline when it's loaded (see _load_state_clip's own
    # time_scale param), so the animation itself visibly speeds up, and
    # can_fire/start_reload (bound to the same real clip - see drawing's own
    # docstring) unblock exactly that much sooner too, automatically.
    draw_speed = 1.0

    @property
    def drawing(self):
        """True while this weapon's draw animation is still ACTUALLY playing
        on the first-person viewmodel (see ViewModel.is_one_shot_active) -
        fire()/start_reload() both refuse while this is true, same idea as
        `reloading`. Resolves False immediately (nothing to wait for) for a
        weapon with no viewmodel at all (e.g. a remote player's own copy -
        see RemotePlayer) or no "draw" clip loaded."""
        return self._one_shot_still_playing("draw")

    def _one_shot_still_playing(self, state):
        vm = self._viewmodel
        return vm is not None and vm.is_one_shot_active(state)

    # ---- accuracy ------------------------------------------------------
    # Shots land within a cone of half-angle "spread" degrees around the aim.
    # It sits at spread_min while the trigger is rested, grows by
    # spread_per_shot with each shot (capped at spread_max) and, once the gun
    # has been quiet for spread_recovery_delay seconds, shrinks back at
    # spread_recovery degrees per second. So how fast you can shoot decides how
    # bad it gets: pick spread_per_shot against the gun's realistic rate of
    # fire - a pistol you can only click a few times a second stays accurate at
    # that pace and only loses it when spammed.
    spread_min = 0.4
    spread_max = 4.0
    spread_per_shot = 1.0
    spread_recovery = 5.0
    spread_recovery_delay = 0.15

    # ---- recoil --------------------------------------------------------
    # Each shot kicks the owner's camera up by recoil_pitch degrees (plus a
    # random sideways kick of up to +-recoil_yaw), quickly (recoil_kick_speed
    # deg/s), then it settles back down by itself at recoil_recovery deg/s.
    # Shooting faster than it settles stacks the kicks, up to recoil_max
    # degrees. See Recoil.
    recoil_pitch = 1.0
    recoil_yaw = 0.0
    recoil_max = 6.0
    recoil_kick_speed = 120.0
    recoil_recovery = 6.0

    # ---- damage --------------------------------------------------------
    damage = 10.0             # health taken from a player each shot hits
    headshot_multiplier = 1.5  # ...times this when the shot hits the head
    max_range = 500.0         # metres a shot's line trace reaches

    # ---- models --------------------------------------------------------
    # First person: viewmodel_model holds the arms, each with its own rig
    # (skin index) and baked animation. The gun normally lives in that same
    # file as a second skin (viewmodel_gun_skin) - pistol.glb's own
    # convention - but viewmodel_gun_model lets it come from a SEPARATE
    # file instead (None means "same file as the arms"), for a weapon whose
    # gun was modeled/exported on its own (e.g. GoudaGun's Gouda_Anims.glb,
    # which has no arms of its own at all - see gouda_gun.py). Either way
    # the gun's rig needs the same joint names as viewmodel_model's arms
    # rig for viewmodel_animations' clips to play on both correctly.
    viewmodel_model = f"{_POSE_DIR}/pistol.glb"
    viewmodel_arms_skin = 0
    viewmodel_gun_model = None
    viewmodel_gun_skin = 1
    worldmodel = f"{_POSE_DIR}/pistolWM.glb"
    # The character joint the world model is held by, and its placement in
    # that joint's frame: (x, y, z) metres and (x, y, z) euler degrees, plus a
    # uniform scale. Tune these to seat the model in the hand.
    worldmodel_bone = "ValveBiped.Bip01_R_Hand"
    worldmodel_position = (0.07, 0.0, 0.02)
    worldmodel_rotation = (-90.0, 0.0, -90.0)
    worldmodel_scale = 1.0
    # The bones bullets (tracers) come out of. The world model has a real muzzle
    # bone; the first-person gun doesn't (Source keeps that as an attachment, not
    # a bone), so it uses the bone at the front of the barrel instead.
    # The particle system (Assets/Particles, see Modules/Particles) played at the
    # muzzle on every shot; None for no flash.
    muzzle_particle = "muzzle_pistols"
    # The flash's colour range, two (r, g, b) 0-255 endpoints each particle picks
    # between - replaces the particle file's own colours (None keeps them).
    muzzle_color = ((255, 180, 0), (255, 235, 180))
    muzzle_size = 0.7          # scale of the whole flash
    # The particle file offsets its particles from the origin (up to ~25 cm sideways
    # at full size), which pulls the flash off the barrel; this scales that (0 = all
    # on the muzzle).
    muzzle_offset_scale = 0.0
    worldmodel_muzzle_bone = "ValveBiped.flash"
    viewmodel_muzzle_bone = "v_weapon.USP_Silencer"

    # ---- animations (state -> .glb, or None for the model's own clip) ----
    # One set for both parts: the arms and the gun share an armature (the gun
    # rig is the arms rig plus the weapon bones), so each file's GUN-rig clip
    # (viewmodel_gun_skin) plays on the arms too - joints the arms don't have
    # are simply skipped. States in viewmodel_one_shot_states play once and
    # then drop back to idle.
    #
    # A value can also be (path, skin_index) instead of a bare path, to read
    # a DIFFERENT rig than viewmodel_gun_skin from that one file - needed for
    # USP's own "reload" entry (see usp.py): unlike pistol_idle.glb/pistol_
    # shoot.glb, pistol_reload.glb's gun rig happens to be its skin 0, not 1
    # (confirmed by inspecting the file directly - it has only one skin).
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
        self._spread_at_last_shot = self.spread_min
        self.recoil = Recoil(self)   # the owner's camera kick - see Recoil.apply
        self._fire_channel = None   # the mixer channel the last shot played on
        self._scene = None
        self._player_model = None
        self.worldmodel_obj = None
        self._viewmodel = None
        self._clips = {}   # (part, state) -> clip name actually loaded
        self._moving = False
        # magazine_size <= 0 means "no ammo tracking" (every weapon's old,
        # only behavior) - ammo then just sits unused, can_fire/fire never
        # consult it, and start_reload always refuses (nothing to refill).
        self.ammo = self.magazine_size
        self._reloading = False

    # ---- firing --------------------------------------------------------

    @property
    def reloading(self):
        return self._reloading

    def can_fire(self, now=None):
        if self._reloading or self.drawing:
            return False
        if self.magazine_size > 0 and self.ammo <= 0:
            return False
        now = time.perf_counter() if now is None else now
        # A tiny epsilon, not a bare >= - two calls exactly fire_interval
        # apart can otherwise miss a shot to plain float error (e.g.
        # 0.4 - 0.3 == 0.09999999999999998 in IEEE 754, just under 0.1)
        # rather than any real timing issue.
        return now - self._last_fire >= self.fire_interval - 1e-9

    def shots_this_frame(self, click_edges, trigger_held):
        """How many times to call fire() this frame - the fire_mode-driven
        replacement for a caller (app.py) directly branching on a bare
        `automatic` bool. click_edges: real trigger pulls since last frame
        (each one a fresh press, even if several land in the same frame -
        see app.py's own trigger_clicks); trigger_held: whether the mouse
        button is down RIGHT NOW, for a frame-by-frame check.

        FireMode.SEMI (default): exactly click_edges - one shot per press,
        holding the trigger down does nothing further until it's released
        and pressed again.

        FireMode.AUTO: 1 every frame the trigger's held (0 otherwise) -
        fire_interval (checked inside fire()/can_fire, not here) is what
        actually paces the real rate of fire; this just keeps offering it
        a shot to take every frame while held.

        FireMode.CUSTOM: this base implementation fires nothing - the whole
        point of CUSTOM is a subclass overrides this method itself with
        bespoke logic (burst fire, a charge-up weapon, ...) instead of
        picking between the two built-in shapes above."""
        if self.fire_mode == FireMode.AUTO:
            return 1 if trigger_held else 0
        if self.fire_mode == FireMode.SEMI:
            return click_edges
        return 0

    def spread_degrees(self, now=None):
        """The weapon's accuracy right now: the half-angle (degrees) of the
        cone the next shot would land in - see the accuracy settings above.
        Reads the clock, so a HUD can poll it every frame."""
        now = time.perf_counter() if now is None else now
        quiet = now - self._last_fire - self.spread_recovery_delay
        spread = self._spread_at_last_shot - self.spread_recovery * max(0.0, quiet)
        return max(self.spread_min, spread)

    def fire(self, scene, position, direction=None, now=None, follow=None):
        """Pulls the trigger: plays the gunshot at `position` (world space)
        with this weapon's distance falloff (`follow`, a callable returning the
        owner's current position, keeps it attached to the owner while it
        plays), starts the shoot animations, and - if `direction` (the aim, a
        unit vector) is given - traces a line from `position` along it, up to
        max_range, through scene.physics, deflected by a random amount within
        the current spread (spread_degrees) and then worsening the spread for
        the next shot. Returns a Shot describing the trace (Shot.victim is a
        struck player's id), or None - doing nothing - if it fired less than
        fire_interval ago. What a hit DOES (health, the network message) is the
        caller's to apply."""
        now = time.perf_counter() if now is None else now
        if not self.can_fire(now):
            return None
        spread = self.spread_degrees(now)
        self._last_fire = now
        if self.magazine_size > 0:
            self.ammo -= 1
        self._spread_at_last_shot = min(self.spread_max, spread + self.spread_per_shot)
        self.play_fire_sound(scene, position, follow)
        self.play("shoot")
        hit = None
        aim = None
        if direction is not None:
            origin = glm.vec3(position)
            aim = _spread_direction(direction, spread)
            hit = scene.physics.raycast(origin, origin + aim * self.max_range)
        # After the trace: the shot goes where the camera was pointing when the
        # trigger was pulled; the kick moves it for the NEXT one.
        self.recoil.kick()
        # A shot that struck a player is a headshot if the same trace also passes through THAT
        # player's head box (a second trace, against head boxes only - see CollisionGroup.HEAD).
        headshot = False
        if hit is not None and hit.owner is not None:
            head = scene.physics.raycast(origin, origin + aim * self.max_range, CollisionGroup.HEAD)
            headshot = head is not None and head.owner == hit.owner
        damage = self.damage * self.headshot_multiplier if headshot else self.damage
        return Shot(hit, damage, aim, spread, headshot)

    # ---- ammo / reload --------------------------------------------------

    def start_reload(self, now=None):
        """Begins a reload (plays the "reload" state - see viewmodel_
        animations/worldmodel_animations) if the magazine isn't already full
        and one isn't already running. update() (call once a frame
        regardless of input) finishes it once the animation actually does
        (see _one_shot_still_playing) - not a guessed duration. Returns
        whether one actually started - False for a weapon with no ammo
        tracking (magazine_size <= 0), one already full, one already
        reloading, or still mid-draw."""
        if self.magazine_size <= 0 or self._reloading or self.drawing or self.ammo >= self.magazine_size:
            return False
        self._reloading = True
        self.play("reload")
        return True

    def update(self, now=None):
        """Call once a frame regardless of input (app.py's main loop does):
        auto-starts a reload the instant the magazine runs dry (so running
        empty mid-fight reloads on its own, same as most games - no need to
        remember the reload key), and finishes an in-progress reload once its
        OWN animation has actually finished playing (see
        _one_shot_still_playing - a weapon with no viewmodel/no reload clip
        at all finishes instantly, same as it always refilling used to when
        this was a guessed timer), refilling the magazine and dropping back
        to idle. A no-op for a weapon with no ammo tracking at all
        (magazine_size <= 0)."""
        if self._reloading:
            if not self._one_shot_still_playing("reload"):
                self._reloading = False
                self.ammo = self.magazine_size
                self.play("idle")
            return
        if self.magazine_size > 0 and self.ammo <= 0:
            self.start_reload(now)

    def muzzle_position(self, scene):
        """Where this weapon's bullets come out, in the world: the first-person
        gun's muzzle bone while that's what the owner is looking at, otherwise
        the world model's (in the character's hand). None if neither is
        attached/posed yet."""
        gun = self._viewmodel.gun if self._viewmodel is not None else None
        if gun is not None and gun.get("viewmodel_visible") and self.viewmodel_muzzle_bone:
            return scene.joint_world_position(gun, self.viewmodel_muzzle_bone)
        if self.worldmodel_obj is not None and self.worldmodel_muzzle_bone:
            return scene.joint_world_position(self.worldmodel_obj, self.worldmodel_muzzle_bone)
        return None

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
            falloff="inverse", muffle=True,
        )
        self._fire_channel = emitter["channel"]
        return emitter

    # ---- equipping -----------------------------------------------------

    def player_clip(self, state="idle"):
        return f"{self.animation_prefix}_{state}"

    def equip_player(self, scene, player_model, activate=True):
        """Puts the world model in the character's hand and merges this
        weapon's player animation, then - unless activate=False - ACTIVATES
        it (see set_active: shows the world model and takes over the
        player's upper-body pose). player_model: a PlayerModel (Modules/
        Player/player_model.py). The pose needs one built with an
        override_upper_body_root_joints split (the local player's is); on
        one without, only the world model is attached.

        activate=False PRIMES this weapon (loads everything) WITHOUT
        touching the player's current pose or showing its world model - for
        pre-loading a weapon the player isn't holding yet (see app.py's own
        setup_game, which primes every non-current slot this way right after
        equipping the real one) without stealing the pose out from under
        whichever weapon actually IS active.

        If this exact weapon is already primed for this (scene, player_model)
        pair - i.e. this isn't the first time it's been equipped there, just
        switching back to it (or re-priming it, which is then just a no-op) -
        this is just set_active(activate): no re-loading, no rebuilding its
        world model from scratch. That's what makes app.py's own weapon
        switching near-instant on anything but a weapon's very first equip
        (see switch_weapon's own docstring) - an earlier version of this
        fully tore down and rebuilt the world model on EVERY switch, which is
        what caused a stutter each time."""
        if scene is self._scene and player_model is self._player_model and self.worldmodel_obj is not None:
            self.set_active(activate)
            return
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

        self._attach_worldmodel(scene, player_obj)
        self.set_active(activate)

    def set_active(self, active):
        """Shows/hides this weapon's world model, and - only while
        active - makes it the player's upper-body pose AND plays "draw" (see
        can_fire/start_reload, both of which refuse until it's actually
        finished playing - drawing) fresh, matching the draw animation
        restarting from the beginning every time. Falls back to whatever
        ViewModel.set_weapon already set (idle) for a weapon with no "draw"
        clip of its own - play()/ViewModel.play() both just skip a state with
        nothing loaded for it rather than clearing what's already showing.
        Near-instant either way: no loading, unlike equip_player's own
        first-time path (see its docstring) - this is what a weapon SWITCH
        actually does once both weapons involved are already primed."""
        if self.worldmodel_obj is not None:
            self.worldmodel_obj["visible_in_color"] = active
            self.worldmodel_obj["cast_shadow"] = active
        if not active:
            return
        if _debug_active():
            print(f"[WB DEBUG] set_active(True) weapon={self.name} -> play('draw')")
        self.play("draw")
        idle = self._clips.get(("player", "idle"))
        if (idle is not None and self._player_model is not None
                and getattr(self._player_model, "_override_upper_body_root_joints", None) is not None):
            self._player_model.set_upper_override(idle)
            offsets = self._upper_offsets()
            if offsets:
                self._player_model.set_upper_rotation_offset(offsets)

    def deactivate(self):
        """Hides this weapon WITHOUT unloading it - see equip_player's own
        docstring on why this (not unequip_player) is what a weapon switch
        calls on the outgoing weapon. Also cancels an in-progress reload
        rather than letting it keep running unseen: the viewmodel it was
        playing on is about to show the NEXT weapon's animations instead
        (there's only one active one-shot slot per ViewModel - see
        is_one_shot_active), so _one_shot_still_playing("reload") would
        read as finished the moment anything else plays, silently
        refilling the magazine in the background whether or not the
        player switches back. Cancelling here means switching back to
        this weapon later needs a fresh reload, same as most shooters -
        the magazine keeps whatever it had before the reload started.

        Also resets recoil: recoil.apply() (app.py's main loop) only runs
        for the currently ACTIVE weapon, so an unsettled kick from a shot
        fired right before switching away would otherwise sit frozen
        (recoil.update never gets called while inactive to drain it) and
        resume easing itself back to neutral the instant this weapon is
        drawn again - since the viewmodel is camera-relative, that read as
        the gun visibly sliding into place right after the (otherwise
        instant, blend_duration=0.0) draw cut."""
        if _debug_active():
            print(f"[WB DEBUG] deactivate() weapon={self.name}")
        if self._reloading:
            self._reloading = False
        self.recoil.reset()
        self.set_active(False)

    def _attach_worldmodel(self, scene, player_obj):
        if not self.worldmodel:
            return
        obj = scene.add_skeletal(self.worldmodel, visible_in_color=True, cast_shadow=True)
        if obj is None:
            return
        obj["specular_strength"] = 0
        obj["frustum_cull"] = (1.4, 0.4, 2.6)      # same as its owner's (its position is the owner's)
        obj["max_draw_distance"] = 30.0            # a pistol that far away is a couple of pixels
        obj["max_shadow_distance"] = 8.0
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

    def equip_viewmodel(self, viewmodel, activate=True):
        """Shows this weapon's first-person gun on `viewmodel` (a ViewModel -
        Modules/Player/viewmodel.py) and gives its arms this weapon's arm
        animations. activate=False PRIMES it (loads everything) without
        showing it or hiding whichever weapon IS currently shown - see
        equip_player's own activate docstring, which this mirrors."""
        self._viewmodel = viewmodel
        if activate:
            viewmodel.set_weapon(self)
        else:
            viewmodel.prime_weapon(self)

    def unequip_player(self):
        if self.worldmodel_obj is not None and self._scene is not None:
            self._scene.remove_skeletal(self.worldmodel_obj)
        self.worldmodel_obj = None
        if self._player_model is not None:
            self._player_model.clear_upper_override()
        self._player_model = None

    def unequip(self):
        self.unequip_player()
        self.recoil.reset()
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
