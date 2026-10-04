"""
Base class for weapons. A weapon is mostly DATA - which sound it makes, which
models show it in the first-person view and in a character's hand, which
animation files pose the owner's arms - plus the little behaviour every weapon
shares (a rate-limited fire() that plays the gunshot, equip/unequip, and
play(state) to switch every attached part to that state's animation).

Subclass it (see usp.py) and override the class attributes below - that part
really does need nothing else touched, Modules/Weapons/registry.py's own
pkgutil-based discovery finds a new subclass automatically. The defaults
describe the pistol set, so a bare WeaponsBase() is a working (USP-sounding)
pistol.

*** BUT, separately: add ONE line to Modules/Weapons/__init__.py too -       ***
*** "from .yourfile import YourWeaponClass" - or the new weapon works fine   ***
*** from source and then silently DISAPPEARS from a built .exe with no      ***
*** error anywhere (confirmed: exactly what happened to Pencil the first    ***
*** time around). See registry.py's own module docstring for the full       ***
*** why (Nuitka/PyInstaller can't see a module only ever reached through a  ***
*** dynamically-computed import name, so it never gets compiled in at all). ***

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
from Modules.Weapons.damage_classes import Bullet
from Modules.Weapons.recoil import Recoil

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


def _load_state_clip(scene, obj, path, clip_name, skin_index=0, time_scale=1.0, source_clip=None):
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
    per-frame, so it costs nothing beyond the one-time merge.

    source_clip: use this exact animation NAME from `path` instead of
    _first_clip_name's own "first clip that animates this skin" heuristic -
    for a file exported with SEVERAL named actions sharing one skin (e.g.
    Pencil.glb's own Idle/Shoot/Reload/DrawAction, all baked into one file -
    see pencil.py) rather than the usual one-clip-per-file pose convention,
    where "the first one" would just pick the same wrong clip for every
    state. See _parse_clip_spec for how a viewmodel_animations/player_
    animations/worldmodel_animations entry asks for this."""
    if obj is None or "skeleton" not in obj:
        return None
    animations = obj["skeleton"].animations
    if path is None:
        return next(iter(animations), None)
    if clip_name in animations:
        return clip_name          # already merged (e.g. another owner of the same rig)
    source = source_clip if source_clip is not None else _first_clip_name(path, skin_index)
    if source is None:
        return None
    added = scene.load_additional_animations(
        obj, path, rename={source: clip_name}, skin_index=skin_index, time_scale=time_scale)
    if clip_name not in added:
        return None
    # load_additional_animations merges EVERY clip `path` has for this skin, not just the
    # one named `source` above - harmless when a file genuinely has only one relevant clip
    # (every OTHER caller's case), but a file with several named actions sharing one skin
    # (Pencil.glb's Idle/Shoot/Reload/DrawAction, all in ONE file - see pencil.py) would
    # otherwise leave the other three sitting in `animations` under their own raw names,
    # then get needlessly re-parsed and re-merged (with a noisy "already exists -
    # overwriting" warning each time) on every one of THIS weapon's OTHER _load_state_clip
    # calls for that same file. Drop anything this call added besides what was actually
    # asked for - whichever state wants one of those loads and renames it on its own call.
    for extra in added:
        if extra != clip_name:
            del animations[extra]
    return clip_name


def _parse_clip_spec(clip_spec, default_skin_index):
    """Normalizes one viewmodel_animations/player_animations/worldmodel_
    animations value into (path, skin_index, source_clip) for _load_state_
    clip. A bare path (or None - see that function's own docstring) just
    uses `default_skin_index` and no source_clip (today's existing "first
    clip in the file" behavior, unchanged). A tuple can add:

      - an int: which SKIN of `path` to read instead of the default - e.g.
        USP's own "reload"/"draw" entries, whose gun rig happens to live at
        skin 0 in those particular files instead of the usual skin 1 (see
        viewmodel_animations' own base-class docstring).
      - a str: which of `path`'s own ANIMATIONS to use by NAME instead of
        _load_state_clip's "first clip that animates this skin" guess - for
        a file exported with several named actions sharing one skin (see
        _load_state_clip's own source_clip docstring).

    A 2-tuple's second element can be EITHER of those (told apart by type -
    int vs str), so (path, "Idle") and (path, 0) are both valid and mean
    different things. A 3-tuple gives both explicitly, always in (path,
    skin_index, source_clip) order, for a file that needs both at once (a
    non-default skin AND a specific named action within it)."""
    if not isinstance(clip_spec, tuple):
        return clip_spec, default_skin_index, None
    if len(clip_spec) == 3:
        return clip_spec
    path, second = clip_spec
    if isinstance(second, str):
        return path, default_skin_index, second
    return path, second, None


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
    # Its id in the weapon registry (registry.py) and over the network; None = the
    # class name in snake_case. Set abstract = True on a base class that isn't a
    # weapon you can carry.
    weapon_id = None
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

    # ---- scope (ADS) -----------------------------------------------------
    # Right-click smoothly eases the CAMERA'S OWN viewmodel-space transform
    # toward a real point ON THE GUN MODEL (see scope_point/update_scope,
    # both called from app.py's main loop every frame), NOT a hand-guessed
    # camera-space offset: the target is MEASURED off the gun's own current
    # pose every frame (same scene.joint_world_position mechanism muzzle_
    # position() already uses), so it moves and rotates exactly with the
    # actual viewmodel - sway, recoil, the draw animation, all of it -
    # instead of chasing a fixed guess that would drift out of register with
    # whatever the gun is actually doing on screen. Alongside that, the same
    # eased fraction also narrows the camera's own FOV toward scope_fov (see
    # app.py) and, once FULLY eased in (see `scoped`), swaps the ordinary
    # crosshair for the scope overlay and hides the viewmodel entirely (see
    # ViewModel.update's own hidden_while_scoped handling) - "looking
    # through the sight" rather than still seeing the gun in the way. False
    # by default: most weapons have nothing worth looking through and
    # right-click does nothing at all for them.
    has_scope = False
    # Which bone of the GUN rig (not the arms) the scope point is measured
    # from, and an optional (x, y, z) offset in THAT bone's own local frame
    # (None = right at the bone) - same shape/convention as viewmodel_
    # muzzle_bone/viewmodel_muzzle_offset. None (the base default) means
    # this weapon has no actual point defined even if has_scope were somehow
    # set True - scope_point() then always returns None.
    viewmodel_scope_bone = None
    viewmodel_scope_offset = None
    # Seconds to fully ease between hip-fire and fully scoped (and back,
    # same rate) - smaller is snappier. Same "how fast", not "how far" role
    # as draw_speed's own knob, just expressed as a duration instead of a
    # multiplier since there's no existing animation clip to scale here.
    scope_time = 0.2
    # Camera FOV (degrees) once FULLY scoped - app.py eases camera.fov
    # between its own normal/hip-fire value and this by the same fraction
    # update_scope returns, same as everything else about this ADS
    # transition. Meaningless for a has_scope False weapon (app.py never
    # reads this unless has_scope is True). Default picked as "a real,
    # noticeable zoom" for whichever future weapon doesn't bother overriding
    # it - see pencil.py's own value for an actual sniper-grade zoom.
    scope_fov = 20.0
    # Multiplies spread_degrees()'s own result once FULLY scoped, eased in by the exact
    # same smoothstepped fraction as everything else about this ADS transition (see
    # spread_degrees' own use of this) - 1.0 (the base default) means scoping gives no
    # accuracy benefit of its own beyond the camera/FOV/sensitivity effects everyone
    # already gets. 0.0 means PERFECTLY accurate once fully scoped, however wide
    # spread_min/spread_max would otherwise be - see pencil.py's own override, a real
    # sniper rifle's whole reason to look through its scope at all.
    scope_accuracy_multiplier = 1.0
    # Played once (not looped, not positional - see update_scope's own add_sound call)
    # the instant aiming actually STARTS/STOPS - a plain scope-lens zoom sound, same for
    # every scoped weapon unless a specific one overrides it. None (the base default)
    # means has_scope True with no sound set just stays silent, same as every other
    # optional sound slot in this file (fire_sound, muzzle_particle, ...).
    scope_zoom_in_sound = None
    scope_zoom_out_sound = None

    def scope_point(self, scene):
        """World position of this weapon's scope point RIGHT NOW - None if this weapon
        has no scope, no viewmodel_scope_bone configured, or the first-person gun isn't
        currently posed/visible (same guards as muzzle_position's own gun branch).
        Reflects whatever the gun is ACTUALLY doing this frame (sway, recoil, the current
        animation pose), since it's measured fresh off the live skeleton every call rather
        than cached - see app.py's own scope-handling for why it's called once a frame,
        right after viewmodel.update() has finished posing the gun for THIS frame."""
        if not self.has_scope or not self.viewmodel_scope_bone:
            return None
        gun = self._viewmodel.gun if self._viewmodel is not None else None
        if gun is None or not gun.get("viewmodel_visible"):
            return None
        return scene.joint_world_position(
            gun, self.viewmodel_scope_bone, local_offset=self.viewmodel_scope_offset)

    def update_scope(self, aiming, dt):
        """Call once a frame (app.py's main loop does, unconditionally - same shape as
        weapon.update()) with whether the player is CURRENTLY holding right-click (and
        every other reason it should count right now - alive, not in a menu, not third-
        person - already folded in by the caller). Eases _scope_blend toward 1.0
        (aiming) or 0.0 (not) at a rate set by scope_time, and returns that fraction
        SMOOTHSTEPPED (3t^2-2t^3, an ease-in/ease-out feel rather than a linear blend) -
        the caller blends the camera between its normal eye position and scope_point()'s
        own measured world position by this fraction (see app.py's own handling), so it
        arrives smoothly instead of snapping the instant the button goes down.

        Always 0.0 for a weapon with has_scope False - a caller can call this
        unconditionally on whatever weapon is active without checking has_scope itself
        first, same as every other per-frame weapon method here (update(), recoil.apply,
        ...).

        Forces `aiming` False outright while self.racking is true (a weapon with a rack
        clip only - see that property's own docstring) - overriding whatever the player
        is still holding right-click for. A real scope sight picture wouldn't show the
        bolt being worked at all (the viewmodel is hidden entirely once fully scoped -
        see ViewModel.update's own hidden_while_scoped), and working a bolt action takes
        your eye off the scope anyway - so the player is kicked back out to see (and be
        gated by) the rack animation instead of staring at a frozen sight picture while
        it plays unseen underneath."""
        if not self.has_scope:
            self._scope_blend = 0.0
            return 0.0
        if self.racking:
            aiming = False
        if aiming != self._was_aiming:
            # Edge-triggered (the instant the input actually CHANGES, not every frame
            # it happens to be held) - a plain scope-lens zoom sound, same one whether
            # aiming started because the player pressed right-click or stopped because
            # racking just forced it back out (see this method's own racking check
            # above) - either way the lens is physically moving. Flat, in both ears, at
            # any distance (universal=True): it's feedback for the player looking
            # through their OWN scope, not a sound anyone else in the world would hear
            # (same reasoning as app.py's own hitmarker sound).
            sound = self.scope_zoom_in_sound if aiming else self.scope_zoom_out_sound
            if sound and self._scene is not None:
                self._scene.sound_manager.add_sound(sound, glm.vec3(0.0), loop=False, universal=True)
            self._was_aiming = aiming
        target = 1.0 if aiming else 0.0
        step = dt / max(self.scope_time, 1e-6)
        if self._scope_blend < target:
            self._scope_blend = min(target, self._scope_blend + step)
        else:
            self._scope_blend = max(target, self._scope_blend - step)
        return self._scope_blend * self._scope_blend * (3.0 - 2.0 * self._scope_blend)

    @property
    def scoped(self):
        """True once fully (or near enough) eased into the scope point - for a later
        overlay to gate itself on, instead of comparing the raw blend fraction itself."""
        return self._scope_blend >= 0.999

    @property
    def drawing(self):
        """True while this weapon's draw animation is still ACTUALLY playing
        on the first-person viewmodel (see ViewModel.is_one_shot_active) -
        fire()/start_reload() both refuse while this is true, same idea as
        `reloading`. Resolves False immediately (nothing to wait for) for a
        weapon with no viewmodel at all (e.g. a remote player's own copy -
        see RemotePlayer) or no "draw" clip loaded."""
        return self._one_shot_still_playing("draw")

    @property
    def racking(self):
        """True from the moment a shot is fired (see fire()) until its "rack" (bolt-
        cycle) animation has actually FINISHED playing - fire() refuses a new shot while
        this is true, same idea as drawing/reloading. Covers two separate waits: the
        time between firing (or a resumed weapon switch - see deactivate()'s own
        comment) and whichever one-shot is currently blocking it ("shoot", or "draw"
        after a resume) actually finishing (_rack_pending - see fire()/update()), since
        starting "rack" any earlier would cut that other one off entirely (there's only
        one active one-shot slot per viewmodel - see ViewModel.play's own docstring),
        and then "rack" actually playing once it starts. Switching away before either
        phase finishes makes the NEXT draw start this all over again from scratch
        instead of silently counting as already cycled (see deactivate()). Always False
        for a weapon with no "rack" clip loaded (every weapon except a bolt-action one -
        see pencil.py's own "rack" entry): _rack_pending still gets set for a beat after
        firing, but the very next update() immediately clears it once nothing's left to
        wait for - same "nothing loaded, nothing to block on" fallback drawing/reloading
        already rely on."""
        return self._rack_pending or self._one_shot_still_playing("rack")

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

    # The particle files this weapon's effects (muzzle_particle, impact_particle)
    # live in, loaded when the game starts: {.pcf path: [system names to load]}
    # (None loads every system in the file). The names those two attributes
    # hold are looked up among what's been loaded.
    particle_files = {}

    # An Explosion (Modules/Weapons/explosion.py) set off where this weapon's
    # shots land, on top of the direct-hit damage; None = no explosion.
    explosion = None

    # An effect (a system in a loaded .pcf) played where this weapon's shots land,
    # instead of the usual bullet impact: its name, an ((r, g, b), (r, g, b)) 0-255
    # colour range replacing the file's own (None keeps them), and a size scale.
    impact_particle = None
    impact_color = None
    impact_size = 1.0

    # The bullet tracer look (a name in Modules/Graphics/tracers.py's STYLES).
    tracer_style = "default"

    # What KIND of damage this weapon deals - right now, just how a player it kills dies
    # (see Modules/Weapons/damage_classes.py's own DamageClass.death_effect). Bullet (gibs)
    # is every weapon's default; a weapon that should dissolve its victims instead sets this
    # to Zap.
    damage_class = Bullet

    # A Modules/Weapons/tracer_spiral.TracerSpiral - a Quake-railgun-style spiral of small
    # star sprites winding around this weapon's own tracer beam - or None (most weapons) for
    # no spiral, just the tracer. Every tunable (colour, tightness, density, size, lifetime)
    # lives on that instance, not here - see spawn_tracer_spiral below and TracerSpiral's own
    # docstring.
    tracer_spiral = None

    # ---- gun-handling sounds ---------------------------------------------
    # {viewmodel state: {frame: sound file}} - played when that state's
    # animation reaches the frame (source frames at handling_fps; see
    # _play_handling_sounds). They follow the gun and have a tiny falloff.
    handling_sounds = {}
    handling_fps = 30.0
    handling_volume = 1.0
    handling_min_distance = 1.0
    handling_max_distance = 6.0

    # ---- damage --------------------------------------------------------
    damage = 10.0             # health taken from a player each shot hits
    headshot_multiplier = 2.0  # ...times this when the shot hits the head
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
    # Mesh node names to load the gun from as SEPARATE objects, one per node -
    # for a gun with several materials (a skinned object keeps only one).
    # None loads the whole gun as one object.
    viewmodel_gun_nodes = None
    # Per-material overrides for the gun's parts, keyed by the material's name in
    # the glb (like add_static's alpha_mode_overrides/roughness_overrides):
    # {name: "OPAQUE" | "MASK" | "BLEND"} and {name: roughness float}.
    viewmodel_alpha_mode_overrides = {}
    viewmodel_roughness_overrides = {}
    # Per-NODE alpha mode, for a node (see viewmodel_gun_nodes) that shares a
    # material with another but needs its own: {node name: mode}. A node listed
    # here ignores viewmodel_alpha_mode_overrides; a None value keeps the mode
    # the glb authored for it.
    viewmodel_node_alpha_mode_overrides = {}
    viewmodel_gun_skin = 1
    worldmodel = f"{_POSE_DIR}/pistolWM.glb"
    # The character joint the world model is held by, and its placement in
    # that joint's frame: (x, y, z) metres and (x, y, z) euler degrees, plus a
    # uniform scale. Tune these to seat the model in the hand.
    worldmodel_bone = "ValveBiped.Bip01_R_Hand"
    worldmodel_position = (0.07, 0.0, 0.02)
    worldmodel_rotation = (-90.0, 0.0, -90.0)
    worldmodel_scale = 1.0
    # A world model with several materials (body/cheese/glass...) loads as one skinned object
    # per mesh node instead of one combined object - same reason and same shape as viewmodel_
    # gun_nodes (a skinned object keeps only ONE material - see skeletal_loader.py's own
    # multi-material scope note). None (most weapons - one material) keeps the existing single-
    # object behavior exactly. Every node needs a skin (this is still add_skeletal underneath,
    # not the static multi-material loader) - a rigid one-bone skin is enough for a prop that
    # doesn't deform, but it still has to actually be skinned in the source file.
    worldmodel_nodes = None
    worldmodel_alpha_mode_overrides = {}
    worldmodel_node_alpha_mode_overrides = {}
    worldmodel_roughness_overrides = {}
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
    # Where the muzzle really is relative to that bone: an (x, y, z) in the bone's
    # own frame and units (None = at the bone). For a gun whose barrel tip isn't
    # where the borrowed bone is.
    viewmodel_muzzle_offset = None

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
    #
    # It can ALSO be (path, source_clip_name) - a string instead of an int -
    # to pick ONE of that file's own named actions by name, instead of the
    # usual "first clip in the file" guess (_first_clip_name): needed for a
    # file exported with SEVERAL actions sharing one rig instead of this
    # project's usual one-clip-per-pose-file convention (e.g. pencil.py's
    # Pencil.glb, which bakes Idle/Shoot/Reload/DrawAction all into ONE
    # file) - the "first clip" guess would otherwise just pick the same one
    # action for every state. A 3-tuple (path, skin_index, source_clip_name)
    # gives both explicitly, for a file that needs a non-default skin AND a
    # specific named action within it. See _parse_clip_spec's own docstring
    # for the exact rules.
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
        self.worldmodel_obj = None     # the FIRST part - see worldmodel_nodes; carries the
                                        # skeleton, so this is what muzzle-bone lookups and
                                        # worldmodel_animations play on regardless of part count
        self.worldmodel_objs = []      # every part (one for a single-material world model too)
        self._viewmodel = None
        self._clips = {}   # (part, state) -> clip name actually loaded
        self._moving = False
        # Whether the last set_moving() call skipped applying the moving-yaw spine
        # correction because an emote was playing - see set_moving's own comment.
        self._moving_offset_suppressed_by_emote = False
        self._scope_blend = 0.0   # 0 = hip-fire, 1 = fully scoped - see update_scope
        self._was_aiming = False  # last frame's `aiming` input - see update_scope's own zoom sound edge-trigger
        # magazine_size <= 0 means "no ammo tracking" (every weapon's old,
        # only behavior) - ammo then just sits unused, can_fire/fire never
        # consult it, and start_reload always refuses (nothing to refill).
        self.ammo = self.magazine_size
        self._reloading = False
        # Set by fire() right after a shot, waiting for "shoot" to finish before "rack"
        # itself can start (see racking's own docstring) - cleared either once "rack"
        # actually starts (update()) or on a weapon switch away mid-wait (deactivate()).
        self._rack_pending = False
        self._handling_last = (None, 0.0)   # (state, animation time) at the last handling-sound check

    # ---- firing --------------------------------------------------------

    @property
    def reloading(self):
        return self._reloading

    def can_fire(self, now=None):
        if self._reloading or self.drawing or self.racking:
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
        Reads the clock, so a HUD can poll it every frame.

        Scaled by scope_accuracy_multiplier once scoped in (has_scope only - a weapon
        that can't scope at all has nothing to ease here), by the exact same smoothstepped
        fraction driving the camera move/FOV/sensitivity (see update_scope) rather than a
        hard on/off switch the instant aiming starts - fully resting on spread_min/max
        when not scoped, and all the way down to scope_accuracy_multiplier's own value
        (0.0 for a perfectly accurate scoped weapon - see pencil.py) once fully eased
        in."""
        now = time.perf_counter() if now is None else now
        quiet = now - self._last_fire - self.spread_recovery_delay
        spread = self._spread_at_last_shot - self.spread_recovery * max(0.0, quiet)
        spread = max(self.spread_min, spread)
        if self.has_scope:
            eased = self._scope_blend * self._scope_blend * (3.0 - 2.0 * self._scope_blend)
            spread *= 1.0 + (self.scope_accuracy_multiplier - 1.0) * eased
        return spread

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
        # Schedules a "rack" (bolt-cycle) animation to start once "shoot" itself
        # finishes (see update()) - NOT right now, since playing it immediately would
        # cut "shoot" off entirely (one active one-shot slot per viewmodel). ONLY for a
        # weapon that actually defines a "rack" entry at all (most don't - this is a
        # bolt-action-specific mechanic, see pencil.py's own) - without this check,
        # EVERY weapon got gated by _rack_pending/racking (see can_fire()) after every
        # shot, waiting on "shoot" to finish plus one more update() tick for a no-op
        # self.play("rack") that has no clip to play - confirmed as a real bug: it
        # silently capped USP's (and every other non-bolt-action weapon's) real rate of
        # fire to "however long its OWN shoot animation takes" instead of its own
        # fire_interval, which happened to read as "as slow as the sniper" since both
        # ended up bottlenecked by an animation length rather than their own numbers.
        # Also only when the magazine isn't now empty from this very shot - an empty
        # magazine reloads instead (see update()'s own auto-reload), and that animation
        # already covers working the bolt, so racking first would just be a redundant
        # extra step before the reload even starts. magazine_size <= 0 (no ammo
        # tracking at all) racks after every shot - there's no "empty" to skip it for.
        self._rack_pending = "rack" in self.viewmodel_animations and (self.magazine_size <= 0 or self.ammo > 0)
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
        # Cancels any rack still owed from a shot BEFORE this reload (see fire()'s own
        # _rack_pending comment) - without this, a manual reload pressed while a rack was
        # still pending/playing left that flag sitting untouched (update()'s reload branch
        # returns early every tick a reload is in progress, so _rack_pending's own check
        # never even runs until reload is done), so it got consumed the moment reload
        # finished and forced an extra rack AFTER it - "Shoot -> Reload -> Rack" instead of
        # the one-or-the-other "Shoot -> Reload OR Rack -> ready" a reload should give:
        # working the bolt to reload already re-chambers on its own, so any older,
        # now-redundant rack debt from before it started should just be forgiven, not
        # carried through to make the player wait out a second animation afterward. Also
        # cuts short an actively PLAYING "rack" one-shot the same way (self.play("reload")
        # right below takes over the viewmodel's one active one-shot slot regardless) -
        # reloading mid-cycle means the bolt's getting worked by the reload instead.
        self._rack_pending = False
        self.play("reload")
        return True

    def play_handling_sound(self, path, position, follow=None):
        """A gun-handling sound (clip out, slide release...): a one-shot that
        rides along with the gun via `follow` (a callable returning the gun's
        current world position), is dropped by the sound manager once it
        finishes, and has a tiny falloff (handling_*_distance)."""
        scene = self._scene
        if scene is None or not path:
            return None
        return scene.sound_manager.add_sound(
            path, glm.vec3(position), volume=self.handling_volume,
            min_distance=self.handling_min_distance, max_distance=self.handling_max_distance,
            loop=False, follow=follow, falloff="inverse",
        )

    def _play_handling_sounds(self, follow):
        """Plays every handling_sounds entry whose frame the viewmodel's current
        one-shot animation has just reached since the last call. Frames are
        source frames (handling_fps); `draw` is sped up by draw_speed at load,
        so its frames are scaled to match."""
        vm = self._viewmodel
        state = vm.one_shot_state if vm is not None else None
        sounds = self.handling_sounds.get(state) if state is not None else None
        if not sounds:
            self._handling_last = (None, 0.0)
            return
        now_time = vm.one_shot_time(state)
        last_state, last_time = self._handling_last
        if last_state != state or now_time < last_time:
            last_time = -1.0
        scale = (1.0 / self.draw_speed) if state == "draw" else 1.0
        for frame, path in sounds.items():
            trigger = frame / self.handling_fps * scale
            if last_time < trigger <= now_time and follow is not None:
                self.play_handling_sound(path, follow(), follow)
        self._handling_last = (state, now_time)

    def update(self, now=None, follow=None):
        """Call once a frame regardless of input (app.py's main loop does).
        follow (a callable returning the gun's world position) is what
        handling sounds are attached to.

        Plays any handling sounds whose frame was just reached, then:
        auto-starts a reload the instant the magazine runs dry (so running
        empty mid-fight reloads on its own, same as most games - no need to
        remember the reload key), and finishes an in-progress reload once its
        OWN animation has actually finished playing (see
        _one_shot_still_playing - a weapon with no viewmodel/no reload clip
        at all finishes instantly, same as it always refilling used to when
        this was a guessed timer), refilling the magazine and dropping back
        to idle. A no-op for a weapon with no ammo tracking at all
        (magazine_size <= 0).

        Also starts a pending "rack" animation (see fire()'s own _rack_
        pending comment) the instant whatever one-shot is currently blocking
        it actually finishes - can't start it any earlier (same one-active-
        one-shot-slot reasoning as every other state here), and checking
        every frame rather than guessing a clip's length is the same "bound
        to the real clip" approach drawing/reloading already use. Normally
        that's "shoot" (fire() just played it); after switching back to a
        weapon that had its rack interrupted mid-cycle (see deactivate()),
        it's "draw" instead (set_active plays that fresh on every re-equip) -
        both are checked so a resumed rack correctly waits for draw to
        finish instead of cutting it short."""
        self._play_handling_sounds(follow)
        if self._reloading:
            if not self._one_shot_still_playing("reload"):
                self._reloading = False
                self.ammo = self.magazine_size
                self.play("idle")
            return
        if self._rack_pending and not self._one_shot_still_playing("shoot") and not self.drawing:
            self._rack_pending = False
            self.play("rack")
        if self.magazine_size > 0 and self.ammo <= 0:
            self.start_reload(now)

    def spawn_tracer_spiral(self, particles, start, end, camera_pos=None):
        """Spawns this weapon's own tracer_spiral (see Modules/Weapons/tracer_spiral.py) along
        this exact shot's beam - a no-op if it doesn't have one (the overwhelming majority of
        weapons). The actual particle maths (the helix shape, the camera-distance coverage
        falloff for a beam too long to spiral at full density) is ParticleManager.spawn_beam_
        spiral's job, in Modules/Particles/particle_system.py - this method's only work is
        handing it this weapon's own configured TracerSpiral instance plus a group cap shared
        across every weapon (so a burst of shots from ANY of them can't pile up unlimited
        spiral effects, the same idea as impacts' own cap). That split is the whole point:
        "spawn particles, and honour how far away the camera is" is a capability every weapon
        gets for free from this base class, while each weapon's own LOOK (colour, tightness,
        density, size, lifetime - see TracerSpiral's own docstring) stays entirely its own
        business, set on its own tracer_spiral attribute and nowhere else.

        particles: the game's ParticleManager (app.py's own `particles`). camera_pos: this
        frame's camera position, if the caller has it - see spawn_beam_spiral's own docstring
        on what it's used for."""
        if self.tracer_spiral is None:
            return None
        return particles.spawn_beam_spiral(start, end, self.tracer_spiral, group="tracer_spiral",
                                           camera_pos=camera_pos)

    def muzzle_position(self, scene):
        """Where this weapon's bullets come out, in the world: the first-person
        gun's muzzle bone while that's what the owner is looking at, otherwise
        the world model's (in the character's hand). None if neither is
        attached/posed yet."""
        gun = self._viewmodel.gun if self._viewmodel is not None else None
        if gun is not None and gun.get("viewmodel_visible") and self.viewmodel_muzzle_bone:
            return scene.joint_world_position(
                gun, self.viewmodel_muzzle_bone, local_offset=self.viewmodel_muzzle_offset)
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

        for state, spec in self.player_animations.items():
            path, skin_index, source_clip = _parse_clip_spec(spec, 0)
            clip = _load_state_clip(scene, player_obj, path, self.player_clip(state),
                                    skin_index=skin_index, source_clip=source_clip)
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
        for obj in self.worldmodel_objs:
            obj["visible_in_color"] = active
            obj["cast_shadow"] = active
        if not active:
            return
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
        instant, blend_duration=0.0) draw cut.

        Also handles an in-progress "rack" (see fire()/update()'s own _rack_pending):
        unlike reload, this does NOT just cancel it - a bolt-action player who switches
        away mid-cycle hasn't actually finished working the bolt, so it's left pending,
        forcing a FRESH rack (from set_active's own "draw" all the way through) the next
        time this weapon comes back out, same as a real bolt-action rifle's own bolt
        doesn't un-cycle itself just because you looked away. self.racking already covers
        both phases this could be interrupted in - still waiting for "shoot"/"draw" to
        finish before "rack" even starts, or "rack" itself actually playing - either way
        becomes a fresh self._rack_pending = True; genuinely not mid-cycle at all (already
        idle) stays False, exactly like today."""
        if self._reloading:
            self._reloading = False
        self._rack_pending = self.racking
        # Snaps back to hip-fire immediately rather than easing out - the NEXT weapon's
        # own view takes over this same frame, so there's nothing left to visibly ease
        # (and a lingering nonzero blend would otherwise silently offset the viewmodel
        # the instant this weapon is drawn again, before update_scope's own first call
        # that frame has a chance to start easing it back down).
        self._scope_blend = 0.0
        self._handling_last = (None, 0.0)
        self.recoil.reset()
        self.set_active(False)

    def _attach_worldmodel(self, scene, player_obj):
        if not self.worldmodel:
            return
        if self.worldmodel_nodes:
            # One skinned object per material node - see worldmodel_nodes' own docstring.
            objs = []
            for node in self.worldmodel_nodes:
                node_modes = self.worldmodel_node_alpha_mode_overrides
                ignore = node in node_modes     # this node opts out of the by-material overrides
                obj = scene.add_skeletal(
                    self.worldmodel, visible_in_color=True, cast_shadow=True, node_names=(node,))
                if obj is None:
                    continue
                name = obj.get("material_name")
                if not ignore and self.worldmodel_alpha_mode_overrides.get(name):
                    obj["alpha_mode"] = self.worldmodel_alpha_mode_overrides[name]
                if node_modes.get(node):
                    obj["alpha_mode"] = node_modes[node]
                if self.worldmodel_roughness_overrides.get(name) is not None:
                    obj["roughness"] = float(self.worldmodel_roughness_overrides[name])
                objs.append(obj)
        else:
            obj = scene.add_skeletal(self.worldmodel, visible_in_color=True, cast_shadow=True)
            objs = [obj] if obj is not None else []
        if not objs:
            return

        rotation = glm.mat4(glm.quat(glm.radians(glm.vec3(*self.worldmodel_rotation))))
        local = (
            glm.translate(glm.mat4(1.0), glm.vec3(*self.worldmodel_position))
            * rotation * glm.scale(glm.mat4(1.0), glm.vec3(self.worldmodel_scale))
        )
        attached = []
        for obj in objs:
            obj["specular_strength"] = 0
            obj["frustum_cull"] = (1.4, 0.4, 2.6)  # same as its owner's (its position is the owner's)
            obj["max_draw_distance"] = 30.0        # a pistol that far away is a couple of pixels
            obj["max_shadow_distance"] = 8.0
            if scene.attach_skeletal(obj, player_obj, self.worldmodel_bone, local) is None:
                scene.remove_skeletal(obj)
                continue
            attached.append(obj)
        if not attached:
            return

        self.worldmodel_objs = attached
        self.worldmodel_obj = attached[0]      # carries the skeleton muzzle-bone lookups use -
                                                # animations still load on EVERY part below (each
                                                # is its own add_skeletal call, so its own skeleton
                                                # - see _load_state_clip's own docstring)
        for state, spec in self.worldmodel_animations.items():
            path, skin_index, source_clip = _parse_clip_spec(spec, 0)
            clip_name = f"{self.player_clip(state)}_wm"
            for obj in attached:
                clip = _load_state_clip(scene, obj, path, clip_name,
                                        skin_index=skin_index, source_clip=source_clip)
                if clip is not None:
                    self._clips[("worldmodel", state)] = clip
        idle = self._clips.get(("worldmodel", "idle"))
        if idle is not None:
            for obj in attached:
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
        pose set by equip_player, crossfading via the usual offset change.

        Ignored entirely while the player model is mid-emote (see PlayerModel.
        play_emote/is_emoting): this correction exists to tweak the WEAPON-
        holding idle pose while walking, which has nothing to do with an
        emote's own full-body clip sharing the same Spine4-rooted joints
        (play_emote widens to that same split - see set_upper_override's own
        docstring) - applying it on top would skew the emote's spine/neck
        orientation. _moving_offset_suppressed_by_emote forces a fresh
        reapply the instant an emote that suppressed this correction ends,
        even if `moving` itself didn't change while it was suppressed -
        otherwise the correction would stay missing until some LATER,
        unrelated moving/not-moving edge happened to trigger it again."""
        model = self._player_model
        currently_emoting = model is not None and model.is_emoting()
        resumed_from_emote = self._moving_offset_suppressed_by_emote and not currently_emoting
        self._moving_offset_suppressed_by_emote = currently_emoting
        moving = horizontal_speed > self.player_moving_speed
        if moving == self._moving and not resumed_from_emote:
            return
        self._moving = moving
        if currently_emoting:
            return
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
        if self._scene is not None:
            for obj in self.worldmodel_objs:
                self._scene.remove_skeletal(obj)
        self.worldmodel_obj = None
        self.worldmodel_objs = []
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
