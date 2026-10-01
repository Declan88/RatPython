import gc
import sys
import time
import pygame
import os
import ctypes
import glm


def _is_frozen_build():
    """True for a Nuitka-compiled run ("__compiled__" injected into this
    entry module's globals - Nuitka's own documented detection method),
    False for an ordinary `python app.py` dev run. Shared by every
    frozen-only check in this file (currently just the asset-path chdir)
    so they can't silently drift out of sync with each other."""
    return "__compiled__" in globals()


def _chdir_for_frozen_build():
    """Nuitka's --onefile mode re-executes itself from its own unpacked
    temp directory before any of this code runs, so sys.executable
    already points at that unpacked location by the time this runs (for
    --standalone too - there's just no separate temp dir to unpack to,
    sys.executable is simply where the built exe already sits). Every
    asset path in this codebase (torus_scene.py's model/texture/audio
    paths, etc.) is a plain relative string like "Assets/Models/...",
    which only resolves correctly if the process's current working
    directory happens to BE that location. Since none of those call
    sites can be reached before this runs (they're all behind imports/
    calls below), chdir'ing here once, before anything else executes,
    makes every existing relative path keep working unmodified instead
    of touching dozens of individual asset-loading call sites."""
    if _is_frozen_build():
        os.chdir(os.path.dirname(os.path.abspath(sys.executable)))


_chdir_for_frozen_build()


def _disable_window_ghosting():
    """Loading blocks the main thread for seconds at a time, and OpenGL only works from that
    thread, so window messages can't be answered meanwhile. If the player clicks the window
    in that time Windows swaps it for a white "ghost" and offers to end the process; this
    tells Windows not to. The clicks simply queue up (and are dropped once loading is done)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.DisableProcessWindowsGhosting()
        except Exception:
            pass


_disable_window_ghosting()

# NOTE on Optimus/switchable-graphics GPU selection for OpenGL: Windows'
# own "High performance" registry hint (Settings > System > Display >
# Graphics, or the equivalent HKCU\Software\Microsoft\DirectX\
# UserGpuPreference key an app could write itself) isn't reliably
# honored for a raw OpenGL context (moderngl/pygame-ce, via WGL) the way
# it is for Direct3D/DXGI apps - an earlier version of this file wrote
# that key directly and confirmed it did nothing for this project, so
# it was removed rather than kept as inert dead code. The mechanism
# that DOES work is exporting two symbols -
# NvOptimusEnablement and AmdPowerXpressRequestHighPerformance - but
# these MUST be in the actual .exe's own PE export table specifically;
# a DLL loaded at runtime does nothing, regardless of how early it's
# loaded here - an earlier version of this file tried exactly that (a
# companion gpu_hint.dll) and it had no effect, and the old PyInstaller
# build needed a patched bootloader to get this right for the same
# underlying reason (that workaround was PyInstaller-specific and
# doesn't carry over to Nuitka).
#
# Nothing in THIS Python file can express that fix - there is no
# per-build-invocation flag or Python-level hook for it - but
# nuitka_gpu_preference_plugin.py (project root, loaded via
# BuildCMD_Nuitka's own --user-plugin= flag) now does, using Nuitka's
# getExtraCodeFiles() plugin hook. Confirmed working (see that file's
# own docstring for the pefile-verified proof and the two non-obvious
# details that made it work: which compile stage a --onefile build's
# "onefile_"-prefixed extra-code-file key actually reaches, and why the
# onefile bootstrap .exe - not some separate extracted exe - is the
# right target in the first place).

from Modules.Window.window import WindowManager
from Modules.Window.loading_screen import show_loading_screen
from Modules.UI import UIManager
from Modules.UI.demo import build_demo
from Modules.UI import Anchor, Crosshair, Hitmarker, Label, ProgressBar
from Modules.UI.death_screen import DeathScreen
from Modules.UI.pause_menu import PauseMenu
from Modules.Graphics.tracers import Tracers
from Modules.Particles import ParticleManager
from Modules.Particles.blood import register_blood
from Modules.Particles.lightning import register_lightning
from Modules.Particles.dissolve_sparks import register_dissolve_sparks
from Modules.Gore import GibManager
from Modules.Debug import FrameProfiler
from Modules.Particles.impacts import (IMPACT_FILE, IMPACT_GROUP, MATERIAL_ALIASES, LIFETIME_SCALE, MAX_IMPACT_EFFECTS,
                                       RADIUS_SCALE, keep_frame, spawn_impact, play_impact_sound)
from Modules.Physics.physics_world import CollisionGroup
from Modules.UI.nametags import NameTags
from Modules.UI.scoreboard import Scoreboard

SPAWN_POSITION = (0.0, 2.0, 3.0)   # where the player (re)spawns: hull centre, metres
RESPAWN_SECONDS = 3.0
# A standing, shootable target near spawn purely for testing the kill feed/
# sound without a second PC - see setup_game/the shooting code's own dummy
# branch. 1 is a reserved id no real Steam64 id can ever equal (those are all
# 17 digits), so it can never collide with a real player's.
TEST_DUMMY_STEAM_ID = 1
TEST_DUMMY_OFFSET = (3.0, 0.0, -2.0)   # metres from SPAWN_POSITION, feet-height
TEST_DUMMY_RESPAWN_SECONDS = 1.5
TEST_DUMMY_HEALTH = 100.0
STARTING_LOADOUT = ["usp", "gouda_gun"]   # weapon ids (see Modules/Weapons/registry.py) everyone spawns with

# /smite's sound pair - see app.py's own spawn_smite. Distances are Minecraft's own attenuation
# (given in blocks) carried straight over as metres, this project's own world scale (a block and
# a metre are both "about one big stride" - the same correspondence Assets/Particles' own
# UNIT_SCALE comment uses converting Source's inches).
#
# NOT the cause of an earlier "can't hear the explosion unless I'm the one smited" report, even
# though it looked like one at first (SoundManager's own falloff="inverse" curve is steeper than
# Minecraft's, and this got pushed as high as 2000m/8,000,000m chasing that theory without it
# helping) - the REAL cause was show_kill_feed's own Kill.wav (universal=True, no distance
# falloff at all, GAIN=4.0 - see KILL_SOUND_GAIN) firing at the same instant for any admin kill
# OTHER than a self-smite (no kill credit for suicide, so it never fires there), reliably
# drowning out this pair regardless of range. Restored to the original conversion now that
# show_kill_feed takes play_sound=False for a zap kill instead.
SMITE_EXPLOSION_SOUND = "Assets/Audio/Effects/Lightning/Explosion3.wav"
SMITE_THUNDER_SOUND = "Assets/Audio/Effects/Lightning/Thunder1.wav"
SMITE_EXPLOSION_RANGE = 16.0        # Minecraft: 16 blocks
SMITE_THUNDER_RANGE = 160_000.0     # Minecraft: 160,000 blocks
HITMARKER_SOUND = "Assets/Audio/Player/hitmarker.wav"
HITMARKER_VOLUME = 1.0   # the sound manager applies a volume twice (on the sound and on its channel), so 1.0 is what plays at full
HITMARKER_HEADSHOT_PITCH = 1.45   # a headshot's hitmarker sound is this much higher (playback speed)
HITMARKER_GAIN = 10.0     # ...so anything louder has to amplify the samples themselves (soft-clipped)
KILL_SOUND = "Assets/Audio/player/Kill.wav"
KILL_SOUND_VOLUME = 1.0
KILL_SOUND_GAIN = 4.0
from Modules.Graphics.paper_doll import PaperDoll
from Modules.UI.killfeed import KillFeed
from Modules.UI.damage_indicator import DamageIndicator
from Modules.UI.chatbox import ChatBox
from Modules.Chat import commands
from Modules.Camera.camera import Camera
from Modules.Camera.camera_boom_arm import CameraBoomArm
from Modules.Scenes.torus_scene import TorusScene
from Modules.Scenes.mainmap_scene import MainMapScene
from Modules.Scenes.main_menu_scene import MainMenuScene
from Modules.Physics.character_controller import CharacterController
from Modules.Player.player_model import PlayerModel
from Modules.Player.rat_colors import RAT_TINT_MASK_PATH
from Modules.Player.viewmodel import ViewModel
from Modules.Weapons import Inventory, get_weapon_class, weapon_ids
from Modules.Weapons.damage_classes import DISSOLVE, get_damage_class
from Modules.Weapons.registry import weapon_id_of
from Modules.Weapons.weapons_base import WeaponsBase
from Modules.UI.weapon_hud import WeaponHUD


def load_steam_api_dll():
    # One shared library name per platform - Valve's Steamworks SDK ships
    # a differently-named/formatted binary for each (steam_api64.dll on
    # Windows, libsteam_api.dylib on macOS - a universal x86_64+arm64
    # binary here, so one file covers both Intel and Apple Silicon -
    # libsteam_api.so on Linux). ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)
    # and RTLD_GLOBAL itself are both real cross-platform POSIX/ctypes
    # concepts, not Windows-specific, so the loading logic below is
    # identical for all three - only the filename differs.
    lib_name = {
        "win32": "steam_api64.dll",
        "darwin": "libsteam_api.dylib",
    }.get(
        sys.platform, "libsteam_api.so"
    )  # covers "linux" and any other POSIX platform

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # _chdir_for_frozen_build() (called at module import time, above this
    # function's own call further down) already puts the process's cwd
    # at the right unpacked location for a Nuitka build on every
    # platform, so the os.getcwd() candidate below covers a frozen build
    # on its own - no separate extraction-folder special-case needed.
    candidates = [
        os.path.join(script_dir, lib_name),
        os.path.join(os.getcwd(), lib_name),
    ]

    env_override = os.environ.get("STEAM_API_DLL_PATH")
    if env_override:
        candidates.insert(0, env_override)

    for path in candidates:
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(os.path.dirname(path))
                print(f"Pre-loaded {lib_name} from: {path}")
                return
            except Exception as e:
                print(f"Found {lib_name} at {path} but failed to load it: {e}")

    print(f"Warning: {lib_name} not found.")


# Pre-load the platform's Steamworks shared library globally before any
# modules import py_steam_net.
load_steam_api_dll()

from Modules.Networking.network_manager import NetworkManager
from Modules.Networking.remote_player import RemotePlayer
from Modules.GameModes import Deathmatch


def main():
    print("Yo wsg")
    window = WindowManager(800, 600, "RatWar")
    camera = Camera(position=(0.0, 0.0, 3.0), aspect=window.width / window.height)

    # Scene classes, not instances - see get_or_load_scene below. Only
    # ever constructed the first time something actually switches to
    # that key, not all up front: TorusScene is kept around purely as a
    # K_1-selectable test/comparison scene (see the switch handler
    # below), and building it (parsing its glb, uploading GPU
    # resources, baking lightmaps) is real, non-trivial work worth
    # skipping entirely for a run that never visits it.
    SCENE_CLASSES = {"torus": TorusScene, "mainmap": MainMapScene, "mainmenu": MainMenuScene}
    scenes = {}

    def get_or_load_scene(key):
        """Returns the already-built scenes[key] if it exists, else
        constructs it first - showing one loading-screen frame right
        before that (still fully synchronous/blocking - see loading_
        screen.py's own docstring for why) construction runs, so
        switching to a not-yet-visited scene reads as "loading"
        instead of as a frozen window."""
        if key not in scenes:
            show_loading_screen(window, f"Loading {key}...")
            scenes[key] = SCENE_CLASSES[key](window.ctx)
            scenes[key].gibs = GibManager(scenes[key], particles)
            # See WindowManager.reset_frame_timer's own docstring - a
            # multi-second construction just ran on this same thread,
            # and the NEXT dt computed would otherwise include all of it.
            window.reset_frame_timer()
        return scenes[key]

    # Gameplay state. Nothing below exists until setup_game() runs (the
    # main menu's Play button) - it is what loads mainmap, so the menu
    # doesn't pay for that load up front. The main loop only reaches any of
    # this after in_menu goes False.
    current_scene_key = None
    current_scene = None
    player = None
    player_height = None
    local_player_model = None
    viewmodel = None  # first-person arms glued to the camera
    weapon = None     # the local player's CURRENT weapon (Modules/Weapons) - inventory.current
    # Every weapon the local player carries, and which one's out (any weapon in the
    # registry can be in it, by id - see Modules/Weapons/inventory.py) - see
    # switch_weapon, called by setup_game (below) for the initial equip and by the
    # scroll-wheel handling further down.
    inventory = Inventory(STARTING_LOADOUT)
    boom_arm = None
    third_person = False
    toggle_third_person = lambda key: None

    def apply_third_person_visibility():
        """The current weapon's world model follows third_person exactly like the local
        player's own body does (see local_player_model's own visible_in_color=False,
        set_visible_in_color calls) - WeaponsBase.set_active always shows it regardless of
        first/third person (a weapon switch has no idea which one is active), so this has
        to be reapplied after anything that could have just turned it back on: the third-
        person toggle itself, and a weapon switch/the very first equip (which defaults to
        visible - correct for third person, wrong for the first-person default). Defined here,
        not inside setup_game, so switch_weapon (a sibling of setup_game, not nested inside
        it) can see it too - it needs weapon/third_person, both already live at this scope."""
        if weapon is not None:
            for obj in weapon.worldmodel_objs:
                obj["visible_in_color"] = third_person
    test_dummy = None  # a standing, shootable RemotePlayer for testing the kill feed/sound alone - see setup_game
    # deaths: the dummy's own death counter (receive_state's "x" - see RemotePlayer);
    # respawn_at: perf_counter() time to show it again, or None while it's up.
    # pos/yaw: its fixed spot, resent on every receive_state call (required every time).
    dummy_state = {"deaths": 0, "respawn_at": None, "pos": None, "yaw": 180.0, "health": 100.0}

    def setup_game(scene_key):
        nonlocal boom_arm, current_scene, current_scene_key, local_player_model, viewmodel, weapon, player, player_height, third_person, toggle_third_person, test_dummy
        current_scene_key = scene_key
        current_scene = get_or_load_scene(current_scene_key)

        # Player capsule: walks/collides against current_scene's static
        # geometry (see torus_scene.py's collision=True statics) via the
        # scene's own physics world. Spawned above the floor so it falls
        # and settles on first update rather than starting embedded in it.
        # max_slope_degrees raised slightly past Source's 45.57 default - the
        # TorusScene staircase's clip-brush ramp (see torus_scene.py) sits at
        # ~46.3 degrees, fit to the actual tread-nosing line rather than a
        # shallower approximation, so it needs a hair more headroom to count
        # as walkable floor instead of a wall.
        # CharacterController.get_eye_offset() (non-crouched, see
        # character_controller.py) puts the camera at height/2 +
        # height*_DEFAULT_EYE_RATIO above the feet, where _DEFAULT_EYE_RATIO
        # is 0.7/1.8 - i.e. eye level = height * 8/9. The rat.glb model's
        # actual measured eye level is 1.39225m above its feet, so height is
        # set here to whatever makes that formula land exactly there
        # (1.39225 * 9/8), rather than picking an arbitrary height and
        # letting the eye level fall wherever the generic ratio puts it -
        # keeps the first-person camera at the same height the visible/
        # shadow-casting model's own eyes actually are.
        player_height = 1.39225 * 9 / 8
        player = CharacterController(
            current_scene.physics,
            position=game_mode.choose_spawn_position(),
            height=player_height,
            max_slope_degrees=47.0,
        )

        # The local player's own visual body - shadow-only (visible_in_color
        # =False) since a first-person player never sees their own model,
        # only what it casts onto the ground. Model path/animation are
        # hardcoded HERE, at the call site, rather than inside PlayerModel
        # itself, so swapping the local player's visual model later is a
        # one-line change - PlayerModel (Modules/Player/player_model.py)
        # stays fully generic, not tied to this one asset.
        # All of this project's rat/rifle animations were baked with
        # Blender's factory-default scene frame rate (24fps) left unchanged,
        # even though they were actually authored/intended for 30fps - so
        # every raw keyframe time in these files is 30/24 = 1.25x too long
        # (each file plays back 25% slower than intended). RAT_ANIM_TIME_SCALE
        # corrects for that by rescaling every keyframe time by 24/30 before
        # it's used - see skeletal_loader.load_skinned_glb/
        # load_animation_clips' own time_scale docstring for the mechanism.
        RAT_ANIM_TIME_SCALE = 24.0 / 30.0

        # Blend-state table: (name, clip, min_speed), min_speed in m/s.
        # Source's own velwalk/velrun (90/220 Source units/s * 0.0254 - the
        # same constants character_controller.py's footstep-sound code uses)
        # decide when "walk"/"run" kick in. rat.glb's own baked-in "New" clip
        # is actually a dancing animation, not an idle one - "rifle_idle"/
        # "rifle_walk"/"rifle_run" (loaded below from the separate pose-only
        # files, sharing rat.glb's armature) are the real poses. All three
        # are full-body clips (legs included, not just arms), so the same
        # table is reused for the upper-body split below too - appending a
        # new blend state later (crouch-walk, sprint, whatever) is just
        # adding another triple here, PlayerModel needs no other change.
        player_animation_states = (
            ("idle", "rifle_idle", 0.0),
            ("walk", "rifle_walk", 90.0 * 0.0254),
            ("run", "rifle_run", 220.0 * 0.0254),
        )

        # Facing-relative directional overrides for the "walk" state only -
        # RifleWalkS/E/W/NE/NW/SE/SW.glb (loaded below) exist alongside
        # RifleWalkN.glb, but there's no equivalent run-direction set yet,
        # so "run" deliberately has no entry here at all and keeps playing
        # rifle_run regardless of direction until those are authored too -
        # see PlayerModel's directional_clips docstring for exactly how a
        # missing state/direction falls back to the plain (or nearest-
        # cardinal) clip.
        player_directional_clips = {
            "walk": {
                "N": "rifle_walk",
                "S": "rifle_walk_s",
                "E": "rifle_walk_e",
                "W": "rifle_walk_w",
                "NE": "rifle_walk_ne",
                "NW": "rifle_walk_nw",
                "SE": "rifle_walk_se",
                "SW": "rifle_walk_sw",
            },
        }

        local_player_model = PlayerModel(
            current_scene,
            "Assets/Models/rat.glb",
            visible_in_color=False,
            cast_shadow=True,
            time_scale=RAT_ANIM_TIME_SCALE,
            tint_mask_path=RAT_TINT_MASK_PATH,
            states=player_animation_states,
            directional_clips=player_directional_clips,
            # Splits the model into a lower body (locomotion - legs, spine,
            # neck, head - above) and an upper body (gun-holding pose)
            # driven independently. Deliberately just the two clavicles, NOT
            # their shared parent Spine4 - Spine4 also parents Neck1 (->
            # Head1), and Bip01_Spine/Spine1/Spine2 above IT drive the torso
            # twist - rooting the split there pulled the spine AND neck/head
            # into the "upper body" mask too, so a gun-holding pose (or the
            # shotgun-idle override) would visibly override the head/neck
            # look-direction and spine lean along with the arms, not just
            # the arms. Rooting at the clavicles instead confines the swap
            # to exactly the shoulder/arm/hand chains - spine, neck, and
            # head always follow the LOWER body's own locomotion clip
            # (idle/walk/run/jump/crouch) no matter what the upper body is
            # doing.
            upper_body_root_joints=[
                "ValveBiped.Bip01_R_Clavicle",
                "ValveBiped.Bip01_L_Clavicle",
            ],
            # A weapon's held pose (set_upper_override - see WeaponsBase.
            # equip_player) uses a WIDER split instead, rooted
            # at Spine4 - which also parents Neck1 (-> Head1) alongside both
            # clavicle chains in this rig, so this pulls in arms, neck, AND
            # head as one unit. Locomotion-driven upper-body poses above
            # stay narrower (clavicles only) on purpose, but an override
            # like pistol_idle was authored with its own Spine4/neck
            # rotation as PART of the pose - splitting it at the clavicles
            # instead fed that clip's clavicle rotation the wrong PARENT
            # transform (whatever Spine4 orientation the current locomotion
            # clip happened to be using), which is what actually made the
            # arm look rotated wrong. Widening the override's own mask to
            # include Spine4 lets it supply its own correct pose for
            # everything above the shoulders, matching how it was actually
            # authored - no manual per-joint rotation correction needed.
            override_upper_body_root_joints=["ValveBiped.Bip01_Spine4"],
            upper_states=player_animation_states,
            # RifleJump.glb (loaded below) - a one-shot takeoff pose, not a
            # speed-driven blend state like the three above, so it's its own
            # param rather than another `states` entry: PlayerModel.update()
            # overrides BOTH bodies with it the instant is_grounded reads
            # False, holding its last frame (see add_skeletal's own loop
            # param) until is_on_ground() is true again, at which point
            # normal idle/walk/run switching resumes.
            jump_animation="rifle_jump",
            # RifleCrouch.glb (loaded below) - a static held pose (2
            # identical keyframes, confirmed via raw glb inspection - not a
            # cycle), not a speed-driven blend state, so it's its own param
            # like jump_animation above rather than another `states` entry.
            # PlayerModel.update() overrides BOTH bodies with it for as long
            # as is_crouched (already passed below via player.is_crouched())
            # reads True, resuming normal idle/walk/run switching the
            # instant it reads False again.
            crouch_animation="rifle_crouch",
        )
        # Matches torus_scene.py's own decorative rat.glb character's
        # shading exactly (same flat "character["specular_strength"] = 0"
        # mutation there) - PlayerModel has no constructor knob for this
        # (bind_material's own default is 1.0, a normal specular highlight),
        # so it's set directly on the obj dict here, same as the decorative
        # one does.
        if local_player_model.obj is not None:
            local_player_model.obj["specular_strength"] = 0
        # First-person arms (see Modules/Player/viewmodel.py) - drawn only
        # while the camera is in first person, tinted with the same fur color.
        viewmodel = ViewModel(current_scene)
        # rifleidle.glb/RifleWalkN.glb are separate pose-only exports sharing
        # rat.glb's own armature (see load_additional_animations) - both
        # files happen to name their one clip "New" (same name rat.glb's own
        # base clip already uses), hence the rename to keep all three
        # distinct on the merged skeleton.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/rifleidle.glb",
            rename={"New": "rifle_idle"},
            time_scale=RAT_ANIM_TIME_SCALE,
        )
        # RifleWalkN.glb specifically has ALREADY been re-exported at a
        # correct 30fps (confirmed: its own keyframes are spaced at exactly
        # 1/30s, unlike rat.glb/rifleidle.glb above, still at 1/24s) - no
        # time_scale correction here, or this would double-correct an
        # already-fixed file and play it 20% too fast.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkN.glb",
            rename={"New": "rifle_walk"},
        )
        # RifleWalkS.glb/RifleWalkE.glb - re-exported since this was first
        # wired in and now BOTH already bake at a correct 30fps (confirmed
        # via raw glb inspection: 28 keyframes/0.9s and 23 keyframes/0.7333s
        # respectively, both exactly 30fps) - no time_scale correction, same
        # as RifleWalkN.glb.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkS.glb",
            rename={"New": "rifle_walk_s"},
        )
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkE.glb",
            rename={"New": "rifle_walk_e"},
        )
        # RifleWalkW.glb - unlike S/E above, still baked at 24fps (confirmed:
        # 18 keyframes over 0.7083s), so it still needs the correction or it
        # plays 25% slower than intended.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkW.glb",
            rename={"New": "rifle_walk_w"},
            time_scale=RAT_ANIM_TIME_SCALE,
        )
        # RifleWalkNE/NW/SE/SW.glb - each of these bundles 34 clips (a full
        # animation library re-export, not just the one directional pose),
        # all byte-identical to each other across the 4 files EXCEPT the one
        # literally named "New" - confirmed by directly comparing every
        # clip's keyframe data across all 4 files. That's the same clip name
        # every other single-pose file here (rifleidle.glb, RifleWalkN.glb,
        # etc.) uses for its real authored pose, so it's used here too rather
        # than the more prominent-looking "Diag" clip, which turned out to be
        # byte-for-byte identical junk shared by all 4 files (a leftover
        # reference/placeholder track, not the actual walk motion). All 4
        # confirmed at 24fps via the same keyframe-spacing check as
        # RifleWalkW.glb above, so all 4 need the correction.
        for _direction, _filename in (
            ("ne", "RifleWalkNE.glb"),
            ("nw", "RifleWalkNW.glb"),
            ("se", "RifleWalkSE.glb"),
            ("sw", "RifleWalkSW.glb"),
        ):
            current_scene.load_additional_animations(
                local_player_model.obj,
                f"Assets/Animations/Poses/Rifle/{_filename}",
                rename={"New": f"rifle_walk_{_direction}"},
                time_scale=RAT_ANIM_TIME_SCALE,
            )
        # RifleRunN.glb - also already baked at a correct 30fps (confirmed
        # the same way as RifleWalkN.glb), no time_scale correction needed.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleRunN.glb",
            rename={"New": "rifle_run"},
        )
        # RifleJump.glb - confirmed already baked at a correct 30fps (same
        # raw-keyframe-spacing check as RifleWalkN.glb/RifleRunN.glb), no
        # time_scale correction needed.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleJump.glb",
            rename={"New": "rifle_jump"},
        )
        # RifleCrouch.glb - a static pose (its own first/last keyframe are
        # identical, confirmed via raw glb inspection), so its authored
        # frame rate/duration don't matter - no time_scale correction needed.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleCrouch.glb",
            rename={"New": "rifle_crouch"},
        )
        # Third-person mode - see Modules/Camera/camera_boom_arm.py. Off by
        # default (first person): the local player's own model stays
        # shadow-only (visible_in_color=False above) until this is toggled.
        third_person = False
        boom_arm = CameraBoomArm(current_scene.physics)

        def toggle_third_person(key):
            nonlocal third_person
            if key == pygame.K_v:
                third_person = not third_person
                # The local player's body is normally shadow-only (a first-
                # person player never sees their own model - see
                # local_player_model's own visible_in_color=False above) -
                # third person needs it actually drawn instead.
                local_player_model.set_visible_in_color(third_person)
                apply_third_person_visibility()

        # The current weapon decides everything about how the player holds it:
        # its gun and arm animations on the first-person viewmodel, its world
        # model in the character's hand, and the pose of the player rig's upper
        # body (plus that pose's rotation corrections - see WeaponsBase).
        # force=True: a fresh map means a fresh scene/player_model to equip
        # onto even if the SLOT itself (inventory.index) hasn't changed - the
        # ammo/reload state on the weapon itself carries over unaffected
        # (the inventory is built once, outside setup_game, not recreated
        # per map).
        switch_weapon(inventory.index, force=True)
        # Prime every OTHER slot too (loads its models/animations WITHOUT
        # showing them or disturbing the real current weapon's pose - see
        # equip_player/equip_viewmodel's own activate=False), so the FIRST
        # scroll to it is just as instant as every switch after - see
        # switch_weapon's own docstring for why a switch between
        # already-primed weapons has no loading cost at all.
        for other in inventory.weapons:
            if other is not weapon:
                prime_weapon(other)
        # Bake every weapon's clips now, so the first draw/reload/switch doesn't
        # hitch (see ViewModel.warm_animations).
        viewmodel.warm_animations()

    def prime_weapon(other):
        """Loads a carried weapon's models/animations without showing it (see setup_game)."""
        other.equip_viewmodel(viewmodel, activate=False)
        other.equip_player(current_scene, local_player_model, activate=False)

    def on_weapon_added(new_weapon):
        if viewmodel is not None and current_scene is not None:      # in a match: load it now, not at first use
            prime_weapon(new_weapon)
            viewmodel.warm_animations()
    inventory.on_added = on_weapon_added

    def give_weapon(weapon_id):
        """/give: adds a weapon (by registry id) to the inventory."""
        if get_weapon_class(weapon_id) is None:
            return f"No weapon '{weapon_id}'. Available: {', '.join(weapon_ids())}."
        if inventory.add(weapon_id) is None:
            return "You already have that (or the inventory is full)."
        return f"Gave {get_weapon_class(weapon_id).name}."

    def switch_weapon(slot_index, force=False):
        """Switches to inventory slot slot_index (a no-op if it's already
        out, unless force) - the initial equip in setup_game above (which
        also primes every OTHER slot up front, right after), and every
        later scroll-wheel switch (see the main loop).

        The outgoing weapon (if any) is only DEACTIVATED (hidden), not
        unequipped - WeaponsBase.equip_player/equip_viewmodel both notice a
        weapon that's already primed for the current scene and just
        reactivate it instead of reloading anything, which is what makes
        switching back and forth near-instant instead of the load-time
        stutter a full unequip/re-equip cycle used to cause on every single
        switch (see WeaponsBase.equip_player's own docstring)."""
        nonlocal weapon
        if not force and weapon is not None and slot_index % len(inventory) == inventory.index:
            return
        if weapon is not None:
            weapon.deactivate()
        weapon = inventory.select(slot_index)
        net_mgr.local_weapon = weapon_id_of(type(weapon))      # so other players draw the same gun
        weapon.equip_viewmodel(viewmodel)
        weapon.equip_player(current_scene, local_player_model)
        # equip_player always shows the new weapon's world model (WeaponsBase.set_active has no
        # idea whether the local player is first- or third-person) - mask it back down to match,
        # same as toggle_third_person does for the toggle itself. Covers the very first equip
        # too (third_person defaults False, so this hides it immediately instead of it being
        # visible in first person until the player happens to press V once).
        apply_third_person_visibility()

    def spawn_test_dummy():
        """The test dummy: reuses RemotePlayer wholesale (the same model,
        hitboxes - body AND head, for headshot testing too - weapon, gib-on-
        death and auto-hide/show-again machinery a real remote player gets)
        rather than building any of that again - see the shooting code's own
        dummy branch for how a hit on it is turned into a kill without any
        network traffic at all (it isn't a real match participant). NOT
        spawned automatically - only the /adddummy command (see
        Modules/Chat/commands.py) calls this, so it never shows up
        uninvited. Calling it again while one's already up replaces it
        (fresh position/state) rather than stacking a second one."""
        nonlocal test_dummy
        if current_scene is None:
            return "No map loaded."
        if test_dummy is not None:
            test_dummy.destroy()
        dummy_xz = (SPAWN_POSITION[0] + TEST_DUMMY_OFFSET[0], SPAWN_POSITION[2] + TEST_DUMMY_OFFSET[2])
        ground = current_scene.physics.raycast(
            glm.vec3(dummy_xz[0], SPAWN_POSITION[1] + 5.0, dummy_xz[1]),
            glm.vec3(dummy_xz[0], SPAWN_POSITION[1] - 20.0, dummy_xz[1]),
            CollisionGroup.STATIC,
        )
        dummy_feet_y = ground.position.y if ground is not None else SPAWN_POSITION[1] + TEST_DUMMY_OFFSET[1]
        dummy_state["deaths"] = 0
        dummy_state["health"] = TEST_DUMMY_HEALTH
        dummy_state["respawn_at"] = None
        dummy_state["pos"] = (dummy_xz[0], dummy_feet_y, dummy_xz[1])
        test_dummy = RemotePlayer(current_scene, TEST_DUMMY_STEAM_ID)
        test_dummy.name = "Test Dummy"
        test_dummy.receive_state({
            "p": list(dummy_state["pos"]), "y": dummy_state["yaw"], "v": 0.0,
            "c": 0, "g": 1, "s": 0, "d": [0.0, 0.0], "j": 0, "x": 0, "a": 1,
        })
        def dummy_died(pos, vel, color, damage_class):
            if get_damage_class(damage_class).death_effect == DISSOLVE:
                pass   # the RemotePlayer dissolve state machine (shared - test_dummy IS one) handles the body itself
            elif current_scene.gibs is not None:
                current_scene.gibs.spawn(pos, vel, push=dummy_state.get("push"), tint=color)
            if damage_class == "zap":
                # Keyed on the damage class name directly, not a separate "was this a smite"
                # flag - today Zap is ONLY ever dealt by /smite, so they're equivalent; if a
                # future weapon deals Zap damage WITHOUT wanting the full lightning-strike
                # fanfare, this is the place to split that back out into its own flag.
                spawn_smite(pos)
        test_dummy.on_death = dummy_died
        test_dummy.on_dissolve_spark = spawn_dissolve_spark_burst
        test_dummy.on_dissolve_arc = spawn_dissolve_arc
        return "Test dummy spawned."

    ui = UIManager(window)
    ui_demo = build_demo(ui)

    # TEMPORARY health bar - a placeholder until a real damage system exists.
    # H takes 10 damage, J heals 10 (see on_key_down); nothing else reads or
    # writes `health` yet. Hidden on the main menu, shown once a map starts.
    name_tags = NameTags(ui)
    scoreboard = Scoreboard(ui)
    health ={"value": 100.0, "max": 100.0}
    health_bar = ProgressBar(
        value=health["value"], max_value=health["max"], label_format="{value:.0f} / {max:.0f}",
        anchor=Anchor.BOTTOM_LEFT, offset=(130, -49), size=(300, 26), visible=False,
        fill_color=(90, 200, 110, 255),
    )
    ui.root.add(health_bar)
    # Screen-centre crosshair; its gap is the current weapon's real spread.
    crosshair = ui.root.add(Crosshair(visible=False))
    # Current weapon + ammo, bottom-right, with the other weapon slots shown
    # dim above it - see Modules/UI/weapon_hud.py. Shown/hidden alongside
    # the crosshair; updated every frame from the shooting code below.
    weapon_hud = ui.root.add(WeaponHUD())
    # Diagonal ticks over the crosshair when one of our shots hits another player.
    hitmarker = ui.root.add(Hitmarker())
    # Arrow on a ring around the crosshair pointing at whoever just shot US (see
    # Modules/UI/damage_indicator.py) - updated every frame from the camera below,
    # triggered from on_damaged.
    damage_indicator = ui.root.add(DamageIndicator())
    # F9: a live breakdown of where each frame's time goes (see Modules/Debug/frame_profiler.py).
    prof = FrameProfiler()
    profile_label = ui.root.add(Label("", font_size=17, color=(255, 255, 170, 255), visible=False, offset=(14, 14)))
    hit_sound_channel = [None]   # every hit plays on the same channel, cutting off the last
    kill_sound_channel = [None]  # ditto, for Kill.wav
    # Kill feed, top-left - see net_mgr.on_killfeed wiring below.
    kill_feed = ui.root.add(KillFeed())
    # Text chat + admin commands (Modules/Chat/commands.py) - Enter opens it
    # (see on_key_down), bottom-left. commands_ctx is filled in below once
    # change_health/spawn_test_dummy exist; the callables inside are only
    # ever invoked later, once chat is actually used, so the empty dict here
    # is fine in the meantime (mirrors particles.raycast's own forward
    # reference to current_scene just above setup_game).
    commands_ctx = {}
    chatbox = ui.root.add(ChatBox(
        is_admin=lambda: commands.is_admin(net_mgr.local_steam_id),
        local_name=lambda: net_mgr.display_name(net_mgr.local_steam_id),
        send_chat=lambda text: net_mgr.send_chat(text),
        ctx=commands_ctx,
    ))

    # Dying: health reaching 0 (from any source) flags `pending`; the main loop starts
    # the death at a safe point (begin_death) and ends it (end_death) when the
    # screen's countdown runs out. `eye_drop` eases the camera down to the floor.
    death = {"pending": False, "active": False, "eye_drop": 0.0, "push": None, "push_time": 0.0,
             "smite": False,          # set by smite_self right before the killing change_health call
             "damage_class": "bullet"}   # set by whatever's ABOUT to kill us - on_damaged/kill_self/
                                          # smite_self/explode's own self-damage branch, right before
                                          # each one's own change_health call - begin_death reads and
                                          # resets it once the death it names actually happens
    death_screen = ui.root.add(DeathScreen(RESPAWN_SECONDS))
    # ESC in a game: resume / disconnect to the main menu / quit. Added last so it draws on top.
    pause = {"on": False, "quit": False}
    pause_menu = ui.root.add(PauseMenu(
        on_resume=lambda: set_paused(False), on_disconnect=lambda: disconnect(),
        on_quit=lambda: pause.update(quit=True)))

    def change_health(delta):
        if death["active"] and delta < 0.0:
            return   # already dead
        health["value"] = max(0.0, min(health["max"], health["value"] + delta))
        health_bar.value = health["value"]
        low = health["value"] <= health["max"] * 0.3
        health_bar.fill_color = (220, 70, 70, 255) if low else (90, 200, 110, 255)
        if health["value"] <= 0.0 and not death["active"]:
            death["pending"] = True

    # The game starts on the main menu and loads nothing else. Host/Join
    # (see Modules/UI/lobby_menu.py) talk to NetworkManager - created here,
    # before any map exists, since the menu needs Steam callbacks pumped -
    # and once a lobby is hosted/joined the menu reports the chosen map via
    # request_start. The real work (setup_game: map, player, models, each
    # behind its own loading frame) runs from the main loop, not inside
    # that callback, which can fire from inside Steam's callback dispatch.
    MAPS = [("Main Map", "mainmap"), ("Torus (test)", "torus")]
    in_menu = True
    pending_start = None
    net_mgr = NetworkManager(camera, None)
    # The active game mode: decides (re)spawn position and owns scoring's
    # display (see Modules/GameModes/game_mode.py) - a single hardcoded
    # spawn point, same as SPAWN_POSITION always was, until a map defines
    # real spawn markers to hand it instead.
    game_mode = Deathmatch(net_mgr, spawn_points=[SPAWN_POSITION])

    def dummy_or_display_name(steam_id):
        return "Test Dummy" if steam_id == TEST_DUMMY_STEAM_ID else net_mgr.display_name(steam_id)

    def show_kill_feed(killer_id, victim_id, weapon_name, headshot, play_sound=True):
        """A row in the kill feed (see Modules/UI/killfeed.py) - shown to
        EVERY player for EVERY kill (net_mgr.on_killfeed below fires this the
        same way on every client - see NetworkManager's own _broadcast_kill/
        _receive_killfeed), plus the local-only "we got a kill" feedback -
        Kill.wav, universal=True like the hitmarker's own feedback sound, so
        only WE hear it, not something other players' games play - for a kill
        WE scored. Also the path a hit on TEST_DUMMY takes directly (see the
        shooting code below), bypassing the network entirely since that's a
        local-only target, not a real match kill.

        play_sound=False skips Kill.wav specifically - kill_dummy passes this for a smite
        (damage_class == "zap"): KILL_SOUND_GAIN=4.0 and universal=True (always full volume,
        no distance falloff at all) means it reliably drowns out spawn_smite's own positional
        explosion/thunder pair playing at the very same instant - confirmed by a smite on
        anyone OTHER than yourself (which routes through here) being effectively inaudible
        over it, while smiting yourself (which never calls show_kill_feed at all - no kill
        credit for a suicide) let the explosion come through clearly. The visual kill feed
        entry itself is unaffected either way."""
        killer_name = dummy_or_display_name(killer_id)
        victim_name = dummy_or_display_name(victim_id)
        kill_feed.add_kill(
            killer_name, victim_name, weapon_name, headshot,
            killer_mine=killer_id == net_mgr.local_steam_id,
            victim_mine=victim_id == net_mgr.local_steam_id,
        )
        if play_sound and killer_id == net_mgr.local_steam_id and current_scene is not None:
            kill_emitter = current_scene.sound_manager.add_sound(
                KILL_SOUND, camera.position, volume=KILL_SOUND_VOLUME, loop=False,
                universal=True, channel=kill_sound_channel[0], gain=KILL_SOUND_GAIN)
            kill_sound_channel[0] = kill_emitter["channel"]

    def on_killfeed(killer_id, victim_id, weapon_name, headshot):
        show_kill_feed(killer_id, victim_id, weapon_name, headshot)
        game_mode.on_kill(killer_id, victim_id, weapon_name)
    net_mgr.on_killfeed = on_killfeed
    net_mgr.on_chat = lambda sender_id, text: chatbox.push_message(net_mgr.display_name(sender_id), text)
    # An admin's /kill <player> targeting a REAL remote player (see
    # kill_player below) arrives here on THAT player's own client - we only
    # ever act on it if the sender really is the hardcoded admin (checked
    # independently on this end, not trusted from the sender - see
    # NetworkManager.send_admin_kill's own docstring).
    net_mgr.on_admin_kill = (
        lambda sender_id, damage_class="bullet":
            (smite_self() if damage_class == "zap" else kill_self()) if commands.is_admin(sender_id) else None)

    def damage_dummy(amount, weapon_name="", headshot=False, push=None, damage_class="bullet"):
        """Hurts the test dummy like a real player: it has TEST_DUMMY_HEALTH and only
        dies (kill_dummy) once that's gone. Ignored if there's no dummy or it's already down."""
        if test_dummy is None or test_dummy.dead:
            return
        dummy_state["health"] -= amount
        if dummy_state["health"] <= 0.0:
            kill_dummy(weapon_name, headshot, push=push, damage_class=damage_class)

    def spawn_smite(position):
        """The /smite effect at `position` - a lightning strike (Modules/Particles/lightning.py:
        a bright flash + a burst of electric arcs) plus the explosion/thunder sound pair, at
        this project's own converted-from-Minecraft attenuation (see SMITE_EXPLOSION_RANGE/
        SMITE_THUNDER_RANGE's own comment). Always called LOCALLY, once per client that needs
        to actually show/hear it - the local player's own death (begin_death), the test
        dummy's (kill_dummy), and every remote player's (remote_death, driven by the smite
        flag on their death packet - see NetworkManager.notify_death) - never sent over the
        network itself; the DEATH is what's replicated, and each client reacts to it."""
        particles.spawn("smite_flash", position)
        particles.spawn("smite_bolt", position)
        particles.spawn("smite_bolt_sky", position)
        particles.spawn("smite_crackle", position)
        current_scene.sound_manager.add_sound(
            SMITE_EXPLOSION_SOUND, position, min_distance=1.0, max_distance=SMITE_EXPLOSION_RANGE,
            loop=False, falloff="inverse")
        current_scene.sound_manager.add_sound(
            SMITE_THUNDER_SOUND, position, min_distance=1.0, max_distance=SMITE_THUNDER_RANGE,
            loop=False, falloff="inverse")

    def spawn_dissolve_spark_burst(position):
        """One spark/glow burst at `position` (Modules/Particles/dissolve_sparks.py) - the
        ambient dust, fired LOCALLY, purely visual, every DISSOLVE_SPARK_INTERVAL while a
        Zap-dissolving body is still animating (see RemotePlayer.on_dissolve_spark, wired up
        below for both the test dummy and every real remote player) - same "each client
        reacts locally, nothing extra goes over the network" shape as spawn_smite above."""
        particles.spawn("dissolve_spark", position)
        particles.spawn("dissolve_glow", position)

    def spawn_dissolve_arc(position):
        """One tesla-arc burst at `position` (Modules/Particles/dissolve_sparks.py) - the
        actual "electricity" read, kept separate from spawn_dissolve_spark_burst above since
        RemotePlayer fires this one several times at once from different points on the body,
        on its own (faster-ramping) interval - see RemotePlayer.on_dissolve_arc/
        DISSOLVE_ARC_INTERVAL_START's own docstring."""
        particles.spawn("dissolve_arc", position)

    def kill_dummy(weapon_name="", headshot=False, push=None, damage_class="bullet"):
        """Kills the test dummy right now, if one's up and not already down -
        shared by a real hit on it (see the shooting code below) and the
        /kill (or /smite) command's own player-target path (kill_player/smite_player).
        damage_class picks gibs vs. a dissolve - see dummy_died/Modules/Weapons/
        damage_classes.py."""
        if test_dummy is None:
            return "No test dummy - use /adddummy first."
        if test_dummy.dead:
            return "The test dummy is already down."
        dummy_state["deaths"] += 1
        dummy_state["push"] = push      # the gibs' shove (see spawn_test_dummy's on_death)
        test_dummy.receive_state({
            "p": list(dummy_state["pos"]), "y": dummy_state["yaw"], "v": 0.0,
            "c": 0, "g": 1, "s": 0, "d": [0.0, 0.0], "j": 0,
            "x": dummy_state["deaths"], "a": 0, "dc": str(damage_class),
        })
        dummy_state["respawn_at"] = time.perf_counter() + TEST_DUMMY_RESPAWN_SECONDS
        show_kill_feed(net_mgr.local_steam_id, TEST_DUMMY_STEAM_ID, weapon_name, headshot,
                       play_sound=damage_class != "zap")
        return "Test dummy killed."

    def explode(source_weapon, hit):
        """Sets off source_weapon.explosion where `hit` landed: everyone inside its radius
        takes damage (the shooter only a proportion of it, see Explosion) and, if it kills
        them, their gibs are thrown away from the blast."""
        explosion = source_weapon.explosion
        origin = glm.vec3(hit.position) + glm.vec3(hit.normal) * 0.1     # just off the surface
        targets = [(steam_id, other.body_center) for steam_id, other in net_mgr.remote_players.items()
                   if not other.dead]
        if test_dummy is not None and not test_dummy.dead:
            targets.append((TEST_DUMMY_STEAM_ID, test_dummy.body_center))
        me = None if death["active"] or death["pending"] else (net_mgr.local_steam_id, player.get_position())
        blast_hits = explosion.hits(
            origin, targets, invoker=me,
            blocked=lambda a, b: current_scene.physics.raycast(a, b, CollisionGroup.STATIC) is not None)
        hurt_someone = False
        for blast in blast_hits:
            if blast.is_invoker:
                death["damage_class"] = source_weapon.damage_class.name
                change_health(-blast.damage)
                # Self-damage from our OWN explosion never goes through net_mgr.on_damage at
                # all (nothing to send - we already know about it) - that's the only other
                # place a hit normally triggers the damage indicator, so it needs triggering
                # directly here instead, pointed at the blast itself (origin), not some
                # notion of "attacker" (that's also us, standing wherever we fired FROM, not
                # where the blast actually was - origin is the only reading that makes sense
                # for your own splash catching you).
                to_blast = origin - camera.position
                damage_indicator.trigger((to_blast.x, to_blast.z))
            elif blast.target_id == TEST_DUMMY_STEAM_ID:
                hurt_someone = True
                damage_dummy(blast.damage, source_weapon.name, False, push=blast.push,
                            damage_class=source_weapon.damage_class.name)
            else:
                hurt_someone = True
                net_mgr.send_damage(blast.target_id, blast.damage, source_weapon.name, push=blast.push,
                                    damage_class=source_weapon.damage_class.name, origin=tuple(origin))
        if hurt_someone:
            hitmarker.trigger(headshot=False)

    def kill_self():
        """The /kill command's default (no argument) - suicide, for testing
        death/respawn without needing something around to shoot you. Also
        what a REMOTE admin-kill (see net_mgr.on_admin_kill above) runs on
        the targeted player's own client."""
        if death["active"] or death["pending"]:
            return "You're already dead."
        # Explicit, not relying on whatever's left over from a previous non-fatal hit - a
        # player who took Zap damage earlier and survived, then used /kill on themselves,
        # should die a plain death, not dissolve from a hit that didn't actually kill them.
        death["damage_class"] = "bullet"
        change_health(-(health["max"] + 1.0))
        return "You died."

    def smite_self():
        """/smite's default (no argument) - same as kill_self but marks the death as a smite
        (begin_death reads death["smite"] and both shows the lightning locally and tells
        notify_death to replicate it) and its damage class as Zap (dissolve instead of gibs -
        see begin_death/remote_death's own damage_class branch) - also what a remote
        /smite <player> (see net_mgr.on_admin_kill above) runs on the targeted player's own
        client."""
        if death["active"] or death["pending"]:
            return "You're already dead."
        death["smite"] = True
        death["damage_class"] = "zap"
        change_health(-(health["max"] + 1.0))
        return "Smitten."

    def kill_targets():
        """Candidate names for /kill's own argument (see commands.Command's
        arg_source docstring) - yourself, the test dummy if it's up, and
        every other connected player."""
        names = [net_mgr.display_name(net_mgr.local_steam_id)]
        if test_dummy is not None and not test_dummy.dead:
            names.append("Test Dummy")
        names.extend(
            player.name or net_mgr.display_name(steam_id)
            for steam_id, player in net_mgr.remote_players.items()
        )
        return names

    def kill_player(name):
        """/kill <name>'s own dispatch: resolves name (case-insensitive) to
        yourself, the test dummy, or another connected player, and kills
        whichever it is. A remote player can only ever be told to kill
        THEMSELVES (see NetworkManager.send_admin_kill) - there's no way to
        reach into their game and do it directly - so that branch can only
        ever report the request was sent, not that it actually happened."""
        target = name.strip().lower()
        if target in ("", "me", "you", "self", net_mgr.display_name(net_mgr.local_steam_id).lower()):
            return kill_self()
        if target in ("dummy", "test dummy"):
            return kill_dummy("/kill")
        for steam_id, player in net_mgr.remote_players.items():
            if (player.name or "").lower() == target:
                if net_mgr.send_admin_kill(steam_id):
                    return f"Kill request sent to {player.name}."
                return f"Couldn't reach {player.name}."
        return f"No connected player named '{name}'."

    def smite_player(name):
        """/smite <name>'s own dispatch - identical shape to kill_player, just marking the
        death (or the remote request) as a smite instead of a plain kill."""
        target = name.strip().lower()
        if target in ("", "me", "you", "self", net_mgr.display_name(net_mgr.local_steam_id).lower()):
            return smite_self()
        if target in ("dummy", "test dummy"):
            return kill_dummy("/smite", damage_class="zap")
        for steam_id, player in net_mgr.remote_players.items():
            if (player.name or "").lower() == target:
                if net_mgr.send_admin_kill(steam_id, damage_class="zap"):
                    return f"Smite request sent to {player.name}."
                return f"Couldn't reach {player.name}."
        return f"No connected player named '{name}'."

    commands_ctx["add_dummy"] = spawn_test_dummy
    commands_ctx["give_weapon"] = give_weapon
    commands_ctx["weapon_ids"] = weapon_ids
    commands_ctx["kill_self"] = kill_self
    commands_ctx["kill_player"] = kill_player
    commands_ctx["kill_targets"] = kill_targets
    commands_ctx["smite_self"] = smite_self
    commands_ctx["smite_player"] = smite_player
    tracers = Tracers(window.ctx)
    particles = ParticleManager(window.ctx, material_aliases=MATERIAL_ALIASES,
                                radius_scale=RADIUS_SCALE, frame_filter=keep_frame,
                                lifetime_scale=LIFETIME_SCALE)
    particles.group_limits[IMPACT_GROUP] = MAX_IMPACT_EFFECTS     # at most this many impacts alive at once
    particles.group_limits["tracer_spiral"] = 8   # a burst of shots can't pile up unlimited spiral effects
    particles.load("Assets/Particles/Muzzle/muzzleflashes.pcf")
    particles.load(IMPACT_FILE)
    # Whatever particle files the weapons ask for (see WeaponsBase.particle_files).
    for weapon_id in weapon_ids():    # every weapon, not just carried ones: other players' guns too
        for pcf_path, system_names in get_weapon_class(weapon_id).particle_files.items():
            particles.load(pcf_path, only=system_names)
    register_blood(particles)
    register_lightning(particles)
    register_dissolve_sparks(particles)
    # Particles that collide (impact debris) trace against the level only.
    particles.raycast = lambda a, b: (
        current_scene.physics.raycast(a, b, CollisionGroup.STATIC) if current_scene is not None else None)
    particles.raycast_fast = lambda ax, ay, az, bx, by, bz: (
        current_scene.physics.raycast_light(ax, ay, az, bx, by, bz) if current_scene is not None else None)

    def remote_shot(start, end, follow=None, shooter_weapon=None):
        """Another player's shot, drawn with THEIR weapon's looks (tracer, impact, muzzle
        flash - whichever gun they hold, see RemotePlayer): its tracer, a muzzle flash that
        stays on their gun, and the impact where it ended (looked up here - only the end
        point is sent)."""
        shooter_weapon = shooter_weapon or WeaponsBase
        tracers.add(start, end, style=shooter_weapon.tracer_style)
        shooter_weapon.spawn_tracer_spiral(particles, start, end, camera_pos=camera.position)
        aim = glm.vec3(end) - glm.vec3(start)
        if glm.length(aim) > 1e-3:
            if current_scene is not None:
                hit = current_scene.physics.raycast(start, glm.vec3(end) + glm.normalize(aim) * 0.1)
                if shooter_weapon.impact_particle and hit is not None:
                    particles.spawn_surface(
                        shooter_weapon.impact_particle, hit.position, hit.normal,
                        colors=shooter_weapon.impact_color, size=shooter_weapon.impact_size, group=IMPACT_GROUP)
                    play_impact_sound(current_scene.sound_manager, hit)
                else:
                    spawn_impact(particles, hit, sound_manager=current_scene.sound_manager)
            if shooter_weapon.muzzle_particle:
                particles.spawn(shooter_weapon.muzzle_particle, start, forward=aim,
                                colors=shooter_weapon.muzzle_color, size=shooter_weapon.muzzle_size,
                                offset_scale=shooter_weapon.muzzle_offset_scale, follow=follow)
    net_mgr.on_tracer = remote_shot

    def remote_footstep(material, position, volume):
        """Another player's footstep, replicated the same way as their shots/jumps - see
        NetworkManager.notify_footstep/RemotePlayer's own docstrings."""
        if current_scene is not None:
            current_scene.play_footstep_sound(material, position, volume=volume)
    net_mgr.on_footstep = remote_footstep
    net_mgr.on_dissolve_spark = spawn_dissolve_spark_burst
    net_mgr.on_dissolve_arc = spawn_dissolve_arc

    def remote_death(position, velocity, color=None, damage_class="bullet"):
        """Another player died: their body bursts into gibs (in their fur colour) where they
        stood, UNLESS damage_class dissolves instead (Zap - see Modules/Weapons/
        damage_classes.py; RemotePlayer's own dissolve state machine handles the body itself,
        already running by the time this fires - see its _set_dead) - and if it was a
        /smite specifically, the lightning effect too (see dummy_died's own comment on why
        that's keyed on the damage class name rather than a separate flag). spawn_smite
        itself is never called over the network - each client reacts to the replicated
        death/damage_class on its own."""
        if current_scene is None:
            return
        if get_damage_class(damage_class).death_effect != DISSOLVE and current_scene.gibs is not None:
            current_scene.gibs.spawn(position, velocity, tint=color)
        if damage_class == "zap":
            spawn_smite(position)
    net_mgr.on_death = remote_death
    # Another player's shot hit us: the shooter decided that, we apply it.
    def on_damaged(amount, attacker_id, weapon_name, headshot, push=None, damage_class="bullet", origin=None):
        death["damage_class"] = damage_class   # what killed us, if this hit does - see begin_death
        change_health(-amount)
        if push is not None:      # an explosion's shove: if this kills us, our gibs fly that way
            death["push"], death["push_time"] = glm.vec3(*push), time.perf_counter()
        # Points the damage indicator at wherever this actually came from. origin (the
        # shooter's own hit/blast position - see network_manager.py's send_damage docstring)
        # is preferred when given: exact for a direct hit, and the only correct choice for
        # splash damage (an explosion can reach someone standing well off to the side of
        # wherever the attacker themselves is). Falls back to the attacker's own CURRENT
        # position (looked up live off their replicated state - body_center is continuously
        # updated from their movement stream) only for an older/odd message with no origin;
        # None either way (they disconnected/despawned between firing and this landing, or
        # this is the test dummy, which isn't a RemotePlayer at all, AND no origin was sent)
        # just skips showing an arrow rather than guessing.
        if origin is not None:
            to_source = glm.vec3(*origin) - camera.position
            damage_indicator.trigger((to_source.x, to_source.z))
        else:
            attacker = net_mgr.remote_players.get(attacker_id)
            if attacker is not None:
                to_attacker = attacker.body_center - camera.position
                damage_indicator.trigger((to_attacker.x, to_attacker.z))
    net_mgr.on_damage = on_damaged
    from Modules.Scenes import scene_base as _scene_base
    _scene_base.LOAD_PUMP = net_mgr.pump_callbacks
    menu_scene = get_or_load_scene("mainmenu")
    game_look = (camera.yaw, camera.pitch, glm.vec3(camera.front))

    def request_start(map_key):
        nonlocal pending_start
        pending_start = map_key

    paper_doll = None  # head of the local rat beside the health bar - built in start_game

    def start_game(map_key):
        nonlocal in_menu, paper_doll
        menu_ui.visible = False
        # Coming back to the map we disconnected from: everything for it is still built.
        reuse = current_scene is not None and current_scene_key == map_key and player is not None
        if reuse:
            current_scene.sound_manager.resume_all()
            player.teleport(game_mode.choose_spawn_position())
            change_health(health["max"])
            if weapon is not None:
                weapon.recoil.reset()
        else:
            setup_game(map_key)
        health_bar.visible = True
        crosshair.visible = True
        weapon_hud.visible = True
        name_tags.visible = True
        if local_player_model is not None and local_player_model.obj is not None:
            local_player_model.set_hat(net_mgr.local_hat)
            local_player_model.set_tint(net_mgr.local_color)
            viewmodel.set_tint(net_mgr.local_color)
            if not reuse:
                paper_doll = PaperDoll(window.ctx, local_player_model.obj)
        if not reuse:
            prime_effects()
        in_menu = False
        ui.set_cursor_free(False)
        camera.yaw, camera.pitch, camera.front = game_look
        window.reset_frame_timer()
        net_mgr.scene = current_scene
        net_mgr.in_game = True

    def set_local_body_visible(visible):
        """Shows or hides the local player's body (and gun) - hidden while dead."""
        for obj in (local_player_model.obj, *getattr(weapon, "worldmodel_objs", ())):
            if obj is not None:
                obj["visible_in_color"] = visible and third_person
                obj["cast_shadow"] = visible

    def set_paused(on):
        pause["on"] = on
        pause_menu.visible = on
        ui.set_cursor_free(on)

    def disconnect():
        """Leaves the lobby and returns to the main menu (the map stays loaded in case it's picked again)."""
        nonlocal in_menu
        set_paused(False)
        net_mgr.leave_lobby()
        if death["active"] or death["pending"]:
            death["active"] = death["pending"] = False
            death_screen.hide()
            set_local_body_visible(True)
        current_scene.sound_manager.pause_all()
        health_bar.visible = False
        crosshair.visible = False
        weapon_hud.visible = False
        name_tags.visible = False
        menu_ui.visible = True
        in_menu = True
        ui.set_cursor_free(True)
        window.reset_frame_timer()

    def begin_death():
        death["pending"] = False
        death["active"] = True
        death["eye_drop"] = 0.0
        feet = player.get_position() - glm.vec3(0.0, player_height / 2.0, 0.0)
        push = death["push"] if time.perf_counter() - death["push_time"] < 1.0 else None
        death["push"] = None
        damage_class, death["damage_class"] = death["damage_class"], "bullet"
        # No gibs for a dissolve death (Zap) - matches what everyone ELSE sees us do (see
        # RemotePlayer's own dissolve state machine): the local player's own body is always
        # hidden instantly either way (never watches its own third-person death - see
        # set_local_body_visible below), so there's no local dissolve ANIMATION to show, just
        # this one difference in what flies out of us when we die.
        if current_scene.gibs is not None and get_damage_class(damage_class).death_effect != DISSOLVE:
            current_scene.gibs.spawn(feet, player.velocity, push=push, tint=net_mgr.local_color)
        smite, death["smite"] = death["smite"], False
        if smite:
            spawn_smite(feet)
        net_mgr.notify_death(push=push, damage_class=damage_class)
        set_local_body_visible(False)
        crosshair.visible = False
        weapon_hud.visible = False
        player.set_move_direction(glm.vec3(0.0))
        player.set_sprinting(False)
        death_screen.show()

    def end_death():
        death["active"] = False
        player.teleport(game_mode.choose_spawn_position())
        change_health(health["max"])
        if weapon is not None:
            weapon.recoil.reset()
        set_local_body_visible(True)
        crosshair.visible = True
        weapon_hud.visible = True
        death_screen.hide()
        net_mgr.notify_respawn()

    def prime_shooting(prime_camera):
        """Plays a shot for real - the gun and arm animations (and the player rig's upper-body
        override), the muzzle flash pass between the world and the viewmodels, a tracer, the
        impact effects - for about a second into the back buffer, so the first real shot doesn't
        build any of it (pose caches, buffers, blend setups) in the middle of a frame."""
        if weapon is None:
            return
        previous_hook = current_scene.before_viewmodels
        current_scene.before_viewmodels = lambda: particles.render(prime_camera, overlay=True)
        muzzle = glm.vec3(prime_camera.position) + prime_camera.front * 0.6
        end = glm.vec3(prime_camera.position) + prime_camera.front * 8.0
        weapon.play("shoot")
        if weapon.muzzle_particle:
            particles.spawn(weapon.muzzle_particle, muzzle, forward=prime_camera.front, up=prime_camera.up,
                            overlay=True, colors=weapon.muzzle_color, size=weapon.muzzle_size,
                            offset_scale=weapon.muzzle_offset_scale)
        tracers.add(muzzle, end)
        for _ in range(10):
            current_scene.update(0.1)
            window.ctx.clear(0.1, 0.1, 0.1, 1.0)
            current_scene.render(prime_camera, None)
            tracers.render(prime_camera)
            particles.update(0.1)
            particles.render(prime_camera, overlay=False)
        weapon.play("idle")
        current_scene.update(0.1)
        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        current_scene.render(prime_camera, None)
        current_scene.before_viewmodels = previous_hook
        particles.clear()
        player.teleport(SPAWN_POSITION)       # the frames above let it fall a little
        weapon.recoil.reset()

    def prime_effects():
        """Warms everything a first death, shot or hit would load or compile lazily - gib
        meshes and materials, blood and impact textures, sounds, the hit marker and death screen
        - so none of it hitches mid-game. All drawn into the back buffer only."""
        show_loading_screen(window, "Loading effects...")
        eye = glm.vec3(*SPAWN_POSITION) + glm.vec3(0.0, 0.5, 0.0)
        prime_camera = Camera(position=eye, aspect=camera.aspect)
        prime_camera.fov = camera.fov
        prime_camera.yaw, prime_camera.pitch = -90.0, 0.0
        prime_camera.update_vectors()
        sounds = current_scene.sound_manager
        if weapon is not None and weapon.fire_sound:
            sounds.preload(weapon.fire_sound, muffle=True)
        sounds.preload(HITMARKER_SOUND, gain=HITMARKER_GAIN)
        sounds.preload(HITMARKER_SOUND, gain=HITMARKER_GAIN, pitch=HITMARKER_HEADSHOT_PITCH)
        sounds.preload(KILL_SOUND, gain=KILL_SOUND_GAIN)
        if current_scene.gibs is not None:
            if local_player_model is not None and local_player_model.obj is not None:
                current_scene.gibs.match_material(local_player_model.obj)
            current_scene.gibs.prime(prime_camera, tint=net_mgr.local_color)
        particles.prime(prime_camera)
        prime_shooting(prime_camera)
        hitmarker.prime()
        damage_indicator.prime()
        death_screen.show()       # one invisible frame builds its text
        ui.render()
        death_screen.hide()
        # Everything built so far lives for the rest of the game: keep the garbage collector from
        # re-scanning it (a full collection mid-game is a visible hitch).
        gc.collect()
        gc.freeze()
        window.reset_frame_timer()

    menu_ui = menu_scene.build_ui(ui, net_mgr, MAPS, on_start=request_start)
    ui.set_cursor_free(True)

    def toggle_ui_demo(key):
        if key == pygame.K_F1:
            ui_demo.visible = not ui_demo.visible
            ui.set_cursor_free(ui_demo.visible)

    trigger_clicks = [0]   # left clicks since the last frame's firing check
    wheel_delta = [0]      # net scroll wheel motion since the last frame's weapon-switch check

    def count_click(event):
        """window.handle_events' event filter: counts left clicks (from the
        event queue, so none is lost between frames) and then lets the UI have
        the event as before."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and not in_menu:
            if chatbox.focused:
                return ui.handle_event(event)   # let TextInput's own Escape close chat instead of pausing
            set_paused(not pause["on"])
            return True
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and not in_menu and not ui.cursor_free):
            trigger_clicks[0] += 1
        if event.type == pygame.MOUSEWHEEL and not in_menu and not ui.cursor_free:
            # Accumulated, not consumed here directly, in case several land in
            # one frame - the main loop turns the total into at most one
            # weapon switch (see wheel_delta below).
            wheel_delta[0] += event.y
            return True
        return ui.handle_event(event)

    def on_key_down(key):
        if in_menu:
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER) and not pause["on"] and not chatbox.focused:
            chatbox.open()
            return
        # "/" opens chat so commands don't need Enter first, straight into command mode -
        # passed explicitly as the opened box's initial text (see ChatBox.open's own
        # docstring) rather than relying on the TEXTINPUT event that happens to follow this
        # same keypress: that worked the first time chat was ever opened in a session, but
        # not reliably on a second open (SDL's start_text_input, toggled off by the previous
        # close and back on here in reaction to this very keydown, doesn't consistently
        # still catch the OS's already-in-flight character event for THIS press the way it
        # does from a cold start) - passing "/" directly sidesteps that timing dependency
        # entirely.
        if key in (pygame.K_SLASH, pygame.K_KP_DIVIDE) and not pause["on"] and not chatbox.focused:
            chatbox.open("/")
            return
        if chatbox.focused:
            return   # chat has keyboard focus - ui.handle_event already routed this, see count_click
        if key == pygame.K_F9:
            prof.set_enabled(not prof.enabled)
            profile_label.visible = prof.enabled
            profile_label.text = ""
        if death["active"]:
            return
        toggle_third_person(key)
        toggle_ui_demo(key)
        if key == pygame.K_r and weapon is not None and not ui.cursor_free:
            weapon.start_reload()
        if key == pygame.K_t and weapon is not None and weapon.reloading and viewmodel is not None:
            reload_time = viewmodel.one_shot_time("reload")
            if reload_time is not None:
                print(f"[reload] {weapon.name}: frame {reload_time * 30.0:.1f} (30fps) / {reload_time:.3f}s")
        if key == pygame.K_h:
            change_health(-10.0)
        elif key == pygame.K_j:
            change_health(10.0)

    net_mgr.profiler = prof
    prof.header = lambda: (
        f"{'HOST' if net_mgr.is_host else 'CLIENT'}  other players: {len(net_mgr.remote_players)}  "
        f"window {window.width}x{window.height}  vsync {window.vsync}")

    running = True
    while running:
        prof.begin()
        running, dt = window.handle_events(
            camera, on_key_down=on_key_down, event_filter=count_click
        )

        if pause["quit"]:
            break
        if pending_start is not None:
            start_map, pending_start = pending_start, None
            start_game(start_map)
        prof.mark("events")

        if in_menu:
            net_mgr.update()
            menu_scene.update(dt)
            if menu_scene.menu is not None:
                menu_scene.menu.update(dt)
            menu_scene.apply_camera(camera)
            window.ctx.clear(0.1, 0.1, 0.1, 1.0)
            menu_scene.render(camera, None)
            ui.render()
            window.flip()
            continue

        keys = pygame.key.get_pressed()

        # Ground-relative movement, driven by camera yaw (mouse-look)
        # but ignoring pitch - walking shouldn't speed up/slow down
        # just from looking up or down.
        if death["pending"] and not death["active"]:
            begin_death()
        alive = not death["active"]
        move_dir = glm.vec3(0.0)
        if keys[pygame.K_w]:
            move_dir += camera.get_flat_forward()
        if keys[pygame.K_s]:
            move_dir -= camera.get_flat_forward()
        if keys[pygame.K_a]:
            move_dir -= camera.get_flat_right()
        if keys[pygame.K_d]:
            move_dir += camera.get_flat_right()
        if alive and (pause["on"] or chatbox.focused):
            player.set_move_direction(glm.vec3(0.0))
            player.set_sprinting(False)
        elif alive:
            player.set_move_direction(move_dir)
            player.set_sprinting(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            player.set_crouching(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
            if keys[pygame.K_SPACE]:
                player.jump()
        # No player.update(dt) here - CharacterController's Source-style
        # ground/air movement math runs once per fixed physics tick (a
        # PhysicsWorld pre-substep callback, registered in its __init__),
        # not once per variable-length render frame; current_scene.update()
        # below is what actually advances physics.

        # Call update to drive scene animations (like the rotating
        # torus) and step physics - camera position then follows
        # wherever physics moved the player capsule to this frame.
        current_scene.update(dt)
        if current_scene.gibs is not None:
            current_scene.gibs.update(dt)     # before particles.update: the blood follows the gibs
        prof.mark("input + scene.update")
        eye_position = player.get_position() + glm.vec3(
            0.0, player.get_eye_offset(), 0.0
        )
        if not alive:
            # The view sinks to the floor over half a second, like the body dropping.
            death["eye_drop"] = min(1.0, death["eye_drop"] + dt / 0.5)
            ease = death["eye_drop"] * death["eye_drop"] * (3.0 - 2.0 * death["eye_drop"])
            eye_position.y -= (player.get_eye_offset() - 0.25) * ease
        if third_person:
            # Same pivot first-person already uses (the player's own eye
            # position) - camera.front/yaw/pitch (mouse look) and all
            # movement math are completely unaffected by this branch,
            # only WHERE the camera itself sits changes.
            camera.position = boom_arm.get_camera_position(
                eye_position, camera.yaw, camera.pitch
            )
        else:
            camera.position = eye_position
        # After the camera has its final position/look for this frame.
        # Recoil moves the camera a little further each frame (see Recoil) -
        # after mouse look, before anything reads camera.front for this frame.
        if weapon is not None and alive:
            weapon.recoil.apply(camera, dt)
        viewmodel.update(camera, not third_person and alive, dt)
        if weapon is not None:
            weapon.update(follow=lambda: camera.position)   # finishes an in-progress reload once its time is up

        # Scroll wheel switches weapons (see switch_weapon) - up goes to the
        # previous slot, down to the next, wrapping around either way.
        # Ignored with only one weapon (nothing to switch to) or dead.
        wheel, wheel_delta[0] = wheel_delta[0], 0
        if wheel != 0 and alive and len(inventory) > 1:
            switch_weapon(inventory.index + (-1 if wheel > 0 else 1))

        # Left mouse fires: how many times is up to the weapon's own
        # fire_mode (see WeaponsBase.shots_this_frame) - one shot per click
        # for SEMI, continuously while held (paced by fire_interval) for
        # AUTO. Clicks are only counted with the cursor captured (see
        # count_click), so clicking menus/UI never shoots.
        clicks, trigger_clicks[0] = trigger_clicks[0], 0
        if weapon is not None:
            crosshair.set_spread(weapon.spread_degrees(), camera.fov)
            weapon_hud.update(weapon, inventory.weapons, inventory.index)
        if weapon is not None and not ui.cursor_free and alive:
            shots = weapon.shots_this_frame(clicks, pygame.mouse.get_pressed()[0])
            for _ in range(shots):
                # A line trace from the camera along the aim (see PhysicsWorld.
                # raycast): the first thing it meets - wall, prop or another
                # player's hitbox - is what the shot hit.
                shot = weapon.fire(
                    current_scene, camera.position, direction=camera.front,
                    follow=lambda: camera.position,
                )
                if shot is not None:
                    end = (shot.hit.position if shot.hit is not None
                           else camera.position + shot.direction * weapon.max_range)
                    net_mgr.notify_shot(glm.vec3(end))
                    # From the gun's muzzle bone to where the trace ended.
                    muzzle = weapon.muzzle_position(current_scene)
                    if muzzle is None:   # no gun model posed yet: from just off the eye
                        right = glm.normalize(glm.cross(camera.front, camera.up))
                        muzzle = camera.position + camera.front * 0.6 + right * 0.14 - camera.up * 0.12
                    tracers.add(muzzle, end, style=weapon.tracer_style)
                    weapon.spawn_tracer_spiral(particles, muzzle, end, camera_pos=camera.position)
                    if weapon.impact_particle and shot.hit is not None:
                        particles.spawn_surface(
                            weapon.impact_particle, shot.hit.position, shot.hit.normal,
                            colors=weapon.impact_color, size=weapon.impact_size, group=IMPACT_GROUP)
                        play_impact_sound(current_scene.sound_manager, shot.hit)
                    else:
                        spawn_impact(particles, shot.hit, sound_manager=current_scene.sound_manager)
                    if weapon.explosion is not None and shot.hit is not None:
                        explode(weapon, shot.hit)
                    if weapon.muzzle_particle:
                        # overlay while the first-person gun is what's on screen (its depth is squashed)
                        particles.spawn(weapon.muzzle_particle, muzzle, forward=shot.direction,
                                        up=camera.up, overlay=not third_person,
                                        colors=weapon.muzzle_color, size=weapon.muzzle_size,
                                        offset_scale=weapon.muzzle_offset_scale,
                                        follow=lambda: weapon.muzzle_position(current_scene),
                                        follow_frame=lambda: (camera.front, camera.up))
                    if shot.victim in net_mgr.remote_players or shot.victim == TEST_DUMMY_STEAM_ID:
                        hitmarker.trigger(headshot=shot.headshot)
                        # Flat, in both ears, at any distance: it's feedback for us, not a sound in the world.
                        hit_emitter = current_scene.sound_manager.add_sound(
                            HITMARKER_SOUND, camera.position, volume=HITMARKER_VOLUME, loop=False,
                            universal=True, channel=hit_sound_channel[0], gain=HITMARKER_GAIN,
                            pitch=HITMARKER_HEADSHOT_PITCH if shot.headshot else 1.0)
                        hit_sound_channel[0] = hit_emitter["channel"]
                        if shot.victim == TEST_DUMMY_STEAM_ID:
                            # Not a real match participant - no network traffic,
                            # no real health/score, just an instant "kill" (any
                            # hit, any number of times) via the same kill_dummy
                            # the /kill command's own dummy-target path uses.
                            damage_dummy(shot.damage, weapon.name, shot.headshot,
                                        damage_class=weapon.damage_class.name)
                        else:
                            # origin: where the shot came FROM, for the victim's own damage-
                            # direction indicator (see on_damaged/DamageIndicator) - our own
                            # position at the moment we fired, not shot.hit.position (where the
                            # bullet struck THEM, a point on their own hitbox a few tenths of a
                            # metre across that has nothing to do with our direction from them;
                            # confirmed as exactly why the indicator read as inaccurate,
                            # especially at range, where a few tenths of a metre of "where on my
                            # body did it land" noise swings the shown bearing by a lot more
                            # degrees than it does up close).
                            net_mgr.send_damage(shot.victim, shot.damage, weapon.name, headshot=shot.headshot,
                                                damage_class=weapon.damage_class.name,
                                                origin=tuple(camera.position))

        prof.mark("camera + weapons")

        # get_position() is the hull CENTER, not feet - subtract half
        # the standing height (the same player_height passed to
        # CharacterController above, not a re-read of any private/
        # crouch-varying internal) to get where the model should
        # actually stand. horiz_speed matches the flat/horizontal
        # convention already used elsewhere in this file (get_flat_
        # forward, footstep gating) so vertical jump/fall speed never
        # triggers a "running" pose.
        feet_position = player.get_position() - glm.vec3(0.0, player_height / 2.0, 0.0)
        horiz_speed = glm.length(glm.vec3(player.velocity.x, 0.0, player.velocity.z))
        # A one-shot consumed event (see CharacterController.pop_jumped's
        # own docstring, mirroring pop_footstep below) - popped BEFORE
        # the update() call it feeds, so PlayerModel can play the jump
        # takeoff pose the instant a jump actually executes instead of
        # waiting on is_on_ground()'s own debounced reading (see
        # PlayerModel.update()'s own just_jumped docstring for why that
        # debounce alone made a real jump feel delayed).
        just_jumped = player.pop_jumped()
        if weapon is not None:
            weapon.set_moving(horiz_speed)
        local_player_model.update(
            dt,
            feet_position,
            camera.yaw,
            horiz_speed,
            is_crouched=player.is_crouched(),
            is_grounded=player.is_on_ground(),
            # Walk-vs-run is now driven by the actual sprint key input,
            # not momentum - see PlayerModel.update()'s own is_sprinting
            # docstring for why (this was the real fix for the run
            # animation "randomly restarting", which turned out to be
            # physics speed noise crossing a threshold, not an animation
            # data or looping-code problem).
            is_sprinting=player.is_sprinting(),
            # Same raw WASD-derived direction vector already passed to
            # player.set_move_direction() above - drives
            # player_directional_clips' facing-relative N/S/E/W walk
            # selection (see PlayerModel.update()'s own move_direction
            # docstring). Reused as-is rather than reading it back off
            # CharacterController, since that's exactly the vector this
            # was built from a few lines up.
            move_direction=move_dir,
            just_jumped=just_jumped,
        )
        if test_dummy is not None:
            test_dummy.update(dt)
        # The same values driving the local model, sent to other players so
        # their copy of us moves and animates identically (see NetworkManager.
        # set_local_state for why this isn't just the camera position).
        net_mgr.set_local_state(
            feet_position, camera.yaw, camera.pitch, horiz_speed,
            player.is_crouched(), player.is_on_ground(), player.is_sprinting(),
            move_dir, just_jumped,
        )

        footstep = player.pop_footstep()
        if footstep is not None:
            material, volume = footstep
            current_scene.play_footstep_sound(
                material, player.get_position(), volume=volume
            )
            net_mgr.notify_footstep(material, volume)   # so other players hear it too

        prof.mark("player model + state")
        current_scene.update_audio(camera)
        prof.mark("audio")

        if death["active"]:
            death_screen.update()
            if death_screen.finished:
                end_death()
        kill_feed.update()
        chatbox.update()
        damage_indicator.update(camera)
        if (dummy_state["respawn_at"] is not None and time.perf_counter() >= dummy_state["respawn_at"]
                and test_dummy is not None):
            dummy_state["respawn_at"] = None
            dummy_state["health"] = TEST_DUMMY_HEALTH
            test_dummy.receive_state({
                "p": list(dummy_state["pos"]), "y": dummy_state["yaw"], "v": 0.0,
                "c": 0, "g": 1, "s": 0, "d": [0.0, 0.0], "j": 0,
                "x": dummy_state["deaths"], "a": 1,
            })

        prof.mark("death screen")
        net_mgr.update()
        prof.mark("net_mgr.update")
        tagged_players = net_mgr.remote_players
        if test_dummy is not None:
            # Not a real network peer (no steam_id in net_mgr.remote_players at all - see
            # spawn_test_dummy's own docstring), so it needs adding here for a tag same as
            # everyone else gets. A fresh dict each frame (cheap - at most a handful of
            # players) rather than mutating remote_players itself, which would leak a
            # fake "peer" into every OTHER place that dict is used (scoreboard, kill credit).
            tagged_players = dict(tagged_players)
            tagged_players[TEST_DUMMY_STEAM_ID] = test_dummy
        name_tags.update(tagged_players, camera, window.ctx.screen.size, scene=current_scene)
        scoreboard.visible = bool(keys[pygame.K_TAB]) and not chatbox.focused
        if scoreboard.visible:
            scoreboard.update(net_mgr, game_mode)
        prof.mark("name tags")

        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        # First-person muzzle flashes (overlay effects) are drawn by the scene just
        # before the viewmodels, so the gun and arms sit in front of them.
        current_scene.before_viewmodels = lambda: particles.render(camera, overlay=True)
        current_scene.render(camera, None)
        prof.mark("scene.render (CPU)")
        tracers.render(camera)
        particles.update(dt)
        particles.render(camera, overlay=False)
        current_scene.present()      # copies an offscreen scene (SSR) to the window; a no-op otherwise
        prof.mark("tracers + particles")
        if prof.enabled:
            text = "\n".join(prof.lines)
            if text != profile_label.text:
                prof.ignore_frame()       # re-drawing the overlay's own text is a spike we caused
                profile_label.text = text
        ui.render()
        prof.mark("ui.render")
        if paper_doll is not None:
            paper_doll.render(window.ctx.screen.size)
        prof.mark("paper doll")
        window.flip()
        prof.mark("flip (GPU wait / vsync)")
        prof.end_frame()

    particles.destroy()
    tracers.destroy()
    ui.destroy()
    window.quit()
    sys.exit()


def _write_crash_log(exc):
    """Appends a full traceback (plus a few basics: frozen or not,
    platform) to crash_log.txt next to the running exe (or the project
    root, for a dev `python app.py` run - see _is_frozen_build's own
    docstring for that distinction). Exists specifically because
    BuildCMD_Nuitka builds with --windows-console-mode=disable - a
    frozen build has NO console for an unhandled exception's traceback
    to print to at all, so today an unhandled crash (a GL context that
    fails to create because a given GPU/driver doesn't support some
    requested feature is the leading suspect for a reported AMD-only
    crash - see the entry point below) is completely silent: the window
    just flashes and closes, with nothing anywhere explaining why. This
    doesn't fix any particular crash - it exists so the NEXT one leaves
    behind something to actually diagnose it from, rather than staying
    a guess. Never lets a failure IN this logging itself replace the
    original exception - see the entry point's own re-raise.

    Deliberately NOT written "next to the exe" via sys.executable - a
    first version of this did exactly that, and it was WRONG for a
    --onefile build specifically: Nuitka's own onefile bootstrap
    unpacks the real program into a TEMPORARY directory and runs it
    from there (confirmed directly in Nuitka's own C source,
    OnefileBootstrap.c) - sys.executable inside that process points
    INTO that temp directory, which the bootstrap deletes the moment
    this process exits, crash or not. A crash log written there was
    already gone by the time anyone went looking for it - it looked
    exactly like no log had been written at all. ~/.ratwar isn't
    touched by that cleanup and survives the process exiting, which is
    the entire point of a crash log."""
    import traceback
    import datetime
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".ratwar")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "crash_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== crash at {datetime.datetime.now().isoformat()} ===\n")
            f.write(f"frozen build: {_is_frozen_build()}\n")
            f.write(f"platform: {sys.platform}\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _write_crash_log(e)
        raise
