import gc
import sys
import time
import pygame
import os
import ctypes
import glm


def _is_frozen_build():
    """True in a Nuitka build (Nuitka injects "__compiled__" into this module's globals)."""
    return "__compiled__" in globals()


def _chdir_for_frozen_build():
    """Every asset path is a plain relative "Assets/..." string, so a built exe must run from its own folder."""
    if _is_frozen_build():
        os.chdir(os.path.dirname(os.path.abspath(sys.executable)))


_chdir_for_frozen_build()

# Packed (encrypted) assets: when the build ships Assets/pack.idx, every read under Assets/ comes from the pack
# (Modules/Assets/pack_runtime.py). Must run before anything reads an asset at import time; a no-op from source.
from Modules.Assets import pack_runtime
pack_runtime.install()


def _disable_window_ghosting():
    """Loading blocks the GL thread for seconds; without this, a click meanwhile makes Windows swap the window
    for a white "not responding" ghost. The clicks just queue up instead."""
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.DisableProcessWindowsGhosting()
        except Exception:
            pass


_disable_window_ghosting()

# Dedicated-GPU selection on Optimus/switchable laptops: only the NvOptimusEnablement/
# AmdPowerXpressRequestHighPerformance exports in the .exe itself work (not the Windows registry hint, not a
# DLL) - nuitka_gpu_preference_plugin.py adds them at build time.

from Modules.Window.window import WindowManager, frame_rate_limit
from Modules.Window.nvidia_profile import ensure_threaded_optimization
from Modules.Window.loading_screen import hide_loading_screen, show_loading_screen
from Modules.UI import UIManager
from Modules.UI.demo import build_demo
from Modules.UI import Anchor, Crosshair, Hitmarker, Label, ProgressBar
from Modules.UI.scope_overlay import ScopeOverlay
from Modules.UI.death_screen import DeathScreen
from Modules.UI.pause_menu import PauseMenu
from Modules.UI.emote_wheel import EmoteWheel
from Modules.Graphics.tracers import Tracers
from Modules.Particles import ParticleManager
from Modules.Particles.blood import register_blood
from Modules.Particles.lightning import register_lightning
from Modules.Particles.dissolve_sparks import register_dissolve_sparks
from Modules.Gore import GibManager
from Modules.Debug import FrameProfiler
from Modules.Window.cpu_affinity import (
    prefer_performance_cores, keep_other_threads_off_main_cores, apply_process_cpu_mask_from_env,
    apply_priority_from_env, cpu_affinity_label)
from Modules.Particles.impacts import (IMPACT_FILE, IMPACT_GROUP, MATERIAL_ALIASES, LIFETIME_SCALE, MAX_IMPACT_EFFECTS,
                                       RADIUS_SCALE, keep_frame, spawn_impact, play_impact_sound)
from Modules.Physics.physics_world import CollisionGroup
from Modules.UI.nametags import NameTags
from Modules.UI.scoreboard import Scoreboard
from Modules.Input import Action, AimAssist, InputManager, CONTROLLER
from Modules.Input.bindings import RUMBLE_DAMAGE, RUMBLE_DEATH, RUMBLE_FIRE, RUMBLE_HIT, WHEEL_STICK_SPEED
from Modules.UI import gamepad_nav
from Modules.settings import settings

SPAWN_POSITION = (0.0, 2.5, 6.0)   # where the player (re)spawns: hull centre, metres
RESPAWN_SECONDS = 3.0
# A shootable target for testing kills alone (/adddummy). 1 can never equal a real (17-digit) Steam64 id.
TEST_DUMMY_STEAM_ID = 1
TEST_DUMMY_OFFSET = (3.0, 0.0, -2.0)   # metres from SPAWN_POSITION, feet-height
TEST_DUMMY_RESPAWN_SECONDS = 1.5
TEST_DUMMY_HEALTH = 100.0
STARTING_LOADOUT = ["usp", "gouda_gun", "pencil", "corroder"]   # weapon ids (Modules/Weapons/registry.py)
# Number-row keys 1-9 select inventory slots 0-8 (only the slots that exist).
WEAPON_SLOT_KEYS = {getattr(pygame, f"K_{n}"): n - 1 for n in range(1, 10)}

# /smite's sound pair. Ranges are Minecraft's attenuation in blocks, taken as metres. (Kill.wav, not these
# ranges, was what once drowned the explosion out - show_kill_feed's play_sound=False covers that.)
SMITE_EXPLOSION_SOUND = "Assets/Audio/Effects/Lightning/Explosion3.wav"
SMITE_THUNDER_SOUND = "Assets/Audio/Effects/Lightning/Thunder1.wav"
SMITE_EXPLOSION_RANGE = 16.0
SMITE_THUNDER_RANGE = 160_000.0
HITMARKER_SOUND = "Assets/Audio/Player/hitmarker.wav"
HITMARKER_VOLUME = 1.0   # the sound manager applies a volume twice (sound and channel), so 1.0 is full
HITMARKER_HEADSHOT_PITCH = 1.45   # a headshot's hitmarker plays this much faster
HITMARKER_GAIN = 10.0     # anything louder has to amplify the samples themselves (soft-clipped)
KILL_SOUND = "Assets/Audio/player/Kill.wav"
KILL_SOUND_VOLUME = 1.0
KILL_SOUND_GAIN = 4.0

# The rat's animations. Most were exported at Blender's default 24 fps although authored for 30, so their key
# times are 1.25x too long; RAT_ANIM_TIME_SCALE corrects that for the files flagged below (checked per file
# from the raw keyframe spacing - the others were re-exported at 30 fps).
RAT_ANIM_TIME_SCALE = 24.0 / 30.0
RIFLE_POSE_DIR = "Assets/Animations/Poses/Rifle"
# (file, clip name on the merged skeleton, needs the 24 -> 30 fps fix). Each file's real pose is its "New" clip
# (the NE/NW/SE/SW files also carry 33 identical leftover clips, "Diag" among them, which are ignored).
RIFLE_POSES = (
    ("rifleidle.glb", "rifle_idle", True),
    ("RifleWalkN.glb", "rifle_walk", False),
    ("RifleWalkS.glb", "rifle_walk_s", False),
    ("RifleWalkE.glb", "rifle_walk_e", False),
    ("RifleWalkW.glb", "rifle_walk_w", True),
    ("RifleWalkNE.glb", "rifle_walk_ne", True),
    ("RifleWalkNW.glb", "rifle_walk_nw", True),
    ("RifleWalkSE.glb", "rifle_walk_se", True),
    ("RifleWalkSW.glb", "rifle_walk_sw", True),
    ("RifleRunN.glb", "rifle_run", False),
    ("RifleJump.glb", "rifle_jump", False),
    ("RifleCrouch.glb", "rifle_crouch", False),   # a static pose: frame rate doesn't matter
)
# (state, clip, min speed m/s) - Source's velwalk/velrun (90/220 units/s). Used for both body halves.
PLAYER_ANIMATION_STATES = (
    ("idle", "rifle_idle", 0.0),
    ("walk", "rifle_walk", 90.0 * 0.0254),
    ("run", "rifle_run", 220.0 * 0.0254),
)
# Facing-relative walk and crouch-walk clips (no directional run set exists yet, so "run" always plays rifle_run).
# The crouch-walk names are remote_player.CROUCH_WALK_CLIPS' (loaded below).
PLAYER_DIRECTIONAL_CLIPS = {
    "walk": {"N": "rifle_walk",
             **{d.upper(): f"rifle_walk_{d}" for d in ("s", "e", "w", "ne", "nw", "se", "sw")}},
    "crouch_walk": {"N": "crouch_walk",
                    **{d.upper(): f"crouch_walk_{d}" for d in ("s", "e", "w", "ne", "nw", "se", "sw")}},
}
# Rat eye level is 1.39225 m above the feet; CharacterController puts the eye at height * 8/9.
PLAYER_HEIGHT = 1.39225 * 9 / 8

from Modules.Graphics.paper_doll import PaperDoll
from Modules.UI.killfeed import KillFeed
from Modules.UI.damage_indicator import DamageIndicator
from Modules.UI.damage_numbers import DamageNumbers
from Modules.UI.chatbox import ChatBox
from Modules.Chat import commands
from Modules.Camera.camera import Camera
from Modules.Camera.camera_boom_arm import CameraBoomArm
from Modules.Scenes.torus_scene import TorusScene
from Modules.Scenes.mainmap_scene import MainMapScene
from Modules.Scenes.sewer_scene import SewerScene
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
    """Pre-loads the platform's Steamworks library globally (RTLD_GLOBAL) before py_steam_net is imported.
    STEAM_API_DLL_PATH overrides where it's looked for."""
    lib_name = {"win32": "steam_api64.dll", "darwin": "libsteam_api.dylib"}.get(sys.platform, "libsteam_api.so")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # A frozen build already chdir'd to its own folder (above), so the cwd candidate covers it.
    candidates = [os.path.join(script_dir, lib_name), os.path.join(os.getcwd(), lib_name)]
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


load_steam_api_dll()

from Modules.Networking.network_manager import NetworkManager
from Modules.Networking.remote_player import (
    AIM_LEVEL_JOINT, AIM_PITCH_JOINTS, CROUCH_STATES, CROUCH_WALK_CLIPS, CROUCH_WALK_FILE, ROOT_OFFSETS,
    UPPER_BODY_ROOT_JOINTS, WEAPON_POSE_WEIGHTS, RemotePlayer)
from Modules.GameModes import Deathmatch, coerce_map, create_game_mode, game_mode_ids, get_game_mode_class


def main():
    print("Yo wsg")
    # CPU placement (Modules/Window/cpu_affinity.py): on a hybrid CPU, the process (the graphics driver's threads
    # included) on the P-cores, and this main thread on P-cores of its own - re-applied to the other threads after
    # the window and each load, since the driver starts threads lazily. RATWAR_CPU_MASK replaces it (experiments).
    apply_priority_from_env()
    if not apply_process_cpu_mask_from_env():
        prefer_performance_cores()
    # NVIDIA's threaded optimization, forced on for this exe (Modules/Window/nvidia_profile.py) - left on "Auto"
    # the driver turned it off by itself in most sessions, halving fps. Before the GL context exists.
    ensure_threaded_optimization()
    window = WindowManager(800, 600, "Rat King")
    keep_other_threads_off_main_cores()
    camera = Camera(position=(0.0, 0.0, 3.0), aspect=window.width / window.height)
    # Hip-fire FOV and sensitivity: scoping narrows both by the same ratio and eases back to these.
    BASE_FOV = camera.fov
    BASE_SENSITIVITY = camera.sensitivity

    # Scene classes, built the first time something switches to them (TorusScene is only a test scene).
    SCENE_CLASSES = {"torus": TorusScene, "mainmap": MainMapScene, "sewer": SewerScene, "mainmenu": MainMenuScene}
    scenes = {}

    def get_or_load_scene(key):
        """scenes[key], building it first (behind a loading-screen frame) if needed."""
        if key not in scenes:
            show_loading_screen(window, f"Loading {key}...")
            try:
                scenes[key] = SCENE_CLASSES[key](window.ctx)
                scenes[key].gibs = GibManager(scenes[key], particles)
            finally:
                hide_loading_screen()
            keep_other_threads_off_main_cores()   # any thread the load started
            window.reset_frame_timer()   # keep the multi-second load out of the next dt
        return scenes[key]

    # Gameplay state - nothing below exists until setup_game() runs (a match start).
    current_scene_key = None
    current_scene = None
    player = None
    local_player_model = None
    viewmodel = None  # first-person arms glued to the camera
    weapon = None     # the local player's current weapon - inventory.current
    inventory = Inventory(STARTING_LOADOUT)   # built once: loadout and ammo carry across matches
    boom_arm = None
    # Purely state-driven: third person while dead or emoting, first person otherwise.
    third_person = False

    def apply_third_person_visibility():
        """The current weapon's world model shows only in third person (equipping always shows it)."""
        if weapon is not None:
            for obj in weapon.worldmodel_objs:
                obj["visible_in_color"] = third_person
    test_dummy = None  # a shootable RemotePlayer for testing alone - see spawn_test_dummy
    # deaths: its death counter (RemotePlayer's "x"); respawn_at: perf_counter() time to show it again, or None.
    dummy_state = {"deaths": 0, "respawn_at": None, "pos": None, "yaw": 180.0, "health": 100.0}

    def dummy_packet(alive, damage_class=None):
        """The movement packet that puts the test dummy at its spot, alive or dead."""
        packet = {"p": list(dummy_state["pos"]), "y": dummy_state["yaw"], "v": 0.0, "c": 0, "g": 1, "s": 0,
                  "d": [0.0, 0.0], "j": 0, "x": dummy_state["deaths"], "a": int(alive)}
        if damage_class is not None:
            packet["dc"] = str(damage_class)
        return packet

    def setup_game(scene_key):
        nonlocal boom_arm, current_scene, current_scene_key, local_player_model, viewmodel, player, third_person
        nonlocal test_dummy, paper_doll
        # Every match is a full rebuild: tear down the last one (even on the same map) so no actor, physics body
        # or GL resource carries over, and drop the old scene so a fresh one is built.
        for leftover in (player, local_player_model, paper_doll, test_dummy):
            if leftover is not None:
                leftover.destroy()
        player = local_player_model = paper_doll = test_dummy = None
        if current_scene is not None:
            current_scene.destroy()
            scenes.pop(current_scene_key, None)
            current_scene = None
        current_scene_key = scene_key
        current_scene = get_or_load_scene(current_scene_key)

        game_mode.use_scene_spawns(current_scene, PLAYER_HEIGHT)   # the map's "character_spawn" markers
        player = CharacterController(
            current_scene.physics,
            position=game_mode.choose_spawn_position(),
            height=PLAYER_HEIGHT,
            max_slope_degrees=47.0,   # a hair past Source's 45.57: the torus scene's stair ramp is ~46.3
            owner=net_mgr.local_steam_id,   # tags our movement ghost, so actor triggers can tell who we are
        )

        # Our own body: shadow-only in first person.
        local_player_model = PlayerModel(
            current_scene,
            "Assets/Models/rat.glb",
            visible_in_color=False,
            cast_shadow=True,
            time_scale=RAT_ANIM_TIME_SCALE,
            tint_mask_path=RAT_TINT_MASK_PATH,
            states=PLAYER_ANIMATION_STATES,
            directional_clips=PLAYER_DIRECTIONAL_CLIPS,
            root_offsets=ROOT_OFFSETS,         # per-clip body offsets (the crouch pose's raised feet)
            aim_pitch_joints=AIM_PITCH_JOINTS,  # the spine bends to where we look...
            aim_level_joint=AIM_LEVEL_JOINT,    # ...and the head faces straight ahead at pitch 0
            # The upper/lower body split and how a held weapon's pose is layered on - the same for every player
            # (see remote_player's UPPER_BODY_ROOT_JOINTS/WEAPON_POSE_WEIGHTS).
            upper_body_root_joints=UPPER_BODY_ROOT_JOINTS,
            override_upper_body_root_joints=WEAPON_POSE_WEIGHTS,
            override_upper_body_component_space=True,
            upper_states=PLAYER_ANIMATION_STATES,
            jump_animation="rifle_jump",       # held on both bodies while airborne
            crouch_animation=CROUCH_STATES,    # both bodies while crouched: the pose, then the crouch walk
        )
        if local_player_model.obj is not None:
            local_player_model.obj["specular_strength"] = 0   # flat shading, like the torus scene's rat
        viewmodel = ViewModel(current_scene)
        for filename, clip, fix_fps in RIFLE_POSES:
            current_scene.load_additional_animations(
                local_player_model.obj, f"{RIFLE_POSE_DIR}/{filename}", rename={"New": clip},
                time_scale=RAT_ANIM_TIME_SCALE if fix_fps else 1.0)
        current_scene.load_additional_animations(
            local_player_model.obj, CROUCH_WALK_FILE, rename=CROUCH_WALK_CLIPS, time_scale=RAT_ANIM_TIME_SCALE)
        # Whatever clips Emotes.glb holds become the emote wheel's catalogue (none collide with the names above).
        emote_catalogue = current_scene.load_additional_animations(
            local_player_model.obj, "Assets/Animations/Emotes/Emotes.glb",
        )
        emote_wheel.set_pages(emote_catalogue)
        third_person = False
        boom_arm = CameraBoomArm(current_scene.physics)

        # force=True: a fresh scene/player model needs equipping even though the slot is unchanged.
        switch_weapon(inventory.index, force=True)
        # Load every other carried weapon too (hidden), so the first switch to it doesn't hitch.
        for other in inventory.weapons:
            if other is not weapon:
                prime_weapon(other)
        viewmodel.warm_animations()   # bake every clip now, not at the first draw/reload

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
        """Switches to inventory slot slot_index (a no-op if it's already out, unless force). The outgoing weapon
        is only hidden; equipping one that's already primed for this scene just reactivates it, so switching
        costs no loading."""
        nonlocal weapon
        if not force and weapon is not None and slot_index % len(inventory) == inventory.index:
            return
        if weapon is not None:
            weapon.deactivate()
        weapon = inventory.select(slot_index)
        net_mgr.local_weapon = weapon_id_of(type(weapon))      # so other players draw the same gun
        weapon.equip_viewmodel(viewmodel)
        weapon.equip_player(current_scene, local_player_model)
        apply_third_person_visibility()

    def spawn_test_dummy():
        """/adddummy: a RemotePlayer standing near spawn (same model, hitboxes, gibs and dissolve as a real one),
        hit locally with no network traffic. Replaces an existing dummy."""
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
        test_dummy.receive_state(dummy_packet(alive=True))

        def dummy_died(pos, vel, color, damage_class):
            # A dissolve is handled by RemotePlayer's own dissolve state machine.
            if get_damage_class(damage_class).death_effect != DISSOLVE and current_scene.gibs is not None:
                current_scene.gibs.spawn(pos, vel, push=dummy_state.get("push"), tint=color)
            if damage_class == "smite":
                spawn_smite(pos)
        test_dummy.on_death = dummy_died
        test_dummy.on_dissolve_spark = spawn_dissolve_spark_burst
        test_dummy.on_dissolve_arc = spawn_dissolve_arc
        return "Test dummy spawned."

    ui = UIManager(window)
    ui_demo = build_demo(ui)

    # The full-screen scope sight picture, shown instead of the crosshair once ADS has fully eased in. Added
    # first so it sits behind every other HUD element (ui.root draws children in add order).
    reticle = ui.root.add(ScopeOverlay("Assets/Textures/UI/Reticle.png", visible=False))

    name_tags = NameTags(ui)
    scoreboard = Scoreboard(ui)
    health = {"value": 100.0, "max": 100.0}
    health_bar = ProgressBar(
        value=health["value"], max_value=health["max"], label_format="{value:.0f} / {max:.0f}",
        anchor=Anchor.BOTTOM_LEFT, offset=(130, -49), size=(300, 26), visible=False,
        fill_color=(90, 200, 110, 255),
    )
    ui.root.add(health_bar)
    # Round countdown, top-right - also hidden whenever the mode has no round timer (round_remaining() is None).
    round_timer_label = ui.root.add(Label(
        "", font_size=30, color=(255, 255, 255, 255), align="right",
        anchor=Anchor.TOP_RIGHT, offset=(-16, 16), visible=False,
    ))
    crosshair = ui.root.add(Crosshair(visible=False))   # its gap is the current weapon's real spread
    weapon_hud = ui.root.add(WeaponHUD())   # current weapon + ammo, other slots dim above it
    hitmarker = ui.root.add(Hitmarker())   # ticks over the crosshair when one of our shots hits a player
    damage_indicator = ui.root.add(DamageIndicator())   # arrow pointing at whoever just hurt us
    damage_numbers = DamageNumbers(ui)   # the damage each of our hits did, popping up where it landed
    game_hud = (health_bar, round_timer_label, crosshair, weapon_hud, name_tags, damage_numbers)   # shown only in a match

    def set_game_hud_visible(on):
        for widget in game_hud:
            widget.visible = on

    def point_damage_indicator(source):
        """Points the damage indicator at a world position."""
        to_source = glm.vec3(source) - camera.position
        damage_indicator.trigger((to_source.x, to_source.z))
    prof = FrameProfiler()   # F9: where each frame's time goes (Modules/Debug/frame_profiler.py)
    profile_label = ui.root.add(Label("", font_size=17, color=(255, 255, 170, 255), visible=False, offset=(14, 14)))
    hit_sound_channel = [None]   # every hit plays on the same channel, cutting off the last
    kill_sound_channel = [None]  # ditto, for Kill.wav
    kill_feed = ui.root.add(KillFeed())
    # Text chat + admin commands (Modules/Chat/commands.py). commands_ctx is filled in further down; it's only
    # read once someone actually runs a command.
    commands_ctx = {}
    chatbox = ui.root.add(ChatBox(
        is_admin=lambda: commands.is_admin(net_mgr.local_steam_id),
        local_name=lambda: net_mgr.display_name(net_mgr.local_steam_id),
        send_chat=lambda text: net_mgr.send_chat(text),
        ctx=commands_ctx,
    ))

    # Dying: health reaching 0 flags `pending`; the main loop starts the death at a safe point (begin_death) and
    # ends it (end_death) when the death screen's countdown runs out. `eye_drop` eases the camera to the floor.
    # smite/damage_class are set by whatever is about to kill us, and read (and reset) by begin_death.
    death = {"pending": False, "active": False, "eye_drop": 0.0, "push": None, "push_time": 0.0,
             "smite": False, "damage_class": "bullet"}
    death_screen = ui.root.add(DeathScreen(RESPAWN_SECONDS))
    emote_wheel = ui.root.add(EmoteWheel())   # hold B; its pages come from Emotes.glb in setup_game
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

    def kill_local(damage_class):
        """Kills the local player outright (a command, or falling out of the map)."""
        death["damage_class"] = damage_class
        change_health(-(health["max"] + 1.0))

    # The game starts on the main menu with nothing else loaded. The menu talks to NetworkManager (it needs Steam
    # callbacks pumped) and reports the chosen map via request_start; the load itself runs from the main loop,
    # not from inside that callback (which can fire during Steam's callback dispatch).
    MAPS = [("Boardwalk", "mainmap"), ("Sewer", "sewer")]
    # Every registered game mode (Modules/GameModes - remember its __init__ import), Cheese War first.
    MODES = [(get_game_mode_class(mode_id).name, mode_id)
             for mode_id in sorted(game_mode_ids(), key=lambda m: m != "cheese_war")]
    in_menu = True
    in_lobby = False   # the lobby waiting room - gameplay keys are off there too
    pending_start = None
    net_mgr = NetworkManager(camera, None)
    # A placeholder until a match starts - start_game replaces it with the mode the host picked.
    game_mode = Deathmatch(net_mgr, spawn_points=[SPAWN_POSITION])

    def dummy_or_display_name(steam_id):
        return "Test Dummy" if steam_id == TEST_DUMMY_STEAM_ID else net_mgr.display_name(steam_id)

    def show_kill_feed(killer_id, victim_id, weapon_name, headshot, play_sound=True):
        """A kill feed row (every client shows every kill), plus Kill.wav when the kill is ours. Kill.wav is
        loud and has no falloff, so play_sound=False (a smite) keeps it from drowning out the smite's own
        explosion and thunder."""
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
    net_mgr.on_player_joined = lambda steam_id, name: chatbox.push_system(f"{name} joined the game")
    net_mgr.on_player_left = lambda steam_id, name: chatbox.push_system(f"{name} left the game")
    # An admin's /kill or /smite aimed at us arrives here; the sender is checked on this end, not trusted.
    net_mgr.on_admin_kill = (
        lambda sender_id, damage_class="bullet":
            kill_by_name("", "smite" if damage_class == "smite" else "bullet")
            if commands.is_admin(sender_id) else None)

    def damage_dummy(amount, weapon_name="", headshot=False, push=None, damage_class="bullet"):
        """Hurts the test dummy like a real player: it dies once its TEST_DUMMY_HEALTH is gone."""
        if test_dummy is None or test_dummy.dead:
            return
        dummy_state["health"] -= amount
        if dummy_state["health"] <= 0.0:
            kill_dummy(weapon_name, headshot, push=push, damage_class=damage_class)

    def spawn_smite(position):
        """The /smite effect: a lightning strike plus the explosion/thunder pair. Never sent over the network -
        each client plays it locally when it learns of a smite death."""
        for system in ("smite_flash", "smite_bolt", "smite_bolt_sky", "smite_crackle"):
            particles.spawn(system, position)
        for sound, max_distance in ((SMITE_EXPLOSION_SOUND, SMITE_EXPLOSION_RANGE),
                                    (SMITE_THUNDER_SOUND, SMITE_THUNDER_RANGE)):
            current_scene.sound_manager.add_sound(
                sound, position, min_distance=1.0, max_distance=max_distance, loop=False, falloff="inverse")

    def spawn_dissolve_spark_burst(position):
        """Ambient sparks/glow on a Zap-dissolving body (fired periodically by RemotePlayer, purely local)."""
        particles.spawn("dissolve_spark", position)
        particles.spawn("dissolve_glow", position)

    def spawn_dissolve_arc(position):
        """One tesla arc on a dissolving body (RemotePlayer fires several at once, on its own interval)."""
        particles.spawn("dissolve_arc", position)

    def kill_dummy(weapon_name="", headshot=False, push=None, damage_class="bullet"):
        """Kills the test dummy now (a fatal hit, or /kill / /smite naming it). damage_class picks gibs or a
        dissolve."""
        if test_dummy is None:
            return "No test dummy - use /adddummy first."
        if test_dummy.dead:
            return "The test dummy is already down."
        dummy_state["deaths"] += 1
        dummy_state["push"] = push      # the gibs' shove (see spawn_test_dummy's dummy_died)
        test_dummy.receive_state(dummy_packet(alive=False, damage_class=damage_class))
        dummy_state["respawn_at"] = time.perf_counter() + TEST_DUMMY_RESPAWN_SECONDS
        show_kill_feed(net_mgr.local_steam_id, TEST_DUMMY_STEAM_ID, weapon_name, headshot,
                       play_sound=damage_class != "smite")
        return "Test dummy killed."

    def explode(source_weapon, hit):
        """Sets off source_weapon.explosion where `hit` landed: everyone inside its radius takes damage (the
        shooter only a proportion of it, see Explosion) and, if it kills them, their gibs fly away from it.
        Returns {target id: (position, damage)} for every other player it hurt (for the damage numbers)."""
        explosion = source_weapon.explosion
        damage_class = source_weapon.damage_class.name
        origin = glm.vec3(hit.position) + glm.vec3(hit.normal) * 0.1     # just off the surface
        targets = [(steam_id, other.body_center) for steam_id, other in net_mgr.remote_players.items()
                   if not other.dead]
        if test_dummy is not None and not test_dummy.dead:
            targets.append((TEST_DUMMY_STEAM_ID, test_dummy.body_center))
        me = None if death["active"] or death["pending"] else (net_mgr.local_steam_id, player.get_position())
        blast_hits = explosion.hits(
            origin, targets, invoker=me,
            blocked=lambda a, b: current_scene.physics.raycast(a, b, CollisionGroup.STATIC) is not None)
        target_positions = dict(targets)
        splash = {}
        for blast in blast_hits:
            if blast.is_invoker:
                death["damage_class"] = damage_class
                change_health(-blast.damage)
                controls.rumble(RUMBLE_DAMAGE, blast.damage / 40.0)
                point_damage_indicator(origin)   # our own splash never comes through net_mgr.on_damage
                continue
            splash[blast.target_id] = (target_positions.get(blast.target_id, origin), blast.damage)
            if blast.target_id == TEST_DUMMY_STEAM_ID:
                damage_dummy(blast.damage, source_weapon.name, False, push=blast.push, damage_class=damage_class)
            else:
                net_mgr.send_damage(blast.target_id, blast.damage, source_weapon.name, push=blast.push,
                                    damage_class=damage_class, origin=tuple(origin))
        if splash:
            hitmarker.trigger(headshot=False)
        return splash

    def show_damage_numbers(shot, splash):
        """Damage numbers for one of our shots: the direct hit (if it struck a player) and its explosion's splash
        (explode's result) on each player it caught. With combine_explosion_damage on, a player both hit and caught
        in the blast gets one number for the total instead of two."""
        if not settings.damage_numbers:
            return
        splash = dict(splash)
        if shot.victim in net_mgr.remote_players or shot.victim == TEST_DUMMY_STEAM_ID:
            total = shot.damage
            if settings.combine_explosion_damage and shot.victim in splash:
                total += splash.pop(shot.victim)[1]
            damage_numbers.add(shot.hit.position, total, shot.headshot)
        for position, amount in splash.values():
            damage_numbers.add(position, amount)

    def die_by_command(damage_class, message):
        """/kill or /smite on ourselves (also what an admin's remote /kill or /smite runs here). Always this
        damage class, never whatever an earlier survived hit left in death["damage_class"]."""
        if death["active"] or death["pending"]:
            return "You're already dead."
        death["smite"] = damage_class == "smite"   # begin_death shows the lightning and replicates it
        kill_local(damage_class)
        return message

    def kill_targets():
        """Autocomplete candidates for /kill and /smite: you, the test dummy if it's up, and everyone else."""
        names = [net_mgr.display_name(net_mgr.local_steam_id)]
        if test_dummy is not None and not test_dummy.dead:
            names.append("Test Dummy")
        names.extend(other.name or net_mgr.display_name(steam_id)
                     for steam_id, other in net_mgr.remote_players.items())
        return names

    def kill_by_name(name, damage_class):
        """/kill or /smite [name] (damage_class "bullet" or "smite"): you, the test dummy, or another player. A
        remote player can only be asked to kill themselves (NetworkManager.send_admin_kill), so that case only
        reports the request was sent."""
        smite = damage_class == "smite"
        target = name.strip().lower()
        if target in ("", "me", "you", "self", net_mgr.display_name(net_mgr.local_steam_id).lower()):
            return die_by_command(damage_class, "Smitten." if smite else "You died.")
        if target in ("dummy", "test dummy"):
            return kill_dummy("/smite" if smite else "/kill", damage_class=damage_class)
        for steam_id, other in net_mgr.remote_players.items():
            if (other.name or "").lower() == target:
                if net_mgr.send_admin_kill(steam_id, damage_class=damage_class):
                    return f"{'Smite' if smite else 'Kill'} request sent to {other.name}."
                return f"Couldn't reach {other.name}."
        return f"No connected player named '{name}'."

    commands_ctx.update(
        add_dummy=spawn_test_dummy,
        give_weapon=give_weapon,
        weapon_ids=weapon_ids,
        kill_targets=kill_targets,
        kill_self=lambda: kill_by_name("", "bullet"),
        kill_player=lambda name: kill_by_name(name, "bullet"),
        smite_self=lambda: kill_by_name("", "smite"),
        smite_player=lambda name: kill_by_name(name, "smite"),
    )
    tracers = Tracers(window.ctx)
    particles = ParticleManager(window.ctx, material_aliases=MATERIAL_ALIASES,
                                radius_scale=RADIUS_SCALE, frame_filter=keep_frame,
                                lifetime_scale=LIFETIME_SCALE)
    particles.group_limits[IMPACT_GROUP] = MAX_IMPACT_EFFECTS     # at most this many impacts alive at once
    particles.group_limits["tracer_spiral"] = 8   # a burst of shots can't pile up unlimited spiral effects
    particles.load("Assets/Particles/Muzzle/muzzleflashes.pcf")
    particles.load(IMPACT_FILE)
    # Whatever particle files the weapons ask for - every weapon, not just carried ones: other players' guns too.
    for weapon_id in weapon_ids():
        for pcf_path, system_names in get_weapon_class(weapon_id).particle_files.items():
            particles.load(pcf_path, only=system_names)
    # The Gouda flash's ring is the one part the source effect doesn't lock to the muzzle: left free, it drifts
    # ahead of the gun and lingers as a ghost of the flash.
    particles.lock_to_control_point("bo3_raygun_muzzleflash_ring")
    register_blood(particles)
    register_lightning(particles)
    register_dissolve_sparks(particles)
    # Particles that collide (impact debris) trace against the level only.
    particles.raycast = lambda a, b: (
        current_scene.physics.raycast(a, b, CollisionGroup.STATIC) if current_scene is not None else None)
    particles.raycast_fast = lambda ax, ay, az, bx, by, bz: (
        current_scene.physics.raycast_light(ax, ay, az, bx, by, bz) if current_scene is not None else None)

    def spawn_shot_impact(gun, hit):
        """The impact effect and sound where a shot from `gun` ended (hit may be None: a miss)."""
        if gun.impact_particle and hit is not None:
            particles.spawn_surface(gun.impact_particle, hit.position, hit.normal,
                                    colors=gun.impact_color, size=gun.impact_size, group=IMPACT_GROUP)
            play_impact_sound(current_scene.sound_manager, hit, gun=gun)
        else:
            spawn_impact(particles, hit, sound_manager=current_scene.sound_manager, gun=gun)

    def remote_shot(start, end, follow=None, shooter_weapon=None):
        """Another player's shot, drawn with their weapon's looks: tracer, muzzle flash on their gun, and the
        impact where it ended (looked up here - only the end point is sent)."""
        shooter_weapon = shooter_weapon or WeaponsBase
        tracers.add(start, end, style=shooter_weapon.tracer_style)
        shooter_weapon.spawn_tracer_spiral(particles, start, end, camera_pos=camera.position)
        aim = glm.vec3(end) - glm.vec3(start)
        if glm.length(aim) > 1e-3:
            if current_scene is not None:
                spawn_shot_impact(shooter_weapon,
                                  current_scene.physics.raycast(start, glm.vec3(end) + glm.normalize(aim) * 0.1))
            if shooter_weapon.muzzle_particle:
                particles.spawn(shooter_weapon.muzzle_particle, start, forward=aim,
                                colors=shooter_weapon.muzzle_color, size=shooter_weapon.muzzle_size,
                                offset_scale=shooter_weapon.muzzle_offset_scale, follow=follow)
    net_mgr.on_tracer = remote_shot

    def remote_footstep(material, position, volume, steam_id=None):
        if current_scene is not None:
            current_scene.play_footstep_sound(material, position, volume=volume, character=steam_id)
    net_mgr.on_footstep = remote_footstep
    net_mgr.on_dissolve_spark = spawn_dissolve_spark_burst
    net_mgr.on_dissolve_arc = spawn_dissolve_arc

    def remote_death(position, velocity, color=None, damage_class="bullet"):
        """Another player died: gibs in their fur colour (unless they dissolve - RemotePlayer animates that
        itself), plus the lightning for a smite."""
        if current_scene is None:
            return
        if get_damage_class(damage_class).death_effect != DISSOLVE and current_scene.gibs is not None:
            current_scene.gibs.spawn(position, velocity, tint=color)
        if damage_class == "smite":
            spawn_smite(position)
    net_mgr.on_death = remote_death

    def on_damaged(amount, attacker_id, weapon_name, headshot, push=None, damage_class="bullet", origin=None):
        """Another player's shot hit us: the shooter decided that, we apply it."""
        death["damage_class"] = damage_class   # what killed us, if this hit does - see begin_death
        change_health(-amount)
        controls.rumble(RUMBLE_DAMAGE, amount / 40.0)
        if push is not None:      # an explosion's shove: if this kills us, our gibs fly that way
            death["push"], death["push_time"] = glm.vec3(*push), time.perf_counter()
        # origin (where the shot or blast came from) is exact, and the only right answer for splash damage; an
        # old message without one falls back to the attacker's current position, or no arrow at all.
        if origin is not None:
            point_damage_indicator(origin)
        else:
            attacker = net_mgr.remote_players.get(attacker_id)
            if attacker is not None:
                point_damage_indicator(attacker.body_center)
    net_mgr.on_damage = on_damaged
    from Modules.Scenes import scene_base as _scene_base
    _scene_base.LOAD_PUMP = net_mgr.pump_callbacks
    menu_scene = get_or_load_scene("mainmenu")

    def request_start(map_key, mode_key=None):
        nonlocal pending_start
        pending_start = (map_key, mode_key)

    def start_from_lobby(map_key, mode_key=None):
        """The menu's "begin gameplay" signal: the lobby room's Start button, or singleplayer."""
        nonlocal in_lobby
        in_lobby = False
        request_start(map_key, mode_key)

    def on_lobby_enter():
        nonlocal in_lobby
        in_lobby = True

    def on_lobby_leave():
        nonlocal in_lobby, in_menu
        in_lobby = False
        in_menu = True

    paper_doll = None  # head of the local rat beside the health bar - built in start_game

    def refill_player():
        """Full health and magazines, recoil settled - every (re)spawn."""
        change_health(health["max"])
        for w in inventory.weapons:
            if w.magazine_size > 0:   # magazine_size <= 0: no ammo tracking
                w.ammo = w.magazine_size
        inventory.recoil.reset()

    def apply_spawn_rotation():
        """Snaps the camera to the game mode's fixed spawn facing."""
        camera.yaw, camera.pitch = game_mode.choose_spawn_rotation()
        camera.update_vectors()

    def start_game(map_key, mode_key=None):
        nonlocal in_menu, paper_doll, game_mode
        menu_ui.visible = False
        map_key = coerce_map(mode_key, map_key)   # e.g. the Sewer is Deathmatch only
        # No mode chosen, or one the registry doesn't know (stale lobby data): Deathmatch.
        game_mode = (
            create_game_mode(mode_key, net_mgr, spawn_points=[SPAWN_POSITION])
            or Deathmatch(net_mgr, spawn_points=[SPAWN_POSITION])
        )
        # Only one peer writes the shared round start time (every peer reads it back); in singleplayer there's
        # no lobby, so we're it.
        if (net_mgr.is_host or not net_mgr.current_lobby_id) and game_mode.round_duration is not None:
            net_mgr.start_round_timer(game_mode.round_duration)
        setup_game(map_key)
        refill_player()   # however the last match ended
        set_game_hud_visible(True)
        if local_player_model is not None and local_player_model.obj is not None:
            local_player_model.set_hat(net_mgr.local_hat)
            local_player_model.set_tint(net_mgr.local_color)
            viewmodel.set_tint(net_mgr.local_color)
            paper_doll = PaperDoll(window.ctx, local_player_model.obj)
        prime_effects()
        keep_other_threads_off_main_cores()
        in_menu = False
        ui.set_cursor_free(False)
        apply_spawn_rotation()
        window.reset_frame_timer()
        net_mgr.scene = current_scene
        net_mgr.in_game = True

    def set_local_body_visible(visible):
        """Shows or hides the local player's body (and gun) - hidden while dead, seen only in third person."""
        for obj in (local_player_model.obj, *getattr(weapon, "worldmodel_objs", ())):
            if obj is not None:
                obj["visible_in_color"] = visible and third_person
                obj["cast_shadow"] = visible

    def set_paused(on):
        pause["on"] = on
        pause_menu.visible = on
        ui.set_cursor_free(on)

    def disconnect():
        """Leaves the lobby and returns to the main menu. The scene is only paused (the menu branch of the main
        loop doesn't update it); the next start_game rebuilds everything anyway."""
        nonlocal in_menu
        set_paused(False)
        net_mgr.leave_lobby()
        net_mgr.scene = None
        if death["active"] or death["pending"]:
            death["active"] = death["pending"] = False
            death_screen.hide()
            set_local_body_visible(True)
        current_scene.sound_manager.pause_all()
        set_game_hud_visible(False)
        menu_ui.visible = True
        in_menu = True
        ui.set_cursor_free(True)
        window.reset_frame_timer()

    def stop_moving():
        player.set_move_direction(glm.vec3(0.0))
        player.set_sprinting(False)

    def begin_death():
        nonlocal third_person
        death["pending"] = False
        death["active"] = True
        death["eye_drop"] = 0.0
        controls.rumble(RUMBLE_DEATH)
        feet = player.get_feet_position()
        push = death["push"] if time.perf_counter() - death["push_time"] < 1.0 else None
        death["push"] = None
        damage_class, death["damage_class"] = death["damage_class"], "bullet"
        # Our body is hidden either way; a dissolve death just throws no gibs (matching what others see).
        if current_scene.gibs is not None and get_damage_class(damage_class).death_effect != DISSOLVE:
            current_scene.gibs.spawn(feet, player.velocity, push=push, tint=net_mgr.local_color)
        smite, death["smite"] = death["smite"], False
        if smite:
            spawn_smite(feet)
        net_mgr.notify_death(push=push, damage_class=damage_class)
        set_local_body_visible(False)
        local_player_model.cancel_emote()   # don't dance into the death pose
        third_person = True   # watch the gibs instead of the inside of our hidden body
        crosshair.visible = False
        weapon_hud.visible = False
        player.halt()          # the body's gone: no sliding or flying on with the momentum we died with
        for w in inventory.weapons:
            w.reset()          # a reload, rack or scope from this life doesn't carry on (or play sounds) while dead
        if weapon is not None:
            weapon.play("idle")
        death_screen.show()

    def end_death():
        nonlocal third_person
        death["active"] = False
        third_person = False   # before set_local_body_visible, whose result depends on it
        player.teleport(game_mode.choose_spawn_position())
        apply_spawn_rotation()
        refill_player()
        if weapon is not None:
            weapon.set_active(True)   # draws the gun afresh, like a weapon switch (before the visibility below)
        set_local_body_visible(True)
        crosshair.visible = True
        weapon_hud.visible = True
        death_screen.hide()
        net_mgr.notify_respawn()

    def prime_shooting(prime_camera):
        """Plays a shot for real for about a second into the back buffer (gun/arm animations, upper-body
        override, muzzle flash, tracer, impact), so the first real shot builds none of it mid-frame."""
        if weapon is None:
            return
        previous_hook = current_scene.before_viewmodels
        current_scene.before_viewmodels = lambda: particles.render(prime_camera, overlay=True)
        muzzle = glm.vec3(prime_camera.position) + prime_camera.front * 0.6
        end = glm.vec3(prime_camera.position) + prime_camera.front * 8.0
        home = player.get_position()          # wherever the spawn put us - restored below
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
        player.teleport(home)                 # the frames above let it fall a little
        inventory.recoil.reset()

    def prime_effects():
        """Warms everything a first death, shot or hit would load or compile lazily (gibs, blood and impact
        textures, sounds, hit marker, death screen), drawing into the back buffer only."""
        show_loading_screen(window, "Loading effects...")
        eye = glm.vec3(player.get_position()) + glm.vec3(0.0, 0.5, 0.0)
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
        hide_loading_screen()
        # Everything built so far lives for the whole match: keep the garbage collector from re-scanning it (a
        # full collection mid-game is a visible hitch).
        gc.collect()
        gc.freeze()
        window.reset_frame_timer()

    menu_ui = menu_scene.build_ui(
        ui, net_mgr, MAPS, MODES, on_start=start_from_lobby,
        on_lobby_enter=on_lobby_enter, on_lobby_leave=on_lobby_leave,
    )
    ui.set_cursor_free(True)

    # Keyboard, mouse and controllers (Modules/Input - bindings.py says what does what).
    controls = InputManager()
    aim_assist = AimAssist()

    def controller_connected(name):
        chatbox.push_system(f"Controller connected: {name}")

    def controller_disconnected(name, was_in_use):
        chatbox.push_system(f"Controller disconnected: {name}")
        # Lost the controller mid-match (flat battery, cable): pause, like a console game, instead of running on.
        if was_in_use and not in_menu and not in_lobby and not pause["on"]:
            set_paused(True)

    controls.on_connected = controller_connected
    controls.on_disconnected = controller_disconnected

    def controller_menu_input(dt):
        """While the cursor is free (menus, the pause menu) and the controller is in use (the cursor is hidden): the
        D-pad or left stick jumps between controls (steps a slider, moves through an open dropdown's options, and on
        a colour wheel the stick moves its marker), A clicks, the right stick scrolls, and B closes an open dropdown,
        then the pause menu."""
        wheel = gamepad_nav.hovered_wheel(ui)
        directions = controls.nav_presses(dt, stick=wheel is None)   # every frame, so a held direction repeats
        if controls.device != CONTROLLER:
            return
        dropdown = ui.popup if hasattr(ui.popup, "pick_hovered") else None
        controls.drive_ui_cursor(dt, click=dropdown is None)
        if dropdown is not None and controls.pressed(Action.UI_CLICK):
            # Pick the highlighted option and put the cursor back on the box, not on whatever is under the list.
            dropdown.pick_hovered()
            x, y, w, h = dropdown.rect
            pygame.mouse.set_pos((round((x + w / 2) * ui.scale), round((y + h / 2) * ui.scale)))
            controls.cursor_moved()
            return
        if wheel is not None and gamepad_nav.drive_wheel(ui, wheel, controls.left_stick(), dt, WHEEL_STICK_SPEED):
            controls.cursor_moved()
        for direction in directions:
            if gamepad_nav.navigate(ui, direction):
                controls.cursor_moved()
        if controls.pressed(Action.UI_BACK):
            if ui.popup is not None:
                ui.popup.close()
            elif pause["on"]:
                set_paused(False)
    pending_emote = [None]   # set when the emote wheel's button is released over a slot, consumed by the main loop
    emote_was_playing = [False]   # last frame's is_emoting(): its falling edge means back to first person

    def line_of_sight(a, b):
        return current_scene.physics.raycast(a, b, CollisionGroup.STATIC) is None

    def count_click(event):
        """window.handle_events' event filter: every event goes to the input manager first (it never consumes one),
        then pause, then the UI. Keys and clicks only count as game input with the cursor captured and chat
        closed, judged before the UI reacts - so the click on a menu's Resume button doesn't also fire."""
        playing = not in_menu and not in_lobby
        controls.handle_event(event, gameplay=playing and not ui.cursor_free and not chatbox.focused)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and playing:
            if chatbox.focused:
                return ui.handle_event(event)   # Escape closes chat instead of pausing
            set_paused(not pause["on"])
            return True
        # In play the wheel switches weapons (input manager); with the cursor free it scrolls the UI.
        if event.type == pygame.MOUSEWHEEL and playing and not ui.cursor_free:
            return True
        return ui.handle_event(event)

    def on_key_down(key):
        nonlocal third_person
        if in_menu or in_lobby:
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER) and not pause["on"] and not chatbox.focused:
            chatbox.open()
            return
        # "/" opens chat straight into command mode. The "/" is passed in rather than left to the TEXTINPUT
        # event that follows: SDL doesn't reliably deliver that one when text input was just switched back on.
        if key in (pygame.K_SLASH, pygame.K_KP_DIVIDE) and not pause["on"] and not chatbox.focused:
            chatbox.open("/")
            return
        if chatbox.focused:
            return   # chat has keyboard focus - ui.handle_event already routed this
        if key == pygame.K_F9:
            prof.set_enabled(not prof.enabled)
            profile_label.visible = prof.enabled
            profile_label.text = ""
        if key == pygame.K_F6 and player is not None:
            print(f"[movedbg] {'ON' if player.toggle_debug_log() else 'OFF'}")   # one line per physics tick
        if death["active"]:
            return
        if key == pygame.K_F1:
            ui_demo.visible = not ui_demo.visible
            ui.set_cursor_free(ui_demo.visible)
        if key == pygame.K_v:
            # TEMPORARY (animation checking): V toggles third person. Remove before release.
            third_person = not third_person
            set_local_body_visible(True)   # re-evaluated for the new view (body and world model)
        slot = WEAPON_SLOT_KEYS.get(key)
        if (slot is not None and weapon is not None and not ui.cursor_free
                and len(inventory) > 1 and slot < len(inventory)):
            switch_weapon(slot)
        if key == pygame.K_t and weapon is not None and weapon.reloading and viewmodel is not None:
            reload_time = viewmodel.one_shot_time("reload")
            if reload_time is not None:
                print(f"[reload] {weapon.name}: frame {reload_time * 30.0:.1f} (30fps) / {reload_time:.3f}s")
        if key == pygame.K_h:
            change_health(-10.0)
        elif key == pygame.K_j:
            change_health(10.0)

    net_mgr.profiler = prof
    prof.extra = lambda: f"drawn {len(getattr(current_scene, '_frame_opaque_visible', None) or ())} opaque"
    prof.header = lambda: (
        f"{'HOST' if net_mgr.is_host else 'CLIENT'}  other players: {len(net_mgr.remote_players)}  "
        f"window {window.width}x{window.height}  vsync {window.vsync}  "
        f"gpu {window.ctx.info.get('GL_RENDERER', '?')}  "
        f"{cpu_affinity_label()}")

    # Built once rather than as fresh closures every frame.
    def camera_position():
        return camera.position

    def muzzle_position():
        return weapon.muzzle_position(current_scene)

    def camera_frame():
        return camera.front, camera.up

    def draw_overlay_particles():
        """First-person muzzle flashes, drawn by the scene just before the viewmodels so the gun sits in front."""
        particles.sync_followers()     # onto the gun's pose from this frame
        particles.render(camera, overlay=True)

    running = True
    while running:
        prof.begin()
        # The frame rate limit (the setting can change any time; the refresh rate if the window changes display).
        window.max_fps = frame_rate_limit(settings.fps_limit, window.refresh_rate)
        controls.begin_frame()
        running, dt = window.handle_events(
            camera, on_key_down=on_key_down, event_filter=count_click
        )
        controls.update(dt)
        ui.set_cursor_hidden(controls.pointer_hidden)

        if pause["quit"]:
            break
        if pending_start is not None:
            (start_map, start_mode), pending_start = pending_start, None
            start_game(start_map, start_mode)
        prof.mark("events")

        if in_menu or in_lobby:
            controller_menu_input(dt)
            net_mgr.update()
            menu_scene.update(dt)
            if in_menu and menu_scene.menu is not None:
                menu_scene.menu.update(dt)
            menu_scene.apply_camera(camera)
            window.ctx.clear(0.1, 0.1, 0.1, 1.0)
            menu_scene.render(camera, None)
            menu_scene.present()
            ui.render()
            window.flip()
            continue

        if death["pending"] and not death["active"]:
            begin_death()
        alive = not death["active"]
        using_pad = controls.device == CONTROLLER
        in_play = not ui.cursor_free and not chatbox.focused   # the cursor's free in menus, the pause menu, the wheel

        # Controller: Menu toggles pause; in the pause menu the stick/D-pad/A/B drive it.
        if controls.pressed(Action.PAUSE) and not chatbox.focused:
            set_paused(not pause["on"])
        elif ui.cursor_free and not emote_wheel.visible:
            controller_menu_input(dt)

        move_x, move_y = controls.move_vector()
        # Right-stick look (the mouse turns the camera in window.handle_events), with aim assist on a controller.
        if alive and in_play:
            look_yaw, look_pitch = controls.look_delta(dt, scale=camera.fov / BASE_FOV)
            if using_pad and settings.aim_assist:
                targets = [(steam_id, other.body_center) for steam_id, other in net_mgr.remote_players.items()
                           if not other.dead]
                if test_dummy is not None and not test_dummy.dead:
                    targets.append((TEST_DUMMY_STEAM_ID, test_dummy.body_center))
                scale, pull_yaw, pull_pitch = aim_assist.update(camera, targets, dt, line_of_sight)
                look_yaw, look_pitch = look_yaw * scale + pull_yaw, look_pitch * scale + pull_pitch
            if look_yaw or look_pitch:
                camera.rotate(look_yaw, look_pitch)
        # Ground-relative movement from the camera's yaw (pitch ignored); a stick's tilt sets the speed.
        move_dir = camera.get_flat_forward() * move_y + camera.get_flat_right() * move_x

        # Emote wheel: opens while its button is held, picks with the mouse or right stick, plays on release.
        if controls.pressed(Action.EMOTE_WHEEL) and in_play and not emote_wheel.visible:
            emote_wheel.open()
            ui.set_cursor_free(True)
        if emote_wheel.visible:
            if not emote_wheel.is_closing():   # a closing wheel stops changing its highlight
                if not controls.held(Action.EMOTE_WHEEL):
                    pending_emote[0] = emote_wheel.confirmed_emote_name()
                    emote_wheel.close()
                    ui.set_cursor_free(False)
                elif using_pad:
                    sx, sy = controls.right_stick()
                    if sx or sy:
                        x, y, w, h = emote_wheel.rect
                        emote_wheel.update_hover(x + w / 2.0 + sx * 150.0, y + h / 2.0 + sy * 150.0)
                else:
                    mx, my = pygame.mouse.get_pos()
                    emote_wheel.update_hover(mx / ui.scale, my / ui.scale)
            emote_wheel.update()
        if pending_emote[0] is not None:
            name, pending_emote[0] = pending_emote[0], None
            # Only from a standstill on the ground (real velocity, not input): momentum would cancel the emote a
            # frame or two in.
            still = glm.length(glm.vec3(player.velocity.x, 0.0, player.velocity.z)) < 0.1
            if alive and player.is_on_ground() and still:
                local_player_model.play_emote(name)
                third_person = True
                set_local_body_visible(True)   # re-evaluate now that third_person flipped
                net_mgr.notify_emote(name)
        if alive and (pause["on"] or chatbox.focused or emote_wheel.visible):
            stop_moving()
            # Also zeroed for the player model below, so a held key doesn't read as moving (and cancel an emote).
            move_dir = glm.vec3(0.0)
        elif alive:
            player.set_move_direction(move_dir)
            player.set_sprinting(controls.held(Action.SPRINT))
            player.set_crouching(controls.held(Action.CROUCH))
            if controls.held(Action.JUMP):
                player.jump()
        # Movement itself runs per fixed physics tick (CharacterController's pre-substep callback), inside
        # current_scene.update() below.

        # Positions for actor proximity triggers (Modules/Actors) - without us while dead.
        player_positions = [
            (steam_id, other.body_center) for steam_id, other in net_mgr.remote_players.items()
        ]
        if alive:
            player_positions.append((net_mgr.local_steam_id, player.get_position()))
        prof.mark("input")
        current_scene.profiler = prof
        current_scene.update(dt, player_positions=player_positions, game_mode=game_mode)
        prof.mark("scene.update")
        game_mode.update(dt)
        if current_scene.gibs is not None:
            current_scene.gibs.update(dt)     # before particles.update: the blood follows the gibs
        prof.mark("game mode + gibs")
        # Fell out of the map (Scene.min_height): the same death as /kill.
        if alive and current_scene.min_height is not None and player.get_position().y < current_scene.min_height:
            kill_local("bullet")
        eye_position = player.get_position() + glm.vec3(0.0, player.get_eye_offset(), 0.0)
        if not alive:
            # The view sinks to the floor over half a second, like the body dropping.
            death["eye_drop"] = min(1.0, death["eye_drop"] + dt / 0.5)
            ease = death["eye_drop"] * death["eye_drop"] * (3.0 - 2.0 * death["eye_drop"])
            eye_position.y -= (player.get_eye_offset() - 0.25) * ease
        # Third person only moves where the camera sits; the look direction and movement are unchanged.
        if third_person:
            camera.position = boom_arm.get_camera_position(eye_position, camera.yaw, camera.pitch)
        else:
            camera.position = eye_position
        # Recoil (the inventory's, so a kick keeps settling across a weapon switch), before anything reads
        # camera.front this frame.
        if alive:
            inventory.recoil.apply(camera, dt)
        # Right-click ADS: update_scope eases a 0..1 blend that moves the viewmodel to the scope and narrows the
        # FOV; sensitivity scales by the same ratio (felt from the next frame's mouse motion).
        scope_blend = 0.0
        if weapon is not None:
            aiming = alive and not third_person and not ui.cursor_free and controls.held(Action.AIM)
            scope_blend = weapon.update_scope(aiming, dt)
        camera.fov = BASE_FOV + (weapon.scope_fov - BASE_FOV) * scope_blend if (
            weapon is not None and weapon.has_scope) else BASE_FOV
        camera.sensitivity = BASE_SENSITIVITY * (camera.fov / BASE_FOV)
        viewmodel.update(camera, not third_person and alive, dt, scope_blend)
        if weapon is not None:
            weapon.update(follow=camera_position)   # finishes an in-progress reload once its time is up
        # Once ADS has fully eased in, the scope's sight picture replaces the crosshair. Same gating as aiming,
        # so pausing or dying drops it at once.
        fully_scoped = (weapon is not None and weapon.has_scope and weapon.scoped
                        and alive and not third_person and not ui.cursor_free)
        reticle.visible = fully_scoped
        # Only during active play: pause/death/menus own the crosshair's visibility otherwise.
        if weapon is not None and not ui.cursor_free and alive:
            crosshair.visible = not fully_scoped

        # Next/previous weapon (wheel, RB/LB/Y/D-pad), wrapping; reload (R, X).
        steps = controls.pressed(Action.NEXT_WEAPON) - controls.pressed(Action.PREV_WEAPON)
        if steps and alive and in_play and len(inventory) > 1:
            switch_weapon(inventory.index + (1 if steps > 0 else -1))
        if controls.pressed(Action.RELOAD) and weapon is not None and alive and in_play:
            weapon.start_reload()

        # Fire (left mouse, RT); the weapon's fire mode decides how many shots (one per press for SEMI, paced while
        # held for AUTO). Presses only count as game input with the cursor captured, so clicking UI never shoots.
        if weapon is not None:
            crosshair.set_spread(weapon.spread_degrees(), camera.fov)
            weapon_hud.update(weapon, inventory.weapons, inventory.index)
        shots_fired_this_frame = 0
        if weapon is not None and in_play and alive:
            shots = weapon.shots_this_frame(controls.pressed(Action.FIRE), controls.held(Action.FIRE))
            shots_fired_this_frame = shots
            for _ in range(shots):
                # A line trace along the aim: the first wall, prop or player hitbox it meets is what was hit.
                shot = weapon.fire(current_scene, camera.position, direction=camera.front, follow=camera_position)
                if shot is None:      # not ready (fire rate, reloading, empty): no shot, no rumble
                    continue
                controls.rumble(RUMBLE_FIRE)
                end = (shot.hit.position if shot.hit is not None
                       else camera.position + shot.direction * weapon.max_range)
                net_mgr.notify_shot(glm.vec3(end))
                muzzle = weapon.muzzle_position(current_scene)
                if muzzle is None:   # no gun model posed yet: from just off the eye
                    right = glm.normalize(glm.cross(camera.front, camera.up))
                    muzzle = camera.position + camera.front * 0.6 + right * 0.14 - camera.up * 0.12
                tracers.add(muzzle, end, style=weapon.tracer_style)
                weapon.spawn_tracer_spiral(particles, muzzle, end, camera_pos=camera.position)
                spawn_shot_impact(weapon, shot.hit)
                splash = {}
                if weapon.explosion is not None and shot.hit is not None:
                    splash = explode(weapon, shot.hit)
                show_damage_numbers(shot, splash)
                if weapon.muzzle_particle:
                    # overlay while the first-person gun is on screen (its depth is squashed)
                    particles.spawn(weapon.muzzle_particle, muzzle, forward=shot.direction,
                                    up=camera.up, overlay=not third_person,
                                    colors=weapon.muzzle_color, size=weapon.muzzle_size,
                                    offset_scale=weapon.muzzle_offset_scale,
                                    follow=muzzle_position, follow_frame=camera_frame)
                if shot.victim in net_mgr.remote_players or shot.victim == TEST_DUMMY_STEAM_ID:
                    hitmarker.trigger(headshot=shot.headshot)
                    controls.rumble(RUMBLE_HIT, 1.0 if shot.headshot else 0.6)
                    # Flat, in both ears, at any distance: feedback for us, not a sound in the world.
                    hit_emitter = current_scene.sound_manager.add_sound(
                        HITMARKER_SOUND, camera.position, volume=HITMARKER_VOLUME, loop=False,
                        universal=True, channel=hit_sound_channel[0], gain=HITMARKER_GAIN,
                        pitch=HITMARKER_HEADSHOT_PITCH if shot.headshot else 1.0)
                    hit_sound_channel[0] = hit_emitter["channel"]
                    if shot.victim == TEST_DUMMY_STEAM_ID:
                        damage_dummy(shot.damage, weapon.name, shot.headshot, damage_class=weapon.damage_class.name)
                    else:
                        # origin = where we fired from, for their damage indicator (not where it hit their
                        # hitbox: that noise swung the arrow badly at range).
                        net_mgr.send_damage(shot.victim, shot.damage, weapon.name, headshot=shot.headshot,
                                            damage_class=weapon.damage_class.name,
                                            origin=tuple(camera.position))

        prof.mark("camera + weapons")

        # The model stands at the hull's bottom (crouched, that's the shorter hull's). Horizontal speed only, so
        # jumping or falling never reads as running.
        feet_position = player.get_feet_position()
        horiz_speed = glm.length(glm.vec3(player.velocity.x, 0.0, player.velocity.z))
        just_jumped = player.pop_jumped()   # the takeoff pose plays the moment a jump executes
        grounded = player.is_on_ground() or player.is_on_ladder()
        if weapon is not None:
            weapon.set_moving(horiz_speed)
        local_player_model.update(
            dt,
            feet_position,
            camera.yaw,
            horiz_speed,
            is_crouched=player.is_crouched(),
            is_grounded=grounded,
            is_sprinting=player.is_sprinting(),   # sprint input, not speed: speed noise restarted the run clip
            move_direction=move_dir,              # picks the directional walk clip
            just_jumped=just_jumped,
            just_shot=shots_fired_this_frame > 0,   # firing cancels an emote
            pitch_degrees=camera.pitch,           # bends the spine toward where we look (others get it too)
        )
        was_emoting, is_emoting_now = emote_was_playing[0], local_player_model.is_emoting()
        if was_emoting and not is_emoting_now and not death["active"]:
            # The emote finished or was cancelled: back to first person (unless dying, which wants third).
            third_person = False
            set_local_body_visible(True)
        emote_was_playing[0] = is_emoting_now
        if test_dummy is not None:
            test_dummy.update(dt)
        # The same values driving our model, so other players' copy of us moves and animates identically.
        net_mgr.set_local_state(
            feet_position, camera.yaw, camera.pitch, horiz_speed,
            player.is_crouched(), grounded, player.is_sprinting(),
            move_dir, just_jumped,
        )

        footstep = player.pop_footstep()
        if footstep is not None:
            material, volume = footstep
            current_scene.play_footstep_sound(
                material, player.get_position(), volume=volume, character=net_mgr.local_steam_id
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
        _t_round = time.perf_counter() if prof.enabled else 0.0
        remaining = game_mode.round_remaining()
        if prof.enabled:
            prof.add("  hud: round timer read", time.perf_counter() - _t_round)
        round_timer_label.visible = remaining is not None
        if remaining is not None:
            minutes, seconds = divmod(int(remaining), 60)
            round_timer_label.text = f"{minutes}:{seconds:02d}"
        chatbox.update()
        damage_indicator.update(camera)
        damage_numbers.update(camera, window.ctx.screen.size)
        if (dummy_state["respawn_at"] is not None and time.perf_counter() >= dummy_state["respawn_at"]
                and test_dummy is not None):
            dummy_state["respawn_at"] = None
            dummy_state["health"] = TEST_DUMMY_HEALTH
            test_dummy.receive_state(dummy_packet(alive=True))

        prof.mark("death screen")
        net_mgr.update()
        prof.mark("net_mgr.update")
        tagged_players = net_mgr.remote_players
        if test_dummy is not None:
            # A copy with the dummy added: putting it in remote_players itself would leak a fake peer into the
            # scoreboard and kill credit.
            tagged_players = dict(tagged_players)
            tagged_players[TEST_DUMMY_STEAM_ID] = test_dummy
        name_tags.update(tagged_players, camera, window.ctx.screen.size, scene=current_scene)
        scoreboard.visible = controls.held(Action.SCOREBOARD) and not chatbox.focused
        if scoreboard.visible:
            scoreboard.update(net_mgr, game_mode)
        prof.mark("name tags")

        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        current_scene.before_viewmodels = draw_overlay_particles
        current_scene.render(camera, None)
        prof.mark("scene.render (CPU)")
        tracers.render(camera)
        prof.mark("tracers")
        particles.update(dt)
        prof.mark("particles.update")
        particles.render(camera, overlay=False)
        prof.mark("particles.render")
        current_scene.present()      # resolves the scene's offscreen MSAA target into the window
        prof.mark("present")
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
    """Appends the traceback to ~/.ratwar/crash_log.txt: a build has no console to print it to. Not next to the
    exe - a onefile build runs from a temp dir that's deleted on exit. Never replaces the original exception."""
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
