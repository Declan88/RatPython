"""
Every player action and what triggers it, per device. Data only - InputManager (input_manager.py) reads these, so a
rebind (or a new action) is an edit here.

Controller bindings use SDL's GameController layout: any Xbox-style pad, named as on an Xbox controller.
"""

import pygame


class Action:
    JUMP = "jump"
    CROUCH = "crouch"
    SPRINT = "sprint"
    FIRE = "fire"
    AIM = "aim"
    RELOAD = "reload"
    NEXT_WEAPON = "next_weapon"
    PREV_WEAPON = "prev_weapon"
    EMOTE_WHEEL = "emote_wheel"
    SCOREBOARD = "scoreboard"
    PAUSE = "pause"
    UI_CLICK = "ui_click"     # menus: click where the (stick-driven) cursor is
    UI_BACK = "ui_back"       # menus: back out (close a dropdown, then the pause menu)
    UI_UP = "ui_up"           # menus: jump to the next control in that direction (Modules/UI/gamepad_nav.py)
    UI_DOWN = "ui_down"
    UI_LEFT = "ui_left"
    UI_RIGHT = "ui_right"


# Movement keys, as (forward, back, left, right).
MOVE_KEYS = ((pygame.K_w,), (pygame.K_s,), (pygame.K_a,), (pygame.K_d,))

KEYBOARD = {
    Action.JUMP: (pygame.K_SPACE,),
    Action.CROUCH: (pygame.K_LCTRL, pygame.K_RCTRL),
    Action.SPRINT: (pygame.K_LSHIFT, pygame.K_RSHIFT),
    Action.RELOAD: (pygame.K_r,),
    Action.EMOTE_WHEEL: (pygame.K_b,),
    Action.SCOREBOARD: (pygame.K_TAB,),
}

MOUSE_BUTTONS = {             # pygame button numbers: 1 left, 3 right
    Action.FIRE: (1,),
    Action.AIM: (3,),
}
# The mouse wheel: scrolling down = next weapon, up = previous.
WHEEL_DOWN = Action.NEXT_WEAPON
WHEEL_UP = Action.PREV_WEAPON

CONTROLLER_BUTTONS = {
    Action.JUMP: (pygame.CONTROLLER_BUTTON_A,),
    Action.CROUCH: (pygame.CONTROLLER_BUTTON_B,),
    Action.SPRINT: (pygame.CONTROLLER_BUTTON_LEFTSTICK,),
    Action.RELOAD: (pygame.CONTROLLER_BUTTON_X,),
    Action.NEXT_WEAPON: (pygame.CONTROLLER_BUTTON_Y, pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
                         pygame.CONTROLLER_BUTTON_DPAD_RIGHT),
    Action.PREV_WEAPON: (pygame.CONTROLLER_BUTTON_LEFTSHOULDER, pygame.CONTROLLER_BUTTON_DPAD_LEFT),
    Action.EMOTE_WHEEL: (pygame.CONTROLLER_BUTTON_DPAD_DOWN,),
    Action.SCOREBOARD: (pygame.CONTROLLER_BUTTON_BACK,),
    Action.PAUSE: (pygame.CONTROLLER_BUTTON_START,),
    Action.UI_CLICK: (pygame.CONTROLLER_BUTTON_A,),
    Action.UI_BACK: (pygame.CONTROLLER_BUTTON_B,),
    Action.UI_UP: (pygame.CONTROLLER_BUTTON_DPAD_UP,),
    Action.UI_DOWN: (pygame.CONTROLLER_BUTTON_DPAD_DOWN,),
    Action.UI_LEFT: (pygame.CONTROLLER_BUTTON_DPAD_LEFT,),
    Action.UI_RIGHT: (pygame.CONTROLLER_BUTTON_DPAD_RIGHT,),
}

CONTROLLER_TRIGGERS = {       # analog triggers, pressed past TRIGGER_THRESHOLD
    Action.FIRE: pygame.CONTROLLER_AXIS_TRIGGERRIGHT,
    Action.AIM: pygame.CONTROLLER_AXIS_TRIGGERLEFT,
}
TRIGGER_THRESHOLD = 0.35

# Sticks.
STICK_DEADZONE = 0.15         # radial; the rest of the range is rescaled to start at 0
LOOK_YAW_SPEED = 260.0        # degrees/second at full right-stick deflection
LOOK_PITCH_SPEED = 180.0
LOOK_CURVE = 2.2              # >1: fine control near centre, full speed at the edge
SPRINT_RELEASE = 0.3          # L3 sprint stays on until the left stick drops below this
CURSOR_SPEED = 1400.0         # pixels/second at full deflection, menus
SCROLL_INTERVAL = 0.12        # seconds between wheel steps with the right stick held, menus
NAV_REPEAT_DELAY = 0.35       # menus: a held D-pad direction repeats after this...
NAV_REPEAT_INTERVAL = 0.09    # ...then this often

# Rumble: (low-frequency motor, high-frequency motor, seconds) - the game scales the motors by strength.
RUMBLE_FIRE = (0.10, 0.30, 0.06)
RUMBLE_HIT = (0.0, 0.25, 0.05)       # one of our shots landed
RUMBLE_DAMAGE = (0.55, 0.35, 0.18)   # we were hit (scaled by the damage)
RUMBLE_DEATH = (1.0, 0.6, 0.45)
