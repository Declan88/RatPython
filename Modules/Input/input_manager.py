"""
One place the game asks about input, whichever device it comes from: keyboard + mouse or an Xbox-style controller
(any number, hot-pluggable, all merged). See bindings.py for what triggers what.

Per frame:
    input.begin_frame()                       # before the window's events are read
    ... for each event: input.handle_event(event, gameplay=...)
    input.update(dt)                          # polls the controllers
    input.held(Action.JUMP), input.pressed(Action.FIRE), input.move_vector(), input.look_delta(dt) ...

`device` is whichever was used last ("keyboard" or "controller") - aim assist only runs for the controller, and the
menus are driven by the stick only while it's the controller. Presses are counted from events, so none is lost
between frames; `gameplay` tells handle_event whether a key/click is game input at all (not while typing in chat
or clicking a menu).
"""

import math
import time
from collections import Counter

import pygame

from Modules.settings import settings
from . import bindings
from .bindings import Action

_NAV = ((Action.UI_UP, "up"), (Action.UI_DOWN, "down"), (Action.UI_LEFT, "left"), (Action.UI_RIGHT, "right"))

try:
    from pygame._sdl2 import controller as _sdl_controller
except ImportError:       # a pygame without SDL's GameController API: keyboard and mouse only
    _sdl_controller = None

KEYBOARD = "keyboard"
CONTROLLER = "controller"

_KEY_ACTIONS = {}
for _action, _keys in bindings.KEYBOARD.items():
    for _key in _keys:
        _KEY_ACTIONS.setdefault(_key, []).append(_action)
_MOUSE_ACTIONS = {}
for _action, _buttons in bindings.MOUSE_BUTTONS.items():
    for _button in _buttons:
        _MOUSE_ACTIONS.setdefault(_button, []).append(_action)


def _stick(x, y):
    """Raw stick axes (-1..1) -> (x, y) with a radial deadzone, rescaled so it starts at 0 just past it."""
    magnitude = math.hypot(x, y)
    if magnitude <= bindings.STICK_DEADZONE:
        return 0.0, 0.0
    scaled = min(1.0, (magnitude - bindings.STICK_DEADZONE) / (1.0 - bindings.STICK_DEADZONE))
    return x / magnitude * scaled, y / magnitude * scaled


class InputManager:
    def __init__(self):
        self.device = KEYBOARD
        self._controllers = {}            # instance id -> pygame._sdl2.controller.Controller
        self._pressed = Counter()         # action -> presses this frame
        self._pad_held = set()            # controller actions held at the last poll
        self._left = self._right = (0.0, 0.0)
        self._triggers = {}               # axis -> 0..1
        self._sprint_latched = False
        self._cursor_moved_at = 0.0       # when the stick last moved the menu cursor (its own motion events)
        self._scroll_wait = 0.0
        self._nav_held = {}               # D-pad direction -> seconds until it repeats
        self._keys = None
        self._mouse = (False, False, False)
        # Set by the game: on_connected(name), on_disconnected(name, was_in_use).
        self.on_connected = None
        self.on_disconnected = None
        if _sdl_controller is not None:
            try:
                _sdl_controller.init()
                for index in range(_sdl_controller.get_count()):
                    self._open(index)
            except pygame.error as e:
                print(f"[input] controller support unavailable: {e}")

    # ---- devices -----------------------------------------------------

    def _open(self, index):
        if _sdl_controller is None or not _sdl_controller.is_controller(index):
            return
        try:
            pad = _sdl_controller.Controller(index)
            instance_id = pad.as_joystick().get_instance_id()
            if instance_id in self._controllers:
                return      # SDL also announces pads that were already connected (and opened) at startup
            self._controllers[instance_id] = pad
            print(f"[input] controller connected: {pad.name}")
            if self.on_connected is not None:
                self.on_connected(pad.name)
        except pygame.error as e:
            print(f"[input] couldn't open controller {index}: {e}")

    def _close(self, instance_id):
        pad = self._controllers.pop(instance_id, None)
        if pad is not None:
            print(f"[input] controller disconnected: {pad.name}")
            was_in_use = self.device == CONTROLLER
            if not self._controllers and was_in_use:
                self.device = KEYBOARD
            if self.on_disconnected is not None:
                self.on_disconnected(pad.name, was_in_use)

    @property
    def has_controller(self):
        return bool(self._controllers)

    # ---- per frame ---------------------------------------------------

    def begin_frame(self):
        """Forgets last frame's presses. Call before the frame's events are handled."""
        self._pressed.clear()

    def handle_event(self, event, gameplay=True):
        """Feed every pygame event (it's never consumed here). gameplay: whether a key or click counts as game
        input right now; it switches the active device either way."""
        t = event.type
        if t == pygame.CONTROLLERDEVICEADDED:
            self._open(event.device_index)
        elif t == pygame.CONTROLLERDEVICEREMOVED:
            self._close(event.instance_id)
        elif getattr(event, "synthetic", False):
            return                          # posted by drive_ui_cursor on the controller's behalf
        elif t == pygame.KEYDOWN:
            self.device = KEYBOARD
            if gameplay:
                for action in _KEY_ACTIONS.get(event.key, ()):
                    self._pressed[action] += 1
        elif t == pygame.MOUSEBUTTONDOWN:
            self.device = KEYBOARD
            if gameplay:
                for action in _MOUSE_ACTIONS.get(event.button, ()):
                    self._pressed[action] += 1
        elif t == pygame.MOUSEWHEEL:
            self.device = KEYBOARD
            if gameplay and event.y:
                self._pressed[bindings.WHEEL_UP if event.y > 0 else bindings.WHEEL_DOWN] += 1
        elif t == pygame.MOUSEMOTION:
            # The stick moving the menu cursor makes motion events too: only a real, deliberate move counts.
            if time.perf_counter() - self._cursor_moved_at > 0.15 and abs(event.rel[0]) + abs(event.rel[1]) > 3:
                self.device = KEYBOARD

    def update(self, dt):
        """Polls keyboard/mouse state and the controllers, once per frame after the events."""
        self._keys = pygame.key.get_pressed()
        self._mouse = pygame.mouse.get_pressed()
        held = set()
        left = right = (0.0, 0.0)
        triggers = {}
        for pad in self._controllers.values():
            try:
                for action, buttons in bindings.CONTROLLER_BUTTONS.items():
                    if any(pad.get_button(b) for b in buttons):
                        held.add(action)
                for action, axis in bindings.CONTROLLER_TRIGGERS.items():
                    value = max(0.0, pad.get_axis(axis) / 32767.0)
                    triggers[axis] = max(triggers.get(axis, 0.0), value)
                    if value >= bindings.TRIGGER_THRESHOLD:
                        held.add(action)
                pl = _stick(pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX) / 32767.0,
                            pad.get_axis(pygame.CONTROLLER_AXIS_LEFTY) / 32767.0)
                pr = _stick(pad.get_axis(pygame.CONTROLLER_AXIS_RIGHTX) / 32767.0,
                            pad.get_axis(pygame.CONTROLLER_AXIS_RIGHTY) / 32767.0)
            except pygame.error:
                continue
            # Several pads: whichever stick is pushed further wins.
            if math.hypot(*pl) > math.hypot(*left):
                left = pl
            if math.hypot(*pr) > math.hypot(*right):
                right = pr
        newly = held - self._pad_held
        for action in newly:
            self._pressed[action] += 1
        if held or left != (0.0, 0.0) or right != (0.0, 0.0) or any(v > 0.05 for v in triggers.values()):
            self.device = CONTROLLER
        self._pad_held, self._left, self._right, self._triggers = held, left, right, triggers
        # L3 sprint latches on until the stick (nearly) stops.
        if Action.SPRINT in newly:
            self._sprint_latched = True
        if math.hypot(*left) < bindings.SPRINT_RELEASE:
            self._sprint_latched = False

    # ---- queries -----------------------------------------------------

    def pressed(self, action):
        """How many times `action` was pressed this frame (0 if not)."""
        return self._pressed.get(action, 0)

    def held(self, action):
        """Whether `action` is held on any device."""
        if action == Action.SPRINT and self._sprint_latched:
            return True
        if action in self._pad_held:
            return True
        if self._keys is not None and any(self._keys[k] for k in bindings.KEYBOARD.get(action, ())):
            return True
        return any(self._mouse[b - 1] for b in bindings.MOUSE_BUTTONS.get(action, ()) if 1 <= b <= 3)

    def move_vector(self):
        """(right, forward), each -1..1: the left stick (its tilt = speed) or WASD."""
        if self._left != (0.0, 0.0):
            return self._left[0], -self._left[1]
        if self._keys is None:
            return 0.0, 0.0
        forward, back, left, right = (any(self._keys[k] for k in keys) for keys in bindings.MOVE_KEYS)
        return float(right) - float(left), float(forward) - float(back)

    def look_delta(self, dt, scale=1.0):
        """(yaw, pitch) degrees to turn this frame from the right stick (0, 0 for the mouse - the window applies
        mouse look itself). scale: e.g. the scope's zoom ratio."""
        x, y = self._right
        if x == 0.0 and y == 0.0:
            return 0.0, 0.0
        magnitude = math.hypot(x, y)
        curved = magnitude ** bindings.LOOK_CURVE / magnitude
        scale *= settings.controller_sensitivity
        pitch_sign = 1.0 if settings.controller_invert_y else -1.0   # stick up (negative y) looks up
        return (x * curved * bindings.LOOK_YAW_SPEED * dt * scale,
                pitch_sign * y * curved * bindings.LOOK_PITCH_SPEED * dt * scale)

    def right_stick(self):
        return self._right

    def rumble(self, effect, strength=1.0):
        """Vibrates the controllers - effect: (low motor, high motor, seconds), see bindings.RUMBLE_*. Only while
        the controller is the device in use, and if vibration is on in the settings."""
        if self.device != CONTROLLER or not settings.controller_vibration:
            return
        low, high, seconds = effect
        strength = max(0.0, min(1.0, strength))
        for pad in self._controllers.values():
            try:
                pad.rumble(low * strength, high * strength, int(seconds * 1000))
            except (pygame.error, AttributeError):
                pass

    # ---- menus -------------------------------------------------------

    def cursor_moved(self):
        """Call after moving the mouse cursor on the controller's behalf (menu navigation), so the motion that
        causes isn't taken for the mouse being used."""
        self._cursor_moved_at = time.perf_counter()

    def nav_presses(self, dt):
        """D-pad directions ("up", "down", "left", "right") to move menu focus this frame: on press, then
        repeating while held."""
        out = []
        for action, direction in _NAV:
            if action not in self._pad_held:
                self._nav_held.pop(direction, None)
            elif direction not in self._nav_held:
                self._nav_held[direction] = bindings.NAV_REPEAT_DELAY
                out.append(direction)
            else:
                self._nav_held[direction] -= dt
                if self._nav_held[direction] <= 0.0:
                    self._nav_held[direction] = bindings.NAV_REPEAT_INTERVAL
                    out.append(direction)
        return out

    def drive_ui_cursor(self, dt, window_size):
        """While the controller is in use and the cursor is free (menus): the left stick (or the right stick's
        vertical, for scrolling) drives the real mouse cursor and A clicks - as synthetic pygame events, so the UI
        needs no controller code."""
        if self.device != CONTROLLER:
            return
        x, y = self._left
        if x or y:
            magnitude = math.hypot(x, y)
            speed = bindings.CURSOR_SPEED * magnitude ** 1.5 / magnitude * dt
            cx, cy = pygame.mouse.get_pos()
            nx = min(max(cx + x * speed, 0), window_size[0] - 1)
            ny = min(max(cy + y * speed, 0), window_size[1] - 1)
            pygame.mouse.set_pos((round(nx), round(ny)))
            self.cursor_moved()
        pos = pygame.mouse.get_pos()
        for _ in range(self.pressed(Action.UI_CLICK)):
            for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                pygame.event.post(pygame.event.Event(kind, button=1, pos=pos, synthetic=True))
        scroll = self._right[1]
        self._scroll_wait -= dt
        if abs(scroll) > 0.5 and self._scroll_wait <= 0.0:
            pygame.event.post(pygame.event.Event(
                pygame.MOUSEWHEEL, x=0, y=-1 if scroll > 0 else 1, flipped=False, synthetic=True))
            self._scroll_wait = bindings.SCROLL_INTERVAL
