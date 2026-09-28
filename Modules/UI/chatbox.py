"""
In-game text chat: Enter opens a one-line input (frees the cursor while
typing, exactly like the pause menu does - see ChatBox.open/_close), and
recent messages fade into a short log just above it. A message starting with
"/" is an admin-only command (see Modules/Chat/commands.py) instead of a
chat line - typing one shows a live list of matching commands, and Tab
completes to the best match, both admin-only; a non-admin's "/text" is just
sent as an ordinary chat message.
"""

import time

import pygame

from . import theme
from .inputs import TextInput
from .widgets import Anchor, Label, Panel
from Modules.Chat import commands

LOG_LINES = 8
LOG_LINE_HEIGHT = 24
LOG_DISPLAY_SECONDS = 8.0
LOG_FADE_SECONDS = 1.0
MAX_MESSAGE_LENGTH = 200
MAX_SUGGESTIONS = 6


class ChatInput(TextInput):
    """TextInput plus Tab-to-complete - everything else (typing, Enter,
    Escape, selection...) is exactly TextInput's own behavior."""

    def __init__(self, on_tab=None, **kw):
        super().__init__(**kw)
        self.on_tab = on_tab

    def on_key(self, event):
        if event.key == pygame.K_TAB and self.on_tab is not None:
            self.on_tab()
            return
        super().on_key(event)


class ChatBox(Panel):
    def __init__(self, is_admin=lambda: False, local_name=lambda: "You",
                 send_chat=lambda text: None, ctx=None, **kw):
        kw.setdefault("anchor", Anchor.BOTTOM_LEFT)
        # Well above the health bar/paper doll (bottom-left, up to about y=-150
        # - see app.py's health_bar offset and PaperDoll's own circle/head
        # extent) so the chat log and input never sit behind either.
        kw.setdefault("offset", (24, -210))
        kw.setdefault("size", (560, 0))
        super().__init__(color=(0, 0, 0, 0), **kw)
        self.is_admin = is_admin
        self.local_name = local_name
        self.send_chat = send_chat
        # Plain callables a command handler needs (Modules/Chat/commands.py's
        # own docstring) - app.py builds this once (starting EMPTY, filled in
        # once the callables it needs exist - see app.py's own commands_ctx)
        # and hands it straight through; nothing here reads its contents.
        # NOT "ctx or {}" - that discards the caller's dict as soon as it's
        # empty (empty dicts are falsy), which is exactly its state at this
        # point, silently detaching self.ctx from app.py's own commands_ctx
        # and leaving every command's ctx["..."] lookup a KeyError forever.
        self.ctx = ctx if ctx is not None else {}

        self._log = []   # [{"start": t, "label": Label}], oldest first
        self.log_panel = self.add(Panel(color=(0, 0, 0, 0), anchor=Anchor.BOTTOM_LEFT,
                                        pivot=(0.0, 1.0), size=(560, 0)))
        # The suggestion list floats ABOVE the input (negative offset - this
        # panel is itself bottom-anchored, see class docstring) - built fresh
        # each frame it's showing (see _update_suggestions), cheap since it's
        # at most MAX_SUGGESTIONS short labels and only rebuilt while the
        # input is actually open.
        self.suggestions = self.add(Panel(color=theme.CARD, visible=False, anchor=Anchor.BOTTOM_LEFT,
                                          pivot=(0.0, 1.0), offset=(0, -48), layout="vertical",
                                          padding=10, spacing=4, fit_content=True))
        self.input = self.add(ChatInput(
            placeholder="Press Enter to chat, / for commands", visible=False,
            max_length=MAX_MESSAGE_LENGTH, size=(560, 42), font_size=22,
            anchor=Anchor.BOTTOM_LEFT,
            on_submit=self._submit, on_tab=self._tab_complete,
        ))
        self._open = False
        self._suggestion_key = None

    @property
    def focused(self):
        return self._open

    # ---- open/close -------------------------------------------------

    def open(self):
        if self.manager is None or self._open:
            return
        self._open = True
        self.input.visible = True
        self.input.set_text("", notify=False)
        self.manager.set_cursor_free(True)
        self.manager.set_focus(self.input)

    def close(self):
        """Public close - e.g. app.py's disconnect() calls this to make sure
        chat isn't left open (and the cursor free) after leaving a match."""
        if self._open:
            self._close()

    def _close(self):
        self._open = False
        self.input.visible = False
        self.suggestions.visible = False
        self._suggestion_key = None
        if self.manager is not None:
            # set_cursor_free(False) already clears focus itself (see
            # UIManager.set_cursor_free) - nothing more to do here.
            self.manager.set_cursor_free(False)

    # ---- per-frame ----------------------------------------------------

    def update(self):
        """Call once a frame (app.py's main loop does, unconditionally - cheap
        while closed and empty, same shape as KillFeed/DeathScreen)."""
        # Escape (handled by TextInput itself, which just clears focus) or any
        # other way focus moved off the input closes chat and restores the
        # cursor - polled rather than an on_blur hook so EVERY way focus can
        # leave (not just Escape) is covered by one path.
        if self._open and (self.manager is None or self.manager.focus is not self.input):
            self._close()
        if self._open:
            self._update_suggestions()
        if not self._log:
            return
        now = time.perf_counter()
        total = LOG_DISPLAY_SECONDS + LOG_FADE_SECONDS
        changed = False
        for entry in list(self._log):
            elapsed = now - entry["start"]
            if elapsed >= total:
                self._log.remove(entry)
                self.log_panel.remove(entry["label"])
                changed = True
                continue
            alpha = 255
            if elapsed > LOG_DISPLAY_SECONDS:
                alpha = round(255 * (1.0 - (elapsed - LOG_DISPLAY_SECONDS) / LOG_FADE_SECONDS))
            entry["label"].color = (*entry["color"][:3], round(entry["color"][3] * alpha / 255))
        if changed:
            self._reflow_log()

    # ---- sending / commands --------------------------------------------

    def _submit(self, text):
        text = text.strip()
        self._close()
        if not text:
            return
        if text.startswith("/"):
            if self.is_admin():
                self._run_command(text[1:])
            else:
                self.push_system("Only an admin can use commands.")
            return
        self.push_message(self.local_name(), text, mine=True)
        self.send_chat(text)

    def _run_command(self, rest):
        parts = rest.split()
        if not parts:
            return
        result = commands.run(parts[0], parts[1:], self.ctx)
        if result:
            self.push_system(result)

    def _suggestions(self):
        """(mode, items) for whatever's being typed right now - shared by the
        live suggestion list and Tab-completion so they never disagree.

        mode "command", items [Command, ...]: still typing the command's own
        name (no space yet) - matched by name prefix.

        mode "arg", items [str, ...]: a space follows a recognized command
        that declares an arg_source (see commands.Command's own docstring,
        e.g. /kill's "kill_targets") - candidates from that ctx-provided
        list, matched by prefix against whatever's typed after the space.
        Player names can contain spaces themselves ("Test Dummy"), so the
        WHOLE remainder after the command name is the filter text, not just
        its last word.

        (None, []): not admin, not in "/" mode, or the matched command has
        no argument to autocomplete."""
        text = self.input.text
        if not (text.startswith("/") and self.is_admin()):
            return None, []
        body = text[1:]
        if " " not in body:
            typed = body.lower()
            return "command", [c for c in commands.all_commands() if c.name.startswith(typed)]
        cmd_name, rest = body.split(" ", 1)
        cmd = commands.find(cmd_name)
        if cmd is None or not cmd.arg_source:
            return None, []
        provider = self.ctx.get(cmd.arg_source)
        options = provider() if provider is not None else []
        typed = rest.lower()
        return "arg", [name for name in options if name.lower().startswith(typed)]

    def _tab_complete(self):
        mode, items = self._suggestions()
        if not items:
            return
        if mode == "command":
            self.input.set_text(f"/{items[0].name} ", notify=False)
        elif mode == "arg":
            cmd_name = self.input.text[1:].split(" ", 1)[0]
            self.input.set_text(f"/{cmd_name} {items[0]}", notify=False)

    def _update_suggestions(self):
        mode, items = self._suggestions()
        if mode is None:
            self.suggestions.visible = False
            self._suggestion_key = None
            return
        items = items[:MAX_SUGGESTIONS]
        key = (mode, tuple(items) if mode == "arg" else tuple(c.name for c in items))
        self.suggestions.visible = True
        if key == self._suggestion_key:
            return
        self._suggestion_key = key
        self.suggestions.clear_children()
        if not items:
            self.suggestions.add(Label(
                "No matches" if mode == "arg" else "No matching commands",
                font_size=18, color=theme.MUTED))
        elif mode == "command":
            for i, c in enumerate(items):
                color = theme.ACCENT if i == 0 else theme.TEXT
                self.suggestions.add(Label(f"/{c.usage}", font_size=18, color=color))
                self.suggestions.add(Label(c.help_text, font_size=15, color=theme.MUTED, offset=(14, 0)))
        else:
            for i, name in enumerate(items):
                self.suggestions.add(Label(name, font_size=18, color=theme.ACCENT if i == 0 else theme.TEXT))

    # ---- log ------------------------------------------------------------

    def push_message(self, sender_name, text, mine=False):
        self._push_line(f"{sender_name}: {text}", theme.ACCENT if mine else theme.TEXT)

    def push_system(self, text):
        self._push_line(text, theme.WARN)

    def _push_line(self, text, color):
        lbl = self.log_panel.add(Label(text, font_size=20, color=color, shadow=True))
        self._log.append({"start": time.perf_counter(), "label": lbl, "color": color})
        while len(self._log) > LOG_LINES:
            old = self._log.pop(0)
            self.log_panel.remove(old["label"])
        self._reflow_log()

    def _reflow_log(self):
        # Newest closest to the input (the bottom, since this panel is
        # bottom-pivoted - see __init__): index n-1 (newest) gets the
        # smallest upward offset, index 0 (oldest) the largest.
        n = len(self._log)
        for i, entry in enumerate(self._log):
            entry["label"].offset = (0, -(n - i) * LOG_LINE_HEIGHT)
