"""
In-game text chat: Enter opens a one-line input (frees the cursor while
typing, exactly like the pause menu does - see ChatBox.open/_close), and
recent messages fade into a short log just above it. A message starting with
"/" is an admin-only command (see Modules/Chat/commands.py) instead of a
chat line - typing one shows a live list of matching commands, and Tab
cycles through matches (repeated presses step to the next one, wrapping
around - see _tab_complete), both admin-only; a non-admin's "/text" is just
sent as an ordinary chat message. Up/Down recall past submitted lines
(chat messages and commands alike), same as a shell's own history.
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
    """TextInput plus Tab-to-cycle-complete and Up/Down history recall -
    everything else (typing, Enter, Escape, selection...) is exactly
    TextInput's own behavior."""

    def __init__(self, on_tab=None, on_history=None, **kw):
        super().__init__(**kw)
        self.on_tab = on_tab
        self.on_history = on_history
        # Set for exactly one TEXTINPUT event right after Tab/Up/Down or a caller-seeded
        # open() - none of those are meant to type a character, but the OS/SDL can still
        # deliver a genuine TEXTINPUT echo of the SAME physical keypress that triggered
        # them (confirmed for "/" reopening chat - see app.py's own comment on that key),
        # which would otherwise land as an ordinary typed character right afterward.
        self.swallow_next_text = False

    def on_key(self, event):
        if event.key == pygame.K_TAB and self.on_tab is not None:
            self.on_tab()
            self.swallow_next_text = True
            return
        if event.key == pygame.K_UP and self.on_history is not None:
            self.on_history(1)
            self.swallow_next_text = True
            return
        if event.key == pygame.K_DOWN and self.on_history is not None:
            self.on_history(-1)
            self.swallow_next_text = True
            return
        super().on_key(event)

    def on_text(self, text):
        if self.swallow_next_text:
            self.swallow_next_text = False
            return
        super().on_text(text)


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
            on_submit=self._submit, on_tab=self._tab_complete, on_history=self._history_step,
            on_change=self._on_input_changed,
        ))
        self._open = False
        self._suggestion_key = None
        # Tab-cycle state (see _tab_complete's own docstring) - _tab_prefix is the text a
        # cycle started from, kept fixed across repeated Tab presses so each one re-filters
        # from what was actually TYPED rather than from the completion the last press wrote.
        # Ended by _on_input_changed (a REAL edit) rather than by comparing text against
        # what the last Tab press wrote - a plain string comparison is one more thing an
        # unrelated stray event could quietly defeat (see ChatInput's own swallow_next_text
        # comment on why those aren't as rare as they sound), where an edit CALLBACK can't
        # fire without on_change actually running.
        self._tab_prefix = None
        self._tab_index = -1
        # Command/chat history (see _history_step) - oldest first; _history_pos -1 means
        # "editing a live line", 0 the most recently submitted entry, increasing = older.
        self._history = []
        self._history_pos = -1
        self._history_draft = ""   # what was being typed before the first Up press

    @property
    def focused(self):
        return self._open

    # ---- open/close -------------------------------------------------

    def open(self, initial_text=""):
        """Opens the chat input, optionally with some text already typed (app.py
        opens it with "/" when / is pressed, straight into command mode)."""
        if self.manager is None or self._open:
            return
        self._open = True
        self.input.visible = True
        self.input.set_text(initial_text, notify=bool(initial_text))
        if initial_text:
            self.input.swallow_next_text = True
        self._reset_tab_cycle()
        self._history_pos = -1
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
        self._reset_tab_cycle()
        self._history_pos = -1
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
        if not self._history or self._history[-1] != text:
            self._history.append(text)
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

    def _suggestions_for(self, text):
        """(mode, items) as if `text` were currently typed - see _suggestions for the
        full mode/items contract. Split out from _suggestions so _tab_complete can keep
        re-deriving the SAME candidate list from the prefix the user actually TYPED,
        instead of from whatever completion the last Tab press already wrote into the
        input (which would otherwise become its own filter and mostly just match
        itself)."""
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
        return self._suggestions_for(self.input.text)

    def _reset_tab_cycle(self):
        self._tab_prefix = None
        self._tab_index = -1

    def _on_input_changed(self, text):
        """Fires on every REAL edit (typing, backspace...) - never on Tab-completion or
        history recall, which both write via set_text(..., notify=False) specifically to
        avoid re-triggering this. Ends whatever Tab-cycle was in progress, so the NEXT Tab
        press starts fresh from what's now typed instead of continuing to cycle a
        candidate list computed from stale, already-superseded text."""
        self._reset_tab_cycle()

    def _tab_complete(self):
        """Cycles through matches instead of always jumping to the first: a fresh cycle
        starts from whatever's actually typed right now, and keeps advancing to the next
        candidate (wrapping back to the first after the last) on every further Tab press
        UNTIL a real edit ends it (see _on_input_changed) - not until the input merely
        stops matching what the last press wrote, which is one plain-string-comparison
        away from silently breaking (see ChatInput's own swallow_next_text comment)."""
        if self._tab_prefix is None:
            self._tab_prefix = self.input.text
            self._tab_index = -1
        mode, items = self._suggestions_for(self._tab_prefix)
        if not items:
            return
        self._tab_index = (self._tab_index + 1) % len(items)
        if mode == "command":
            applied = f"/{items[self._tab_index].name} "
            if len(items) == 1:
                # A single, definite command name (not one of several still competing for
                # this prefix) - the completion below adds a trailing space, moving into
                # argument territory, but does so via set_text(..., notify=False)
                # specifically so it does NOT end the cycle (see _on_input_changed) the way
                # a real edit would. Without this, _tab_prefix would stay frozen on the
                # pre-space text forever, and every further Tab press would keep re-deriving
                # "command" mode from it and re-picking this same one match - looking
                # exactly like Tab got stuck, since it'd never notice a space now follows
                # the name and this became an argument to cycle instead. Advancing the
                # prefix here lets the NEXT press naturally re-derive "arg" mode from it.
                self._tab_prefix = applied
                self._tab_index = -1
        else:
            cmd_name = self._tab_prefix[1:].split(" ", 1)[0]
            applied = f"/{cmd_name} {items[self._tab_index]}"
        self.input.set_text(applied, notify=False)

    def _history_step(self, direction):
        """direction +1 (Up) moves toward older entries, -1 (Down) toward newer - same
        convention as a shell's own history recall. Whatever was being typed before the
        FIRST Up press is remembered (self._history_draft) so Down past the newest
        history entry restores it instead of leaving the last recalled line sitting
        there with no way back to what was actually being composed."""
        if not self._history:
            return
        if self._history_pos == -1:
            if direction < 0:
                return   # already on the live line, nothing newer to go to
            self._history_draft = self.input.text
        new_pos = self._history_pos + direction
        if new_pos < -1 or new_pos >= len(self._history):
            return
        self._history_pos = new_pos
        text = self._history_draft if new_pos == -1 else self._history[-(new_pos + 1)]
        self.input.set_text(text, notify=False)
        self._reset_tab_cycle()   # browsing history shouldn't chain into an unrelated tab-cycle

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
