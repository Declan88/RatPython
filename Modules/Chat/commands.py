"""
Admin slash-commands, typed into chat as "/name arg1 arg2 ..." - see
Modules/UI/chatbox.py for the input box/autocomplete/dispatch itself (this
file only holds the registry and each command's own logic, no UI code).
Only usable by ADMIN_STEAM_ID - chatbox.py checks is_admin() before it ever
calls run() at all, so an unregistered/non-admin "/"-message can't reach any
handler here.

Add a new command by writing a function below decorated with @register(...)
and nothing else - the chat box's autocomplete list, its help text, and
dispatch are all driven off this one registry, not a second list to keep in
sync.

A handler takes (args, ctx): args is the command's own text split on
whitespace (never including the command NAME itself, already stripped by
run()), and ctx is a small dict app.py builds (see its own commands_ctx) of
plain callables into the running game a handler needs - so this file stays
independent of app.py's own internals (a handler never reaches into globals,
just whatever ctx handed it) and easy to unit-test on its own. Returns a
string to show as a local system message in chat, or None for no feedback.
"""

# The one Steam account allowed to run commands at all. Not a real
# server-enforced permission system - there's no server here - every
# client independently decides whether ITS OWN player may run a command
# against ITS OWN game state (add a dummy only it can see, kill only
# itself); nothing here can affect another player's game. Change this to
# your own Steam64 id (printed at startup as "Your Steam ID: ...") to make
# yourself the admin on your own machine.
ADMIN_STEAM_ID = 76561198283674475


def is_admin(steam_id):
    return steam_id == ADMIN_STEAM_ID


class Command:
    __slots__ = ("name", "handler", "help_text", "arg_hint", "arg_source")

    def __init__(self, name, handler, help_text, arg_hint, arg_source):
        self.name = name
        self.handler = handler
        self.help_text = help_text
        self.arg_hint = arg_hint
        # None, or a key into ctx (see the module docstring) naming a
        # zero-arg callable that returns [str, ...] candidates for this
        # command's one argument - e.g. /kill's "kill_targets" (app.py's own
        # list of live player names) - so the chat box (Modules/UI/chatbox.py)
        # can autocomplete/list them once the command name itself is typed
        # and a space follows it, the same way it autocompletes command
        # names themselves. A command that takes no argument, or an
        # unstructured one autocomplete can't help with, leaves this None.
        self.arg_source = arg_source

    @property
    def usage(self):
        return f"{self.name} {self.arg_hint}".strip()


_registry = {}


def register(help_text, arg_hint="", arg_source=None):
    """Decorator: @register("what it does", "<optional arg hint>",
    arg_source="ctx key naming the argument's autocomplete source"). The
    command's own name is the decorated function's name with a leading
    "_cmd_" stripped (see the commands below) - just naming convention, nothing
    reads the function's __name__ at runtime besides this decorator itself."""
    def deco(fn):
        name = fn.__name__
        if name.startswith("_cmd_"):
            name = name[len("_cmd_"):]
        name = name.lower()
        _registry[name] = Command(name, fn, help_text, arg_hint, arg_source)
        return fn
    return deco


def all_commands():
    """Every registered command, alphabetical - what the chat box's
    autocomplete list shows."""
    return sorted(_registry.values(), key=lambda c: c.name)


def find(name):
    return _registry.get(name.lower())


def run(name, args, ctx):
    """Looks up name and calls its handler(args, ctx). Returns the handler's
    own feedback string, or an "unknown command" message if name isn't
    registered - always returns SOMETHING to show in chat, so a typo doesn't
    just silently do nothing."""
    cmd = find(name)
    if cmd is None:
        return f"Unknown command: /{name} (press / to see what's available)"
    try:
        return cmd.handler(args, ctx)
    except Exception as e:
        return f"/{name} failed: {e}"


# ---- commands ------------------------------------------------------------

@register("Spawns the test dummy near your spawn point (or respawns it if it's already up).")
def _cmd_adddummy(args, ctx):
    return ctx["add_dummy"]()


@register("Adds a weapon to your inventory by id (tab-complete lists them).",
          "<weapon>", arg_source="weapon_ids")
def _cmd_give(args, ctx):
    if not args:
        return "Usage: /give <weapon> - " + ", ".join(ctx["weapon_ids"]())
    return ctx["give_weapon"](args[0].lower())


@register("Kills you, or a named connected player/the test dummy if given "
          "(tab-complete their name) - defaults to yourself.",
          "[player]", arg_source="kill_targets")
def _cmd_kill(args, ctx):
    if not args:
        return ctx["kill_self"]()
    return ctx["kill_player"](" ".join(args))


@register("Same as /kill, but with a lightning strike (particle, explosion + thunder) on "
          "whoever it hits - you, a named connected player, or the test dummy - defaults "
          "to yourself.", "[player]", arg_source="kill_targets")
def _cmd_smite(args, ctx):
    if not args:
        return ctx["smite_self"]()
    return ctx["smite_player"](" ".join(args))
