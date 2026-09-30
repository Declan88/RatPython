"""
Damage classes: what KIND of damage something deals, independent of which weapon deals it -
right now, just how the KILLED player's body reacts (see DamageClass.death_effect below).
A weapon opts in via its own damage_class class attribute (see WeaponsBase); /smite
(Modules/Chat/commands.py, app.py's own smite_self/smite_player) uses Zap directly, with
no weapon involved at all.

Carried over the network as a short name string (see NetworkManager.notify_death/send_damage's
own damage_class params), not the class itself - get_damage_class() turns that back into one on
the receiving end, the same shape as Modules/Weapons/registry.py's own get_weapon_class.
"""

GIBS = "gibs"          # WeaponsBase's own default - the existing gib-burst death
DISSOLVE = "dissolve"  # a Source-style dissolve (see Modules/Graphics/skeletal_shader.py's
                        # own u_dissolve_amount/RemotePlayer's dissolve state machine) instead


class DamageClass:
    name = "bullet"
    death_effect = GIBS


class Bullet(DamageClass):
    """The default for every weapon that doesn't set its own damage_class - ordinary
    bullet/explosive damage, dies with the existing gib burst."""
    name = "bullet"
    death_effect = GIBS


class Zap(DamageClass):
    """Electrical damage - dies with a dissolve instead of gibs. /smite's own damage class;
    nothing else currently deals it, but any weapon can opt in via its own damage_class."""
    name = "zap"
    death_effect = DISSOLVE
    dissolve_color = (120, 210, 255)   # cool electric blue-cyan - the glowing edge's colour


_registry = {cls.name: cls for cls in (Bullet, Zap)}


def get_damage_class(name):
    """The DamageClass for `name` (see DamageClass.name), or Bullet if name is empty/unknown -
    same "always returns something sane" contract as Modules/Weapons/registry.py's own
    get_weapon_class: a damage class named by a packet from a newer/older build than this
    client's own should still just resolve to a plain kill, not fail to parse the packet at
    all over one unrecognized field."""
    return _registry.get(name, Bullet)
