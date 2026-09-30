"""
Explosions any weapon can use. A weapon sets `explosion = Explosion(...)`; when
its shot lands, app.py calls explosion.hits(...) and applies the result: damage
to everyone inside the radius, the invoker (whoever fired) taking only a
proportion of it, plus a shove for the gibs of anyone it kills - away from the
blast.

    Explosion(radius=3.0, damage=10.0, self_damage_fraction=0.5)

This file is only the maths - who's where and how damage is delivered
(network, health, gibs) stays in app.py.
"""

import glm


class ExplosionHit:
    __slots__ = ("target_id", "damage", "push", "is_invoker")

    def __init__(self, target_id, damage, push, is_invoker):
        self.target_id = target_id
        self.damage = damage          # health this target loses
        self.push = push              # glm.vec3 (m/s): which way, and how hard, the blast throws it
        self.is_invoker = is_invoker


class Explosion:
    def __init__(self, radius, damage, self_damage_fraction=0.5, falloff=True,
                 push_speed=10.0, push_lift=0.35, blocked_by_walls=True):
        """radius: metres from the blast point. damage: dealt at the centre.
        self_damage_fraction: the proportion of the damage the INVOKER takes if
        inside (0 = never hurts them, 1 = same as anyone). falloff: damage
        fades linearly to nothing at the edge (False = full damage anywhere
        inside). push_speed: gib speed (m/s) at the centre, fading the same way;
        push_lift: extra upward bias on the shove. blocked_by_walls: level
        geometry between the blast and a target shields it."""
        self.radius = radius
        self.damage = damage
        self.self_damage_fraction = self_damage_fraction
        self.falloff = falloff
        self.push_speed = push_speed
        self.push_lift = push_lift
        self.blocked_by_walls = blocked_by_walls

    def strength(self, distance):
        """0..1 of the full effect at `distance` from the centre."""
        if distance >= self.radius:
            return 0.0
        return 1.0 - distance / self.radius if self.falloff else 1.0

    def hits(self, origin, targets, invoker=None, blocked=None):
        """Who the blast at `origin` (world) reaches, as a list of ExplosionHit.
        targets: [(id, centre position)] of everyone else. invoker: (id, centre)
        of the shooter, or None. blocked: callable(origin, position) -> True if
        something solid is in the way (used when blocked_by_walls)."""
        origin = glm.vec3(origin)
        result = []
        candidates = [(tid, pos, False) for tid, pos in targets]
        if invoker is not None:
            candidates.append((invoker[0], invoker[1], True))
        for target_id, position, is_invoker in candidates:
            position = glm.vec3(position)
            offset = position - origin
            distance = glm.length(offset)
            scale = self.strength(distance)
            if scale <= 0.0:
                continue
            if self.blocked_by_walls and blocked is not None and blocked(origin, position):
                continue
            direction = offset / distance if distance > 1e-4 else glm.vec3(0.0, 1.0, 0.0)
            direction = glm.normalize(direction + glm.vec3(0.0, self.push_lift, 0.0))
            damage = self.damage * scale * (self.self_damage_fraction if is_invoker else 1.0)
            result.append(ExplosionHit(target_id, damage, direction * (self.push_speed * scale), is_invoker))
        return result
