"""The standard tracer: a short bright streak flying from the muzzle to the impact."""

from .base import TracerStyle


class Streak(TracerStyle):
    name = "default"
    width = 1.0
    speed = 320.0     # metres per second the head travels
    length = 9.0      # metres from head to tail

    fragment = """
    vec3 shade(vec2 uv, float fade, float metres, float total, float seed) {
        float across = 1.0 - abs(uv.y);
        float core = across * across;
        float halo = across * 0.35;
        float along = clamp(uv.x, 0.0, 1.0);
        float body = 0.15 + 0.85 * along * along;         // bright head, fading tail
        float nose = 1.0 - smoothstep(0.93, 1.0, along);  // soft tip
        vec3 hot = vec3(1.0, 0.96, 0.82);
        vec3 warm = vec3(1.0, 0.55, 0.15);
        return mix(warm, hot, core) * (core + halo) * body * nose * 1.6 * fade;
    }
    """

    def segment(self, age, distance):
        head = self.speed * (age + 1.0 / 60.0)
        tail = head - self.length
        if tail >= distance:
            return None
        a, b = max(tail, 0.0), min(head, distance)
        return a, b, (a - tail) / self.length, (b - tail) / self.length, 1.0


STYLE = Streak()
