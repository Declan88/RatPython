"""
A small lightning bolt: a jagged electric-blue line from the muzzle to the impact,
re-jittering a few times as it fades so it crackles instead of sitting still. One bolt per
shot, kept deliberately cheap to shade (4 hashes + 3 segment distances per pixel).

Built from the ribbon every tracer gets: the bolt is a polyline of random kinks drawn
INSIDE the ribbon (uv.y is -1..1 across it), so the ribbon is made wider than a normal
tracer (`width`) to leave room for the zigzag. The kinks are spaced along the ribbon's
SCREEN-space length (see tracers.py's v_ribbon_x), so the zigzag looks the same size from the
muzzle to the impact despite perspective. Kink offsets come from a hash of the segment index
and the shot's seed, so every shot is a different bolt but nothing is stored per shot.

Used by pointing a weapon at it: `tracer_style = "lightning"` (see the Corroder).
"""

from .base import TracerStyle


class Lightning(TracerStyle):
    name = "lightning"
    width = 8.0           # ribbon room for the zigzag; the bolt itself is a thin line inside it
    shrink = 0.0          # constant width while fading, so the zigzag keeps its shape
    lifetime = 0.22       # seconds for the bolt to die out

    fragment = """
    // Tweak the look here. Everything is measured in ribbon half-widths (the unit of uv.y),
    // along the SCREEN-space length of the bolt (v_ribbon_x/v_ribbon_len - see tracers.py), not
    // in metres: perspective squeezes metres hard (half a metre is hundreds of pixels next to
    // the muzzle and a couple at the far end), which made a metre-spaced zigzag start straight
    // and turn into a dense scribble. In screen space the zigzag is the same size all along.
    const vec3  HOT  = vec3(0.92, 0.98, 1.0);    // white-hot core
    const vec3  GLOW = vec3(0.30, 0.65, 1.0);    // electric blue halo
    const float KINK_LENGTH = 0.95;               // between kinks, along the bolt
    const float AMPLITUDE = 0.62;                 // how far a kink strays from the straight line
    const float CORE_WIDTH = 0.085;               // half-width of the bright core
    const float HALO_WIDTH = 0.42;                // how far the glow reaches
    const float PIN_LENGTH = 0.25;                // at each end the bolt is pulled onto the line
    const float JITTER_STEPS = 4.0;               // how many times the shape re-rolls while it fades

    // sin-free hash (sin() with a big argument is imprecise on a lot of GPUs)
    float hash(float n) {
        n = fract(n * 0.1031 + 0.37);
        n *= n + 33.33;
        n *= n + n;
        return fract(n);
    }

    // The k-th kink's sideways offset: pinned to 0 at the muzzle and the impact.
    float kink(float k, float seed, float len) {
        float x = k * KINK_LENGTH;
        float pin = smoothstep(0.0, PIN_LENGTH, min(x, len - x));
        return (hash(k * 1.37 + seed) * 2.0 - 1.0) * AMPLITUDE * pin;
    }

    // Distance from p to segment a-b.
    float seg_dist(vec2 p, vec2 a, vec2 b) {
        vec2 ab = b - a;
        float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
        return length(p - (a + ab * t));
    }

    // Distance from (x, y) to the bolt polyline, looking at the three nearest segments. The
    // four kinks they share are computed once each (a segment's end is the next one's start).
    float bolt_dist(float x, float y, float seed, float len) {
        float i = floor(x / KINK_LENGTH);
        vec2 p = vec2(x, y);
        vec2 k0 = vec2((i - 1.0) * KINK_LENGTH, kink(i - 1.0, seed, len));
        vec2 k1 = vec2(i * KINK_LENGTH,         kink(i,       seed, len));
        vec2 k2 = vec2((i + 1.0) * KINK_LENGTH, kink(i + 1.0, seed, len));
        vec2 k3 = vec2((i + 2.0) * KINK_LENGTH, kink(i + 2.0, seed, len));
        return min(seg_dist(p, k0, k1), min(seg_dist(p, k1, k2), seg_dist(p, k2, k3)));
    }

    vec3 shade(vec2 uv, float fade, float metres, float total, float seed) {
        // Re-roll the shape a few times over the bolt's life (fade falls 1 -> 0) for a crackle.
        float step_id = floor((1.0 - sqrt(max(fade, 0.0))) * JITTER_STEPS);
        float s = seed * 91.0 + step_id * 17.0;

        float d = bolt_dist(v_ribbon_x, v_ribbon_y, s, v_ribbon_len);
        // (1 - smoothstep(0, w, d), not smoothstep(w, 0, d): reversed edges are undefined in GLSL)
        float core = 1.0 - smoothstep(0.0, CORE_WIDTH, d);
        float halo = pow(1.0 - smoothstep(0.0, HALO_WIDTH, d), 2.0);

        // No hard cut at the very start.
        float start = smoothstep(0.0, 0.25, metres);
        vec3 color = mix(GLOW, HOT, core) * (core + halo * 0.6);
        return color * start * fade * 2.4;
    }
    """

    def segment(self, age, distance):
        if age >= self.lifetime:
            return None
        fade = 1.0 - age / self.lifetime
        return 0.0, distance, 0.0, 1.0, fade * fade


STYLE = Lightning()
