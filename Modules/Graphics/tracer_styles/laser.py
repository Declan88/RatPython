"""
A laser: one static beam from the muzzle to the impact with a noise texture along
and across it, faded out at both ends, that fades away quickly.
"""

from .base import TracerStyle


class Laser(TracerStyle):
    name = "laser"
    width = 1.7
    shrink = 0.0          # constant width the whole way, even while fading
    lifetime = 0.45       # seconds for the beam to fade out completely

    fragment = """
    // Tweak the look here.
    const vec3  HOT  = vec3(1.0, 0.97, 0.72);   // bright core
    const vec3  GLOW = vec3(1.0, 0.72, 0.10);   // outer glow
    const float END_FADE = 0.4;                  // metres faded in at each end (kept tiny: just no hard cut)
    const float NOISE_SCALE = 1.4;               // noise cells per metre along the beam
    const float NOISE_ACROSS = 3.0;              // noise cells across the beam

    float hash(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
    }
    float vnoise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
                   mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
    }

    vec3 shade(vec2 uv, float fade, float metres, float total, float seed) {
        float across = 1.0 - abs(uv.y);
        // Faded near the beginning and the end (in metres, so a long beam isn't
        // mostly fade).
        float end = min(metres, total - metres);
        float ends = smoothstep(0.0, END_FADE, end);
        vec2 p = vec2(metres * NOISE_SCALE, uv.y * NOISE_ACROSS) + seed * 37.0;
        float n = 0.6 * vnoise(p) + 0.4 * vnoise(p * 2.7 + 11.0);
        // The cross-section shape depends only on `across`, so the beam is the
        // same width everywhere; the noise only varies its brightness.
        float core = pow(across, 3.0);
        float halo = across * 0.5;
        float flicker = mix(0.5, 1.0, n);
        return mix(GLOW, HOT, core) * (core + halo) * flicker * ends * fade * 2.2;
    }
    """

    def segment(self, age, distance):
        if age >= self.lifetime:
            return None
        fade = 1.0 - age / self.lifetime
        return 0.0, distance, 0.0, 1.0, fade * fade


STYLE = Laser()
