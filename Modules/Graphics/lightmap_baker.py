"""
GPU lightmap bake pass.

Renders a static object's geometry using its lightmap UV as clip-space
position (instead of a camera MVP), so the rasterizer fills a small
texture atlas one texel per shader invocation - each texel's fragment
shader invocation receives the interpolated world position/normal for
that point on the mesh, and computes real (diffuse-only) lighting there.

Only POINT lights get baked here. The directional light is deliberately
never baked - it stays fully real-time via CascadedShadowMap for every
object, static or dynamic (see scene.py's _render_shadows). Baking it
here too would double-count it: the runtime shader already adds a
real-time, correctly-shadowed directional term for every object
regardless of whether it has a lightmap, so an object with both a baked
directional contribution AND the real-time one would show it twice, at
roughly double brightness. This module used to support baking the
directional light too (a u_mode switch selecting directional vs point
behavior) - that's been removed along with the double-counting bug it
caused.

Bakes ONE LIGHT AT A TIME, additively blended into the lightmap texture,
rather than declaring uniform arrays sized for every light in the scene
at once. This is deliberate: an earlier version tried to bind a
per-scene-light-count array of shadow-face samplers/matrices into the
bake shader, and the equivalent real-time version of that array was what
caused a GLSL "Constant register limit exceeded" link error once light
counts grew - large mat4/sampler arrays burn through a legacy shader
compiler's fixed register budget fast, scaling with light count. Baking
one light at a time means this shader only ever needs uniform space for
a single point light's worth of data (position/color/radius + one
6-face shadow cube), a small fixed size regardless of how many
"stationary" lights the scene actually has.

Reuses Scene.shadow_program / each object's "shadow_vao" for the
temporary per-light point shadow cubes - same depth-only program
already used for real-time shadows, just fed different light_mvps.
"""

import moderngl

BAKE_VERTEX_SHADER = """
#version 330
uniform mat4 u_model;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_lightmap_uv;

out vec3 v_world_pos;
out vec3 v_normal;

void main() {
    v_world_pos = (u_model * vec4(in_position, 1.0)).xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    gl_Position = vec4(in_lightmap_uv * 2.0 - 1.0, 0.0, 1.0);
}
"""

BAKE_FRAGMENT_SHADER = """
#version 330

uniform vec3 u_light_color;
uniform vec3 u_point_pos;
uniform float u_point_radius;
uniform int u_has_shadow;
uniform sampler2D u_point_shadow_faces[6];
uniform mat4 u_point_light_mvps[6];

in vec3 v_world_pos;
in vec3 v_normal;
out vec4 fragColor;

// Must match PointShadowMap.FACE_DIRECTIONS: 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z
int get_cube_face(vec3 dir) {
    vec3 a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) return dir.x > 0.0 ? 0 : 1;
    if (a.y >= a.x && a.y >= a.z) return dir.y > 0.0 ? 2 : 3;
    return dir.z > 0.0 ? 4 : 5;
}

// Normal-offset bias, not a flat depth-comparison fudge factor: nudges
// the tested point a small real world-space distance off the surface
// along its own normal before doing the shadow lookup, rather than
// tweaking the comparison threshold in NDC space. A flat bias forces a
// trade-off (too small = acne, too large = light bleeding through thin
// geometry, since the fudge eats into the true occluded region) -
// normal offset avoids that trade-off by actually separating the point
// from the surface it's sitting on, which is what was actually causing
// the acne/bleed in the first place. 0.02 is tuned for a small-scale
// scene; widen it if thin walls still leak, narrow it if thick geometry
// shows detachment (shadows floating slightly off their casters).
float calc_point_shadow(vec3 world_pos, vec3 normal) {
    vec3 offset_pos = world_pos + normal * 0.02;
    int face = get_cube_face(offset_pos - u_point_pos);
    vec4 ls = u_point_light_mvps[face] * vec4(offset_pos, 1.0);
    if (ls.w <= 0.00001) return 0.0;
    vec3 c = (ls.xyz / ls.w) * 0.5 + 0.5;
    if (any(lessThan(c, vec3(0.0))) || any(greaterThan(c, vec3(1.0)))) return 0.0;
    return (c.z > texture(u_point_shadow_faces[face], c.xy).r) ? 1.0 : 0.0;
}

void main() {
    vec3 N = normalize(v_normal);

    vec3 to_light = u_point_pos - v_world_pos;
    float dist = length(to_light);
    float NdotL = max(dot(N, to_light / max(dist, 0.0001)), 0.0);

    float radius = max(u_point_radius, 0.01);
    float falloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
    float atten = (falloff * falloff) / (dist * dist + 1.0);

    float shadow = (u_has_shadow == 1) ? calc_point_shadow(v_world_pos, N) : 0.0;

    fragColor = vec4(u_light_color * NdotL * atten * (1.0 - shadow), 1.0);
}
"""


def create_bake_program(ctx):
    return ctx.program(vertex_shader=BAKE_VERTEX_SHADER, fragment_shader=BAKE_FRAGMENT_SHADER)


def create_lightmap(ctx, resolution=256):
    """A fresh, black-cleared lightmap texture, ready for point-light
    passes to be additively blended into.

    Uses a floating-point format (dtype="f2", half-float) with 4
    components (RGBA), not the default 8-bit unsigned normalized RGB.
    Two separate things matter here:

    1. Float, not 8-bit: additive blending (ONE, ONE) into an 8-bit
       target clips each channel to [0, 1] the instant it saturates,
       independently per channel and per light pass. With several
       lights baked into the same texture, that produces exactly the
       kind of blown-out, wrong-hued result you'd expect from clipping
       (e.g. red and blue channels maxing out before green does,
       turning "very bright" into garish magenta) - by the time the
       runtime shader's Reinhard tonemap sees the value, the actual
       brightness information is already gone.

    2. RGBA, not RGB: GL_RGB16F (3-component half-float) is NOT a
       required color-renderable format per the OpenGL spec - only
       GL_RGBA16F, GL_R16F, and GL_RG16F are guaranteed to work as a
       framebuffer color attachment on every implementation. RGB16F
       support is implementation-dependent, and requesting it produced
       a texture whose actual storage didn't match its reported
       size - not a clean error, just corrupted readback. RGBA16F is
       safe everywhere; the alpha channel is simply unused (the bake
       shader already writes vec4, and the runtime sampler already
       reads only .rgb, so no shader changes were needed for this)."""
    texture = ctx.texture((resolution, resolution), 4, dtype="f2")
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo = ctx.framebuffer(color_attachments=[texture])
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    fbo.release()
    return texture


def bake_point_light(ctx, bake_program, obj, model_matrix, light, shadow_map=None):
    """Additively bakes one point light's contribution into
    obj["lightmap_texture"]. shadow_map, if given, is a PointShadowMap
    whose 6 faces already contain rendered static-geometry depth for
    just this light (see Scene.bake_static_lighting for how it's built
    and torn down around this call)."""
    if not obj.get("has_lightmap_uv") or obj.get("lightmap_vao") is None or obj.get("lightmap_texture") is None:
        return

    fbo = ctx.framebuffer(color_attachments=[obj["lightmap_texture"]])
    fbo.use()

    bake_program["u_model"].write(model_matrix.to_bytes())
    bake_program["u_light_color"].value = tuple(c * light["intensity"] for c in light["color"])
    bake_program["u_point_pos"].value = tuple(light["position"])
    bake_program["u_point_radius"].value = light["radius"]

    if shadow_map is not None:
        bake_program["u_has_shadow"].value = 1
        for face in range(6):
            shadow_map.live_textures[face].use(location=face)
        # Whole-array assignment, not per-index bracket names - see
        # pbr_shader.py's docstring for why that distinction matters.
        if "u_point_shadow_faces" in bake_program:
            bake_program["u_point_shadow_faces"].value = tuple(range(6))
        if "u_point_light_mvps" in bake_program:
            bake_program["u_point_light_mvps"].write(b"".join(m.to_bytes() for m in shadow_map.light_mvps))
    else:
        bake_program["u_has_shadow"].value = 0

    obj["lightmap_vao"].render()
    fbo.release()