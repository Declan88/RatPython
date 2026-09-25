"""
GPU lightmap bake pass.

Renders a static object's geometry using its lightmap UV as clip-space
position (instead of a camera MVP), so the rasterizer fills a small
texture atlas one texel per shader invocation - each texel's fragment
shader invocation receives the interpolated world position/normal for
that point on the mesh, and computes real (diffuse-only) lighting there.

Both point lights AND the directional (sun) light get baked here now -
u_mode switches the single bake fragment shader between the two
behaviors, since a lightmapped object's bake VAO is built against THIS
program specifically (see Scene._load_object) and can't be reused
against a second, separately-compiled program.

To avoid double-counting the sun (baking it here AND still adding a
real-time directional term at runtime would show it twice, at roughly
double brightness), pbr_shader.py's runtime fragment shader now skips
its own real-time directional calculation entirely for any object that
has a lightmap (u_has_lightmap == 1) - see that file's main(). Only
objects WITHOUT a lightmap (dynamic/skeletal - moving objects, which
can't be baked since their transform changes every frame) still
compute the sun's contribution live, and still cast/receive real-time
shadows via CascadedShadowMap exactly as before. Static geometry still
RENDERS into that same real-time cascade pass too (see Scene.
_render_shadows) - not for its own lighting anymore, but so it still
correctly occludes the sun for a moving object standing behind/under
it (matching Unreal's stationary-light model: static shadowing is
baked, but movable actors still get real-time shadows cast onto AND by
static geometry).

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
import numpy as np

from Modules.Graphics.pbr_shader import _has_uniform

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

// 0 = point light (default - see bake_point_light), 1 = directional
// (sun) light (see bake_directional_light). One shader/program handles
// both because a lightmapped object's "lightmap_vao" is built once,
// against THIS specific compiled program (see Scene._load_object) -
// baking the sun through a second, separate program would need a
// second VAO per object just for that, for no real benefit.
uniform int u_mode;

uniform vec3 u_light_color;

// u_mode == 0 (point light) uniforms.
uniform vec3 u_point_pos;
uniform float u_point_radius;
uniform int u_has_shadow;
uniform sampler2D u_point_shadow_faces[6];
uniform mat4 u_point_light_mvps[6];
uniform float u_point_shadow_texel;

// u_mode == 1 (directional/sun) uniforms. Unlike the point-light path,
// there's no falloff/attenuation (a directional light has no position,
// every point in the scene receives the same intensity) and always a
// single flat shadow map, never a 6-face cube - see Scene.
// bake_static_lighting for how that shadow map is built (one static-
// scene-covering orthographic depth render, done once for the whole
// bake, not per object).
uniform vec3 u_light_dir;
uniform sampler2D u_directional_shadow_map;
uniform mat4 u_directional_light_mvp;
uniform float u_directional_shadow_texel;

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
//
// 3x3 PCF (percentage-closer filtering): a single hard binary shadow
// test forces every edge into a jagged staircase along shadow-map texel
// boundaries. Averaging 9 neighboring samples turns that hard cutoff
// into a smooth gradient instead - same technique pbr_shader.py's
// directional calculate_shadow already uses. u_point_shadow_texel is
// passed in from Python (1.0 / actual shadow map resolution) rather
// than hardcoded, so raising the shadow map's resolution can't silently
// leave the PCF step size mismatched with it.
float calc_point_shadow(vec3 world_pos, vec3 normal) {
    vec3 offset_pos = world_pos + normal * 0.02;
    int face = get_cube_face(offset_pos - u_point_pos);
    vec4 ls = u_point_light_mvps[face] * vec4(offset_pos, 1.0);
    if (ls.w <= 0.00001) return 0.0;
    vec3 c = (ls.xyz / ls.w) * 0.5 + 0.5;
    if (any(lessThan(c, vec3(0.0))) || any(greaterThan(c, vec3(1.0)))) return 0.0;

    float shadow = 0.0;
    vec2 texel_size = vec2(u_point_shadow_texel);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 uv = clamp(c.xy + vec2(x, y) * texel_size, vec2(0.001), vec2(0.999));
            if (c.z > texture(u_point_shadow_faces[face], uv).r) {
                shadow += 1.0;
            }
        }
    }
    return shadow / 9.0;
}

// Same normal-offset-bias + 3x3 PCF approach as calc_point_shadow above
// (see that function's own comment) and as pbr_shader.py's real-time
// calculate_shadow - just against a single flat orthographic depth map
// instead of a cube face or a cascade array. 0.05 offset rather than
// calc_point_shadow's 0.02: this shadow map covers the WHOLE static
// scene in one orthographic projection (see Scene.bake_static_lighting),
// so its world-space texel size is coarser than a tightly-fit per-light
// point shadow cube, and needs a proportionally bigger offset to stay
// ahead of that.
float calc_directional_shadow(vec3 world_pos, vec3 normal) {
    vec3 offset_pos = world_pos + normal * 0.05;
    vec4 ls = u_directional_light_mvp * vec4(offset_pos, 1.0);
    if (ls.w <= 0.00001) return 0.0;
    vec3 c = (ls.xyz / ls.w) * 0.5 + 0.5;
    if (any(lessThan(c, vec3(0.0))) || any(greaterThan(c, vec3(1.0)))) return 0.0;

    float shadow = 0.0;
    vec2 texel_size = vec2(u_directional_shadow_texel);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 uv = clamp(c.xy + vec2(x, y) * texel_size, vec2(0.001), vec2(0.999));
            if (c.z > texture(u_directional_shadow_map, uv).r) {
                shadow += 1.0;
            }
        }
    }
    return shadow / 9.0;
}

void main() {
    vec3 N = normalize(v_normal);

    if (u_mode == 1) {
        // Diffuse only, same as the point-light path below - baked
        // lighting here never carries a specular term (a lightmap
        // stores irradiance, not a view-dependent highlight; see this
        // file's own module docstring). The real-time shader still
        // adds the sun's specular highlight for objects WITHOUT a
        // lightmap (moving objects - see pbr_shader.py's main()), just
        // never a baked one.
        vec3 L = normalize(u_light_dir);
        float NdotL = max(dot(N, L), 0.0);
        float shadow = calc_directional_shadow(v_world_pos, N);
        fragColor = vec4(u_light_color * NdotL * (1.0 - shadow), 1.0);
        return;
    }

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


# alpha=0 (create_lightmap's own clear value) marks a texel no light
# pass has ever rasterized over at all - both bake_point_light's and
# bake_directional_light's fragment shaders write alpha=1.0
# unconditionally wherever a triangle DOES cover a texel (see their own
# fragColor lines), regardless of how dark that texel's actual lit
# color comes out - so alpha here is a pure coverage flag, never
# confusable with "covered but black". Every eligible object gets AT
# LEAST the sun baked into it (see Scene.bake_static_lighting's own
# docstring), so this is always a reliable signal even for an object
# with zero point lights ever reaching it.
DILATE_VERTEX_SHADER = """
#version 330
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

DILATE_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_source;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 center = texture(u_source, v_uv);
    if (center.a > 0.0) {
        fragColor = center;
        return;
    }
    // Grow the valid region outward by one texel: average whichever of
    // the 8 neighbors are themselves already valid (alpha > 0). Several
    // passes (see dilate_lightmap's own iterations) push this outward
    // one ring at a time, same as any standard texture-margin dilation
    // ("push-pull") technique.
    vec3 sum = vec3(0.0);
    float count = 0.0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            vec4 s = texture(u_source, v_uv + vec2(float(dx), float(dy)) * u_texel_size);
            if (s.a > 0.0) {
                sum += s.rgb;
                count += 1.0;
            }
        }
    }
    fragColor = count > 0.0 ? vec4(sum / count, 1.0) : vec4(0.0);
}
"""


def create_dilate_program(ctx):
    return ctx.program(vertex_shader=DILATE_VERTEX_SHADER, fragment_shader=DILATE_FRAGMENT_SHADER)


def create_dilate_quad_vao(ctx, dilate_program):
    """A single fullscreen triangle-strip quad, reused for every object's
    dilate_lightmap call this Scene ever makes - the geometry never
    changes, only which lightmap texture is bound as u_source."""
    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(dilate_program, [(vbo, "2f", "in_position")])
    return vao, vbo


def dilate_lightmap(ctx, dilate_program, quad_vao, texture, iterations=6):
    """Grows each chart's own baked lighting outward by `iterations`
    texels into the empty padding gap generate_lightmap_uvs left around
    it (see that function's own padding_texels docstring - it explicitly
    expects a dilation step like this one to exist, sized to match).

    iterations MUST be sized against the REAL resolution `texture` was
    actually baked at, not generate_lightmap_uvs' own internal
    DEFAULT_RESOLUTION assumption (256) - that function reserves a
    margin that's a FIXED FRACTION of UV space, computed without ever
    knowing what resolution this texture would really end up being
    baked at (that's decided later, independently - see Scene.
    bake_static_lighting), so the real, actual TEXEL width of that
    margin scales up proportionally with however much higher the real
    resolution is than DEFAULT_RESOLUTION. The default of 6 here only
    ever matches DEFAULT_RESOLUTION itself (256) - Scene.bake_static_
    lighting's own call site computes a real value scaled to whatever
    lightmap_resolution it's actually baking at instead of relying on
    this default, which is kept only for any other/future caller that
    doesn't. Getting this wrong doesn't error or crash - it silently
    leaves most of the padding gap as raw, never-written clear color
    instead of a safe color extension, which is exactly what was
    confirmed to cause visible bleeding/seams between unrelated charts
    packed into the same lightmap (this project bakes at 2048-4096,
    8-16x DEFAULT_RESOLUTION, while this default of 6 alone would only
    ever ACTUALLY cover a 256-resolution bake, leaving most of the real
    gap at those higher resolutions completely unfilled).

    Without this, the padding gap is still just whatever create_lightmap
    cleared it to (transparent black) - bilinear sampling at a chart's
    OWN edge (not just between two DIFFERENT charts, which the padding
    gap alone already keeps apart) blends toward that black, reading as
    a dark seam/border around every lightmapped surface. Dilating first
    fills the gap with a plausible extension of the chart's own nearby
    color, so a sample landing anywhere in the (now no longer really
    "empty") gap blends between real lit colors instead of black.

    Ping-pongs between `texture` and a same-sized scratch texture
    (NEAREST-filtered for the duration - bilinear sampling DURING
    dilation would reintroduce the exact bleeding this is meant to fix,
    by blending toward not-yet-dilated black neighbors mid-pass) and
    copies the final result back into `texture` itself if it landed in
    the scratch texture instead, so the caller's own object dict entry
    (obj["lightmap_texture"]) never needs to change identity. Restores
    `texture`'s normal LINEAR filtering before returning either way."""
    resolution = texture.size
    scratch = ctx.texture(resolution, 4, dtype="f2")
    scratch.filter = (moderngl.NEAREST, moderngl.NEAREST)
    original_filter = texture.filter
    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    texel_size = (1.0 / resolution[0], 1.0 / resolution[1])
    dilate_program["u_texel_size"].value = texel_size

    src, dst = texture, scratch
    for _ in range(iterations):
        fbo = ctx.framebuffer(color_attachments=[dst])
        fbo.use()
        src.use(location=0)
        dilate_program["u_source"].value = 0
        quad_vao.render(moderngl.TRIANGLE_STRIP)
        fbo.release()
        src, dst = dst, src

    if src is not texture:
        # Landed in the scratch texture after an odd number of
        # iterations - blit it back into the object's own texture
        # object rather than handing the caller a different one.
        src_fbo = ctx.framebuffer(color_attachments=[src])
        dst_fbo = ctx.framebuffer(color_attachments=[texture])
        ctx.copy_framebuffer(dst_fbo, src_fbo)
        src_fbo.release()
        dst_fbo.release()

    scratch.release()
    texture.filter = original_filter


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

    bake_program["u_mode"].value = 0
    bake_program["u_model"].write(model_matrix.to_bytes())
    bake_program["u_light_color"].value = tuple(c * light["intensity"] for c in light["color"])
    bake_program["u_point_pos"].value = tuple(light["position"])
    bake_program["u_point_radius"].value = light["radius"]

    if shadow_map is not None:
        bake_program["u_has_shadow"].value = 1
        bake_program["u_point_shadow_texel"].value = 1.0 / shadow_map.resolution
        for face in range(6):
            shadow_map.live_textures[face].use(location=face)
        # Whole-array assignment, not per-index bracket names - see
        # pbr_shader.py's docstring for why that distinction matters.
        if _has_uniform(bake_program, "u_point_shadow_faces"):
            bake_program["u_point_shadow_faces"].value = tuple(range(6))
        if _has_uniform(bake_program, "u_point_light_mvps"):
            bake_program["u_point_light_mvps"].write(b"".join(m.to_bytes() for m in shadow_map.light_mvps))
    else:
        bake_program["u_has_shadow"].value = 0

    obj["lightmap_vao"].render()
    fbo.release()


def bake_directional_light(ctx, bake_program, obj, model_matrix, light_dir, light_color,
                            light_intensity, shadow_texture, shadow_light_mvp, shadow_resolution):
    """Additively bakes the sun's contribution into obj["lightmap_
    texture"] - the directional counterpart to bake_point_light above,
    called once per eligible static object from Scene.bake_static_
    lighting for the SAME single static-scene shadow map (built once
    for the whole bake, not per object - a directional light has no
    position for a per-object shadow cube to make sense of anyway).

    shadow_texture: a single depth texture (not a 6-face cube - see
    calc_directional_shadow's own comment) already containing every
    static object's depth as seen from the light.
    shadow_light_mvp: the glm.mat4 that produced it.
    shadow_resolution: shadow_texture's resolution, for sizing the PCF
    kernel's texel step (see calc_point_shadow's own u_point_shadow_
    texel for why this is passed in rather than hardcoded)."""
    if not obj.get("has_lightmap_uv") or obj.get("lightmap_vao") is None or obj.get("lightmap_texture") is None:
        return

    fbo = ctx.framebuffer(color_attachments=[obj["lightmap_texture"]])
    fbo.use()

    bake_program["u_mode"].value = 1
    bake_program["u_model"].write(model_matrix.to_bytes())
    bake_program["u_light_color"].value = tuple(c * light_intensity for c in light_color)
    bake_program["u_light_dir"].value = tuple(light_dir)

    shadow_texture.use(location=0)
    if _has_uniform(bake_program, "u_directional_shadow_map"):
        bake_program["u_directional_shadow_map"].value = 0
    if _has_uniform(bake_program, "u_directional_light_mvp"):
        bake_program["u_directional_light_mvp"].write(shadow_light_mvp.to_bytes())
    bake_program["u_directional_shadow_texel"].value = 1.0 / shadow_resolution

    obj["lightmap_vao"].render()
    fbo.release()