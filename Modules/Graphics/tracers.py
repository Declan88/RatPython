"""
Bullet tracers, Garry's Mod style: each shot sends a short, bright streak flying
from the muzzle to wherever the bullet ended up, faster than the eye can follow
but slow enough to see. A tracer is just a start point and an end point; the
streak is a camera-facing ribbon (thin in world units, widened a little with
distance so it stays visible far away) drawn additively, so it glows over dark
areas and never darkens anything.

Drawn after the scene (see app.py) with the depth test on and depth writes off,
so walls hide the part of a tracer behind them but a tracer never hides
anything itself. One shared dynamic buffer holds them all, drawn with one call per style
(each look lives in its own file in tracer_styles/).
"""

import random
import time

import glm
import moderngl
import numpy as np

from Modules.Graphics.tracer_styles import get_style

HALF_WIDTH = 0.0035  # base ribbon half-width, as a fraction of the screen height (constant on screen)
MAX_TRACERS = 96
NEAR_MARGIN = 0.15     # metres in front of the camera a streak is trimmed to
_FLOATS_PER_VERTEX = 12   # pos 3, other end 3, uv 2, metres 1, fade 1, total 1, seed 1

# Additive blending (see render()'s own ctx.blend_func) only ADDS a tracer's color on top of
# whatever's already drawn - it never darkens or replaces the background. Far away, the beam's
# own on-screen coverage per pixel is thin enough that this small addition can't outshine a
# bright patch of sky behind it (this project's borealis skybox has teal/green bands, so a dim
# tracer over one of those reads as the SKY's teal showing through, not the tracer's own color
# - confirmed the shader's own two colors, HOT/GLOW in tracer_styles/laser.py, both have low
# blue/green - there's no path to it computing teal itself). Compensated here by boosting the
# additive color with distance from the CAMERA (not along the beam - v_metres/v_total are
# already used for the beam's own end-fade) - capped well short of blowing out a close-up shot.
DISTANCE_BRIGHTNESS_BOOST = 0.02   # extra multiplier per metre from the camera
DISTANCE_BRIGHTNESS_MAX = 4.0      # cap - a shot hundreds of metres away shouldn't flare out

_VERT = """
#version 330
uniform mat4 u_view_proj;
uniform vec2 u_half_px;      // ribbon half-width in NDC units (x, y)
uniform vec2 u_aspect;       // (aspect, 1): puts NDC direction in square units
uniform float u_width;       // the style's width multiplier
uniform float u_shrink;      // how much thinner it gets as it fades
in vec3 in_pos;
in vec3 in_other;            // the streak's other end, to know which way it runs on screen
in vec2 in_uv;               // x: 0 at the tail -> 1 at the head, y: -1..1 across
in float in_metres;          // distance along the path from the muzzle
in float in_fade;            // intensity 0..1 (a fading tracer also gets thinner)
in float in_total;           // the whole path's length in metres
in float in_seed;            // random per shot
out vec2 v_uv;
out float v_fade;
out float v_metres;
out float v_total;
out float v_seed;
out vec3 v_world_pos;
void main() {
    v_uv = in_uv;
    v_fade = in_fade;
    v_metres = in_metres;
    v_total = in_total;
    v_seed = in_seed;
    v_world_pos = in_pos;
    vec4 c = u_view_proj * vec4(in_pos, 1.0);
    vec4 o = u_view_proj * vec4(in_other, 1.0);
    vec2 dir = (o.xy / o.w - c.xy / c.w) * u_aspect;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 normal = vec2(-dir.y, dir.x) / u_aspect;
    // Expanded in screen space so the streak is the same thickness along its
    // whole length however close to the camera it starts.
    float width = u_width * mix(1.0 - u_shrink, 1.0, in_fade);
    gl_Position = c + vec4(normal * in_uv.y * width * u_half_px * c.w, 0.0, 0.0);
}
"""

_FRAG_HEAD = """
#version 330
in vec2 v_uv;
in float v_fade;
in float v_metres;
in float v_total;
in float v_seed;
in vec3 v_world_pos;
uniform vec3 u_cam_pos;
uniform float u_dist_boost;
uniform float u_dist_boost_max;
out vec4 fragColor;
"""

_FRAG_TAIL = """
void main() {
    float boost = min(1.0 + distance(v_world_pos, u_cam_pos) * u_dist_boost, u_dist_boost_max);
    fragColor = vec4(shade(v_uv, v_fade, v_metres, v_total, v_seed) * boost, 1.0);
}
"""


class _Tracer:
    __slots__ = ("start", "direction", "distance", "born", "style", "seed")

    def __init__(self, start, end, born, style):
        offset = end - start
        self.distance = glm.length(offset)
        self.direction = offset / self.distance
        self.start = start
        self.born = born
        self.style = style
        self.seed = random.random()


class Tracers:
    def __init__(self, ctx):
        self.ctx = ctx
        self.buffer = ctx.buffer(reserve=MAX_TRACERS * 6 * _FLOATS_PER_VERTEX * 4, dynamic=True)
        self._programs = {}    # style name -> (program, vao), built the first time it's drawn
        self.active = []

    def _program_for(self, style):
        entry = self._programs.get(style.name)
        if entry is None:
            program = self.ctx.program(
                vertex_shader=_VERT, fragment_shader=_FRAG_HEAD + style.fragment + _FRAG_TAIL)
            # A style that doesn't use an input (e.g. no noise -> no seed) has it
            # optimised out of the program, so that slot is skipped as padding.
            layout = (("in_pos", "3f"), ("in_other", "3f"), ("in_uv", "2f"), ("in_metres", "1f"),
                      ("in_fade", "1f"), ("in_total", "1f"), ("in_seed", "1f"))
            formats = [fmt if name in program else f"{fmt[0]}x4" for name, fmt in layout]
            names = [name for name, _ in layout if name in program]
            vao = self.ctx.vertex_array(program, [(self.buffer, " ".join(formats), *names)])
            entry = self._programs[style.name] = (program, vao)
        return entry

    def add(self, start, end, now=None, style=None):
        """Starts a tracer from `start` to `end` (world-space points), in the
        style called `style` (see Modules/Graphics/tracer_styles; None or an
        unknown name = the default)."""
        start, end = glm.vec3(start), glm.vec3(end)
        if glm.length(end - start) < 0.5:
            return
        if len(self.active) >= MAX_TRACERS:
            self.active.pop(0)
        self.active.append(_Tracer(start, end, time.perf_counter() if now is None else now, get_style(style)))

    def clear(self):
        self.active.clear()

    def render(self, camera, now=None):
        if not self.active:
            return
        now = time.perf_counter() if now is None else now
        cam_pos = glm.vec3(camera.position)
        forward = glm.vec3(camera.front)
        by_style = {}          # style name -> (style, [vertex floats])
        alive = []
        for t in self.active:
            segment = t.style.segment(now - t.born, t.distance)
            if segment is None:
                continue
            alive.append(t)
            a, b, u0, u1, fade = segment
            if b <= a:
                continue
            # Keep the ribbon in front of the camera (a projected point behind
            # it would flip): trim the part closer than the near margin.
            f0 = glm.dot(t.start + t.direction * a - cam_pos, forward)
            f1 = glm.dot(t.start + t.direction * b - cam_pos, forward)
            if f1 <= NEAR_MARGIN:
                continue
            if f0 < NEAR_MARGIN:
                k = (NEAR_MARGIN - f0) / (f1 - f0)
                a, u0 = a + (b - a) * k, u0 + (u1 - u0) * k
            p0 = t.start + t.direction * a
            p1 = t.start + t.direction * b
            # (this end, the other end, u, metres, side) for the four corners.
            corners = ((p0, p1, u0, a, -1.0), (p0, p1, u0, a, 1.0),
                       (p1, p0, u1, b, 1.0), (p1, p0, u1, b, -1.0))
            verts = by_style.setdefault(t.style.name, (t.style, []))[1]
            for i in (0, 1, 2, 0, 2, 3):
                p, o, u, m, v = corners[i]
                verts.extend((p.x, p.y, p.z, o.x, o.y, o.z, u, v, m, fade, t.distance, t.seed))
        self.active = alive
        if not by_style:
            return

        data = []
        for _style, verts in by_style.values():
            data.extend(verts)
        self.buffer.orphan()
        self.buffer.write(np.array(data, dtype="f4").tobytes())
        half = HALF_WIDTH * 2.0   # NDC spans 2 units over the screen height
        view_proj = (camera.get_projection_matrix() * camera.get_view_matrix()).to_bytes()

        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.fbo.depth_mask = False
        first = 0
        for style, verts in by_style.values():
            program, vao = self._program_for(style)
            program["u_half_px"].value = (half, half)
            program["u_aspect"].value = (camera.aspect, 1.0)
            program["u_width"].value = style.width
            program["u_shrink"].value = style.shrink
            program["u_view_proj"].write(view_proj)
            program["u_cam_pos"].value = tuple(cam_pos)
            program["u_dist_boost"].value = DISTANCE_BRIGHTNESS_BOOST
            program["u_dist_boost_max"].value = DISTANCE_BRIGHTNESS_MAX
            count = len(verts) // _FLOATS_PER_VERTEX
            vao.render(moderngl.TRIANGLES, vertices=count, first=first)
            first += count
        ctx.fbo.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = "back"

    def destroy(self):
        for program, vao in self._programs.values():
            vao.release()
            program.release()
        self._programs.clear()
        self.buffer.release()
