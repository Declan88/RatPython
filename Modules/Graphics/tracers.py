"""
Bullet tracers, Garry's Mod style: each shot sends a short, bright streak flying
from the muzzle to wherever the bullet ended up, faster than the eye can follow
but slow enough to see. A tracer is just a start point and an end point; the
streak is a camera-facing ribbon (thin in world units, widened a little with
distance so it stays visible far away) drawn additively, so it glows over dark
areas and never darkens anything.

Drawn after the scene (see app.py) with the depth test on and depth writes off,
so walls hide the part of a tracer behind them but a tracer never hides
anything itself. One shared dynamic buffer draws them all in a single call.
"""

import time

import glm
import moderngl
import numpy as np

SPEED = 320.0          # metres per second the streak's head travels
LENGTH = 9.0           # metres from head to tail
HALF_WIDTH = 0.0035  # ribbon half-width, as a fraction of the screen height (constant on screen)
MAX_TRACERS = 96
NEAR_MARGIN = 0.15     # metres in front of the camera a streak is trimmed to

_VERT = """
#version 330
uniform mat4 u_view_proj;
uniform vec2 u_half_px;      // ribbon half-width in NDC units (x, y)
uniform vec2 u_aspect;       // (aspect, 1): puts NDC direction in square units
in vec3 in_pos;
in vec3 in_other;            // the streak's other end, to know which way it runs on screen
in vec2 in_uv;               // x: 0 at the tail -> 1 at the head, y: -1..1 across
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    vec4 c = u_view_proj * vec4(in_pos, 1.0);
    vec4 o = u_view_proj * vec4(in_other, 1.0);
    vec2 dir = (o.xy / o.w - c.xy / c.w) * u_aspect;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 normal = vec2(-dir.y, dir.x) / u_aspect;
    // Expanded in screen space so the streak is the same thickness along its
    // whole length however close to the camera it starts.
    gl_Position = c + vec4(normal * in_uv.y * u_half_px * c.w, 0.0, 0.0);
}
"""

# u: 0 at the streak's tail -> 1 at its head; v: -1..1 across the ribbon.
_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
void main() {
    float across = 1.0 - abs(v_uv.y);
    float core = across * across;
    float halo = across * 0.35;
    float along = clamp(v_uv.x, 0.0, 1.0);
    float body = 0.15 + 0.85 * along * along;        // bright head, fading tail
    float nose = 1.0 - smoothstep(0.93, 1.0, along); // soft tip
    vec3 hot = vec3(1.0, 0.96, 0.82);
    vec3 warm = vec3(1.0, 0.55, 0.15);
    vec3 colour = mix(warm, hot, core) * (core + halo) * body * nose;
    fragColor = vec4(colour * 1.6, 1.0);
}
"""


class _Tracer:
    __slots__ = ("start", "direction", "distance", "born")

    def __init__(self, start, end, born):
        offset = end - start
        self.distance = glm.length(offset)
        self.direction = offset / self.distance
        self.start = start
        self.born = born


class Tracers:
    def __init__(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.buffer = ctx.buffer(reserve=MAX_TRACERS * 6 * 8 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "3f 3f 2f", "in_pos", "in_other", "in_uv")])
        self.active = []

    def add(self, start, end, now=None):
        """Starts a tracer from `start` to `end` (world-space points)."""
        start, end = glm.vec3(start), glm.vec3(end)
        if glm.length(end - start) < 0.5:
            return
        if len(self.active) >= MAX_TRACERS:
            self.active.pop(0)
        self.active.append(_Tracer(start, end, time.perf_counter() if now is None else now))

    def clear(self):
        self.active.clear()

    def render(self, camera, now=None):
        if not self.active:
            return
        now = time.perf_counter() if now is None else now
        cam_pos = glm.vec3(camera.position)
        forward = glm.vec3(camera.front)
        verts = []
        alive = []
        for t in self.active:
            head = SPEED * (now - t.born + 1.0 / 60.0)
            tail = head - LENGTH
            if tail >= t.distance:
                continue
            alive.append(t)
            a, b = max(tail, 0.0), min(head, t.distance)
            if b <= a:
                continue
            # Keep the ribbon in front of the camera (a projected point behind
            # it would flip): trim the part closer than the near margin.
            f0 = glm.dot(t.start + t.direction * a - cam_pos, forward)
            f1 = glm.dot(t.start + t.direction * b - cam_pos, forward)
            if f1 <= NEAR_MARGIN:
                continue
            if f0 < NEAR_MARGIN:
                a += (b - a) * (NEAR_MARGIN - f0) / (f1 - f0)
            p0 = t.start + t.direction * a
            p1 = t.start + t.direction * b
            u0, u1 = (a - tail) / LENGTH, (b - tail) / LENGTH
            # (this end, the other end, u, side) for the four corners.
            corners = ((p0, p1, u0, -1.0), (p0, p1, u0, 1.0),
                       (p1, p0, u1, 1.0), (p1, p0, u1, -1.0))
            for i in (0, 1, 2, 0, 2, 3):
                p, o, u, v = corners[i]
                verts.extend((p.x, p.y, p.z, o.x, o.y, o.z, u, v))
        self.active = alive
        if not verts:
            return

        self.buffer.write(np.array(verts, dtype="f4").tobytes())
        half = HALF_WIDTH * 2.0   # NDC spans 2 units over the screen height
        self.program["u_half_px"].value = (half, half)
        self.program["u_aspect"].value = (camera.aspect, 1.0)
        self.program["u_view_proj"].write(
            (camera.get_projection_matrix() * camera.get_view_matrix()).to_bytes())

        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.screen.depth_mask = False
        self.vao.render(moderngl.TRIANGLES, vertices=len(verts) // 8)
        ctx.screen.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = "back"

    def destroy(self):
        self.vao.release()
        self.buffer.release()
        self.program.release()
