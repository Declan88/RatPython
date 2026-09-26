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
BASE_WIDTH = 0.03      # ribbon half-width at the camera, metres
WIDTH_PER_METRE = 0.004  # extra half-width per metre of distance from the camera
MAX_WIDTH = 0.35
MAX_TRACERS = 96

_VERT = """
#version 330
uniform mat4 u_view_proj;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = u_view_proj * vec4(in_pos, 1.0);
}
"""

# u: 0 at the streak's tail -> 1 at its head; v: -1..1 across the ribbon.
_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
void main() {
    float across = 1.0 - abs(v_uv.y);
    float core = across * across * across;
    float halo = across * across * 0.3;
    float along = clamp(v_uv.x, 0.0, 1.0);
    float body = 0.1 + 0.9 * along * along;          // bright head, fading tail
    float nose = 1.0 - smoothstep(0.93, 1.0, along); // soft round-ish tip
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
        self.buffer = ctx.buffer(reserve=MAX_TRACERS * 6 * 5 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "3f 2f", "in_pos", "in_uv")])
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
            p0 = t.start + t.direction * a
            p1 = t.start + t.direction * b
            mid = (p0 + p1) * 0.5
            to_cam = cam_pos - mid
            side = glm.cross(t.direction, to_cam)
            if glm.length(side) < 1e-6:     # looking straight down the streak: it's a dot
                continue
            width = min(MAX_WIDTH, BASE_WIDTH + WIDTH_PER_METRE * glm.length(to_cam))
            side = glm.normalize(side) * width
            u0, u1 = (a - tail) / LENGTH, (b - tail) / LENGTH
            corners = ((p0 - side, u0, -1.0), (p0 + side, u0, 1.0),
                       (p1 + side, u1, 1.0), (p1 - side, u1, -1.0))
            for i in (0, 1, 2, 0, 2, 3):
                p, u, v = corners[i]
                verts.extend((p.x, p.y, p.z, u, v))
        self.active = alive
        if not verts:
            return

        self.buffer.write(np.array(verts, dtype="f4").tobytes())
        self.program["u_view_proj"].write(
            (camera.get_projection_matrix() * camera.get_view_matrix()).to_bytes())

        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.screen.depth_mask = False
        self.vao.render(moderngl.TRIANGLES, vertices=len(verts) // 5)
        ctx.screen.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = "back"

    def destroy(self):
        self.vao.release()
        self.buffer.release()
        self.program.release()
