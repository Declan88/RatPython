"""
Source-style particle effects, driven by .pcf files (see pcf.py).

Efficiency: every live effect is simulated with numpy (a handful of vector ops
per operator per frame, whatever the particle count) and ALL particles of all
effects are drawn as instanced camera-facing quads: one dynamic buffer write and
one draw call per material per frame. Effects with no particles cost nothing
and are dropped; nothing is allocated per particle.

    particles = ParticleManager(ctx)
    particles.load("Assets/Particles/Muzzle/muzzleflashes.pcf")
    particles.spawn("muzzle_pistols", position, forward=..., up=...)   # fire and forget
    ...each frame:  particles.update(dt);  particles.render(camera)    # after the scene

Units: Source works in inches with Z up. Distances, speeds and gravity are
scaled by `unit_scale` (metres per Source unit) and a Source vector maps to the
world with X->X, Y->-Z, Z->Y. A spawn's forward/up give the effect's control
point 0 frame, which "local space" options and local speeds use (Source's
X = forward, Y = left, Z = up).

Materials: a system's material ("particle\\muzzleflash\\noisecloud1.vmt") is
looked up as an image (png/tga/jpg/...) with the same name under the manager's
texture folders, and as a .vmt beside it for its $basetexture / $additive /
$translucent. With no image at all a soft round sprite is used, so an effect is
never invisible just for a missing file. Without a .vmt, a sprite is additive
if its image has no transparency (or "additive" is in the name) and alpha
blended otherwise.

Sprite sheets: Source picks a frame ("sequence") per particle and packs the
frames into one image at rectangles listed in the .vtf. If the .vtf is beside
the image (or found by the material's name), its table is used as is - sequence
N is that table's N-th entry, animated through its frames by the renderer's
animation rate. Otherwise a grid can be given with a "<image name>.sheet" text
file - "columns rows [frames_per_sequence]" - or $sheet_columns / $sheet_rows /
$sheet_frames_per_sequence in the .vmt; no sheet at all = one frame.

Sprite trails (sparks, debris) are stretched along the particle's velocity.

Collision ("Collision via traces") needs a line trace: assign
ParticleManager.raycast = callable(from_vec3, to_vec3) -> RayHit or None.

Supported functions are listed in _EMITTERS/_INITIALIZERS/_OPERATORS below; any
other function in a file is skipped (with one warning) rather than failing.
"""

import os
import re

import glm
import moderngl
import numpy as np
import pygame

from .pcf import read_pcf
from .vtf_sheet import read_vtf_sheet

# Source units -> metres, and Source (x, y, z-up) -> this engine (x, y-up, -z).
UNIT_SCALE = 0.0254
_SOURCE_TO_WORLD = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, -1.0, 0.0]], dtype="f4")

_IMAGE_EXTENSIONS = (".png", ".tga", ".jpg", ".jpeg", ".bmp")
_FLOATS_PER_PARTICLE = 14    # position 3, radius 1, rotation 1, rgba 4, velocity 3, trail length 1, sheet cell 1
MAX_TOTAL_PARTICLES = 16384
_MAX_EFFECT_PARTICLES = 4096

_VERT = """
#version 330
uniform mat4 u_view_proj;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_cam_pos;
uniform sampler2D u_rects;    // per sheet frame: left, top, right, bottom (0-1, from the image's top left)
in vec2 in_corner;          // -1..1
in vec3 in_center;
in float in_radius;
in float in_rotation;       // radians
in vec4 in_color;
in vec3 in_vel;
in float in_trail;          // > 0: a trail sprite this many metres long
in float in_cell;           // sprite sheet cell
out vec2 v_uv;
out vec4 v_color;
void main() {
    vec2 uv = in_corner * 0.5 + 0.5;
    vec3 world;
    if (in_trail > 0.0) {
        // A streak from the particle back along its velocity, camera-facing across.
        float speed = length(in_vel);
        vec3 d = speed > 1e-6 ? in_vel / speed : vec3(0.0, 1.0, 0.0);
        vec3 side = cross(d, u_cam_pos - in_center);
        float sl = length(side);
        side = sl > 1e-6 ? side / sl : u_right;
        world = in_center - d * in_trail * (1.0 - uv.x) + side * in_corner.y * in_radius;
        // Trail textures run along their height: the bright head at the image's bottom.
        uv = vec2(in_corner.y * 0.5 + 0.5, 1.0 - uv.x);
    } else {
        float c = cos(in_rotation), s = sin(in_rotation);
        vec2 p = vec2(c * in_corner.x - s * in_corner.y, s * in_corner.x + c * in_corner.y);
        world = in_center + (u_right * p.x + u_up * p.y) * in_radius;
    }
    vec4 r = texelFetch(u_rects, ivec2(int(in_cell + 0.5), 0), 0);
    v_uv = vec2(mix(r.x, r.z, uv.x), 1.0 - mix(r.w, r.y, uv.y));
    v_color = in_color;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
"""

_FRAG = """
#version 330
uniform sampler2D u_texture;
in vec2 v_uv;
in vec4 v_color;
out vec4 fragColor;
void main() {
    // The texture is premultiplied by its alpha (see ParticleManager._material), so
    // additive sprites (blended ONE, ONE) get shaped by it too, and blended ones
    // (ONE, ONE_MINUS_SRC_ALPHA) don't pick up colour from their transparent texels.
    vec4 t = texture(u_texture, v_uv);
    fragColor = vec4(t.rgb * v_color.rgb * v_color.a, t.a * v_color.a);
}
"""


def _material_key(name):
    """A material path as used by the alias/scale tables: lower case, forward
    slashes, no .vmt."""
    key = name.replace("\\", "/").lower()
    return key[:-4] if key.endswith(".vmt") else key


def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


def _rand(rng, low, high, count, exponent=1.0):
    u = rng.random(count, dtype="f4")
    if exponent != 1.0:
        u = u ** exponent
    return low + (high - low) * u


# ---------------------------------------------------------------- effects


class Effect:
    """One running instance of a particle system definition (plus its
    delayed children). Made by ParticleManager.spawn."""

    def __init__(self, manager, definition, origin, basis, delay=0.0, overlay=False, colors=None,
                 size=1.0, follow=None, offset_scale=1.0, **options):
        self.manager = manager
        self.definition = definition
        self.origin = np.asarray(origin, dtype="f4")
        self.basis = basis           # world = basis @ local (columns: forward, left, up)
        self.rng = manager.rng
        self.delay = delay
        self.overlay = overlay       # drawn over everything (no depth test), e.g. a first-person muzzle flash
        self.colors = colors         # (rgb, rgb) replacing every Color Random's range, or None
        self.size = size             # scales the whole effect: radii, offsets, speeds
        self.radius_scale = manager.radius_scale.get(_material_key(definition.material), 1.0)
        self.lifetime_scale = manager.lifetime_scale.get(_material_key(definition.material), 1.0)
        self.follow = follow         # callable -> world position it stays attached to (or None)
        # Extra options (passed on to child systems): follow_particles=False keeps
        # already-emitted particles where they are as the anchor moves (blood left
        # behind a flying gib); inherit_velocity=k gives new particles k * the
        # anchor's current velocity.
        self.options = options
        self.follow_particles = options.get("follow_particles", True)
        self.inherit_velocity = float(options.get("inherit_velocity", 0.0))
        self.anchor_velocity = np.zeros(3, "f4")
        self.offset_scale = offset_scale   # scales Position Modify Offset (0 keeps it on the origin)
        self.time = 0.0
        self.stopped = False         # no more emission (running particles finish)
        cap = int(max(1, min(definition.params.get("max_particles", 1000), _MAX_EFFECT_PARTICLES)))
        self.capacity = cap
        self.count = 0
        self.pos = np.zeros((cap, 3), "f4")
        self.vel = np.zeros((cap, 3), "f4")
        self.age = np.zeros(cap, "f4")
        self.life = np.ones(cap, "f4")
        self.radius = np.zeros(cap, "f4")
        self.radius0 = np.zeros(cap, "f4")        # as spawned, for operators that scale it
        self.rotation = np.zeros(cap, "f4")       # degrees
        self.rotation_speed = np.zeros(cap, "f4")  # degrees/second
        self.color = np.ones((cap, 3), "f4")
        self.alpha = np.ones(cap, "f4")
        self.alpha0 = np.ones(cap, "f4")
        self.seed = np.zeros((cap, 4), "f4")      # per-particle random numbers operators can lean on
        self.sequence = np.zeros(cap, "f4")       # sprite sheet sequence
        self.trail_time = np.full(cap, 0.1, "f4")  # seconds of velocity a trail sprite stretches over
        self.created = np.zeros(cap, "f4")        # effect time at spawn
        self._arrays = (self.pos, self.vel, self.age, self.life, self.radius, self.radius0,
                        self.rotation, self.rotation_speed, self.color, self.alpha, self.alpha0,
                        self.seed, self.sequence, self.trail_time, self.created)
        self.local_time = 0.0
        renderer = definition.renderers[0] if definition.renderers else None
        self.renderer = renderer.name if renderer is not None else "render_animated_sprites"
        self.renderer_params = renderer.params if renderer is not None else {}
        self._constraints = [fn for fn in definition.constraints if fn.name in _CONSTRAINTS]
        for fn in definition.constraints:
            if fn.name not in _CONSTRAINTS:
                manager.warn_unsupported(fn.name)
        self._emitters = [(fn, manager.emitters.get(fn.name)) for fn in definition.emitters]
        self._initializers = [(fn, manager.initializers.get(fn.name)) for fn in definition.initializers]
        self._operators = [(fn, manager.operators.get(fn.name)) for fn in definition.operators]
        self._emit_state = [{"emitted": 0.0, "done": False, "total": 0} for _ in self._emitters]
        for fn, impl in self._emitters + self._initializers + self._operators:
            if impl is None:
                manager.warn_unsupported(fn.name)
        self._children = [(child, d) for child, d in definition.children]
        self._children_spawned = False

    # ---- helpers the function implementations use ----

    def local(self, vec):
        """A control-point-local Source vector as a world vector (scaled)."""
        return self.basis @ (np.asarray(vec, "f4") * self.scale)

    def world(self, vec):
        """A Source world-space vector as an engine world vector (scaled)."""
        return _SOURCE_TO_WORLD @ (np.asarray(vec, "f4") * self.scale)

    def frame(self, use_local):
        """Matrix mapping Source vectors (unscaled) to the world."""
        return (self.basis if use_local else _SOURCE_TO_WORLD) * self.scale

    @property
    def scale(self):
        """Metres per Source unit for this effect (its size setting included)."""
        return self.manager.unit_scale * self.size

    # ---- lifecycle ----

    @property
    def finished(self):
        return self.count == 0 and self.time >= self.delay and (
            self.stopped or all(state["done"] for state in self._emit_state))

    def _follow_anchor(self, dt):
        """Moves the effect (origin and every live particle) by however far
        its anchor has moved since last frame, so it stays attached to it.
        Translation only - its facing is fixed at spawn."""
        if self.follow is None:
            return
        anchor = self.follow()
        if anchor is None:
            self.follow = None     # the thing it was attached to is gone: stay where it was
            return
        anchor = np.array((anchor[0], anchor[1], anchor[2]), "f4")
        delta = anchor - self.origin
        self.origin = anchor
        if dt > 1e-6:
            self.anchor_velocity = delta / dt
        if self.count and self.follow_particles:
            self.pos[:self.count] += delta

    def stop(self):
        """Stops emitting; particles already alive finish."""
        self.stopped = True

    def emit(self, amount):
        amount = min(int(amount), self.capacity - self.count)
        if amount <= 0:
            return
        p = self.definition.params
        n0, n1 = self.count, self.count + amount
        sl = slice(n0, n1)
        self.pos[sl] = self.origin
        self.vel[sl] = self.anchor_velocity * self.inherit_velocity
        self.age[sl] = 0.0
        self.life[sl] = 1.0
        self.radius[sl] = float(p.get("radius", 5.0)) * self.scale * self.radius_scale
        self.rotation[sl] = float(p.get("rotation", 0.0))
        self.rotation_speed[sl] = float(p.get("rotation_speed", 0.0))
        r, g, b, a = p.get("color", (255, 255, 255, 255))
        self.color[sl] = (r / 255.0, g / 255.0, b / 255.0)
        self.alpha0[sl] = a / 255.0
        self.seed[sl] = self.rng.random((amount, 4), dtype="f4")
        self.sequence[sl] = 0.0
        self.trail_time[sl] = 0.1
        self.created[sl] = self.local_time
        for fn, impl in self._initializers:
            if impl is not None:
                impl(self, fn.params, sl, amount)
        self.alpha[sl] = self.alpha0[sl]
        self.radius0[sl] = self.radius[sl]
        self.count = n1

    def update(self, dt):
        dt = min(dt, float(self.definition.params.get("maximum time step", 0.1)) or dt)
        self.time += dt
        if self.time < self.delay:
            return
        local_time = self.time - self.delay
        self.local_time = local_time
        self._follow_anchor(dt)
        if not self._children_spawned:
            self._children_spawned = True
            for child, child_delay in self._children:
                self.manager.spawn_definition(child, self.origin, self.basis, delay=child_delay, overlay=self.overlay,
                                              colors=self.colors, size=self.size,
                                              follow=self.follow, offset_scale=self.offset_scale, **self.options)
        if not self.stopped:
            for (fn, impl), state in zip(self._emitters, self._emit_state):
                if impl is not None and not state["done"]:
                    impl(self, fn.params, state, local_time, dt)
        n = self.count
        if n == 0:
            return
        sl = slice(0, n)
        self.age[sl] += dt
        self.rotation[sl] += self.rotation_speed[sl] * dt
        self.alpha[sl] = self.alpha0[sl]      # alpha operators multiply it down from the spawn value
        collide = self._constraints and self.manager.raycast is not None
        if collide:
            before = self.pos[sl].copy()
        for fn, impl in self._operators:
            if impl is not None:
                impl(self, fn.params, sl, n, dt)
        if collide:
            for fn in self._constraints:
                _CONSTRAINTS[fn.name](self, fn.params, before, n)
        alive = self.age[sl] < self.life[sl]
        if not alive.all():
            keep = np.nonzero(alive)[0]
            k = len(keep)
            for array in self._arrays:
                array[:k] = array[keep]
            self.count = k

    def write_instances(self, out, n, material):
        """Fills `out` (n, _FLOATS_PER_PARTICLE) float32 with the first n
        particles; `material` gives the sprite sheet layout."""
        out[:, 0:3] = self.pos[:n]
        out[:, 3] = self.radius[:n]
        out[:, 4] = np.radians(self.rotation[:n])
        out[:, 5:8] = self.color[:n]
        out[:, 8] = self.alpha[:n]
        out[:, 9:12] = self.vel[:n]
        rp = self.renderer_params
        if self.renderer == "render_sprite_trail":
            speed = np.linalg.norm(self.vel[:n], axis=1)
            length = speed * self.trail_time[:n]
            length = np.clip(length, float(rp.get("min length", 0.0)) * self.scale,
                             float(rp.get("max length", 2000.0)) * self.scale)
            fade = float(rp.get("length fade in time", 0.0))
            if fade > 0.0:
                length *= _clamp01(self.age[:n] / fade)
            out[:, 12] = np.maximum(length, 1e-4)
        else:
            out[:, 12] = 0.0
        if len(material.rects) > 1:
            seq = np.clip(self.sequence[:n].astype("i4"), 0, len(material.seq_start) - 1)
            frames = material.seq_count[seq]
            cell = material.seq_start[seq].astype("f4")
            multi = frames > 1
            if multi.any():
                rate = float(rp.get("animation rate", 0.1))
                progress = (self.age[:n] / self.life[:n]) if rp.get("animation_fit_lifetime") else self.age[:n] * rate
                cell += np.where(multi, np.minimum(frames - 1, np.floor(_clamp01(progress) * frames)), 0)
            out[:, 13] = cell
        else:
            out[:, 13] = 0.0


# ---- emitters: impl(effect, params, state, time_since_start, dt) ----


def _emit_instantaneously(fx, params, state, t, dt):
    if t < float(params.get("emission_start_time", 0.0)):
        return
    # "maximum emission per frame" (-1/0 = unlimited) spreads a burst over frames.
    total = int(params.get("num_to_emit", 100))
    per_frame = int(params.get("maximum emission per frame", -1))
    amount = total - state["total"]
    if per_frame > 0:
        amount = min(amount, per_frame)
    state["total"] += amount
    if state["total"] >= total:
        state["done"] = True
    fx.emit(amount)


def _emit_continuously(fx, params, state, t, dt):
    start = float(params.get("emission_start_time", 0.0))
    duration = float(params.get("emission_duration", 0.0))
    if t < start:
        return
    if duration > 0.0 and t >= start + duration:
        state["done"] = True
        return
    state["emitted"] += float(params.get("emission_rate", 100.0)) * dt
    whole = int(state["emitted"])
    state["emitted"] -= whole
    fx.emit(whole)


# ---- initializers: impl(effect, params, slice, count) ----


def _init_position_sphere(fx, p, sl, k):
    rng = fx.rng
    dirs = rng.standard_normal((k, 3)).astype("f4")
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
    dist = _rand(rng, float(p.get("distance_min", 0.0)), float(p.get("distance_max", 0.0)), k)
    bias = np.asarray(p.get("distance_bias", (1.0, 1.0, 1.0)), "f4")
    frame = fx.frame(bool(p.get("bias in local system", False)))
    fx.pos[sl] += (dirs * dist[:, None] * bias) @ frame.T
    speed = _rand(rng, float(p.get("speed_min", 0.0)), float(p.get("speed_max", 0.0)), k,
                  float(p.get("speed_random_exponent", 1.0)))
    fx.vel[sl] += dirs @ _SOURCE_TO_WORLD.T * (speed[:, None] * fx.scale)
    low = np.asarray(p.get("speed_in_local_coordinate_system_min", (0, 0, 0)), "f4")
    high = np.asarray(p.get("speed_in_local_coordinate_system_max", (0, 0, 0)), "f4")
    if low.any() or high.any():
        local = low + (high - low) * rng.random((k, 3), dtype="f4")
        fx.vel[sl] += local @ (fx.basis * fx.scale).T


def _init_offset_random(fx, p, sl, k):
    low = np.asarray(p.get("offset min", (0, 0, 0)), "f4")
    high = np.asarray(p.get("offset max", (0, 0, 0)), "f4")
    offset = low + (high - low) * fx.rng.random((k, 3), dtype="f4")
    offset *= fx.offset_scale
    offset = offset @ fx.frame(bool(p.get("offset in local space 0/1", False))).T
    if p.get("offset proportional to radius 0/1", False):
        offset *= (fx.radius[sl] / fx.scale)[:, None]
    fx.pos[sl] += offset


def _init_rotation_random(fx, p, sl, k):
    fx.rotation[sl] = float(p.get("rotation_initial", 0.0)) + _rand(
        fx.rng, float(p.get("rotation_offset_min", 0.0)), float(p.get("rotation_offset_max", 360.0)),
        k, float(p.get("rotation_random_exponent", 1.0)))


def _init_rotation_speed_random(fx, p, sl, k):
    fx.rotation_speed[sl] = float(p.get("rotation_speed_constant", 0.0)) + _rand(
        fx.rng, float(p.get("rotation_speed_random_min", 0.0)),
        float(p.get("rotation_speed_random_max", 0.0)), k,
        float(p.get("rotation_speed_random_exponent", 1.0)))


def _init_lifetime_random(fx, p, sl, k):
    fx.life[sl] = np.maximum(1e-4, _rand(
        fx.rng, float(p.get("lifetime_min", 0.0)), float(p.get("lifetime_max", 0.0)), k,
        float(p.get("lifetime_random_exponent", 1.0))) * fx.lifetime_scale)


def _init_radius_random(fx, p, sl, k):
    fx.radius[sl] = _rand(
        fx.rng, float(p.get("radius_min", 1.0)), float(p.get("radius_max", 1.0)), k,
        float(p.get("radius_random_exponent", 1.0))) * fx.scale * fx.radius_scale


def _init_color_random(fx, p, sl, k):
    c1, c2 = fx.colors or (p.get("color1", (255, 255, 255)), p.get("color2", (255, 255, 255)))
    c1 = np.asarray(c1[:3], "f4") / 255.0
    c2 = np.asarray(c2[:3], "f4") / 255.0
    fx.color[sl] = c1 + (c2 - c1) * fx.rng.random((k, 1), dtype="f4")


def _init_alpha_random(fx, p, sl, k):
    fx.alpha0[sl] = _rand(
        fx.rng, float(p.get("alpha_min", 255)), float(p.get("alpha_max", 255)), k,
        float(p.get("alpha_random_exponent", 1.0))) / 255.0


def _init_velocity_random(fx, p, sl, k):
    low = np.asarray(p.get("speed_in_local_coordinate_system_min", (0, 0, 0)), "f4")
    high = np.asarray(p.get("speed_in_local_coordinate_system_max", (0, 0, 0)), "f4")
    local = low + (high - low) * fx.rng.random((k, 3), dtype="f4")
    fx.vel[sl] += local @ (fx.basis * fx.scale).T


def _init_sequence_random(fx, p, sl, k):
    low, high = int(p.get("sequence_min", 0)), int(p.get("sequence_max", 0))
    fx.sequence[sl] = fx.rng.integers(min(low, high), max(low, high) + 1, k)


def _init_trail_length_random(fx, p, sl, k):
    fx.trail_time[sl] = _rand(fx.rng, float(p.get("length_min", 0.1)), float(p.get("length_max", 0.1)),
                              k, float(p.get("length_random_exponent", 1.0)))


def _init_velocity_noise(fx, p, sl, k):
    # Source samples a noise field; a random velocity in the same range is
    # the same look for a burst of one-shot particles.
    low = np.asarray(p.get("output minimum", (0, 0, 0)), "f4")
    high = np.asarray(p.get("output maximum", (0, 0, 0)), "f4")
    velocity = low + (high - low) * fx.rng.random((k, 3), dtype="f4")
    fx.vel[sl] += velocity @ fx.frame(bool(p.get("Apply Velocity in Local Space (0/1)", False))).T


def _init_remap_initial_scalar(fx, p, sl, k):
    # Only the "effect time -> alpha / radius" mapping (all the impact files use).
    if int(p.get("input field", -1)) != 8:
        return
    lo, hi = float(p.get("input minimum", 0.0)), float(p.get("input maximum", 1.0))
    f = min(1.0, max(0.0, (fx.local_time - lo) / max(hi - lo, 1e-6)))
    value = float(p.get("output minimum", 0.0)) + (float(p.get("output maximum", 1.0))
                                                    - float(p.get("output minimum", 0.0))) * f
    field = int(p.get("output field", -1))
    if field == 7:
        fx.alpha0[sl] = value
    elif field == 3:
        fx.radius[sl] = value * fx.scale * fx.radius_scale


def _init_preage(fx, p, sl, k):
    start = _rand(fx.rng, float(p.get("start age minimum", 0.0)), float(p.get("start age maximum", 0.0)), k)
    fx.age[sl] = np.minimum(start, fx.life[sl] * 0.9)


def _init_nothing(fx, p, sl, k):
    pass


# ---- operators: impl(effect, params, slice, count, dt) ----


def _op_movement_basic(fx, p, sl, n, dt):
    gravity = fx.world(p.get("gravity", (0, 0, 0)))
    drag = float(p.get("drag", 0.0))
    vel = fx.vel[sl]
    vel += gravity * dt
    if drag:
        vel *= max(0.0, 1.0 - drag * dt * 30.0)   # Source applies drag per 1/30 s tick
    fx.pos[sl] += vel * dt


def _op_alpha_fade_and_decay(fx, p, sl, n, dt):
    t = fx.age[sl] / fx.life[sl]
    start_a = float(p.get("start_alpha", 1.0))
    end_a = float(p.get("end_alpha", 0.0))
    in0, in1 = float(p.get("start_fade_in_time", 0.0)), float(p.get("end_fade_in_time", 0.5))
    out0, out1 = float(p.get("start_fade_out_time", 0.5)), float(p.get("end_fade_out_time", 1.0))
    fade_in = _clamp01((t - in0) / max(in1 - in0, 1e-6))
    fade_out = _clamp01((t - out0) / max(out1 - out0, 1e-6))
    factor = (start_a + (1.0 - start_a) * fade_in) * (1.0 + (end_a - 1.0) * fade_out)
    fx.alpha[sl] *= factor


def _op_rotation_basic(fx, p, sl, n, dt):
    fx.rotation[sl] += float(p.get("rotation_rate", 0.0)) * dt


def _op_radius_scale(fx, p, sl, n, dt):
    t = fx.age[sl] / fx.life[sl]
    t0, t1 = float(p.get("start_time", 0.0)), float(p.get("end_time", 1.0))
    s0, s1 = float(p.get("radius_start_scale", 1.0)), float(p.get("radius_end_scale", 1.0))
    f = _clamp01((t - t0) / max(t1 - t0, 1e-6))
    fx.radius[sl] = fx.radius0[sl] * (s0 + (s1 - s0) * f)


def _op_alpha_fade_out_random(fx, p, sl, n, dt):
    low, high = float(p.get("fade out time min", 0.0)), float(p.get("fade out time max", 0.0))
    fade = np.maximum(low + (high - low) * fx.seed[sl, 0] ** float(p.get("fade out time exponent", 1.0)), 1e-4)
    if p.get("proportional 0/1", True):
        remaining = 1.0 - fx.age[sl] / fx.life[sl]
    else:
        remaining = fx.life[sl] - fx.age[sl]
    fx.alpha[sl] *= _clamp01(remaining / fade)


def _op_alpha_fade_in_random(fx, p, sl, n, dt):
    low, high = float(p.get("fade in time min", 0.0)), float(p.get("fade in time max", 0.0))
    fade = np.maximum(low + (high - low) * fx.seed[sl, 1] ** float(p.get("fade in time exponent", 1.0)), 1e-4)
    elapsed = fx.age[sl] / fx.life[sl] if p.get("proportional 0/1", True) else fx.age[sl]
    fx.alpha[sl] *= _clamp01(elapsed / fade)


def _op_rotation_spin_roll(fx, p, sl, n, dt):
    fx.rotation[sl] += float(p.get("spin_rate_degrees", 0.0)) * dt


def _oscillation(fx, p, sl, dt, rate_low, rate_high):
    """Per-particle sine wave amounts for this frame: (delta of the sine
    since last frame) * rate, so an oscillated field never drifts. The rate
    and frequency are picked per particle between the file's min and max."""
    seed = fx.seed[sl]
    freq = (np.asarray(p.get("oscillation frequency min", 1.0), "f4")
            + (np.asarray(p.get("oscillation frequency max", 1.0), "f4")
               - np.asarray(p.get("oscillation frequency min", 1.0), "f4")) * (seed[:, 1:2] if np.ndim(rate_low) else seed[:, 1]))
    phase = float(p.get("oscillation start phase", 0.0))
    age = fx.age[sl]
    life = fx.life[sl]
    t_frac = age / life
    now = age[:, None] if np.ndim(rate_low) else age
    before = now - dt
    delta = np.sin(2.0 * np.pi * (freq * now + phase)) - np.sin(2.0 * np.pi * (freq * before + phase))
    rate = rate_low + (rate_high - rate_low) * (seed[:, 0:1] if np.ndim(rate_low) else seed[:, 0])
    start = float(p.get("start time min", 0.0))
    end = float(p.get("end time max", 1.0))
    window = (t_frac >= start) & (t_frac <= end) if p.get("start/end proportional", True) else (age >= start) & (age <= end)
    return delta * rate * (window[:, None] if np.ndim(rate_low) else window)


def _op_oscillate_scalar(fx, p, sl, n, dt):
    field = int(p.get("oscillation field", -1))
    low, high = float(p.get("oscillation rate min", 0.0)), float(p.get("oscillation rate max", 0.0))
    delta = _oscillation(fx, p, sl, dt, low, high)
    if field == 3:          # radius
        fx.radius[sl] += delta * (fx.radius0[sl] if p.get("proportional 0/1", False) else fx.scale)
    elif field == 4:        # rotation
        fx.rotation[sl] += delta * (np.maximum(np.abs(fx.rotation[sl]), 1.0) if p.get("proportional 0/1", False) else 1.0)
    elif field == 7:        # alpha
        fx.alpha[sl] = np.clip(fx.alpha[sl] + delta, 0.0, 1.0)


def _op_oscillate_vector(fx, p, sl, n, dt):
    if int(p.get("oscillation field", -1)) != 0:      # only position is supported
        return
    low = np.asarray(p.get("oscillation rate min", (0, 0, 0)), "f4")
    high = np.asarray(p.get("oscillation rate max", (0, 0, 0)), "f4")
    delta = _oscillation(fx, p, sl, dt, low, high)
    fx.pos[sl] += delta @ _SOURCE_TO_WORLD.T * fx.scale


# ---- constraints: impl(effect, params, positions_before_the_move, count) ----


def _constraint_collision(fx, p, before, n):
    """Keeps particles out of the world: a trace from each one's last position to
    its new one; on a hit it's put on the surface and its velocity is turned
    into a bounce/slide along it."""
    raycast = fx.manager.raycast
    bounce = float(p.get("amount of bounce", 0.0))
    slide = float(p.get("amount of slide", 0.0))
    radius = fx.radius
    for i in range(n):
        a = before[i]
        b = fx.pos[i]
        move = b - a
        dist = float(np.linalg.norm(move))
        if dist < 1e-6:
            continue
        # Look a little past the new position so a particle resting on a floor is
        # still "touching" it, and stay a hair off the surface.
        direction = move / dist
        reach = min(float(radius[i]) * 0.1, 0.05)
        hit = raycast(glm.vec3(*a), glm.vec3(*(b + direction * reach)))
        if hit is None:
            continue
        normal = np.array((hit.normal.x, hit.normal.y, hit.normal.z), "f4")
        fx.pos[i] = np.array((hit.position.x, hit.position.y, hit.position.z), "f4") + normal * 0.003
        v = fx.vel[i]
        along = float(v @ normal)
        if along < 0.0:
            tangent = v - along * normal
            fx.vel[i] = tangent * (1.0 - slide) - along * bounce * normal


_CONSTRAINTS = {"Collision via traces": _constraint_collision}


_EMITTERS = {
    "emit_instantaneously": _emit_instantaneously,
    "emit_continuously": _emit_continuously,
}
_INITIALIZERS = {
    "Position Within Sphere Random": _init_position_sphere,
    "Position Modify Offset Random": _init_offset_random,
    "Rotation Random": _init_rotation_random,
    "Rotation Speed Random": _init_rotation_speed_random,
    "Lifetime Random": _init_lifetime_random,
    "Radius Random": _init_radius_random,
    "Color Random": _init_color_random,
    "Alpha Random": _init_alpha_random,
    "Velocity Random": _init_velocity_random,
    "Sequence Random": _init_sequence_random,
    "Sequence Two Random": _init_nothing,       # a second sprite layer - not drawn
    "Rotation Yaw Random": _init_nothing,       # yaw only matters for oriented (non-camera-facing) sprites
    "Rotation Yaw Flip Random": _init_nothing,
    "Trail Length Random": _init_trail_length_random,
    "Velocity Noise": _init_velocity_noise,
    "remap initial scalar": _init_remap_initial_scalar,
    "Lifetime Pre-Age Noise": _init_preage,
}
_OPERATORS = {
    "Movement Basic": _op_movement_basic,
    "Alpha Fade and Decay": _op_alpha_fade_and_decay,
    "Rotation Basic": _op_rotation_basic,
    "Radius Scale": _op_radius_scale,
    "Alpha Fade Out Random": _op_alpha_fade_out_random,
    "Alpha Fade In Random": _op_alpha_fade_in_random,
    "Rotation Spin Roll": _op_rotation_spin_roll,
    "Oscillate Scalar": _op_oscillate_scalar,
    "Oscillate Vector": _op_oscillate_vector,
    "Lifespan Decay": lambda fx, p, sl, n, dt: None,   # every particle already dies at its lifetime
    "Decay": lambda fx, p, sl, n, dt: None,
}
# Renderers: render_animated_sprites (camera-facing) and render_sprite_trail
# (stretched along the velocity); anything else draws as plain sprites.


# ---------------------------------------------------------------- materials


class _Material:
    def __init__(self, texture, additive, rects_texture, rects, seq_start, seq_count):
        self.texture = texture
        self.additive = additive
        self.rects_texture = rects_texture   # float texture holding `rects`, one frame per texel
        self.rects = rects                   # (frames, 4) left, top, right, bottom
        self.seq_start = seq_start           # sequence -> index of its first frame
        self.seq_count = seq_count           # sequence -> its number of frames


def _grid_layout(columns, rows, per_sequence):
    """Rectangles and sequences for an evenly divided sheet."""
    rects = np.array([(c / columns, r / rows, (c + 1) / columns, (r + 1) / rows)
                      for r in range(rows) for c in range(columns)], "f4")
    per_sequence = max(1, per_sequence)
    sequences = len(rects) // per_sequence
    return rects, np.arange(sequences) * per_sequence, np.full(sequences, per_sequence)


def _filter_frames(layout, image, key, keep):
    """The layout without the sheet frames `keep(key, mean rgb)` rejects (a
    sequence left with none keeps its first). image: (h, w, 4) uint8, top row first."""
    rects, start, count = layout
    height, width = image.shape[:2]
    kept_rects, new_start, new_count = [], [], []
    for first, n in zip(start, count):
        frames = []
        for rect in rects[first:first + n]:
            l, t, r, b = rect
            region = image[int(t * height):max(int(b * height), int(t * height) + 1),
                           int(l * width):max(int(r * width), int(l * width) + 1), :3]
            if keep(key, tuple(region.reshape(-1, 3).mean(0))):
                frames.append(rect)
        if not frames:
            frames = [rects[first]]
        new_start.append(len(kept_rects))
        new_count.append(len(frames))
        kept_rects.extend(frames)
    return np.array(kept_rects, "f4"), np.array(new_start), np.array(new_count)


def _sheet_layout(sheet):
    """Rectangles and sequences from a vtf_sheet.Sheet."""
    rects, start, count = [], [], []
    for frames in sheet.sequences:
        if not frames:       # a hole in the sequence numbering: use the first frame
            frames = [(0.0, (0.0, 0.0, 1.0, 1.0))]
        start.append(len(rects))
        count.append(len(frames))
        rects.extend(rect for _seconds, rect in frames)
    return np.array(rects, "f4"), np.array(start), np.array(count)


def _read_vmt(path):
    """$key -> value of a .vmt's top-level lines (enough for $basetexture and
    the translucency flags)."""
    values = {}
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = re.match(r'\s*"?\$?([\w]+)"?\s+"?([^"\s]+)"?', line)
                if m:
                    values[m.group(1).lower()] = m.group(2)
    except OSError:
        pass
    return values


class ParticleManager:
    def __init__(self, ctx, texture_dirs=("Assets/Particles", "Assets/Particles/ParticleIMGs", "Assets/Textures"),
                 unit_scale=UNIT_SCALE, default_additive=True, depth_test=True, material_aliases=None,
                 radius_scale=None, frame_filter=None, lifetime_scale=None):
        """material_aliases: {material path (no .vmt, forward slashes, lower case):
        texture name} for materials whose .vmt isn't available, saying which image
        they use. radius_scale: {material key: factor} shrinking (or growing) only
        the sprites of systems using that material. frame_filter: optional
        callable(material key, (r, g, b) 0-255 mean of a sheet frame) -> bool,
        dropping the frames it rejects from that material's sequences.
        lifetime_scale: {material key: factor} for how long its particles live."""
        self.ctx = ctx
        self.texture_dirs = list(texture_dirs)
        self.unit_scale = unit_scale
        self.default_additive = default_additive
        self.depth_test = depth_test
        self.material_aliases = {k.lower(): v for k, v in (material_aliases or {}).items()}
        self.radius_scale = {_material_key(k): v for k, v in (radius_scale or {}).items()}
        self.frame_filter = frame_filter
        self.lifetime_scale = {_material_key(k): v for k, v in (lifetime_scale or {}).items()}
        self.rng = np.random.default_rng()
        self.emitters = dict(_EMITTERS)
        self.initializers = dict(_INITIALIZERS)
        self.operators = dict(_OPERATORS)
        self.definitions = {}
        self.effects = []
        self._materials = {}
        self._warned = set()

        self.program = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        corners = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype="f4")
        self._quad = ctx.buffer(corners.tobytes())
        self._instances = ctx.buffer(reserve=MAX_TOTAL_PARTICLES * _FLOATS_PER_PARTICLE * 4, dynamic=True)
        self._scratch = np.zeros((MAX_TOTAL_PARTICLES, _FLOATS_PER_PARTICLE), "f4")
        self.vao = ctx.vertex_array(self.program, [
            (self._quad, "2f", "in_corner"),
            (self._instances, "3f 1f 1f 4f 3f 1f 1f/i", "in_center", "in_radius", "in_rotation",
             "in_color", "in_vel", "in_trail", "in_cell"),
        ])
        self._fallback = None
        # callable(from_vec3, to_vec3) -> RayHit or None, for particle collision.
        self.raycast = None

    # ---- loading ----

    def warn_unsupported(self, name):
        if name not in self._warned:
            self._warned.add(name)
            print(f"[Particles] '{name}' isn't supported yet - skipped")

    def load(self, path):
        """Reads a .pcf and registers its systems; returns their names."""
        systems = read_pcf(path)
        self.definitions.update(systems)
        # Textures next to the file are searched first.
        folder = os.path.dirname(path)
        if folder and folder not in self.texture_dirs:
            self.texture_dirs.insert(0, folder)
        return list(systems)

    def required_images(self, names=None):
        """{material: image path or None} for the systems (all loaded by
        default) - what to put on disk for an effect to look right."""
        wanted = names or list(self.definitions)
        out = {}
        for name in wanted:
            material = self.definitions[name].material
            if material:
                vmt = self._find_vmt(material)
                found = self._find_image(material, _read_vmt(vmt) if vmt else None)
                out[material] = found
        return out

    def _texture_names(self, material, vmt_values=None):
        """The names a material's texture may go by: the .vmt's $basetexture,
        the alias set for it (see material_aliases), then the material's own."""
        stem = material.replace("\\", "/")
        if stem.lower().endswith(".vmt"):
            stem = stem[:-4]
        names = []
        if vmt_values and vmt_values.get("basetexture"):
            names.append(vmt_values["basetexture"].replace("\\", "/"))
        alias = self.material_aliases.get(stem.lower())
        if alias:
            names.append(alias)
        names.append(stem)
        return names

    def _find_file(self, names, extensions):
        for name in names:
            base = os.path.basename(name)
            for folder in self.texture_dirs:
                for candidate in (os.path.join(folder, name), os.path.join(folder, base)):
                    for ext in extensions:
                        if os.path.isfile(candidate + ext):
                            return candidate + ext
        return None

    def _find_image(self, material, vmt_values=None):
        return self._find_file(self._texture_names(material, vmt_values), _IMAGE_EXTENSIONS)

    def _find_vmt(self, material):
        stem = material.replace("\\", "/")
        if not stem.lower().endswith(".vmt"):
            stem += ".vmt"
        for folder in self.texture_dirs:
            for candidate in (os.path.join(folder, stem), os.path.join(folder, os.path.basename(stem))):
                if os.path.isfile(candidate):
                    return candidate
        return None

    def _material(self, name):
        material = self._materials.get(name)
        if material is not None:
            return material
        vmt_path = self._find_vmt(name) if name else None
        vmt = _read_vmt(vmt_path) if vmt_path else {}
        additive = self.default_additive
        if "additive" in name.lower():
            additive = True
        image = self._find_image(name, vmt) if name else None
        layout = None
        texture = None
        if image is not None:
            # A .vtf's sheet table, else a grid from a .sheet file or the .vmt.
            vtf = self._find_file(self._texture_names(name, vmt) + [os.path.splitext(image)[0]], (".vtf",))
            sheet = read_vtf_sheet(vtf) if vtf else None
            if sheet is not None:
                layout = _sheet_layout(sheet)
            else:
                grid = (int(vmt.get("sheet_columns", 1)), int(vmt.get("sheet_rows", 1)),
                        int(vmt.get("sheet_frames_per_sequence", 1)))
                sheet_file = os.path.splitext(image)[0] + ".sheet"
                if os.path.isfile(sheet_file):
                    try:
                        numbers = [int(x) for x in open(sheet_file).read().split()]
                        grid = (numbers[0], numbers[1], numbers[2] if len(numbers) > 2 else 1)
                    except (ValueError, IndexError, OSError):
                        print(f"[Particles] couldn't read {sheet_file} (expected: columns rows [frames per sequence])")
                if grid[0] * grid[1] > 1:
                    layout = _grid_layout(*grid)
            try:
                surface = pygame.image.load(image)
                width, height = surface.get_size()
                image_rows = np.frombuffer(pygame.image.tobytes(surface, "RGBA", True), "u1").reshape(height, width, 4)
                if "additive" not in name.lower():
                    # No .vmt says otherwise: an image with real transparency is
                    # alpha blended (smoke), a fully opaque one is additive (glows).
                    additive = bool(image_rows[..., 3].min() == 255)
                if layout is not None and self.frame_filter is not None:
                    layout = _filter_frames(layout, image_rows[::-1], _material_key(name), self.frame_filter)
                premultiplied = image_rows.copy()
                premultiplied[..., :3] = (image_rows[..., :3].astype("f4") * image_rows[..., 3:4] / 255.0).astype("u1")
                texture = self.ctx.texture((width, height), 4, premultiplied.tobytes())
                texture.build_mipmaps()
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                texture.repeat_x = texture.repeat_y = False
            except Exception as e:
                print(f"[Particles] couldn't load {image}: {e}")
        if vmt:
            if vmt.get("additive", "0") == "1":
                additive = True
            elif vmt.get("translucent", "0") == "1":
                additive = False
        if texture is None:
            if name and name not in self._warned:
                self._warned.add(name)
                print(f"[Particles] no image found for material '{name}' - using a soft round sprite")
            texture = self._fallback_texture()
        if layout is None:
            layout = (np.array([(0.0, 0.0, 1.0, 1.0)], "f4"), np.zeros(1, "i4"), np.ones(1, "i4"))
        rects, seq_start, seq_count = layout
        rects_texture = self.ctx.texture((len(rects), 1), 4, rects.astype("f4").tobytes(), dtype="f4")
        rects_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        material = _Material(texture, additive, rects_texture, rects, seq_start, seq_count)
        self._materials[name] = material
        return material

    def _fallback_texture(self):
        if self._fallback is None:
            size = 64
            axis = np.linspace(-1.0, 1.0, size, dtype="f4")
            r = np.sqrt(axis[None, :] ** 2 + axis[:, None] ** 2)
            a = _clamp01(1.0 - r) ** 1.5
            data = np.zeros((size, size, 4), "u1")
            data[..., :3] = (a * 255).astype("u1")[..., None]   # additive sprites read the colour, not the alpha
            data[..., 3] = (a * 255).astype("u1")
            self._fallback = self.ctx.texture((size, size), 4, data.tobytes())
            self._fallback.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._fallback.repeat_x = self._fallback.repeat_y = False
        return self._fallback

    # ---- spawning ----

    def spawn(self, name, position, forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), overlay=False,
              colors=None, size=1.0, follow=None, offset_scale=1.0, **options):
        """Starts effect `name` at `position` (world) with control point 0
        facing `forward`/`up`. Returns the Effect (call .stop() to end a
        continuous one), or None if the name isn't loaded. overlay: draw it over
        the scene ignoring depth (for effects on the first-person gun). colors:
        ((r, g, b), (r, g, b)) 0-255, replaces the colour range the file's
        Color Random initializers pick from. size: scales the whole effect.
        follow: callable returning the world position it stays attached to
        (moving the live particles along); offset_scale: scales the file's
        Position Modify Offset (0 = particles start on the spawn point). Other
        keyword options: follow_particles, inherit_velocity (see Effect)."""
        definition = self.definitions.get(name)
        if definition is None:
            return None
        f = glm.normalize(glm.vec3(forward))
        u = glm.vec3(up)
        left = glm.cross(u, f)
        if glm.length(left) < 1e-4:
            left = glm.cross(glm.vec3(0.0, 0.0, 1.0), f)
        left = glm.normalize(left)
        u = glm.cross(f, left)
        basis = np.array([[f.x, left.x, u.x], [f.y, left.y, u.y], [f.z, left.z, u.z]], "f4")
        return self.spawn_definition(definition, (position[0], position[1], position[2]), basis, overlay=overlay, colors=colors,
                                     size=size, follow=follow, offset_scale=offset_scale, **options)

    def spawn_surface(self, name, position, normal, **kwargs):
        """Starts effect `name` on a surface: control point 0's up (Source's
        +Z, what impact effects throw their particles along) is the surface
        `normal`. Takes spawn's other keyword arguments."""
        n = glm.normalize(glm.vec3(normal))
        helper = glm.vec3(0.0, 1.0, 0.0) if abs(n.y) < 0.99 else glm.vec3(1.0, 0.0, 0.0)
        return self.spawn(name, position, forward=glm.cross(helper, n), up=n, **kwargs)

    def spawn_definition(self, definition, origin, basis, delay=0.0, overlay=False, colors=None,
                         size=1.0, follow=None, offset_scale=1.0, **options):
        effect = Effect(self, definition, origin, basis, delay, overlay, colors, size, follow, offset_scale, **options)
        self.effects.append(effect)
        return effect

    def prime(self, camera):
        """Loads every material the registered systems use (images, sprite sheets) and draws one
        of each system once, so neither happens on the first shot or death. The effects appear
        in front of `camera` in the back buffer only (nothing is presented) and are cleared
        again. Call once when the game starts, with the camera in the world."""
        for definition in self.definitions.values():
            if definition.material:
                self._material(definition.material)
        self._fallback_texture()
        front = glm.normalize(glm.vec3(camera.front))
        where = glm.vec3(camera.position) + front * 4.0
        for i, name in enumerate(self.definitions):
            self.spawn(name, where, overlay=(i % 2 == 0))
        self.update(0.02)
        self.update(0.02)          # children start once their parent has run
        self.render(camera)
        self.clear()

    def clear(self):
        self.effects.clear()

    # ---- per frame ----

    def update(self, dt):
        if not self.effects:
            return
        for effect in self.effects:
            effect.update(dt)
        self.effects = [e for e in self.effects if not e.finished]

    def render(self, camera, overlay=None):
        """Draws the live effects. overlay: True = only overlay effects (see
        spawn), False = only the others, None = all."""
        live = [e for e in self.effects if e.count > 0 and e.time >= e.delay
                and (overlay is None or e.overlay == overlay)]
        if not live:
            return
        # One batch per material: the effects sharing one are packed together.
        batches = {}
        for effect in live:
            batches.setdefault((effect.definition.material, effect.overlay), []).append(effect)

        cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z], "f4")
        front = glm.normalize(glm.vec3(camera.front))
        right = glm.normalize(glm.cross(front, glm.vec3(camera.up)))
        up = glm.cross(right, front)
        program = self.program
        program["u_view_proj"].write((camera.get_projection_matrix() * camera.get_view_matrix()).to_bytes())
        program["u_right"].value = tuple(right)
        program["u_up"].value = tuple(up)
        program["u_texture"].value = 0
        program["u_rects"].value = 1
        program["u_cam_pos"].value = tuple(cam_pos)

        ctx = self.ctx
        ctx.depth_func = "<"
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.screen.depth_mask = False

        filled = 0
        front_np = np.array(tuple(front), "f4")
        for (material_name, overlay), effects in batches.items():
            start = filled
            material = self._material(material_name)
            for effect in effects:
                n = min(effect.count, MAX_TOTAL_PARTICLES - filled)
                if n <= 0:
                    break
                effect.write_instances(self._scratch[filled:filled + n], n, material)
                filled += n
            if filled == start:
                continue
            if overlay or not self.depth_test:
                ctx.disable(moderngl.DEPTH_TEST)
            else:
                ctx.enable(moderngl.DEPTH_TEST)
            batch = self._scratch[start:filled]
            if not material.additive:      # blended sprites: far ones first
                batch = batch[np.argsort(-((batch[:, 0:3] - cam_pos) @ front_np))]
            self._instances.write(np.ascontiguousarray(batch).tobytes())
            if material.additive:
                ctx.blend_func = moderngl.ONE, moderngl.ONE
            else:
                ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            material.rects_texture.use(1)
            material.texture.use(0)
            self.vao.render(moderngl.TRIANGLES, vertices=6, instances=filled - start)

        ctx.screen.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = "back"

    def destroy(self):
        self.vao.release()
        self._quad.release()
        self._instances.release()
        self.program.release()
        for material in self._materials.values():
            material.rects_texture.release()
            if material.texture is not self._fallback:
                material.texture.release()
        if self._fallback is not None:
            self._fallback.release()
