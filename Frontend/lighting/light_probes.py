"""
A sparse 3D grid of precomputed point-light data covering a static
scene's geometry - what Scene._nearest_point_lights/_sample_probe_
irradiance sample (trilinearly interpolated) to give a moving object
(the player, ...) a smooth, flicker-free, unlimited-light-count
substitute for real-time point-light shadow testing, at its CURRENT
position, instead of testing that live against the physics world (or a
capped per-light shader array) every single frame.

This plays the same role Unreal's Indirect Lighting Cache / volumetric
lightmap volume samples play for movable objects: precomputed data
sampled at the object's current position, smoothly interpolated between
neighboring samples, rather than a hard per-frame live query or a
shader-register-limited per-light loop. Two separate things are stored
per probe cell:

- "visibility": one 0..1 scalar per point light (which light this is is
  given by that light's own "_probe_index" - see Scene.add_point_light).
  Multiplied into that light's color by Scene._nearest_point_lights
  before its (still capped, still real-time) position/color/radius get
  sent to the shader for the live SPECULAR highlight - see pbr_shader.py's
  calculate_point_light_specular. A raw per-frame raycast here (the
  first version of this feature) flickered hard: a fresh independent 0/1
  sample every frame disagrees with the previous frame right at the
  exact moment an occluder's silhouette crosses the light-to-object
  line. Precomputing on a grid and interpolating BETWEEN cells turns
  that hard per-frame coin-flip into a smooth, continuous function of
  position - there's no live raycast left to disagree frame to frame.

- "irradiance": one combined RGB value, summed across EVERY point light
  in the scene (not just the shader's MAX_POINT_LIGHTS-sized array),
  already distance-attenuated and shadow-tested. Sampled once per real-
  time-lit object per frame and used as that object's whole diffuse
  point-light term (see pbr_shader.py's u_probe_irradiance) - the same
  role a baked lightmap texture already plays for STATIC geometry
  (Scene.bake_static_lighting), just sampled from an interpolated 3D
  position instead of a 2D UV. This is what actually removes
  MAX_POINT_LIGHTS as a correctness constraint for a moving object: that
  shader-side cap exists only to avoid a real, previously-hit GLSL
  "Constant register limit exceeded" link error (see pbr_shader.py's
  own module docstring) for a per-light uniform ARRAY, which has nothing
  to do with how many lights this Python-side bake loop can sum over -
  exactly the same reasoning that already lets Scene.bake_static_lighting
  bake an unlimited number of lights into a STATIC lightmap despite the
  same cap.

GRID SCOPE - why this is a sparse dict keyed by integer cell coordinate,
not one dense array covering the whole scene's combined bounding box:
a real level is mostly EMPTY space (open sky above a terrain, void below
it, the gap between disconnected structures) - gridding that at a fixed
resolution wastes almost the entire probe budget on cells nothing will
ever stand in, which in turn forces a dense grid's spacing to coarsen
(to keep total cell count bounded) far past what a small, disconnected
structure - a tunnel, a basement - actually needs. A grid built instead
from the union of every static object's OWN world AABB (each expanded
by a small cell margin) spends its whole budget near geometry a dynamic
object can actually stand next to, at a fixed spacing that doesn't
degrade just because the rest of the map is large - a narrow tunnel far
from the main terrain gets exactly the same resolution as everything
else, instead of "wherever the global bounding box happened to leave
enough count budget for.\""""

import numpy as np
import glm

# World units between adjacent probe cells. Not coarsened by total scene
# volume (see the module docstring's GRID SCOPE section) - the sparse,
# per-object-scoped cell set is what keeps total cell count bounded
# instead, so this can stay fixed/fine regardless of how large or
# sprawling the map is.
DEFAULT_PROBE_SPACING = 2.0

# How many cells of margin to add around each static object's own AABB
# when deciding which cells are "near" it - a light or a dynamic object
# sitting right at an object's edge still needs real (not clamped-to-
# nothing) neighboring cells to interpolate from.
_MARGIN_CELLS = 2

# Backstop against a pathologically detailed static scene (geometry
# covering nearly its whole bounding box, e.g. a dense city) turning
# this into a multi-minute bake: if the near-geometry cell set would
# exceed this, spacing is coarsened (not skipped) until it fits - a
# coarser grid is still strictly better than no grid at all.
_MAX_PROBES = 40000


def _object_cells(aabb_min, aabb_max, origin, spacing, margin_cells):
    lo = np.floor((np.asarray(aabb_min, dtype="f8") - origin) / spacing).astype(np.int64) - margin_cells
    hi = np.ceil((np.asarray(aabb_max, dtype="f8") - origin) / spacing).astype(np.int64) + margin_cells
    lo = np.maximum(lo, 0)
    for ix in range(lo[0], hi[0] + 1):
        for iy in range(lo[1], hi[1] + 1):
            for iz in range(lo[2], hi[2] + 1):
                yield (int(ix), int(iy), int(iz))


def build_light_probe_grid(static_aabbs, point_lights, visibility_test,
                            spacing=DEFAULT_PROBE_SPACING):
    """static_aabbs: list of (mins, maxs) numpy arrays, ONE PER static
    object (not one combined box - see the module docstring's GRID SCOPE
    section). point_lights: Scene.point_lights - each dict needs
    "position", "radius", "color", "intensity", and "_probe_index"
    (assigned by Scene.add_point_light; both output arrays are indexed
    by it). visibility_test(from_pos, to_pos): a callable returning True
    if nothing blocks a straight line between the two (see PhysicsWorld.
    line_of_sight) - passed in rather than imported directly so this
    module has no Physics-package dependency.

    Returns None if there are no point lights or no static geometry to
    build a grid from at all - callers should treat a None grid as
    "every light always fully visible, zero baked irradiance", the same
    behavior as before this feature existed.

    A light farther than its own radius from a given probe cell is
    skipped for that cell (visibility stays at its default of 1.0,
    irradiance keeps its default of 0) without spending a raycast on it
    - a light that far away already contributes ~0 via its own falloff
    curve regardless of what this grid says, so shadow-testing it here
    would be wasted work with no visible effect."""
    if not point_lights or not static_aabbs:
        return None

    origin = np.minimum.reduce([mins for mins, _ in static_aabbs])

    cells = set()
    for mins, maxs in static_aabbs:
        cells.update(_object_cells(mins, maxs, origin, spacing, _MARGIN_CELLS))

    while len(cells) > _MAX_PROBES:
        spacing *= 1.25
        cells = set()
        for mins, maxs in static_aabbs:
            cells.update(_object_cells(mins, maxs, origin, spacing, _MARGIN_CELLS))

    num_lights = len(point_lights)
    lights_in_range = [
        (light["_probe_index"], light["position"],
         light["color"] * light["intensity"], float(light["radius"]))
        for light in point_lights
    ]

    visibility = {}
    irradiance = {}
    for cell in cells:
        ix, iy, iz = cell
        probe_pos = glm.vec3(
            origin[0] + ix * spacing, origin[1] + iy * spacing, origin[2] + iz * spacing,
        )
        vis_vec = np.ones(num_lights, dtype=np.float32)
        irr_vec = np.zeros(3, dtype=np.float32)

        for probe_index, light_pos, light_color, radius in lights_in_range:
            radius = max(radius, 0.01)
            delta = light_pos - probe_pos
            dist_sq = glm.dot(delta, delta)
            if dist_sq > radius * radius:
                continue

            visible = visibility_test(light_pos, probe_pos)
            vis_vec[probe_index] = 1.0 if visible else 0.0
            if not visible:
                continue

            # Same falloff/attenuation curve as pbr_shader.py's real-time
            # calculate_point_light_specular and lightmap_baker.py's own
            # bake_point_light, MINUS their per-fragment NdotL term - a
            # probe has no single fixed surface normal to weight by (it
            # represents an omnidirectional sample point, not a surface),
            # so this is plain incident irradiance, matching how a real
            # volumetric lightmap's SH-0 (ambient) term works.
            dist = dist_sq ** 0.5
            falloff = max(0.0, 1.0 - (dist / radius) ** 4)
            atten = (falloff * falloff) / (dist * dist + 1.0)
            irr_vec += np.asarray(light_color, dtype=np.float32) * atten

        visibility[cell] = vis_vec
        irradiance[cell] = irr_vec

    return {
        "origin": origin,
        "spacing": float(spacing),
        "num_lights": num_lights,
        "visibility": visibility,
        "irradiance": irradiance,
    }


def _sample_sparse_grid(cell_dict, default_vec, origin, spacing, position):
    """Trilinear interpolation over a sparse {cell: vector} dict - a
    missing corner cell (outside the near-geometry region built above)
    falls back to `default_vec` rather than raising, since a position
    right at the edge of the populated region legitimately has some
    real cells and some missing ones among its 8 interpolation corners."""
    rel = (np.asarray(position, dtype="f8") - origin) / spacing
    i0 = np.floor(rel).astype(np.int64)
    frac = rel - i0
    fx, fy, fz = frac

    def get(dx, dy, dz):
        return cell_dict.get((int(i0[0]) + dx, int(i0[1]) + dy, int(i0[2]) + dz), default_vec)

    c00 = get(0, 0, 0) * (1 - fx) + get(1, 0, 0) * fx
    c10 = get(0, 1, 0) * (1 - fx) + get(1, 1, 0) * fx
    c01 = get(0, 0, 1) * (1 - fx) + get(1, 0, 1) * fx
    c11 = get(0, 1, 1) * (1 - fx) + get(1, 1, 1) * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz


def sample_light_probe_visibility(grid, position, num_lights):
    """Trilinearly interpolated per-light visibility (0..1, shape
    (num_lights,)) at world-space `position`. grid=None returns all-1.0
    - see build_light_probe_grid's own docstring."""
    if grid is None:
        return np.ones(num_lights, dtype=np.float32)
    return _sample_sparse_grid(
        grid["visibility"], np.ones(grid["num_lights"], dtype=np.float32),
        grid["origin"], grid["spacing"], position,
    )


def sample_light_probe_irradiance(grid, position):
    """Trilinearly interpolated combined RGB irradiance (shape (3,)) at
    world-space `position` - see the module docstring's "irradiance"
    bullet. grid=None returns zero (no point lights contribute)."""
    if grid is None:
        return np.zeros(3, dtype=np.float32)
    return _sample_sparse_grid(
        grid["irradiance"], np.zeros(3, dtype=np.float32),
        grid["origin"], grid["spacing"], position,
    )
