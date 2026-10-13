"""
View-frustum culling - Gribb & Hartmann's method (2001): the six
frustum planes fall directly out of linear combinations of a view-
projection matrix's own rows, with no separate construction from FOV/
aspect/near/far needed. This is the "easy to set up" version of frustum
culling specifically because of that - it needs nothing from the camera
beyond the exact view_proj matrix already being built every frame for
rendering (see Scene._render_scene/_render_transparent_objects's own
pbr_view_proj/blend_view_proj), and a per-object world-space AABB.

Not lightweight-general-purpose: this only ever answers "is this AABB
provably fully outside the frustum" - a conservative test (see
aabb_outside_frustum's own docstring) suited to "should I skip drawing
this" decisions, not to exact intersection/collision queries.
"""

import numpy as np


def extract_frustum_planes(view_proj):
    """view_proj: a glm.mat4 (projection * view - clip = view_proj *
    vec4(world_pos, 1.0)). Returns a list of 6 plain (a, b, c, d) float
    tuples - NOT a numpy array (see aabb_outside_frustum's own comment
    on why: iterating/unpacking numpy array rows in a Python for-loop,
    hundreds of times a frame, was measured as real, avoidable overhead
    - the same "numpy per-element overhead dominates at this scale"
    issue already hit once before in this codebase, see AnimationChannel.
    sample's own docstring in skeletal_loader.py). Building the planes
    themselves still uses numpy (cheap - a handful of vector ops, done
    once per frame, not once per object); only the OUTPUT converts to
    plain Python before returning. A point (x, y, z) is on the INSIDE
    half-space of a plane when a*x + b*y + c*z + d >= 0 - order is
    left, right, bottom, top, near, far (arbitrary, nothing downstream
    depends on it)."""
    # .to_bytes() is glm's own column-major byte layout - reshape(4, 4)
    # on that flat buffer therefore yields the TRANSPOSE of the "true"
    # column-major matrix (verified against glm's own point transform
    # in Scene._bake_directional_shadow_map's own comment on this exact
    # conversion) - so "row i of the true view_proj" is column i of
    # this reshaped array, i.e. m[:, i], not m[i, :].
    m = np.frombuffer(view_proj.to_bytes(), dtype="f4").reshape(4, 4).astype("f8")
    r0, r1, r2, r3 = (m[:, i] for i in range(4))

    # Standard OpenGL clip-space bounds: a clip-space point is visible
    # when -w <= x,y,z <= w. Each pair below is exactly that inequality
    # rewritten as (row3 +/- rowi) . vec4(world_pos, 1) >= 0.
    planes = np.array([
        r3 + r0,  # left:   x >= -w
        r3 - r0,  # right:  x <=  w
        r3 + r1,  # bottom: y >= -w
        r3 - r1,  # top:    y <=  w
        r3 + r2,  # near:   z >= -w
        r3 - r2,  # far:    z <=  w
    ])
    # Normalizing (a, b, c) to unit length isn't required for a pure
    # inside/outside sign test, but keeps the plane's "d" comparable in
    # world units rather than whatever scale view_proj happens to carry
    # - harmless and cheap (6 sqrt calls, once per frame, not per
    # object).
    lengths = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    lengths[lengths < 1e-9] = 1.0
    return (planes / lengths).tolist()


def aabb_outside_frustum(mins, maxs, planes):
    """True if the world-space AABB (mins, maxs - each a length-3
    sequence of plain floats, NOT numpy scalars - see Scene._get_world_
    aabb, which converts before caching for the same per-call-overhead
    reason as `planes` below) is PROVABLY entirely outside at least one
    of `planes` (a plain list of (a, b, c, d) tuples - see
    extract_frustum_planes) - the standard "positive vertex" AABB/plane
    test: for each plane, the one AABB corner that would score highest
    against that plane's normal is checked, and if even that corner is
    on the outside half-space, nothing else in the box can be inside
    either.

    Conservative in one direction only: an AABB that actually straddles
    a frustum corner without crossing any single plane head-on can
    still test as "not outside" (a false negative - draws one object
    that turn out fully offscreen). It can never produce a false
    positive (never culls something actually visible) - the safe
    tradeoff for a "should I skip this draw" decision."""
    mx0, mx1, mx2 = maxs
    mn0, mn1, mn2 = mins
    for a, b, c, d in planes:
        px = mx0 if a >= 0.0 else mn0
        py = mx1 if b >= 0.0 else mn1
        pz = mx2 if c >= 0.0 else mn2
        if a * px + b * py + c * pz + d < 0.0:
            return True
    return False
