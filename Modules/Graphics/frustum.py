"""
View-frustum culling, Gribb & Hartmann's method (2001): the six frustum planes are sums/differences of the
view-projection matrix's rows, so nothing beyond the frame's view_proj and an object's world-space AABB is needed.
Only answers "is this AABB provably outside" - a conservative test for skipping draws, not an intersection query.
"""

import math


def extract_frustum_planes(view_proj):
    """view_proj: a glm.mat4 (clip = view_proj * world). Returns 6 [a, b, c, d] lists - left, right, bottom, top,
    near, far - with a point inside a plane when a*x + b*y + c*z + d >= 0. Normalized so d is in world units.
    Plain Python, not numpy: numpy's per-call overhead was most of the cost for a 4x4 matrix."""
    rows = [[view_proj[col][row] for col in range(4)] for row in range(4)]   # glm indexes [column][row]
    r3 = rows[3]
    planes = []
    for i in range(3):   # -w <= x, y, z <= w, as (row3 +/- row_i) . (world, 1) >= 0
        for sign in (1.0, -1.0):
            plane = [r3[k] + sign * rows[i][k] for k in range(4)]
            length = math.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2])
            if length < 1e-9:
                length = 1.0
            planes.append([v / length for v in plane])
    return planes


def aabb_outside_frustum(mins, maxs, planes):
    """True if the box (mins, maxs: plain float triples) is entirely outside at least one plane - the "positive
    vertex" test. It can miss a box past a frustum corner (drawing it anyway) but never culls a visible one."""
    mx0, mx1, mx2 = maxs
    mn0, mn1, mn2 = mins
    for a, b, c, d in planes:
        px = mx0 if a >= 0.0 else mn0
        py = mx1 if b >= 0.0 else mn1
        pz = mx2 if c >= 0.0 else mn2
        if a * px + b * py + c * pz + d < 0.0:
            return True
    return False
