"""
Skybox rendering: six independent flat quads, each with its own plain
2D texture, arranged into a box around the camera.

This replaced an earlier version built on a real GL cubemap
(ctx.texture_cube / samplerCube). That approach was architecturally
wrong for what Source actually does here: a hardware cubemap always
enforces a perfect CUBE with a fixed direction-to-face mapping - there
is no way to make GL_TEXTURE_CUBE_MAP render a SHORT, non-cubic box
with independently-proportioned faces. Real-world observation (in-game
in GMod) showed the actual skybox is a short box - roughly half height,
floor sitting at the horizon - with its side textures displayed
completely UNSTRETCHED, contradicting the earlier assumption that
Source's $basetexturetransform "scale 1 2" was stretching a half-height
image to fill a square cube face.

This also explains something that didn't make sense before: if the
box's floor sits at the horizon, a player standing on the ground
essentially never sees the "down" face at all - which is exactly why
Source maps get away with a disposable low-res placeholder there.

Box shape is fully configurable (top_height, bottom_height,
half_extent). The default (half_extent=1.0, bottom_height=0.0,
top_height=1.0) makes each side face exactly 2:1 (width:height) in
world-space proportions, matching the observed 1024x512 native pixel
aspect of these particular textures with zero stretching needed - not
independently verified pixel-exact against the original map, just the
geometrically consistent choice given what's been confirmed so far.

Face order matches the same convention used elsewhere in this project
(PointShadowMap.FACE_DIRECTIONS): 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z.

Six separate draw calls (one per face, each binding its own texture) -
trivially cheap for something drawn once per frame regardless.
"""

import moderngl
import numpy as np
from PIL import Image

# Face order: 0:+X 1:-X 2:+Y(up) 3:-Y(down) 4:+Z(front) 5:-Z(back).
# Which neighboring face borders each face on each UV side - verified
# directly against _build_box_geometry's actual vertex data (not
# assumed from memory), since getting this wrong would blend each face
# toward the WRONG neighbor's color.
_FACE_NEIGHBORS = {
    0: {"left": 5, "right": 4, "bottom": 3, "top": 2},  # +X
    1: {"left": 4, "right": 5, "bottom": 3, "top": 2},  # -X
    2: {"left": 1, "right": 0, "bottom": 4, "top": 5},  # +Y (up)
    3: {"left": 1, "right": 0, "bottom": 5, "top": 4},  # -Y (down)
    4: {"left": 0, "right": 1, "bottom": 3, "top": 2},  # +Z (front)
    5: {"left": 1, "right": 0, "bottom": 3, "top": 2},  # -Z (back)
}


SKYBOX_VERTEX_SHADER = """
#version 330
uniform mat4 u_view;
uniform mat4 u_projection;

in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;

    // Strip translation from the view matrix - the skybox should never
    // appear to move as the camera moves, only rotate with it.
    mat4 rot_view = mat4(mat3(u_view));
    vec4 pos = u_projection * rot_view * vec4(in_position, 1.0);

    // Force depth to the far plane after the perspective divide by
    // setting z = w, so the skybox renders behind everything else
    // regardless of the box's actual (arbitrary, small) size.
    gl_Position = pos.xyww;
}
"""

SKYBOX_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_face;
uniform float u_edge_fade;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 texColor = texture(u_face, v_uv);

    // Fades each face toward black near its own edges - doesn't blend
    // the actual content between two different face textures (that
    // needs real geometry overlap across every edge, a much bigger
    // change), but two adjacent faces both darkening toward black as
    // they approach their shared boundary turns a hard, jarring color
    // jump into a soft dark seam instead - much less noticeable,
    // especially against an already-dark tinted sky. u_edge_fade is
    // the UV-space margin (0.0 = no fade at all, e.g. 0.05 fades
    // starting 5% of the way in from each edge).
    float edge_dist = min(min(v_uv.x, 1.0 - v_uv.x), min(v_uv.y, 1.0 - v_uv.y));
    float fade = u_edge_fade > 0.0 ? smoothstep(0.0, u_edge_fade, edge_dist) : 1.0;

    fragColor = vec4(texColor.rgb * fade, 1.0);
}
"""


def create_skybox_program(ctx):
    return ctx.program(vertex_shader=SKYBOX_VERTEX_SHADER, fragment_shader=SKYBOX_FRAGMENT_SHADER)


def _quad(corners, uvs):
    """corners/uvs: 4 entries each, going around the quad in order.
    Returns 6 (position, uv) vertices - two triangles (0,1,2) and
    (0,2,3) - as a flat list of floats: x,y,z,u,v per vertex."""
    order = [0, 1, 2, 0, 2, 3]
    verts = []
    for i in order:
        verts.extend(corners[i])
        verts.extend(uvs[i])
    return verts


def _build_box_geometry(top_height, bottom_height, half_extent):
    """Returns a flat float list: 6 faces x 6 vertices x 5 floats
    (x,y,z,u,v) = 180 floats, in face order +X,-X,+Y,-Y,+Z,-Z. Winding
    isn't carefully verified for inside-vs-outside correctness since
    culling is disabled for the skybox draw either way - only the UV
    orientation matters for how each texture appears, and that's the
    kind of thing that needs a visual check (use the rotate_uv helper
    below per-face if an image looks mirrored/rotated wrong)."""
    he = half_extent
    b = bottom_height
    t = top_height

    uv_std = [(0, 0), (1, 0), (1, 1), (0, 1)]

    faces = []

    # +X (right)
    faces.append(_quad(
        [(he, b, -he), (he, b, he), (he, t, he), (he, t, -he)],
        uv_std
    ))
    # -X (left)
    faces.append(_quad(
        [(-he, b, he), (-he, b, -he), (-he, t, -he), (-he, t, he)],
        uv_std
    ))
    # +Y (up)
    faces.append(_quad(
        [(-he, t, he), (he, t, he), (he, t, -he), (-he, t, -he)],
        uv_std
    ))
    # -Y (down)
    faces.append(_quad(
        [(-he, b, -he), (he, b, -he), (he, b, he), (-he, b, he)],
        uv_std
    ))
    # +Z (front)
    faces.append(_quad(
        [(he, b, he), (-he, b, he), (-he, t, he), (he, t, he)],
        uv_std
    ))
    # -Z (back)
    faces.append(_quad(
        [(-he, b, -he), (he, b, -he), (he, t, -he), (-he, t, -he)],
        uv_std
    ))

    flat = []
    for face in faces:
        flat.extend(face)
    return flat


def load_skybox_textures(ctx, face_paths, tint=None, padding=0):
    """face_paths: a sequence of exactly 6 image file paths, in order
    +X, -X, +Y, -Y, +Z, -Z. Unlike the earlier cubemap-based version,
    faces are NOT forced to be square or the same size as each other -
    each is its own independent plain 2D texture, uploaded at its
    native resolution and aspect ratio.

    tint: optional color correction applied to the pixel data before
    upload - matches Source's per-material $color parameter. Pass
    either None (no tint), a single (r,g,b) tuple (applied to all 6),
    or a list of 6 (r,g,b) tuples (one per face, if they genuinely
    differ - check each face's .vmt rather than assuming they match).

    padding: pixels of edge-replication padding added to each face
    before upload (default 0, no padding). NOTE: since repeat_x/repeat_y
    are already disabled below (clamp-to-edge), sampling past a face's
    boundary already just holds that edge pixel's color - padding
    produces the same result as clamp-to-edge already does on its own,
    so with the current per-face UV mapping this won't visibly change
    anything. It won't fix the harsh seam where two different,
    unrelated face textures meet either - that's caused by zero
    blending between separate textures, not by edge-sampling behavior,
    and padding doesn't touch that at all."""
    if len(face_paths) != 6:
        raise ValueError(f"Skybox needs exactly 6 face images (+X,-X,+Y,-Y,+Z,-Z), got {len(face_paths)}")

    if tint is None:
        tints = [(1.0, 1.0, 1.0)] * 6
    elif len(tint) == 3 and all(isinstance(c, (int, float)) for c in tint):
        tints = [tint] * 6
    else:
        if len(tint) != 6:
            raise ValueError(f"tint must be a single (r,g,b) or a list of 6, got {len(tint)}")
        tints = tint

    textures = []
    for path, face_tint in zip(face_paths, tints):
        img = Image.open(path).convert("RGB")

        if face_tint != (1.0, 1.0, 1.0):
            arr = np.asarray(img, dtype=np.float32) * np.array(face_tint, dtype=np.float32)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(img)

        if padding > 0:
            arr = np.pad(arr, ((padding, padding), (padding, padding), (0, 0)), mode="edge")

        size = (arr.shape[1], arr.shape[0])
        data = arr.tobytes()

        tex = ctx.texture(size, 3, data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # moderngl textures default to repeat_x/repeat_y = True (GL_REPEAT).
        # Combined with linear filtering, sampling right at a face's edge
        # (every face's UV spans exactly 0..1, so this happens exactly at
        # every seam) can blend the boundary pixel with the WRAPPED-AROUND
        # pixel from the opposite side of this same texture - a genuinely
        # wrong color mixed in right where two faces meet, not just an
        # unblended hard edge. Clamping instead means sampling at/near the
        # boundary just holds the actual edge pixel's own color.
        tex.repeat_x = False
        tex.repeat_y = False
        textures.append(tex)

    return textures


def create_skybox_vao(ctx, prog, top_height=1.0, bottom_height=-0.05, half_extent=1.0, rotations=None):
    """Builds one VBO containing all 6 faces (6 vertices each, 36
    total) and a single VAO - render a specific face later via
    vao.render(vertices=6, first=face_index*6).

    top_height/bottom_height/half_extent: box shape in world units.
    IMPORTANT: since the skybox's view matrix has translation stripped,
    the camera is always effectively positioned at local origin (0,0,0)
    within this box - for every viewing direction to actually hit the
    box's geometry, the origin MUST sit strictly INSIDE the box's
    volume, not exactly on its boundary. bottom_height=0.0 would put
    the floor plane exactly AT the camera's position - a degenerate
    case where looking level or downward never intersects the box at
    all (a hollow box with a hole exactly where the camera stands),
    which is exactly what produced a mostly-black view with only
    strange slivers of geometry visible at steep upward angles. The
    default -0.05 keeps the floor just barely below the camera -
    visually close enough to "floor at the horizon" while keeping the
    camera safely inside the box. Keep bottom_height negative (or at
    least clearly less than 0) for the same reason if you change it.

    The defaults otherwise make each side face's WIDTH still work out
    to 2:1 against a *nominal* height of 1.0 (matching these textures'
    native 1024x512 aspect) - the small negative bottom_height barely
    changes that ratio in practice.

    rotations: optional list of 6 degree values (0/90/180/270 - other
    values aren't snapped to anything meaningful for a UV rotation),
    rotating that face's texture coordinates - for a face whose source
    material rotates its texture. Like the box shape, get this right by
    checking the actual rendered result rather than assuming.

    Returns (vao, vbo) - caller should hang onto vbo for cleanup."""
    flat = _build_box_geometry(top_height, bottom_height, half_extent)
    data = np.array(flat, dtype="f4").reshape(-1, 5)  # 36 rows of (x,y,z,u,v)

    if rotations is not None:
        if len(rotations) != 6:
            raise ValueError(f"rotations must be a list of 6 degree values, got {len(rotations)}")
        for face_index, degrees in enumerate(rotations):
            steps = int(round(degrees / 90.0)) % 4
            if steps == 0:
                continue
            start = face_index * 6
            for row in range(start, start + 6):
                u, v = data[row, 3], data[row, 4]
                for _ in range(steps):
                    u, v = v, 1.0 - u
                data[row, 3], data[row, 4] = u, v

    vbo = ctx.buffer(data.astype("f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 2f", "in_position", "in_uv")])
    return vao, vbo


def render_skybox(ctx, prog, vao, textures, camera, edge_fade=0.05):
    """Call this AFTER rendering all other scene geometry (shadows and
    the color pass), so the skybox only shows through where nothing
    else was drawn. textures: list of 6 (from load_skybox_textures),
    in the same +X,-X,+Y,-Y,+Z,-Z face order as the vao's geometry.

    edge_fade: UV-space margin (0..~0.2 is reasonable) each face fades
    toward black over, softening the seam where two different face
    textures meet - see the fragment shader's comment for why this
    doesn't blend actual content, just makes the boundary less harsh.
    0.0 disables it entirely (hard, unfaded edges, matching the
    original behavior)."""
    prog["u_view"].write(camera.get_view_matrix().to_bytes())
    prog["u_projection"].write(camera.get_projection_matrix().to_bytes())
    prog["u_edge_fade"].value = edge_fade

    ctx.disable(moderngl.CULL_FACE)
    ctx.depth_func = "<="

    for face_index, tex in enumerate(textures):
        tex.use(location=0)
        prog["u_face"].value = 0
        vao.render(moderngl.TRIANGLES, vertices=6, first=face_index * 6)

    # ctx.depth_func is write-only in moderngl (reading it back raises
    # NotImplementedError) - hardcode the restore to "<", the normal
    # depth func used everywhere else in this project.
    ctx.depth_func = "<"
    ctx.enable(moderngl.CULL_FACE)