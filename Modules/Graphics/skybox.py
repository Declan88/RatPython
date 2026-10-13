"""
Skybox rendering. Two independent options:

1. Six independent flat quads, each with its own plain 2D texture,
   arranged into a box around the camera (create_skybox_program /
   load_skybox_textures / create_skybox_vao / render_skybox). Replaces
   an earlier real GL cubemap (ctx.texture_cube / samplerCube) version -
   a hardware cubemap always enforces a perfect CUBE with a fixed
   direction-to-face mapping, which can't represent a short, non-cubic
   skybox shape. Face order matches PointShadowMap.FACE_DIRECTIONS
   elsewhere in this project: 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z.

2. A single equirectangular HDR panorama (.exr), sampled directly by
   direction vector (create_equirect_skybox_program /
   load_equirect_texture / create_equirect_skybox_vao /
   render_equirect_skybox). One continuous image, no discrete faces, so
   the seam problem that motivated option 1's edge-fade/neighbor-blend
   work doesn't exist here - the cleaner option when the source asset
   is already a single panorama rather than 6 separate face images.

   NOTE: loading requires the OpenEXR package (pip install OpenEXR),
   same as lightmap_cache_io.py's .exr path elsewhere in this project,
   and hasn't been verified against a real .exr file or a live GPU in
   this environment. The equirect UV mapping below is the standard
   Y-up formula used across many shader references, but if a specific
   panorama looks vertically flipped or seamed at an unexpected
   longitude, that's worth checking for, not already confirmed correct.

Both options are independent - use whichever matches your actual
source assets. Don't call both add_skybox() and an equirect setup in
the same scene; whichever renders wouldn't be dangerous, just redundant.
"""

import math

import glm
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

# Plain position-only unit cube (36 vertices, no UVs needed) - used by
# the equirect path, where the vertex position itself doubles directly
# as the sampling direction vector, same technique as the original
# cubemap-based skybox before it was replaced by the 6-flat-quad system.
_DIRECTION_CUBE_VERTICES = np.array([
    -1, -1, -1,  1, -1, -1,  1,  1, -1,   1,  1, -1, -1,  1, -1, -1, -1, -1,  # back
    -1, -1,  1,  1, -1,  1,  1,  1,  1,   1,  1,  1, -1,  1,  1, -1, -1,  1,  # front
    -1,  1,  1, -1,  1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  1, -1,  1,  1,  # left
     1,  1,  1,  1,  1, -1,  1, -1, -1,   1, -1, -1,  1, -1,  1,  1,  1,  1,  # right
    -1, -1, -1,  1, -1, -1,  1, -1,  1,   1, -1,  1, -1, -1,  1, -1, -1, -1,  # bottom
    -1,  1, -1,  1,  1, -1,  1,  1,  1,   1,  1,  1, -1,  1,  1, -1,  1, -1,  # top
], dtype="f4")


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
uniform vec3 u_neighbor_left;
uniform vec3 u_neighbor_right;
uniform vec3 u_neighbor_bottom;
uniform vec3 u_neighbor_top;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 result = texture(u_face, v_uv).rgb;

    // Fades each edge toward the ACTUAL average color of whatever
    // face is really on the other side of that edge (see
    // _FACE_NEIGHBORS), instead of fading toward black. Doesn't
    // continue real image content (cloud shapes, detail) across the
    // seam - true content-accurate blending needs real geometry
    // overlap across every edge, a much bigger change - but landing on
    // the neighbor's real color instead of black means the transition
    // reads as "smoothly becoming the sky next to it" rather than "a
    // dark line/wedge", which is what a hard color mismatch or a
    // fade-to-black both looked like before this.
    if (u_edge_fade > 0.0) {
        float fade_left = 1.0 - smoothstep(0.0, u_edge_fade, v_uv.x);
        float fade_right = 1.0 - smoothstep(0.0, u_edge_fade, 1.0 - v_uv.x);
        float fade_bottom = 1.0 - smoothstep(0.0, u_edge_fade, v_uv.y);
        float fade_top = 1.0 - smoothstep(0.0, u_edge_fade, 1.0 - v_uv.y);

        result = mix(result, u_neighbor_left, fade_left);
        result = mix(result, u_neighbor_right, fade_right);
        result = mix(result, u_neighbor_bottom, fade_bottom);
        result = mix(result, u_neighbor_top, fade_top);
    }

    fragColor = vec4(result, 1.0);
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
    kind of thing that needs a visual check (use the rotations param of
    create_skybox_vao if an image looks mirrored/rotated wrong)."""
    he, b, t = half_extent, bottom_height, top_height
    uv_std = [(0, 0), (1, 0), (1, 1), (0, 1)]

    face_corners = [
        [(he, b, -he), (he, b, he), (he, t, he), (he, t, -he)],       # +X
        [(-he, b, he), (-he, b, -he), (-he, t, -he), (-he, t, he)],   # -X
        [(-he, t, he), (he, t, he), (he, t, -he), (-he, t, -he)],     # +Y
        [(-he, b, -he), (he, b, -he), (he, b, he), (-he, b, he)],     # -Y
        [(he, b, he), (-he, b, he), (-he, t, he), (he, t, he)],       # +Z
        [(-he, b, -he), (he, b, -he), (he, t, -he), (-he, t, -he)],   # -Z
    ]

    flat = []
    for corners in face_corners:
        flat.extend(_quad(corners, uv_std))
    return flat


def load_skybox_textures(ctx, face_paths, tint=None, padding=0):
    """face_paths: a sequence of exactly 6 image file paths, in order
    +X, -X, +Y, -Y, +Z, -Z. Unlike the earlier cubemap-based version,
    faces are NOT forced to be square or the same size as each other -
    each is its own independent plain 2D texture, uploaded at its
    native resolution and aspect ratio.

    tint: optional (r,g,b) color correction applied to the pixel data
    before upload - matches Source's per-material $color parameter.
    Pass None (no tint), a single tuple (applied to all 6), or a list
    of 6 (one per face, if they genuinely differ).

    padding: pixels of edge-replication padding added to each face
    (default 0). NOTE: since repeat_x/repeat_y are already disabled
    below (clamp-to-edge), sampling past a face's boundary already just
    holds that edge pixel's color, so with the current per-face UV
    mapping this won't visibly change anything.

    Returns (textures, average_colors) - average_colors is a list of 6
    (r,g,b) tuples (each 0..1), the mean color of that face's pixel
    data post-tint, used by render_skybox to fade each edge toward its
    actual neighboring face's color."""
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
    average_colors = []
    for path, face_tint in zip(face_paths, tints):
        img = Image.open(path).convert("RGB")

        if face_tint != (1.0, 1.0, 1.0):
            arr = np.asarray(img, dtype=np.float32) * np.array(face_tint, dtype=np.float32)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(img)

        average_colors.append(tuple((arr.astype(np.float32).mean(axis=(0, 1)) / 255.0).tolist()))

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

    return textures, average_colors


def create_skybox_vao(ctx, prog, top_height=1.0, bottom_height=-1.0, half_extent=1.0, rotations=None):
    """Builds one VBO containing all 6 faces (6 vertices each, 36
    total) and a single VAO - render a specific face later via
    vao.render(vertices=6, first=face_index*6).

    top_height/bottom_height/half_extent: box shape in world units.
    Defaults make a full symmetric cube (every face, including the top/
    bottom caps, exactly 1:1) - the right choice when all 6 textures
    are the same square size. For mismatched/stretched textures (e.g.
    side faces stored at half height relative to the caps), set
    bottom_height/top_height asymmetrically instead.

    IMPORTANT: since the skybox's view matrix has translation stripped,
    the camera is always effectively positioned at local origin (0,0,0)
    within this box - for every viewing direction to actually hit the
    box's geometry, the origin MUST sit strictly INSIDE the box's
    volume, not exactly on its boundary. bottom_height=0.0 (or any
    value >= 0) would put the floor plane AT or ABOVE the camera's
    position, a degenerate case where looking level or downward never
    intersects the box at all. The symmetric default (-1.0 to 1.0)
    keeps the camera safely at the exact center.

    rotations: optional list of 6 degree values (0/90/180/270), rotating
    that face's texture coordinates - for a face whose source material
    rotates its texture. Verify visually rather than assuming.

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


def render_skybox(ctx, prog, vao, textures, average_colors, camera, edge_fade=0.05):
    """Call this AFTER rendering all other scene geometry (shadows and
    the color pass), so the skybox only shows through where nothing
    else was drawn. textures/average_colors: from load_skybox_textures,
    in the same +X,-X,+Y,-Y,+Z,-Z face order as the vao's geometry.

    edge_fade: UV-space margin (0..~0.2 is reasonable) each face fades
    toward its actual neighboring face's average color over, softening
    the seam where two different face textures meet - see the fragment
    shader's comment for what this does and doesn't achieve. 0.0
    disables it entirely (hard, unfaded edges)."""
    prog["u_view"].write(camera.get_view_matrix().to_bytes())
    prog["u_projection"].write(camera.get_projection_matrix().to_bytes())
    prog["u_edge_fade"].value = edge_fade

    ctx.disable(moderngl.CULL_FACE)
    ctx.depth_func = "<="

    for face_index, tex in enumerate(textures):
        tex.use(location=0)
        prog["u_face"].value = 0

        neighbors = _FACE_NEIGHBORS[face_index]
        prog["u_neighbor_left"].value = average_colors[neighbors["left"]]
        prog["u_neighbor_right"].value = average_colors[neighbors["right"]]
        prog["u_neighbor_bottom"].value = average_colors[neighbors["bottom"]]
        prog["u_neighbor_top"].value = average_colors[neighbors["top"]]

        vao.render(moderngl.TRIANGLES, vertices=6, first=face_index * 6)

    # ctx.depth_func is write-only in moderngl (reading it back raises
    # NotImplementedError) - hardcode the restore to "<", the normal
    # depth func used everywhere else in this project.
    ctx.depth_func = "<"
    ctx.enable(moderngl.CULL_FACE)


# =================================================================
# EQUIRECTANGULAR HDR PANORAMA (single .exr file, no discrete faces)
# =================================================================

EQUIRECT_VERTEX_SHADER = """
#version 330
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_yaw;       // radians: turns the panorama around the vertical axis (0 = as authored)

in vec3 in_position;
out vec3 v_direction;

void main() {
    float c = cos(u_yaw), s = sin(u_yaw);
    v_direction = vec3(c * in_position.x + s * in_position.z, in_position.y,
                       -s * in_position.x + c * in_position.z);

    mat4 rot_view = mat4(mat3(u_view));
    vec4 pos = u_projection * rot_view * vec4(in_position, 1.0);
    gl_Position = pos.xyww;
}
"""

EQUIRECT_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_equirect;
uniform float u_exposure;
uniform bool u_apply_tonemap;
uniform float u_yaw;            // radians - the same uniform the vertex shader turns the sky by
// "Flat" mode (u_flat_span > 0): the panorama is laid straight onto the screen by its own longitude/
// latitude instead of being projected through a lens, so nothing is stretched however much of it is
// shown. u_flat_span is the vertical span in degrees (180 = the image's full height); the horizontal
// span follows from the window's shape. u_view_center is the (u, v) of the screen's centre.
uniform float u_flat_span;
uniform vec2 u_viewport;
uniform vec2 u_view_center;
// Optional haze toward the scene's fog colour at the horizon (Scene.set_fog's sky_density; 0 = none).
uniform vec3 u_fog_color;
uniform float u_fog_sky_density;

in vec3 v_direction;
out vec4 fragColor;

const float PI = 3.14159265359;

void main() {
    vec3 dir = normalize(v_direction);

    // Standard Y-up equirectangular mapping: longitude (rotation
    // around Y) -> u, latitude (elevation) -> v. V is flipped (0.5 -
    // asin(...)/PI rather than 0.5 + asin(...)/PI) to correct for the
    // standard mismatch between how image files store rows (top-to-
    // bottom) and OpenGL's texture V-coordinate convention (bottom-to-
    // top) - confirmed needed empirically (zenith/nadir were swapped
    // without this).
    float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    if (u_flat_span > 0.0) {
        vec2 n = gl_FragCoord.xy / u_viewport - 0.5;
        float h_span = u_flat_span * u_viewport.x / u_viewport.y;
        // (minus yaw: the same sense as the projected mode - a larger yaw slides the scenery right)
        u = u_view_center.x + n.x * h_span / 360.0 - u_yaw / (2.0 * PI);
        v = u_view_center.y - n.y * u_flat_span / 180.0;
    }

    vec3 color = texture(u_equirect, vec2(u, v)).rgb * u_exposure;

    // Only for HDR (.exr) sources: raw radiance values are unbounded
    // and need Reinhard tonemap + gamma to display sensibly, same as
    // the main PBR shader (pbr_shader.py). An LDR (.png/.jpg) source
    // is ALREADY normal 0..1, display-ready, sRGB-encoded data -
    // running it through this same tonemap would incorrectly darken/
    // wash it out (Reinhard maps an input of 1.0 down to 0.5), so it's
    // skipped entirely for that case.
    if (u_apply_tonemap) {
        color = color / (color + vec3(1.0));
        color = pow(color, vec3(1.0 / 2.2));
    }

    if (u_fog_sky_density > 0.0 && u_flat_span <= 0.0) {
        float haze = u_fog_sky_density * pow(1.0 - abs(clamp(dir.y, -1.0, 1.0)), 3.0);
        color = mix(color, u_fog_color, haze);
    }

    fragColor = vec4(color, 1.0);
}
"""


def create_equirect_skybox_program(ctx):
    return ctx.program(vertex_shader=EQUIRECT_VERTEX_SHADER, fragment_shader=EQUIRECT_FRAGMENT_SHADER)


def irradiance_sh_from_equirect(linear_arr, max_width=256):
    """Diffuse "image based lighting" for an equirect panorama, as 9 spherical-harmonic
    coefficients (a (9, 3) float array, RGB each; order Y00, Y1-1 (y), Y10 (z), Y11 (x), Y2-2 (xy),
    Y2-1 (yz), Y20, Y21 (xz), Y22 (x^2-y^2)). pbr_shader.py's env_sh_ambient evaluates it at a surface
    normal to get the light that surface picks up from the whole sky: the sun-like glows are lit from
    where they are, the dark ground from below. The coefficients are already cosine-convolved and
    divided by pi, so a uniform sky of colour c gives exactly c - the same scale as the plain
    sky/ground average this replaces, and a Lambert surface is just albedo * evaluation.

    linear_arr: (H, W, 3) LINEAR colour (see load_equirect_texture). Uses the same direction
    convention as EQUIRECT_FRAGMENT_SHADER's mapping: u = atan2(z, x) / 2pi + 0.5, v = 0.5 - asin(y) / pi.
    Done once at load, on a copy shrunk to max_width (lighting is smooth, so nothing is lost)."""
    height, width = linear_arr.shape[:2]
    step = max(1, width // max_width)
    h, w = height // step, width // step
    small = linear_arr[:h * step, :w * step].astype(np.float64).reshape(h, step, w, step, 3).mean(axis=(1, 3))

    lon = ((np.arange(w) + 0.5) / w - 0.5) * 2.0 * np.pi          # atan2(z, x)
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi                # asin(y)
    lon, lat = np.meshgrid(lon, lat)
    x, y, z = np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)
    solid_angle = (2.0 * np.pi / w) * (np.pi / h) * np.cos(lat)   # per pixel

    basis = np.stack([
        0.282095 * np.ones_like(x),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z, 0.546274 * (x * x - y * y),
    ])                                                            # (9, h, w)
    coefficients = np.einsum("khw,hwc,hw->kc", basis, small, solid_angle)
    convolve = np.array([1.0] + [2.0 / 3.0] * 3 + [0.25] * 5)     # cosine lobe / pi, per band
    return coefficients * convolve[:, None]


def _hemisphere_colors_from_equirect(linear_arr):
    """linear_arr: (H, W, 3) float array already in LINEAR color space
    (see load_equirect_texture - HDR sources already are, LDR sources
    need gamma-decoding first). Returns (sky_color, ground_color), each
    an (r, g, b) tuple - solid-angle-weighted averages of the upper
    (dir.y > 0) and lower (dir.y < 0) hemispheres, for use as a
    hemisphere/"skylight"-style ambient term (Scene.add_equirect_skybox
    stores these; pbr_shader.py blends between them by the surface
    normal's up-component).

    Matches the same Y-up equirect convention EQUIRECT_FRAGMENT_SHADER's
    dir->uv mapping uses: row v=0 is the +Y zenith, row v=1 is the -Y
    nadir. Weighted by sin(theta) per row (theta = polar angle from +Y,
    so theta=0 at the zenith row and theta=PI at the nadir row) rather
    than a flat average, since equirect rows near the poles cover far
    less real solid angle than rows near the equator - a flat average
    would let a handful of zenith/nadir pixels dominate the result."""
    height = linear_arr.shape[0]
    row_v = (np.arange(height, dtype=np.float64) + 0.5) / height
    weights = np.sin(np.pi * row_v)  # zero at the poles, peaks at the equator

    row_means = linear_arr.mean(axis=1).astype(np.float64)  # (H, 3) - flat per-row average across longitude

    half = height // 2
    sky_color = np.average(row_means[:half], axis=0, weights=weights[:half])
    ground_color = np.average(row_means[half:], axis=0, weights=weights[half:])

    return tuple(sky_color.tolist()), tuple(ground_color.tolist())


def load_equirect_texture(ctx, path):
    """Loads a single equirectangular panorama as a plain 2D texture -
    HDR (.exr, needs the OpenEXR package) or LDR (.png/.jpg/etc, via
    PIL, no extra dependency) are both supported, auto-detected from
    the file extension.

    Returns (texture, is_hdr, sky_color, ground_color, environment_sh). environment_sh is the
    panorama's diffuse lighting as 9 spherical-harmonic coefficients (see
    irradiance_sh_from_equirect). is_hdr tells
    render_equirect_skybox whether to apply HDR tonemapping - .exr's
    raw radiance values are unbounded and need it; a normal LDR image
    is already display-ready and would be incorrectly darkened by
    running it through the same tonemap (see the fragment shader's
    comment). sky_color/ground_color: see
    _hemisphere_colors_from_equirect - both already in LINEAR space,
    ready to feed straight into the (also-linear) lighting math in
    pbr_shader.py, with no further gamma handling needed by the caller.

    Not verified against a real .exr file or a live GPU in this
    environment - the PNG/PIL path is the same well-established loading
    code used elsewhere in this project (e.g. model_loader.py's
    textures), so it's on much more solid ground than the .exr path."""
    path_str = str(path)

    if path_str.lower().endswith(".exr"):
        import OpenEXR
        with OpenEXR.File(path_str) as infile:
            arr = infile.channels()["RGB"].pixels.astype(np.float16)
        height, width = arr.shape[0], arr.shape[1]
        tex = ctx.texture((width, height), 3, arr.tobytes(), dtype="f2")
        is_hdr = True
        # Raw HDR radiance is already linear - no gamma decode needed.
        linear_arr = arr.astype(np.float32)
    else:
        img = Image.open(path_str).convert("RGB")
        tex = ctx.texture(img.size, 3, img.tobytes())
        is_hdr = False
        # LDR pixels are sRGB-encoded, display-ready data - decode to
        # linear the same way the main shader decodes albedo textures
        # (pow(x, 2.2)) before using them in lighting math.
        linear_arr = np.power(np.asarray(img, dtype=np.float32) / 255.0, 2.2)

    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # repeat_x=True is deliberate and correct here, unlike the 6-face
    # system's repeat_x=False fix: this is ONE continuous panorama, and
    # its u=0 and u=1 edges genuinely ARE the same physical seam (360
    # degrees around) - wrapping/blending across them is the actually
    # correct behavior, not the earlier bug where wrapping blended two
    # UNRELATED separate face textures together. repeat_y stays off -
    # the top/bottom (poles) shouldn't wrap into each other.
    tex.repeat_x = True
    tex.repeat_y = False

    sky_color, ground_color = _hemisphere_colors_from_equirect(linear_arr)
    environment_sh = irradiance_sh_from_equirect(linear_arr)

    return tex, is_hdr, sky_color, ground_color, environment_sh


def create_equirect_skybox_vao(ctx, prog):
    """Position-only cube - the vertex position doubles directly as
    the sampling direction, no UVs needed since the equirect mapping is
    computed from the direction vector in the fragment shader.

    Returns (vao, vbo) - caller should hang onto vbo for cleanup."""
    vbo = ctx.buffer(_DIRECTION_CUBE_VERTICES.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f", "in_position")])
    return vao, vbo


def render_equirect_skybox(ctx, prog, vao, texture, camera, exposure=1.0, apply_tonemap=True,
                           yaw_degrees=0.0, fov_degrees=None, flat_vertical_span=None, fog=None):
    """Call this AFTER rendering all other scene geometry (shadows and
    the color pass), so the skybox only shows through where nothing
    else was drawn.

    exposure: multiplier on the raw values before tonemapping - the
    standard way HDRI panoramas expose a brightness control, since raw
    radiance values don't have one inherent "correct" display
    brightness. Still applies for LDR sources too (just a plain
    brightness multiplier there), defaulting to 1.0 (no change).

    apply_tonemap: whether to run the result through Reinhard tonemap +
    gamma - set this to whatever load_equirect_texture's is_hdr
    returned (True for .exr, False for .png/.jpg/etc) - see the
    fragment shader's comment for why this matters.

    yaw_degrees: spins the panorama around the vertical axis by this much (a slowly
    turning menu backdrop passes an angle that grows over time); 0 leaves it as authored.

    fov_degrees: draws the sky with its own vertical field of view instead of the camera's (None =
    the camera's). The sky is a backdrop at infinity - it never occludes or lines up with anything
    else - so it can be widened without distorting the scene in front of it. A wider FOV still
    stretches the sky toward the screen's edges, as any perspective projection does.

    flat_vertical_span: draws the panorama FLAT instead (fov_degrees is then ignored): the screen
    shows the image as it is authored, this many degrees of its height (180 = all of it, 90 = the
    middle half, zoomed in) and as much width as the window's shape allows, with no stretching at
    any size. The yaw scrolls it sideways. None = the normal projected sky.

    fog: Scene.fog (or None). The sky is not fogged like geometry (as in Source, it is infinitely far
    away) - only if the fog's sky_density is above 0 is it hazed toward the fog colour at the horizon."""
    texture.use(location=0)
    prog["u_equirect"].value = 0
    prog["u_exposure"].value = exposure
    prog["u_yaw"].value = math.radians(yaw_degrees)
    prog["u_apply_tonemap"].value = apply_tonemap
    prog["u_view"].write(camera.get_view_matrix().to_bytes())
    if fov_degrees is None:
        projection = camera.get_projection_matrix()
    else:
        projection = glm.perspective(glm.radians(fov_degrees), camera.aspect, camera.near, camera.far)
    prog["u_projection"].write(projection.to_bytes())
    front = camera.front
    haze = float(fog["sky_density"]) if fog is not None else 0.0
    prog["u_fog_sky_density"].value = haze
    if haze > 0.0:
        prog["u_fog_color"].value = tuple(fog["color"])
    prog["u_flat_span"].value = float(flat_vertical_span or 0.0)
    if flat_vertical_span:
        _x, _y, view_w, view_h = ctx.viewport
        prog["u_viewport"].value = (float(view_w), float(view_h))
        prog["u_view_center"].value = (
            math.atan2(front.z, front.x) / (2.0 * math.pi) + 0.5,
            0.5 - math.asin(max(-1.0, min(1.0, front.y))) / math.pi,
        )

    ctx.disable(moderngl.CULL_FACE)
    ctx.depth_func = "<="

    vao.render(moderngl.TRIANGLES)

    ctx.depth_func = "<"
    ctx.enable(moderngl.CULL_FACE)