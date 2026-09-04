import moderngl
import numpy as np


VERTEX_SHADER = """
#version 330

uniform mat4 u_mvp;
uniform mat4 u_model;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;

void main()
{
    vec4 world_pos =
        u_model *
        vec4(in_position, 1.0);

    v_position = world_pos.xyz;

    v_normal =
        mat3(transpose(inverse(u_model))) *
        in_normal;

    v_color = in_color;
    v_uv = in_uv;

    gl_Position =
        u_mvp *
        vec4(in_position, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330

uniform vec3 u_light_dir;
uniform vec3 u_eye_pos;
uniform mat4 u_view_matrix;

uniform float u_metallic;
uniform float u_roughness;
uniform vec3 u_emissive;

uniform sampler2D u_texture;
uniform int u_has_texture;

uniform sampler2D u_metallic_roughness_texture;
uniform int u_has_metallic_roughness_texture;

uniform sampler2D u_shadow_maps[3];
uniform mat4 u_light_mvps[3];
uniform float u_cascade_splits[3];
uniform int u_has_shadows;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;

out vec4 fragColor;

const float PI = 3.14159265359;


/* ================================================================
   PBR
   ================================================================ */

float distributionGGX(
    vec3 N,
    vec3 H,
    float roughness
)
{
    float a =
        roughness *
        roughness;

    float a2 =
        max(
            a * a,
            0.00001
        );

    float NdotH =
        max(
            dot(N, H),
            0.0
        );

    float NdotH2 =
        NdotH *
        NdotH;

    float denom =
        NdotH2 *
        (a2 - 1.0)
        + 1.0;

    denom =
        PI *
        denom *
        denom;

    return a2 /
           max(
               denom,
               0.0001
           );
}


float geometrySchlickGGX(
    float NdotV,
    float roughness
)
{
    float r =
        roughness + 1.0;

    float k =
        (r * r) /
        8.0;

    return NdotV /
           (
               NdotV *
               (1.0 - k)
               + k
           );
}


float geometrySmith(
    vec3 N,
    vec3 V,
    vec3 L,
    float roughness
)
{
    float NdotV =
        max(
            dot(N, V),
            0.0
        );

    float NdotL =
        max(
            dot(N, L),
            0.0
        );

    float ggx1 =
        geometrySchlickGGX(
            NdotV,
            roughness
        );

    float ggx2 =
        geometrySchlickGGX(
            NdotL,
            roughness
        );

    return ggx1 * ggx2;
}


vec3 fresnelSchlick(
    float cosTheta,
    vec3 F0
)
{
    return F0 +
           (1.0 - F0) *
           pow(
               clamp(
                   1.0 - cosTheta,
                   0.0,
                   1.0
               ),
               5.0
           );
}


/* ================================================================
   CASCADE SELECTION
   ================================================================ */

int get_cascade_index(
    float view_depth
)
{
    if (
        view_depth <
        u_cascade_splits[0]
    )
    {
        return 0;
    }

    if (
        view_depth <
        u_cascade_splits[1]
    )
    {
        return 1;
    }

    return 2;
}


/* ================================================================
   SHADOW MAP
   ================================================================ */

float calculate_shadow(
    vec3 world_pos,
    float view_depth,
    vec3 normal,
    vec3 light_dir
)
{
    if (u_has_shadows == 0)
    {
        return 0.0;
    }

    /*
     * Don't attempt to use a cascade if the fragment is behind
     * the camera.
     */
    if (view_depth <= 0.0)
    {
        return 0.0;
    }

    int cascade =
        get_cascade_index(
            view_depth
        );

    /*
     * Transform the world-space fragment into the selected
     * light-space projection.
     */
    vec4 light_space =
        u_light_mvps[cascade] *
        vec4(
            world_pos,
            1.0
        );

    if (
        light_space.w <= 0.00001
    )
    {
        return 0.0;
    }

    vec3 ndc =
        light_space.xyz /
        light_space.w;

    /*
     * OpenGL NDC is [-1, +1].
     * Texture coordinates are [0, 1].
     */
    vec3 shadow_coord =
        ndc * 0.5 +
        0.5;

    /*
     * Outside the cascade's shadow projection means there is
     * no shadow contribution from this map.
     */
    if (
        shadow_coord.x < 0.0 ||
        shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 ||
        shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 ||
        shadow_coord.z > 1.0
    )
    {
        return 0.0;
    }

    float current_depth =
        shadow_coord.z;

    /*
     * Stronger bias for surfaces facing away from the light.
     *
     * The previous bias was very small and could produce acne,
     * while increasing the cascade number arbitrarily was not
     * particularly robust.
     */
    float NdotL =
        clamp(
            dot(
                normal,
                light_dir
            ),
            0.0,
            1.0
        );

    float bias =
        max(
            0.0005,
            0.003 *
            (1.0 - NdotL)
        );

    /*
     * Keep a little more bias on farther cascades because their
     * orthographic projections cover a larger world-space area
     * per texel.
     */
    if (cascade == 1)
    {
        bias *= 1.5;
    }
    else if (cascade == 2)
    {
        bias *= 2.0;
    }

    /*
     * The shadow maps are created at 2048x2048.
     */
    vec2 texel_size =
        vec2(
            1.0 / 2048.0,
            1.0 / 2048.0
        );

    /*
     * 3x3 percentage-closer filtering.
     */
    float shadow = 0.0;

    for (
        int x = -1;
        x <= 1;
        x++
    )
    {
        for (
            int y = -1;
            y <= 1;
            y++
        )
        {
            vec2 offset =
                vec2(
                    float(x),
                    float(y)
                ) *
                texel_size;

            vec2 uv =
                shadow_coord.xy +
                offset;

            /*
             * Clamp the PCF samples to the map.
             */
            uv =
                clamp(
                    uv,
                    vec2(0.001),
                    vec2(0.999)
                );

            float map_depth =
                texture(
                    u_shadow_maps[cascade],
                    uv
                ).r;

            if (
                current_depth - bias >
                map_depth
            )
            {
                shadow += 1.0;
            }
        }
    }

    return shadow / 9.0;
}


/* ================================================================
   MAIN
   ================================================================ */

void main()
{
    vec3 N =
        normalize(
            v_normal
        );

    vec3 V =
        normalize(
            u_eye_pos -
            v_position
        );

    /*
     * light_dir is defined as the direction from the surface
     * toward the light.
     */
    vec3 L =
        normalize(
            u_light_dir
        );

    vec3 H =
        normalize(
            V + L
        );

    /* ------------------------------------------------------------
       Albedo
       ------------------------------------------------------------ */

    vec3 raw_albedo =
        u_has_texture == 1
            ? texture(
                u_texture,
                v_uv
              ).rgb
            : v_color;

    vec3 albedo =
        pow(
            max(
                raw_albedo,
                vec3(0.0)
            ),
            vec3(2.2)
        );

    /* ------------------------------------------------------------
       Material
       ------------------------------------------------------------ */

    float metal =
        clamp(
            u_metallic,
            0.0,
            1.0
        );

    float rough =
        clamp(
            u_roughness,
            0.04,
            1.0
        );

    if (
        u_has_metallic_roughness_texture == 1
    )
    {
        vec4 mr =
            texture(
                u_metallic_roughness_texture,
                v_uv
            );

        /*
         * glTF:
         *   Green = roughness
         *   Blue  = metallic
         */
        rough *= mr.g;
        metal *= mr.b;
    }

    rough =
        clamp(
            rough,
            0.04,
            1.0
        );

    /* ------------------------------------------------------------
       BRDF
       ------------------------------------------------------------ */

    vec3 F0 =
        mix(
            vec3(0.04),
            albedo,
            metal
        );

    float NDF =
        distributionGGX(
            N,
            H,
            rough
        );

    float G =
        geometrySmith(
            N,
            V,
            L,
            rough
        );

    vec3 F =
        fresnelSchlick(
            max(
                dot(H, V),
                0.0
            ),
            F0
        );

    vec3 numerator =
        NDF *
        G *
        F;

    float denominator =
        4.0 *
        max(
            dot(N, V),
            0.0
        ) *
        max(
            dot(N, L),
            0.0
        )
        + 0.0001;

    vec3 specular =
        numerator /
        denominator;

    vec3 kS =
        F;

    vec3 kD =
        (
            vec3(1.0) -
            kS
        ) *
        (
            1.0 -
            metal
        );

    float NdotL =
        max(
            dot(N, L),
            0.0
        );

    /* ------------------------------------------------------------
       Shadow
       ------------------------------------------------------------ */

    float view_depth =
        -(
            u_view_matrix *
            vec4(
                v_position,
                1.0
            )
        ).z;

    float shadow =
        calculate_shadow(
            v_position,
            view_depth,
            N,
            L
        );

    /*
     * Make the shadow clearly visible.
     *
     * 0.0 = completely unlit by the direct light.
     * 1.0 = fully lit.
     */
    float shadow_attenuation =
        1.0 -
        shadow;

    /* ------------------------------------------------------------
       Lighting
       ------------------------------------------------------------ */

    vec3 diffuse =
        kD *
        albedo /
        PI;

    vec3 radiance =
        vec3(2.0);

    vec3 direct_light =
        (
            diffuse +
            specular
        ) *
        radiance *
        NdotL;

    direct_light *=
        shadow_attenuation;

    /*
     * Small ambient term so completely shadowed areas aren't
     * pitch black.
     */
    vec3 ambient =
        vec3(0.025) *
        albedo;

    vec3 color =
        direct_light +
        ambient +
        u_emissive;

    /* ------------------------------------------------------------
       Tonemapping / Gamma
       ------------------------------------------------------------ */

    color =
        color /
        (
            color +
            vec3(1.0)
        );

    color =
        pow(
            color,
            vec3(1.0 / 2.2)
        );

    fragColor =
        vec4(
            color,
            1.0
        );
}
"""


def create_program(ctx):
    return ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAGMENT_SHADER
    )


def bind_material(
    prog,
    item_data,
    model_matrix,
    camera,
    light_dir,
    shadow_manager=None
):
    view = camera.get_view_matrix()
    projection = camera.get_projection_matrix()

    mvp = projection * view * model_matrix

    uniforms = {
        "u_mvp":
            mvp.to_bytes(),

        "u_model":
            model_matrix.to_bytes(),

        "u_view_matrix":
            view.to_bytes(),

        "u_light_dir":
            tuple(light_dir),

        "u_eye_pos":
            tuple(camera.position),

        "u_metallic":
            item_data.get(
                "metallic",
                0.0
            ),

        "u_roughness":
            min(
                item_data.get(
                    "roughness",
                    1.0
                ),
                1.0
            ),

        "u_emissive":
            tuple(
                item_data.get(
                    "emissive",
                    [
                        0.0,
                        0.0,
                        0.0
                    ]
                )
            ),

        "u_has_texture":
            item_data.get(
                "has_texture",
                0
            ),

        "u_has_metallic_roughness_texture":
            item_data.get(
                "has_metallic_roughness_texture",
                0
            ),

        "u_has_shadows":
            0
    }

    for name, value in uniforms.items():
        if name not in prog:
            continue

        if isinstance(
            value,
            (bytes, bytearray)
        ):
            prog[name].write(value)
        else:
            prog[name].value = value

    # -------------------------------------------------------------
    # Shadow maps
    # -------------------------------------------------------------

    if shadow_manager is not None:
        for i, tex in enumerate(
            shadow_manager.depth_textures
        ):
            tex.use(
                location=2 + i
            )

        if "u_shadow_maps" in prog:
            prog[
                "u_shadow_maps"
            ].value = (
                2,
                3,
                4
            )

        if "u_has_shadows" in prog:
            prog[
                "u_has_shadows"
            ].value = 1

        if "u_cascade_splits" in prog:
            splits = list(
                shadow_manager.splits
            )

            while len(splits) < 3:
                splits.append(
                    shadow_manager.far
                )

            data = np.array(
                splits[:3],
                dtype=np.float32
            )

            prog[
                "u_cascade_splits"
            ].write(
                data.tobytes()
            )

        if "u_light_mvps" in prog:
            matrices = (
                shadow_manager.light_mvps
            )

            data = b"".join(
                matrix.to_bytes()
                for matrix in matrices[:3]
            )

            prog[
                "u_light_mvps"
            ].write(data)

    # -------------------------------------------------------------
    # Base color texture
    # -------------------------------------------------------------

    texture = item_data.get(
        "texture"
    )

    if (
        texture is not None
        and "u_texture" in prog
    ):
        texture.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR
        )

        texture.use(
            location=0
        )

        prog[
            "u_texture"
        ].value = 0

    # -------------------------------------------------------------
    # Metallic / roughness texture
    # -------------------------------------------------------------

    mr_texture = item_data.get(
        "metallic_roughness_texture"
    )

    if (
        mr_texture is not None
        and
        "u_metallic_roughness_texture"
        in prog
    ):
        mr_texture.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR
        )

        mr_texture.use(
            location=1
        )

        prog[
            "u_metallic_roughness_texture"
        ].value = 1