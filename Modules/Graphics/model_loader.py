import trimesh
import numpy as np
from pathlib import Path
from PIL import Image
import moderngl


def _has_attribute(prog, name):
    """
    Safely determine whether a ModernGL program contains
    a particular vertex attribute.
    """
    try:
        prog[name]
        return True
    except (KeyError, IndexError, AttributeError):
        return False


def load_glb(filepath, ctx, prog):
    path = Path(filepath)

    if not path.exists():
        print(
            f"[Warning] Model file not found: "
            f"{path.resolve()}. Skipping load."
        )
        return None

    vbo = None
    normal_vbo = None
    color_vbo = None
    uv_vbo = None
    ibo = None
    vao = None
    texture_obj = None
    metallic_roughness_texture_obj = None

    try:
        # ---------------------------------------------------------
        # Load mesh
        # ---------------------------------------------------------

        scene_or_mesh = trimesh.load(str(path))

        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh

        if mesh is None:
            raise RuntimeError("Trimesh returned no mesh.")

        if len(mesh.vertices) == 0:
            raise RuntimeError("Mesh contains no vertices.")

        if len(mesh.faces) == 0:
            raise RuntimeError("Mesh contains no faces.")

        vertices = np.asarray(
            mesh.vertices,
            dtype="f4"
        )

        faces = np.asarray(
            mesh.faces,
            dtype="i4"
        )

        # ---------------------------------------------------------
        # Normals
        # ---------------------------------------------------------

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(
            v1 - v0,
            v2 - v0
        )

        norm_lens = np.linalg.norm(
            face_normals,
            axis=1,
            keepdims=True
        )

        norm_lens[norm_lens < 0.000001] = 1.0

        face_normals /= norm_lens

        normals = np.zeros_like(vertices)

        for i in range(3):
            np.add.at(
                normals,
                faces[:, i],
                face_normals
            )

        # ---------------------------------------------------------
        # Average normals across identical positions
        # ---------------------------------------------------------

        unique_pos, inverse_indices = np.unique(
            vertices,
            axis=0,
            return_inverse=True
        )

        shared_normals = np.zeros_like(
            unique_pos
        )

        np.add.at(
            shared_normals,
            inverse_indices,
            normals
        )

        norm_lens = np.linalg.norm(
            shared_normals,
            axis=1,
            keepdims=True
        )

        norm_lens[norm_lens < 0.000001] = 1.0

        shared_normals /= norm_lens

        normals = (
            shared_normals[inverse_indices]
            .astype("f4")
        )

        # ---------------------------------------------------------
        # UVs
        # ---------------------------------------------------------

        if (
            hasattr(mesh.visual, "uv")
            and mesh.visual.uv is not None
        ):
            uvs = np.asarray(
                mesh.visual.uv,
                dtype="f4"
            )

            if len(uvs) == len(vertices):
                uvs = uvs.copy()
                uvs[:, 1] = 1.0 - uvs[:, 1]
            else:
                uvs = np.zeros(
                    (len(vertices), 2),
                    dtype="f4"
                )
        else:
            uvs = np.zeros(
                (len(vertices), 2),
                dtype="f4"
            )

        # ---------------------------------------------------------
        # Material defaults
        # ---------------------------------------------------------

        base_color = np.array(
            [0.8, 0.8, 0.8],
            dtype="f4"
        )

        metallic = 0.1
        roughness = 0.5

        emissive = np.array(
            [0.0, 0.0, 0.0],
            dtype="f4"
        )

        # ---------------------------------------------------------
        # Material
        # ---------------------------------------------------------

        if (
            hasattr(mesh.visual, "material")
            and mesh.visual.material is not None
        ):
            mat = mesh.visual.material

            # -----------------------------------------------------
            # Base color
            # -----------------------------------------------------

            main_color = getattr(
                mat,
                "main_color",
                None
            )

            if main_color is not None:
                color = np.asarray(
                    main_color[:3],
                    dtype="f4"
                )

                if np.max(color) > 1.0:
                    color /= 255.0

                base_color = color

            else:
                base_factor = getattr(
                    mat,
                    "baseColorFactor",
                    None
                )

                if base_factor is not None:
                    color = np.asarray(
                        base_factor[:3],
                        dtype="f4"
                    )

                    if np.max(color) > 1.0:
                        color /= 255.0

                    base_color = color

                else:
                    diffuse = getattr(
                        mat,
                        "diffuse",
                        None
                    )

                    if diffuse is not None:
                        color = np.asarray(
                            diffuse[:3],
                            dtype="f4"
                        )

                        if np.max(color) > 1.0:
                            color /= 255.0

                        base_color = color

            # -----------------------------------------------------
            # Metallic
            # -----------------------------------------------------

            metallic_factor = getattr(
                mat,
                "metallicFactor",
                None
            )

            if metallic_factor is not None:
                metallic = float(
                    metallic_factor
                )

            # -----------------------------------------------------
            # Roughness
            # -----------------------------------------------------

            roughness_factor = getattr(
                mat,
                "roughnessFactor",
                None
            )

            if roughness_factor is not None:
                roughness = float(
                    roughness_factor
                )

            # -----------------------------------------------------
            # Emissive
            # -----------------------------------------------------

            emissive_factor = getattr(
                mat,
                "emissiveFactor",
                None
            )

            if emissive_factor is not None:
                emissive = np.asarray(
                    emissive_factor[:3],
                    dtype="f4"
                )

                if np.max(emissive) > 1.0:
                    emissive /= 255.0

            # -----------------------------------------------------
            # Base color texture
            # -----------------------------------------------------

            img = getattr(
                mat,
                "image",
                None
            )

            if (
                img is None
                and hasattr(mat, "baseColorTexture")
            ):
                img = getattr(
                    mat,
                    "baseColorTexture",
                    None
                )

            if (
                img is None
                and isinstance(
                    scene_or_mesh,
                    trimesh.Scene
                )
                and hasattr(
                    scene_or_mesh,
                    "textures"
                )
            ):
                textures = scene_or_mesh.textures

                if textures:
                    img = list(
                        textures.values()
                    )[0]

            if img is not None:
                if not isinstance(
                    img,
                    Image.Image
                ):
                    img = Image.fromarray(
                        np.asarray(img)
                    )

                img = img.convert("RGB")

                texture_obj = ctx.texture(
                    img.size,
                    3,
                    img.tobytes()
                )

                texture_obj.filter = (
                    moderngl.LINEAR,
                    moderngl.LINEAR
                )

                texture_obj.repeat_x = True
                texture_obj.repeat_y = True

            # -----------------------------------------------------
            # Metallic / roughness texture
            # -----------------------------------------------------

            mr_img = getattr(
                mat,
                "metallicRoughnessTexture",
                None
            )

            if mr_img is not None:
                if not isinstance(
                    mr_img,
                    Image.Image
                ):
                    mr_img = Image.fromarray(
                        np.asarray(mr_img)
                    )

                mr_img = mr_img.convert("RGB")

                metallic_roughness_texture_obj = (
                    ctx.texture(
                        mr_img.size,
                        3,
                        mr_img.tobytes()
                    )
                )

                metallic_roughness_texture_obj.filter = (
                    moderngl.LINEAR,
                    moderngl.LINEAR
                )

                metallic_roughness_texture_obj.repeat_x = True
                metallic_roughness_texture_obj.repeat_y = True

        # ---------------------------------------------------------
        # Vertex colors
        # ---------------------------------------------------------

        colors = np.tile(
            base_color,
            (len(vertices), 1)
        ).astype("f4")

        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(vertices)
        ):
            v_cols = np.asarray(
                mesh.visual.vertex_colors[:, :3],
                dtype="f4"
            )

            v_cols /= 255.0

            # Only use vertex colors when they actually contain
            # useful variation.
            if not np.allclose(
                v_cols,
                v_cols[0]
            ):
                colors = v_cols

        # ---------------------------------------------------------
        # Create individual buffers
        #
        # Using separate buffers is intentional.
        # It means shadow shaders can use only in_position,
        # while PBR shaders can use all four attributes.
        # ---------------------------------------------------------

        if _has_attribute(
            prog,
            "in_position"
        ):
            vbo = ctx.buffer(
                vertices.tobytes()
            )

        if _has_attribute(
            prog,
            "in_normal"
        ):
            normal_vbo = ctx.buffer(
                normals.tobytes()
            )

        if _has_attribute(
            prog,
            "in_color"
        ):
            color_vbo = ctx.buffer(
                colors.tobytes()
            )

        if _has_attribute(
            prog,
            "in_uv"
        ):
            uv_vbo = ctx.buffer(
                uvs.tobytes()
            )

        ibo = ctx.buffer(
            faces.tobytes()
        )

        # ---------------------------------------------------------
        # Build VAO
        # ---------------------------------------------------------

        vao_content = []

        if vbo is not None:
            vao_content.append(
                (
                    vbo,
                    "3f",
                    "in_position"
                )
            )

        if normal_vbo is not None:
            vao_content.append(
                (
                    normal_vbo,
                    "3f",
                    "in_normal"
                )
            )

        if color_vbo is not None:
            vao_content.append(
                (
                    color_vbo,
                    "3f",
                    "in_color"
                )
            )

        if uv_vbo is not None:
            vao_content.append(
                (
                    uv_vbo,
                    "2f",
                    "in_uv"
                )
            )

        if not vao_content:
            raise RuntimeError(
                "Shader contains no recognized "
                "vertex attributes."
            )

        vao = ctx.vertex_array(
            prog,
            vao_content,
            ibo
        )

        return {
            "vao": vao,

            "vbo": vbo,
            "normal_vbo": normal_vbo,
            "color_vbo": color_vbo,
            "uv_vbo": uv_vbo,
            "ibo": ibo,

            "texture": texture_obj,

            "metallic_roughness_texture":
                metallic_roughness_texture_obj,

            "metallic": metallic,
            "roughness": roughness,

            "emissive": [
                float(emissive[0]),
                float(emissive[1]),
                float(emissive[2])
            ],

            "has_texture":
                1 if texture_obj is not None else 0,

            "has_metallic_roughness_texture":
                1
                if metallic_roughness_texture_obj is not None
                else 0
        }

    except Exception as e:
        print(
            f"[Error] Failed to parse model "
            f"{path}: {e}"
        )

        if vao is not None:
            try:
                vao.release()
            except Exception:
                pass

        for buffer in (
            vbo,
            normal_vbo,
            color_vbo,
            uv_vbo,
            ibo
        ):
            if buffer is not None:
                try:
                    buffer.release()
                except Exception:
                    pass

        if texture_obj is not None:
            try:
                texture_obj.release()
            except Exception:
                pass

        if metallic_roughness_texture_obj is not None:
            try:
                metallic_roughness_texture_obj.release()
            except Exception:
                pass

        return None