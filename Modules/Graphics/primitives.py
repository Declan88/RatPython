import moderngl
import numpy as np

def create_cube(ctx, prog):
    vertices = np.array([
        # Position          # Color / Normal placeholders
        -0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
         0.5, -0.5, -0.5,   0.0, 1.0, 0.0,
         0.5,  0.5, -0.5,   0.0, 0.0, 1.0,
        -0.5,  0.5, -0.5,   1.0, 1.0, 0.0,
        -0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
         0.5, -0.5,  0.5,   0.0, 1.0, 1.0,
         0.5,  0.5,  0.5,   1.0, 1.0, 1.0,
        -0.5,  0.5,  0.5,   0.0, 0.0, 0.0,
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 4, 7, 7, 3, 0,
        1, 5, 6, 6, 2, 1,
        3, 2, 6, 6, 7, 3,
        0, 1, 5, 5, 4, 0
    ], dtype=np.int32)

    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(indices.tobytes())

    # Safely check which attributes exist in the given shader program
    attributes = ['in_position']
    format_str = '3f'
    
    if 'in_color' in prog:
        attributes.append('in_color')
        format_str += ' 3f'
    elif 'in_normal' in prog:
        attributes.append('in_normal')
        format_str += ' 3f'

    return ctx.vertex_array(prog, [(vbo, format_str, *attributes)], ibo)