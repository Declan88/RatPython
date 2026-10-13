"""
Frontend: a standalone copy of the RatWar graphics pipeline.

Nothing here imports from Modules/ or app.py, and nothing in the game imports
from here - it is a copy, not a refactor. Each subpackage is independent of
the ones it doesn't list below, so import only what you need:

    loading    glTF models, morph targets, skeletal rigs/animations, actor markers
    lighting   punctual lights, baked lightmaps (+ UV generation/caches), light probes
    shadows    cascaded directional shadows, point-light shadows
    rendering  PBR + skeletal shaders, skybox, frustum culling, render targets, pose batching
    physics    Bullet collision world, collision groups, player character controller
    camera     camera + physics-aware third-person boom arm
    window     window/GL context + loading screen
    audio      positional sound, footstep materials
    actors     ActorBase + registry for map-placed props (register your own)
    scene      Scene: ties all of the above together (the "renderer")
    settings   graphics options (~/.frontend/settings.json)

Nothing is imported eagerly here, so `import Frontend` is free.
"""
