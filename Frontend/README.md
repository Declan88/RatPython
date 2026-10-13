# Frontend

A standalone **copy** of RatWar's graphics pipeline. It is not wired into the
game: nothing in `Modules/` or `app.py` imports from here, and nothing here
imports from `Modules/`. Edit it freely without touching the game.

Run from the project root (or anywhere `Frontend` is importable). Requires:
`moderngl`, `pygame-ce`, `PyGLM`, `numpy`, `Pillow`, `trimesh`, `panda3d`,
`pygltflib`; optional `OpenEXR` (HDR lightmap caches).

## Layout

| Package | What it does | Depends on |
|---|---|---|
| `loading/` | glTF meshes, morph targets, skeletal rigs + animations, `actor_<id>` marker extraction | `lighting` (UV generation) |
| `lighting/` | glTF punctual lights, baked lightmaps (+ UV generation, disk caches), light probes | `rendering` (shader uniform helper) |
| `shadows/` | `CascadedShadowMap` (sun), `PointShadowMap` | none |
| `rendering/` | PBR + skeletal shaders, skybox, frustum culling, `SceneTarget`, pose batching | none |
| `physics/` | Bullet `PhysicsWorld`, `CollisionGroup`, `CharacterController` | `loading`, `audio` |
| `camera/` | `Camera`, physics-aware `CameraBoomArm` | `physics` (boom arm only) |
| `window/` | `WindowManager` (window + GL context), loading screen | none |
| `audio/` | `SoundManager`, footstep material lookup | none |
| `actors/` | `ActorBase`, registry, `register_actor()` | `physics` |
| `scene/` | `Scene`: ties everything together (the renderer) | all of the above |
| `settings.py` | graphics options, stored in `~/.frontend/settings.json` | none |

`import Frontend` imports nothing eagerly, so import just the subpackage you need.

## Minimal use

```python
from Frontend.window import WindowManager
from Frontend.scene import Scene
from Frontend.camera import Camera

scene = Scene(window.ctx, recalculate_shadows=True)
scene.add_static("Assets/Models/torus.glb", collision=True)
scene.bake_static_lighting()
scene.update(dt)
scene.render(camera, None)
```

Subclass `Scene` for a level (see how `Modules/Scenes/torus_scene.py` does it).

## Adding your own map actors

The Frontend ships no actors. Subclass `ActorBase` in your own code and call
`register_actor(MyActor)` before loading a map whose `actor_<id>` markers
should spawn it.

## Things to know

- **Asset paths are relative to the working directory**: `Assets/...` for
  models, `Assets/Lightmaps/<SceneClass>` for baked lightmaps,
  `Assets/Lightmaps/_uv_cache`, `Assets/Physics/_collision_cache`, and
  `Assets/Audio/footsteps`. Run from a folder that has (or will get) those.
- **Left out on purpose** (game-specific, not pipeline): player paper-doll
  preview, bullet tracers, bullet-impact sounds, the Cheese Pile actor,
  gibs/particles, UI, networking.
- `Scene.update()` still accepts `game_mode` and `player_positions`; they are
  just passed through to actors.
- Some docstrings still mention `Modules/...` paths and game terms (weapons,
  gibs, viewmodels). They are comments inherited from the original, not
  dependencies.
