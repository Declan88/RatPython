"""
Gibs: when a player dies their body bursts into chunks (Assets/Models/Gibs) that fly
out, tumble and bounce off the level with blood spurting from each, then shrink
away after a few seconds so no physics objects stay around.

Cost: everything expensive happens once, in GibManager.__init__ - the chunks are
split out of the model, uploaded to the GPU and given collision shapes then. A death
afterwards is just a handful of pooled draw objects (sharing that GPU data), five
small Bullet bodies and their blood effects; nothing is uploaded, loaded or allocated
on the GPU, so it can't hitch. Live gibs cost one matrix per part per frame, and when
they expire the bodies go and the draw objects return to the pool.

    scene.gibs = GibManager(scene, particles)
    scene.gibs.spawn(feet_position, velocity)      # someone died here
    ...each frame, before particles.update():   scene.gibs.update(dt)
"""

from Modules.settings import settings
import math
import os
import random
import re
import shutil
import tempfile

import glm
import numpy as np
import trimesh

from Modules.Graphics.pbr_shader import invalidate_material_ubo
from Modules.Physics.physics_world import CollisionGroup
from Modules.Player.rat_colors import RAT_TINT_MASK_PATH

GIB_MODEL = "Assets/Models/Gibs/Gibs.glb"
GORE_SOUND = "Assets/Audio/Player/Gore.wav"

# The fur material of the gib model (the same one the player rat uses): the part of a gib that
# takes the player's fur colour, matches the player's shading and uses the same tint mask.
FUR_MATERIAL = "funnyrat.001"

LIFETIME = 6.0            # seconds a gib exists
SHRINK_TIME = 0.9         # ...of which the last is spent shrinking to nothing
MAX_LIVE_GIBS = 25        # more than this (several deaths at once) retires the oldest early
_MASS_PER_M3 = 60.0       # a chunk's mass from its bounding volume: light, so it flies


class _Part:
    """One chunk of the gib model: its draw objects (one per material), collision
    shape and where it sat in the assembled body."""

    def __init__(self, name, objects, shape, center, mass, size):
        self.name = name
        self.objects = objects      # ORIGINAL objects from Scene.load_prop_groups
        self.shape = shape
        self.center = center        # glm.vec3: chunk centre relative to the body's feet
        self.mass = mass
        self.size = size            # largest dimension, metres
        self.radius = size * 0.5 * 1.15   # bounding sphere, for culling
        self.free = []              # pooled sets of object copies, ready to use


class _Live:
    __slots__ = ("part", "objects", "body", "age", "position", "alive")

    def __init__(self, part, objects, body):
        self.part = part
        self.objects = objects
        self.body = body
        self.age = 0.0
        self.position = glm.vec3(0.0)
        self.alive = True


def _split_model(path):
    """({chunk name: [trimesh geometries recentred on the chunk]}, {name: centre}).
    The glb keeps every chunk in one pose, each as one mesh per material.

    Vertex normals are carried through by hand (rotated with each node's transform, flipped for a
    mirrored copy) and set explicitly: trimesh would otherwise recompute them on export - through
    scipy, which isn't installed, so it logged a traceback per mesh before falling back."""
    scene = trimesh.load(path)
    groups = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        source = scene.geometry[geometry_name]
        normals = np.asarray(source.vertex_normals, dtype="f8")      # as stored in the file
        geometry = source.copy()
        geometry.apply_transform(transform)
        rotation = np.asarray(transform, dtype="f8")[:3, :3]
        rotated = normals @ np.linalg.inv(rotation)                  # (R^-1)^T applied to row vectors
        rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
        name = re.sub(r"_[0-9a-f]{6}$", "", node)     # trimesh suffixes duplicate node names
        groups.setdefault(name, []).append((geometry, rotated))
    result = {}
    centers = {}
    for name, entries in groups.items():
        geometries = [g for g, _ in entries]
        low = np.min([g.bounds[0] for g in geometries], axis=0)
        high = np.max([g.bounds[1] for g in geometries], axis=0)
        # A chunk modelled as one HALF of the body (a mesh whose edge sits exactly on the
        # x = 0 mirror plane - the torso was exported without its mirror modifier applied)
        # is completed by mirroring it, so it isn't an open shell.
        if abs(high[0]) < 1e-4 or abs(low[0]) < 1e-4:
            for geometry, normals in list(entries):
                mirrored = geometry.copy()
                mirrored.vertices = mirrored.vertices * np.array([-1.0, 1.0, 1.0])
                mirrored.invert()      # mirroring flips the winding: turn the faces back out
                entries.append((mirrored, normals * np.array([-1.0, 1.0, 1.0])))
            geometries = [g for g, _ in entries]
            low = np.min([g.bounds[0] for g in geometries], axis=0)
            high = np.max([g.bounds[1] for g in geometries], axis=0)
        center = (low + high) / 2.0
        for geometry, normals in entries:
            geometry.apply_translation(-center)
            geometry.vertex_normals = normals      # last: transforms/invert above rewrite them
        result[name] = geometries
        centers[name] = center
    return result, centers


class GibManager:
    def __init__(self, scene, particles, model_path=GIB_MODEL, tint_mask_path=RAT_TINT_MASK_PATH):
        self.scene = scene
        self.particles = particles
        self.parts = []
        self.live = []
        self._tint_mask = None
        self._deaths = 0
        self._fur_overrides = {"specular_strength": 0}   # what the player rat's shading sets (see app.py)
        self._build(model_path)
        # The fur takes the player's colour through the same mask texture the rat uses
        # (the gib meshes share the rat's UV layout), uploaded once for all of them.
        self._tint_mask = scene._load_tint_mask(tint_mask_path)
        for part in self.parts:
            for obj in part.objects:
                if obj.get("material_name") == FUR_MATERIAL:
                    obj["tint_mask_texture"] = self._tint_mask
                    obj.update(self._fur_overrides)
                else:
                    obj["cast_shadow"] = False     # the small flesh cut faces add nothing to a shadow
                obj["ssr_depth"] = False           # nor are tumbling chunks worth reflecting
                obj["light_cell"] = 2.0            # lit from a lookup shared with the chunks around it

    def match_material(self, source_obj):
        """Makes the fur match another object's shading - the player's skeletal object:
        copies its metallic/roughness/specular/emissive/normal settings."""
        for key in ("metallic", "roughness", "specular_strength", "emissive", "normal_scale"):
            if key in source_obj:
                self._fur_overrides[key] = source_obj[key]
        for part in self.parts:
            for objects in [part.objects] + part.free:
                for obj in objects:
                    if obj.get("material_name") == FUR_MATERIAL:
                        obj.update(self._fur_overrides)
                        invalidate_material_ubo(obj)

    def _dress(self, objects, tint):
        """Puts the current fur colour (rgb 0-1, or None for the model's own) on a set of
        draw objects, rebuilding a material buffer only when the colour actually changed."""
        new = None if tint is None else tuple(float(c) for c in tint)
        for obj in objects:
            if obj.get("material_name") == FUR_MATERIAL:
                if obj.get("tint_color") != new:
                    obj["tint_color"] = new
                    invalidate_material_ubo(obj)
                obj["material_group"] = ("gib fur", new)      # same textures + material buffer
            else:
                obj["material_group"] = ("gib flesh",)

    # ---- one-time setup ----

    def _build(self, model_path):
        groups, centers = _split_model(model_path)
        folder = tempfile.mkdtemp(prefix="ratwar_gibs_")
        try:
            for name, geometries in groups.items():
                path = os.path.join(folder, f"{name}.glb")
                trimesh.Scene(geometries).export(path)
                objects = self.scene.load_prop_groups(path)
                if not objects:
                    continue
                points = np.concatenate([np.asarray(g.vertices, "f4") for g in geometries])
                extent = points.max(axis=0) - points.min(axis=0)
                shape = self.scene.physics.make_hull_shape(points)
                self.parts.append(_Part(
                    name, objects, shape, glm.vec3(*[float(x) for x in centers[name]]),
                    max(1.0, float(np.prod(extent)) * _MASS_PER_M3), float(extent.max())))
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    # ---- spawning ----

    def spawn(self, feet_position, velocity=(0.0, 0.0, 0.0), push=None, sound=True, tint=None):
        """Bursts a body standing at `feet_position` into gibs. velocity: the body's own
        velocity (the chunks carry some of it); push: an extra shove for all of them
        (a shot's direction). tint: the dead player's fur colour (rgb 0-1, None = the model's own).
        Also plays the gore sound and throws blood around."""
        if not self.parts:
            return
        feet = glm.vec3(feet_position)
        if not settings.gibs:
            self._cheap_death(feet, sound)
            return
        inherited = glm.vec3(velocity) * 0.6 + (glm.vec3(push) if push is not None else glm.vec3(0.0))
        physics = self.scene.physics
        centre_of_mass = feet + glm.vec3(0.0, 0.8, 0.0)

        group = []      # this death's gibs (their wounds share one blood effect)
        self._deaths += 1
        light_group = ("death", self._deaths)
        for part in self.parts:
            if len(self.live) >= MAX_LIVE_GIBS:
                self._retire(self.live[0])
            position = feet + part.center
            # Outward from the body's axis (random when the chunk is on it), upward, and
            # tumbling: light chunks (limbs) are thrown harder than the torso.
            outward = glm.vec3(part.center.x, 0.0, part.center.z) * 3.0
            theta = random.uniform(0.0, math.tau)
            outward += glm.vec3(math.cos(theta), 0.0, math.sin(theta))
            outward = glm.normalize(outward) if glm.length(outward) > 1e-4 else glm.vec3(1.0, 0.0, 0.0)
            light = 1.0 if part.mass < 5.0 else 0.6
            linear = (outward * random.uniform(2.5, 6.0) * light
                      + glm.vec3(0.0, random.uniform(3.5, 7.5) * (0.6 + 0.4 * light), 0.0) + inherited)
            angular = glm.vec3(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
            angular = glm.normalize(angular) * random.uniform(7.0, 16.0)
            rotation = glm.quat(glm.vec3(0.0))

            body = physics.add_dynamic_shape(
                part.shape, position, rotation, part.mass, CollisionGroup.GIB,
                restitution=0.38, friction=0.8, linear_damping=0.04, angular_damping=0.12,
                ccd_radius=part.size * 0.3, name="gib")
            physics.set_body_velocity(body, linear, angular)

            objects = part.free.pop() if part.free else [dict(o) for o in part.objects]
            self._dress(objects, tint)
            live = _Live(part, objects, body)
            live.position = glm.vec3(position)
            self._place(objects, part.radius, position, rotation, 1.0)
            self.live.append(live)
            group.append(live)

        # Added to the scene fur first, then flesh, so objects sharing a material draw one after
        # another (the renderer skips re-binding textures and material buffers between them).
        drawn = [obj for live in group for obj in live.objects]
        drawn.sort(key=lambda o: o.get("material_name") != FUR_MATERIAL)
        for obj in drawn:
            obj["light_group"] = light_group      # one lighting sample for the whole burst
            self.scene.add_prop_object(obj)

        # Blood spurts from every wound and trails behind the chunks as they fly: one effect
        # for the whole death, emitting from each gib's current position.
        def wounds(group=group):
            return [(g.position.x, g.position.y, g.position.z) for g in group if g.alive] or None
        self.particles.spawn(
            "gore_blood_spurt", centre_of_mass, anchors=wounds, follow_particles=False,
            inherit_velocity=0.6)

        self.particles.spawn("gore_blood_burst", centre_of_mass)
        self.particles.spawn("gore_blood_cloud", centre_of_mass)
        if sound:
            self.scene.sound_manager.add_sound(
                GORE_SOUND, centre_of_mass, volume=1.0, min_distance=6.0, max_distance=90.0,
                loop=False, falloff="inverse", muffle=True)

    def _cheap_death(self, feet, sound):
        """The death effect with gibs turned off: no bodies, no physics, no extra draws - just
        the blood burst and cloud and the gore sound."""
        centre = feet + glm.vec3(0.0, 0.8, 0.0)
        self.particles.spawn("gore_blood_burst", centre)
        self.particles.spawn("gore_blood_cloud", centre)
        if sound:
            self.scene.sound_manager.add_sound(
                GORE_SOUND, centre, volume=1.0, min_distance=6.0, max_distance=90.0,
                loop=False, falloff="inverse", muffle=True)

    @staticmethod
    def _place(objects, radius, position, rotation, scale):
        """Moves a gib's draw objects: one matrix (and one bounding sphere / box, for culling)
        shared by all of them."""
        matrix = glm.translate(glm.mat4(1.0), position) * glm.mat4_cast(rotation)
        if scale != 1.0:
            matrix = matrix * glm.scale(glm.mat4(1.0), glm.vec3(scale))
        r = radius * scale
        aabb = ((position.x - r, position.y - r, position.z - r), (position.x + r, position.y + r, position.z + r))
        sphere = (position, r)
        for obj in objects:
            obj["transform"] = matrix
            obj["position"] = position
            obj["aabb_world"] = aabb
            obj["shadow_sphere"] = sphere

    # ---- per frame ----

    def update(self, dt):
        """Moves the draw objects to their bodies, shrinks expiring gibs and retires
        finished ones. Call once a frame after the scene's physics stepped."""
        if not self.live:
            return
        physics = self.scene.physics
        expired = []
        for live in self.live:
            live.age += dt
            if live.age >= LIFETIME:
                expired.append(live)
                continue
            position, rotation = physics.get_body_pose(live.body)
            live.position = position
            remaining = LIFETIME - live.age
            scale = 1.0
            if remaining < SHRINK_TIME:
                t = remaining / SHRINK_TIME
                scale = t * t * (3.0 - 2.0 * t)
            self._place(live.objects, live.part.radius, position, rotation, scale)
        for live in expired:
            self._retire(live)

    def _retire(self, live):
        live.alive = False
        self.scene.physics.remove_body(live.body)
        for obj in live.objects:
            self.scene.remove_prop_object(obj)
        live.part.free.append(live.objects)
        try:
            self.live.remove(live)
        except ValueError:
            pass

    def prime(self, camera, tint=None):
        """Does everything a first death would otherwise do lazily - so it can't hitch: builds
        the draw-object pool and their material buffers, warms the gore sound (and its
        distance-muffled copies), and draws a set of gibs once. They spawn in front of `camera`
        into the back buffer (never presented) and are removed again before any physics step.
        Call once when the game starts, with the camera in the world."""
        self.scene.sound_manager.preload(GORE_SOUND, muffle=True)
        front = glm.normalize(glm.vec3(camera.front))
        self.spawn(glm.vec3(camera.position) + front * 3.0 - glm.vec3(0.0, 0.8, 0.0), sound=False, tint=tint)
        self.scene.render(camera, None)
        self.clear()
        self.particles.clear()

    def clear(self):
        """Removes every live gib at once (respawn, leaving the map)."""
        for live in list(self.live):
            self._retire(live)

    def destroy(self):
        """Frees the GPU data. The scene calls this before it releases its own objects."""
        self.clear()
        for part in self.parts:
            for objects in part.free:
                for obj in objects:
                    self._release_copy(obj)
            part.free.clear()
            for obj in part.objects:
                obj["tint_mask_texture"] = None      # one texture shared by all: freed once, below
                self.scene.release_prop_object(obj)
        self.parts.clear()
        if self._tint_mask is not None:
            self._tint_mask.release()
            self._tint_mask = None

    @staticmethod
    def _release_copy(obj):
        """A pooled copy shares the original's GPU data but grew its own material
        uniform buffers when first drawn - those are the only thing it owns."""
        for key in ("_material_ubo", "_material_ubo_untinted"):
            buffer = obj.get(key)
            if buffer is not None:
                try:
                    buffer.release()
                except Exception:
                    pass
                obj[key] = None
