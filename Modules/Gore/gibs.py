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

import math
import os
import random
import re
import shutil
import tempfile

import glm
import numpy as np
import trimesh

from Modules.Physics.physics_world import CollisionGroup

GIB_MODEL = "Assets/Models/Gibs/Gibs.glb"
GORE_SOUND = "Assets/Audio/Player/Gore.wav"

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
        self.free = []              # pooled sets of object copies, ready to use


class _Live:
    __slots__ = ("part", "objects", "body", "age", "position", "effect")

    def __init__(self, part, objects, body):
        self.part = part
        self.objects = objects
        self.body = body
        self.age = 0.0
        self.position = glm.vec3(0.0)
        self.effect = None


def _split_model(path):
    """({chunk name: [trimesh geometries recentred on the chunk]}, {name: centre}).
    The glb keeps every chunk in one pose, each as one mesh per material."""
    scene = trimesh.load(path)
    groups = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        name = re.sub(r"_[0-9a-f]{6}$", "", node)     # trimesh suffixes duplicate node names
        groups.setdefault(name, []).append(geometry)
    centers = {}
    for name, geometries in groups.items():
        low = np.min([g.bounds[0] for g in geometries], axis=0)
        high = np.max([g.bounds[1] for g in geometries], axis=0)
        center = (low + high) / 2.0
        for g in geometries:
            g.apply_translation(-center)
        centers[name] = center
    return groups, centers


class GibManager:
    def __init__(self, scene, particles, model_path=GIB_MODEL):
        self.scene = scene
        self.particles = particles
        self.parts = []
        self.live = []
        self._build(model_path)

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

    def spawn(self, feet_position, velocity=(0.0, 0.0, 0.0), push=None, sound=True):
        """Bursts a body standing at `feet_position` into gibs. velocity: the body's own
        velocity (the chunks carry some of it); push: an extra shove for all of them
        (a shot's direction). Also plays the gore sound and throws blood around."""
        if not self.parts:
            return
        feet = glm.vec3(feet_position)
        inherited = glm.vec3(velocity) * 0.6 + (glm.vec3(push) if push is not None else glm.vec3(0.0))
        physics = self.scene.physics
        centre_of_mass = feet + glm.vec3(0.0, 0.8, 0.0)

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
            live = _Live(part, objects, body)
            live.position = glm.vec3(position)
            for obj in objects:
                self._place(obj, position, rotation, 1.0)
                self.scene.add_prop_object(obj)
            # Blood spurts from the wound and trails behind the chunk as it flies.
            live.effect = self.particles.spawn(
                "gore_blood_spurt", position, follow=lambda live=live: live.position,
                follow_particles=False, inherit_velocity=0.6)
            self.live.append(live)

        self.particles.spawn("gore_blood_burst", centre_of_mass)
        self.particles.spawn("gore_blood_cloud", centre_of_mass)
        if sound:
            self.scene.sound_manager.add_sound(
                GORE_SOUND, centre_of_mass, volume=1.0, min_distance=6.0, max_distance=90.0,
                loop=False, falloff="inverse", muffle=True)

    @staticmethod
    def _place(obj, position, rotation, scale):
        matrix = glm.translate(glm.mat4(1.0), position) * glm.mat4_cast(rotation)
        if scale != 1.0:
            matrix = matrix * glm.scale(glm.mat4(1.0), glm.vec3(scale))
        obj["transform"] = matrix
        obj["position"] = position

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
            position, _ = physics.get_transform(live.body)
            rotation = physics.get_body_quat(live.body)
            live.position = position
            remaining = LIFETIME - live.age
            scale = 1.0
            if remaining < SHRINK_TIME:
                t = remaining / SHRINK_TIME
                scale = t * t * (3.0 - 2.0 * t)
            for obj in live.objects:
                self._place(obj, position, rotation, scale)
        for live in expired:
            self._retire(live)

    def _retire(self, live):
        if live.effect is not None:
            live.effect.stop()
            live.effect.follow = None
        self.scene.physics.remove_body(live.body)
        for obj in live.objects:
            self.scene.remove_prop_object(obj)
        live.part.free.append(live.objects)
        try:
            self.live.remove(live)
        except ValueError:
            pass

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
                self.scene.release_prop_object(obj)
        self.parts.clear()

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
