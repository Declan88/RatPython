"""
Collision/physics backend for Scene: wraps Panda3D's Bullet bindings
(panda3d.bullet) - Panda3D is already an installed dependency here, so
this needs no new pip install, and gets a maintained physics engine
(rigid bodies, capsule sweeping, a ready-made kinematic character
controller) instead of hand-rolling AABB/capsule-vs-mesh tests. Used
standalone, without Panda3D's own renderer/ShowBase - a bare NodePath
tree just hosts the Bullet node attachments Panda3D needs for
transform tracking; nothing here is ever drawn by Panda3D.

COORDINATE SYSTEMS: this project's renderer (moderngl/glm) is Y-up.
BulletCharacterControllerNode's fall/step/jump bookkeeping is hardcoded
to Panda3D's native Z-up axis - it ignores both BulletWorld's gravity
direction and the capsule shape's own "up" parameter for that internal
logic (confirmed empirically, not documented anywhere). So this module
keeps physics space Z-up internally and converts every value crossing
the render/physics boundary through the to_physics_*/to_render_*
helpers below - nothing else in this codebase should touch a Panda3D
Point3/Vec3/Quat directly with a raw render-space value.

COLLISION FILTERING: each body gets one BitMask32 (CollisionGroup
flags, combinable with |) via its collision_mask parameter. Panda3D-
Bullet's default filter is a plain bitwise AND check between two
bodies' masks - not a separate "what am I" vs "what do I hit" pair -
so two bodies collide only if (a.mask & b.mask) != 0. Leaving
collision_mask at its default (CollisionGroup.ALL) means "collide with
everything"; narrow it on both sides of a pairing to carve out
exceptions (see add_static_box's docstring for a worked example).
"""

import glm
import trimesh
import numpy as np
from panda3d.core import NodePath, PandaNode, Point3, Vec3, BitMask32
from panda3d.core import Quat as PandaQuat
from panda3d.bullet import (
    BulletWorld, BulletRigidBodyNode, BulletBoxShape, BulletSphereShape,
    BulletTriangleMesh, BulletTriangleMeshShape, BulletConvexHullShape,
)

# Proper (determinant +1) rotation, +90 degrees about X, mapping render
# Y-up -> physics Z-up while preserving handedness.
_AXIS_SWAP = glm.angleAxis(glm.radians(90.0), glm.vec3(1.0, 0.0, 0.0))
_AXIS_SWAP_INV = glm.inverse(_AXIS_SWAP)

# doPhysics(dt, max_substeps, fixed_timestep) can only ever simulate up
# to max_substeps * fixed_timestep of time in one call, no matter how
# large dt is - PhysicsWorld.step() clamps dt to that ceiling so a
# frame-rate stutter makes physics briefly run in slow motion instead
# of silently falling further and further behind real time each
# subsequent frame (a "spiral of death").
_PHYSICS_MAX_SUBSTEPS = 10
_PHYSICS_FIXED_TIMESTEP = 1.0 / 120.0
_PHYSICS_MAX_STEP_DT = _PHYSICS_MAX_SUBSTEPS * _PHYSICS_FIXED_TIMESTEP


def to_physics_pos(v):
    r = _AXIS_SWAP * glm.vec3(v)
    return Point3(r.x, r.y, r.z)


def to_render_pos(p):
    r = _AXIS_SWAP_INV * glm.vec3(p.x, p.y, p.z)
    return glm.vec3(r.x, r.y, r.z)


def to_physics_vec(v):
    """For values where sign/direction matters - velocities, movement
    directions, plane normals. NOT for sizes (see to_physics_extent)."""
    r = _AXIS_SWAP * glm.vec3(v)
    return Vec3(r.x, r.y, r.z)


def to_render_vec(v):
    r = _AXIS_SWAP_INV * glm.vec3(v.x, v.y, v.z)
    return glm.vec3(r.x, r.y, r.z)


def to_physics_extent(v):
    """For pure magnitudes (box half-extents) - permutes axes like the
    others but skips the sign flip a real rotation would introduce,
    since a negative half-extent is meaningless to BulletBoxShape."""
    v = glm.vec3(v)
    return Vec3(abs(v.x), abs(v.z), abs(v.y))


def to_physics_quat(q):
    r = _AXIS_SWAP * q * _AXIS_SWAP_INV
    return PandaQuat(r.w, r.x, r.y, r.z)


def to_render_quat(pq):
    r = glm.quat(pq.getW(), pq.getI(), pq.getJ(), pq.getK())
    return _AXIS_SWAP_INV * r * _AXIS_SWAP


def _euler_to_quat(rotation):
    """rotation: glm.vec3 of radians, same XYZ order
    Scene._get_model_matrix applies (rotate X, then Y, then Z)."""
    rotation = glm.vec3(rotation)
    return (
        glm.angleAxis(rotation.x, glm.vec3(1.0, 0.0, 0.0)) *
        glm.angleAxis(rotation.y, glm.vec3(0.0, 1.0, 0.0)) *
        glm.angleAxis(rotation.z, glm.vec3(0.0, 0.0, 1.0))
    )


class CollisionGroup:
    """Combinable collision-mask bits (up to 32 available). Two bodies
    only collide if (a.collision_mask & b.collision_mask) != 0 - see
    the module docstring. ALL is the default for every add_* method:
    ordinary solid geometry never needs to think about this at all."""
    STATIC = BitMask32.bit(0)
    DYNAMIC = BitMask32.bit(1)
    PLAYER = BitMask32.bit(2)
    TRIGGER = BitMask32.bit(3)
    ALL = BitMask32.all_on()


def _load_mesh(model_path, scale):
    """Loads model_path independently through trimesh (cheap - no GPU
    upload, separate from model_loader.py's render-focused load) and
    returns (vertices, faces), vertices pre-scaled by `scale` if given."""
    mesh = trimesh.load(str(model_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype="f8")
    if scale is not None:
        vertices = vertices * np.asarray(glm.vec3(scale), dtype="f8")
    return vertices, np.asarray(mesh.faces, dtype="i4")


class PhysicsWorld:
    """One per Scene (see Scene.__init__'s self.physics). Owns the
    Bullet simulation for that scene's static/dynamic colliders and
    lets a CharacterController (character_controller.py) attach a
    player capsule into the same world."""

    def __init__(self, gravity=9.81):
        self._root = NodePath(PandaNode("physics_root"))
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -gravity))
        self._node_paths = []
        self._accumulator = 0.0
        self._pre_substep_callbacks = []

    def add_pre_substep_callback(self, callback):
        """callback(dt) is invoked once for every fixed-size physics
        substep this world runs (see step()) - dt is always exactly
        _PHYSICS_FIXED_TIMESTEP, never a variable render-frame dt. Used
        by CharacterController to run its Source-engine-style ground/
        air acceleration at a stable, framerate-independent tick rate -
        that math needs to happen every physics tick, not once per
        render frame, the same way Source's own movement code runs on
        the server's fixed tick rather than the client's render rate."""
        self._pre_substep_callbacks.append(callback)

    def remove_pre_substep_callback(self, callback):
        if callback in self._pre_substep_callbacks:
            self._pre_substep_callbacks.remove(callback)

    def step(self, dt):
        """Call once per frame (Scene.update already does this).

        Runs its own fixed-size accumulator loop instead of handing
        Bullet's doPhysics(dt, max_substeps, fixed_timestep) the raw
        variable per-frame dt directly. That looks like it should be
        fps-independent (Bullet documents doPhysics as internally
        sub-stepping to a fixed timestep for exactly this reason), but
        empirically it is NOT once dt drops below fixed_timestep - which
        happens at any framerate above ~120fps, i.e. as soon as vsync is
        off. Confirmed by direct measurement: walking for a fixed
        simulated 2 seconds at speed 3 covers the correct ~6.0 units at
        30/60fps, but only ~1.44 units at 500fps and ~0.14 units at
        5000fps - distance scales down roughly as 1/fps past the
        threshold, which is why disabling vsync made the player slower.
        Feeding doPhysics exactly one fixed_timestep-sized step at a
        time (never smaller) instead of a variable dt sidesteps whatever
        internal accounting causes that, and measured flat/correct
        across 30-5000fps in the same test."""
        self._accumulator = min(self._accumulator + dt, _PHYSICS_MAX_STEP_DT)
        while self._accumulator >= _PHYSICS_FIXED_TIMESTEP:
            for callback in self._pre_substep_callbacks:
                callback(_PHYSICS_FIXED_TIMESTEP)
            self.world.doPhysics(_PHYSICS_FIXED_TIMESTEP, 1, _PHYSICS_FIXED_TIMESTEP)
            self._accumulator -= _PHYSICS_FIXED_TIMESTEP

    def get_interpolation_alpha(self):
        """Fraction (0-1) of a fixed timestep that's elapsed since the
        last completed substep - i.e. how far real time has crept into
        the NEXT tick that hasn't been simulated yet. Above ~120fps
        (see step()'s docstring), most render frames land between two
        physics substeps, so anything reading a raw physics position
        every render frame sees it hold still for a frame or two and
        then jump - most visible at low speed, where each tick's actual
        movement is small next to that jump. CharacterController.
        get_position() blends between last tick's and this tick's
        position by this fraction to smooth that out (the standard
        fixed-timestep-with-interpolation fix - see e.g. Glenn
        Fiedler's "Fix Your Timestep!")."""
        return self._accumulator / _PHYSICS_FIXED_TIMESTEP

    # ---------------------------------------------------------------
    # STATIC COLLIDERS (mass = 0, never move - level geometry)
    # ---------------------------------------------------------------

    def add_static_mesh(self, model_path, position=None, rotation=None,
                         scale=None, collision_mask=CollisionGroup.ALL,
                         exclude_local_bounds=None):
        """Exact per-triangle collision built by loading model_path a
        second time through trimesh (cheap - no GPU upload, independent
        of model_loader.py's render-focused load), so level geometry
        collides exactly like it looks. Only valid for STATIC bodies -
        Bullet requires convex shapes for anything that moves.

        exclude_local_bounds: optional ((min_x,min_y,min_z), (max_x,
        max_y,max_z)) in the model's own untransformed vertex space -
        any triangle whose centroid falls inside this box is left out
        of the collision mesh entirely. For carving out a region that's
        getting its own simplified collider instead (e.g. a sloped
        add_static_box standing in for a staircase too fine-grained for
        the player hull to climb step-by-step - see torus_scene.py).
        Leaving the fine mesh triangles in there TOO, underneath/around
        that simplified collider, was confirmed to cause erratic
        movement (a step-up/settle sweep landing on whichever of the two
        overlapping surfaces happens to be closest that tick, flipping
        surface normal and update-direction mid-slide) - this excludes
        them at the source instead of trying to make two independent
        colliders agree."""
        vertices, faces = _load_mesh(model_path, scale)

        if exclude_local_bounds is not None:
            mins, maxs = np.asarray(exclude_local_bounds[0]), np.asarray(exclude_local_bounds[1])
            centroids = vertices[faces].mean(axis=1)
            inside = np.all((centroids >= mins) & (centroids <= maxs), axis=1)
            faces = faces[~inside]

        tri_mesh = BulletTriangleMesh()
        # glTF exports (especially flat-shaded ones) commonly duplicate a
        # vertex's POSITION once per adjacent face so each triangle can
        # carry its own normal - so two triangles that are visually
        # coplanar and share an edge often don't share a vertex index at
        # all in the raw buffer trimesh hands back. Without welding here,
        # Bullet has no way to know those triangles are connected, so it
        # treats their shared edge as an "internal edge" a sweep can snag
        # on (spurious edge/vertex normal instead of the flat face normal)
        # - confirmed as the cause of walking jitter on flat multi-
        # triangle surfaces (see this module's docstring, and torus_
        # scene.py's plane.glb/floorbase.glb comments). remove_duplicate_
        # vertices=True + a tiny welding distance merges any positions
        # that coincide to within floating-point noise back into shared
        # indices, restoring the adjacency info Bullet needs.
        tri_mesh.setWeldingDistance(1e-8)
        for a, b, c in faces:
            tri_mesh.addTriangle(
                to_physics_pos(vertices[a]),
                to_physics_pos(vertices[b]),
                to_physics_pos(vertices[c]),
                True,
            )

        shape = BulletTriangleMeshShape(tri_mesh, dynamic=False)
        return self._add_body(shape, 0.0, position, rotation, collision_mask, "static_mesh")

    def add_static_box(self, half_extents, position=None, rotation=None,
                        collision_mask=CollisionGroup.ALL):
        """Cheaper approximate collider - an oriented box. Useful for
        simple blocking volumes (invisible walls, primitive-shaped
        props) where exact per-triangle collision isn't needed.

        Example - a trigger-style volume only the player should enter,
        that other dynamic debris passes straight through:
            physics.add_static_box(
                (2, 3, 2), position=(0, 1.5, 0),
                collision_mask=CollisionGroup.TRIGGER | CollisionGroup.PLAYER,
            )
        and give the player capsule collision_mask=CollisionGroup.ALL
        (the default) so it still shares the TRIGGER bit; ordinary
        debris left at CollisionGroup.ALL shares STATIC/DYNAMIC/PLAYER
        bits but not TRIGGER's, so (debris.mask & box.mask) == 0."""
        shape = BulletBoxShape(to_physics_extent(half_extents))
        return self._add_body(shape, 0.0, position, rotation, collision_mask, "static_box")

    # ---------------------------------------------------------------
    # DYNAMIC COLLIDERS (mass > 0 - pushed around by the simulation)
    # ---------------------------------------------------------------

    def add_dynamic_box(self, half_extents, position=None, rotation=None,
                         mass=1.0, collision_mask=CollisionGroup.ALL, gravity=True, kinematic=False):
        shape = BulletBoxShape(to_physics_extent(half_extents))
        return self._add_body(
            shape, mass, position, rotation, collision_mask, "dynamic_box",
            gravity=gravity, kinematic=kinematic,
        )

    def add_dynamic_sphere(self, radius, position=None, mass=1.0,
                            collision_mask=CollisionGroup.ALL, gravity=True, kinematic=False):
        shape = BulletSphereShape(float(radius))
        return self._add_body(
            shape, mass, position, None, collision_mask, "dynamic_sphere",
            gravity=gravity, kinematic=kinematic,
        )

    def add_dynamic_mesh(self, model_path, position=None, rotation=None,
                          scale=None, mass=1.0, collision_mask=CollisionGroup.ALL,
                          gravity=True, kinematic=False):
        """Convex-hull collision built from model_path's vertices -
        Bullet doesn't allow exact triangle-mesh shapes on moving
        bodies, so this is the closest a dynamic prop can get to its
        actual visual shape."""
        vertices, _ = _load_mesh(model_path, scale)

        shape = BulletConvexHullShape()
        for v in vertices:
            shape.addPoint(to_physics_pos(v))

        return self._add_body(
            shape, mass, position, rotation, collision_mask, "dynamic_mesh",
            gravity=gravity, kinematic=kinematic,
        )

    # ---------------------------------------------------------------
    # BOUNDS-DERIVED COLLIDERS (box/sphere sized automatically from a
    # model's own vertex bounds - convenient when a cheap collider just
    # needs to roughly match an object's size, without hand-measuring
    # half_extents/radius yourself)
    # ---------------------------------------------------------------

    def _bounds_center_and_half_extents(self, model_path, position, rotation, scale):
        vertices, _ = _load_mesh(model_path, scale)
        mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
        half_extents = glm.vec3((maxs - mins) / 2.0)
        local_center = glm.vec3((maxs + mins) / 2.0)

        rotated_center = _euler_to_quat(rotation) * local_center if rotation is not None else local_center
        world_center = (glm.vec3(position) if position is not None else glm.vec3(0.0)) + rotated_center
        return world_center, half_extents

    def add_static_box_from_bounds(self, model_path, position=None, rotation=None,
                                    scale=None, collision_mask=CollisionGroup.ALL):
        """Like add_static_box, but derives half_extents (and re-centers
        for an off-origin mesh) automatically from model_path's own
        vertex bounds."""
        center, half_extents = self._bounds_center_and_half_extents(model_path, position, rotation, scale)
        return self.add_static_box(half_extents, position=center, rotation=rotation, collision_mask=collision_mask)

    def add_dynamic_box_from_bounds(self, model_path, position=None, rotation=None, scale=None,
                                     mass=1.0, collision_mask=CollisionGroup.ALL, gravity=True, kinematic=False):
        """Like add_dynamic_box, but derives half_extents automatically
        from model_path's own vertex bounds."""
        center, half_extents = self._bounds_center_and_half_extents(model_path, position, rotation, scale)
        return self.add_dynamic_box(
            half_extents, position=center, rotation=rotation, mass=mass,
            collision_mask=collision_mask, gravity=gravity, kinematic=kinematic,
        )

    def add_dynamic_sphere_from_bounds(self, model_path, position=None, scale=None,
                                        mass=1.0, collision_mask=CollisionGroup.ALL, gravity=True, kinematic=False):
        """Like add_dynamic_sphere, but derives a radius automatically
        from model_path's own vertex bounds (the largest half-extent
        across the 3 axes - errs on the side of a slightly-too-big
        sphere for a non-cubic mesh rather than clipping through it)."""
        center, half_extents = self._bounds_center_and_half_extents(model_path, position, None, scale)
        radius = max(half_extents.x, half_extents.y, half_extents.z)
        return self.add_dynamic_sphere(
            radius, position=center, mass=mass, collision_mask=collision_mask,
            gravity=gravity, kinematic=kinematic,
        )

    # ---------------------------------------------------------------

    def _add_body(self, shape, mass, position, rotation, collision_mask, name, gravity=True, kinematic=False):
        node = BulletRigidBodyNode(name)
        node.addShape(shape)

        if kinematic:
            # A kinematic body is immovable by any physical force or
            # collision impulse - completely unlike gravity=False
            # (still a normal mass-having dynamic body that anything
            # can shove around, which is exactly why a "spin in place"
            # object built that way flew off the moment it touched
            # anything). Its transform is instead driven entirely by
            # the application every frame (see PhysicsWorld.
            # set_transform / Scene.update's handling of a _kinematic
            # object) - Bullet reads that transform for collision
            # purposes but never writes to it. Conventionally mass=0,
            # same as a static body, but flagged kinematic so Bullet
            # still treats it as capable of moving (a mass=0 body
            # without this flag is assumed permanently fixed).
            node.setMass(0.0)
            node.setKinematic(True)
            node.setDeactivationEnabled(False)
        else:
            node.setMass(mass)
            if mass > 0.0:
                # Dynamic bodies here are typically just a handful of
                # props, not enough to need sleep/wake bookkeeping -
                # always simulating avoids "why did that stop
                # responding" surprises.
                node.setDeactivationEnabled(False)
        node.setIntoCollideMask(collision_mask)

        node_path = self._root.attachNewNode(node)
        node_path.setPos(to_physics_pos(position) if position is not None else Point3(0, 0, 0))
        if rotation is not None:
            node_path.setQuat(to_physics_quat(_euler_to_quat(rotation)))

        self.world.attachRigidBody(node)

        if not kinematic and mass > 0.0 and not gravity:
            # A per-body override (BulletRigidBodyNode.setGravity), not
            # the world's own gravity - this body still has mass and
            # full contact response (things can push it, it can push
            # them), it just isn't pulled down by gravity. Meaningless
            # for mass=0 (static/kinematic) bodies, which gravity never
            # affects anyway. Set AFTER attachRigidBody, not before -
            # confirmed empirically that attachRigidBody overwrites any
            # gravity set beforehand with the world's own, silently
            # discarding a pre-attach override.
            node.setGravity(Vec3(0, 0, 0))

        self._node_paths.append(node_path)
        return node_path

    def get_transform(self, node_path):
        """Returns (position, rotation) in render space - glm.vec3
        position, glm.vec3 XYZ-radians rotation - call after step() to
        sync a dynamic Scene object's transform to wherever physics
        moved it (see Scene.update)."""
        return to_render_pos(node_path.getPos()), glm.eulerAngles(to_render_quat(node_path.getQuat()))

    def set_transform(self, node_path, position, rotation):
        """The inverse of get_transform() - pushes a render-space
        position/rotation (glm.vec3 position, glm.vec3 XYZ-radians
        rotation) ONTO a physics node_path, rather than reading physics
        state off of it. For a KINEMATIC body (see add_dynamic_box's
        kinematic parameter): unlike a normal dynamic body, whose
        transform the simulation itself owns and Scene.update() reads
        FROM every frame, a kinematic body's transform is driven by the
        application - Bullet reads it back out each step to know where
        the body is for collision purposes, but never writes to it
        itself. Call this every frame for a kinematic body instead of
        get_transform()."""
        node_path.setPos(to_physics_pos(position))
        node_path.setQuat(to_physics_quat(_euler_to_quat(rotation)))

    def destroy(self):
        for node_path in self._node_paths:
            node = node_path.node()
            if isinstance(node, BulletRigidBodyNode):
                try:
                    self.world.removeRigidBody(node)
                except Exception:
                    pass
            node_path.removeNode()
        self._node_paths.clear()
