"""
Base class for a map "actor" - a scripted, self-contained prop spawned from
a plain Empty node in a map's own .glb named "actor_<id>" (see
Modules/Graphics/gltf_actors.py and Scene.add_actors_from_glb) - the
marker node carries no geometry of its own; it's purely a spawn point. An
actor's own mesh/collision/logic all come from its own class, the same way
a weapon's class points at its own gun model rather than expecting the map
to contain one.

Subclass this (see cheesepile.py for a real example), set `meshes` to a
tuple of ActorMesh instances (see that class's own docstring - this is
the ONE place an actor declares what it loads, replacing what used to be
a flat model_path/dynamic/has_collision/collision_shape/collision_mask/
collision_model_path/has_morph/mass/gravity/kinematic set of attributes
with a small list: each entry is independently visible and/or collidable,
so adding a THIRD (or fourth, ...) mesh with its own purpose - a second
LOD, a different hit-scan proxy, whatever - is just another list entry,
not another ad-hoc pair of attributes bolted onto this class), and
register it by importing it in Modules/Actors/__init__.py (see
registry.py's own module docstring for why that import is required, not
optional, once this project is built with Nuitka/PyInstaller).
"""

import glm

from Frontend.physics.physics_world import CollisionGroup


class ActorMesh:
    """Describes ONE mesh an actor loads - see ActorBase's own `meshes`
    tuple, which is just a list of these. An actor's VISUAL side and its
    COLLISION side are independent here: a mesh can be visible-only (an
    ordinary rendered piece with no collider of its own), collision-only
    (visible=False - never rendered at all, e.g. a hand-authored,
    invisible proxy), or both at once.

    Collision is ALWAYS a separate, static, one-time PhysicsWorld.
    add_static_mesh/add_static_box_from_bounds call against this mesh's
    own path - regardless of whether the VISUAL side (if any) is
    dynamic or morphing. This is a deliberate choice, not a limitation:
    Bullet requires a CONVEX shape for anything that moves (so an exact
    concave per-triangle collider is static-only by construction
    either way - see PhysicsWorld.add_dynamic's own docstring), and
    rebuilding a collider every frame to track a LIVE animation (a
    morph target, say) was measured directly and found not worth it for
    a cosmetic deformation (~1.4ms per rebuild for a ~450-triangle mesh,
    a real fraction of a 60fps frame budget, for one single object) -
    see CheesePile's own Cheesepile.glb entry for exactly this case: its
    collision is a cheap, one-time snapshot of the mesh's REST pose,
    not something that tracks its own live squash/stretch.

    path: the glb to load from.

    visible: True (default) - also rendered. dynamic/has_morph below
    only matter when this is True.

    dynamic: only meaningful when visible=True. False (default): an
    ordinary immovable prop (Scene.add_static). True: Scene.add_dynamic
    - lets a FUTURE mesh (not used by anything yet) have real mass and
    fall under gravity; collision is never bundled into this call
    regardless (see collision_mask below, always a separate body).

    has_morph: only meaningful when visible=True and dynamic=True.
    Loads via Scene.add_dynamic_morph instead of add_dynamic - a glTF
    shape key (e.g. "Empty" on Cheesepile.glb) driven at runtime via
    ActorBase.set_morph_weight(weight).

    collision_mask: None (default) - no collision body at all from this
    mesh. Otherwise, the CollisionGroup bits a SEPARATE static collider
    built from this same path should carry - e.g. CollisionGroup.
    PLAYER_CLIP (solid for the player's own movement only, never a
    weapon hit-scan) or CollisionGroup.BULLET (the reverse: a real
    hit-scan target, never solid to movement) - combine with | for
    more than one. See both bits' own docstrings in physics_world.py.

    collision_shape: "mesh" (default, exact per-triangle - PhysicsWorld.
    add_static_mesh) or "box" (bounds-derived - add_static_box_from_
    bounds) - which PhysicsWorld call builds the collider above."""

    def __init__(self, path, visible=True, dynamic=False, has_morph=False,
                 collision_mask=None, collision_shape="mesh"):
        self.path = path
        self.visible = visible
        self.dynamic = dynamic
        self.has_morph = has_morph
        self.collision_mask = collision_mask
        self.collision_shape = collision_shape


class ActorBase:
    # True ONLY on this class itself - never inherited as True by a real
    # subclass unless it explicitly redeclares it (see registry.py's own
    # _discover, which checks cls.__dict__, not getattr, for exactly this
    # reason) - excludes ActorBase itself from the registry while letting
    # an intermediate abstract subclass (e.g. a shared base for several
    # similar actors) exist without becoming spawnable on its own.
    abstract = True

    # The id a map's "actor_<id>" marker node names this actor by - falls
    # back to snake_case of the class name (see registry.actor_id_of) if
    # left None, same convention weapon_id already uses.
    actor_id = None

    # Every mesh this actor loads - see ActorMesh's own docstring. Empty
    # by default (an actor with purely invisible logic - a trigger
    # volume, say - is a reasonable thing to want, same as before).
    meshes = ()

    # Footstep-sound tag (see footstep_materials.py) for every mesh's
    # own collider, same meaning as Scene.add_static's own
    # physical_material param. None falls back to the default material set.
    physical_material = None

    cast_shadow = True
    visible_in_color = True

    # How many update() calls between contactTest sweeps of this actor's
    # own self._shapes (see update() below) - 1 (default) checks every
    # single frame, same as always. Raise this on an actor whose shapes
    # are pure proximity/event detectors with no need for frame-perfect
    # response (most of them - see CheesePile's own 4, "~4 frames of
    # latency on a Cheese War proximity print is imperceptible"):
    # contactTest's own cost scales with how many OTHER bodies exist in
    # the physics world, not with this actor's own shape - confirmed
    # directly on mainmap.glb (88 separate per-object static colliders)
    # at ~0.15ms per call vs ~0.0002ms in an empty world, a ~700x
    # difference that showed up as a measurable, disproportionate (tens
    # of fps at this project's own very high uncapped framerates) per-
    # frame cost for checking a single small sphere once a frame.
    shape_check_interval = 1

    def __init__(self, scene, position, rotation, scale):
        self.scene = scene
        self.position = position
        self.rotation = rotation
        self.scale = scale
        # Every VISIBLE mesh's own rendered object(s), flattened across
        # every entry in self.meshes - a subclass reading self.objs
        # (e.g. to post-process visible_in_color per-object) never needs
        # to know how many separate meshes contributed to it.
        self.objs = []
        # Subset of self.objs that also has a morph target - what
        # set_morph_weight below actually drives (NOT self.objs as a
        # whole, since a non-morph VISIBLE mesh sharing this actor would
        # otherwise also get a (meaningless) weight write).
        self._morph_objs = []
        # Every collision body load_meshes built (one per ActorMesh with
        # collision_mask set, visible or not) - what destroy() tears
        # down. self.body is kept as a convenience alias to the FIRST
        # one, for simple single-mesh actors that don't want to dig
        # through self._collision_bodies themselves.
        self._collision_bodies = []
        self.body = None
        # Every collision shape this actor owns, by the id create_
        # collision_shape(shape_id, ...) was given - see its own docstring.
        # {shape_id: {"node_path", "blocks", "inside", "on_enter",
        # "on_exit"}}. "inside": the set of owner ids (see
        # create_collision_shape) overlapping THIS shape as of the last
        # update() call, so on_enter/on_exit only fire on the frame
        # something actually starts/stops overlapping, not every frame
        # it's merely still inside.
        self._shapes = {}
        # Counts update() calls - see shape_check_interval's own
        # docstring and update() below for how it's actually used.
        self._update_count = 0
        # Set by destroy() - see its own docstring. Scene.update's own
        # actor loop prunes any actor with this True from self.actors right
        # after calling update() on everything, so an actor that destroys
        # itself mid-update (e.g. cheesepile.py's own gamemode check) is
        # gone by the very next frame without Scene needing to know why.
        self._destroyed = False
        self.load_meshes()
        self.load_shapes()

    def load_meshes(self):
        """Loads every ActorMesh in self.meshes - see that class's own
        docstring for exactly what each field controls. Override this
        (instead of self.meshes) for a visual/collision combination it
        doesn't cover (a skeletal/animated visual, a hand-built
        collision shape unrelated to any mesh's own geometry, ...).

        visible_in_color/cast_shadow are NOT passed to add_static/
        add_dynamic itself - unlike add_skeletal, neither has such
        constructor params at all (confirmed directly - passing them to
        add_static raised a TypeError on every single actor spawn, which
        is why CheesePile never actually appeared, before this was
        fixed). Every returned object dict already defaults to both
        True when the key is simply absent, exactly matching this
        class's own defaults - so they're only set explicitly, as a
        post-hoc dict override (the same pattern add_static's own
        metallic/roughness overrides already use internally), when a
        subclass actually wants a non-default value."""
        for mesh in self.meshes:
            objs = []
            if mesh.visible:
                if mesh.has_morph:
                    objs = self.scene.add_dynamic_morph(
                        mesh.path, position=self.position, rotation=self.rotation, scale=self.scale,
                        physical_material=self.physical_material,
                    ) or []
                    self._morph_objs.extend(objs)
                elif mesh.dynamic:
                    obj = self.scene.add_dynamic(
                        mesh.path, position=self.position, rotation=self.rotation, scale=self.scale,
                        collision=False, physical_material=self.physical_material,
                    )
                    objs = [obj] if obj is not None else []
                else:
                    objs = self.scene.add_static(
                        mesh.path, position=self.position, rotation=self.rotation, scale=self.scale,
                        collision=False,
                    )
                if objs and (not self.visible_in_color or not self.cast_shadow):
                    for obj in objs:
                        obj["visible_in_color"] = self.visible_in_color
                        obj["cast_shadow"] = self.cast_shadow
                self.objs.extend(objs)

            if mesh.collision_mask is not None:
                if mesh.collision_shape == "box":
                    collider = self.scene.physics.add_static_box_from_bounds(
                        mesh.path, position=self.position, rotation=self.rotation, scale=self.scale,
                        collision_mask=mesh.collision_mask, material=self.physical_material,
                    )
                else:
                    collider = self.scene.physics.add_static_mesh(
                        mesh.path, position=self.position, rotation=self.rotation, scale=self.scale,
                        collision_mask=mesh.collision_mask, material=self.physical_material,
                    )
                self._collision_bodies.append(collider)
                if self.body is None:
                    self.body = collider

    def set_morph_weight(self, weight):
        """Drives every morph-target mesh's own shape-key weight
        (self._morph_objs - empty, a no-op, if no ActorMesh in self.
        meshes has has_morph=True, so a subclass's update() can call
        this unconditionally without checking first). Forwards straight
        to Scene.set_morph_weight - see its own docstring for why this
        is deliberately NOT clamped to [0, 1]."""
        if self._morph_objs:
            self.scene.set_morph_weight(self._morph_objs, weight)

    def create_collision_shape(self, shape_id, shape="sphere", radius=None, size=None,
                                position=None, blocks=False, collision_mask=None, owner=None,
                                on_enter=None, on_exit=None):
        """The ONE place an actor builds a hand-held (not mesh-based -
        see ActorMesh/load_meshes above for that) Bullet collision
        shape - every primitive shape an actor owns (a detector sphere,
        a solid blocker, several of either on the same actor) goes
        through this one call instead of hand-rolling Bullet code per
        actor. Every shape built here is tracked under shape_id in
        self._shapes, so this same actor can own several INDEPENDENT
        shapes at once, each with its own callbacks - shape_id is just
        a name this actor's own code picks, to tell them apart (passed
        back to on_enter/on_exit, so a shared callback can tell which
        shape fired it).

        shape: "sphere" (radius=...) or "box" (size=..., a glm.vec3 of full
        XYZ extents) - add more cases here, following the same pattern, as
        actors need them.

        position: world position this shape is centered on - defaults to
        self.position (most actors want their shape right where they
        spawned) but can be offset for a shape that shouldn't sit at the
        actor's own origin.

        blocks=False (default): this shape OVERLAPS only - a REAL
        non-solid Bullet ghost (attachGhost, never attachRigidBody), so it
        can NEVER physically block or push anything, only ever be found by
        a manual contactTest (see update() below and PhysicsWorld.
        add_sphere_trigger/add_hitbox's own docstrings - this is the exact
        same mechanism). blocks=True: this shape is SOLID - a real static
        rigid body that actually collides/blocks for real, in addition to
        still being found by the same contactTest (so a solid shape's
        on_enter/on_exit still fire - "when something hits it").

        collision_mask: which groups this shape overlaps/blocks (default
        CollisionGroup.ALL - matches everything regardless of the OTHER
        body's own mask, so a detector shape never misses an overlap, and a
        blocking shape blocks everything, purely because of its own
        blocks=True/False rather than needing a hand-picked mask).

        on_enter(other_owner)/on_exit(other_owner): called from update()
        below the frame something (identified by its "owner" python tag -
        see PhysicsWorld.add_hitbox/CharacterController's own owner params)
        starts/stops overlapping THIS shape specifically - not every frame
        it's merely still inside. Either can be left None if this shape has
        no use for one side of that."""
        pos = position if position is not None else self.position
        mask = collision_mask if collision_mask is not None else CollisionGroup.ALL
        if shape == "sphere":
            if blocks:
                node_path = self.scene.physics.add_dynamic_sphere_from_bounds(
                    self.meshes[0].path if self.meshes else None, position=pos, scale=glm.vec3(radius * 2.0),
                    mass=0.0, collision_mask=mask,
                )
            else:
                node_path = self.scene.physics.add_sphere_trigger(
                    radius, position=pos, owner=owner, collision_mask=mask,
                )
        elif shape == "box":
            if size is None:
                raise ValueError("create_collision_shape: shape='box' requires size (full XYZ extents)")
            if blocks:
                # add_static_box takes HALF-extents (PhysicsWorld's own
                # convention - see its own docstring) - size here is
                # documented as FULL extents, so halve it here rather
                # than leaking that distinction out to every caller.
                node_path = self.scene.physics.add_static_box(
                    glm.vec3(size) * 0.5, position=pos, rotation=self.rotation,
                    collision_mask=mask, material=self.physical_material,
                )
            else:
                raise ValueError("create_collision_shape: no box-ghost helper yet - "
                                  "add PhysicsWorld.add_box_trigger, following add_sphere_trigger's own shape")
        else:
            raise ValueError(f"create_collision_shape: unsupported shape {shape!r}")
        self._shapes[shape_id] = {
            "node_path": node_path, "blocks": blocks, "inside": set(),
            "on_enter": on_enter, "on_exit": on_exit,
        }
        return node_path

    def load_shapes(self):
        """Override to call create_collision_shape(...) for each collision
        shape this actor owns - a no-op by default (most actors have no
        use for one). See cheesepile.py for a real example (one detector
        sphere, on_enter/on_exit wired to its own print methods)."""

    def update(self, dt, player_positions=(), game_mode=None):
        """Call once per frame (Scene.update() does, for every actor in
        self.actors, generically). player_positions ([(player_id, glm.vec3
        position), ...]) and game_mode (the active Modules.GameModes.
        GameMode instance - see Scene.update's own docstring for both) are
        accepted for signature compatibility with that generic call and for
        any OTHER per-frame logic a subclass might want (see cheesepile.py's
        own override, which uses game_mode to self-destroy outside "Cheese
        War"), but NEITHER is used by the shape-overlap check below - that's
        answered by a real physics query (self._shapes, see
        create_collision_shape) instead of hand-rolled distance math
        against a position list, so it stays correct regardless of what
        shape a player's own collider actually is, without this class
        needing to know or care. A subclass overriding this for its own
        per-frame logic should normally call super().update(dt,
        player_positions, game_mode) too, unless it has no shapes at all.

        For every shape this actor owns (self._shapes - empty for most
        actors, the common case where load_shapes never called
        create_collision_shape): contactTest's against that shape's own
        node (the same immediate, on-demand query
        CharacterController._can_stand already prefers over the
        separately-cached, can-go-stale getOverlappingNodes() - see its own
        docstring), reads each contact's "owner" python tag to identify WHO
        is actually overlapping (a plain wall/floor contact has no such tag
        and is silently skipped), and fires that shape's own
        on_enter(owner)/on_exit(owner) exactly once each, on the frame an
        owner id actually starts/stops appearing among its contacts - not
        every frame it's inside/outside - by diffing against that shape's
        own "inside" set (last frame's), not the whole actor's.

        shape_check_interval > 1 (see its own class docstring) skips
        the contactTest sweep entirely on every call that isn't a
        multiple of it - still counted via self._update_count either
        way, so the actual check cadence stays even (every Nth call),
        not reset/drifting if an actor goes dormant and resumes later."""
        self._update_count += 1
        if self._shapes and self._update_count % self.shape_check_interval != 0:
            return
        for info in self._shapes.values():
            node_path = info["node_path"]
            if node_path is None:
                continue
            node = node_path.node()
            result = self.scene.physics.world.contactTest(node)
            now_inside = set()
            for i in range(result.getNumContacts()):
                contact = result.getContact(i)
                other = contact.getNode1() if contact.getNode0() == node else contact.getNode0()
                if other.hasPythonTag("owner"):
                    owner = other.getPythonTag("owner")
                    if owner is not None:
                        now_inside.add(owner)
            entered = now_inside - info["inside"]
            exited = info["inside"] - now_inside
            info["inside"] = now_inside
            if info["on_enter"]:
                for owner in entered:
                    info["on_enter"](owner)
            if info["on_exit"]:
                for owner in exited:
                    info["on_exit"](owner)

    def destroy(self):
        """Fully tears this actor down: releases every visible mesh's
        own objects (Scene.remove_dynamic_object for one loaded dynamic/
        morph, remove_static_object otherwise - tracked per-object via
        the "_physics_body"/dynamic-ness Scene itself already stamps on
        a dynamic object, so this doesn't need its OWN bookkeeping of
        which mesh built which entry), every collision body in self.
        _collision_bodies (one per ActorMesh with collision_mask set -
        PhysicsWorld.remove_body handles a rigid body either way) and
        every shape in self._shapes (same remove_body method - it
        handles a rigid body or a ghost either way, see its own
        docstring), then marks self._destroyed = True so Scene.update's
        own actor loop prunes this out of self.actors on the very next
        pass (see its own comment - NOT removed from that list here
        directly, to avoid mutating it while it's mid-iteration if this
        runs from inside an actor's own update()). Idempotent - a second
        call is a safe no-op, since everything it would touch is already
        empty/True. Override and call super().destroy() if a subclass
        allocates anything of its own beyond what's listed here."""
        if self._destroyed:
            return
        for obj in self.objs:
            remove = self.scene.remove_dynamic_object if obj in self.scene.dynamic_objects else self.scene.remove_static_object
            remove(obj)
        self.objs = []
        self._morph_objs = []
        for collider in self._collision_bodies:
            self.scene.physics.remove_body(collider)
        self._collision_bodies = []
        self.body = None
        for info in self._shapes.values():
            if info["node_path"] is not None:
                self.scene.physics.remove_body(info["node_path"])
        self._shapes = {}
        self._destroyed = True
