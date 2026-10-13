"""
A decorative prop actor with two meshes (see self.meshes below, both
ActorMesh instances - the modular mesh system actor_base.py's own
docstring describes):

  - Cheesepile.glb: the real, VISIBLE, morphing wedge+plate. Its own
    collision_mask is CollisionGroup.BULLET - a cheap, one-time static
    collider snapshotting its REST pose, hit by weapon hit-scan only,
    never solid to the player's own movement (see CollisionGroup.
    BULLET's own docstring in physics_world.py for exactly why: an
    exact collider that actually TRACKED the morph animation live was
    measured directly and found not worth it - ~1.4ms per rebuild, a
    real fraction of a 60fps frame, for one cosmetic decoration).

  - Cheese_Col.glb: an invisible (visible=False - never rendered),
    hand-authored proxy cylinder, collision_mask=CollisionGroup.
    PLAYER_CLIP - solid for the player's own movement only, never hit
    by a weapon's shot.

Between the two: a player walks on the pile via the simple, stable
cylinder proxy, while a bullet fired at it registers a real per-
triangle hit against the actual wedge shape - neither one ever
collides with the OTHER'S intended target (confirmed via both bits'
own docstrings/CharacterController's and PhysicsWorld.raycast's own
default masks).

It also has its own non-solid "detector" ghost sphere (see load_shapes
below, via ActorBase.create_collision_shape - a hand-held primitive
shape, not a mesh, so it's unrelated to self.meshes) for the Cheese War
proximity callbacks. Place a node named "actor_cheesepile" (or
"actor_cheesepile.001", etc. for more than one) in a map's .glb and this
spawns there automatically.

"Cheese War"-exclusive: see update()'s own override below - this despawns
itself (ActorBase.destroy()) the moment it finds out the active game mode
isn't Cheese War, rather than ever really belonging to a Deathmatch (or any
other) match it happened to get placed in the same map for.
"""

from Modules.Physics.physics_world import CollisionGroup

from .actor_base import ActorBase, ActorMesh

# The exact display name Modules/GameModes/cheese_war.py's own CheeseWar.name
# is set to - compared against game_mode.name rather than isinstance-checking
# the CheeseWar class directly, so this file doesn't need to import anything
# from Modules.GameModes at all (an actor has no other reason to know that
# package exists).
_CHEESE_WAR_MODE_NAME = "Cheese War"

_DETECTOR_RADIUS = 6

# The shape key runs from -1 (pile full) to 1 (pile empty) - -1 is already
# PAST the "normal" 0-1 a shape key slider usually implies (Cheesepile.glb's
# own mesh.weights is authored at -1 already), and neither Scene.
# set_morph_weight nor morph_program's own GLSL clamp this in any way - see
# both docstrings. The pile's size itself lives in the game mode (CheeseWar).


class CheesePile(ActorBase):
    actor_id = "cheesepile"
    # The "detector" sphere below is a pure proximity/event print, not
    # anything needing frame-perfect response - checking it every 4th
    # frame (~15Hz at 60fps) instead of every single one cuts most of
    # the cost confirmed to be the actual cause of a measurable fps
    # drop (ActorBase.shape_check_interval's own docstring has the full
    # numbers: contactTest's cost scales with how many OTHER bodies
    # exist in the level - ~0.15ms/call on mainmap.glb's 88 separate
    # static colliders vs ~0.0002ms in an empty world - not with this
    # actor's own tiny sphere).
    shape_check_interval = 4
    # dynamic=True (-> Scene.add_dynamic_morph, not Scene.add_static) is
    # what keeps the visible wedge out of bake_static_lighting(), which
    # only ever iterates static_objects (see its own docstring) -
    # leaving this static was exactly what was wrong before: it got a
    # baked lightmap built for it like any other piece of immovable
    # level geometry, despite being a prop that's supposed to move.
    meshes = (
        ActorMesh(
            "Assets/Models/Cheese/Cheesepile.glb",
            visible=True,
            dynamic=True,
            has_morph=True,
            collision_mask=CollisionGroup.BULLET,
        ),
        ActorMesh(
            "Assets/Models/Cheese/Cheese_Col.glb",
            visible=False,
            collision_mask=CollisionGroup.PLAYER_CLIP,
        ),
    )

    def __init__(self, scene, position, rotation, scale):
        # Ids of every player currently inside the detector (local and
        # remote) - a set, so two overlapping players, one leaving,
        # still leaves the other counted.
        self._inside = set()
        super().__init__(scene, position, rotation, scale)
        # Starts at -1 explicitly (add_dynamic_morph's own default is
        # 0.0, matching an UN-morphed, no-key-applied rest pose, not the
        # -1 a full pile is meant to start at) - has to run AFTER
        # super().__init__ above, since that's what actually builds
        # self._morph_objs (set_morph_weight is a no-op before then).
        self.set_morph_weight(-1.0)

    def load_shapes(self):
        # blocks=False: a pure overlap detector, never physically collides
        # with anything - see create_collision_shape's own docstring for
        # why that's required for a shape meant to notice a player at
        # range without also pushing them.
        self.create_collision_shape(
            "detector",
            shape="sphere",
            radius=_DETECTOR_RADIUS,
            blocks=False,
            # PLAYER only: the local player's own ghost (the thing this
            # overlap test needs to find) carries exactly that bit, while
            # CharacterController's movement sweeps deliberately EXCLUDE it
            # - with the default ALL mask the sweep (sweepTestClosest does
            # hit ghosts) treated the whole detector sphere as a solid wall.
            # Remote players' hitboxes (HITBOX bit) aren't detected, which is
            # fine: Cheese War only ever credits the LOCAL player's own cheese.
            collision_mask=CollisionGroup.PLAYER,
            on_enter=self._on_player_enter,
            on_exit=self._on_player_exit,
        )

    def _on_player_enter(self, player_id):
        self._inside.add(player_id)

    def _on_player_exit(self, player_id):
        self._inside.discard(player_id)

    def update(self, dt, player_positions=(), game_mode=None):
        # game_mode is None very briefly before app.py's own request_start/
        # start_game flow has picked a real mode - deliberately NOT treated
        # as "despawn", since defaulting to "assume not Cheese War" there
        # would despawn every cheese pile for a frame or two on every single
        # map load, Cheese War included, before the real mode is known.
        if (
            game_mode is not None
            and getattr(game_mode, "name", None) != _CHEESE_WAR_MODE_NAME
        ):
            self.destroy()
            return
        if game_mode is not None:
            # Who's inside goes to the mode, which owns the cheese economy
            # (see CheeseWar) - this actor just shows its result.
            game_mode.report_pile_occupants(id(self), self._inside)
            # Shape key follows the shared pile: full (CHEESE_TOTAL) = -1,
            # empty = 1, linear between.
            self.set_morph_weight(1.0 - 2.0 * game_mode.pile_fraction)
        super().update(dt, player_positions, game_mode)
