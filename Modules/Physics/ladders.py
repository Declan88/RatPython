"""
Ladder volumes, after Source's func_ladder: an invisible box in the level that,
while the player is touching it and moving INTO it, takes over their movement -
no gravity, forward/back climbs up/down, strafing moves sideways along it, jump
leaps off backwards - until they let go, jump, reach the bottom or climb off the
top. Collision still applies throughout (CharacterController keeps using its
normal sweep), so a ladder never carries you through a wall.

A volume is authored in the map as an Empty named "ladder" (or "ladder.001",
"ladder_roof"...) - see gltf_actors.extract_ladder_markers - and registered with
PhysicsWorld.add_ladder. CharacterController reads PhysicsWorld.ladders each tick.
"""

import re

import glm

# What a map node must be called to be a ladder volume: "ladder", "ladder_", "ladder.001",
# "ladder_exit", "ladder-east"... (any text after a ".", "_" or "-"). Shared by the map-marker
# reader (gltf_actors.extract_ladder_markers) and the model loader, which keeps such nodes out of
# the rendered and collidable level - a volume is a marker, never visible or solid.
LADDER_NODE_RE = re.compile(r"^ladder(?:[._\-].*)?$", re.IGNORECASE)


def is_ladder_node(node_name):
    return bool(LADDER_NODE_RE.match(node_name or ""))


# Source's numbers (game/shared/gamemovement.cpp), in metres:
CLIMB_SPEED = 200.0 * 0.0254        # MAX_CLIMB_SPEED: up/down/sideways along a ladder
JUMP_OFF_SPEED = 270.0 * 0.0254     # the push away from the ladder when you jump off it
STEP_INTERVAL = 0.35                # STEPSOUNDTIME_ON_LADDER (350 ms) between rung sounds

# How directly you must be moving into the ladder to grab it (dot of your move direction with
# the direction into the ladder face): just brushing past one doesn't catch you.
GRAB_DOT = 0.5
# Climbing up past the top: once your feet are this close to the top of the volume you hop off
# onto whatever is up there (forward over the lip, a little up), and can't re-grab straight away.
TOP_DISMOUNT_REACH = 0.3
TOP_HOP_FORWARD = 2.0
TOP_HOP_UP = 3.0
REGRAB_COOLDOWN = 0.35
# While on a ladder the hull is eased out to sit flat against the face of the volume (up to this speed):
# grabbing at an edge, half over solid ground, would otherwise leave it unable to slide down past it.
STANDOFF_SPEED = 2.0


class LadderVolume:
    """An oriented box. rotation: glm.mat3 whose columns are the box's local X/Y/Z axes in world
    space; half_extents: its half size along those. The climbable face is the thin horizontal
    axis (the smaller of the local X and Z half extents) - on whichever side the player is."""

    def __init__(self, center, rotation, half_extents):
        self.center = glm.vec3(center)
        self.rotation = glm.mat3(rotation)
        self.half_extents = glm.vec3(half_extents)
        flat_x = glm.vec3(self.rotation[0].x, 0.0, self.rotation[0].z)
        flat_z = glm.vec3(self.rotation[2].x, 0.0, self.rotation[2].z)
        face = flat_x if self.half_extents.x <= self.half_extents.z else flat_z
        self.face_axis = glm.normalize(face) if glm.length(face) > 1e-6 else glm.vec3(0.0, 0.0, 1.0)
        self.face_half = min(self.half_extents.x, self.half_extents.z)     # half the volume's thickness along face_axis
        reach_y = sum(abs(self.rotation[i].y) * self.half_extents[i] for i in range(3))
        self.top = self.center.y + reach_y
        self.bottom = self.center.y - reach_y

    def overlaps(self, position, half_xz, half_y):
        """Whether a world-axis-aligned box (the player's hull: centre `position`, half size half_xz
        in X and Z, half_y in Y) touches this volume - the separating-axis test on the volume's own
        three axes (a hair generous near a rotated volume's corners, which is the safe direction)."""
        offset = glm.vec3(position) - self.center
        for i in range(3):
            axis = self.rotation[i]
            reach = half_xz * (abs(axis.x) + abs(axis.z)) + half_y * abs(axis.y)
            if abs(glm.dot(offset, axis)) > self.half_extents[i] + reach:
                return False
        return True

    def facing(self, position):
        """Horizontal unit vector pointing from the ladder out toward `position` - the side the
        player is climbing from (so one volume works from either side)."""
        normal = glm.vec3(self.face_axis)
        if glm.dot(glm.vec3(position) - self.center, normal) < 0.0:
            normal = -normal
        return normal
