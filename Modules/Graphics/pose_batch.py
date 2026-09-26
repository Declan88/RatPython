"""
Fast skeletal pose evaluation, for many characters at once.

The reference implementation (Skeleton.compute_bone_matrices_multi) walks every joint of every
blended clip in Python: about a quarter of a millisecond per character, so ten players cost as
much as the rest of the frame. This does the same job in a handful of numpy operations that don't
grow with the number of characters:

  * Clips are BAKED once (per process, shared by every character using the same clip data) into
    dense per-frame tables of local translation / rotation / scale for all joints. Sampling a clip
    is then two table rows and a lerp - no per-joint work.
  * The blend (several clips folded together by weight, plus the crossfade from the previous
    clip) is done for every character in the batch together with vectorised lerp/slerp.
  * Local matrices and the joint hierarchy walk are batched matrix products over all characters,
    one per joint.

The result is a BoneMatrices - the same skinning palette, held as one numpy array, that behaves
like the list of glm.mat4 the rest of the engine expects (indexing gives a glm.mat4; the GPU
upload uses the array's bytes directly).

Only the common case is handled here - a plain (unmasked) clip blend on a skeleton whose clips
give every joint translation, rotation and scale. Anything else (upper-body splits, an active pose
snapshot, a partial clip) returns None from make_request and the caller uses the reference path.
"""

import math

import glm
import numpy as np

_BAKE_STEP = 1.0 / 60.0        # baked frames are at most this far apart (source keys are 30 fps)
_baked = {}                    # clip fingerprint -> _Baked or False (can't be baked)


class _Baked:
    __slots__ = ("frames", "step", "translation", "rotation", "scale")

    def __init__(self, frames, step, translation, rotation, scale):
        self.frames = frames          # number of intervals (tables have frames + 1 rows)
        self.step = step              # seconds between rows
        self.translation = translation   # (frames + 1, J, 3) float32
        self.rotation = rotation         # (frames + 1, J, 4) float32, x y z w
        self.scale = scale               # (frames + 1, J, 3) float32


def _fingerprint(clip, joint_count):
    first = clip.channels[0] if clip.channels else None
    probe = first.values[len(first.values) // 2] if first is not None and first.values else ()
    return (clip.name, joint_count, len(clip.channels), round(clip.duration, 6), tuple(round(v, 5) for v in probe))


def baked_clip(skeleton, name):
    """The baked tables for a clip of `skeleton` (built on first use, shared afterwards), or None
    if it can't be baked (unknown, empty, or missing a joint's translation/rotation/scale)."""
    clip = skeleton.animations.get(name)
    if clip is None or not clip.channels:
        return None
    joint_count = len(skeleton.joints)
    key = _fingerprint(clip, joint_count)
    cached = _baked.get(key)
    if cached is None:
        cached = _baked[key] = _bake(clip, joint_count) or False
    return cached or None


def _bake(clip, joint_count):
    frames = max(1, int(math.ceil(clip.duration / _BAKE_STEP))) if clip.duration > 0.0 else 1
    step = clip.duration / frames if clip.duration > 0.0 else 1.0
    translation = np.zeros((frames + 1, joint_count, 3), "f4")
    rotation = np.zeros((frames + 1, joint_count, 4), "f4")
    scale = np.ones((frames + 1, joint_count, 3), "f4")
    for k in range(frames + 1):
        t, r, s = clip.sample_pose(joint_count, k * step)
        for j in range(joint_count):
            if t[j] is None or r[j] is None or s[j] is None:
                return None            # a joint without all three channels: use the reference path
            translation[k, j] = (t[j].x, t[j].y, t[j].z)
            rotation[k, j] = (r[j].x, r[j].y, r[j].z, r[j].w)
            scale[k, j] = (s[j].x, s[j].y, s[j].z)
    return _Baked(frames, step, translation, rotation, scale)


class _SkeletonTables:
    """Per-skeleton constants for the hierarchy walk."""

    def __init__(self, skeleton):
        joints = skeleton.joints
        self.count = len(joints)
        self.parents = [j.parent_joint_index for j in joints]
        # Row-major math (M @ v with column vectors) - glm matrices are column-major, so a glm
        # matrix's bytes are the TRANSPOSE of the numpy array holding the same matrix.
        self.external = [_to_numpy(j.external_root_matrix) for j in joints]
        self.inverse_bind = np.stack([_to_numpy(j.inverse_bind_matrix) for j in joints])   # (J, 4, 4)
        self.in_order = all(p < i for i, p in enumerate(self.parents))
        # The joints grouped by depth in the hierarchy: everything at one depth can be multiplied
        # by its parent's world matrix in one batched product.
        depth = []
        for i, p in enumerate(self.parents):
            depth.append(0 if p == -1 else depth[p] + 1)
        self.levels = []
        for d in range(max(depth) + 1 if depth else 0):
            joints_here = [i for i, dd in enumerate(depth) if dd == d]
            parents_here = [self.parents[i] for i in joints_here]
            self.levels.append((np.array(joints_here), np.array(parents_here),
                                np.stack([self.external[i] for i in joints_here]) if d == 0 else None))


def _to_numpy(matrix):
    return np.frombuffer(matrix.to_bytes(), dtype="f4").reshape(4, 4).T.astype("f8")


def _tables(skeleton):
    cached = getattr(skeleton, "_pose_tables", None)
    if cached is None:
        cached = skeleton._pose_tables = _SkeletonTables(skeleton)
    return cached


class BoneMatrices:
    """A skinning palette (one 4x4 per joint) held as a numpy array; acts like the list of glm.mat4
    the rest of the engine expects."""
    __slots__ = ("_m", "_packed")

    def __init__(self, matrices, packed):
        self._m = matrices           # (J, 4, 4) row-major math matrices, float32
        self._packed = packed        # (MAX_BONES, 4, 4) float32 already transposed for the GPU

    def __len__(self):
        return len(self._m)

    def __bool__(self):
        return len(self._m) > 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self._m)))]
        return glm.mat4.from_bytes(np.ascontiguousarray(self._m[index].T).tobytes())

    def __iter__(self):
        return (self[i] for i in range(len(self._m)))

    def gpu_bytes(self):
        """The palette padded to MAX_BONES, in the layout the bone uniform buffer wants."""
        return self._packed.tobytes()


class LazyPose:
    """The final local (translation, rotation, scale) of every joint - what Skeleton.last_pose holds
    for pose snapshots - kept as arrays and turned into glm values only if someone iterates."""
    __slots__ = ("t", "r", "s")

    def __init__(self, t, r, s):
        self.t, self.r, self.s = t, r, s

    def __iter__(self):
        for i in range(len(self.t)):
            yield (glm.vec3(*map(float, self.t[i])),
                   glm.quat(float(self.r[i][3]), float(self.r[i][0]), float(self.r[i][1]), float(self.r[i][2])),
                   glm.vec3(*map(float, self.s[i])))

    def __len__(self):
        return len(self.t)


def structure_key(skeleton):
    """Skeletons that can share one batch: same joint hierarchy and bind pose (the same model)."""
    key = getattr(skeleton, "_pose_key", None)
    if key is None:
        joints = skeleton.joints
        key = skeleton._pose_key = (
            len(joints), tuple(j.parent_joint_index for j in joints),
            joints[0].inverse_bind_matrix.to_bytes() if joints else b"",
            joints[-1].inverse_bind_matrix.to_bytes() if joints else b"")
    return key


def make_request(skeleton, weighted, prev_name, prev_time, blend_weight):
    """A pose request for evaluate(), or None when the reference path must be used. `weighted` is
    Skeleton.compute_bone_matrices_multi's list of (clip name, time, weight)."""
    if getattr(skeleton, "_pose_snap", None) is not None or getattr(skeleton, "_pose_fast", True) is False:
        return None
    entries = []
    for name, time, weight in weighted:
        if weight > 0.0 and name in skeleton.animations:
            table = baked_clip(skeleton, name)
            if table is None:
                skeleton._pose_fast = False
                return None
            entries.append((table, time, weight))
    if not entries:
        return None
    prev = None
    if prev_name is not None and blend_weight < 1.0:
        table = baked_clip(skeleton, prev_name)
        if table is None:
            return None
        prev = (table, prev_time, blend_weight)
    return (skeleton, entries, prev)


def _rows(table, time):
    """(row a, row b, fraction) for sampling a baked clip at `time`: the two nearest baked rows."""
    pos = time / table.step if table.step > 0.0 else 0.0
    if pos <= 0.0:
        return 0, 0, 0.0
    if pos >= table.frames:
        last = table.frames
        return last, last, 0.0
    i = int(pos)
    return i, i + 1, pos - i


def _nlerp_rows(q0, q1, f):
    """Normalised lerp between baked rotation rows (adjacent rows are nearly identical, so this equals a
    slerp to well below what matters). q0, q1: (..., 4); f: (..., 1)."""
    sign = np.where((q0 * q1).sum(-1, keepdims=True) < 0.0, -1.0, 1.0)
    q = q0 + (q1 * sign - q0) * f
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _slerp(q0, q1, t):
    """Vectorised glm.slerp (shortest path; nearly parallel quaternions use a normalised lerp).
    q0, q1: (..., 4); t: broadcastable to (..., 1)."""
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    close = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.where(close, 1.0, np.sin(theta))
    w0 = np.where(close, 1.0 - t, np.sin((1.0 - t) * theta) / sin_theta)
    w1 = np.where(close, t, np.sin(t * theta) / sin_theta)
    q = q0 * w0 + q1 * w1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def evaluate(requests):
    """Poses every request (see make_request): returns a list of (BoneMatrices, LazyPose), one per
    request, in order. All requests must be for skeletons with the same joint structure (they are:
    one character model)."""
    R = len(requests)
    tables = _tables(requests[0][0])
    J = tables.count

    # 1. Collect every (request, clip, time) sample - blend entries and crossfade sources - as row
    # indices into the baked tables, and gather them all at once.
    ta, tb, qa, qb, sa, sb, frac = [], [], [], [], [], [], []
    owner = []          # request index of each sample
    first = [0] * R     # sample index of each request's first entry
    later = {}          # level -> (request indices, sample indices, blend factors)
    fades = []          # (request index, sample index, weight)
    for r, (skeleton, entries, prev) in enumerate(requests):
        total = sum(w for _, _, w in entries)
        running = 0.0
        for level, (table, time, w) in enumerate(entries):
            a, b, f = _rows(table, time)
            n = len(frac)
            ta.append(table.translation[a]); tb.append(table.translation[b])
            qa.append(table.rotation[a]); qb.append(table.rotation[b])
            sa.append(table.scale[a]); sb.append(table.scale[b])
            frac.append(f)
            running += w / total
            if level == 0:
                first[r] = n
            else:
                blend = (w / total) / running if running > 0.0 else 0.0
                lv = later.setdefault(level, ([], [], []))
                lv[0].append(r); lv[1].append(n); lv[2].append(blend)
        if prev is not None:
            table, time, weight = prev
            a, b, f = _rows(table, time)
            n = len(frac)
            ta.append(table.translation[a]); tb.append(table.translation[b])
            qa.append(table.rotation[a]); qb.append(table.rotation[b])
            sa.append(table.scale[a]); sb.append(table.scale[b])
            frac.append(f)
            fades.append((r, n, weight))
    f = np.array(frac, "f8")[:, None, None]
    TA = np.stack(ta).astype("f8"); QA = np.stack(qa).astype("f8"); SA = np.stack(sa).astype("f8")
    ST = TA + (np.stack(tb).astype("f8") - TA) * f
    SS = SA + (np.stack(sb).astype("f8") - SA) * f
    SQ = _nlerp_rows(QA, np.stack(qb).astype("f8"), f)

    # 2. Each request's pose = its first entry, with the later entries folded in level by level (see
    # Skeleton._sample_weighted for why the running weights work out this way).
    idx0 = np.array(first)
    T, Q, S = ST[idx0], SQ[idx0], SS[idx0]
    for level in sorted(later):
        reqs = np.array(later[level][0]); samples = np.array(later[level][1])
        b = np.array(later[level][2], "f8")[:, None, None]
        T[reqs] = T[reqs] + (ST[samples] - T[reqs]) * b
        Q[reqs] = _slerp(Q[reqs], SQ[samples], b)
        S[reqs] = S[reqs] + (SS[samples] - S[reqs]) * b
    # 3. Crossfade from the previous clip (frozen at the moment the transition began).
    if fades:
        reqs = np.array([r for r, _, _ in fades]); samples = np.array([n for _, n, _ in fades])
        w = np.array([w for _, _, w in fades], "f8")[:, None, None]
        T[reqs] = ST[samples] + (T[reqs] - ST[samples]) * w
        Q[reqs] = _slerp(SQ[samples], Q[reqs], w)
        S[reqs] = SS[samples] + (S[reqs] - SS[samples]) * w

    # Local matrices: translate * rotate * scale (glm's order).
    x, y, z, w = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]
    L = np.zeros((R, J, 4, 4), "f8")
    L[..., 0, 0] = (1 - 2 * (y * y + z * z)) * S[..., 0]
    L[..., 0, 1] = (2 * (x * y - z * w)) * S[..., 1]
    L[..., 0, 2] = (2 * (x * z + y * w)) * S[..., 2]
    L[..., 1, 0] = (2 * (x * y + z * w)) * S[..., 0]
    L[..., 1, 1] = (1 - 2 * (x * x + z * z)) * S[..., 1]
    L[..., 1, 2] = (2 * (y * z - x * w)) * S[..., 2]
    L[..., 2, 0] = (2 * (x * z - y * w)) * S[..., 0]
    L[..., 2, 1] = (2 * (y * z + x * w)) * S[..., 1]
    L[..., 2, 2] = (1 - 2 * (x * x + y * y)) * S[..., 2]
    L[..., :3, 3] = T
    L[..., 3, 3] = 1.0

    # Hierarchy walk: one batched product per joint (parents come first).
    W = np.empty_like(L)
    for joints_here, parents_here, external in tables.levels:
        if external is not None:           # roots: their (usually identity) external matrix
            W[:, joints_here] = external[None] @ L[:, joints_here]
        else:
            W[:, joints_here] = W[:, parents_here] @ L[:, joints_here]
    B = (W @ tables.inverse_bind[None]).astype("f4")           # (R, J, 4, 4)

    from Modules.Graphics.skeletal_shader import MAX_BONES     # (imported late: avoids a cycle)
    packed = np.tile(np.eye(4, dtype="f4"), (R, MAX_BONES, 1, 1))
    count = min(J, MAX_BONES)
    packed[:, :count] = B[:, :count].transpose(0, 1, 3, 2)

    return [(BoneMatrices(B[r], packed[r]), LazyPose(T[r], Q[r], S[r])) for r in range(R)]
