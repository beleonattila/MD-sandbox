"""In-between frames in internal coordinates (display only).

Reference implementation of the frame interpolation used by ``mdbench-motion.html``
(the JavaScript there mirrors this module; keep the two in step). Saved MD frames are
1 ps apart; drawing a straight-line blend of Cartesian positions between them shortens
bonds and collapses angles whenever a group rotates. Here each atom is instead placed
from a bond length, a bond angle and a dihedral relative to three already-placed atoms
(a Z-matrix over a spanning tree of the bond graph, rooted at one atom whose two bonded
neighbours fix the root frame), and those internal coordinates are blended:

- bond lengths and bond angles linearly;
- dihedrals along the shorter arc;
- atoms sharing the same three reference atoms (the H's of a CH3, an sp2 pair) as one
  rigid rotor: only the first sibling's dihedral is blended as a torsion, the others
  keep a blended offset from it. Blending each H separately lets them take arcs in
  opposite directions when a methyl turns ~60° between frames, which squeezed H-C-H
  angles to ~15° on the ala2 trajectory;
- the root frame's orientation by quaternion slerp and its origin linearly.

At u = 0 and u = 1 the input frames are reproduced exactly. The in-between structures
are not simulated; they are for drawing only and never feed a result.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

TAU = 2 * np.pi


@dataclass(frozen=True)
class ZMatrix:
    """Placement order for ``n_atoms`` atoms.

    ``root, root_a, root_b``: the root atom and two of its bonded neighbours.
    ``refs[k] = (j, a, b, c, s)``: atom ``j`` is placed at distance |j-a|, angle j-a-b
    and dihedral j-a-b-c; ``s`` is the row of its primary sibling (``k`` itself if first).
    """

    n_atoms: int
    root: int
    root_a: int
    root_b: int
    refs: tuple[tuple[int, int, int, int, int], ...]


@dataclass(frozen=True)
class Internals:
    """One frame in internal coordinates: root origin, orientation, root triangle, Z-matrix rows."""

    origin: np.ndarray  # (3,)
    quat: np.ndarray  # (4,) w, x, y, z
    d_a: float  # |root - root_a|
    d_b: float  # |root - root_b|
    theta: float  # angle root_a - root - root_b
    ic: np.ndarray  # (n_refs, 3): length, angle, dihedral (rad)


def build_zmatrix(
    n_atoms: int, bonds: list[tuple[int, int]], root: int, root_a: int, root_b: int
) -> ZMatrix:
    """Breadth-first spanning tree from ``root``; raises if the graph is disconnected."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    par = [-1] * n_atoms
    seen = [False] * n_atoms
    seen[root] = True
    order, queue = [root], deque([root])
    while queue:
        x = queue.popleft()
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                par[y] = x
                order.append(y)
                queue.append(y)
    if len(order) != n_atoms:
        raise ValueError(f"bond graph is disconnected: reached {len(order)} of {n_atoms} atoms")
    if par[root_a] != root or par[root_b] != root:
        raise ValueError("root_a and root_b must both be bonded to root")
    refs: list[tuple[int, int, int, int, int]] = []
    primary: dict[tuple[int, int, int], int] = {}
    for j in order:
        if j in (root, root_a, root_b):
            continue
        a = par[j]
        if a == root:
            b, c = root_a, root_b
        else:
            b = par[a]
            c = (root_b if a == root_a else root_a) if b == root else par[b]
        s = primary.setdefault((a, b, c), len(refs))
        refs.append((j, a, b, c, s))
    return ZMatrix(n_atoms, root, root_a, root_b, tuple(refs))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _frame(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray) -> np.ndarray:
    """Rows: unit b->c, in-plane normal, plane normal (the local frame at ``p_c``)."""
    bc = _unit(p_c - p_b)
    n = _unit(np.cross(p_b - p_a, bc))
    return np.array([bc, np.cross(n, bc), n])


def _wrap(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % TAU - np.pi


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    t = np.trace(m)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        return np.array(
            [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Shortest-path spherical interpolation of unit quaternions."""
    c = float(np.dot(q0, q1))
    if c < 0:
        q1, c = -q1, -c
    if c > 0.9995:
        q = q0 + (q1 - q0) * u
        return q / np.linalg.norm(q)
    w = np.arccos(c)
    return (np.sin((1 - u) * w) * q0 + np.sin(u * w) * q1) / np.sin(w)


def internals(zm: ZMatrix, xyz: np.ndarray) -> Internals:
    """Internal coordinates of one frame ``xyz`` (n_atoms, 3)."""
    o = xyz[zm.root]
    va, vb = xyz[zm.root_a] - o, xyz[zm.root_b] - o
    e1 = _unit(va)
    e2 = _unit(vb - e1 * np.dot(vb, e1))
    e3 = np.cross(e1, e2)
    d_a, d_b = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    theta = float(np.arccos(np.clip(np.dot(e1, vb) / d_b, -1, 1)))
    ic = np.empty((len(zm.refs), 3))
    for k, (j, a, b, c, _) in enumerate(zm.refs):
        loc = _frame(xyz[c], xyz[b], xyz[a]) @ (xyz[j] - xyz[a])
        r = np.linalg.norm(loc)
        ic[k] = r, np.arccos(np.clip(-loc[0] / r, -1, 1)), np.arctan2(loc[2], loc[1])
    return Internals(
        o.copy(), _quat_from_matrix(np.column_stack([e1, e2, e3])), d_a, d_b, theta, ic
    )


def blend(zm: ZMatrix, i0: Internals, i1: Internals, u: float) -> np.ndarray:
    """Cartesian structure at fraction ``u`` in [0, 1] between two frames' internals."""
    lerp = lambda a, b: a + (b - a) * u  # noqa: E731
    prim = np.array([r[4] for r in zm.refs], dtype=int)
    ph0, ph1 = i0.ic[:, 2], i1.ic[:, 2]
    ph_prim = ph0 + _wrap(ph1 - ph0) * u  # each row as its own torsion; used for primaries
    off0, off1 = _wrap(ph0 - ph0[prim]), _wrap(ph1 - ph1[prim])
    ph = np.where(
        prim == np.arange(len(prim)), ph_prim, ph_prim[prim] + off0 + _wrap(off1 - off0) * u
    )
    r, t = lerp(i0.ic[:, 0], i1.ic[:, 0]), lerp(i0.ic[:, 1], i1.ic[:, 1])

    loc = np.zeros((zm.n_atoms, 3))
    d_a, d_b, th = lerp(i0.d_a, i1.d_a), lerp(i0.d_b, i1.d_b), lerp(i0.theta, i1.theta)
    loc[zm.root_a] = d_a, 0, 0
    loc[zm.root_b] = d_b * np.cos(th), d_b * np.sin(th), 0
    for k, (j, a, b, c, _) in enumerate(zm.refs):
        f = _frame(loc[c], loc[b], loc[a])
        d2 = np.array(
            [
                -r[k] * np.cos(t[k]),
                r[k] * np.sin(t[k]) * np.cos(ph[k]),
                r[k] * np.sin(t[k]) * np.sin(ph[k]),
            ]
        )
        loc[j] = loc[a] + f.T @ d2
    rot = _matrix_from_quat(slerp(i0.quat, i1.quat, u))
    return lerp(i0.origin, i1.origin) + loc @ rot.T


def interpolate(zm: ZMatrix, xyz0: np.ndarray, xyz1: np.ndarray, u: float) -> np.ndarray:
    """Structure at fraction ``u`` between frames ``xyz0`` and ``xyz1`` (n_atoms, 3)."""
    return blend(zm, internals(zm, xyz0), internals(zm, xyz1), u)


def upsample(zm: ZMatrix, xyz: np.ndarray, factor: int) -> np.ndarray:
    """Insert ``factor - 1`` in-between structures per gap.

    (n_frames, n_atoms, 3) -> ((n_frames - 1) * factor + 1, n_atoms, 3); saved frames kept exactly.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    ics = [internals(zm, x) for x in xyz]
    out = [xyz[0]]
    for f in range(len(xyz) - 1):
        out.extend(blend(zm, ics[f], ics[f + 1], s / factor) for s in range(1, factor))
        out.append(xyz[f + 1])
    return np.array(out)


def angle_triples(n_atoms: int, bonds: list[tuple[int, int]]) -> np.ndarray:
    """Every bonded angle i-c-j as rows (i, c, j)."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return np.array(
        [(x, c, y) for c in range(n_atoms) for k, x in enumerate(adj[c]) for y in adj[c][k + 1 :]]
    )


def geometry(
    xyz: np.ndarray, bonds: np.ndarray, triples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bond lengths (Å) and bond angles (degrees) of one frame."""
    bl = np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=1)
    v1, v2 = xyz[triples[:, 0]] - xyz[triples[:, 1]], xyz[triples[:, 2]] - xyz[triples[:, 1]]
    cos = (v1 * v2).sum(1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
    return bl, np.degrees(np.arccos(np.clip(cos, -1, 1)))


def excursion(value: np.ndarray, end0: np.ndarray, end1: np.ndarray) -> np.ndarray:
    """How far ``value`` falls outside the range spanned by the two endpoint values (0 inside)."""
    lo, hi = np.minimum(end0, end1), np.maximum(end0, end1)
    return np.maximum(lo - value, 0) + np.maximum(value - hi, 0)


__all__ = [
    "Internals",
    "ZMatrix",
    "angle_triples",
    "blend",
    "build_zmatrix",
    "excursion",
    "geometry",
    "internals",
    "interpolate",
    "slerp",
    "upsample",
]
