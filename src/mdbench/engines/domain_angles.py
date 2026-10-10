"""Adenylate kinase domain-angle collective variables (NMP-CORE and LID-CORE).

Definitions (E. coli AdK numbering), from Beckstein, Denning, Perilla & Woolf,
J. Mol. Biol. 394:160-176 (2009), doi:10.1016/j.jmb.2009.09.009, Figure 1 legend:

    "The NMP-CORE angle theta_NMP is formed by the centres of geometry of the backbone
    and C-beta atoms in residues 115-125 (CORE/LID), 90-100 (CORE), and 35-55 (NMP) of
    E. coli AdK. theta_LID is defined equivalently as the angle between 179-185 (CORE),
    115-125 (CORE/hinge/LID), and 125-153 (LID)."

The middle group is the vertex. "Backbone" is N, CA, C, O (MDAnalysis ``backbone``), so
hydrogens never enter and the CV is independent of protonation. Residue 125 belongs to
two LID-angle groups, exactly as in the published definition. The same selections are in
Beckstein's MDAnalysis course material (becksteinlab.physics.asu.edu, SimBioNano 2013).

Two implementations that must agree (see ``tests/test_adk_domain_angles.py``):

- :func:`domain_angles` / :func:`domain_angle_timeseries`: MDAnalysis, degrees.
- :func:`plumed_domain_angles`: a PLUMED input snippet (``CENTER`` + ``ANGLE``),
  radians, as PLUMED reports angles. Atom numbers are 1-based OpenMM particle indices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# Group -> residue range (inclusive). Vertex is "B".
NMP_GROUPS: dict[str, tuple[int, int]] = {"A": (115, 125), "B": (90, 100), "C": (35, 55)}
LID_GROUPS: dict[str, tuple[int, int]] = {"A": (179, 185), "B": (115, 125), "C": (125, 153)}
ANGLES: dict[str, dict[str, tuple[int, int]]] = {"theta_nmp": NMP_GROUPS, "theta_lid": LID_GROUPS}
ATOM_SELECTION = "protein and (backbone or name CB)"

CITATION_DOI = "10.1016/j.jmb.2009.09.009"


def group_selection(resids: tuple[int, int], extra: str | None = None) -> str:
    """MDAnalysis selection string for one centroid group."""
    sel = f"{ATOM_SELECTION} and resid {resids[0]}:{resids[1]}"
    return f"({sel}) and ({extra})" if extra else sel


def angle_from_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC (vertex ``b``) in degrees."""
    ba = np.asarray(a, float) - np.asarray(b, float)
    bc = np.asarray(c, float) - np.asarray(b, float)
    cosang = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def group_indices(universe: Any, extra: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    """0-based atom indices of every centroid group: ``{angle: {A|B|C: indices}}``.

    ``extra`` is ANDed into every selection (e.g. ``"segid A"`` or ``"chainID A"``).
    Raises ``ValueError`` if a group is empty or a residue in it lacks backbone atoms.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, groups in ANGLES.items():
        out[name] = {}
        for key, rng in groups.items():
            ag = universe.select_atoms(group_selection(rng, extra))
            n_res = rng[1] - rng[0] + 1
            if ag.n_atoms == 0 or len(np.unique(ag.resids)) != n_res:
                raise ValueError(
                    f"{name} group {key} (resid {rng[0]}:{rng[1]}): found {ag.n_atoms} atoms "
                    f"in {len(np.unique(ag.resids))}/{n_res} residues"
                )
            out[name][key] = ag.indices.copy()
    return out


def domain_angles(universe: Any, extra: str | None = None) -> dict[str, float]:
    """theta_nmp and theta_lid (degrees) for the current frame of ``universe``."""
    idx = group_indices(universe, extra)
    pos = universe.atoms.positions
    return {
        name: angle_from_points(*(pos[g[k]].mean(axis=0) for k in ("A", "B", "C")))
        for name, g in idx.items()
    }


def domain_angles_from_positions(
    positions: np.ndarray, groups: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, float]:
    """Domain angles (degrees) from a raw (n_atoms, 3) array and precomputed groups."""
    pos = np.asarray(positions, float)
    return {
        name: angle_from_points(*(pos[np.asarray(g[k])].mean(axis=0) for k in ("A", "B", "C")))
        for name, g in groups.items()
    }


def domain_angle_timeseries(
    universe: Any,
    extra: str | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> np.ndarray:
    """Array (n_frames, 3): time in ps, theta_nmp, theta_lid (degrees).

    Centroids are taken from the coordinates as stored; trajectories must have whole
    molecules (OpenMM XTC output does, since positions are not wrapped per atom).
    """
    idx = group_indices(universe, extra)
    rows = []
    for ts in universe.trajectory[start:stop:step]:
        ang = domain_angles_from_positions(universe.atoms.positions, idx)
        rows.append((float(ts.time), ang["theta_nmp"], ang["theta_lid"]))
    return np.asarray(rows, float).reshape(-1, 3)


def _plumed_atoms(indices: Sequence[int]) -> str:
    """Compress 0-based indices into PLUMED 1-based ranges, e.g. ``10-20,25``."""
    one = sorted(int(i) + 1 for i in indices)
    parts: list[str] = []
    start = prev = one[0]
    for i in one[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(parts)


def plumed_domain_angles(
    groups: Mapping[str, Mapping[str, Sequence[int]]],
    labels: Mapping[str, str] | None = None,
    nopbc: bool = False,
    print_stride: int | None = None,
    colvar_file: str = "COLVAR",
) -> str:
    """PLUMED input defining the two domain angles (radians).

    ``groups`` is the output of :func:`group_indices` (0-based). The centres are
    geometric (``CENTER`` without ``MASS``), matching the published definition.
    ``labels`` maps ``theta_nmp``/``theta_lid`` to PLUMED labels (default: same names).
    With ``print_stride`` a ``PRINT`` line writes both CVs to ``colvar_file``.
    """
    labels = dict(labels or {k: k for k in ANGLES})
    flag = " NOPBC" if nopbc else ""
    lines = [
        "# AdK domain angles, Beckstein et al. 2009 (doi:10.1016/j.jmb.2009.09.009); radians",
    ]
    for name in ANGLES:
        lab = labels[name]
        for key in ("A", "B", "C"):
            lines.append(f"{lab}_{key}: CENTER ATOMS={_plumed_atoms(groups[name][key])}{flag}")
        lines.append(f"{lab}: ANGLE ATOMS={lab}_A,{lab}_B,{lab}_C{flag}")
    if print_stride:
        args = ",".join(labels[n] for n in ANGLES)
        lines.append(f"PRINT ARG={args} STRIDE={int(print_stride)} FILE={colvar_file}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ANGLES",
    "ATOM_SELECTION",
    "CITATION_DOI",
    "LID_GROUPS",
    "NMP_GROUPS",
    "angle_from_points",
    "domain_angle_timeseries",
    "domain_angles",
    "domain_angles_from_positions",
    "group_indices",
    "group_selection",
    "plumed_domain_angles",
]
