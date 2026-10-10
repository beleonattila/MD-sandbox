"""Collective variables: backbone dihedrals, PLUMED input generation, COLVAR parsing.

Angles are radians internally, in ``[-pi, pi)``, with the IUPAC sign convention (the
same as PLUMED ``TORSION`` and MDAnalysis). Degrees only appear at the config/report
boundary.

Nothing here touches OpenMM state; it is pure numpy plus text, so it is cheap to test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Quad = tuple[int, int, int, int]


def wrap_angle(x: np.ndarray | float) -> np.ndarray:
    """Wrap angles (radians) into ``[-pi, pi)``."""
    a = np.asarray(x, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_deg(x: np.ndarray | float) -> np.ndarray:
    """Wrap angles (degrees) into ``[-180, 180)``."""
    a = np.asarray(x, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Minimum-image difference ``a - b`` of angles in radians, in ``[-pi, pi)``."""
    return wrap_angle(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def dihedral(xyz: np.ndarray) -> np.ndarray:
    """Dihedral angle(s) in radians for points of shape ``(..., 4, 3)``.

    IUPAC convention: looking along b->c, the angle from a to d, positive clockwise;
    ``[-pi, pi)``. Works on a single quadruplet or a stack of frames.
    """
    p = np.asarray(xyz, dtype=float)
    if p.shape[-2:] != (4, 3):
        raise ValueError(f"expected shape (..., 4, 3), got {p.shape}")
    b0 = p[..., 0, :] - p[..., 1, :]
    b1 = p[..., 2, :] - p[..., 1, :]
    b2 = p[..., 3, :] - p[..., 2, :]
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return wrap_angle(np.arctan2(y, x))


def dihedrals_from_frames(frames: np.ndarray, quads: Mapping[str, Quad]) -> dict[str, np.ndarray]:
    """Dihedral time series for each named quadruplet; ``frames`` is ``(n_frames, n_atoms, 3)``."""
    f = np.asarray(frames, dtype=float)
    if f.ndim == 2:
        f = f[None]
    return {name: dihedral(f[:, list(q), :]) for name, q in quads.items()}


def backbone_quads(atoms: Sequence[tuple[str, int, str]]) -> dict[str, Quad]:
    """phi/psi atom indices for a single central residue of a capped peptide.

    ``atoms`` is a sequence of ``(residue_name, residue_index, atom_name)`` in atom
    order. The central residue is the first residue that has N, CA and C (the caps
    ACE/NME have no CA). phi = C(i-1)-N-CA-C, psi = N-CA-C-N(i+1).
    """
    by_res: dict[int, dict[str, int]] = {}
    order: list[int] = []
    for idx, (_resn, resi, name) in enumerate(atoms):
        if resi not in by_res:
            by_res[resi] = {}
            order.append(resi)
        by_res[resi].setdefault(name, idx)
    for pos, resi in enumerate(order):
        r = by_res[resi]
        if {"N", "CA", "C"} <= r.keys():
            if pos == 0 or pos == len(order) - 1:
                raise ValueError("central residue must have a preceding and following residue")
            prev, nxt = by_res[order[pos - 1]], by_res[order[pos + 1]]
            if "C" not in prev or "N" not in nxt:
                raise ValueError("neighbouring residues lack C / N atoms")
            return {
                "phi": (prev["C"], r["N"], r["CA"], r["C"]),
                "psi": (r["N"], r["CA"], r["C"], nxt["N"]),
            }
    raise ValueError("no residue with N, CA and C found")


def topology_atoms(topology: Any) -> list[tuple[str, int, str]]:
    """``(resname, resindex, atomname)`` for every atom of an OpenMM Topology."""
    return [(a.residue.name, a.residue.index, a.name) for a in topology.atoms()]


# --------------------------------------------------------------------------- PLUMED


def _fmt(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, Sequence):
        return ",".join(_fmt(v) for v in x)
    return f"{float(x):.10g}" if isinstance(x, float) else str(x)


def torsion_lines(quads: Mapping[str, Quad]) -> list[str]:
    """PLUMED ``TORSION`` actions (PLUMED atom numbers are 1-based)."""
    return [f"{name}: TORSION ATOMS={','.join(str(i + 1) for i in q)}" for name, q in quads.items()]


def metad_script(
    quads: Mapping[str, Quad],
    params: Mapping[str, Any],
    temperature_K: float,
    colvar: str = "COLVAR",
    hills: str = "HILLS",
) -> str:
    """Well-tempered METAD on the given torsions, with ``CALC_RCT`` for reweighting.

    ``params`` keys: sigma_rad, height_kj_mol, pace_steps, biasfactor, grid_bin,
    colvar_stride_steps; optional rct_ustride. Printed fields: the CVs, ``metad.bias``
    and ``metad.rbias`` (bias minus c(t), the Tiwary-Parrinello reweighting factor).
    """
    names = list(quads)
    n = len(names)

    def per_cv(key: str) -> list[Any]:
        v = params[key]
        return list(v) if isinstance(v, Sequence) and not isinstance(v, str) else [v] * n

    lines = ["UNITS LENGTH=nm ENERGY=kj/mol TIME=ps", *torsion_lines(quads)]
    metad = [
        "metad: METAD",
        f"ARG={','.join(names)}",
        f"SIGMA={_fmt(per_cv('sigma_rad'))}",
        f"HEIGHT={_fmt(float(params['height_kj_mol']))}",
        f"PACE={int(params['pace_steps'])}",
        f"BIASFACTOR={_fmt(float(params['biasfactor']))}",
        f"TEMP={_fmt(float(temperature_K))}",
        f"GRID_MIN={','.join(['-pi'] * n)}",
        f"GRID_MAX={','.join(['pi'] * n)}",
        f"GRID_BIN={_fmt([int(b) for b in per_cv('grid_bin')])}",
        "CALC_RCT",
        f"RCT_USTRIDE={int(params.get('rct_ustride', 1))}",
        f"FILE={hills}",
    ]
    lines.append(" ".join(metad))
    lines.append(
        f"PRINT ARG={','.join(names)},metad.bias,metad.rbias "
        f"STRIDE={int(params['colvar_stride_steps'])} FILE={colvar}"
    )
    return "\n".join(lines) + "\n"


def read_colvar(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a PLUMED COLVAR file (``#! FIELDS`` header) into column arrays.

    Repeated headers (restarts) are tolerated as long as the fields do not change.
    """
    fields: list[str] | None = None
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#! FIELDS"):
            new = line.split()[2:]
            if fields is not None and new != fields:
                raise ValueError(f"{path}: COLVAR fields changed mid-file")
            fields = new
            continue
        if not line.strip() or line.startswith("#"):
            continue
        rows.append([float(v) for v in line.split()])
    if fields is None:
        raise ValueError(f"{path}: no '#! FIELDS' header")
    arr = np.asarray(rows, dtype=float).reshape(-1, len(fields))
    return {f: arr[:, i] for i, f in enumerate(fields)}


__all__ = [
    "angle_diff",
    "backbone_quads",
    "dihedral",
    "dihedrals_from_frames",
    "metad_script",
    "read_colvar",
    "topology_atoms",
    "torsion_lines",
    "wrap_angle",
    "wrap_deg",
]
