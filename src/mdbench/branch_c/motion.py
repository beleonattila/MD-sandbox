"""Export ala2 trajectories for the in-browser motion viewer.

Reads one unbiased MD replicate (``md_histogram``) and one metadynamics replicate
(``wtmetad_reweight``) from ``runs/`` and writes a single JSON file:

- ``md``: every frame of the solute, aligned on the backbone core (ACE C, ALA N CA CB C,
  NME N) so phi/psi rotation shows instead of tumbling. Coordinates are int16 in units
  of 0.01 Å, little-endian, base64; phi/psi are float32 radians from the run's ``cv.npz``.
  Bonds are guessed from frame-0 distances, because the topology PDB only has CONECT
  records for the HETATM caps.
- ``metad``: every hill from ``HILLS`` (time, phi, psi, height as stored by PLUMED, which
  for well-tempered runs already carries the gamma/(gamma-1) factor) and the COLVAR
  walker. COLVAR repeats rows after a restart; the last row per time is kept.

The output is a view of ``runs/``, not a result: it is regenerable and is written under
``runs/`` by default.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from mdbench.engines.plumed_cv import read_colvar

CORE_SELECTION = (
    "(resname ACE and name C) or (resname ALA and name N CA CB C) or (resname NME and name N)"
)
BOND_CUTOFF_A = {"heavy": 1.75, "hydrogen": 1.25}
COORD_SCALE = 100  # int16 units per Å


def b64(a: np.ndarray, dtype: str) -> str:
    """Base64 of ``a`` as a contiguous little-endian ``dtype`` buffer."""
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def guess_bonds(xyz: np.ndarray, elements: list[str]) -> list[tuple[int, int]]:
    """Bonded pairs by distance: X-H under 1.25 Å, heavy-heavy under 1.75 Å."""
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    out = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            kind = "hydrogen" if "H" in (elements[i], elements[j]) else "heavy"
            if d[i, j] < BOND_CUTOFF_A[kind]:
                out.append((i, j))
    return out


def last_per_time(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop rows whose ``time`` repeats (restarts), keeping the last; sort by time."""
    t = cols["time"]
    _, first_in_reversed = np.unique(t[::-1], return_index=True)
    keep = len(t) - 1 - first_in_reversed
    return {k: v[keep] for k, v in cols.items()}


def quantize(xyz: np.ndarray) -> np.ndarray:
    """Å to int16 centi-Å; refuses coordinates the format cannot hold."""
    q = np.round(xyz * COORD_SCALE)
    if np.abs(q).max() > np.iinfo(np.int16).max:
        raise ValueError("coordinates exceed the int16 range; is the solute unwrapped?")
    return q.astype("<i2")


def aligned_frames(topology: Path, trajectory: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    """All frames centred and rotated onto frame 0's backbone core; plus atom labels."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    u = mda.Universe(str(topology), str(trajectory))
    core = u.select_atoms(CORE_SELECTION)
    if core.n_atoms != 6:
        raise ValueError(f"backbone core selection found {core.n_atoms} atoms, expected 6")
    u.trajectory[0]
    ref = core.positions - core.center_of_geometry()
    out = np.empty((len(u.trajectory), u.atoms.n_atoms, 3), dtype=np.float32)
    for i, _ in enumerate(u.trajectory):
        c = core.center_of_geometry()
        rot, _ = align.rotation_matrix(core.positions - c, ref)
        out[i] = (u.atoms.positions - c) @ rot.T
    atoms = [
        {"n": a.name, "r": a.resname, "e": getattr(a, "element", "") or a.name[0]} for a in u.atoms
    ]
    return out, atoms


def md_block(rep_dir: Path) -> dict[str, Any]:
    """The ``md`` section from one ``md_histogram`` replicate directory."""
    xyz, atoms = aligned_frames(rep_dir / "topology.pdb", rep_dir / "traj.xtc")
    cv = np.load(rep_dir / "cv.npz")
    if len(cv["phi"]) != len(xyz):
        raise ValueError(f"{rep_dir}: {len(cv['phi'])} CV rows but {len(xyz)} frames")
    return {
        "n_frames": len(xyz),
        "dt_ps": float(cv["time_ps"][1] - cv["time_ps"][0]),
        "atoms": atoms,
        "bonds": guess_bonds(xyz[0], [a["e"] for a in atoms]),
        "xyz_i16_centiA": b64(quantize(xyz), "<i2"),
        "phi": b64(cv["phi"], "<f4"),
        "psi": b64(cv["psi"], "<f4"),
        "source": str(rep_dir),
    }


def metad_block(rep_dir: Path) -> dict[str, Any]:
    """The ``metad`` section from one ``wtmetad_reweight`` replicate directory."""
    hills = read_colvar(rep_dir / "HILLS")
    walker = last_per_time(read_colvar(rep_dir / "COLVAR"))
    return {
        "n_hills": len(hills["time"]),
        "t_ps": b64(hills["time"], "<f4"),
        "phi": b64(hills["phi"], "<f4"),
        "psi": b64(hills["psi"], "<f4"),
        "height": b64(hills["height"], "<f4"),
        "sigma": float(hills["sigma_phi"][0]),
        "biasf": float(hills["biasf"][0]),
        "cv_t": b64(walker["time"], "<f4"),
        "cv_phi": b64(walker["phi"], "<f4"),
        "cv_psi": b64(walker["psi"], "<f4"),
        "source": str(rep_dir),
    }


def export_motion(md_rep: Path, metad_rep: Path, out: Path) -> Path:
    """Write the viewer JSON for one MD replicate and one metad replicate."""
    data = {"md": md_block(md_rep), "metad": metad_block(metad_rep)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    return out


__all__ = ["export_motion", "guess_bonds", "last_per_time", "quantize"]
