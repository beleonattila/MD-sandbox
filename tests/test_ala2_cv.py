"""Fast tests for the CV engine (dihedrals, atom selection, PLUMED text, COLVAR parsing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdbench import REPO_ROOT
from mdbench.engines import plumed_cv as cv


def _quad_with_angle(theta: float) -> np.ndarray:
    """Points a,b,c,d with dihedral ``theta`` (IUPAC) about the b->c (x) axis."""
    a = np.array([0.0, 1.0, 0.0])
    b = np.zeros(3)
    c = np.array([1.0, 0.0, 0.0])
    d = c + np.array([0.0, np.cos(theta), np.sin(theta)])
    return np.stack([a, b, c, d])


@pytest.mark.parametrize("deg", [-179.0, -120.0, -60.0, 0.0, 45.0, 90.0, 150.0])
def test_dihedral_known_geometry(deg: float) -> None:
    # Sign convention checked against MDAnalysis (same as PLUMED TORSION).
    from MDAnalysis.lib.distances import calc_dihedrals

    p = _quad_with_angle(np.radians(deg))
    ref = calc_dihedrals(*(p[i][None].astype(np.float64) for i in range(4)))[0]
    assert cv.dihedral(p) == pytest.approx(ref, abs=1e-6)  # MDAnalysis is ~float32-accurate
    assert abs(abs(np.degrees(cv.dihedral(p))) - abs(deg)) < 1e-6


def test_dihedral_vectorised_and_wrapped() -> None:
    angles = np.radians(np.linspace(-179, 179, 25))
    stack = np.stack([_quad_with_angle(t) for t in angles])
    out = cv.dihedral(stack)
    assert out.shape == (25,)
    assert np.all(out >= -np.pi) and np.all(out < np.pi)
    # Translation and rotation invariance.
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = stack @ q.T + rng.normal(size=3)
    assert np.allclose(cv.dihedral(moved), out * np.sign(np.linalg.det(q)), atol=1e-9)


def test_wrap_and_diff() -> None:
    assert cv.wrap_deg(190.0) == pytest.approx(-170.0)
    assert cv.wrap_deg(-180.0) == pytest.approx(-180.0)
    assert float(cv.angle_diff(np.radians(170), np.radians(-170))) == pytest.approx(np.radians(-20))


def test_backbone_quads_on_built_ala2() -> None:
    from openmm import app

    pdb = app.PDBFile(str(REPO_ROOT / "systems/ala2/ala2_tleap.pdb"))
    q = cv.backbone_quads(cv.topology_atoms(pdb.topology))
    assert q == {"phi": (4, 6, 8, 14), "psi": (6, 8, 14, 16)}
    # The leap-built chain is fully extended: phi, psi ~ 180.
    x = np.asarray(pdb.getPositions(asNumpy=True)._value)
    ang = cv.dihedrals_from_frames(x, q)
    assert abs(abs(np.degrees(ang["phi"][0])) - 180) < 5
    assert abs(abs(np.degrees(ang["psi"][0])) - 180) < 5


def test_metad_script_and_colvar_roundtrip(tmp_path: Path) -> None:
    quads = {"phi": (4, 6, 8, 14), "psi": (6, 8, 14, 16)}
    params = {
        "sigma_rad": 0.35,
        "height_kj_mol": 1.2,
        "pace_steps": 500,
        "biasfactor": 6.0,
        "grid_bin": 200,
        "colvar_stride_steps": 500,
    }
    s = cv.metad_script(quads, params, 300.0)
    assert "phi: TORSION ATOMS=5,7,9,15" in s
    assert "psi: TORSION ATOMS=7,9,15,17" in s
    assert "SIGMA=0.35,0.35" in s and "BIASFACTOR=6" in s and "TEMP=300" in s
    assert "CALC_RCT" in s and "metad.rbias" in s
    col = tmp_path / "COLVAR"
    col.write_text(
        "#! FIELDS time phi psi metad.bias metad.rbias\n"
        "#! SET min_phi -pi\n"
        " 0.0 -1.0 2.0 0.0 0.0\n"
        " 1.0 -1.1 2.1 0.5 0.2\n"
        "#! FIELDS time phi psi metad.bias metad.rbias\n"
        " 2.0 -1.2 2.2 0.9 0.3\n"
    )
    d = cv.read_colvar(col)
    assert list(d) == ["time", "phi", "psi", "metad.bias", "metad.rbias"]
    assert np.allclose(d["time"], [0, 1, 2])
    assert np.allclose(d["metad.rbias"], [0, 0.2, 0.3])


def test_plumed_driver_accepts_generated_script(tmp_path: Path) -> None:
    """The generated METAD input runs in this env's PLUMED (``plumed driver``, no MD)."""
    import shutil
    import subprocess

    exe = shutil.which("plumed")
    if exe is None:
        pytest.skip("plumed executable not on PATH")
    quads = {"phi": (0, 1, 2, 3), "psi": (1, 2, 3, 4)}
    params = {
        "sigma_rad": 0.35,
        "height_kj_mol": 1.2,
        "pace_steps": 2,
        "biasfactor": 6.0,
        "grid_bin": 50,
        "colvar_stride_steps": 1,
    }
    (tmp_path / "plumed.dat").write_text(cv.metad_script(quads, params, 300.0))
    rng = np.random.default_rng(1)
    frames = [rng.normal(scale=0.2, size=(5, 3)) for _ in range(5)]
    with (tmp_path / "traj.xyz").open("w") as fh:
        for f in frames:
            fh.write("5\n5.0 5.0 5.0\n")
            fh.writelines(f"X {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in f)
    subprocess.run(
        [exe, "--no-mpi", "driver", "--plumed", "plumed.dat", "--ixyz", "traj.xyz",
         "--timestep", "0.002"],
        cwd=tmp_path, check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    d = cv.read_colvar(tmp_path / "COLVAR")
    assert "metad.rbias" in d and len(d["time"]) == 5
    assert d["phi"][0] == pytest.approx(cv.dihedral(frames[0][[0, 1, 2, 3]]), abs=1e-4)
    assert (tmp_path / "HILLS").exists()
