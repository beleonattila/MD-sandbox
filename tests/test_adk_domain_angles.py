"""AdK domain angles: MDAnalysis and PLUMED implementations agree on both crystals."""

from __future__ import annotations

import contextlib
import gc
import math
from pathlib import Path

import numpy as np
import pytest

from mdbench import REPO_ROOT
from mdbench.engines import domain_angles as da

RAW = REPO_ROOT / "systems" / "adk" / "raw"
PREP = REPO_ROOT / "systems" / "adk" / "prepared"
STRUCTURES = ["4AKE_A", "4AKE_B", "1AKE_A", "1AKE_B"]


def _universe(path: Path):
    mda = pytest.importorskip("MDAnalysis")
    return mda.Universe(str(path))


def test_angle_from_points_right_angle():
    assert da.angle_from_points([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert da.angle_from_points([2, 0, 0], [1, 0, 0], [3, 0, 0]) == pytest.approx(0.0)


def test_plumed_atom_ranges_compress():
    assert da._plumed_atoms([0, 1, 2, 5, 7, 8]) == "1-3,6,8-9"
    assert da._plumed_atoms([4]) == "5"


@pytest.mark.parametrize("pdb_id", ["4AKE", "1AKE"])
def test_crystal_angles_raw(pdb_id: str):
    """Raw crystal chains: open 4AKE has theta_LID > 140 deg (Beckstein et al. 2009 text);
    closed 1AKE has both angles much smaller. Chains A and B agree within 2 deg."""
    u = _universe(RAW / f"{pdb_id}.pdb")
    a = da.domain_angles(u, "chainID A")
    b = da.domain_angles(u, "chainID B")
    for k in a:
        assert abs(a[k] - b[k]) < 2.0
    if pdb_id == "4AKE":
        assert a["theta_lid"] > 140.0 and a["theta_nmp"] > 65.0
    else:
        assert a["theta_lid"] < 115.0 and a["theta_nmp"] < 50.0


@pytest.mark.parametrize("name", STRUCTURES)
def test_prepared_matches_raw(name: str):
    """Preparation (H added, ligand/water removed) leaves the CV unchanged: no H in it."""
    pdb_id, chain = name.split("_")
    raw = da.domain_angles(_universe(RAW / f"{pdb_id}.pdb"), f"chainID {chain}")
    prep = da.domain_angles(_universe(PREP / f"{name}_protein.pdb"))
    for k in raw:
        assert prep[k] == pytest.approx(raw[k], abs=1e-3)


def _plumed_angles(path: Path, workdir: Path) -> dict[str, float]:
    """Evaluate the generated PLUMED snippet through openmm-plumed on the PDB coordinates."""
    openmm = pytest.importorskip("openmm")
    plumed = pytest.importorskip("openmmplumed")
    from openmm import app, unit

    pdb = app.PDBFile(str(path))
    groups = da.group_indices(_universe(path))
    script = da.plumed_domain_angles(groups, print_stride=1, colvar_file="COLVAR")
    script += "FLUSH STRIDE=1\n"  # do not depend on when the PLUMED object is destroyed
    system = openmm.System()
    for atom in pdb.topology.atoms():
        system.addParticle(atom.element.mass)
    force = plumed.PlumedForce(script)
    system.addForce(force)
    integ = openmm.VerletIntegrator(1e-6 * unit.picoseconds)
    workdir.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(workdir):
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(pdb.positions)
        integ.step(1)
        ctx.getState(getEnergy=True)
        del ctx, integ, force, system
        gc.collect()  # destroying the context finalises PLUMED and closes COLVAR
    lines = (workdir / "COLVAR").read_text().splitlines()
    fields = next(ln for ln in lines if ln.startswith("#! FIELDS")).split()[2:]
    first = next(ln for ln in lines if not ln.startswith("#")).split()
    vals = dict(zip(fields, map(float, first), strict=True))
    return {k: math.degrees(vals[k]) for k in ("theta_nmp", "theta_lid")}


@pytest.mark.parametrize("name", STRUCTURES)
def test_mdanalysis_and_plumed_agree(name: str, tmp_path: Path):
    path = PREP / f"{name}_protein.pdb"
    ref = da.domain_angles(_universe(path))
    got = _plumed_angles(path, tmp_path / name)
    for k in ref:
        # PDB coordinates are 0.001 A; PLUMED prints 8 significant digits in radians.
        assert got[k] == pytest.approx(ref[k], abs=1e-3), (k, got[k], ref[k])


def test_plumed_snippet_structure():
    groups = {
        n: {k: np.arange(3) + 10 * i for i, k in enumerate("ABC")}
        for n in ("theta_nmp", "theta_lid")
    }
    s = da.plumed_domain_angles(groups, print_stride=5)
    assert "theta_nmp: ANGLE ATOMS=theta_nmp_A,theta_nmp_B,theta_nmp_C" in s
    assert "theta_lid_B: CENTER ATOMS=11-13" in s
    assert "MASS" not in s  # geometric centres, as published
    assert s.strip().endswith("PRINT ARG=theta_nmp,theta_lid STRIDE=5 FILE=COLVAR")


def test_group_selection_uses_published_ranges():
    assert da.NMP_GROUPS == {"A": (115, 125), "B": (90, 100), "C": (35, 55)}
    assert da.LID_GROUPS == {"A": (179, 185), "B": (115, 125), "C": (125, 153)}
    assert da.group_selection((35, 55)) == "protein and (backbone or name CB) and resid 35:55"
