"""Engine tests on a tiny water box, CPU platform (the GPU may be shared)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mdbench.config import load_defaults
from mdbench.engines.md import run_md, select_atoms


def _md(**kw):
    md = dict(load_defaults()["md"])
    md.update(platform="CPU", fallback_platform=None, minimize_max_iterations=200)
    md.update(kw)
    return md


def _energies(log: Path) -> np.ndarray:
    with log.open() as fh:
        rows = list(csv.DictReader(fh))
    key = next(k for k in rows[0] if "Total Energy" in k)
    return np.array([float(r[key]) for r in rows])


def test_nve_energy_conservation(tiny_water, tmp_path: Path):
    """Velocity Verlet, 1 fs, rigid water, PME: total-energy drift must stay small.

    Threshold: |linear drift| < 0.05 kT per degree of freedom per ns, and peak-to-peak
    total energy < 0.005 kT/dof over the run. Measured on this box with CPU PME
    (tolerance 1e-5), SETTLE and a 0.7-0.8 nm LJ switch: ~0.01 and ~0.0002.
    """
    top, system, pos = tiny_water
    import copy

    system = copy.deepcopy(system)
    md = _md(
        integrator="verlet",
        timestep_fs=1.0,
        constraint_tolerance=1e-8,
        equilibration_steps=200,
        n_steps=2000,
        log_interval_steps=50,
        traj_interval_steps=0,
        checkpoint_interval_steps=0,
    )
    res = run_md(top, system, pos, md, tmp_path / "nve")
    e = _energies(Path(res.log))
    t_ns = np.arange(1, len(e) + 1) * 50 * 1e-6
    slope = np.polyfit(t_ns, e, 1)[0]  # kJ/mol/ns
    n_dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    kt = 0.0083144626 * 300.0
    drift = abs(slope) / (n_dof * kt)
    assert drift < 0.05, f"NVE drift {drift:.4f} kT/dof/ns"
    assert np.ptp(e) / (n_dof * kt) < 0.005


def test_langevin_outputs_and_resume(tiny_water, tmp_path: Path):
    top, system, pos = tiny_water
    import copy

    md = _md(
        n_steps=200,
        traj_interval_steps=50,
        log_interval_steps=50,
        checkpoint_interval_steps=100,
        xtc_selection=list(range(9)),  # three waters stand in for a solute
        seed=1234,
    )
    rd = tmp_path / "lang"
    res = run_md(top, copy.deepcopy(system), pos, md, rd)
    assert res.gpu == "cpu" and res.platform == "CPU"
    assert res.wall_clock_s > 0 and res.n_steps == 200
    for name in ("traj.xtc", "md.log", "checkpoint.chk", "topology.pdb", "md_result.json"):
        assert (rd / name).exists(), name
    assert json.loads((rd / "md_result.json").read_text())["n_atoms"] == top.getNumAtoms()

    import MDAnalysis as mda

    u = mda.Universe(str(rd / "topology.pdb"), str(rd / "traj.xtc"))
    assert u.atoms.n_atoms == 9
    assert len(u.trajectory) == 4

    # Extend the same run directory: resumes from the checkpoint and appends.
    res2 = run_md(top, copy.deepcopy(system), pos, dict(md, n_steps=300), rd)
    assert res2.resumed_from_step == 200 and res2.n_steps == 100
    u = mda.Universe(str(rd / "topology.pdb"), str(rd / "traj.xtc"))
    assert len(u.trajectory) == 6


def test_solute_selection_falls_back(tiny_water):
    top, _, _ = tiny_water
    assert len(select_atoms(top, "solute")) == top.getNumAtoms()
    with pytest.raises(ValueError):
        select_atoms(top, "protein")


def test_plumed_hook(tiny_water, tmp_path: Path):
    pytest.importorskip("openmmplumed")
    top, system, pos = tiny_water
    import copy

    script = "d: DISTANCE ATOMS=1,4\nPRINT ARG=d STRIDE=10 FILE=COLVAR\n"
    md = _md(n_steps=50, traj_interval_steps=0, log_interval_steps=0, checkpoint_interval_steps=0)
    run_md(top, copy.deepcopy(system), pos, md, tmp_path / "plumed", plumed_script=script)
    colvar = (tmp_path / "plumed" / "COLVAR").read_text().splitlines()
    assert colvar[0].startswith("#! FIELDS")
    assert len([ln for ln in colvar if not ln.startswith("#")]) >= 5
