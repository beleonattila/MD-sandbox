"""AdK branch C: preparation record, configs, estimators and the cheap runners."""

from __future__ import annotations

import contextlib
import gc
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from mdbench import REPO_ROOT
from mdbench.branch_c import adk
from mdbench.config import load_config
from mdbench.registry import get_runner

CONFIGS = REPO_ROOT / "configs"
RUN_CONFIGS = sorted(
    p for p in CONFIGS.glob("c_adk_*.yml") if p.stem not in ("c_adk_prep", "c_adk_base")
)
PREP_RECORD = REPO_ROOT / "systems" / "adk" / "prep.yml"


# ----------------------------------------------------------------- provenance and prep


@pytest.mark.parametrize("pdb_id", ["4AKE", "1AKE"])
def test_raw_entries_verify(pdb_id: str):
    prep = yaml.safe_load((CONFIGS / "c_adk_prep.yml").read_text())["prep"]
    found = adk.parse_pdb_header(REPO_ROOT / "systems" / "adk" / "raw" / f"{pdb_id}.pdb")
    checks = adk.verify_entry(found, prep["structures"][pdb_id]["expected"])
    assert checks["all_ok"], checks
    wrong = dict(prep["structures"][pdb_id]["expected"], organism_taxid=9606)
    assert not adk.verify_entry(found, wrong)["all_ok"]


def test_prep_record_checksums_match_files():
    rec = yaml.safe_load(PREP_RECORD.read_text())
    for s in rec["structures"].values():
        assert adk.sha256_file(REPO_ROOT / s["raw_file"]) == s["raw_sha256"]
        assert s["verification"]["all_ok"]
        for ch in s["chains"].values():
            assert adk.sha256_file(REPO_ROOT / ch["protein_pdb"]) == ch["protein_pdb_sha256"]
            assert ch["net_charge_e"] == pytest.approx(-4.0)
            if "solvated" in ch:
                sv = ch["solvated"]
                assert adk.sha256_file(REPO_ROOT / sv["pdb"]) == sv["pdb_sha256"]
                assert sv["net_charge_e"] == pytest.approx(0.0, abs=1e-3)
    assert rec["histidine_states_all_structures_identical"]
    for key in ("protonation_tool", "pdbfixer", "openmm"):
        assert rec["tools"][key]


def test_raw_checksum_file_matches():
    raw = REPO_ROOT / "systems" / "adk" / "raw"
    for line in (raw / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert adk.sha256_file(raw / name) == digest, name


# ----------------------------------------------------------------- configs


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.stem)
def test_configs_resolve_and_have_runner(path: Path):
    cfg = load_config(path)
    assert cfg["branch"] == "c" and cfg["system"] == "adk"
    get_runner(cfg["branch"], cfg["system"], cfg["level"], cfg["method"])
    for state in cfg["adk"]["endpoints"].values():
        for p in state.values():
            assert (REPO_ROOT / p).exists()
    assert (REPO_ROOT / cfg["adk"]["solvated_pdb"]).exists()


# ----------------------------------------------------------------- estimators


def test_jeffreys_floor_nonzero():
    assert adk.jeffreys_sd(0, 5) == pytest.approx(math.sqrt(0.5 * 5.5 / (36 * 7)))
    assert adk.jeffreys_sd(0, 5) > 0


def test_population_rows_never_zero_uncertainty():
    an = {"closed": {"theta_nmp_max": 58.3, "theta_lid_max": 125.9}, "n_blocks": 5}
    nmp = np.full(100, 70.0)
    lid = np.full(100, 147.0)
    rows, diag = adk.population_rows(nmp, lid, an)
    by = {r["system"]: r for r in rows}
    assert by["adk_apo_p_closed"]["prediction"] == 0.0
    assert by["adk_apo_p_closed"]["uncertainty"] > 0.05
    assert by["adk_apo_theta_lid_mean"]["prediction"] == pytest.approx(147.0)
    assert len(diag["p_closed_cumulative"]["p_closed"]) == 10


def test_population_rows_weighted():
    an = {"closed": {"theta_nmp_max": 58.3, "theta_lid_max": 125.9}, "n_blocks": 4}
    nmp = np.array([40.0, 70.0] * 50)
    lid = np.array([100.0, 150.0] * 50)
    w = np.array([3.0, 1.0] * 50)
    rows, _ = adk.population_rows(nmp, lid, an, weights=w)
    assert rows[0]["prediction"] == pytest.approx(0.75)


def test_mbar_recovers_unbiased_gaussian():
    pytest.importorskip("pymbar")
    rng = np.random.default_rng(0)
    kT, kappa, mu0, s0 = 2.494, 500.0, 2.4, 0.15  # unbiased N(mu0, s0^2) in "radians"
    k0 = kT / s0**2
    centers = np.linspace(2.0, 2.8, 9)
    samples = []
    for c in centers:  # product of two Gaussians is Gaussian
        prec = k0 + kappa
        mean = (k0 * mu0 + kappa * c) / prec
        samples.append(rng.normal(mean, math.sqrt(kT / prec), 2000))
    w, _ = adk.mbar_unbiased_weights(samples, centers, kappa, kT)
    x = np.concatenate(samples)
    m = np.sum(w * x)
    assert m == pytest.approx(mu0, abs=0.01)
    assert math.sqrt(np.sum(w * (x - m) ** 2)) == pytest.approx(s0, rel=0.05)


def test_msm_two_state_population():
    pytest.importorskip("deeptime")
    rng = np.random.default_rng(1)
    t = np.array([[0.98, 0.02], [0.06, 0.94]])  # stationary p_closed = 0.25
    feats, angs = [], []
    for _ in range(4):
        s = [0]
        for _ in range(4999):
            s.append(rng.choice(2, p=t[s[-1]]))
        s = np.array(s)
        nmp = np.where(s == 1, 44.0, 72.0) + rng.normal(0, 2.0, len(s))
        lid = np.where(s == 1, 106.0, 146.0) + rng.normal(0, 2.0, len(s))
        a = np.stack([nmp, lid], axis=1)
        angs.append(a)
        feats.append(np.radians(a))
    m = {
        "tica_dim": None, "n_clusters": 10, "seed": 1, "its_lags_frames": [1, 2, 5],
        "n_its": 1, "msm_lag_frames": 2, "n_bayes_samples": 20, "ck_steps": 3, "ck_sets": 2,
    }  # fmt: skip
    out = adk.build_msm(feats, angs, m, {"theta_nmp_max": 58.3, "theta_lid_max": 125.9})
    p, sd = out["p_closed"]
    assert p == pytest.approx(0.25, abs=0.05)
    assert sd > 0
    assert isinstance(out["ck_max_abs_deviation"], float)


# ----------------------------------------------------------------- PLUMED inputs parse


def _plumed_eval(script: str, pdb_path: Path, workdir: Path) -> dict[str, np.ndarray]:
    openmm = pytest.importorskip("openmm")
    plumed = pytest.importorskip("openmmplumed")
    from openmm import app, unit

    pdb = app.PDBFile(str(pdb_path))
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
        integ.step(3)
        del ctx, integ, force, system
        gc.collect()
    return adk.read_colvar(workdir / "COLVAR")


def test_metad_and_umbrella_scripts_run_in_plumed(tmp_path: Path):
    pdb = REPO_ROOT / "systems" / "adk" / "prepared" / "4AKE_A_protein.pdb"
    groups = adk.cv_groups_for(pdb)
    mcfg = load_config(CONFIGS / "c_adk_l3_metad.yml")["metad"]
    mcfg = dict(mcfg, print_stride_steps=1, pace_steps=1)
    col = _plumed_eval(adk.metad_script(groups, mcfg, 300.0), pdb, tmp_path / "metad")
    assert {"theta_nmp", "theta_lid", "metad.bias", "metad.rbias"} <= set(col)
    assert math.degrees(col["theta_lid"][0]) == pytest.approx(146.54, abs=0.05)
    ucfg = dict(load_config(CONFIGS / "c_adk_l3_umbrella.yml")["umbrella"], print_stride_steps=1)
    col = _plumed_eval(adk.umbrella_script(groups, 140.0, ucfg), pdb, tmp_path / "umb")
    dl = col["theta_lid"][0] - math.radians(140.0)
    assert col["res.bias"][0] == pytest.approx(0.5 * ucfg["kappa_kj_mol_rad2"] * dl**2, rel=1e-4)


# ----------------------------------------------------------------- L1 end to end


def test_l1_runner_end_to_end(isolated_results: Path, tmp_path: Path):
    from mdbench.cli import cmd_run
    from mdbench.schema import read_results

    cfg_path = tmp_path / "l1.yml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "extends": str(CONFIGS / "c_adk_l1.yml"),
                "l1": {"n_mc": 50},
                "output": {"root": str(tmp_path / "runs")},
            }
        )
    )
    rows = cmd_run(cfg_path)
    by = {r["system"]: r for r in rows}
    assert by["adk_open_theta_lid"]["prediction"] == pytest.approx(145.67, abs=0.05)
    assert by["adk_closed_theta_nmp"]["prediction"] == pytest.approx(44.11, abs=0.05)
    assert all(r["uncertainty"] > 0 for r in rows)
    assert read_results("c").num_rows == len(rows)
