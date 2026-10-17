"""AdK branch B: configs, endpoints, the cheap runners, MD analysis on synthetic frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mdbench import REPO_ROOT
from mdbench.branch_b import adk
from mdbench.config import load_config
from mdbench.engines import modes as nm
from mdbench.registry import get_runner
from mdbench.schema import validate_row

CONFIGS = REPO_ROOT / "configs"
RUN_CONFIGS = sorted(p for p in CONFIGS.glob("b_adk_*.yml") if p.stem != "b_adk_base")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIGS / "b_adk_l0_anm.yml")


@pytest.fixture(scope="module")
def ep(cfg) -> adk.Endpoints:
    return adk.Endpoints(cfg)


def _full_row(row: dict, cfg: dict) -> dict:
    from mdbench.cli import complete_row

    return complete_row(row, cfg, 1.0)


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.stem)
def test_configs_resolve_and_have_runner(path: Path):
    c = load_config(path)
    assert c["branch"] == "b" and c["system"] == "adk"
    get_runner(c["branch"], c["system"], c["level"], c["method"])
    for state in c["adk"]["endpoints"].values():
        for p in state.values():
            assert (REPO_ROOT / p).exists()
    assert (REPO_ROOT / c["adk"]["solvated_pdb"]).exists()


def test_closed_box_matches_branch_c():
    b = load_config(CONFIGS / "b_adk_l0_anm.yml")
    c = load_config(CONFIGS / "c_adk_l1.yml")
    assert b["analysis"]["closed"] == c["analysis"]["closed"]


def test_bfactors_read_from_raw_file():
    b = adk.crystal_ca_bfactors(REPO_ROOT / "systems/adk/raw/4AKE.pdb", "A")
    assert len(b) == 214
    assert b[12] == pytest.approx(32.73)  # ATOM 78 CA GLY A 12 in 4AKE.pdb


def test_endpoints_and_displacement(ep):
    assert len(ep.resids) == 214
    assert set(ep.states) == {"open", "closed"}
    d = ep.displacement("4AKE_A", "1AKE_A")
    rmsd = np.sqrt(np.sum(d**2) / 214)
    # Superposed CA RMSD between the open and closed crystals: several Angstrom.
    assert 4.0 < rmsd < 10.0
    # Copies of the same state are much closer than open vs closed.
    same = np.sqrt(np.sum(ep.displacement("4AKE_A", "4AKE_B") ** 2) / 214)
    assert same < rmsd / 3


def test_score_subspace_perfect_mode(ep):
    d = ep.displacement("4AKE_A", "1AKE_A")
    modes = np.stack([d / np.linalg.norm(d)] + [np.zeros_like(d)] * 0, axis=1)
    rng = np.random.default_rng(0)
    modes = np.concatenate([modes, rng.normal(size=(d.size, 9))], axis=1)
    s = adk.score_subspace(ep, "4AKE_A", modes, 10)
    assert set(s) == {"open2closed"}
    assert s["open2closed"]["1AKE_A"]["overlap_max"] == pytest.approx(1.0)
    assert s["open2closed"]["1AKE_A"]["best_mode"] == 1
    assert s["open2closed"]["1AKE_A"]["cumoverlap"] == pytest.approx(1.0)


def _check_rows(rows: list[dict], c: dict) -> None:
    for r in rows:
        validate_row(_full_row(r, c), "b")
        if "overlap" in r["system"]:
            assert r["reference"] == 1.0 and r["reference_source"]
            assert 0.0 <= r["prediction"] <= 1.0
        else:
            assert "reference" not in r


def test_null_runner(tmp_path):
    c = load_config(CONFIGS / "b_adk_l0_null.yml")
    c["null_model"]["n_draws"] = 50
    rows = get_runner("b", "adk", "L0", "null_random_centroid")(c, tmp_path)
    names = {r["system"] for r in rows}
    assert names == {
        "adk_open2closed_overlap_max", "adk_open2closed_cumoverlap_k10",
        "adk_closed2open_overlap_max", "adk_closed2open_cumoverlap_k10",
        "adk_open_bfactor_pearson", "adk_closed_bfactor_pearson",
    }  # fmt: skip
    _check_rows(rows, c)
    cum = next(r for r in rows if r["system"] == "adk_open2closed_cumoverlap_k10")
    assert cum["prediction"] == pytest.approx(np.sqrt(10 / 636), rel=0.3)
    assert (tmp_path / "l0_null.json").exists()


def test_anm_runner_beats_null(tmp_path, cfg):
    rows = get_runner("b", "adk", "L0", "anm_ca")(cfg, tmp_path)
    _check_rows(rows, cfg)
    by = {r["system"]: r for r in rows}
    # The open->closed motion of AdK is a textbook low-mode motion; ANM must beat the
    # random-subspace null (~0.13 cumulative) by a wide margin.
    assert by["adk_open2closed_cumoverlap_k10"]["prediction"] > 0.4
    assert by["adk_open_bfactor_pearson"]["prediction"] > 0.0
    detail = json.loads((tmp_path / "l0_anm.json").read_text())
    assert set(detail["cutoffs_A"]) == {12.0, 15.0, 18.0}


def test_fd_hessian_matches_analytic_harmonic_bond():
    import openmm
    from openmm import unit

    from mdbench.engines.nma import fd_hessian

    k, r0 = 1000.0, 0.15
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, r0, k)
    system.addForce(bond)
    ctx = openmm.Context(
        system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    x = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    ctx.setPositions(x * unit.nanometer)
    h, asym = fd_hessian(ctx, x, 1e-5)
    # At the minimum the only stiffness is k along the bond axis.
    exp = np.zeros((6, 6))
    exp[0, 0] = exp[3, 3] = k
    exp[0, 3] = exp[3, 0] = -k
    assert np.allclose(h, exp, atol=1e-3)
    assert asym < 1e-6


def _synthetic_run(root: Path, name: str, ep: adk.Endpoints, closing: bool, start: str) -> Path:
    """A run dir in the pca_short_md layout with C-alpha frames and domain angles."""
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(len(name))
    x0, x1 = ep.xyz["4AKE_A"], nm.superpose(ep.xyz["1AKE_A"], ep.xyz["4AKE_A"])
    n = 200
    lam = np.linspace(0, 1 if closing else 0.2, n)
    xyz = x0[None] + lam[:, None, None] * (x1 - x0)[None] + rng.normal(0, 0.3, (n, 214, 3))
    t = np.arange(n) * 10.0
    np.savez_compressed(d / "ca_traj.npz", time_ps=t, xyz_A=xyz.astype(np.float32))
    nmp = 72.6 - lam * 28.5
    lid = 145.7 - lam * 39.5
    np.savetxt(d / "cv_timeseries.csv", np.column_stack([t, nmp, lid]), delimiter=",",
               header="time_ps,theta_nmp_deg,theta_lid_deg", comments="")  # fmt: skip
    (d / "md_result.json").write_text(json.dumps({"wall_clock_s": 100.0, "gpu": "test"}))
    (d / "config.resolved.yml").write_text(yaml.safe_dump({"adk": {"solvated_pdb": start}}))
    return d


def test_single_run_analysis_on_synthetic_frames(tmp_path, ep):
    c = load_config(CONFIGS / "b_adk_l2_pca.yml")
    d = _synthetic_run(tmp_path, "rep", ep, closing=True, start="4AKE_A_solvated.pdb.gz")
    t, xyz = adk.ca_frames(d, ep.resids)
    cv = np.loadtxt(d / "cv_timeseries.csv", delimiter=",", skiprows=1)
    rows, diag = adk.analyse_single_run(c, ep, "4AKE_A", t, xyz, cv)
    by = {r["system"]: r for r in rows}
    # Frames move along the crystal displacement, so PC1 must be that displacement.
    assert by["adk_open2closed_overlap_max"]["prediction"] > 0.9
    assert by["adk_open_md_min_rmsd_to_closed_A"]["prediction"] < 1.5
    assert diag["visited_closed"]
    _check_rows(rows, c)


def test_pooled_runner_counts_transitions_and_filters_start(tmp_path, ep, monkeypatch):
    c = load_config(CONFIGS / "b_adk_l2_pooled.yml")
    _synthetic_run(tmp_path, "r0", ep, closing=True, start="systems/adk/prepared/4AKE_A.pdb.gz")
    _synthetic_run(tmp_path, "r1", ep, closing=False, start="systems/adk/prepared/4AKE_A.pdb.gz")
    _synthetic_run(tmp_path, "r2", ep, closing=False, start="systems/adk/prepared/4AKE_A.pdb.gz")
    _synthetic_run(tmp_path, "r3", ep, closing=True, start="systems/adk/prepared/1AKE_A.pdb.gz")
    c["pooled"]["runs"] = [str(tmp_path / "r*")]
    c["pooled"]["n_boot"] = 10
    out = tmp_path / "out"
    out.mkdir()
    rows = get_runner("b", "adk", "L2", "pooled_short_md")(c, out)
    by = {r["system"]: r for r in rows}
    assert by["adk_open_md_frac_reached_closed"]["prediction"] == pytest.approx(1 / 3)
    assert by["adk_open_md_frac_reached_closed"]["uncertainty"] > 0
    assert all(r["wall_clock_s"] >= 300.0 for r in rows)  # 3 pooled runs x 100 s of MD
    diag = json.loads((out / "analysis.json").read_text())
    assert diag["n_runs"] == 3 and len(diag["skipped"]) == 1
    _check_rows(rows, c)


def test_scorer_puts_null_beside_each_level():
    import pandas as pd

    from mdbench.registry import get_scorer

    base = {
        "branch": "b", "replicate": 0, "reference": 1.0, "reference_source": "x",
        "wall_clock_s": 1.0, "gpu": "none", "code_version": "t", "config_hash": "h",
        "system": "s", "level": "L0", "uncertainty": 0.05,
    }  # fmt: skip
    df = pd.DataFrame(
        [
            {**base, "method": "null_random_centroid", "prediction": 0.1},
            {**base, "method": "anm_ca", "prediction": 0.6},
        ]
    )
    out = get_scorer("b")(df)
    anm = out[out["method"] == "anm_ca"].iloc[0]
    assert anm["null_prediction"] == pytest.approx(0.1)
    assert anm["abs_err"] == pytest.approx(0.4)
    assert anm["z_vs_null"] == pytest.approx(0.5 / np.sqrt(0.005))
