"""Fast tests for FES/basin/block/reweighting/MBAR/MSM code and the ala2 configs."""

from __future__ import annotations

import numpy as np
import pytest

from mdbench import REPO_ROOT
from mdbench.engines import fes

BASINS = {
    "C7eq_beta": [[-180, 0, 50, 180], [-180, 0, -180, -120], [120, 180, -180, 180]],
    "alphaR": [[-180, 0, -120, 50]],
    "alphaL_C7ax": [[0, 120, -180, 180]],
}
KT = fes.kT_kj(300.0)


def test_basins_tile_plane_and_assign() -> None:
    assert fes.validate_basins(BASINS) == list(BASINS)
    g = np.arange(-179.5, 180, 1.0)
    phi, psi = np.meshgrid(g, g)
    lab = fes.assign_basins(phi.ravel(), psi.ravel(), BASINS)
    assert np.all(lab >= 0)  # full tiling
    pts = fes.assign_basins(
        np.array([-65.0, -70.0, 60.0, 175.0, -150.0, 200.0]),
        np.array([140.0, -40.0, 40.0, 0.0, -170.0, 140.0]),
        BASINS,
    )
    assert pts.tolist() == [0, 1, 2, 0, 0, 0]  # 200 deg wraps to -160


def test_overlapping_basins_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        fes.validate_basins({"a": [[-180, 10, -180, 180]], "b": [[0, 180, -180, 180]]})


def test_populations_and_delta_g() -> None:
    lab = np.array([0] * 70 + [1] * 25 + [2] * 5)
    p = fes.populations(lab, 3)
    assert np.allclose(p, [0.7, 0.25, 0.05])
    dg = fes.delta_g_kj(p, 0, KT)
    assert dg[0] == 0 and dg[1] == pytest.approx(-KT * np.log(0.25 / 0.7))
    w = np.where(lab == 2, 10.0, 1.0)
    assert fes.populations(lab, 3, w)[2] == pytest.approx(50 / 145)
    assert np.allclose(fes.boltzmann_populations(np.array([0.0, KT * np.log(2), np.inf]), KT),
                       [2 / 3, 1 / 3, 0])  # fmt: skip


def test_histogram_fes_minimum_zero() -> None:
    rng = np.random.default_rng(0)
    phi = rng.normal(-70, 10, 5000)
    psi = rng.normal(140, 10, 5000)
    f, edges = fes.histogram_fes(phi, psi, KT, 36)
    assert f.shape == (36, 36) and len(edges) == 37
    assert np.nanmin(f) == 0.0
    i, _j = np.unravel_index(np.argmin(np.where(np.isfinite(f), f, np.inf)), f.shape)
    assert -80 <= 0.5 * (edges[i] + edges[i + 1]) <= -60


def _two_state_chain(n: int, p_switch: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.empty(n, dtype=int)
    s[0] = 0
    flips = rng.random(n) < p_switch
    for t in range(1, n):
        s[t] = 1 - s[t - 1] if flips[t] else s[t - 1]
    return s


def test_block_analysis_grows_with_correlation() -> None:
    fast = fes.block_populations(_two_state_chain(20000, 0.4, 1), 2)
    slow = fes.block_populations(_two_state_chain(20000, 0.002, 1), 2)
    assert np.all(np.isfinite(fast.sem)) and np.all(np.isfinite(slow.sem))
    assert slow.sem[0] > 3 * fast.sem[0]
    # Uncorrelated limit: SEM ~ sqrt(p(1-p)/N) within a factor of ~2.
    iid = fes.block_populations(np.random.default_rng(2).integers(0, 2, 20000), 2)
    assert 0.5 < iid.sem[0] / np.sqrt(0.25 / 20000) < 2.0
    ends, run = fes.running_estimate(_two_state_chain(1000, 0.4, 3), 2, n_points=10)
    assert len(ends) == len(run) and np.allclose(run.sum(axis=1), 1)


def test_metad_weights_recover_unbiased_populations() -> None:
    """Reweighting a known static bias recovers the unbiased populations."""
    rng = np.random.default_rng(4)
    true = np.array([0.8, 0.2])
    bias = np.array([KT * np.log(4.0), 0.0])  # raises the deep well: 0.8/0.2 -> 0.5/0.5
    biased = true * np.exp(-bias / KT)
    biased /= biased.sum()
    lab = rng.choice(2, size=40000, p=biased)
    w = fes.metad_weights(bias[lab], KT)
    assert fes.populations(lab, 2, w) == pytest.approx(true, abs=0.01)
    assert 0 < fes.effective_sample_size(w) <= len(w)


def test_umbrella_mbar_on_analytic_double_well() -> None:
    """Exact sampling of biased windows on a 1D periodic double well; MBAR recovers p."""
    pytest.importorskip("pymbar")
    rng = np.random.default_rng(5)
    grid = np.linspace(-np.pi, np.pi, 3601)[:-1]
    f0 = 4.0 * KT * np.cos(2 * grid) + 1.0 * KT * np.sin(grid)  # wells near +-pi/2
    centers = np.radians(np.arange(-180, 180, 20) + 10.0)
    k = 60.0
    xs, labs = [], []
    lab_of = (grid > 0).astype(int)  # basin 0: x<=0, basin 1: x>0
    for c in centers:
        u = f0 + fes.umbrella_bias_kj(grid, c, k)
        p = np.exp(-(u - u.min()) / KT)
        idx = rng.choice(len(grid), size=600, p=p / p.sum())
        xs.append(grid[idx])
        labs.append(lab_of[idx])
    res = fes.umbrella_mbar(xs, labs, centers, k, KT, 2, 0, profile_bins=36, subsample=False)
    p_true = np.exp(-(f0 - f0.min()) / KT)
    p_true = np.array([p_true[lab_of == b].sum() for b in (0, 1)]) / p_true.sum()
    assert res.pops == pytest.approx(p_true, abs=4 * res.pop_err.max() + 0.01)
    dg_true = -KT * np.log(p_true[1] / p_true[0])
    assert abs(res.dg_kj[1] - dg_true) < 4 * res.dg_err_kj[1] + 0.2
    assert res.overlap_min > 0.01


def test_msm_on_three_state_chain() -> None:
    pytest.importorskip("deeptime")
    from mdbench.engines import msm

    t = np.array([[0.97, 0.02, 0.01], [0.04, 0.95, 0.01], [0.05, 0.05, 0.90]])
    rng = np.random.default_rng(6)
    trajs = []
    for _ in range(3):
        s = [0]
        for _ in range(6000):
            s.append(rng.choice(3, p=t[s[-1]]))
        trajs.append(np.array(s))
    macro = np.array([0, 1, 2])
    model = msm.estimate_msm(trajs, 1)
    evals, evecs = np.linalg.eig(t.T)
    pi = np.real(evecs[:, np.argmax(np.real(evals))])
    pi /= pi.sum()
    assert msm.macro_populations(model, macro, 3) == pytest.approx(pi, abs=0.04)
    its = msm.implied_timescales(trajs, [1, 2, 4], 2)
    assert its.shape == (3, 2) and np.all(its > 0)
    ck = msm.ck_test(trajs, 1, macro, 3, (1, 2, 4))
    assert ck.max_abs_dev < 0.05  # the chain is exactly Markovian
    samples = msm.bayesian_macro_populations(trajs, 1, macro, 3, n_samples=20)
    assert samples.shape == (20, 3) and np.allclose(samples.sum(axis=1), 1)
    ang = msm.centres_to_angles_deg(msm.periodic_features([np.radians([30.0, -150.0])]))
    assert np.allclose(ang[:, 0], [30.0, -150.0])


@pytest.mark.parametrize(
    "name", ["l0", "l1", "l2", "l3_metad", "l3_umbrella", "l3_msm"]
)  # fmt: skip
def test_ala2_configs_resolve(name: str) -> None:
    from mdbench.config import load_config
    from mdbench.registry import get_runner

    cfg = load_config(REPO_ROOT / f"configs/c_ala2_{name}.yml")
    assert cfg["branch"] == "c" and cfg["system"] == "ala2"
    fes.validate_basins(cfg["analysis"]["basins"])
    assert cfg["analysis"]["reference_basin"] in cfg["analysis"]["basins"]
    assert get_runner("c", "ala2", cfg["level"], cfg["method"]) is not None


def test_observable_rows_drop_nonfinite() -> None:
    from mdbench.branch_c.ala2 import observable_rows

    rows = observable_rows(["a", "b", "c"], 0, np.array([0.6, 0.4, 0.0]),
                           np.array([0.05, 0.05, 0.0]), KT)  # fmt: skip
    systems = [r["system"] for r in rows]
    assert systems == ["ala2.pop.a", "ala2.pop.b", "ala2.dG.b"]  # unvisited c: no rows
    dg = rows[-1]["prediction"]
    assert dg == pytest.approx(fes.as_kcal(-KT * np.log(0.4 / 0.6)))
