"""Branch A, CB7: configs, prepared inputs, the null, docking types, MM/GBSA, ABFE algebra."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from mdbench import REPO_ROOT
from mdbench.branch_a import common
from mdbench.branch_a import ladder as cb7
from mdbench.config import load_config
from mdbench.engines import abfe
from mdbench.registry import get_runner

CONFIGS = REPO_ROOT / "configs"
RUN_CONFIGS = sorted(
    p for p in CONFIGS.glob("a_cb7_*.yml") if p.stem not in ("a_cb7_base", "a_cb7_prep")
)
GT = yaml.safe_load((REPO_ROOT / "data/experimental/cb7_binding_dG.yml").read_text())


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIGS / "a_cb7_l0_null.yml")


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.stem)
def test_configs_resolve_and_have_runner(path: Path):
    c = load_config(path)
    assert c["branch"] == "a" and c["system"] == "cb7"
    get_runner(c["branch"], c["system"], c["level"], c["method"])
    assert set(cb7.guest_ids(c)) <= set(GT["guests"])


def test_ground_truth_entries_are_complete():
    assert len(GT["guests"]) == 15
    for g in GT["guests"].values():
        assert g["source"] in GT["primary_sources"]
        assert -20 < g["dG_kcal_mol"] < -5 and 0 < g["sem_kcal_mol"] < 0.4
    assert GT["doi"] == "10.1146/annurev-biophys-070816-033654"


def test_prepared_files_exist_and_poses_match_table(cfg):
    from mdbench.branch_a.cb7_prep import FILES, skeleton_smiles

    for gid, g in GT["guests"].items():
        for key in FILES:
            assert cb7.prepared(cfg, gid, key).exists(), (gid, key)
        host, guest = cb7.pose_rdkit(cfg, gid)
        assert host.GetNumAtoms() == 126
        assert skeleton_smiles(guest) == skeleton_smiles(g["smiles"])
        idx = json.loads(cb7.prepared(cfg, gid, "indices").read_text())
        assert len(idx["guest"]) == guest.GetNumAtoms()
        # Enough counterions in each box for a net-neutral co-alchemical perturbation.
        q = idx["guest_formal_charge"]
        ion = "cl" if q > 0 else "na"
        for leg in ("complex", "solvent"):
            assert len(idx[leg].get(ion, [])) >= abs(q)


def test_reference_fields(cfg):
    r = common.reference_fields(GT, "cb7_s1_g23")
    assert r["reference"] == -17.19
    assert "10.1146/annurev-biophys-070816-033654" in r["reference_source"]
    assert "10.1021/ja055013x" in r["reference_source"]


def test_heavy_atom_count_and_loo():
    assert common.heavy_atom_count("[NH3+]C12CC3CC(CC(C3)C1)C2") == 11
    x = np.arange(8.0)
    pred, sd = common.loo_linear(x, 2.0 - 0.5 * x)
    assert np.allclose(pred, 2.0 - 0.5 * x)
    assert np.all(sd < 1e-6)
    rng = np.random.default_rng(0)
    y = -0.5 * x + rng.normal(0, 1, 8)
    pred, sd = common.loo_linear(x, y)
    assert np.all(sd > 0.3)


def test_series_of():
    assert common.series_of("cb7_s1_g23") == ["cb7", "cb7_s1"]
    assert common.series_of("t4l_p_xylene") == ["t4l"]


def test_ad4_types():
    from rdkit import Chem

    from mdbench.engines.dock import ad4_type, rotatable_bonds, to_pdbqt

    def types(smi: str) -> dict[str, list[str]]:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        out: dict[str, list[str]] = {}
        for a in m.GetAtoms():
            out.setdefault(a.GetSymbol(), []).append(ad4_type(a))
        return out

    assert sorted(set(types("[NH3+]Cc1ccncc1")["N"])) == ["N", "NA"]  # ammonium N, pyridine NA
    assert set(types("Nc1ccc(N)cc1")["N"]) == {"N"}  # aniline N is not an acceptor
    assert set(types("c1cc[n+](C)cc1")["N"]) == {"N"}  # pyridinium
    assert set(types("CNC(=O)C")["N"]) == {"N"} and types("CNC(=O)C")["O"] == ["OA"]
    assert set(types("CCN")["N"]) == {"NA"}  # neutral amine
    t = types("Cc1ccccc1")
    assert "A" in t["C"] and "C" in t["C"]
    assert set(types("CCO")["H"]) == {"H", "HD"}
    assert rotatable_bonds(Chem.MolFromSmiles("[NH3+]C12CC3CC(CC(C3)C1)C2")) == 0
    assert rotatable_bonds(Chem.MolFromSmiles("[NH3+]Cc1ccc(C[NH3+])cc1")) == 2
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(m, randomSeed=1)
    pq = to_pdbqt(m, rigid_ligand=True, torsdof=1)
    assert pq.startswith("ROOT") and "TORSDOF 1" in pq
    assert sum(ln.startswith("ATOM") for ln in pq.splitlines()) == 4  # C, C, O, HO


def test_min_image_triclinic():
    a = 3.0
    box = np.array([[a, 0, 0], [0, a, 0], [a / 2, a / 2, a / math.sqrt(2)]])
    d = np.array([0.1, 0.2, 0.3])
    for n in ([1, 0, 0], [0, -1, 1], [2, 1, -1]):
        shifted = d + np.array(n) @ box
        assert np.allclose(cb7.min_image(shifted, box), d, atol=1e-9)


def test_mmgbsa_vanishes_at_separation_and_binds_at_pose(cfg):
    from mdbench.branch_a.cb7_prep import read_system
    from mdbench.engines.mmgbsa import GBScorer

    gid = "cb7_s1_g23"
    host, guest = cb7.pose_rdkit(cfg, gid)
    el, nb = cb7._elements_and_neighbours(host, guest)
    idx = json.loads(cb7.prepared(cfg, gid, "indices").read_text())
    sc = GBScorer(read_system(cb7.prepared(cfg, gid, "vacuum_system")), el, nb,
                  idx["host"], idx["guest"])  # fmt: skip
    x = np.vstack([host.GetConformer().GetPositions(), guest.GetConformer().GetPositions()]) / 10
    bound = sc.score(x[None])[0] * common.KCAL_PER_KJ
    far = x.copy()
    far[len(idx["host"]) :] += np.array([20.0, 0, 0])
    apart = sc.score(far[None])[0] * common.KCAL_PER_KJ
    assert bound < -10.0
    assert abs(apart) < 0.1


def test_restraint_volume_and_standard_state():
    t = 298.15
    k = abfe.spring_constant(0.1, t)
    assert abfe.restraint_volume_nm3(k, t) == pytest.approx((2 * math.pi) ** 1.5 * 0.1**3)
    assert abfe.standard_state_term_kt(k, t) == pytest.approx(
        math.log(1.660539 / ((2 * math.pi) ** 1.5 * 1e-3))
    )


def test_standard_state_matches_openmmtools():
    pytest.importorskip("openmmtools")
    from openmm import unit
    from openmmtools import forces, states, testsystems

    t = 298.15
    k = abfe.spring_constant(0.1, t)
    ts = testsystems.HostGuestVacuum()
    r = forces.HarmonicRestraintForce(
        spring_constant=k * unit.kilojoule_per_mole / unit.nanometer**2,
        restrained_atom_indices1=list(range(126)),
        restrained_atom_indices2=list(range(126, ts.system.getNumParticles())),
    )
    st = states.ThermodynamicState(ts.system, t * unit.kelvin)
    # openmmtools returns -ln(V0 / V_r); ours is +ln(V0 / V_r).
    assert -r.compute_standard_state_correction(st) == pytest.approx(
        abfe.standard_state_term_kt(k, t), rel=1e-3
    )


def test_schedule_validation():
    c = load_config(CONFIGS / "a_cb7_l3_abfe.yml")["abfe"]
    for leg in ("complex", "solvent"):
        s = abfe.schedule(c, leg)
        assert len({len(v) for v in s.values()}) == 1
    bad = {
        "schedule": {"solvent": {"lambda_sterics": [1.0, 0.5], "lambda_electrostatics": [1.0, 0.5]}}
    }
    with pytest.raises(ValueError):
        abfe.schedule(bad, "solvent")


def test_l3_campaign_configs_cover_every_guest_once_with_the_dense_schedule():
    base = load_config(CONFIGS / "a_cb7_l3_abfe.yml")["abfe"]["schedule"]
    dense = load_config(CONFIGS / "a_cb7_l3_abfe_g23_dense.yml")
    assert dense["abfe"]["schedule"] == base  # the checked schedule is the campaign's
    assert len(base["complex"]["lambda_sterics"]) == 30
    assert len(base["solvent"]["lambda_sterics"]) == 27
    pilot = load_config(CONFIGS / "a_cb7_l3_abfe_g23.yml")["abfe"]["schedule"]
    assert len(pilot["complex"]["lambda_sterics"]) == 24  # the first pilot stays reproducible
    seen = [g for p in CONFIGS.glob("a_cb7_l3_abfe_s*.yml") for g in cb7.guest_ids(load_config(p))]
    assert sorted([*seen, "cb7_s1_g23"]) == sorted(GT["guests"])
    t4l = [g for p in CONFIGS.glob("a_t4l_l3_abfe_*.yml") for g in cb7.guest_ids(load_config(p))]
    assert sorted(t4l) == sorted(
        yaml.safe_load((REPO_ROOT / "data/experimental/t4l_l99a_binding_dG.yml").read_text())[
            "ligands"
        ]
    )


def test_energy_context_cache_holds_one_context(tmp_path):
    """Under NPT, ``sampler.minimize`` builds a barostat-free copy of the system per state, so
    an unbounded energy cache keeps one context per state (~2.5 GB of GPU memory for 24 CB7
    states, which exhausted the 6 GB card in the first L3 pilot). The cap keeps one."""
    pytest.importorskip("openmmtools")
    import openmm
    from openmm import unit
    from openmmtools import alchemy, mcmc, multistate, states, testsystems

    ts = testsystems.WaterBox(box_edge=2.2 * unit.nanometer, cutoff=0.9 * unit.nanometer)
    ts.system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
    region = alchemy.AlchemicalRegion(alchemical_atoms=[0, 1, 2])
    alch = alchemy.AbsoluteAlchemicalFactory().create_alchemical_system(ts.system, region)
    thermo = states.ThermodynamicState(alch, 300 * unit.kelvin, 1 * unit.bar)
    therm_states = []
    for lam in (1.0, 0.5, 0.0):
        cs = states.CompoundThermodynamicState(thermo, [alchemy.AlchemicalState.from_system(alch)])
        cs.lambda_sterics = cs.lambda_electrostatics = lam
        therm_states.append(cs)

    def n_contexts(capacity):
        sampler = multistate.ReplicaExchangeSampler(
            mcmc_moves=mcmc.LangevinDynamicsMove(n_steps=1), number_of_iterations=1
        )
        sampler.energy_context_cache = abfe._context_cache({"platform": "CPU"}, capacity=capacity)
        sampler.sampler_context_cache = abfe._context_cache({"platform": "CPU"})
        rep = multistate.MultiStateReporter(str(tmp_path / f"c{capacity}.nc"))
        sampler.create(therm_states, states.SamplerState(ts.positions, box_vectors=
                       ts.system.getDefaultPeriodicBoxVectors()), rep)  # fmt: skip
        sampler.minimize(max_iterations=2)
        n = len(sampler.energy_context_cache)
        rep.close()
        return n

    assert n_contexts(None) == 3  # the openmmtools behaviour that made the cap necessary
    assert n_contexts(1) == 1


def test_per_state_samples_and_mbar_on_harmonic_oscillators():
    """MBAR on exact samples of 1D harmonic oscillators recovers the analytic dF."""
    rng = np.random.default_rng(1)
    ks = np.array([1.0, 2.0, 4.0, 8.0])  # reduced spring constants, u = k x^2 / 2
    n_it, n_rep = 400, 4
    x = np.empty((n_it, n_rep))
    state_of_replica = np.empty((n_it, n_rep), int)
    for t in range(n_it):
        perm = rng.permutation(n_rep)
        state_of_replica[t] = perm
        x[t] = rng.normal(0, 1 / np.sqrt(ks[perm]))
    energy = 0.5 * ks[None, None, :] * x[:, :, None] ** 2
    u, n_k = abfe.per_state_samples(energy, state_of_replica, 0, 1)
    assert n_k.tolist() == [n_it] * 4
    f, sf, ov = abfe.mbar_from_u_kln(u, n_k)
    assert f == pytest.approx(0.5 * math.log(ks[-1] / ks[0]), abs=4 * sf + 1e-3)
    assert ov.shape == (4, 4)


def test_combine_sign_convention():
    legs = {
        "solvent": {"dF_kT": 10.0, "ddF_kT": 0.1, "dF_kT_first_half": 10.0,
                    "dF_kT_second_half": 10.0, "wall_clock_s": 1.0},
        "complex": {"dF_kT": 30.0, "ddF_kT": 0.2, "dF_kT_first_half": 29.8,
                    "dF_kT_second_half": 30.2, "wall_clock_s": 2.0},
    }  # fmt: skip
    c = {"restraint_sigma_nm": 0.1}
    r = abfe.combine(legs, c, 298.15)
    kt = 0.0083144626 * 298.15 / 4.184
    ssc = abfe.standard_state_term_kt(abfe.spring_constant(0.1, 298.15), 298.15)
    assert r["dG_kcal"] == pytest.approx((10.0 - 30.0 + ssc) * kt)
    assert r["dG_kcal"] < 0  # decoupling costs more in the complex: binding is favourable
    assert r["sigma_kcal"] == pytest.approx(math.sqrt(0.1**2 + 0.2**2 + 0.2**2) * kt)
    assert r["wall_clock_s"] == 3.0


def test_null_runner(tmp_path, cfg):
    rows = get_runner("a", "cb7", "L0", "null_heavy_atom_count")(cfg, tmp_path)
    assert len(rows) == 15
    for r in rows:
        assert r["reference"] is not None and r["uncertainty"] > 0
    assert (tmp_path / "l0_null.json").exists()


def test_scorer_puts_null_beside_each_level():
    import pandas as pd

    from mdbench.registry import get_scorer

    ref = [-6.0, -8.0, -10.0, -12.0, -14.0]
    base = {"branch": "a", "replicate": 0, "reference_source": "x", "wall_clock_s": 1.0,
            "gpu": "none", "code_version": "t", "config_hash": "h", "uncertainty": 0.5}  # fmt: skip
    rows = []
    for i, r in enumerate(ref):
        sys_ = f"cb7_s1_g{i}"
        rows.append({**base, "system": sys_, "level": "L0", "method": common.NULL_METHOD,
                     "prediction": -10.0 + (i % 2), "reference": r})  # fmt: skip
        rows.append({**base, "system": sys_, "level": "L3", "method": "abfe_hrex_mbar",
                     "prediction": r - 1.0, "reference": r})  # fmt: skip
    out = get_scorer("a")(pd.DataFrame(rows), n_boot=50, seed=0)
    l3 = out[(out["method"] == "abfe_hrex_mbar") & (out["series"] == "cb7")].iloc[0]
    assert l3["spearman"] == pytest.approx(1.0)
    assert l3["pairwise"] == pytest.approx(1.0)
    assert l3["rmse"] == pytest.approx(1.0)
    assert np.isfinite(l3["null_rmse"])
    assert set(out["series"]) == {"cb7", "cb7_s1"}


# --------------------------------------------------------------------------- T4L L99A

T4L_GT = yaml.safe_load((REPO_ROOT / "data/experimental/t4l_l99a_binding_dG.yml").read_text())


def test_t4l_ground_truth_and_prepared_files():
    from mdbench.branch_a.cb7_prep import FILES

    c = load_config(CONFIGS / "a_t4l_l0_null.yml")
    assert len(T4L_GT["ligands"]) == 7
    for lid in T4L_GT["ligands"]:
        for key in FILES:
            assert cb7.prepared(c, lid, key).exists(), (lid, key)
        idx = json.loads(cb7.prepared(c, lid, "indices").read_text())
        assert len(idx["elements"]) == len(idx["host"]) + len(idx["guest"])
        assert len(idx["complex"]["restraint_host"]) >= 8  # pocket C-alpha anchor
        assert idx["guest_formal_charge"] == 0
    r = common.reference_fields(T4L_GT, "t4l_benzene")
    assert r["reference"] == -5.19 and "10.1021/bi00027a006" in r["reference_source"]


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("a_t4l_*.yml")), ids=lambda p: p.stem)
def test_t4l_configs_resolve(path: Path):
    if path.stem in ("a_t4l_base", "a_t4l_prep"):
        return
    c = load_config(path)
    assert c["system"] == "t4l"
    get_runner("a", "t4l", c["level"], c["method"])
    assert set(cb7.guest_ids(c)) <= set(T4L_GT["ligands"])


def test_protein_ad4_types():
    from mdbench.engines.dock import protein_ad4_type

    assert protein_ad4_type("PHE", "CZ", "C", 1, "") == "A"
    assert protein_ad4_type("LEU", "CD1", "C", 3, "") == "C"
    assert protein_ad4_type("HIS", "NE2", "N", 0, "") == "NA"
    assert protein_ad4_type("HIS", "NE2", "N", 1, "") == "N"
    assert protein_ad4_type("ALA", "N", "N", 1, "") == "N"
    assert protein_ad4_type("SER", "HG", "H", 0, "O") == "HD"
    assert protein_ad4_type("ALA", "HB1", "H", 0, "C") == "H"


def test_interaction_entropy_matches_gaussian_limit():
    from mdbench.engines.mmgbsa import interaction_entropy

    kt = common.KB_KCAL_MOL_K * 298.15
    e = np.random.default_rng(0).normal(-30.0, 1.5, 200_000)
    r = interaction_entropy(e, kt)
    assert r["c2"] == pytest.approx(1.5**2 / (2 * kt), rel=0.02)
    assert r["ie"] == pytest.approx(r["c2"], rel=0.05)  # cumulants above 2 vanish for a Gaussian
    assert interaction_entropy(np.full(10, -5.0), kt)["ie"] == pytest.approx(0.0)


def test_mbar_backend_stays_off_the_gpu():
    """pymbar's JAX would otherwise preallocate 75% of the 6 GB card during the first leg's
    analysis and starve the second leg (seen on the T4L complex)."""
    jax = pytest.importorskip("jax")
    import mdbench.engines.abfe  # noqa: F401  (sets JAX_PLATFORMS before any backend use)

    assert {d.platform for d in jax.devices()} == {"cpu"}
