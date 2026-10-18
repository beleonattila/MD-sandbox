"""Branch A, the L0-L3 runners for both systems: CB7 host-guest and T4 lysozyme L99A.

Inputs are the prepared files from :mod:`mdbench.branch_a.cb7_prep` and
:mod:`mdbench.branch_a.t4l_prep` (identical force-field parameters at every level). The
config block named after the system (``cb7:`` or ``t4l:``) points at them. Each runner
emits one row per ligand: ``system`` is the ligand id (``cb7_s1_g23``, ``t4l_benzene``),
``prediction`` a standard binding free energy in kcal/mol, and ``reference`` the measured
value from the system's ground-truth file in ``data/experimental/``.

Runners (``method``):

- L0 ``null_heavy_atom_count``: leave-one-out linear fit of dG on heavy-atom count.
- L1 ``vina_rigid_ensemble``: AutoDock Vina score of the best docked pose (rigid
  receptor, rigid ligand conformer ensemble). Never sees the bound poses; for T4L the
  search box is centred on the cavity of the receptor structure.
- L2 ``mmgbsa_obc2``: explicit-solvent MD of the complex, frames rescored by MM/GBSA.
- L3 ``abfe_hrex_mbar``: absolute binding free energy, alchemical decoupling with
  Hamiltonian replica exchange and MBAR (runs in ``mdbench-fe``; see
  :mod:`mdbench.engines.abfe`).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mdbench.branch_a.common import (
    KB_KCAL_MOL_K,
    KCAL_PER_KJ,
    entries,
    heavy_atom_count,
    loo_linear,
    reference_fields,
)
from mdbench.branch_c.adk import repo_path
from mdbench.registry import register

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- inputs


SYSTEMS = ("cb7", "t4l")


def block(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """The config block named after the system (``cfg['cb7']`` or ``cfg['t4l']``)."""
    return cfg[cfg["system"]]


def ground_truth(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The ground-truth YAML named in the config."""
    return yaml.safe_load(repo_path(block(cfg)["ground_truth"]).read_text())


def guest_ids(cfg: Mapping[str, Any]) -> list[str]:
    """Ligands selected by ``<system>.guests`` (``all`` or a list), in ground-truth order."""
    gt = ground_truth(cfg)
    sel = block(cfg).get("guests", "all")
    ids = list(entries(gt))
    if sel == "all":
        return ids
    unknown = sorted(set(sel) - set(ids))
    if unknown:
        raise KeyError(f"unknown guests {unknown}")
    return [g for g in ids if g in sel]


def prepared(cfg: Mapping[str, Any], gid: str, key: str) -> Path:
    """Path of one prepared file for a ligand (see ``cb7_prep.FILES``)."""
    from mdbench.branch_a.cb7_prep import FILES

    return repo_path(block(cfg)["prepared_dir"]) / gid / FILES[key]


def pose_rdkit(cfg: Mapping[str, Any], gid: str) -> tuple[Any, Any]:
    """(host, guest) RDKit molecules with explicit H and the bound-pose coordinates."""
    from rdkit import Chem

    path = repo_path(block(cfg)["poses_dir"]) / f"{gid}.sdf"
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False)]
    if len(mols) != 2 or any(m is None for m in mols):
        raise ValueError(f"{path}: expected host and guest records")
    return mols[0], mols[1]


def ligand_mol(cfg: Mapping[str, Any], gid: str) -> Any:
    """The ligand as simulated (protonation state included), RDKit, no hydrogens.

    CB7 guests come from the identity-checked pose files (their charges are the benchmark's
    protonation states); T4L ligands are neutral and come from the table SMILES.
    """
    from rdkit import Chem

    if "poses_dir" in block(cfg):
        return Chem.RemoveHs(pose_rdkit(cfg, gid)[1])
    return Chem.MolFromSmiles(entries(ground_truth(cfg))[gid]["smiles"])


def guest_smiles(cfg: Mapping[str, Any], gid: str) -> str:
    """The ligand as simulated, canonical SMILES."""
    from rdkit import Chem

    return Chem.MolToSmiles(ligand_mol(cfg, gid))


def _row(gt: Mapping[str, Any], gid: str, pred: float, unc: float, wall: float,
         gpu: str = "none") -> dict[str, Any]:  # fmt: skip
    return {
        "system": gid,
        "prediction": float(pred),
        "uncertainty": float(unc),
        "wall_clock_s": float(wall),
        "gpu": gpu,
        **reference_fields(gt, gid),
    }


def _json(obj: Any, path: Path) -> None:
    def plain(o: Any) -> Any:
        return o.tolist() if hasattr(o, "tolist") else float(o)

    path.write_text(json.dumps(obj, indent=1, default=plain))


# --------------------------------------------------------------------------- L0


@register(branch="a", system="cb7", level="L0", method="null_heavy_atom_count")
@register(branch="a", system="t4l", level="L0", method="null_heavy_atom_count")
def l0_null(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """dG predicted from heavy-atom count alone, leave-one-out over the selected guests."""
    t0 = time.perf_counter()
    gt = ground_truth(cfg)
    ids = guest_ids(cfg)
    hac = np.array([heavy_atom_count(guest_smiles(cfg, g)) for g in ids], float)
    ref = np.array([entries(gt)[g]["dG_kcal_mol"] for g in ids], float)
    pred, sd = loo_linear(hac, ref)
    wall = (time.perf_counter() - t0) / len(ids)
    _json({"guests": ids, "heavy_atoms": hac, "prediction": pred, "sd": sd,
           "full_fit_slope_intercept": np.polyfit(hac, ref, 1).tolist()},
          run_dir / "l0_null.json")  # fmt: skip
    return [_row(gt, g, p, s, wall) for g, p, s in zip(ids, pred, sd, strict=True)]


# --------------------------------------------------------------------------- L1


@register(branch="a", system="cb7", level="L1", method="vina_rigid_ensemble")
@register(branch="a", system="t4l", level="L1", method="vina_rigid_ensemble")
def l1_vina(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Best Vina score per guest; uncertainty is the spread over independent docking seeds."""
    from rdkit import Chem

    from mdbench.engines import dock

    d = cfg["dock"]
    gt = ground_truth(cfg)
    receptor = run_dir / "receptor.pdbqt"
    if "receptor_sdf" in d:  # host-guest: rigid host, box on the host centroid
        host = Chem.MolFromMolFile(str(repo_path(d["receptor_sdf"])), removeHs=False)
        receptor.write_text(dock.to_pdbqt(host))
        center = host.GetConformer().GetPositions().mean(axis=0)
    else:  # protein: box on the crystal ligand of the receptor structure
        receptor.write_text(dock.protein_pdbqt(repo_path(d["receptor_pdb"])))
        ref = d["center_ligand_of"]
        idx = json.loads(prepared(cfg, ref, "indices").read_text())
        from openmm import unit

        from mdbench.branch_c.adk import read_pdb

        vac = read_pdb(prepared(cfg, ref, "vacuum_pdb"))
        xyz = np.array(vac.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        center = xyz[idx["guest"]].mean(axis=0)
    rows, detail = [], {}
    for gid in guest_ids(cfg):
        t0 = time.perf_counter()
        lig = ligand_mol(cfg, gid)
        bests, poses = [], []
        for seed in d["seeds"]:
            confs = dock.conformers(lig, int(d["n_conformers"]), int(seed))
            r = dock.dock_rigid_ensemble(
                str(receptor), confs, center, float(d["box_A"]), int(d["exhaustiveness"]), int(seed)
            )
            bests.append(r["best"])
            poses.append(r["pose"])
        k = int(np.argmin(bests))
        (run_dir / f"{gid}_pose.pdbqt").write_text(poses[k])
        atoms = [ln for ln in poses[k].splitlines() if ln.startswith(("ATOM", "HETATM"))]
        xyz = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in atoms])
        offset = float(np.linalg.norm(xyz.mean(axis=0) - center))
        pred = float(np.mean(bests))
        unc = float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0
        detail[gid] = {"best_per_seed": bests, "torsdof": r["torsdof"],
                       "best_pose_centroid_offset_A": offset,
                       "inside_cavity": offset < float(d["inside_cavity_A"])}  # fmt: skip
        rows.append(_row(gt, gid, pred, unc, time.perf_counter() - t0))
    _json(detail, run_dir / "l1_vina.json")
    return rows


# --------------------------------------------------------------------------- L2


def _elements_and_neighbours(host: Any, guest: Any) -> tuple[list[str], list[str | None]]:
    el, nb = [], []
    for m in (host, guest):
        for a in m.GetAtoms():
            el.append(a.GetSymbol())
            nb.append(a.GetNeighbors()[0].GetSymbol() if a.GetSymbol() == "H" else None)
    return el, nb


def min_image(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Shortest periodic image of displacement(s) ``d`` (nm) for triclinic box rows ``box``.

    Reduced-form boxes (OpenMM convention): subtract c, then b, then a multiples in turn.
    """
    d = np.array(d, float)
    for k in (2, 1, 0):
        d = d - np.outer(np.round(d[..., k] / box[k, k]), box[k]).reshape(d.shape)
    return d


def read_solute_frames(run: Path, n_host: int, n_guest: int) -> tuple[np.ndarray, np.ndarray]:
    """(time_ps, frames_nm) of host + guest from ``traj.xtc``, guest moved next to host.

    The XTC keeps molecules whole but may place guest and host in different periodic
    images; the guest is translated by whole box vectors to the image nearest the host.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.mdamath import triclinic_vectors

    u = mda.Universe(str(run / "topology.pdb"), str(run / "traj.xtc"))
    n = n_host + n_guest
    if u.atoms.n_atoms != n:
        raise ValueError(f"{run}: trajectory has {u.atoms.n_atoms} atoms, expected {n}")
    t, frames = [], []
    for ts in u.trajectory:
        x = u.atoms.positions.astype(float) / 10.0
        box = triclinic_vectors(ts.dimensions) / 10.0
        shift = min_image(x[n_host:].mean(0) - x[:n_host].mean(0), box)
        x[n_host:] += shift - (x[n_host:].mean(0) - x[:n_host].mean(0))
        t.append(ts.time)
        frames.append(x)
    return np.asarray(t), np.asarray(frames)


def _block_sem(x: np.ndarray, n_blocks: int) -> float:
    from mdbench.branch_c.adk import block_sem

    return block_sem(x, n_blocks)


def mmgbsa_one(cfg: Mapping[str, Any], gid: str, out: Path, seed: int) -> dict[str, Any]:
    """One MD replicate of one complex plus its MM/GBSA analysis."""
    import copy

    from mdbench.branch_a.cb7_prep import read_system
    from mdbench.branch_c.adk import read_pdb
    from mdbench.engines.md import run_md
    from mdbench.engines.mmgbsa import GBScorer

    idx = json.loads(prepared(cfg, gid, "indices").read_text())
    solute = idx["host"] + idx["guest"]
    pdb = read_pdb(prepared(cfg, gid, "complex_pdb"))
    system = read_system(prepared(cfg, gid, "complex_system"))
    md = copy.deepcopy(dict(cfg["md"]))
    md["seed"] = int(seed)
    md["xtc_selection"] = solute
    res = run_md(pdb.topology, system, pdb.positions, md, out,
                 box_vectors=pdb.topology.getPeriodicBoxVectors())  # fmt: skip
    t0 = time.perf_counter()
    if "elements" in idx:
        el, nb = idx["elements"], idx["h_neighbour"]
    else:
        el, nb = _elements_and_neighbours(*pose_rdkit(cfg, gid))
    scorer = GBScorer(read_system(prepared(cfg, gid, "vacuum_system")), el, nb,
                      idx["host"], idx["guest"])  # fmt: skip
    t, frames = read_solute_frames(out, len(idx["host"]), len(idx["guest"]))
    keep = t >= float(cfg["mmgbsa"]["discard_ps"])
    e = scorer.score(frames[keep]) * KCAL_PER_KJ
    nh = len(idx["host"])
    # Bound diagnostic: ligand centre vs the restraint anchor (pocket C-alpha atoms for a
    # protein, the whole host for CB7), not the protein centre a buried pocket sits away from.
    anchor = idx["complex"].get("restraint_host") or list(range(nh))
    dist = np.linalg.norm(frames[keep, nh:].mean(1) - frames[keep][:, anchor].mean(1), axis=1)
    np.savez_compressed(out / "mmgbsa.npz", time_ps=t[keep], dE_kcal=e, com_dist_nm=dist)
    nbk = int(cfg["mmgbsa"]["n_blocks"])
    return {
        "seed": int(seed),
        "mean_kcal": float(e.mean()),
        "sem_kcal": _block_sem(e, nbk),
        "sd_kcal": float(e.std(ddof=1)),
        "n_frames": int(keep.sum()),
        "bound_fraction": float(np.mean(dist < float(cfg["mmgbsa"]["bound_com_nm"]))),
        "com_dist_max_nm": float(dist.max()),
        "md_wall_clock_s": res.wall_clock_s,
        "ns_per_day": res.ns_per_day,
        "gpu": res.gpu,
        "analysis_wall_clock_s": time.perf_counter() - t0,
    }


@register(branch="a", system="cb7", level="L2", method="mmgbsa_obc2")
@register(branch="a", system="t4l", level="L2", method="mmgbsa_obc2")
def l2_mmgbsa(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """MM/GBSA over independent MD replicates; uncertainty from the replicate spread."""
    gt = ground_truth(cfg)
    rows, detail = [], {}
    seeds = [int(s) for s in cfg["mmgbsa"]["seeds"]]
    for gid in guest_ids(cfg):
        reps = []
        for k, seed in enumerate(seeds):
            rd = run_dir / gid / f"rep_{k}"
            done = rd / "replicate.json"
            if done.exists():
                reps.append(json.loads(done.read_text()))
                continue
            r = mmgbsa_one(cfg, gid, rd, seed)
            _json(r, done)
            reps.append(r)
        means = np.array([r["mean_kcal"] for r in reps])
        sems = np.array([r["sem_kcal"] for r in reps])
        spread = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else 0.0
        within = float(math.sqrt(np.nanmean(sems**2) / len(sems)))
        wall = sum(r["md_wall_clock_s"] + r["analysis_wall_clock_s"] for r in reps)
        detail[gid] = {"replicates": reps, "spread_sem": spread, "within_sem": within}
        rows.append(_row(gt, gid, means.mean(), max(spread, within), wall, reps[0]["gpu"]))
        log.info("%s: %.2f +- %.2f kcal/mol", gid, means.mean(), max(spread, within))
    _json(detail, run_dir / "l2_mmgbsa.json")
    return rows


@register(branch="a", system="cb7", level="L2", method="mmgbsa_obc2_ie")
@register(branch="a", system="t4l", level="L2", method="mmgbsa_obc2_ie")
def l2_mmgbsa_ie(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """MM/GBSA plus an interaction-entropy term, re-using the MM/GBSA run's frames (no new MD).

    ``ie.source_config`` names the MM/GBSA config whose run directory holds the replicates.
    Per replicate: ``<dE_GB> - T dS``, with -T dS from the gas-phase host-guest interaction
    energy of the same frames (``ie.estimator``: ``c2`` or ``ie``). Uncertainty as at L2.
    The wall clock adds the MM/GBSA replicate's cost, since this level cannot run without it.
    """
    from mdbench.branch_a.cb7_prep import read_system
    from mdbench.config import load_config
    from mdbench.config import run_dir as config_run_dir
    from mdbench.engines.mmgbsa import GBScorer, interaction_entropy

    src_cfg = load_config(repo_path(cfg["ie"]["source_config"]))
    src = config_run_dir(src_cfg)
    est = str(cfg["ie"]["estimator"])
    kt = KB_KCAL_MOL_K * float(src_cfg["md"]["temperature_K"])
    gt = ground_truth(cfg)
    rows, detail = [], {}
    for gid in guest_ids(cfg):
        idx = json.loads(prepared(cfg, gid, "indices").read_text())
        nh, ng = len(idx["host"]), len(idx["guest"])
        el, nb = ((idx["elements"], idx["h_neighbour"]) if "elements" in idx
                  else _elements_and_neighbours(*pose_rdkit(cfg, gid)))  # fmt: skip
        scorer = GBScorer(read_system(prepared(cfg, gid, "vacuum_system")), el, nb,
                          idx["host"], idx["guest"], solvation=False)  # fmt: skip
        reps = []
        for k in range(len(src_cfg["mmgbsa"]["seeds"])):
            rd = src / gid / f"rep_{k}"
            if not (rd / "replicate.json").exists():
                raise FileNotFoundError(f"{rd}: run {cfg['ie']['source_config']} first")
            t0 = time.perf_counter()
            l2 = json.loads((rd / "replicate.json").read_text())
            gb = np.load(rd / "mmgbsa.npz")
            t, frames = read_solute_frames(rd, nh, ng)
            keep = t >= float(src_cfg["mmgbsa"]["discard_ps"])
            if keep.sum() != len(gb["dE_kcal"]):
                raise ValueError(f"{rd}: {keep.sum()} frames vs {len(gb['dE_kcal'])} in mmgbsa.npz")
            e_int = scorer.score(frames[keep]) * KCAL_PER_KJ
            ent = interaction_entropy(e_int, kt)
            out = run_dir / gid / f"rep_{k}"
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "e_int.npz", time_ps=t[keep], e_int_kcal=e_int)
            reps.append({
                "mmgbsa_kcal": l2["mean_kcal"], "mmgbsa_sem_kcal": l2["sem_kcal"],
                "e_int_mean_kcal": float(e_int.mean()), "minus_tds": ent,
                "dG_kcal": l2["mean_kcal"] + ent[est], "gpu": l2["gpu"],
                "wall_clock_s": l2["md_wall_clock_s"] + l2["analysis_wall_clock_s"]
                + time.perf_counter() - t0,
            })  # fmt: skip
        dg = np.array([r["dG_kcal"] for r in reps])
        sems = np.array([r["mmgbsa_sem_kcal"] for r in reps])
        spread = float(dg.std(ddof=1) / math.sqrt(len(dg))) if len(dg) > 1 else 0.0
        within = float(math.sqrt(np.nanmean(sems**2) / len(sems)))
        detail[gid] = {"estimator": est, "replicates": reps, "spread_sem": spread}
        wall = sum(r["wall_clock_s"] for r in reps)
        rows.append(_row(gt, gid, dg.mean(), max(spread, within), wall, reps[0]["gpu"]))
        tds = float(np.mean([r["minus_tds"][est] for r in reps]))
        log.info("%s: %.2f +- %.2f kcal/mol (-TdS %s %.2f)", gid, dg.mean(), max(spread, within),
                 est, tds)  # fmt: skip
    _json(detail, run_dir / "l2_mmgbsa_ie.json")
    return rows


# --------------------------------------------------------------------------- L3


@register(branch="a", system="cb7", level="L3", method="abfe_hrex_mbar")
@register(branch="a", system="t4l", level="L3", method="abfe_hrex_mbar")
def l3_abfe(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Absolute binding free energy per guest (complex and solvent legs, MBAR)."""
    from mdbench.engines import abfe

    gt = ground_truth(cfg)
    rows = []
    for gid in guest_ids(cfg):
        idx = json.loads(prepared(cfg, gid, "indices").read_text())
        legs = {}
        for leg in ("solvent", "complex"):
            out = run_dir / gid / leg
            done = out / "leg_result.json"
            if done.exists():
                legs[leg] = json.loads(done.read_text())
                continue
            legs[leg] = abfe.run_leg(
                leg=leg,
                system_xml=prepared(cfg, gid, f"{leg}_system"),
                pdb=prepared(cfg, gid, f"{leg}_pdb"),
                indices=idx[leg],
                guest_charge=int(idx["guest_formal_charge"]),
                cfg=cfg["abfe"],
                md=cfg["md"],
                out=out,
            )
            _json(legs[leg], done)
        res = abfe.combine(legs, cfg["abfe"], float(cfg["md"]["temperature_K"]))
        _json({"legs": legs, "result": res}, run_dir / gid / "abfe_result.json")
        rows.append(_row(gt, gid, res["dG_kcal"], res["sigma_kcal"], res["wall_clock_s"],
                         legs["complex"]["gpu"]))  # fmt: skip
    return rows


__all__ = [
    "SYSTEMS",
    "block",
    "guest_ids",
    "l0_null",
    "l1_vina",
    "l2_mmgbsa",
    "l3_abfe",
    "min_image",
    "read_solute_frames",
]
