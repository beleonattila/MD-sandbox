"""Branch B, adenylate kinase (E. coli AdK): does a method recover the open<->closed motion?

The ground truth is the crystallographic displacement between the apo open structure
(4AKE) and the closed structure (1AKE, Ap5A removed), on the 214 C-alpha atoms after
least-squares superposition, plus the C-alpha B-factors of each crystal. Structures and
their preparation are shared with branch C (``systems/adk/prep.yml``).

Prediction targets (the ``system`` column; ``<K>`` is ``analysis.n_modes``):

=====================================  ==================================================
``adk_open2closed_overlap_max``        max over the lowest K modes (or top K PCs) of the
                                       open structure of ``|cos(v_k, d)|``, d = closed - open
``adk_open2closed_cumoverlap_k<K>``    ``|P_K d| / |d|``: fraction of the displacement
                                       (as a length) inside the span of those K modes
``adk_closed2open_overlap_max``        the same with modes of the closed structure,
``adk_closed2open_cumoverlap_k<K>``    d = open - closed (L0, L1 only)
``adk_open_bfactor_pearson``           Pearson r between predicted C-alpha mean-square
``adk_closed_bfactor_pearson``         fluctuations and that crystal's C-alpha B-factors
``adk_open_md_min_rmsd_to_closed_A``   L2: closest C-alpha RMSD to 1AKE reached (Angstrom)
``adk_open_md_frac_reached_closed``    pooled MD: fraction of open-start runs that visited
                                       the closed state at least once
=====================================  ==================================================

Overlap rows carry ``reference = 1.0`` (config ``reference``): a displacement lying wholly
inside the predicted subspace scores 1 by definition, so the error is ``1 - overlap``.
B-factor rows have a null reference: crystal B-factors include static disorder and
lattice contacts that no solution model contains, so r = 1 is not the achievable target.
See ``data/experimental/adk_crystal_endpoints.yml``.

Runners (``method``):

- L0 ``null_random_centroid``: random K-dimensional internal subspace; MSF proportional
  to squared distance from the centroid. The thing every method must beat.
- L0 ``anm_ca``: anisotropic network model on C-alpha atoms.
- L1 ``nma_allatom_gbn2``: all-atom normal modes in GBn2 implicit solvent (ff14SB).
- L2 ``pca_short_md``: PCA of one short explicit-solvent MD run from 4AKE.
- L2 ``pooled_short_md`` / L3 ``pooled_long_md``: pooled analysis of a set of MD runs
  (no new MD): transition count, pooled PCA, RMSF.
"""

from __future__ import annotations

import glob
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mdbench.branch_c.adk import (
    KB_KJ_MOL_K,
    _rel,
    build_explicit_system,
    closed_indicator,
    jeffreys_sd,
    read_pdb,
    repo_path,
    trajectory_angles,
)
from mdbench.engines import modes as nm
from mdbench.registry import register, register_scorer

DIRECTIONS: dict[str, tuple[str, str]] = {
    "open2closed": ("open", "closed"),
    "closed2open": ("closed", "open"),
}


# --------------------------------------------------------------------------- endpoints


def crystal_ca_bfactors(raw_pdb: str | Path, chain: str) -> dict[int, float]:
    """C-alpha B-factors (A^2) by residue number for one chain; first altloc only."""
    out: dict[int, float] = {}
    for ln in Path(repo_path(raw_pdb)).read_text().splitlines():
        if not ln.startswith("ATOM") or ln[12:16].strip() != "CA" or ln[21] != chain:
            continue
        if ln[16] not in (" ", "A"):
            continue
        out.setdefault(int(ln[22:26]), float(ln[60:66]))
    return out


def ca_coordinates(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(resids, C-alpha xyz in Angstrom, C-alpha atom indices) of a prepared PDB."""
    from openmm import unit

    pdb = read_pdb(repo_path(path))
    xyz = np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    idx = [a.index for a in pdb.topology.atoms() if a.name == "CA"]
    resids = np.array([int(a.residue.id) for a in pdb.topology.atoms() if a.name == "CA"])
    return resids, xyz[idx], np.asarray(idx)


class Endpoints:
    """C-alpha coordinates and B-factors of every crystallographic copy in the config.

    ``cfg['adk']['endpoints']`` maps state -> {label: prepared PDB}, labels being
    ``<PDB id>_<chain>``; ``cfg['adk']['raw_pdb']`` maps PDB id -> raw file for B-factors.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        a = cfg["adk"]
        self.states: dict[str, list[str]] = {s: list(v) for s, v in a["endpoints"].items()}
        self.paths: dict[str, str] = {
            lab: p for v in a["endpoints"].values() for lab, p in v.items()
        }
        self.xyz: dict[str, np.ndarray] = {}
        self.bfactors: dict[str, np.ndarray] = {}
        resids = None
        for lab, path in self.paths.items():
            r, x, _ = ca_coordinates(path)
            if resids is None:
                resids = r
            elif not np.array_equal(r, resids):
                raise ValueError(f"{lab}: C-alpha residues differ from the other endpoints")
            pdb_id, chain = lab.split("_")
            b = crystal_ca_bfactors(a["raw_pdb"][pdb_id], chain)
            self.xyz[lab] = x
            self.bfactors[lab] = np.array([b[int(i)] for i in r])
        assert resids is not None
        self.resids = resids

    def displacement(self, src: str, tgt: str, frame: np.ndarray | None = None) -> np.ndarray:
        """Flattened tgt - src, both superposed onto ``frame`` (default: the src crystal)."""
        f = self.xyz[src] if frame is None else np.asarray(frame, float)
        return (nm.superpose(self.xyz[tgt], f) - nm.superpose(self.xyz[src], f)).ravel()


def score_subspace(
    ep: Endpoints, src: str, modes_ca: np.ndarray, k: int, frame: np.ndarray | None = None
) -> dict[str, dict[str, Any]]:
    """Overlap scores of C-alpha modes of copy ``src`` against every opposite-state copy.

    Returns ``{direction: {target_label: {overlap_max, cumoverlap, best_mode, overlaps,
    rmsd_A}}}`` for the direction that starts from ``src``'s state.
    """
    state = next(s for s, labs in ep.states.items() if src in labs)
    out: dict[str, dict[str, Any]] = {}
    for direction, (a, b) in DIRECTIONS.items():
        if a != state:
            continue
        out[direction] = {}
        for tgt in ep.states[b]:
            d = ep.displacement(src, tgt, frame)
            ov = nm.overlaps(modes_ca[:, :k], d)
            out[direction][tgt] = {
                "overlap_max": float(ov.max()),
                "cumoverlap": nm.cumulative_overlap(modes_ca, d, k),
                "best_mode": int(np.argmax(ov)) + 1,
                "overlaps": ov.round(4).tolist(),
                "rmsd_A": float(np.sqrt(np.sum(d**2) / (len(d) // 3))),
            }
    return out


def _sd(values: Sequence[float]) -> float:
    v = np.asarray(values, float)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def collect(per_src: Mapping[str, Mapping[str, Any]], k: int) -> dict[str, list[float]]:
    """``{target name: [values over (src, tgt) copy pairs]}`` from per-source scores."""
    vals: dict[str, list[float]] = {}
    for sc in per_src.values():
        for direction, by_tgt in sc["scores"].items():
            for s in by_tgt.values():
                vals.setdefault(f"adk_{direction}_overlap_max", []).append(s["overlap_max"])
                vals.setdefault(f"adk_{direction}_cumoverlap_k{k}", []).append(s["cumoverlap"])
        if "bfactor_r" in sc:
            vals.setdefault(f"adk_{sc['state']}_bfactor_pearson", []).append(sc["bfactor_r"])
    return vals


def make_row(
    cfg: Mapping[str, Any], name: str, prediction: float, uncertainty: float
) -> dict[str, Any]:
    """One result row; overlap targets get the reference from ``cfg['reference']``."""
    row: dict[str, Any] = {
        "system": name,
        "prediction": float(prediction),
        "uncertainty": float(uncertainty),
    }
    if "overlap" in name:
        ref = cfg["reference"]
        row["reference"] = float(ref["overlap_value"])
        row["reference_source"] = str(ref["overlap_source"])
    return row


def evaluate(
    cfg: Mapping[str, Any],
    ep: Endpoints,
    provider: Callable[[str, str], dict[str, Any]],
    sources: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run ``provider(state, label)`` for every source copy and score it.

    The provider returns ``{"modes_ca": (3n, >=K), "msf_ca": (n,), "frame": (n, 3) or
    None, ...diagnostics}``. Result: ``{label: {state, scores, bfactor_r, n, diag}}``.
    """
    k = int(cfg["analysis"]["n_modes"])
    sources = sources or ep.states
    out: dict[str, dict[str, Any]] = {}
    for state, labels in sources.items():
        for lab in labels:
            p = provider(state, lab)
            r = nm.pearson(p["msf_ca"], ep.bfactors[lab])
            out[lab] = {
                "state": state,
                "scores": score_subspace(ep, lab, p["modes_ca"], k, p.get("frame")),
                "bfactor_r": r,
                "n_residues": len(ep.resids),
                "diag": {kk: v for kk, v in p.items() if kk not in ("modes_ca", "msf_ca", "frame")},
            }
    return out


def _json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: np.asarray(o).tolist()))


# --------------------------------------------------------------------------- L0 null


@register(branch="b", system="adk", level="L0", method="null_random_centroid")
def adk_l0_null(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """The null: a random K-dimensional internal subspace, and MSF ~ distance from centre.

    Overlap targets: mean over ``null_model.n_draws`` random orthonormal subspaces (rigid-body
    motion projected out) and over every (source copy, target copy) pair; uncertainty is
    the sd over draws, i.e. the spread of the null itself. For a random direction in D
    dimensions the expected squared cumulative overlap is K/D (D = 3*214 - 6 = 636).

    B-factor target: Pearson r between the crystal B-factors and the squared distance of
    each C-alpha from the centroid (surface residues move more). Uncertainty:
    sqrt(var over copies + Fisher-z standard error of r).
    """
    p = cfg["null_model"]
    k = int(cfg["analysis"]["n_modes"])
    rng = np.random.default_rng(int(p["seed"]))
    ep = Endpoints(cfg)
    draws: dict[str, list[float]] = {}
    for direction, (a, b) in DIRECTIONS.items():
        for src in ep.states[a]:
            ds = [ep.displacement(src, tgt) for tgt in ep.states[b]]
            for _ in range(int(p["n_draws"])):
                q = nm.random_internal_subspace(ep.xyz[src], k, rng)
                for d in ds:
                    draws.setdefault(f"adk_{direction}_overlap_max", []).append(
                        float(nm.overlaps(q, d).max())
                    )
                    draws.setdefault(f"adk_{direction}_cumoverlap_k{k}", []).append(
                        nm.cumulative_overlap(q, d, k)
                    )
    rows = [make_row(cfg, n, np.mean(v), _sd(v)) for n, v in draws.items()]
    diag: dict[str, Any] = {"n_draws": int(p["n_draws"]), "dof_internal": 3 * len(ep.resids) - 6}
    for state, labels in ep.states.items():
        rs = []
        for lab in labels:
            x = ep.xyz[lab]
            msf = np.sum((x - x.mean(axis=0)) ** 2, axis=1)
            rs.append(nm.pearson(msf, ep.bfactors[lab]))
        r = float(np.mean(rs))
        u = math.sqrt(_sd(rs) ** 2 + nm.fisher_se(r, len(ep.resids)) ** 2)
        rows.append(make_row(cfg, f"adk_{state}_bfactor_pearson", r, u))
        diag[f"bfactor_r_{state}"] = dict(zip(labels, rs, strict=True))
    diag["expected_cumoverlap_random"] = math.sqrt(k / diag["dof_internal"])
    _json({"rows": rows, "diag": diag}, run_dir / "l0_null.json")
    return rows


# --------------------------------------------------------------------------- L0 ANM


def anm_provider(cutoff_A: float, gamma: float, k: int, ep: Endpoints) -> Callable:
    """Provider for :func:`evaluate`: ANM on the crystal C-alpha atoms of one copy."""

    def provide(state: str, label: str) -> dict[str, Any]:
        x = ep.xyz[label]
        vals, vecs = nm.internal_modes(nm.anm_hessian(x, cutoff_A, gamma), x)
        return {
            "modes_ca": vecs[:, : max(k, 20)],
            "msf_ca": nm.msf_from_modes(vals, vecs),
            "frame": None,
            "lowest_eigenvalues": vals[:20].round(6).tolist(),
            "n_nonpositive": int(np.sum(vals <= 1e-8)),
        }

    return provide


@register(branch="b", system="adk", level="L0", method="anm_ca")
def adk_l0_anm(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Anisotropic network model on the C-alpha atoms of each crystal copy.

    Prediction: mean over (source copy, target copy) pairs at ``anm.cutoff_A``.
    Uncertainty: sqrt(var over copy pairs + var over ``anm.cutoff_sensitivity_A``), the
    second term being the sd of the copy-mean across the cutoffs (the nominal included);
    B-factor rows add the Fisher-z standard error of r in quadrature.
    """
    p = cfg["anm"]
    k = int(cfg["analysis"]["n_modes"])
    ep = Endpoints(cfg)
    gamma = float(p.get("gamma", 1.0))
    cutoffs = [float(p["cutoff_A"])] + [
        float(c) for c in p.get("cutoff_sensitivity_A", []) if float(c) != float(p["cutoff_A"])
    ]
    per_cutoff = {}
    for c in cutoffs:
        res = evaluate(cfg, ep, anm_provider(c, gamma, k, ep))
        per_cutoff[c] = {"vals": collect(res, k), "detail": res}
    nominal = per_cutoff[cutoffs[0]]["vals"]
    rows = []
    for name, v in nominal.items():
        means = [float(np.mean(per_cutoff[c]["vals"][name])) for c in cutoffs]
        var = _sd(v) ** 2 + _sd(means) ** 2
        if name.endswith("bfactor_pearson"):
            var += nm.fisher_se(float(np.mean(v)), len(ep.resids)) ** 2
        rows.append(make_row(cfg, name, np.mean(v), math.sqrt(var)))
    _json(
        {"cutoffs_A": cutoffs, "per_cutoff": {str(c): d for c, d in per_cutoff.items()}},
        run_dir / "l0_anm.json",
    )
    return rows


# --------------------------------------------------------------------------- L1 NMA


def nma_provider(cfg: Mapping[str, Any], run_dir: Path, ep: Endpoints, state_out: dict) -> Callable:
    """Provider for :func:`evaluate`: all-atom NMA of one minimised crystal copy (GBn2)."""
    import openmm
    from openmm import app, unit

    from mdbench.engines.md import device_name
    from mdbench.engines.nma import fd_hessian, minimise_staged

    p = cfg["nma"]
    k = int(cfg["analysis"]["n_modes"])
    kT = KB_KJ_MOL_K * float(p["temperature_K"])

    def context(system: Any, precision: str) -> Any:
        plat = openmm.Platform.getPlatformByName(str(p["platform"]))
        props = {"Precision": precision} if p["platform"] in ("CUDA", "OpenCL") else {}
        return openmm.Context(system, openmm.VerletIntegrator(0.001), plat, props)

    def provide(state: str, label: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        path = ep.paths[label]
        pdb = read_pdb(repo_path(path))
        ff = app.ForceField(*p["forcefield"])
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        masses = np.array(
            [
                system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(system.getNumParticles())
            ]
        )
        stages = []
        ctx = context(system, str(p["pre_minimise_precision"]))
        ctx.setPositions(pdb.positions)
        stages += minimise_staged(ctx, p["pre_minimise_stages"])
        pos = (
            ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        del ctx
        ctx = context(system, str(p["hessian_precision"]))
        if p["platform"] in ("CUDA", "OpenCL"):
            state_out["gpu"] = device_name(ctx)
        ctx.setPositions(pos)
        stages += minimise_staged(ctx, p["minimise_stages"])
        x_nm = (
            ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        t_min = time.perf_counter() - t0
        hess, asym = fd_hessian(ctx, x_nm, float(p["hessian_step_nm"]))
        del ctx
        t_hess = time.perf_counter() - t0 - t_min
        vals, vecs = nm.internal_modes(hess, x_nm, masses)
        del hess
        msf_nm2 = nm.msf_from_modes(vals, vecs, masses, kT)
        _, ca_ref, ca_idx = ca_coordinates(path)
        pos_idx = np.flatnonzero(vals > 0)
        low = pos_idx[: max(k, int(p.get("n_modes_saved", 50)))]
        cart = nm.cartesian_modes(vecs[:, low], masses)
        modes_ca = nm.restrict(cart, ca_idx)
        x_ca = x_nm[ca_idx] * 10.0
        np.savez_compressed(
            run_dir / f"nma_{label}.npz",
            eigenvalues=vals,
            modes_ca=modes_ca,
            minimised_ca_A=x_ca,
            msf_A2=msf_nm2 * 100.0,
        )
        wn = nm.wavenumbers_cm1(vals)
        return {
            "modes_ca": modes_ca,
            "msf_ca": msf_nm2[ca_idx] * 100.0,
            "frame": x_ca,
            "minimisation": stages,
            "rmsd_minimised_vs_crystal_ca_A": nm.rmsd(nm.superpose(x_ca, ca_ref), ca_ref),
            "n_negative_eigenvalues": int(np.sum(vals < 0)),
            "lowest_wavenumbers_cm1": wn[:20].round(3).tolist(),
            "hessian_asymmetry": asym,
            "ca_mode_norm_fraction": np.linalg.norm(nm.restrict(cart[:, :k], ca_idx), axis=0)
            .round(4)
            .tolist(),
            "wall_clock_s": {
                "minimise": t_min,
                "hessian": t_hess,
                "total": time.perf_counter() - t0,
            },
        }

    return provide


@register(branch="b", system="adk", level="L1", method="nma_allatom_gbn2")
def adk_l1_nma(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """All-atom normal modes of each minimised crystal copy, GBn2 implicit solvent.

    Per copy: minimise (fast mixed-precision stages, then double-precision stages), build
    the central-difference Hessian in double precision, mass-weight, project out rigid
    body, diagonalise fully. The lowest K positive modes are converted to Cartesian
    directions and restricted to the C-alpha atoms (not renormalised per mode; the
    cumulative overlap is a projection, so non-orthogonality does not inflate it).
    MSF uses every positive mode at ``nma.temperature_K``; negative modes (imperfect
    minimum) are dropped and counted in ``l1_nma.json``.

    Uncertainty: sd over (source copy, target copy) pairs; B-factor rows add the
    Fisher-z standard error in quadrature.
    """
    k = int(cfg["analysis"]["n_modes"])
    ep = Endpoints(cfg)
    gpu: dict[str, str] = {}
    res = evaluate(cfg, ep, nma_provider(cfg, run_dir, ep, gpu))
    rows = []
    for name, v in collect(res, k).items():
        var = _sd(v) ** 2
        if name.endswith("bfactor_pearson"):
            var += nm.fisher_se(float(np.mean(v)), len(ep.resids)) ** 2
        rows.append(make_row(cfg, name, np.mean(v), math.sqrt(var)))
    _json(res, run_dir / "l1_nma.json")
    for r in rows:
        r["gpu"] = gpu.get("gpu", "none")
    return rows


# --------------------------------------------------------------------------- L2 MD PCA


def ca_frames(run: Path, resids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(time_ps, C-alpha xyz in Angstrom) of an MD run dir: ``ca_traj.npz`` or the XTC."""
    cache = run / "ca_traj.npz"
    if cache.exists():
        z = np.load(cache)
        return z["time_ps"], z["xyz_A"].astype(float)
    import MDAnalysis as mda

    u = mda.Universe(str(run / "topology.pdb"), str(run / "traj.xtc"))
    ca = u.select_atoms("protein and name CA")
    if not np.array_equal(ca.resids, resids):
        raise ValueError(f"{run}: C-alpha residues do not match the endpoints")
    t, xyz = [], []
    for ts in u.trajectory:
        t.append(float(ts.time))
        xyz.append(ca.positions.copy())
    return np.asarray(t), np.asarray(xyz, float)


def md_subspace(
    cfg: Mapping[str, Any], ep: Endpoints, frames: np.ndarray, src: str
) -> dict[str, Any]:
    """PCA scores of superposed C-alpha ``frames`` against the crystal displacement."""
    k = int(cfg["analysis"]["n_modes"])
    fit = nm.iterative_superpose(frames, ep.xyz[src])
    var, pcs, msf = nm.pca(fit)
    mean = fit.mean(axis=0)
    scores = score_subspace(ep, src, pcs, k, frame=mean)
    return {
        "scores": scores,
        "bfactor_r": nm.pearson(msf, ep.bfactors[src]),
        "variance_fraction_top": (var[:k] / var.sum()).round(4).tolist(),
        "fit": fit,
    }


def min_rmsd_to(frames: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame C-alpha RMSD (A) to ``target`` after optimal superposition."""
    return np.array([nm.rmsd(nm.superpose(x, target), target) for x in frames])


def _half_spread(a: float, b: float) -> float:
    return abs(a - b) / 2.0


@register(branch="b", system="adk", level="L2", method="pca_short_md")
def adk_l2_pca(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """One short explicit-solvent MD run from the open crystal; PCA, RMSF, closest approach.

    Frames after ``analysis.discard_ps`` are superposed (C-alpha, iterated onto the mean)
    and diagonalised. Overlap prediction: mean over the closed target copies.
    Uncertainty: sqrt(var over target copies + (half-split difference / 2)^2), where the
    half-split is the same estimate on the first and second half of the run; that term is
    also the convergence diagnostic. B-factor: Fisher-z SE and half-split in quadrature.
    The replicate-to-replicate spread is in the pooled runner, not here.

    ``ca_traj.npz`` (C-alpha only, all frames) and ``cv_timeseries.csv`` (domain angles)
    are written so the analysis survives deletion of the trajectory.
    """
    from mdbench.engines.md import run_md

    ep = Endpoints(cfg)
    src = str(cfg["adk"]["md_start_copy"])
    top, system, pos, box = build_explicit_system(cfg)
    res = run_md(top, system, pos, cfg["md"], run_dir, box_vectors=box)
    if not res.trajectory:
        raise RuntimeError("pca_short_md needs a trajectory (md.traj_interval_steps > 0)")
    t, xyz = ca_frames(run_dir, ep.resids)
    np.savez_compressed(run_dir / "ca_traj.npz", time_ps=t, xyz_A=xyz.astype(np.float32))
    cv = trajectory_angles(res.topology, res.trajectory)
    np.savetxt(run_dir / "cv_timeseries.csv", cv, delimiter=",",
               header="time_ps,theta_nmp_deg,theta_lid_deg", comments="")  # fmt: skip
    rows, diag = analyse_single_run(cfg, ep, src, t, xyz, cv)
    diag["md_result"] = json.loads((run_dir / "md_result.json").read_text())
    _json(diag, run_dir / "analysis.json")
    for r in rows:
        r["gpu"] = res.gpu
    return rows


def analyse_single_run(
    cfg: Mapping[str, Any],
    ep: Endpoints,
    src: str,
    t: np.ndarray,
    xyz: np.ndarray,
    cv: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows and diagnostics of :func:`adk_l2_pca` from C-alpha frames and domain angles."""
    an = cfg["analysis"]
    k = int(an["n_modes"])
    keep = t >= float(an.get("discard_ps", 0.0))
    frames = xyz[keep]
    if len(frames) < 4:
        raise RuntimeError(f"only {len(frames)} frames after discard_ps")
    half = len(frames) // 2
    full = md_subspace(cfg, ep, frames, src)
    halves = [md_subspace(cfg, ep, f, src) for f in (frames[:half], frames[half:])]
    if src not in ep.states["open"]:
        raise ValueError(f"MD analysis expects an open-state start copy, got {src}")
    rows = []
    direction = "open2closed"
    for key, name in (("overlap_max", "overlap_max"), ("cumoverlap", f"cumoverlap_k{k}")):
        v = [s[key] for s in full["scores"][direction].values()]
        h = [float(np.mean([s[key] for s in x["scores"][direction].values()])) for x in halves]
        u = math.sqrt(_sd(v) ** 2 + _half_spread(*h) ** 2)
        rows.append(make_row(cfg, f"adk_{direction}_{name}", np.mean(v), u))
    r = full["bfactor_r"]
    u = math.sqrt(nm.fisher_se(r, len(ep.resids)) ** 2
                  + _half_spread(halves[0]["bfactor_r"], halves[1]["bfactor_r"]) ** 2)  # fmt: skip
    rows.append(make_row(cfg, "adk_open_bfactor_pearson", r, u))
    closed = ep.states["closed"]
    rm = np.min([min_rmsd_to(xyz, ep.xyz[c]) for c in closed], axis=0)
    hm = [float(rm[: len(rm) // 2].min()), float(rm[len(rm) // 2 :].min())]
    rows.append(make_row(cfg, "adk_open_md_min_rmsd_to_closed_A", float(rm.min()),
                         _half_spread(*hm)))  # fmt: skip
    ind = closed_indicator(cv[:, 1], cv[:, 2], an["closed"])
    diag = {
        "n_frames": len(t),
        "n_frames_analysed": int(keep.sum()),
        "scores": full["scores"],
        "bfactor_r": r,
        "halves": [{"scores": x["scores"], "bfactor_r": x["bfactor_r"]} for x in halves],
        "pc_variance_fraction_top": full["variance_fraction_top"],
        "rmsd_to_closed_min_A": float(rm.min()),
        "rmsd_to_closed_final_A": float(rm[-1]),
        "rmsd_to_open_crystal_final_A": float(min_rmsd_to(xyz[-1:], ep.xyz[src])[0]),
        "visited_closed": bool(ind.any()),
        "fraction_frames_closed": float(ind.mean()),
        "theta_nmp_min_deg": float(cv[:, 1].min()),
        "theta_lid_min_deg": float(cv[:, 2].min()),
    }
    return rows, diag


# --------------------------------------------------------------------------- pooled MD


def _run_dirs(patterns: Sequence[str]) -> list[Path]:
    hits = sorted({p for pat in patterns for p in glob.glob(str(repo_path(pat)))})
    return [Path(h) for h in hits if Path(h).is_dir()]


def _start_structure(run: Path) -> str:
    cfg = yaml.safe_load((run / "config.resolved.yml").read_text())
    return str(cfg["adk"]["solvated_pdb"])


@register(branch="b", system="adk", level="L2", method="pooled_short_md")
@register(branch="b", system="adk", level="L3", method="pooled_long_md")
def adk_pooled_md(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Pooled analysis of finished MD runs (no new MD): how many saw the transition?

    ``pooled.runs`` are glob patterns of run directories (``pca_short_md`` or branch C
    ``unbiased_md`` layout); only runs whose start structure contains
    ``pooled.start_structure`` are used, and every run must be complete
    (``md_result.json`` present). A run "reached closed" if any production frame is
    inside the branch-C closed box (``analysis.closed``, both domain angles).

    Rows: the fraction of runs that reached closed (uncertainty: Jeffreys posterior sd
    with n = number of runs), and the overlap and B-factor targets from PCA of all runs'
    frames pooled (uncertainty: bootstrap sd over runs, ``pooled.n_boot``; B-factor adds
    the Fisher-z SE). ``wall_clock_s`` is the summed MD wall clock of the pooled runs
    plus this analysis, because that is what the estimate cost.
    """
    p = cfg["pooled"]
    an = cfg["analysis"]
    k = int(an["n_modes"])
    t0 = time.perf_counter()
    ep = Endpoints(cfg)
    src = str(cfg["adk"]["md_start_copy"])
    runs, skipped = [], {}
    for d in _run_dirs(p["runs"]):
        if not (d / "md_result.json").exists():
            skipped[_rel(d)] = "incomplete (no md_result.json)"
        elif str(p["start_structure"]) not in _start_structure(d):
            skipped[_rel(d)] = f"start {_start_structure(d)}"
        else:
            runs.append(d)
    if len(runs) < int(p.get("min_runs", 2)):
        raise RuntimeError(f"pooled analysis needs >= {p.get('min_runs', 2)} runs, got {runs}")
    per_run, frames, md_wall, gpus = [], [], 0.0, set()
    for d in runs:
        t, xyz = ca_frames(d, ep.resids)
        cvf = d / "cv_timeseries.csv"
        cv = np.loadtxt(cvf, delimiter=",", skiprows=1, ndmin=2) if cvf.exists() else (
            trajectory_angles(d / "topology.pdb", d / "traj.xtc"))  # fmt: skip
        ind = closed_indicator(cv[:, 1], cv[:, 2], an["closed"])
        rm = np.min([min_rmsd_to(xyz, ep.xyz[c]) for c in ep.states["closed"]], axis=0)
        mdr = json.loads((d / "md_result.json").read_text())
        md_wall += float(mdr["wall_clock_s"])
        gpus.add(str(mdr.get("gpu", "none")))
        keep = t >= float(an.get("discard_ps", 0.0))
        frames.append(xyz[keep])
        per_run.append(
            {
                "run": _rel(d),
                "n_frames": len(t),
                "duration_ns": float(t[-1] - t[0]) / 1000.0 if len(t) > 1 else 0.0,
                "visited_closed": bool(ind.any()),
                "fraction_frames_closed": float(ind.mean()),
                "first_closed_ps": float(cv[np.argmax(ind > 0), 0]) if ind.any() else None,
                "rmsd_to_closed_min_A": float(rm.min()),
            }
        )
    n = len(per_run)
    hits = sum(r["visited_closed"] for r in per_run)
    rows = [make_row(cfg, "adk_open_md_frac_reached_closed", hits / n, jeffreys_sd(hits, n))]

    def estimate(fs: list[np.ndarray]) -> dict[str, float]:
        m = md_subspace(cfg, ep, np.concatenate(fs), src)
        sc = m["scores"]["open2closed"].values()
        return {
            "adk_open2closed_overlap_max": float(np.mean([s["overlap_max"] for s in sc])),
            f"adk_open2closed_cumoverlap_k{k}": float(np.mean([s["cumoverlap"] for s in sc])),
            "adk_open_bfactor_pearson": m["bfactor_r"],
        }

    point = estimate(frames)
    rng = np.random.default_rng(int(p["seed"]))
    boots = [estimate([frames[i] for i in rng.integers(0, n, n)]) for _ in range(int(p["n_boot"]))]
    for name, v in point.items():
        u = _sd([b[name] for b in boots])
        if name.endswith("bfactor_pearson"):
            u = math.sqrt(u**2 + nm.fisher_se(v, len(ep.resids)) ** 2)
        rows.append(make_row(cfg, name, v, u))
    wall = md_wall + (time.perf_counter() - t0)
    gpu = ",".join(sorted(gpus)) or "none"
    _json({"runs": per_run, "skipped": skipped, "n_reached_closed": hits, "n_runs": n,
           "point": point, "md_wall_clock_s": md_wall}, run_dir / "analysis.json")  # fmt: skip
    for r in rows:
        r["wall_clock_s"] = wall
        r["gpu"] = gpu
    return rows


# --------------------------------------------------------------------------- scorer


@register_scorer("b")
def score_b(df: Any, n_boot: int = 1000, seed: int = 0) -> Any:
    """Per (system, level, method): the estimate, its spread, the null beside it, the cost.

    Overlap and B-factor correlations are scores in themselves, so the table reports the
    level's value next to the L0 null's value for the same target (``null_prediction``)
    and the difference in units of the combined uncertainty (``z_vs_null``). For targets
    with a reference the error ``1 - overlap`` is the ``abs_err`` column.
    """
    import pandas as pd

    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["system", "level", "method"], sort=True)
    out = g.agg(
        n=("prediction", "size"),
        prediction=("prediction", "mean"),
        sd_replicates=("prediction", "std"),
        uncertainty=("uncertainty", "mean"),
        reference=("reference", "first"),
        wall_clock_s=("wall_clock_s", "sum"),
    ).reset_index()
    # With one row the spread across replicates is unknown, not zero.
    out["abs_err"] = (out["prediction"] - out["reference"]).abs()
    null = out[out["method"].str.startswith("null")].set_index("system")
    out["null_prediction"] = out["system"].map(null["prediction"])
    nu = out["system"].map(null["uncertainty"])
    out["z_vs_null"] = (out["prediction"] - out["null_prediction"]) / np.sqrt(
        out["uncertainty"] ** 2 + nu**2
    )
    return out.sort_values(["system", "level", "method"]).reset_index(drop=True)


__all__ = [
    "DIRECTIONS",
    "Endpoints",
    "analyse_single_run",
    "ca_coordinates",
    "crystal_ca_bfactors",
    "evaluate",
    "md_subspace",
    "score_subspace",
]
