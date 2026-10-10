"""Alanine dipeptide (ACE-ALA-NME), branch C, levels L0-L3.

Observable: backbone (phi, psi) free energy surface and the populations of the basins
defined in the config (``analysis.basins``). Every runner returns, per basin, a row
``system = "ala2.pop.<basin>"`` (fraction) and, for every non-reference basin, a row
``system = "ala2.dG.<basin>"`` (G_basin - G_reference_basin, kcal/mol). The results
schema has no "observable" column, so the observable rides in ``system`` (the CLI allows
a runner to override ``system``; the scorer groups by it).

Levels (method names are the registry keys):

- L0 ``boltzmann_minima``     minimise from a grid of starts (implicit solvent), Boltzmann
                              over the energies of the distinct minima found. No sampling.
- L1 ``static_cv_harmonic``   static CV values (phi, psi) of those minima, plus harmonic
                              basin free energies from the relaxed (phi, psi) Hessian.
- L2 ``md_histogram``         histogram over unbiased explicit-solvent MD, N replicates.
- L3 ``wtmetad_reweight``     well-tempered metadynamics on phi/psi (PLUMED; OPES is not
                              compiled into this PLUMED build), c(t) reweighting.
- L3 ``umbrella_mbar_phi``    umbrella sampling along phi, MBAR (pymbar 4).
- L3 ``msm_deeptime``         MSM on the L2 trajectories: ITS, CK test, Bayesian errors.

Every level writes a figure ``reports/c_ala2_<level>_<method>.png`` that shows the FES (or
the closest thing the method has) and its convergence diagnostic in the same figure.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mdbench import REPO_ROOT
from mdbench.engines import fes
from mdbench.engines import plumed_cv as cv
from mdbench.registry import register

log = logging.getLogger(__name__)

SYSTEM = "ala2"


# =========================================================================== helpers


def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _reports_dir(cfg: Mapping[str, Any]) -> Path:
    d = _repo_path(cfg.get("analysis", {}).get("reports_dir", "reports"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _basins(cfg: Mapping[str, Any]) -> tuple[dict[str, list[list[float]]], list[str], int]:
    a = cfg["analysis"]
    basins = {str(k): [list(map(float, r)) for r in v] for k, v in a["basins"].items()}
    names = fes.validate_basins(basins)
    ref = names.index(str(a["reference_basin"]))
    return basins, names, ref


def _kT(cfg: Mapping[str, Any]) -> float:
    return fes.kT_kj(float(cfg["md"]["temperature_K"]))


def observable_rows(
    names: Sequence[str],
    ref: int,
    pops: np.ndarray,
    pop_err: np.ndarray,
    kT: float,
    dg_kj: np.ndarray | None = None,
    dg_err_kj: np.ndarray | None = None,
    **common: Any,
) -> list[dict[str, Any]]:
    """Rows for basin populations and ΔG (kcal/mol) vs the reference basin.

    ΔG defaults to ``-kT ln(p_i/p_ref)`` with a delta-method error. A basin with exactly
    zero population (never sampled, or outside the MSM's connected set) gets no rows at
    all: ``0 ± 0`` would claim a certainty the data do not have, and its ΔG is infinite.
    Any other non-finite row is dropped too. Dropped rows are logged as warnings.
    """
    pops = np.asarray(pops, dtype=float)
    pop_err = np.asarray(pop_err, dtype=float)
    if dg_kj is None:
        dg_kj = fes.delta_g_kj(pops, ref, kT)
        dg_err_kj = fes.delta_g_err_kj(pops, pop_err, ref, kT)
    rows: list[dict[str, Any]] = []
    for i, n in enumerate(names):
        if not pops[i] > 0:
            log.warning("basin %s unvisited (population 0): no pop/dG rows emitted", n)
            continue
        rows.append(
            {"system": f"{SYSTEM}.pop.{n}", "prediction": pops[i], "uncertainty": pop_err[i]}
        )
    for i, n in enumerate(names):
        if i == ref:
            continue
        rows.append(
            {
                "system": f"{SYSTEM}.dG.{n}",
                "prediction": float(fes.as_kcal(dg_kj[i])),
                "uncertainty": float(fes.as_kcal(dg_err_kj[i])),
            }
        )
    out = []
    for r in rows:
        if np.isfinite(r["prediction"]) and np.isfinite(r["uncertainty"]):
            out.append({**r, **common})
        else:
            log.warning("dropping non-finite row %s", r)
    return out


def _write_json(path: Path, obj: Any) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default))


# =========================================================================== system prep


def load_peptide(prep: Mapping[str, Any]):
    """The built ACE-ALA-NME structure (tleap output, see systems/ala2/prep.yml)."""
    from openmm import app

    return app.PDBFile(str(_repo_path(prep["structure"])))


def build_solvated(prep: Mapping[str, Any]):
    """Explicit-solvent system: ``(topology, system, positions, box_vectors)``.

    Deterministic (Modeller.addSolvent places a fixed water lattice), so every level and
    replicate starts from the same atoms.
    """
    import openmm
    from openmm import app, unit

    pdb = load_peptide(prep)
    ff = app.ForceField(*prep["forcefield"])
    m = app.Modeller(pdb.topology, pdb.positions)
    box = float(prep["box_nm"])
    m.addSolvent(
        ff,
        model=str(prep["water_model"]),
        boxSize=openmm.Vec3(box, box, box) * unit.nanometer,
        ionicStrength=float(prep.get("ionic_strength_M", 0.0)) * unit.molar,
        neutralize=bool(prep.get("neutralize", True)),
    )
    system = ff.createSystem(
        m.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=float(prep["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=getattr(app, str(prep["constraints"])),
        rigidWater=bool(prep.get("rigid_water", True)),
        ewaldErrorTolerance=float(prep.get("ewald_error_tolerance", 5e-4)),
        hydrogenMass=(
            float(prep["hydrogen_mass_amu"]) * unit.amu if prep.get("hydrogen_mass_amu") else None
        ),
    )
    return m.topology, system, m.positions, m.topology.getPeriodicBoxVectors()


def _quads(topology: Any) -> dict[str, cv.Quad]:
    return cv.backbone_quads(cv.topology_atoms(topology))


def _md_cfg(cfg: Mapping[str, Any], **override: Any) -> dict[str, Any]:
    md = copy.deepcopy(dict(cfg["md"]))
    md.update(override)
    return md


def trajectory_cvs(run_dir: Path) -> dict[str, np.ndarray]:
    """phi/psi (radians) per frame from ``run_dir/{topology.pdb,traj.xtc}``; cached."""
    cache = run_dir / "cv.npz"
    if cache.exists() and cache.stat().st_mtime >= (run_dir / "traj.xtc").stat().st_mtime:
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    import MDAnalysis as mda
    from openmm import app

    top = app.PDBFile(str(run_dir / "topology.pdb")).topology
    quads = _quads(top)
    u = mda.Universe(str(run_dir / "topology.pdb"), str(run_dir / "traj.xtc"))
    idx = sorted({i for q in quads.values() for i in q})
    pos = np.array([u.atoms.positions[idx] for _ in u.trajectory])
    remap = {a: j for j, a in enumerate(idx)}
    local = {k: tuple(remap[i] for i in q) for k, q in quads.items()}
    out = cv.dihedrals_from_frames(pos, local)
    out["time_ps"] = np.array([ts.time for ts in u.trajectory])
    np.savez(cache, **out)
    return out


# =========================================================================== L0 / L1


class _RestrainedMinimizer:
    """Small-molecule minimiser with optional harmonic phi/psi restraints (CPU)."""

    def __init__(self, cfg: Mapping[str, Any]):
        import openmm
        from openmm import app

        l0 = cfg["minima"]
        pdb = load_peptide(cfg["system_prep"])
        self.topology = pdb.topology
        ff = app.ForceField(*l0["forcefield"])
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=getattr(app, str(l0["constraints"])) if l0.get("constraints") else None,
        )
        self.quads = _quads(pdb.topology)
        r = openmm.CustomTorsionForce(
            "0.5*k_res*dd^2; dd=atan2(sin(theta-theta0), cos(theta-theta0))"
        )
        r.addGlobalParameter("k_res", 0.0)
        r.addPerTorsionParameter("theta0")
        for q in self.quads.values():
            r.addTorsion(*q, [0.0])
        r.setForceGroup(1)
        system.addForce(r)
        self.restraint = r
        platform = openmm.Platform.getPlatformByName(str(l0.get("platform", "CPU")))
        self.ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
        self.start = pdb.positions
        self.tol = float(l0["minimize_tolerance_kj_mol_nm"])
        self.max_iter = int(l0.get("minimize_max_iterations", 0))

    def angles(self) -> np.ndarray:
        from openmm import unit

        x = (
            self.ctx.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )
        return np.array([cv.dihedral(x[list(q)]) for q in self.quads.values()])

    def energy(self) -> float:
        from openmm import unit

        st = self.ctx.getState(getEnergy=True, groups={0})
        return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def minimize(
        self, targets_rad: Sequence[float] | None, k_res: float, positions: Any = None
    ) -> tuple[np.ndarray, float, Any]:
        """Minimise (restrained to ``targets`` if given); returns (angles, E_unrestrained, pos)."""
        import openmm

        self.ctx.setPositions(self.start if positions is None else positions)
        if targets_rad is not None:
            for i, (q, t) in enumerate(zip(self.quads.values(), targets_rad, strict=True)):
                self.restraint.setTorsionParameters(i, *q, [float(t)])
            self.restraint.updateParametersInContext(self.ctx)
            self.ctx.setParameter("k_res", float(k_res))
        else:
            self.ctx.setParameter("k_res", 0.0)
        openmm.LocalEnergyMinimizer.minimize(self.ctx, self.tol, self.max_iter)
        pos = self.ctx.getState(getPositions=True).getPositions()
        return self.angles(), self.energy(), pos


def find_minima(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Grid of (phi, psi) starts -> restrained pre-minimisation -> free minimisation.

    Returns per-start final angles/energies, the distinct minima (clustered by angle and
    energy) and each start's minimum id.
    """
    l0 = cfg["minima"]
    mz = _RestrainedMinimizer(cfg)
    step = float(l0["start_grid_deg"])
    grid = np.arange(-180.0 + step / 2, 180.0, step)
    starts, finals, energies, positions = [], [], [], []
    for phi0 in grid:
        for psi0 in grid:
            _, _, pos = mz.minimize(np.radians([phi0, psi0]), float(l0["k_pre_kj_rad2"]))
            ang, e, pos = mz.minimize(None, 0.0, pos)
            starts.append((phi0, psi0))
            finals.append(np.degrees(ang))
            energies.append(e)
            positions.append(pos)
    finals_a = np.array(finals)
    energies_a = np.array(energies)
    tol_deg = float(l0["cluster_tol_deg"])
    tol_e = float(l0["cluster_tol_kj_mol"])
    minima: list[dict[str, Any]] = []
    member = np.full(len(finals_a), -1, dtype=int)
    for i in np.argsort(energies_a):
        for mid, m in enumerate(minima):
            d = np.abs(fes.wrap180(finals_a[i] - m["angles_deg"]))
            if np.all(d < tol_deg) and abs(energies_a[i] - m["energy_kj"]) < tol_e:
                member[i] = mid
                break
        else:
            member[i] = len(minima)
            minima.append(
                {"angles_deg": finals_a[i], "energy_kj": float(energies_a[i]), "pos": positions[i]}
            )
    return {
        "minimizer": mz,
        "starts_deg": np.array(starts),
        "finals_deg": finals_a,
        "energies_kj": energies_a,
        "member": member,
        "minima": minima,
    }


def _basin_of_minima(minima: Sequence[Mapping[str, Any]], basins: fes.Basins) -> np.ndarray:
    ang = np.array([m["angles_deg"] for m in minima])
    return fes.assign_basins(ang[:, 0], ang[:, 1], basins)


def _superposition_pops(
    log_z: np.ndarray, basin_of: np.ndarray, found: np.ndarray, n_basins: int
) -> np.ndarray:
    """Basin populations from per-minimum log partition functions, found-minima only."""
    z = np.zeros(n_basins)
    lz = np.where(found, log_z, -np.inf)
    lz = lz - np.max(lz[np.isfinite(lz)])
    for b in range(n_basins):
        sel = (basin_of == b) & found
        z[b] = np.exp(lz[sel]).sum() if sel.any() else 0.0
    return z / z.sum()


def _bootstrap_pops(
    log_z: np.ndarray,
    basin_of: np.ndarray,
    member: np.ndarray,
    n_basins: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(member)
    out = np.empty((n_boot, n_basins))
    for b in range(n_boot):
        found = np.zeros(len(log_z), dtype=bool)
        found[np.unique(member[rng.integers(0, n, n)])] = True
        out[b] = _superposition_pops(log_z, basin_of, found, n_basins)
    return out


def relaxed_hessian(
    mz: _RestrainedMinimizer, minimum: Mapping[str, Any], h_deg: float, k_stiff: float
) -> tuple[np.ndarray, float]:
    """Relaxed (phi, psi) Hessian (kJ/mol/rad^2) at a minimum by a 3x3 restrained scan.

    Fits ``E = c + g.d + 1/2 d^T H d`` by least squares to the unrestrained energy at the
    *achieved* angles (the stiff restraint does not land exactly on target). Returns
    ``(H, rms_fit_residual_kj)``.
    """
    a0 = np.radians(minimum["angles_deg"])
    h = np.radians(h_deg)
    d_list, e_list = [], []
    for dp in (-h, 0.0, h):
        for ds in (-h, 0.0, h):
            ang, e, _ = mz.minimize(a0 + np.array([dp, ds]), k_stiff, minimum["pos"])
            d = cv.angle_diff(ang, a0)
            d_list.append(d)
            e_list.append(e)
    d = np.array(d_list)
    e = np.array(e_list)
    a = np.column_stack(
        [
            np.ones(len(d)),
            d[:, 0],
            d[:, 1],
            0.5 * d[:, 0] ** 2,
            d[:, 0] * d[:, 1],
            0.5 * d[:, 1] ** 2,
        ]
    )
    coef, *_ = np.linalg.lstsq(a, e, rcond=None)
    hess = np.array([[coef[3], coef[4]], [coef[4], coef[5]]])
    resid = float(np.sqrt(np.mean((a @ coef - e) ** 2)))
    return hess, resid


def _minima_figure(
    cfg: Mapping[str, Any],
    path: Path,
    found: Mapping[str, Any],
    basins: fes.Basins,
    names: Sequence[str],
    boot: np.ndarray,
    pops: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    e = np.array([m["energy_kj"] for m in found["minima"]])
    ang = np.array([m["angles_deg"] for m in found["minima"]])
    rel = fes.as_kcal(e - e.min())
    ax[0].scatter(
        found["starts_deg"][:, 0], found["starts_deg"][:, 1], s=8, c="#c3c2b7", label="starts"
    )
    sc = ax[0].scatter(ang[:, 0], ang[:, 1], c=rel, cmap="Blues_r", s=70, edgecolors="#0b0b0b")
    fig.colorbar(sc, ax=ax[0], label="E - E_min of minimum (kcal/mol)")
    _draw_basins(ax[0], basins)
    ax[0].set(xlim=(-180, 180), ylim=(-180, 180), xlabel="phi (deg)", ylabel="psi (deg)")
    ax[0].set_title("Minima found (implicit solvent)")
    colors = _series_colors(len(names))
    parts = ax[1].violinplot([boot[:, i] for i in range(len(names))], showextrema=False)
    for body, c in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(c)
        body.set_alpha(0.45)
    ax[1].scatter(range(1, len(names) + 1), pops, c=colors, s=40, zorder=3, edgecolors="#0b0b0b")
    ax[1].set_xticks(range(1, len(names) + 1), names)
    ax[1].set_ylabel("population")
    ax[1].set_title("Convergence: bootstrap over starts (violin), point = all starts")
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _series_colors(n: int) -> list[str]:
    # Fixed categorical order (dataviz reference palette, light mode).
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    return pal[:n]


def _draw_basins(ax: Any, basins: fes.Basins) -> None:
    from matplotlib.patches import Rectangle

    for (name, rects), c in zip(basins.items(), _series_colors(len(basins)), strict=True):
        for j, r in enumerate(rects):
            ax.add_patch(
                Rectangle(
                    (r[0], r[2]),
                    r[1] - r[0],
                    r[3] - r[2],
                    fill=False,
                    ec=c,
                    lw=1.5,
                    ls="--",
                    label=name if j == 0 else None,
                )
            )
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)


@register(branch="c", system=SYSTEM, level="L0", method="boltzmann_minima")
def run_l0(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """L0: Boltzmann weights of minimised basin representatives (energies only)."""
    t0 = time.perf_counter()
    basins, names, ref = _basins(cfg)
    kT = _kT(cfg)
    found = find_minima(cfg)
    minima = found["minima"]
    basin_of = _basin_of_minima(minima, basins)
    log_z = -np.array([m["energy_kj"] for m in minima]) / kT
    all_found = np.ones(len(minima), dtype=bool)
    pops = _superposition_pops(log_z, basin_of, all_found, len(names))
    boot = _bootstrap_pops(
        log_z,
        basin_of,
        found["member"],
        len(names),
        int(cfg["minima"]["n_bootstrap"]),
        int(cfg["minima"]["bootstrap_seed"]),
    )
    err = boot.std(axis=0, ddof=1)
    _write_json(
        run_dir / "minima.json",
        {
            "names": names,
            "minima": [
                {"phi_psi_deg": m["angles_deg"], "energy_kj": m["energy_kj"], "basin": int(b)}
                for m, b in zip(minima, basin_of, strict=True)
            ],
            "pops": pops,
            "pop_bootstrap_std": err,
        },
    )
    fig = _reports_dir(cfg) / "c_ala2_L0_boltzmann_minima.png"
    _minima_figure(cfg, fig, found, basins, names, boot, pops, "ala2 L0: Boltzmann over minima")
    return observable_rows(
        names, ref, pops, err, kT, gpu="none", wall_clock_s=time.perf_counter() - t0
    )


@register(branch="c", system=SYSTEM, level="L1", method="static_cv_harmonic")
def run_l1(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """L1: static (phi, psi) of each basin's lowest minimum + harmonic basin populations.

    phi/psi rows: ``system = ala2.phi.<basin>`` / ``ala2.psi.<basin>`` in degrees, with
    the thermal width ``sqrt(kT (H^-1)_ii)`` of the well as the uncertainty. Populations:
    harmonic superposition ``Z_m = exp(-E_m/kT) 2 pi kT / sqrt(det H_m)`` over minima,
    uncertainty = bootstrap over starts combined with the Hessian-step sensitivity.
    """
    t0 = time.perf_counter()
    basins, names, ref = _basins(cfg)
    kT = _kT(cfg)
    l1 = cfg["harmonic"]
    found = find_minima(cfg)
    mz, minima = found["minimizer"], found["minima"]
    basin_of = _basin_of_minima(minima, basins)
    steps = [float(h) for h in l1["hessian_step_deg"]]
    log_z_by_h, hess_main, resid_main = [], [], []
    for j, h in enumerate(steps):
        lz = []
        for m in minima:
            hess, resid = relaxed_hessian(mz, m, h, float(l1["k_stiff_kj_rad2"]))
            det = np.linalg.det(hess)
            pd = det > 0 and hess[0, 0] > 0
            # A non-positive-definite fit means a saddle/flat direction at this resolution;
            # fall back to the energy-only weight (logged in the run dir).
            lz.append(-m["energy_kj"] / kT + (np.log(2 * np.pi * kT / np.sqrt(det)) if pd else 0))
            if j == 0:
                hess_main.append(hess)
                resid_main.append(resid)
        log_z_by_h.append(np.array(lz))
    all_found = np.ones(len(minima), dtype=bool)
    pops_by_h = np.array(
        [_superposition_pops(lz, basin_of, all_found, len(names)) for lz in log_z_by_h]
    )
    pops = pops_by_h[0]
    boot = _bootstrap_pops(
        log_z_by_h[0],
        basin_of,
        found["member"],
        len(names),
        int(cfg["minima"]["n_bootstrap"]),
        int(cfg["minima"]["bootstrap_seed"]),
    )
    h_sens = np.max(np.abs(pops_by_h - pops), axis=0)
    err = np.sqrt(boot.std(axis=0, ddof=1) ** 2 + h_sens**2)
    rows = observable_rows(names, ref, pops, err, kT, gpu="none")
    # Static CV values of each basin's representative (lowest-energy minimum).
    e = np.array([m["energy_kj"] for m in minima])
    for b, n in enumerate(names):
        sel = np.where(basin_of == b)[0]
        if not len(sel):
            continue
        i = sel[np.argmin(e[sel])]
        hess = hess_main[i]
        try:
            cov = kT * np.linalg.inv(hess)
            width = np.degrees(np.sqrt(np.clip(np.diag(cov), 0, None)))
        except np.linalg.LinAlgError:
            width = np.array([np.nan, np.nan])
        for k, label in enumerate(("phi", "psi")):
            if np.isfinite(width[k]) and np.linalg.det(hess) > 0:
                rows.append(
                    {
                        "system": f"{SYSTEM}.{label}.{n}",
                        "prediction": float(minima[i]["angles_deg"][k]),
                        "uncertainty": float(width[k]),
                        "gpu": "none",
                    }
                )
    _write_json(
        run_dir / "harmonic.json",
        {
            "names": names,
            "hessian_steps_deg": steps,
            "minima": [
                {
                    "phi_psi_deg": m["angles_deg"],
                    "energy_kj": m["energy_kj"],
                    "basin": int(b),
                    "hessian_kj_rad2": hm,
                    "fit_rms_kj": r,
                }
                for m, b, hm, r in zip(minima, basin_of, hess_main, resid_main, strict=True)
            ],
            "pops_by_step": pops_by_h,
            "pops": pops,
            "pop_err": err,
        },
    )
    fig = _reports_dir(cfg) / "c_ala2_L1_static_cv_harmonic.png"
    _minima_figure(cfg, fig, found, basins, names, boot, pops, "ala2 L1: harmonic basins")
    wall = time.perf_counter() - t0
    for r in rows:
        r["wall_clock_s"] = wall
    return rows


# =========================================================================== L2


def _fes_axes(
    ax: Any, phi_deg: np.ndarray, psi_deg: np.ndarray, kT: float, bins: int, basins: fes.Basins,
    weights: np.ndarray | None = None, title: str = "", vmax_kcal: float = 8.0,
) -> None:  # fmt: skip
    f, edges = fes.histogram_fes(phi_deg, psi_deg, kT, bins, weights)
    fk = fes.as_kcal(f)
    im = ax.pcolormesh(
        edges, edges, np.ma.masked_invalid(fk).T, cmap="Blues_r", vmin=0, vmax=vmax_kcal
    )
    ax.figure.colorbar(im, ax=ax, label="F (kcal/mol)")
    _draw_basins(ax, basins)
    ax.set(xlim=(-180, 180), ylim=(-180, 180), xlabel="phi (deg)", ylabel="psi (deg)")
    ax.set_title(title)


def _run_replicates(
    cfg: Mapping[str, Any],
    run_dir: Path,
    plumed: bool,
    n_rep: int,
) -> tuple[list[Path], list[Any]]:
    """Run ``n_rep`` independent MD runs into ``run_dir/rep_<r>`` (seed = md.seed + r)."""
    top, system, pos, box = build_solvated(cfg["system_prep"])
    quads = _quads(top)
    dirs, results = [], []
    for r in range(n_rep):
        d = run_dir / f"rep_{r}"
        md = _md_cfg(cfg, seed=int(cfg["md"]["seed"]) + r)
        script = None
        if plumed:
            p = cfg["metad"]
            script = cv.metad_script(quads, p, float(md["temperature_K"]))
            d.mkdir(parents=True, exist_ok=True)
            (d / "plumed.dat").write_text(script)
        res = run_md(top, copy.deepcopy(system), pos, md, d, plumed_script=script, box_vectors=box)
        dirs.append(d)
        results.append(res)
    return dirs, results


def run_md(*args: Any, **kwargs: Any):
    """Thin indirection to :func:`mdbench.engines.md.run_md` (patchable in tests)."""
    from mdbench.engines.md import run_md as _run_md

    return _run_md(*args, **kwargs)


def _pool_replicates(
    per_rep_pops: np.ndarray, per_rep_sem: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over replicates; error = max(replicate SEM, propagated block SEM)."""
    r = len(per_rep_pops)
    mean = per_rep_pops.mean(axis=0)
    rep_sem = per_rep_pops.std(axis=0, ddof=1) / np.sqrt(r) if r > 1 else np.zeros_like(mean)
    block_sem = np.sqrt(np.sum(per_rep_sem**2, axis=0)) / r
    return mean, np.maximum(rep_sem, block_sem)


@register(branch="c", system=SYSTEM, level="L2", method="md_histogram")
def run_l2(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """L2: basin populations from histograms of unbiased explicit-solvent MD replicates."""
    t0 = time.perf_counter()
    basins, names, ref = _basins(cfg)
    kT = _kT(cfg)
    a = cfg["analysis"]
    n_rep = int(cfg["sampling"]["n_replicates"])
    dirs, results = _run_replicates(cfg, run_dir, plumed=False, n_rep=n_rep)
    per_pops, per_sem, phis, psis, curves, runs_ = [], [], [], [], [], []
    for d in dirs:
        c = trajectory_cvs(d)
        phi, psi = np.degrees(c["phi"]), np.degrees(c["psi"])
        lab = fes.assign_basins(phi, psi, basins)
        br = fes.block_populations(lab, len(names), None, a["block_counts"], a["min_blocks"])
        per_pops.append(br.pops)
        per_sem.append(br.sem)
        phis.append(phi)
        psis.append(psi)
        curves.append(br)
        runs_.append(fes.running_estimate(lab, len(names)))
    pops, err = _pool_replicates(np.array(per_pops), np.array(per_sem))
    gpu = results[0].gpu
    _write_json(
        run_dir / "l2_summary.json",
        {
            "names": names,
            "per_replicate_pops": per_pops,
            "per_replicate_block_sem": per_sem,
            "pops": pops,
            "pop_err": err,
            "md": [r.__dict__ for r in results],
        },
    )
    _l2_figure(cfg, phis, psis, curves, runs_, names, basins, kT, results)
    rows = observable_rows(names, ref, pops, err, kT, gpu=gpu)
    wall = time.perf_counter() - t0
    for r in rows:
        r["wall_clock_s"] = wall
    return rows


def _diag_axes(ax_run: Any, ax_blk: Any, runs_: list, curves: list, names: Sequence[str],
               dt_ns: float, label_prefix: str = "rep") -> None:  # fmt: skip
    colors = _series_colors(len(names))
    for r, (ends, p) in enumerate(runs_):
        for b, c in enumerate(colors):
            ax_run.plot(
                ends * dt_ns, p[:, b], color=c, lw=1.2, alpha=0.9,
                label=names[b] if r == 0 else None, ls=["-", "--", ":", "-."][r % 4],
            )  # fmt: skip
    ax_run.set(xlabel="simulated time (ns)", ylabel="cumulative population", yscale="log")
    ax_run.set_title(f"Time-resolved populations (line style = {label_prefix})")
    ax_run.legend(fontsize=7)
    for r, br in enumerate(curves):
        for b, c in enumerate(colors):
            ax_blk.plot(
                br.n_blocks, br.sem_curve[:, b], color=c, lw=1.2, marker="o", ms=3,
                ls=["-", "--", ":", "-."][r % 4], label=names[b] if r == 0 else None,
            )  # fmt: skip
    ax_blk.set(xlabel="number of blocks", ylabel="SEM of population", xscale="log", yscale="log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ax_blk.xaxis.set_major_formatter(ScalarFormatter())
    ax_blk.xaxis.set_minor_formatter(NullFormatter())
    ax_blk.set_title("Block analysis")
    ax_blk.legend(fontsize=7)


def _l2_figure(cfg, phis, psis, curves, runs_, names, basins, kT, results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt_ns = float(cfg["md"]["traj_interval_steps"]) * float(cfg["md"]["timestep_fs"]) * 1e-6
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    total_ns = sum(len(p) for p in phis) * dt_ns
    _fes_axes(
        ax[0], np.concatenate(phis), np.concatenate(psis), kT, cfg["analysis"]["fes_bins"],
        basins, title=f"Unbiased MD FES ({len(phis)} x {total_ns / len(phis):.1f} ns)",
    )  # fmt: skip
    _diag_axes(ax[1], ax[2], runs_, curves, names, dt_ns)
    fig.suptitle("ala2 L2: histogram over unbiased MD (ff14SB/TIP3P, 300 K)")
    fig.savefig(_reports_dir(cfg) / "c_ala2_L2_md_histogram.png", dpi=130)
    plt.close(fig)


# =========================================================================== L3 metad


@register(branch="c", system=SYSTEM, level="L3", method="wtmetad_reweight")
def run_l3_metad(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """L3a: well-tempered metadynamics on (phi, psi), c(t)-reweighted basin populations."""
    t0 = time.perf_counter()
    basins, names, ref = _basins(cfg)
    kT = _kT(cfg)
    a = cfg["analysis"]
    p = cfg["metad"]
    n_rep = int(cfg["sampling"]["n_replicates"])
    dirs, results = _run_replicates(cfg, run_dir, plumed=True, n_rep=n_rep)
    per_pops, per_sem, phis, psis, ws, curves, runs_, ess = [], [], [], [], [], [], [], []
    stride_ps = float(p["colvar_stride_steps"]) * float(cfg["md"]["timestep_fs"]) * 1e-3
    for d in dirs:
        col = cv.read_colvar(d / "COLVAR")
        keep = col["time"] >= float(p["discard_ps"])
        phi = np.degrees(col["phi"][keep])
        psi = np.degrees(col["psi"][keep])
        w = fes.metad_weights(col["metad.rbias"][keep], kT)
        lab = fes.assign_basins(phi, psi, basins)
        br = fes.block_populations(lab, len(names), w, a["block_counts"], a["min_blocks"])
        per_pops.append(br.pops)
        per_sem.append(br.sem)
        phis.append(phi)
        psis.append(psi)
        ws.append(w)
        curves.append(br)
        runs_.append(fes.running_estimate(lab, len(names), w))
        ess.append(fes.effective_sample_size(w))
    pops, err = _pool_replicates(np.array(per_pops), np.array(per_sem))
    _write_json(
        run_dir / "metad_summary.json",
        {
            "names": names,
            "per_replicate_pops": per_pops,
            "per_replicate_block_sem": per_sem,
            "effective_sample_size": ess,
            "pops": pops,
            "pop_err": err,
            "plumed_script": (dirs[0] / "plumed.dat").read_text(),
            "md": [r.__dict__ for r in results],
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    allw = np.concatenate([w / len(ws) for w in ws])
    _fes_axes(
        ax[0], np.concatenate(phis), np.concatenate(psis), kT, a["fes_bins"], basins,
        weights=allw, title=f"WT-metaD reweighted FES ({n_rep} walker(s), c(t) reweighting)",
    )  # fmt: skip
    _diag_axes(ax[1], ax[2], runs_, curves, names, stride_ps * 1e-3, label_prefix="replica")
    ax[1].set_xlabel(f"time after {float(p['discard_ps']) / 1000:.1f} ns discard (ns)")
    fig.suptitle("ala2 L3: well-tempered metadynamics on phi/psi (PLUMED 2.9)")
    fig.savefig(_reports_dir(cfg) / "c_ala2_L3_wtmetad_reweight.png", dpi=130)
    plt.close(fig)
    rows = observable_rows(names, ref, pops, err, kT, gpu=results[0].gpu)
    wall = time.perf_counter() - t0
    for r in rows:
        r["wall_clock_s"] = wall
    return rows


# =========================================================================== L3 umbrella


def umbrella_centers_deg(n_windows: int) -> np.ndarray:
    """Evenly spaced window centres over the periodic [-180, 180) range."""
    return -180.0 + (np.arange(n_windows) + 0.5) * 360.0 / n_windows


@register(branch="c", system=SYSTEM, level="L3", method="umbrella_mbar_phi")
def run_l3_umbrella(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """L3b: umbrella sampling along phi (OpenMM CustomTorsionForce), MBAR reweighting.

    Basin populations in (phi, psi) come from MBAR with psi sampled freely in every window.
    The reported uncertainty combines MBAR's analytic error (decorrelated samples) with
    half the first-half/second-half discrepancy in quadrature.
    """
    import openmm

    t0 = time.perf_counter()
    basins, names, ref = _basins(cfg)
    kT = _kT(cfg)
    u = cfg["umbrella"]
    top, system, pos, box = build_solvated(cfg["system_prep"])
    quads = _quads(top)
    centers = umbrella_centers_deg(int(u["n_windows"]))
    k = float(u["k_kj_rad2"])
    results, phis, psis, labs = [], [], [], []
    for w, c in enumerate(centers):
        s = copy.deepcopy(system)
        f = openmm.CustomTorsionForce(
            "0.5*k_us*dd^2; dd=atan2(sin(theta-theta0), cos(theta-theta0))"
        )
        f.addGlobalParameter("k_us", k)
        f.addPerTorsionParameter("theta0")
        f.addTorsion(*quads["phi"], [float(np.radians(c))])
        s.addForce(f)
        d = run_dir / f"win_{w:03d}"
        md = _md_cfg(cfg, seed=int(cfg["md"]["seed"]) + w)
        results.append(run_md(top, s, pos, md, d, box_vectors=box))
        cvs = trajectory_cvs(d)
        phis.append(cvs["phi"])
        psis.append(cvs["psi"])
        labs.append(fes.assign_basins(np.degrees(cvs["phi"]), np.degrees(cvs["psi"]), basins))
    cr = np.radians(centers)
    nb = len(names)
    bins = int(u["profile_bins"])
    full = fes.umbrella_mbar(phis, labs, cr, k, kT, nb, ref, bins)
    halves = []
    for h in (0, 1):
        sl = [slice(0, len(x) // 2) if h == 0 else slice(len(x) // 2, None) for x in phis]
        halves.append(
            fes.umbrella_mbar(
                [x[s] for x, s in zip(phis, sl, strict=True)],
                [x[s] for x, s in zip(labs, sl, strict=True)],
                cr,
                k,
                kT,
                nb,
                ref,
                bins,
            )
        )
    half_pop = 0.5 * np.abs(halves[0].pops - halves[1].pops)
    half_dg = 0.5 * np.abs(halves[0].dg_kj - halves[1].dg_kj)
    pop_err = np.sqrt(full.pop_err**2 + half_pop**2)
    dg_err = np.sqrt(full.dg_err_kj**2 + half_dg**2)
    _write_json(
        run_dir / "umbrella_summary.json",
        {
            "names": names,
            "centers_deg": centers,
            "pops": full.pops,
            "pop_err_mbar": full.pop_err,
            "pop_err": pop_err,
            "dg_kj": full.dg_kj,
            "dg_err_kj_mbar": full.dg_err_kj,
            "dg_err_kj": dg_err,
            "halves_pops": [hh.pops for hh in halves],
            "statistical_inefficiency": full.g,
            "n_used": full.n_used,
            "min_adjacent_overlap": full.overlap_min,
            "md_wall_clock_s": [r.wall_clock_s for r in results],
        },
    )
    _umbrella_figure(cfg, phis, centers, full, halves, kT, psis, basins)
    rows = observable_rows(
        names, ref, full.pops, pop_err, kT, full.dg_kj, dg_err, gpu=results[0].gpu
    )
    wall = time.perf_counter() - t0
    for r in rows:
        r["wall_clock_s"] = wall
    return rows


def _umbrella_figure(cfg, phis, centers, full, halves, kT, psis, basins) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for x in phis:
        ax[0].hist(np.degrees(x), bins=np.linspace(-180, 180, 181), histtype="step", lw=0.8,
                   color="#2a78d6", alpha=0.7)  # fmt: skip
    ax[0].set(xlabel="phi (deg)", ylabel="samples per window", xlim=(-180, 180))
    ax[0].set_title(f"Window histograms (min adjacent MBAR overlap {full.overlap_min:.3f})")
    ax[1].plot(full.profile_x_deg, fes.as_kcal(full.profile_kj), color="#0b0b0b", lw=2,
               label="all data")  # fmt: skip
    for h, (hh, c) in enumerate(zip(halves, ["#2a78d6", "#eb6834"], strict=True)):
        ax[1].plot(hh.profile_x_deg, fes.as_kcal(hh.profile_kj), color=c, lw=1.2, ls="--",
                   label=["first half", "second half"][h])  # fmt: skip
    ax[1].set(xlabel="phi (deg)", ylabel="F(phi) (kcal/mol)", xlim=(-180, 180))
    ax[1].set_title("MBAR F(phi); convergence = half vs half")
    ax[1].legend(fontsize=7)
    # MBAR-reweighted 2D FES is approximated here by the per-window histogram stitched with
    # each window's unbiasing weight exp(+bias/kT) normalised by the MBAR window free energy;
    # we show instead the raw psi coverage per phi so sampling gaps are visible.
    allphi = np.degrees(np.concatenate(phis))
    allpsi = np.degrees(np.concatenate(psis))
    h2, xe, ye = np.histogram2d(allphi, allpsi, bins=60, range=[[-180, 180], [-180, 180]])
    ax[2].pcolormesh(xe, ye, np.ma.masked_equal(h2, 0).T, cmap="Blues", norm="log")
    _draw_basins(ax[2], basins)
    ax[2].set(xlabel="phi (deg)", ylabel="psi (deg)", xlim=(-180, 180), ylim=(-180, 180))
    ax[2].set_title("(phi, psi) coverage, all windows (biased counts)")
    fig.suptitle("ala2 L3: umbrella sampling along phi + MBAR (pymbar 4)")
    fig.savefig(_reports_dir(cfg) / "c_ala2_L3_umbrella_mbar_phi.png", dpi=130)
    plt.close(fig)


# =========================================================================== L3 MSM


@register(branch="c", system=SYSTEM, level="L3", method="msm_deeptime")
def run_l3_msm(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """L3c: MSM (deeptime) over the L2 trajectories: ITS, CK test, Bayesian populations.

    ``wall_clock_s`` = MSM analysis time + the MD wall clock of the source trajectories
    (the method cannot exist without them).
    """
    from mdbench.config import load_config
    from mdbench.config import run_dir as cfg_run_dir
    from mdbench.engines import msm

    t0 = time.perf_counter()
    basins, names, ref = _basins(cfg)
    kT = _kT(cfg)
    m = cfg["msm"]
    src_cfg = load_config(_repo_path(m["source_config"]))
    src_dir = cfg_run_dir(src_cfg)
    reps = sorted(src_dir.glob("rep_*"))
    if not reps:
        raise FileNotFoundError(f"no L2 trajectories in {src_dir}; run {m['source_config']} first")
    md_wall = 0.0
    gpu = "none"
    angles = []
    for d in reps:
        res = json.loads((d / "md_result.json").read_text())
        md_wall += float(res["wall_clock_s"])
        gpu = res["gpu"]
        c = trajectory_cvs(d)
        angles.append((c["phi"], c["psi"]))
    dt_ps = float(src_cfg["md"]["traj_interval_steps"]) * float(src_cfg["md"]["timestep_fs"]) * 1e-3
    feats = [msm.periodic_features(a) for a in angles]
    dtrajs, centres = msm.discretize(feats, int(m["n_clusters"]), int(m["cluster_seed"]))
    cang = msm.centres_to_angles_deg(centres)
    macro = fes.assign_basins(cang[:, 0], cang[:, 1], basins)
    lag = max(1, round(float(m["lag_ps"]) / dt_ps))
    lags = sorted({max(1, round(float(x) / dt_ps)) for x in m["its_lags_ps"]})
    its = msm.implied_timescales(dtrajs, lags, int(m["n_its"])) * dt_ps
    ck = msm.ck_test(dtrajs, lag, macro, len(names), m["ck_multiples"])
    model = msm.estimate_msm(dtrajs, lag)
    pops_mle = msm.macro_populations(model, macro, len(names))
    samples = msm.bayesian_macro_populations(dtrajs, lag, macro, len(names), int(m["n_samples"]))
    pop_err = samples.std(axis=0, ddof=1)
    with np.errstate(divide="ignore"):
        dg_s = -kT * (np.log(samples) - np.log(samples[:, [ref]]))
    dg = fes.delta_g_kj(pops_mle, ref, kT)
    dg_err = np.where(np.all(np.isfinite(dg_s), axis=0), dg_s.std(axis=0, ddof=1), np.nan)
    dg_err[ref] = 0.0
    n_conn = int(model.n_states)
    _write_json(
        run_dir / "msm_summary.json",
        {
            "names": names,
            "source_run_dir": str(src_dir),
            "dt_ps": dt_ps,
            "lag_frames": lag,
            "its_lags_frames": lags,
            "implied_timescales_ps": its,
            "ck_max_abs_dev": ck.max_abs_dev,
            "n_states_connected": n_conn,
            "n_clusters": int(m["n_clusters"]),
            "pops_mle": pops_mle,
            "pops_bayes_mean": samples.mean(axis=0),
            "pop_err": pop_err,
            "md_wall_clock_s": md_wall,
        },
    )
    _msm_figure(cfg, angles, dtrajs, model, its, lags, lag, dt_ps, ck, names, basins, kT)
    rows = observable_rows(names, ref, pops_mle, pop_err, kT, dg, dg_err, gpu=gpu)
    wall = time.perf_counter() - t0 + md_wall
    for r in rows:
        r["wall_clock_s"] = wall
    return rows


def _msm_figure(cfg, angles, dtrajs, model, its, lags, lag, dt_ps, ck, names, basins, kT) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    # MSM-reweighted FES: each frame weighted by pi_s / N_s of its microstate.
    pi = np.zeros(int(max(d.max() for d in dtrajs)) + 1)
    pi[np.asarray(model.count_model.state_symbols)] = model.stationary_distribution
    dall = np.concatenate(dtrajs)
    n_s = np.bincount(dall, minlength=len(pi)).astype(float)
    w = np.where(n_s[dall] > 0, pi[dall] / np.maximum(n_s[dall], 1), 0.0)
    phi = np.degrees(np.concatenate([a[0] for a in angles]))
    psi = np.degrees(np.concatenate([a[1] for a in angles]))
    ax0 = fig.add_subplot(gs[0, 0])
    _fes_axes(ax0, phi, psi, kT, cfg["analysis"]["fes_bins"], basins, weights=w,
              title=f"MSM-reweighted FES (lag {lag * dt_ps:g} ps)")  # fmt: skip
    ax1 = fig.add_subplot(gs[0, 1:])
    lag_ps = np.array(lags) * dt_ps
    for i, c in enumerate(["#2a78d6", "#eb6834", "#1baf7a"][: its.shape[1]]):
        ax1.plot(lag_ps, its[:, i], marker="o", ms=4, lw=1.5, color=c, label=f"t{i + 2}")
    ax1.fill_between(lag_ps, lag_ps, 1e-3, color="#c3c2b7", alpha=0.5, label="t < lag")
    ax1.axvline(lag * dt_ps, color="#52514e", ls=":", lw=1, label="chosen lag")
    ax1.set(xscale="log", yscale="log", xlabel="lag (ps)", ylabel="implied timescale (ps)",
            ylim=(max(dt_ps, 0.5), None))  # fmt: skip
    ax1.set_title("Implied timescales")
    ax1.legend(fontsize=7)
    nm = len(names)
    colors = _series_colors(nm)
    for i in range(nm):
        ax = fig.add_subplot(gs[1, i])
        t = np.array(ck.lag_multiples) * lag * dt_ps
        for j in range(nm):
            ax.plot(t, ck.estimated[:, i, j], color=colors[j], lw=1.5, marker="o", ms=4,
                    label=f"-> {names[j]} est.")  # fmt: skip
            ax.plot(t, ck.predicted[:, i, j], color=colors[j], lw=1.2, ls="--",
                    label=f"-> {names[j]} pred.")  # fmt: skip
        ax.set(xlabel="time (ps)", ylabel="P(stay/transit)", ylim=(-0.02, 1.02))
        ax.set_title(f"CK test from {names[i]}")
        if i == 0:
            ax.legend(fontsize=6)
    fig.suptitle(f"ala2 L3: MSM (deeptime) on L2 trajectories; CK max |dev| = {ck.max_abs_dev:.3f}")
    fig.savefig(_reports_dir(cfg) / "c_ala2_L3_msm_deeptime.png", dpi=130)
    plt.close(fig)


__all__ = [
    "build_solvated",
    "find_minima",
    "observable_rows",
    "relaxed_hessian",
    "run_l0",
    "run_l1",
    "run_l2",
    "run_l3_metad",
    "run_l3_msm",
    "run_l3_umbrella",
    "trajectory_cvs",
    "umbrella_centers_deg",
]
