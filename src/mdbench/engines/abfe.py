"""Absolute binding free energy by alchemical decoupling (branch A, L3). Runs in ``mdbench-fe``.

Thermodynamic cycle, per guest, two legs sampled with Hamiltonian replica exchange
(openmmtools ``ReplicaExchangeSampler``) and analysed with MBAR:

- **complex**: host + guest in water. State 0 is fully coupled and unrestrained; the
  first states switch on a harmonic restraint between the host and guest heavy-atom
  centres of mass (``lambda_restraints`` 0 -> 1), then guest electrostatics are annihilated
  and sterics decoupled with the restraint on. ``dF_complex = F(decoupled, restrained) -
  F(coupled, free)``.
- **solvent**: guest in water, electrostatics then sterics. ``dF_solvent = F(decoupled) -
  F(coupled)``.

Binding free energy at the 1 M standard state::

    dG_bind = dF_solvent - dF_complex + kT ln(V0 / V_r),   V_r = (2 pi kT / K)^(3/2)

where ``V_r`` is the configurational volume of the decoupled guest under the restraint
(spring constant ``K`` on the centre-of-mass distance) and ``V0 = 1660.539 A^3``. The
restraint does not bias the coupled end state because it is ramped on inside the cycle.

Charged guests: with PME, decoupling a net charge introduces box-size-dependent
artefacts. Each leg therefore decouples the guest *together with* ``|q|`` counterions
(Cl- for a cationic guest) taken from the bulk, farthest from the guest, so every
alchemical perturbation is net-neutral (the co-alchemical ion approach). The ions'
own bulk decoupling appears in both legs and cancels in ``dF_solvent - dF_complex``.

Convergence diagnostics written per leg: MBAR overlap matrix (smallest neighbour
overlap), equilibration iterations and statistical inefficiency (openmmtools analyzer),
replica round trips, and the free energy from the first and second halves of the
production samples. The reduced-potential matrix is saved to ``u_kln.npz`` and the
replica-exchange NetCDF files are deleted afterwards (docs/HARDWARE.md retention policy).
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mdbench import budget

# pymbar's JAX backend has CUDA in mdbench-fe and preallocates 75% of the card the first
# time MBAR runs, then holds it: after one leg's analysis only ~1.4 GB of the 6 GB is left
# for the next leg's contexts, which a 35k-atom T4L complex overruns. MBAR on these
# u_kln sizes takes seconds on the CPU. Must be set before openmmtools imports pymbar.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

log = logging.getLogger(__name__)

KB_KJ_MOL_K = 0.0083144626
KCAL_PER_KJ = 1.0 / 4.184
V0_NM3 = 1.660539  # 1 M standard state volume per molecule, nm^3


def restraint_volume_nm3(spring_kj_mol_nm2: float, temperature_k: float) -> float:
    """Configurational volume of a 3D isotropic harmonic restraint, (2 pi kT / K)^(3/2)."""
    kt = KB_KJ_MOL_K * temperature_k
    return (2.0 * math.pi * kt / spring_kj_mol_nm2) ** 1.5


def standard_state_term_kt(spring_kj_mol_nm2: float, temperature_k: float) -> float:
    """``ln(V0 / V_r)`` (in kT): the cost of confining the decoupled guest to the restraint."""
    return math.log(V0_NM3 / restraint_volume_nm3(spring_kj_mol_nm2, temperature_k))


def spring_constant(sigma_nm: float, temperature_k: float) -> float:
    """K (kJ/mol/nm^2) giving a per-axis positional sd of ``sigma_nm``: K = kT / sigma^2."""
    return KB_KJ_MOL_K * temperature_k / sigma_nm**2


def pick_counterions(
    positions_nm: np.ndarray, guest: Sequence[int], ions: Sequence[int], n: int
) -> list[int]:
    """The ``n`` ions farthest from the guest centre (ignoring periodicity: a ranking only)."""
    if n == 0:
        return []
    if len(ions) < n:
        raise ValueError(f"need {n} counterions, box has {len(ions)}")
    c = positions_nm[list(guest)].mean(axis=0)
    d = np.linalg.norm(positions_nm[list(ions)] - c, axis=1)
    return [int(ions[i]) for i in np.argsort(-d)[:n]]


def schedule(cfg: Mapping[str, Any], leg: str) -> dict[str, list[float]]:
    """Per-state lambda lists for one leg, validated to equal length and monotone order."""
    s = {k: [float(x) for x in v] for k, v in cfg["schedule"][leg].items()}
    n = {len(v) for v in s.values()}
    if len(n) != 1:
        raise ValueError(f"{leg}: lambda lists differ in length {n}")
    for k, v in s.items():
        if k == "lambda_restraints":
            if v[0] != 0.0 or v[-1] != 1.0 or np.any(np.diff(v) < 0):
                raise ValueError(f"{leg}: lambda_restraints must rise 0 -> 1")
        elif v[0] != 1.0 or v[-1] != 0.0 or np.any(np.diff(v) > 0):
            raise ValueError(f"{leg}: {k} must fall 1 -> 0")
    return s


def _read_inputs(system_xml: Path, pdb: Path) -> tuple[Any, np.ndarray, Any]:
    import openmm
    from openmm import app, unit

    opener = gzip.open if str(system_xml).endswith(".gz") else open
    with opener(system_xml, "rt") as fh:
        system = openmm.XmlSerializer.deserialize(fh.read())
    with gzip.open(pdb, "rt") if str(pdb).endswith(".gz") else open(pdb) as fh:
        p = app.PDBFile(fh)
    pos = np.array(p.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return system, pos, p.topology.getPeriodicBoxVectors()


def _heavy(system: Any, atoms: Sequence[int]) -> list[int]:
    from openmm import unit

    return [int(a) for a in atoms
            if system.getParticleMass(int(a)).value_in_unit(unit.dalton) > 1.5]  # fmt: skip


def _restraint_state_class() -> Any:
    """The ``RestraintState`` class (built on first use; openmmtools is imported lazily).

    It is published as the module attribute ``mdbench.engines.abfe.RestraintState`` (see
    :func:`__getattr__`). openmmtools pickles thermodynamic states into the ``.nc`` file by
    class path, so the class must be importable by name for a paused run to be reloaded.
    """
    if "RestraintState" not in globals():
        from openmmtools.states import GlobalParameterState

        class RestraintState(GlobalParameterState):
            """Controls ``lambda_restraints`` of the host-guest centre-of-mass restraint."""

            lambda_restraints = GlobalParameterState.GlobalParameter(
                "lambda_restraints", standard_value=1.0
            )

        RestraintState.__qualname__ = "RestraintState"
        globals()["RestraintState"] = RestraintState
    return globals()["RestraintState"]


def __getattr__(name: str) -> Any:
    if name == "RestraintState":
        return _restraint_state_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_states(
    system: Any,
    leg: str,
    alchemical_atoms: Sequence[int],
    host_heavy: Sequence[int],
    guest_heavy: Sequence[int],
    cfg: Mapping[str, Any],
    temperature_k: float,
    pressure_bar: float | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Compound thermodynamic states for one leg; returns (states, info)."""
    from openmm import unit
    from openmmtools import alchemy, forces, states

    region = alchemy.AlchemicalRegion(
        alchemical_atoms=list(alchemical_atoms),
        annihilate_electrostatics=True,
        annihilate_sterics=False,
        softcore_alpha=float(cfg.get("softcore_alpha", 0.5)),
    )
    factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True)
    alch_system = factory.create_alchemical_system(system, region)
    info: dict[str, Any] = {"n_alchemical_atoms": len(alchemical_atoms)}
    composable = []
    if leg == "complex":
        k = spring_constant(float(cfg["restraint_sigma_nm"]), temperature_k)
        restraint = forces.HarmonicRestraintForce(
            spring_constant=k * unit.kilojoule_per_mole / unit.nanometer**2,
            restrained_atom_indices1=list(host_heavy),
            restrained_atom_indices2=list(guest_heavy),
            controlling_parameter_name="lambda_restraints",
        )
        alch_system.addForce(restraint)
        info["spring_constant_kj_mol_nm2"] = k
    pressure = None if pressure_bar is None else float(pressure_bar) * unit.bar
    thermo = states.ThermodynamicState(alch_system, temperature_k * unit.kelvin, pressure)
    alch_state = alchemy.AlchemicalState.from_system(alch_system)
    composable.append(alch_state)
    if leg == "complex":
        composable.append(_restraint_state_class().from_system(alch_system))
    lam = schedule(cfg, leg)
    out = []
    n = len(next(iter(lam.values())))
    for i in range(n):
        cs = states.CompoundThermodynamicState(thermo, composable)
        for key, vals in lam.items():
            setattr(cs, key, vals[i])
        out.append(cs)
    info["n_states"] = n
    info["schedule"] = lam
    return out, info


def pre_equilibrate(
    system: Any, pos_nm: np.ndarray, box: Any, md: Mapping[str, Any], n_steps: int, seed: int
) -> tuple[np.ndarray, Any]:
    """Minimise and run plain NPT MD on the fully coupled system; returns (positions, box).

    Packmol boxes start at 0.9 g/mL with small voids; relaxing density and solvation
    before replica exchange keeps the alchemical states from starting in strained
    configurations.
    """
    import copy

    import openmm
    from openmm import unit

    sys2 = copy.deepcopy(system)
    temperature = float(md["temperature_K"]) * unit.kelvin
    if md.get("pressure_bar") is not None:
        sys2.addForce(openmm.MonteCarloBarostat(float(md["pressure_bar"]) * unit.bar, temperature))
    integ = openmm.LangevinMiddleIntegrator(
        temperature, float(md["friction_per_ps"]) / unit.picosecond,
        float(md["timestep_fs"]) * unit.femtosecond,
    )  # fmt: skip
    integ.setRandomNumberSeed(int(seed))
    plat = openmm.Platform.getPlatformByName(str(md.get("platform", "CUDA")))
    props = (
        {"Precision": str(md.get("precision", "mixed"))}
        if plat.getName() in ("CUDA", "OpenCL")
        else {}
    )
    ctx = openmm.Context(sys2, integ, plat, props)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(pos_nm * unit.nanometer)
    openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, 0)
    ctx.setVelocitiesToTemperature(temperature, int(seed))
    integ.step(int(n_steps))
    st = ctx.getState(getPositions=True, enforcePeriodicBox=True)
    out = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box_out = st.getPeriodicBoxVectors()
    del ctx, integ
    return out, box_out


def _context_cache(md: Mapping[str, Any], capacity: int | None = None) -> Any:
    """GPU context cache. ``capacity`` bounds it: unbounded, ``sampler.minimize`` keeps one
    context per state alive (~100 MB each on CUDA), which exhausts a 6 GB card."""
    import openmm
    from openmmtools import cache

    plat = openmm.Platform.getPlatformByName(str(md.get("platform", "CUDA")))
    props = (
        {"Precision": str(md.get("precision", "mixed"))}
        if plat.getName() in ("CUDA", "OpenCL")
        else {}
    )
    return cache.ContextCache(platform=plat, platform_properties=props, capacity=capacity)


def _round_trips(state_index: np.ndarray) -> int:
    """Replica round trips 0 -> last -> 0, summed over replicas.

    ``state_index`` is (iteration, replica) -> thermodynamic state.
    """
    n_states = int(state_index.max()) + 1
    trips = 0
    for r in range(state_index.shape[1]):
        target, seen_end = n_states - 1, False
        for s in state_index[:, r]:
            if not seen_end and s == target:
                seen_end = True
            elif seen_end and s == 0:
                trips += 1
                seen_end = False
    return trips


def mbar_from_u_kln(u_kln: np.ndarray, n_k: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(dF first->last, its sd, overlap matrix) in kT from per-state samples."""
    from pymbar import MBAR

    k = u_kln.shape[0]
    u_kn = np.zeros((k, int(n_k.sum())))
    col = 0
    for i in range(k):
        u_kn[:, col : col + n_k[i]] = u_kln[i, :, : n_k[i]]
        col += n_k[i]
    mbar = MBAR(u_kn, n_k, solver_protocol="robust")
    r = mbar.compute_free_energy_differences()
    ov = mbar.compute_overlap()["matrix"]
    return float(r["Delta_f"][0, -1]), float(r["dDelta_f"][0, -1]), ov


def per_state_samples(
    energy: np.ndarray, state_of_replica: np.ndarray, t0: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """u_kln (state sampled, state evaluated, sample) and N_k from replica-ordered energies.

    ``energy`` is (iteration, replica, state) reduced potentials; ``state_of_replica`` is
    (iteration, replica). Iterations ``t0::stride`` are used.
    """
    it = np.arange(t0, energy.shape[0], max(1, int(stride)))
    k = energy.shape[2]
    buckets: list[list[np.ndarray]] = [[] for _ in range(k)]
    for t in it:
        for r in range(energy.shape[1]):
            buckets[int(state_of_replica[t, r])].append(energy[t, r, :])
    n_k = np.array([len(b) for b in buckets])
    u = np.zeros((k, k, int(n_k.max())))
    for i, b in enumerate(buckets):
        if b:
            u[i, :, : len(b)] = np.array(b).T
    return u, n_k


def analyse(nc: Path) -> dict[str, Any]:
    """MBAR analysis of one leg's replica-exchange storage."""
    from openmmtools import multistate

    reporter = multistate.MultiStateReporter(str(nc), open_mode="r")
    try:
        analyzer = multistate.MultiStateSamplerAnalyzer(reporter, unbias_restraint=False)
        df, ddf = analyzer.get_free_energy()
        n_equil = int(analyzer.n_equilibration_iterations)
        g = float(analyzer.statistical_inefficiency)
        energy_rs, _, _, replica_state = analyzer.read_energies()
        # (replica, state, iteration) -> (iteration, replica, state)
        energy = np.transpose(np.asarray(energy_rs), (2, 0, 1))
        state_of_replica = np.asarray(replica_state).T  # (iteration, replica)
        n_iter = energy.shape[0]
    finally:
        reporter.close()
    stride = max(1, round(g))
    u, n_k = per_state_samples(energy, state_of_replica, n_equil, stride)
    f, sf, overlap = mbar_from_u_kln(u, n_k)
    mid = n_equil + (n_iter - n_equil) // 2
    halves = []
    for a, b in ((n_equil, mid), (mid, n_iter)):
        uh, nh = per_state_samples(energy[:b], state_of_replica[:b], a, stride)
        halves.append(mbar_from_u_kln(uh, nh)[0])
    # Time-forward convergence: dF over growing fractions of the production samples.
    forward = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        end = n_equil + max(1, int(frac * (n_iter - n_equil)))
        uf, nf = per_state_samples(energy[:end], state_of_replica[:end], n_equil, stride)
        forward.append([frac, mbar_from_u_kln(uf, nf)[0]])
    neighbour = [float(overlap[i, i + 1]) for i in range(len(overlap) - 1)]
    return {
        "dF_kT": float(df[0, -1]),
        "ddF_kT": float(ddf[0, -1]),
        "dF_kT_own_mbar": f,
        "ddF_kT_own_mbar": sf,
        "n_iterations": int(n_iter),
        "n_equilibration_iterations": n_equil,
        "statistical_inefficiency": g,
        "n_samples_per_state": n_k.tolist(),
        "neighbour_overlap": neighbour,
        "min_neighbour_overlap": float(min(neighbour)),
        "dF_kT_first_half": halves[0],
        "dF_kT_second_half": halves[1],
        "dF_kT_forward": forward,
        "round_trips": _round_trips(state_of_replica[n_equil:]),
        "u_kln": u,
        "n_k": n_k,
    }


def _read_prior_seconds(path: Path) -> float:
    try:
        return float(json.loads(path.read_text()).get("seconds", 0.0))
    except (OSError, ValueError):
        return 0.0


def _run_iterations(sampler: Any, n_iter: int, checkpoint_interval: int, step: int = 5) -> bool:
    """Run the sampler to ``n_iter`` iterations; returns True if a budget stop paused it.

    Resuming an openmmtools multistate run restarts from the last *checkpoint* iteration
    (every ``checkpoint_interval`` iterations), so iterations past it are redone. A stop is
    therefore taken at a checkpoint boundary when that is reachable within the grace period
    (``mdbench run --grace-minutes``) and otherwise immediately, losing the partial block.
    """
    ci = max(1, int(checkpoint_interval))
    t_start = time.perf_counter()
    it_start = int(sampler.iteration)
    while sampler.iteration < n_iter:
        if budget.stop_requested():
            it = int(sampler.iteration)
            to_boundary = (-it) % ci
            done = it - it_start
            per_it = (time.perf_counter() - t_start) / done if done > 0 else float("inf")
            if to_boundary == 0 or per_it * to_boundary > budget.grace_left():
                return True
        k = min(step, n_iter - int(sampler.iteration))
        if budget.stop_requested():  # heading for a boundary: never overshoot it
            k = min(k, (-int(sampler.iteration)) % ci or k)
        sampler.run(n_iterations=k)
    return False


def run_leg(
    leg: str,
    system_xml: Path,
    pdb: Path,
    indices: Mapping[str, Any],
    guest_charge: int,
    cfg: Mapping[str, Any],
    md: Mapping[str, Any],
    out: Path,
) -> dict[str, Any]:
    """Sample and analyse one leg; returns a JSON-able summary (u_kln saved beside it)."""
    from openmm import unit
    from openmmtools import mcmc, multistate, states

    from mdbench.engines.md import _nvidia_smi_name

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    temperature = float(md["temperature_K"])
    system, pos, box = _read_inputs(Path(system_xml), Path(pdb))
    guest = [int(i) for i in indices["guest"]]
    ion_key = "cl" if guest_charge > 0 else "na"
    counter = []
    if cfg.get("coalchemical_ions", True) and guest_charge != 0:
        counter = pick_counterions(pos, guest, indices[ion_key], abs(guest_charge))
    alchemical = guest + counter
    host_heavy = list(indices.get("restraint_host") or _heavy(system, indices.get("host", [])))
    guest_heavy = _heavy(system, guest)
    therm_states, info = build_states(
        system, leg, alchemical, host_heavy, guest_heavy, cfg, temperature,
        md.get("pressure_bar"),
    )  # fmt: skip
    info["counterions"] = counter

    nc = out / f"{leg}.nc"
    n_iter = int(cfg["n_iterations"][leg])
    move = mcmc.LangevinDynamicsMove(
        timestep=float(md["timestep_fs"]) * unit.femtosecond,
        collision_rate=float(md["friction_per_ps"]) / unit.picosecond,
        n_steps=int(cfg["steps_per_iteration"]),
        reassign_velocities=bool(cfg.get("reassign_velocities", True)),
    )
    ready = out / "sampling_ready.json"  # written once minimise + equilibrate are done
    time_file = out / "sampling_time.json"  # cumulative sampling seconds over earlier sessions
    if nc.exists() and not ready.exists():
        nc.unlink()  # stopped before the starting state was prepared: start the leg over
    prior_s = _read_prior_seconds(time_file) if nc.exists() else 0.0
    if nc.exists():
        reporter = multistate.MultiStateReporter(str(nc), open_mode="a")
        sampler = multistate.ReplicaExchangeSampler.from_storage(reporter)
        sampler.energy_context_cache = _context_cache(md, capacity=1)
        sampler.sampler_context_cache = _context_cache(md)
        sampler.number_of_iterations = max(int(sampler.number_of_iterations), n_iter)
    else:
        sampler = multistate.ReplicaExchangeSampler(
            mcmc_moves=move,
            number_of_iterations=n_iter,
            online_analysis_interval=None,
        )
        sampler.energy_context_cache = _context_cache(md, capacity=1)
        sampler.sampler_context_cache = _context_cache(md)
        reporter = multistate.MultiStateReporter(
            str(nc), checkpoint_interval=int(cfg.get("checkpoint_interval", 100))
        )
        pos_eq, box_eq = pre_equilibrate(
            system, pos, box, md, int(cfg["pre_equilibration_steps"]), int(cfg["seed"])
        )
        sampler_state = states.SamplerState(positions=pos_eq * unit.nanometer, box_vectors=box_eq)
        sampler.create(therm_states, sampler_state, reporter)
        sampler.minimize(max_iterations=int(cfg.get("minimize_max_iterations", 1000)))
        sampler.equilibrate(int(cfg.get("equilibration_iterations", 0)))
        ready.write_text("{}")
    paused = _run_iterations(sampler, n_iter, int(cfg.get("checkpoint_interval", 100)))
    if paused:
        stopped_at = int(sampler.iteration)
        reporter.close()
        for c in (sampler.energy_context_cache, sampler.sampler_context_cache):
            c.empty()
        del sampler
        time_file.write_text(json.dumps({"seconds": prior_s + time.perf_counter() - t0}))
        raise budget.RunPaused(f"{leg}: paused at iteration {stopped_at}/{n_iter}")
    reporter.close()
    for c in (sampler.energy_context_cache, sampler.sampler_context_cache):
        c.empty()  # free the GPU contexts before the next leg in this process
    del sampler
    sample_wall = prior_s + time.perf_counter() - t0

    t1 = time.perf_counter()
    res = analyse(nc)
    np.savez_compressed(out / "u_kln.npz", u_kln=res.pop("u_kln"), n_k=res.pop("n_k"))
    if cfg.get("delete_nc_after_analysis", True):
        for f in out.glob("*.nc"):
            f.unlink()
    res.update(
        leg=leg,
        info=info,
        sampling_wall_clock_s=sample_wall,
        analysis_wall_clock_s=time.perf_counter() - t1,
        wall_clock_s=sample_wall + time.perf_counter() - t1,
        simulated_ns_total=n_iter * int(cfg["steps_per_iteration"]) * float(md["timestep_fs"])
        * 1e-6 * info["n_states"],
        gpu=_nvidia_smi_name(0) or "unknown",
    )  # fmt: skip
    (out / "leg_summary.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def combine(legs: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any],
            temperature_k: float) -> dict[str, Any]:  # fmt: skip
    """Binding free energy (kcal/mol) and its uncertainty from the two legs.

    Per-leg sd combines the MBAR sd with half the first/second-half discrepancy (in
    quadrature): MBAR alone ignores slow, unconverged degrees of freedom.
    """
    kt_kcal = KB_KJ_MOL_K * temperature_k * KCAL_PER_KJ
    k = spring_constant(float(cfg["restraint_sigma_nm"]), temperature_k)
    ssc = standard_state_term_kt(k, temperature_k)
    sig2 = 0.0
    parts = {}
    for leg, r in legs.items():
        half = abs(r["dF_kT_first_half"] - r["dF_kT_second_half"]) / 2.0
        s = math.hypot(r["ddF_kT"], half)
        sig2 += s**2
        parts[leg] = {"dF_kT": r["dF_kT"], "sigma_kT": s}
    dg_kt = legs["solvent"]["dF_kT"] - legs["complex"]["dF_kT"] + ssc
    return {
        "dG_kcal": dg_kt * kt_kcal,
        "sigma_kcal": math.sqrt(sig2) * kt_kcal,
        "standard_state_kT": ssc,
        "spring_constant_kj_mol_nm2": k,
        "legs": parts,
        "wall_clock_s": sum(r["wall_clock_s"] for r in legs.values()),
    }


__all__ = [
    "V0_NM3",
    "analyse",
    "build_states",
    "combine",
    "mbar_from_u_kln",
    "per_state_samples",
    "pick_counterions",
    "restraint_volume_nm3",
    "run_leg",
    "schedule",
    "spring_constant",
    "standard_state_term_kt",
]
