"""Generic OpenMM runner shared by every branch.

Input is a prepared system (``openmm.app.Topology``, ``openmm.System``, positions); all
run parameters come from the ``md`` block of a resolved config (see ``defaults.yml``).

Output, in the run directory:

- ``topology.pdb``   solute-only (or the selected subset) reference structure for the XTC
- ``traj.xtc``       sparse XTC of the selected atom subset
- ``md.log``         StateDataReporter CSV
- ``checkpoint.chk`` restart point (resumed automatically when ``md.resume`` is true)
- ``final.pdb``      last frame of the selected subset
- ``md_result.json`` wall clock, ns/day, platform, GPU name

PLUMED: pass ``plumed_script`` (text); it is added as an ``openmmplumed.PlumedForce`` and
the simulation runs with the run directory as the working directory, so relative file
names in the script (COLVAR, HILLS, ...) land next to the trajectory.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import shutil
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import openmm
from openmm import app, unit

log = logging.getLogger(__name__)

# Residue names treated as solvent or ions when selecting the solute.
SOLVENT_RESNAMES: frozenset[str] = frozenset(
    {
        "HOH", "WAT", "TIP3", "TIP4", "TIP5", "SPC", "SOL", "T3P", "T4P", "OPC",
        "NA", "NA+", "Na+", "SOD", "CL", "CL-", "Cl-", "CLA", "K", "K+", "POT",
        "MG", "CA", "CAL", "ZN", "LI", "RB", "CS", "F", "BR", "I",
    }
)  # fmt: skip


@dataclass
class MDResult:
    """Summary of one MD run."""

    run_dir: str
    platform: str
    gpu: str
    n_atoms: int
    n_steps: int
    timestep_fs: float
    wall_clock_s: float
    production_wall_clock_s: float
    ns_per_day: float
    resumed_from_step: int
    trajectory: str | None
    topology: str | None
    log: str
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def solute_indices(topology: app.Topology) -> list[int]:
    """Indices of atoms not in solvent or ion residues."""
    return [a.index for a in topology.atoms() if a.residue.name not in SOLVENT_RESNAMES]


def select_atoms(topology: app.Topology, selection: str | Sequence[int] | None) -> list[int]:
    """Resolve ``md.xtc_selection`` to atom indices.

    ``"solute"`` falls back to all atoms (with a warning) if there is no solute.
    """
    n = topology.getNumAtoms()
    if selection is None or selection == "all":
        return list(range(n))
    if selection == "solute":
        idx = solute_indices(topology)
        if not idx:
            log.warning("no solute atoms found; writing all atoms to the trajectory")
            return list(range(n))
        return idx
    if isinstance(selection, str):
        raise ValueError(f"unknown xtc_selection {selection!r} (use solute, all, or indices)")
    idx = sorted({int(i) for i in selection})
    if idx and (idx[0] < 0 or idx[-1] >= n):
        raise ValueError("xtc_selection index out of range")
    return idx


def _platform_properties(name: str, md: Mapping[str, Any]) -> dict[str, str]:
    props: dict[str, str] = {}
    if name in ("CUDA", "OpenCL"):
        props["Precision"] = str(md.get("precision", "mixed"))
        if md.get("device_index") is not None:
            props["DeviceIndex"] = str(md["device_index"])
    elif name == "CPU" and md.get("cpu_threads"):
        props["Threads"] = str(md["cpu_threads"])
    return props


def make_integrator(md: Mapping[str, Any]) -> openmm.Integrator:
    """Integrator from the ``md`` config block."""
    dt = float(md["timestep_fs"]) * unit.femtoseconds
    kind = str(md.get("integrator", "langevin_middle")).lower()
    if kind == "langevin_middle":
        integ = openmm.LangevinMiddleIntegrator(
            float(md["temperature_K"]) * unit.kelvin,
            float(md["friction_per_ps"]) / unit.picosecond,
            dt,
        )
        integ.setRandomNumberSeed(int(md.get("seed") or 0))
    elif kind == "verlet":
        integ = openmm.VerletIntegrator(dt)
    else:
        raise ValueError(f"unknown integrator {kind!r}")
    integ.setConstraintTolerance(float(md.get("constraint_tolerance", 1e-6)))
    return integ


def _nvidia_smi_name(index: int) -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def device_name(context: openmm.Context) -> str:
    """Human-readable device for the ``gpu`` column: GPU name, or ``cpu``/``reference``."""
    platform = context.getPlatform()
    name = platform.getName()
    if name in ("CUDA", "OpenCL"):
        try:
            dev = platform.getPropertyValue(context, "DeviceName").strip()
        except Exception:
            dev = ""
        if not dev:
            try:
                idx = int(platform.getPropertyValue(context, "DeviceIndex").split(",")[0])
            except Exception:
                idx = 0
            dev = _nvidia_smi_name(idx) or f"{name}-unknown"
        return dev
    return name.lower()


def build_simulation(
    topology: app.Topology,
    system: openmm.System,
    positions: Any,
    md: Mapping[str, Any],
    plumed_script: str | None = None,
    box_vectors: Any = None,
) -> app.Simulation:
    """Create a Simulation with platform selection and optional PLUMED and barostat.

    The ``system`` is modified in place when a barostat or PLUMED force is requested;
    pass a copy if the caller needs the original.
    """
    if md.get("pressure_bar") is not None:
        system.addForce(
            openmm.MonteCarloBarostat(
                float(md["pressure_bar"]) * unit.bar,
                float(md["temperature_K"]) * unit.kelvin,
                int(md.get("barostat_interval", 25)),
            )
        )
    if plumed_script:
        from openmmplumed import PlumedForce

        pf = PlumedForce(plumed_script)
        if hasattr(pf, "setTemperature"):
            pf.setTemperature(float(md["temperature_K"]))
        system.addForce(pf)

    names = [str(md.get("platform", "CUDA"))]
    if md.get("fallback_platform"):
        names.append(str(md["fallback_platform"]))
    last_err: Exception | None = None
    for name in names:
        try:
            platform = openmm.Platform.getPlatformByName(name)
            sim = app.Simulation(
                topology,
                system,
                make_integrator(md),
                platform,
                _platform_properties(name, md),
            )
        except Exception as err:  # OpenMMException for missing/unusable platforms
            log.warning("platform %s unavailable: %s", name, err)
            last_err = err
            continue
        if box_vectors is not None:
            sim.context.setPeriodicBoxVectors(*box_vectors)
        sim.context.setPositions(positions)
        return sim
    raise RuntimeError(f"no usable OpenMM platform among {names}: {last_err}")


def _write_subset_pdb(
    topology: app.Topology, positions: Any, atoms: Sequence[int], path: Path
) -> None:
    keep = set(atoms)
    modeller = app.Modeller(topology, positions)
    if len(keep) < topology.getNumAtoms():
        modeller.delete([a for a in modeller.topology.atoms() if a.index not in keep])
    with path.open("w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)


@contextlib.contextmanager
def _maybe_chdir(path: Path, enabled: bool) -> Iterator[None]:
    if enabled:
        with contextlib.chdir(path):
            yield
    else:
        yield


def run_md(
    topology: app.Topology,
    system: openmm.System,
    positions: Any,
    md: Mapping[str, Any],
    run_dir: str | Path,
    plumed_script: str | None = None,
    box_vectors: Any = None,
    extra_reporters: Sequence[Any] = (),
) -> MDResult:
    """Minimise, equilibrate and run production MD; write outputs into ``run_dir``.

    Resumes from ``run_dir/checkpoint.chk`` when ``md.resume`` is true. The returned
    ``wall_clock_s`` covers context creation through the end of production on this
    invocation; ``ns_per_day`` is from production steps only.
    """
    t0 = time.perf_counter()
    rd = Path(run_dir).resolve()
    rd.mkdir(parents=True, exist_ok=True)
    chk = rd / "checkpoint.chk"
    xtc = rd / "traj.xtc"
    log_path = rd / "md.log"
    top_pdb = rd / "topology.pdb"
    n_steps = int(md["n_steps"])
    dt_fs = float(md["timestep_fs"])
    atoms = select_atoms(topology, md.get("xtc_selection", "solute"))

    with _maybe_chdir(rd, bool(plumed_script)):
        sim = build_simulation(topology, system, positions, md, plumed_script, box_vectors)
        gpu = device_name(sim.context)
        platform = sim.context.getPlatform().getName()
        log.info("platform=%s device=%s atoms=%d", platform, gpu, topology.getNumAtoms())

        resumed_from = 0
        resuming = bool(md.get("resume", True)) and chk.exists()
        if resuming:
            sim.loadCheckpoint(str(chk))
            resumed_from = sim.currentStep
            log.info("resumed from checkpoint at step %d", resumed_from)
        else:
            for stale in (xtc, log_path):
                stale.unlink(missing_ok=True)
            if md.get("minimize", True):
                sim.minimizeEnergy(
                    tolerance=float(md.get("minimize_tolerance_kj_mol_nm", 10.0))
                    * unit.kilojoule_per_mole
                    / unit.nanometer,
                    maxIterations=int(md.get("minimize_max_iterations", 0)),
                )
            temp = float(md["temperature_K"]) * unit.kelvin
            seed = int(md.get("seed") or 0)
            if seed:
                sim.context.setVelocitiesToTemperature(temp, seed)
            else:
                sim.context.setVelocitiesToTemperature(temp)
            eq = int(md.get("equilibration_steps", 0))
            if eq > 0:
                sim.step(eq)
            sim.currentStep = 0
            sim.context.setTime(0.0)
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
            _write_subset_pdb(topology, state.getPositions(), atoms, top_pdb)

        remaining = max(0, n_steps - sim.currentStep)
        append = resuming and log_path.exists()
        traj_int = int(md["traj_interval_steps"])
        log_int = int(md["log_interval_steps"])
        chk_int = int(md["checkpoint_interval_steps"])
        write_traj = 0 < traj_int <= n_steps
        if write_traj:
            sim.reporters.append(
                app.XTCReporter(
                    str(xtc), traj_int, append=resuming and xtc.exists(), atomSubset=atoms
                )
            )
        if log_int > 0:
            sim.reporters.append(
                app.StateDataReporter(
                    str(log_path),
                    log_int,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    speed=True,
                    append=append,
                )
            )
        if chk_int > 0:
            sim.reporters.append(app.CheckpointReporter(str(chk), chk_int))
        sim.reporters.extend(extra_reporters)

        t_prod = time.perf_counter()
        if remaining:
            sim.step(remaining)
        prod_s = time.perf_counter() - t_prod
        sim.saveCheckpoint(str(chk))
        state = sim.context.getState(getPositions=True)
        _write_subset_pdb(topology, state.getPositions(), atoms, rd / "final.pdb")
        # Destroy the context here, inside run_dir, so PLUMED finalises and flushes
        # COLVAR/HILLS before the caller reads them.
        del state, sim
        gc.collect()

    wall = time.perf_counter() - t0
    ns = remaining * dt_fs * 1e-6
    result = MDResult(
        run_dir=str(rd),
        platform=platform,
        gpu=gpu,
        n_atoms=topology.getNumAtoms(),
        n_steps=remaining,
        timestep_fs=dt_fs,
        wall_clock_s=wall,
        production_wall_clock_s=prod_s,
        ns_per_day=(ns / prod_s * 86400.0) if prod_s > 0 and ns > 0 else 0.0,
        resumed_from_step=resumed_from,
        trajectory=str(xtc) if write_traj else None,
        topology=str(top_pdb) if top_pdb.exists() else None,
        log=str(log_path),
    )
    result.write(rd / "md_result.json")
    return result


__all__ = [
    "SOLVENT_RESNAMES",
    "MDResult",
    "build_simulation",
    "device_name",
    "make_integrator",
    "run_md",
    "select_atoms",
    "solute_indices",
]
