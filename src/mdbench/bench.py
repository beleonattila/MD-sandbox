"""Throughput benchmark: a timed OpenMM run on a TIP3P water box of a target atom count.

Built offline with ``openmm.app.Modeller`` (no downloads, no openmmtools). Parameters come
from a YAML file (default ``configs/bench.yml``) with a ``bench`` block and an ``md``
block merged over the package defaults. Prints ns/day and appends a line to
``reports/bench.csv`` so measured numbers can replace the estimates in
``docs/HARDWARE.md``.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openmm
from openmm import app, unit

from mdbench import REPO_ROOT, get_code_version
from mdbench.config import config_hash, deep_merge, load_defaults, load_raw
from mdbench.engines.md import run_md

# TIP3P liquid water at ~300 K: 33.4 molecules/nm³, 3 atoms each.
_ATOMS_PER_NM3 = 3 * 33.4


def build_water_box(
    n_atoms: int,
    forcefield_files: list[str],
    water_model: str,
    cutoff_nm: float,
    ewald_error_tolerance: float = 5e-4,
    switch_distance_nm: float | None = None,
) -> tuple[app.Topology, openmm.System, Any]:
    """Cubic water box with roughly ``n_atoms`` atoms, PME electrostatics, rigid water."""
    edge = (n_atoms / _ATOMS_PER_NM3) ** (1.0 / 3.0)
    ff = app.ForceField(*forcefield_files)
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(
        ff, model=water_model, boxSize=openmm.Vec3(edge, edge, edge) * unit.nanometer
    )
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=ewald_error_tolerance,
        switchDistance=None if switch_distance_nm is None else switch_distance_nm * unit.nanometer,
    )
    return modeller.topology, system, modeller.positions


def load_bench_config(path: str | Path | None) -> dict[str, Any]:
    """Defaults <- configs/bench.yml (or ``path``)."""
    path = Path(path) if path else REPO_ROOT / "configs" / "bench.yml"
    return deep_merge(load_defaults(), load_raw(path))


def run_bench(path: str | Path | None = None, n_atoms: int | None = None) -> dict[str, Any]:
    """Run the benchmark; ``n_atoms`` overrides the config. Returns a summary dict."""
    cfg = load_bench_config(path)
    if n_atoms is not None:
        cfg["bench"]["n_atoms"] = int(n_atoms)
    b = cfg["bench"]
    sw = b.get("switch_distance_nm")
    top, system, pos = build_water_box(
        int(b["n_atoms"]),
        list(b["forcefield"]),
        str(b["water_model"]),
        float(b["cutoff_nm"]),
        float(b.get("ewald_error_tolerance", 5e-4)),
        None if sw is None else float(sw),
    )
    h = config_hash(cfg)
    out_root = Path(cfg.get("output", {}).get("root") or "runs")
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    rd = out_root / "bench" / f"n{top.getNumAtoms()}-{h[:12]}"
    md = dict(cfg["md"], resume=False)
    res = run_md(top, system, pos, md, rd)
    summary = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": res.gpu,
        "platform": res.platform,
        "precision": md.get("precision"),
        "n_atoms": res.n_atoms,
        "timestep_fs": res.timestep_fs,
        "n_steps": res.n_steps,
        "production_s": round(res.production_wall_clock_s, 3),
        "ns_per_day": round(res.ns_per_day, 2),
        "code_version": get_code_version(),
        "config_hash": h,
    }
    (rd / "bench.json").write_text(json.dumps(summary, indent=2))
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    csv_path = reports / "bench.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)
    return summary


__all__ = ["build_water_box", "load_bench_config", "run_bench"]
