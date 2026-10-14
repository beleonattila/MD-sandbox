"""Command line: ``python -m mdbench {run,score,report,bench,export-motion,list}``.

``run <config.yml>`` resolves the config, creates the run directory, writes
``config.resolved.yml`` there, dispatches on (branch, system, level, method) to the
registered runner, fills provenance columns and appends the rows to
``results/<branch>.parquet``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

from mdbench import get_code_version
from mdbench.config import config_hash, load_config, run_dir, write_resolved
from mdbench.registry import get_runner, registered
from mdbench.schema import append_rows


def complete_row(
    row: Mapping[str, Any], cfg: Mapping[str, Any], elapsed_s: float
) -> dict[str, Any]:
    """Fill the provenance columns the CLI owns; runner-provided gpu/wall_clock win."""
    out = dict(row)
    owned = {
        "system": cfg["system"],
        "branch": cfg["branch"],
        "level": cfg["level"],
        "method": cfg["method"],
        "replicate": int(cfg.get("replicate", 0)),
        "code_version": get_code_version(),
        "config_hash": config_hash(cfg),
    }
    # A runner may emit rows for several systems (e.g. a congeneric series); every other
    # owned column must agree with the config.
    for k, v in owned.items():
        if k != "system" and k in out and out[k] != v:
            raise ValueError(f"runner returned {k}={out[k]!r}, config says {v!r}")
        out.setdefault(k, v)
    out.setdefault("gpu", "none")
    out.setdefault("wall_clock_s", elapsed_s)
    out.setdefault("reference", None)
    out.setdefault("reference_source", None)
    return out


def cmd_run(config_path: str) -> list[dict[str, Any]]:
    """Run one config end to end; returns the rows appended."""
    cfg = load_config(config_path)
    fn = get_runner(cfg["branch"], cfg["system"], cfg["level"], cfg["method"])
    rd = run_dir(cfg)
    write_resolved(cfg, rd)
    t0 = time.perf_counter()
    rows = list(fn(cfg, rd))
    elapsed = time.perf_counter() - t0
    full = [complete_row(r, cfg, elapsed) for r in rows]
    path = append_rows(cfg["branch"], full)
    print(f"[run] {len(full)} row(s) -> {path}  (run dir {rd})")
    return full


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="mdbench", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one config and append its rows")
    r.add_argument("config")

    s = sub.add_parser("score", help="score results/<branch>.parquet against references")
    s.add_argument("branch")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)

    sub.add_parser("report", help="ladder plots from results/*.parquet")

    b = sub.add_parser("bench", help="timed MD on a water box; prints ns/day")
    b.add_argument("config", nargs="?", default=None)
    b.add_argument("--n-atoms", type=int, default=None)

    mo = sub.add_parser("export-motion", help="ala2 MD + metad trajectories as viewer JSON")
    mo.add_argument("md_rep", help="an md_histogram replicate dir, e.g. runs/c/ala2/L2-.../rep_2")
    mo.add_argument("metad_rep", help="a wtmetad_reweight replicate dir")
    mo.add_argument("-o", "--out", default="runs/c/ala2/motion.json")

    lst = sub.add_parser("list", help="list registered runners")
    lst.add_argument("branch", nargs="?", default=None)

    a = p.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a.config)
    elif a.cmd == "score":
        import pandas as pd

        from mdbench.score import score_branch

        scores, path = score_branch(a.branch, n_boot=a.n_boot, seed=a.seed)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(scores if not scores.empty else "(no rows with a reference)")
        print(f"[score] -> {path}")
    elif a.cmd == "report":
        from mdbench.report import report_all

        for path in report_all():
            print(f"[report] -> {path}")
    elif a.cmd == "bench":
        from mdbench.bench import run_bench

        summ = run_bench(a.config, a.n_atoms)
        print(
            f"[bench] {summ['n_atoms']} atoms on {summ['gpu']} ({summ['platform']}, "
            f"{summ['precision']}): {summ['ns_per_day']:.1f} ns/day"
        )
    elif a.cmd == "export-motion":
        from pathlib import Path

        from mdbench.branch_c.motion import export_motion

        path = export_motion(Path(a.md_rep), Path(a.metad_rep), Path(a.out))
        print(f"[export-motion] -> {path}")
    elif a.cmd == "list":
        from mdbench.registry import load_branch

        if a.branch:
            load_branch(a.branch)
        for key in registered():
            if a.branch is None or key[0] == a.branch.lower():
                print(" ".join(key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
