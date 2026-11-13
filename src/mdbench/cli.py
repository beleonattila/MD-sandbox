"""Command line: ``python -m mdbench {run,nightly,score,report,bench,export-motion,list}``.

``run <config.yml>`` resolves the config, creates the run directory, writes
``config.resolved.yml`` there, dispatches on (branch, system, level, method) to the
registered runner, fills provenance columns and appends the rows to
``results/<branch>.parquet``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mdbench import budget, get_code_version
from mdbench.config import config_hash, load_config, run_dir, write_resolved
from mdbench.registry import get_runner, is_pausable, registered
from mdbench.schema import SchemaError, append_rows

STATUS_NAME = "status.json"
EXIT_PAUSED = 75  # EX_TEMPFAIL: stopped at a restart point, run the same config again


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


def read_status(rd: Path) -> dict[str, Any]:
    """The run directory's ``status.json`` (state, sessions, cumulative elapsed), or ``{}``."""
    try:
        return json.loads((rd / STATUS_NAME).read_text())
    except (OSError, ValueError):
        return {}


def _write_status(rd: Path, state: str, sessions: int, elapsed_s: float, note: str = "") -> None:
    doc = {
        "state": state,
        "sessions": sessions,
        "elapsed_s": round(elapsed_s, 1),
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": note,
    }
    (rd / STATUS_NAME).write_text(json.dumps(doc, indent=1))


def cmd_run(
    config_path: str, minutes: float | None = None, grace_minutes: float = 0.0
) -> list[dict[str, Any]] | None:
    """Run one config end to end; returns the rows appended.

    With ``minutes`` (and for pausable runners, also on SIGTERM/Ctrl-C) the run stops at its
    next restart point once the time is up and returns ``None``; running the same config
    again continues it. Elapsed time is summed over sessions in ``status.json`` and used as
    the default ``wall_clock_s`` of the rows.
    """
    cfg = load_config(config_path)
    fn = get_runner(cfg["branch"], cfg["system"], cfg["level"], cfg["method"])
    pausable = is_pausable(cfg["branch"], cfg["system"], cfg["level"], cfg["method"])
    rd = run_dir(cfg)
    write_resolved(cfg, rd)
    prior = read_status(rd)
    sessions = int(prior.get("sessions", 0)) + 1
    prior_s = float(prior.get("elapsed_s", 0.0))
    if minutes is not None and not pausable:
        print(
            f"[run] {cfg['branch']}/{cfg['system']}/{cfg['level']}/{cfg['method']} cannot be "
            "paused; ignoring --minutes and running to completion",
            file=sys.stderr,
        )
    if pausable:
        budget.start(minutes, grace_minutes)
        budget.install_signal_handlers()
    _write_status(rd, "running", sessions, prior_s)
    t0 = time.perf_counter()
    try:
        rows = list(fn(cfg, rd))
    except budget.RunPaused as paused:
        total = prior_s + time.perf_counter() - t0
        _write_status(rd, "paused", sessions, total, str(paused))
        print(f"[run] paused after {total / 60:.1f} min in {sessions} session(s): {paused}")
        print(f"[run] rerun the same config to continue (run dir {rd})")
        return None
    except BaseException as err:
        _write_status(rd, "failed", sessions, prior_s + time.perf_counter() - t0, repr(err))
        raise
    finally:
        if pausable:
            budget.stop()
    elapsed = prior_s + time.perf_counter() - t0
    try:
        full = [complete_row(r, cfg, elapsed) for r in rows]
        path = append_rows(cfg["branch"], full)
    except SchemaError:
        # The rows of this exact config are already recorded: the run is done, not failed.
        _write_status(rd, "done", sessions, prior_s, "rows already recorded; nothing appended")
        raise
    except BaseException as err:
        _write_status(rd, "failed", sessions, elapsed, repr(err))
        raise
    _write_status(rd, "done", sessions, elapsed)
    print(f"[run] {len(full)} row(s) -> {path}  (run dir {rd})")
    return full


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="mdbench", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one config and append its rows")
    r.add_argument("config")
    r.add_argument(
        "--minutes", type=float, default=None,
        help="stop at the next restart point after this many minutes (pausable runs only); "
        f"exit code {EXIT_PAUSED} means paused, rerun to continue",
    )  # fmt: skip
    r.add_argument(
        "--grace-minutes", type=float, default=0.0,
        help="may overrun --minutes by up to this long to reach a cheaper restart point "
        "(ABFE checkpoint boundary)",
    )  # fmt: skip

    n = sub.add_parser("nightly", help="run a queue of configs inside a time budget")
    n.add_argument("--minutes", type=float, required=True, help="total budget for this session")
    n.add_argument("--queue", default="queue/nightly.yml")
    n.add_argument("--grace-minutes", type=float, default=0.0)
    n.add_argument("--min-slice-minutes", type=float, default=10.0,
                   help="do not start a run with less than this left")  # fmt: skip
    n.add_argument("--dry-run", action="store_true", help="show the plan, run nothing")

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
        rows = cmd_run(a.config, a.minutes, a.grace_minutes)
        return EXIT_PAUSED if rows is None else 0
    elif a.cmd == "nightly":
        from mdbench.nightly import run_nightly

        return run_nightly(
            Path(a.queue), a.minutes, a.grace_minutes, a.min_slice_minutes, a.dry_run
        )
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
