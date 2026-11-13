"""Nightly queue: run unfinished configs one after another inside a wall-clock budget.

``python -m mdbench nightly --minutes 480`` reads ``queue/nightly.yml``::

    default_env: mdbench-md
    runs:
      - configs/c_adk_l2_unbiased.yml
      - config: configs/a_t4l_l3_abfe_benzene.yml
        env: mdbench-fe

Entries run in order. A finished entry (its ``config_hash`` is in ``results/<branch>.parquet``)
is skipped, so the queue is the same file every night: the first unfinished entry gets the
whole budget, pauses at a restart point when time is up, and continues the next night. When it
finishes early the next entry gets what is left. Entries whose runner cannot pause are
skipped (they would overrun the window); run those by hand.

Each run is a child process started with the environment's own Python (activated once with
``micromamba run``), so SIGTERM reaches the engine directly: stopping the queue, or the
service manager stopping it at the end of the night, pauses the current run cleanly.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mdbench import REPO_ROOT
from mdbench.cli import read_status
from mdbench.config import config_hash, load_config, run_dir
from mdbench.registry import is_pausable
from mdbench.schema import read_results

MICROMAMBA = Path(os.environ.get("MICROMAMBA", Path.home() / ".local/bin/micromamba"))
MAMBA_ROOT = Path(os.environ.get("MAMBA_ROOT", Path.home() / "micromamba"))
LOCK_PATH = REPO_ROOT / "runs" / ".nightly.lock"


@dataclass
class Entry:
    """One queue entry, resolved."""

    config: Path
    env: str
    branch: str
    key: tuple[str, str, str, str]
    chash: str
    rundir: Path


def load_queue(path: Path) -> list[Entry]:
    """Parse the queue file into resolved entries (configs are loaded, so errors surface now)."""
    doc: Mapping[str, Any] = yaml.safe_load(path.read_text()) or {}
    default_env = str(doc.get("default_env", "mdbench-md"))
    out: list[Entry] = []
    for item in doc.get("runs") or []:
        spec = {"config": item} if isinstance(item, str) else dict(item)
        cfg_path = Path(spec["config"])
        if not cfg_path.is_absolute():
            cfg_path = REPO_ROOT / cfg_path
        cfg = load_config(cfg_path)
        out.append(
            Entry(
                config=cfg_path,
                env=str(spec.get("env", default_env)),
                branch=str(cfg["branch"]),
                key=(cfg["branch"], cfg["system"], cfg["level"], cfg["method"]),
                chash=config_hash(cfg),
                rundir=run_dir(cfg),
            )
        )
    return out


def is_done(entry: Entry) -> bool:
    """True when ``results/<branch>.parquet`` already holds a row of this config."""
    try:
        table = read_results(entry.branch)
    except Exception:  # no results file yet
        return False
    if "config_hash" not in table.column_names:
        return False
    return entry.chash in set(table.column("config_hash").to_pylist())


def activated_env(env: str) -> dict[str, str]:
    """The environment variables ``micromamba run -n <env>`` would set (PLUMED_KERNEL, ...)."""
    code = "import json,os;print('@@'+json.dumps(dict(os.environ)))"
    cmd = [str(MICROMAMBA), "run", "-r", str(MAMBA_ROOT), "-n", env, "python", "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith("@@"))
    return json.loads(line[2:])


def _fmt(minutes: float) -> str:
    return f"{minutes / 60:.1f} h" if minutes >= 120 else f"{minutes:.0f} min"


def _say(msg: str) -> None:
    print(f"[nightly {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_nightly(
    queue: Path,
    minutes: float,
    grace_minutes: float = 0.0,
    min_slice_minutes: float = 10.0,
    dry_run: bool = False,
) -> int:
    """Work through the queue until ``minutes`` are used. Returns 0, or 1 if a run failed."""
    if not queue.exists():
        _say(f"no queue file {queue}; copy queue/nightly.example.yml and choose the runs")
        return 2
    entries = load_queue(queue)
    deadline = time.monotonic() + minutes * 60.0

    if dry_run:
        for e in entries:
            st = read_status(e.rundir)
            state = "done" if is_done(e) else st.get("state", "new")
            note = "" if is_pausable(*e.key) else "  [not pausable: will be skipped]"
            spent = float(st.get("elapsed_s", 0.0)) / 3600
            _say(f"{e.config.name:45s} {state:8s} {spent:5.1f} h spent  env={e.env}{note}")
        return 0

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _say("another nightly session holds the lock; exiting")
        return 3

    child: subprocess.Popen[bytes] | None = None
    stopping = False

    def forward(signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)  # the engine pauses at its next restart point

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    failed = False
    env_cache: dict[str, dict[str, str]] = {}
    _say(f"budget {_fmt(minutes)}, queue {queue} ({len(entries)} entries)")
    for e in entries:
        if stopping:
            break
        if is_done(e):
            continue
        if not is_pausable(*e.key):
            _say(f"skip {e.config.name}: runner cannot pause, run it by hand")
            continue
        left = (deadline - time.monotonic()) / 60.0
        if left < min_slice_minutes:
            _say(f"{_fmt(max(left, 0))} left, below the {_fmt(min_slice_minutes)} minimum: stop")
            break
        if e.env not in env_cache:
            env_cache[e.env] = activated_env(e.env)
        env = dict(env_cache[e.env], PYTHONPATH=str(REPO_ROOT / "src"))
        py = env_cache[e.env].get("CONDA_PREFIX", str(MAMBA_ROOT / "envs" / e.env)) + "/bin/python"
        cmd = [py, "-m", "mdbench", "run", str(e.config), "--minutes", f"{left:.2f}"]
        if grace_minutes:
            cmd += ["--grace-minutes", f"{grace_minutes:g}"]
        _say(f"start {e.config.name} with {_fmt(left)} (env {e.env})")
        child = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
        code = child.wait()
        child = None
        # Judge by what the run recorded, not by the exit code alone.
        state = "done" if is_done(e) else read_status(e.rundir).get("state", "?")
        _say(f"{e.config.name}: exit {code}, {state}")
        if state == "paused":
            break  # the budget is spent; the same entry continues next time
        if state != "done":
            failed = True  # leave it for a human; try the next entry
    _say("session over")
    return 1 if failed else 0


__all__ = ["Entry", "activated_env", "is_done", "load_queue", "run_nightly"]
