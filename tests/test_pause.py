"""Clean stop / resume: budget module, run_md chunked stop, CLI status, nightly queue."""

from __future__ import annotations

import copy
import json
import signal
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from mdbench import budget, cli
from mdbench.config import load_defaults
from mdbench.engines.md import run_md
from mdbench.registry import is_pausable, register
from mdbench.schema import read_results


@pytest.fixture(autouse=True)
def _clean_budget():
    yield
    budget.stop()


def test_budget_inactive_by_default():
    assert not budget.active() and not budget.stop_requested()
    assert budget.seconds_left() is None


def test_budget_deadline_and_grace():
    budget.start(0.0)
    assert budget.stop_requested() and budget.grace_left() == 0.0
    budget.stop()
    budget.start(0.0, grace_minutes=5)
    assert budget.stop_requested() and 290 < budget.grace_left() <= 300
    budget.stop()
    budget.start(60.0)
    assert not budget.stop_requested() and budget.seconds_left() > 3500


def test_budget_sigterm_requests_stop_without_deadline():
    budget.start(None)
    budget.install_signal_handlers()
    assert not budget.stop_requested()
    signal.raise_signal(signal.SIGTERM)
    assert budget.stop_requested() and budget.signalled() and budget.grace_left() == 0.0
    with pytest.raises(KeyboardInterrupt):  # a second signal gives up waiting
        signal.raise_signal(signal.SIGINT)


def _md(**kw):
    md = dict(load_defaults()["md"])
    md.update(platform="CPU", fallback_platform=None, minimize_max_iterations=200, seed=77)
    md.update(kw)
    return md


def _stop_after(monkeypatch, n_checks: int) -> None:
    """Make the n-th stop poll (and later ones) report a stop."""
    calls = {"n": 0}

    def fake() -> bool:
        calls["n"] += 1
        return calls["n"] > n_checks

    monkeypatch.setattr(budget, "stop_requested", fake)


def test_run_md_pause_and_resume_keeps_frames_consistent(tiny_water, tmp_path, monkeypatch):
    top, system, pos = tiny_water
    md = _md(
        n_steps=400, equilibration_steps=100, traj_interval_steps=50, log_interval_steps=50,
        checkpoint_interval_steps=0, xtc_selection=list(range(9)),
    )  # fmt: skip
    ref = run_md(top, copy.deepcopy(system), pos, md, tmp_path / "ref")

    rd = tmp_path / "paused"
    budget.start(None)
    _stop_after(monkeypatch, 3)  # chunks of 50 steps: stops at step 150
    with pytest.raises(budget.RunPaused):
        run_md(top, copy.deepcopy(system), pos, md, rd)
    res = json.loads((rd / "md_result.json").read_text())
    assert res["extra"]["paused"] and res["extra"]["stopped_at_step"] == 150
    assert (rd / "checkpoint.chk").exists() and not (rd / "final.pdb").exists()
    monkeypatch.undo()
    budget.stop()

    out = run_md(top, copy.deepcopy(system), pos, md, rd)
    assert out.resumed_from_step == 150 and out.n_steps == 250
    assert out.wall_clock_s >= out.session_wall_clock_s > 0  # cumulative over both sessions
    import MDAnalysis as mda

    u = mda.Universe(str(rd / "topology.pdb"), str(rd / "traj.xtc"))
    u0 = mda.Universe(str(ref.topology), str(ref.trajectory))
    assert len(u.trajectory) == len(u0.trajectory) == 8  # no duplicated or missing frames
    times = [ts.time for ts in u.trajectory]
    assert times == sorted(set(times))
    # Not bit-identical to `ref`: even two uninterrupted CPU runs with one seed diverge.
    assert np.asarray(out.final_positions).shape == np.asarray(ref.final_positions).shape
    assert np.all(np.isfinite(np.asarray(out.final_positions)))


def test_run_md_without_budget_is_unchanged(tiny_water, tmp_path):
    top, system, pos = tiny_water
    md = _md(n_steps=60, traj_interval_steps=0, log_interval_steps=0, checkpoint_interval_steps=0)
    res = run_md(top, copy.deepcopy(system), pos, md, tmp_path / "plain")
    assert res.n_steps == 60 and not res.extra


def test_pause_ignored_when_resume_disabled(tiny_water, tmp_path, monkeypatch):
    top, system, pos = tiny_water
    md = _md(n_steps=60, resume=False, traj_interval_steps=0, log_interval_steps=0,
             checkpoint_interval_steps=0)  # fmt: skip
    budget.start(0.0)
    res = run_md(top, copy.deepcopy(system), pos, md, tmp_path / "nopause")
    assert res.n_steps == 60  # metadynamics-style runs cannot restart, so they never pause


def test_plumed_colvar_continues_across_pause(tiny_water, tmp_path, monkeypatch):
    pytest.importorskip("openmmplumed")
    top, system, pos = tiny_water
    script = "d: DISTANCE ATOMS=1,4\nPRINT ARG=d STRIDE=10 FILE=COLVAR\n"
    md = _md(n_steps=200, traj_interval_steps=0, log_interval_steps=20, checkpoint_interval_steps=0)
    rd = tmp_path / "pl"
    budget.start(None)
    _stop_after(monkeypatch, 2)
    with pytest.raises(budget.RunPaused):
        run_md(top, copy.deepcopy(system), pos, md, rd, plumed_script=script)
    monkeypatch.undo()
    budget.stop()
    run_md(top, copy.deepcopy(system), pos, md, rd, plumed_script=script)
    assert not list(rd.glob("bck.*"))  # PLUMED did not move the first half aside
    from mdbench.engines.plumed_cv import read_colvar

    raw = read_colvar(rd / "COLVAR")["time"]
    assert len(raw) == 22 and np.sum(np.diff(raw) == 0) == 1  # PLUMED repeats the restart row
    t = read_colvar(rd / "COLVAR", dedupe_time=True)["time"]
    assert len(t) == 21 and np.all(np.diff(t) > 0)


# ---- CLI ---------------------------------------------------------------------------------

STATE = {"calls": 0}


@register(branch="zz", system="toy", level="L0", method="pausable_toy", pausable=True)
def _pausable(cfg: dict, run_dir: Path) -> list[dict]:
    STATE["calls"] += 1
    if STATE["calls"] == 1:
        raise budget.RunPaused("toy paused")
    return [{"prediction": 1.0, "uncertainty": 0.1}]


@register(branch="zz", system="toy", level="L0", method="plain_toy")
def _plain(cfg: dict, run_dir: Path) -> list[dict]:
    return [{"prediction": 2.0, "uncertainty": 0.1}]


def _cfg(tmp_path: Path, method: str) -> Path:
    cfg = {"branch": "zz", "system": "toy", "level": "L0", "method": method,
           "output": {"root": str(tmp_path / "runs")}}  # fmt: skip
    p = tmp_path / f"{method}.yml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_cli_pause_then_complete(tmp_path, isolated_results):
    STATE["calls"] = 0
    cfg = _cfg(tmp_path, "pausable_toy")
    assert is_pausable("zz", "toy", "L0", "pausable_toy")
    assert cli.main(["run", str(cfg), "--minutes", "5"]) == cli.EXIT_PAUSED
    rd = next((tmp_path / "runs").rglob(cli.STATUS_NAME)).parent
    st = cli.read_status(rd)
    assert st["state"] == "paused" and st["sessions"] == 1
    assert not budget.active()  # handlers and deadline are cleaned up
    assert read_results("zz").num_rows == 0  # a pause appends nothing
    st["elapsed_s"] = 1000.0  # pretend the first night used 1000 s
    (rd / cli.STATUS_NAME).write_text(json.dumps(st))
    assert cli.main(["run", str(cfg)]) == 0
    st = cli.read_status(rd)
    assert st["state"] == "done" and st["sessions"] == 2
    row = read_results("zz").to_pandas().iloc[0]
    assert row["wall_clock_s"] >= 1000.0  # cost counts every night


def test_cli_minutes_ignored_for_unpausable(tmp_path, isolated_results, capsys):
    p = _cfg(tmp_path, "plain_toy")
    assert cli.main(["run", str(p), "--minutes", "1"]) == 0
    assert "cannot be paused" in capsys.readouterr().err


# ---- nightly -----------------------------------------------------------------------------


def test_nightly_dry_run_and_done_detection(tmp_path, isolated_results, capsys):
    from mdbench import nightly

    STATE["calls"] = 1
    a = _cfg(tmp_path, "pausable_toy")
    q = tmp_path / "q.yml"
    q.write_text(yaml.safe_dump({"runs": [str(a)]}))
    entries = nightly.load_queue(q)
    assert len(entries) == 1 and not nightly.is_done(entries[0])
    assert nightly.run_nightly(q, 30, dry_run=True) == 0
    assert "pausable_toy" in capsys.readouterr().out
    assert cli.main(["run", str(a)]) == 0
    assert nightly.is_done(entries[0])


def test_nightly_requires_queue_file(tmp_path, capsys):
    from mdbench import nightly

    assert nightly.run_nightly(tmp_path / "missing.yml", 30) == 2
    assert "queue" in capsys.readouterr().out


def test_nightly_runs_child_and_stops_on_pause(tmp_path, isolated_results, monkeypatch):
    """Queue loop with a fake child process: first pauses (budget used), second never starts."""
    import subprocess
    import sys

    from mdbench import nightly

    a = _cfg(tmp_path, "pausable_toy")
    q = tmp_path / "q.yml"
    q.write_text(yaml.safe_dump({"runs": [str(a), str(a)]}))
    monkeypatch.setattr(nightly, "activated_env", lambda env: {"CONDA_PREFIX": sys.prefix})
    monkeypatch.setattr(nightly, "LOCK_PATH", tmp_path / "lock")
    launched: list[list[str]] = []
    entry = nightly.load_queue(q)[0]

    class FakeChild:
        def __init__(self, cmd, **kw):
            launched.append(cmd)
            entry.rundir.mkdir(parents=True, exist_ok=True)
            (entry.rundir / cli.STATUS_NAME).write_text(json.dumps({"state": "paused"}))

        def wait(self):
            return 75

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeChild)
    assert nightly.run_nightly(q, 90, min_slice_minutes=5) == 0
    assert len(launched) == 1
    assert launched[0][-2:] == ["--minutes", launched[0][-1]] or "--minutes" in launched[0]
    minutes = float(launched[0][launched[0].index("--minutes") + 1])
    assert 80 < minutes <= 90


def test_nightly_skips_when_too_little_time(tmp_path, isolated_results, monkeypatch):
    from mdbench import nightly

    a = _cfg(tmp_path, "pausable_toy")
    q = tmp_path / "q.yml"
    q.write_text(yaml.safe_dump({"runs": [str(a)]}))
    monkeypatch.setattr(nightly, "LOCK_PATH", tmp_path / "lock")
    t0 = time.monotonic()
    assert nightly.run_nightly(q, 3, min_slice_minutes=10) == 0  # nothing launched
    assert time.monotonic() - t0 < 5


# ---- ABFE stop logic ---------------------------------------------------------------------


class _FakeSampler:
    """Counts iterations; ``run(n_iterations=k)`` advances k (like the openmmtools sampler)."""

    def __init__(self, start: int = 0) -> None:
        self.iteration = start

    def run(self, n_iterations: int) -> None:
        self.iteration += n_iterations


def test_abfe_runs_to_completion_without_a_stop():
    from mdbench.engines.abfe import _run_iterations

    s = _FakeSampler()
    assert _run_iterations(s, 23, checkpoint_interval=10) is False and s.iteration == 23


def test_abfe_stop_is_immediate_without_grace(monkeypatch):
    from mdbench.engines import abfe

    s = _FakeSampler(start=10)
    budget.start(0.0)
    assert abfe._run_iterations(s, 40, checkpoint_interval=10) is True and s.iteration == 10
    budget.stop()

    # A stop arriving mid-block with no grace: stop where we are; the block is redone on resume.
    budget.start(None)
    s = _FakeSampler(start=8)
    monkeypatch.setattr(budget, "stop_requested", lambda: s.iteration >= 13)
    assert abfe._run_iterations(s, 40, checkpoint_interval=8) is True
    assert s.iteration == 13


def test_abfe_grace_runs_on_to_the_checkpoint_boundary(monkeypatch):
    from mdbench.engines import abfe

    budget.start(0.0, grace_minutes=60)  # real grace_left() ~ 3600 s
    s = _FakeSampler(start=0)
    monkeypatch.setattr(budget, "stop_requested", lambda: s.iteration >= 10)
    monkeypatch.setattr(abfe.time, "perf_counter", lambda: float(s.iteration))  # 1 s/iteration
    assert abfe._run_iterations(s, 40, checkpoint_interval=8) is True
    assert s.iteration == 16  # ran 6 more iterations (6 s < grace) to reach the boundary

    budget.stop()
    budget.start(0.0, grace_minutes=0.05)  # 3 s of grace: the 6 s to the boundary is too far
    s = _FakeSampler(start=0)
    monkeypatch.setattr(budget, "stop_requested", lambda: s.iteration >= 10)
    assert abfe._run_iterations(s, 40, checkpoint_interval=8) is True
    assert s.iteration == 10


def test_restraint_state_pickles_by_name():
    pytest.importorskip("openmmtools")
    import pickle

    from mdbench.engines import abfe

    cls = abfe._restraint_state_class()
    assert cls is abfe.RestraintState and cls.__qualname__ == "RestraintState"
    assert pickle.loads(pickle.dumps(cls)) is cls  # openmmtools stores states by class path
