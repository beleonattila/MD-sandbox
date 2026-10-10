from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mdbench import cli
from mdbench.config import RESOLVED_NAME
from mdbench.registry import get_runner, register
from mdbench.schema import SchemaError, read_results

CALLS: list[Path] = []


@register(branch="zz", system="toy", level="L0", method="constant")
def _toy(cfg: dict, run_dir: Path) -> list[dict]:
    CALLS.append(run_dir)
    assert (run_dir / RESOLVED_NAME).exists()
    return [{"prediction": float(cfg["params"]["value"]), "uncertainty": 0.1}]


@register(branch="zz", system="toy", level="L1", method="no_error_bar")
def _bad(cfg: dict, run_dir: Path) -> list[dict]:
    return [{"prediction": 1.0}]


def _write_cfg(tmp_path: Path, **kw) -> Path:
    cfg = {
        "branch": "zz",
        "system": "toy",
        "level": "L0",
        "method": "constant",
        "params": {"value": 4.2},
        "output": {"root": str(tmp_path / "runs")},
    }
    cfg.update(kw)
    p = tmp_path / "toy.yml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_run_dispatch_appends_row(tmp_path: Path, isolated_results: Path):
    cfg = _write_cfg(tmp_path)
    assert cli.main(["run", str(cfg)]) == 0
    with pytest.raises(SchemaError, match="already in"):
        cli.main(["run", str(cfg)])
    df = read_results("zz").to_pandas()
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.system, r.branch, r.level, r.method) == ("toy", "zz", "L0", "constant")
    assert r.prediction == 4.2 and r.uncertainty == 0.1 and r.gpu == "none"
    assert len(r.config_hash) == 64 and r.code_version
    assert CALLS[-1].is_relative_to(tmp_path / "runs" / "zz" / "toy")


def test_run_rejects_missing_uncertainty(tmp_path: Path, isolated_results: Path):
    cfg = _write_cfg(tmp_path, level="L1", method="no_error_bar")
    with pytest.raises(Exception, match="missing"):
        cli.main(["run", str(cfg)])
    assert read_results("zz").num_rows == 0


def test_unknown_runner(tmp_path: Path, isolated_results: Path):
    with pytest.raises(KeyError):
        get_runner("zz", "toy", "L3", "nope")


def test_score_and_list(tmp_path: Path, isolated_results: Path, capsys):
    from mdbench.schema import append_rows
    from mdbench.score import score_frame

    base = {
        "branch": "zz", "level": "L0", "method": "m", "replicate": 0, "uncertainty": 0.1,
        "wall_clock_s": 1.0, "gpu": "none", "code_version": "x", "config_hash": "h",
        "reference_source": "test-only",
    }  # fmt: skip
    rows = [
        {**base, "system": f"s{i}", "prediction": float(i), "reference": float(i) + 0.5}
        for i in range(5)
    ]
    append_rows("zz", rows)
    scores = score_frame(read_results("zz").to_pandas(), n_boot=50)
    across = scores[scores.scope == "across_systems"].iloc[0]
    assert across.spearman == pytest.approx(1.0) and across.pairwise == pytest.approx(1.0)
    assert across.mue == pytest.approx(0.5)
    cli.main(["list", "zz"])
    assert "zz toy L0 constant" in capsys.readouterr().out
