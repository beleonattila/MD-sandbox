from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mdbench.config import (
    RESOLVED_NAME,
    ConfigError,
    config_hash,
    load_config,
    resolve,
    run_dir,
    write_resolved,
)

BASE = {"branch": "c", "system": "ala2", "level": "L2", "method": "unbiased_md"}


def test_hash_stable_under_key_order():
    a = resolve({**BASE, "md": {"n_steps": 100, "temperature_K": 300.0}})
    b = resolve({"md": {"temperature_K": 300.0, "n_steps": 100}, **dict(reversed(BASE.items()))})
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 64


def test_hash_known_value_is_deterministic():
    # Same input twice in fresh resolutions gives the same digest.
    assert config_hash(resolve(BASE)) == config_hash(resolve(dict(BASE)))


def test_hash_changes_with_physics_not_output():
    a = resolve(BASE)
    b = resolve({**BASE, "md": {"temperature_K": 310.0}})
    c = resolve({**BASE, "output": {"root": "/elsewhere"}})
    assert config_hash(a) != config_hash(b)
    assert config_hash(a) == config_hash(c)


def test_defaults_applied_and_required_keys():
    cfg = resolve(BASE)
    assert cfg["md"]["integrator"] == "langevin_middle"
    assert cfg["replicate"] == 0
    with pytest.raises(ConfigError):
        resolve({"branch": "c", "system": "ala2", "level": "L2"})


def test_extends_and_resolved_round_trip(tmp_path: Path):
    (tmp_path / "base.yml").write_text(yaml.safe_dump({**BASE, "md": {"n_steps": 10}}))
    (tmp_path / "child.yml").write_text(
        yaml.safe_dump({"extends": "base.yml", "replicate": 3, "md": {"timestep_fs": 4.0}})
    )
    cfg = load_config(tmp_path / "child.yml")
    assert cfg["md"]["n_steps"] == 10 and cfg["md"]["timestep_fs"] == 4.0
    assert cfg["replicate"] == 3
    out = write_resolved(cfg, tmp_path / "run")
    assert out.name == RESOLVED_NAME
    again = load_config(out)
    assert config_hash(again) == config_hash(cfg)
    meta = yaml.safe_load(out.read_text())["_meta"]
    assert meta["config_hash"] == config_hash(cfg)


def test_run_dir_layout(tmp_path: Path):
    cfg = resolve({**BASE, "output": {"root": str(tmp_path)}})
    rd = run_dir(cfg)
    assert rd.parent == tmp_path / "c" / "ala2"
    assert rd.name.startswith("L2-unbiased_md-r0-")
