"""Run configuration: load YAML, resolve defaults, hash, and record next to the output.

A run config is a YAML mapping with at least ``branch``, ``system``, ``level`` and
``method``. It may name a parent with ``extends: <path>`` (relative to the file), which
is deep-merged underneath it; chains are allowed. :file:`defaults.yml` sits under
everything.

``config_hash`` is the sha256 of the canonical JSON of the resolved config with the
``output`` block removed, so the same physics hashes the same wherever it is written.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mdbench import REPO_ROOT, get_code_version

DEFAULTS_PATH: Path = Path(__file__).with_name("defaults.yml")
REQUIRED_KEYS: tuple[str, ...] = ("branch", "system", "level", "method")
UNHASHED_KEYS: frozenset[str] = frozenset({"output", "_meta"})
RESOLVED_NAME = "config.resolved.yml"


class ConfigError(ValueError):
    """The configuration is malformed or incomplete."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; mappings merge, everything else replaces."""
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_raw(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML config, following ``extends`` chains (without applying defaults)."""
    path = Path(path).resolve()
    if path in _seen:
        raise ConfigError(f"circular extends at {path}")
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(load_raw(parent_path, _seen | {path}), data)


def load_defaults() -> dict[str, Any]:
    """The package defaults (:file:`src/mdbench/defaults.yml`)."""
    return _read_yaml(DEFAULTS_PATH)


def _to_plain(obj: Any) -> Any:
    """Normalise to JSON-compatible types (tuples to lists, Paths to str)."""
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def resolve(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``raw`` onto the defaults and validate required keys."""
    cfg = _to_plain(deep_merge(load_defaults(), raw))
    cfg.pop("_meta", None)
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise ConfigError(f"config is missing required keys: {missing}")
    for key in REQUIRED_KEYS:
        cfg[key] = str(cfg[key])
    if not str(cfg["level"]).startswith("L"):
        raise ConfigError(f"level must look like L0..L4, got {cfg['level']!r}")
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, follow ``extends``, apply defaults and validate."""
    return resolve(load_raw(path))


def canonical_json(cfg: Mapping[str, Any]) -> str:
    """Canonical JSON of the hashed part of ``cfg`` (sorted keys, no whitespace)."""
    hashed = {k: v for k, v in _to_plain(cfg).items() if k not in UNHASHED_KEYS}
    return json.dumps(hashed, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(cfg: Mapping[str, Any]) -> str:
    """sha256 hex digest of :func:`canonical_json` of the resolved config."""
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()


def run_dir(cfg: Mapping[str, Any]) -> Path:
    """Directory for this run's outputs (not created)."""
    out = cfg.get("output") or {}
    if out.get("run_dir"):
        p = Path(out["run_dir"])
        return p if p.is_absolute() else REPO_ROOT / p
    root = Path(out.get("root") or "runs")
    if not root.is_absolute():
        root = REPO_ROOT / root
    name = f"{cfg['level']}-{cfg['method']}-r{cfg.get('replicate', 0)}-{config_hash(cfg)[:12]}"
    return root / str(cfg["branch"]) / str(cfg["system"]) / name


def write_resolved(cfg: Mapping[str, Any], directory: str | Path) -> Path:
    """Write the resolved config, plus a ``_meta`` provenance block, into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    doc = _to_plain(cfg)
    doc["_meta"] = {
        "config_hash": config_hash(cfg),
        "code_version": get_code_version(),
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = directory / RESOLVED_NAME
    with path.open("w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    return path


__all__ = [
    "ConfigError",
    "canonical_json",
    "config_hash",
    "deep_merge",
    "load_config",
    "load_defaults",
    "load_raw",
    "resolve",
    "run_dir",
    "write_resolved",
]
