"""Registry that branch modules plug their methods into.

A branch module (``src/mdbench/branch_<x>/<system>.py``) registers one function per
(system, level, method)::

    from mdbench.registry import register

    @register(branch="c", system="ala2", level="L0", method="boltzmann_endpoints")
    def ala2_l0(cfg: dict, run_dir: Path) -> list[dict]:
        ...
        return [{"prediction": p, "uncertainty": u, "reference": None,
                 "reference_source": None, "gpu": "none", "wall_clock_s": t}]

The function receives the resolved config and its run directory (already created, with
``config.resolved.yml`` written) and returns result rows. The CLI fills the provenance
columns it owns (system, branch, level, method, replicate, code_version, config_hash) and
defaults ``gpu`` to ``"none"`` and ``wall_clock_s`` to the measured call time when the
function leaves them out; ``prediction`` and ``uncertainty`` are never defaulted.

``python -m mdbench run`` imports every module in ``mdbench.branch_<branch>`` before
dispatching, so registration is just "put the file in the package".
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

RunFn = Callable[[dict[str, Any], Path], Iterable[Mapping[str, Any]]]
ScoreFn = Callable[..., Any]
Key = tuple[str, str, str, str]

_RUNNERS: dict[Key, RunFn] = {}
_SCORERS: dict[str, ScoreFn] = {}


def _key(branch: str, system: str, level: str, method: str) -> Key:
    return (str(branch).lower(), str(system), str(level).upper(), str(method))


def register(branch: str, system: str, level: str, method: str) -> Callable[[RunFn], RunFn]:
    """Decorator registering ``fn`` as the runner for (branch, system, level, method)."""

    def deco(fn: RunFn) -> RunFn:
        key = _key(branch, system, level, method)
        existing = _RUNNERS.get(key)
        if existing is not None and existing is not fn:
            same = (existing.__module__, existing.__qualname__) == (fn.__module__, fn.__qualname__)
            if not same:
                raise ValueError(f"duplicate registration for {key}: {existing} vs {fn}")
        _RUNNERS[key] = fn
        return fn

    return deco


def register_scorer(branch: str) -> Callable[[ScoreFn], ScoreFn]:
    """Decorator registering a branch-specific scorer used by ``mdbench score <branch>``.

    Signature: ``fn(table: pandas.DataFrame, n_boot: int, seed: int) -> pandas.DataFrame``.
    """

    def deco(fn: ScoreFn) -> ScoreFn:
        _SCORERS[str(branch).lower()] = fn
        return fn

    return deco


def load_branch(branch: str) -> None:
    """Import ``mdbench.branch_<branch>`` and all of its submodules (if it exists)."""
    name = f"mdbench.branch_{str(branch).lower()}"
    try:
        pkg = importlib.import_module(name)
    except ModuleNotFoundError as err:
        if err.name == name:
            return
        raise
    for info in pkgutil.walk_packages(getattr(pkg, "__path__", []), prefix=f"{name}."):
        importlib.import_module(info.name)


def get_runner(branch: str, system: str, level: str, method: str) -> RunFn:
    """Look up the runner, importing the branch package first."""
    load_branch(branch)
    key = _key(branch, system, level, method)
    try:
        return _RUNNERS[key]
    except KeyError:
        known = sorted(k for k in _RUNNERS if k[0] == key[0])
        raise KeyError(f"no runner registered for {key}; registered for branch: {known}") from None


def get_scorer(branch: str) -> ScoreFn | None:
    """Branch-specific scorer, or ``None`` to use the generic one."""
    load_branch(branch)
    return _SCORERS.get(str(branch).lower())


def registered() -> list[Key]:
    """All registered runner keys (for listing)."""
    return sorted(_RUNNERS)


__all__ = [
    "get_runner",
    "get_scorer",
    "load_branch",
    "register",
    "register_scorer",
    "registered",
]
