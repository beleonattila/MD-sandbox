"""mdbench: a method-ladder benchmark for molecular simulation.

``code_version`` identifies the code that produced a results row: the git short hash,
suffixed ``-dirty`` when code-relevant tracked or untracked files differ from HEAD.
Changes under ``results/``, ``reports/`` and ``runs/`` do not count as dirty, because
writing results is what runs do. Outside a git checkout it falls back to the package
version (``pkg-<version>``).
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

__version__ = "0.1.0"

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Paths whose state defines the code that produced a result.
_CODE_PATHS: tuple[str, ...] = (
    "src",
    "configs",
    "systems",
    "data",
    "env",
    "pyproject.toml",
    "Makefile",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


@functools.cache
def get_code_version() -> str:
    """Return ``<git short hash>[-dirty]`` or ``pkg-<version>`` outside git."""
    short = _git("rev-parse", "--short", "HEAD")
    if not short:
        return f"pkg-{__version__}"
    status = _git("status", "--porcelain", "--", *_CODE_PATHS)
    dirty = bool(status)
    return f"{short}-dirty" if dirty else short


code_version: str = get_code_version()

__all__ = ["REPO_ROOT", "__version__", "code_version", "get_code_version"]
