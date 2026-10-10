"""The results row schema and the only sanctioned writer of ``results/<branch>.parquet``.

One tidy row per calculation. Columns, in order::

    system, branch, level, method, replicate, prediction, uncertainty, reference,
    reference_source, wall_clock_s, gpu, code_version, config_hash

``reference`` and ``reference_source`` may be null (unverified ground truth must stay
null rather than be guessed). Every other column is required, and ``uncertainty`` must be
a finite, non-negative number: a prediction without an error bar is not a result.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mdbench import REPO_ROOT

SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("system", pa.string(), nullable=False),
        pa.field("branch", pa.string(), nullable=False),
        pa.field("level", pa.string(), nullable=False),
        pa.field("method", pa.string(), nullable=False),
        pa.field("replicate", pa.int64(), nullable=False),
        pa.field("prediction", pa.float64(), nullable=False),
        pa.field("uncertainty", pa.float64(), nullable=False),
        pa.field("reference", pa.float64(), nullable=True),
        pa.field("reference_source", pa.string(), nullable=True),
        pa.field("wall_clock_s", pa.float64(), nullable=False),
        pa.field("gpu", pa.string(), nullable=False),
        pa.field("code_version", pa.string(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
    ]
)

COLUMNS: tuple[str, ...] = tuple(SCHEMA.names)
NULLABLE: frozenset[str] = frozenset(f.name for f in SCHEMA if f.nullable)

_LEVEL_RE = re.compile(r"^L[0-4]$")
_BRANCH_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SchemaError(ValueError):
    """A row does not match the results schema."""


def results_dir() -> Path:
    """Directory holding ``<branch>.parquet``; ``$MDBENCH_RESULTS_DIR`` overrides."""
    env = os.environ.get("MDBENCH_RESULTS_DIR")
    return Path(env) if env else REPO_ROOT / "results"


def results_path(branch: str, directory: Path | None = None) -> Path:
    """Path of the parquet file for ``branch``."""
    return (directory or results_dir()) / f"{branch}.parquet"


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_row(row: Mapping[str, Any], branch: str) -> dict[str, Any]:
    """Check one row against the schema and return a normalised copy.

    Raises :class:`SchemaError` on missing or extra columns, nulls in required columns,
    a non-finite or negative uncertainty, a malformed level, or a branch mismatch.
    """
    keys = set(row)
    missing = [c for c in COLUMNS if c not in keys]
    extra = sorted(keys - set(COLUMNS))
    if missing or extra:
        raise SchemaError(f"wrong columns: missing={missing} extra={extra}")

    out: dict[str, Any] = {}
    for col in COLUMNS:
        val = row[col]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            if col in NULLABLE:
                out[col] = None
                continue
            raise SchemaError(f"column {col!r} is required and may not be null/NaN")
        out[col] = val

    if out["branch"] != branch:
        raise SchemaError(f"row branch {out['branch']!r} does not match file branch {branch!r}")
    if not _LEVEL_RE.match(str(out["level"])):
        raise SchemaError(f"level must be L0..L4, got {out['level']!r}")
    for col in ("prediction", "uncertainty", "wall_clock_s"):
        if not _is_finite_number(out[col]):
            raise SchemaError(f"column {col!r} must be a finite number, got {out[col]!r}")
        out[col] = float(out[col])
    if out["uncertainty"] < 0:
        raise SchemaError(f"uncertainty must be >= 0, got {out['uncertainty']}")
    if out["wall_clock_s"] < 0:
        raise SchemaError(f"wall_clock_s must be >= 0, got {out['wall_clock_s']}")
    if out["reference"] is not None:
        if not _is_finite_number(out["reference"]):
            raise SchemaError(
                f"reference must be a finite number or null, got {out['reference']!r}"
            )
        out["reference"] = float(out["reference"])
        if not out["reference_source"]:
            raise SchemaError("a non-null reference requires a reference_source (DOI)")
    if isinstance(out["replicate"], bool) or int(out["replicate"]) != out["replicate"]:
        raise SchemaError(f"replicate must be an integer, got {out['replicate']!r}")
    out["replicate"] = int(out["replicate"])
    for col in ("system", "branch", "level", "method", "gpu", "code_version", "config_hash"):
        if not isinstance(out[col], str) or not out[col]:
            raise SchemaError(f"column {col!r} must be a non-empty string, got {out[col]!r}")
    return out


def rows_to_table(rows: Iterable[Mapping[str, Any]], branch: str) -> pa.Table:
    """Validate ``rows`` and build a table with exactly :data:`SCHEMA`."""
    clean = [validate_row(r, branch) for r in rows]
    return pa.Table.from_pylist(clean, schema=SCHEMA)


def read_results(branch: str, directory: Path | None = None) -> pa.Table:
    """Read ``results/<branch>.parquet``; an empty table if it does not exist."""
    path = results_path(branch, directory)
    if not path.exists():
        return SCHEMA.empty_table()
    table = pq.read_table(path)
    if table.schema.names != list(COLUMNS):
        raise SchemaError(f"{path} has columns {table.schema.names}, expected {list(COLUMNS)}")
    return table.cast(SCHEMA)


def append_rows(
    branch: str, rows: Iterable[Mapping[str, Any]], directory: Path | None = None
) -> Path:
    """Validate ``rows`` and append them to ``results/<branch>.parquet``.

    Existing rows are kept (append, never overwrite). The file is replaced atomically so
    an interrupted write cannot truncate the product. Returns the parquet path.
    """
    if not _BRANCH_RE.match(branch):
        raise SchemaError(f"invalid branch name {branch!r}")
    new = rows_to_table(rows, branch)
    path = results_path(branch, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    if new.num_rows == 0:
        return path
    old = read_results(branch, directory)
    table = pa.concat_tables([old, new]) if old.num_rows else new
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


__all__ = [
    "COLUMNS",
    "NULLABLE",
    "SCHEMA",
    "SchemaError",
    "append_rows",
    "read_results",
    "results_dir",
    "results_path",
    "rows_to_table",
    "validate_row",
]
