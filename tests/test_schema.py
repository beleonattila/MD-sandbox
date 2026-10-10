from __future__ import annotations

import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mdbench.schema import COLUMNS, SCHEMA, SchemaError, append_rows, read_results


def _row(**kw):
    row = {
        "system": "ala2",
        "branch": "c",
        "level": "L0",
        "method": "null",
        "replicate": 0,
        "prediction": 1.5,
        "uncertainty": 0.2,
        "reference": None,
        "reference_source": None,
        "wall_clock_s": 0.01,
        "gpu": "none",
        "code_version": "abc1234",
        "config_hash": "0" * 64,
    }
    row.update(kw)
    return row


def test_columns_exact():
    assert COLUMNS == (
        "system", "branch", "level", "method", "replicate", "prediction", "uncertainty",
        "reference", "reference_source", "wall_clock_s", "gpu", "code_version", "config_hash",
    )  # fmt: skip


def test_round_trip_and_append(isolated_results: Path):
    p = append_rows("c", [_row()])
    append_rows("c", [_row(replicate=1, reference=2.0, reference_source="doi:10.x/test")])
    assert p == isolated_results / "c.parquet"
    t = pq.read_table(p)
    assert t.schema.equals(SCHEMA, check_metadata=False)
    assert t.num_rows == 2
    df = read_results("c").to_pandas()
    assert list(df["replicate"]) == [0, 1]
    assert math.isnan(df["reference"].iloc[0]) or df["reference"].iloc[0] is None
    assert df["reference"].iloc[1] == 2.0


def test_append_rejects_duplicate_calculation(isolated_results: Path):
    append_rows("c", [_row()])
    with pytest.raises(SchemaError, match="already in"):
        append_rows("c", [_row(prediction=9.9, wall_clock_s=0.001)])
    append_rows("c", [_row(code_version="def5678")])
    assert read_results("c").num_rows == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"uncertainty": None},
        {"uncertainty": float("nan")},
        {"uncertainty": -0.1},
        {"prediction": float("inf")},
        {"level": "L7"},
        {"branch": "a"},
        {"gpu": ""},
        {"reference": 1.0, "reference_source": None},
    ],
)
def test_rejects_bad_values(isolated_results: Path, bad):
    with pytest.raises(SchemaError):
        append_rows("c", [_row(**bad)])
    assert not (isolated_results / "c.parquet").exists()


def test_rejects_wrong_columns(isolated_results: Path):
    r = _row()
    del r["uncertainty"]
    with pytest.raises(SchemaError, match="missing"):
        append_rows("c", [r])
    with pytest.raises(SchemaError, match="extra"):
        append_rows("c", [_row(notes="x")])


def test_bad_batch_writes_nothing(isolated_results: Path):
    append_rows("c", [_row()])
    with pytest.raises(SchemaError):
        append_rows("c", [_row(replicate=1), _row(uncertainty=None)])
    assert read_results("c").num_rows == 1
