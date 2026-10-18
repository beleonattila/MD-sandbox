"""Generic scorer: bootstrap CIs are stable under unrelated rows and paired across methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdbench.score import score_frame


def _rows(method: str, preds, refs, systems):
    base = {"branch": "a", "level": "L1", "replicate": 0, "uncertainty": 0.5,
            "reference_source": "x", "wall_clock_s": 1.0, "gpu": "none",
            "code_version": "t", "config_hash": "h"}  # fmt: skip
    return [{**base, "method": method, "system": s, "prediction": p, "reference": r}
            for s, p, r in zip(systems, preds, refs, strict=True)]  # fmt: skip


def _ci(df, method):
    sc = score_frame(df, n_boot=200, seed=0)
    r = sc[(sc["scope"] == "across_systems") & (sc["method"] == method)].iloc[0]
    return r["spearman_lo"], r["spearman_hi"], r["rmse_lo"], r["rmse_hi"]


def test_ci_does_not_move_when_unrelated_rows_are_added():
    rng = np.random.default_rng(3)
    systems = [f"s{i}" for i in range(8)]
    ref = rng.normal(-8, 2, 8)
    a = pd.DataFrame(_rows("m_a", ref + rng.normal(0, 1, 8), ref, systems))
    other = pd.DataFrame(_rows("m_b", ref + rng.normal(0, 3, 8), ref, systems))
    extra = pd.DataFrame(_rows("m_c", [1.0, 2.0], [-5.0, -6.0], ["x1", "x2"]))
    alone = _ci(a, "m_a")
    assert _ci(pd.concat([other, a, extra], ignore_index=True), "m_a") == alone


def test_methods_on_the_same_systems_are_resampled_identically():
    systems = [f"s{i}" for i in range(6)]
    ref = [-4.0, -5.0, -6.0, -7.0, -8.0, -9.0]
    df = pd.DataFrame(_rows("m_a", ref, ref, systems) + _rows("m_b", ref, ref, systems))
    assert _ci(df, "m_a") == _ci(df, "m_b")
