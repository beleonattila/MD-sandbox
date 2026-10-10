"""Generic scoring of a branch's results against reference values.

For each (system, level, method) and for each (level, method) across systems, rows with a
non-null ``reference`` are scored. Metrics (``docs/GROUND_TRUTH.md``): Spearman ρ,
pairwise ranking accuracy, RMSE and MUE, each with a percentile bootstrap CI. Spearman
and pairwise accuracy need at least two reference points with distinct values.

A branch can override this by registering a scorer with
:func:`mdbench.registry.register_scorer`.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mdbench import REPO_ROOT
from mdbench.registry import get_scorer
from mdbench.schema import read_results


def pairwise_accuracy(pred: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of pairs (with distinct reference) ordered the same way as the reference."""
    good = total = 0
    for i, j in combinations(range(len(ref)), 2):
        dr = ref[i] - ref[j]
        if dr == 0:
            continue
        total += 1
        good += int(np.sign(pred[i] - pred[j]) == np.sign(dr))
    return good / total if total else float("nan")


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    err = pred - ref
    out = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mue": float(np.mean(np.abs(err))),
        "spearman": float("nan"),
        "pairwise": float("nan"),
    }
    if len(ref) >= 2 and np.ptp(ref) > 0 and np.ptp(pred) > 0:
        out["spearman"] = float(spearmanr(pred, ref).statistic)
        out["pairwise"] = pairwise_accuracy(pred, ref)
    return out


def score_frame(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Score a results DataFrame; one output row per group with point values and 95% CIs."""
    rng = np.random.default_rng(seed)
    df = df[df["reference"].notna()]
    records = []
    groupings = {
        "per_system": ["system", "level", "method"],
        "across_systems": ["level", "method"],
    }
    for scope, keys in groupings.items():
        for gkey, g in df.groupby(keys, sort=True):
            if scope == "per_system":
                pred, ref = g["prediction"].to_numpy(float), g["reference"].to_numpy(float)
            else:
                # One point per system: mean over replicates.
                agg = g.groupby("system").agg(
                    prediction=("prediction", "mean"), reference=("reference", "first")
                )
                pred, ref = agg["prediction"].to_numpy(float), agg["reference"].to_numpy(float)
            point = _metrics(pred, ref)
            boots: dict[str, list[float]] = {k: [] for k in point}
            n = len(pred)
            for _ in range(n_boot if n >= 2 else 0):
                idx = rng.integers(0, n, n)
                for k, v in _metrics(pred[idx], ref[idx]).items():
                    boots[k].append(v)
            rec: dict[str, object] = dict(zip(keys, np.atleast_1d(gkey), strict=True))
            rec.update(scope=scope, n=n, wall_clock_s=float(g["wall_clock_s"].sum()))
            for k, v in point.items():
                b = np.asarray(boots[k], float)
                b = b[np.isfinite(b)]
                rec[k] = v
                rec[f"{k}_lo"] = float(np.percentile(b, 2.5)) if b.size else float("nan")
                rec[f"{k}_hi"] = float(np.percentile(b, 97.5)) if b.size else float("nan")
            records.append(rec)
    return pd.DataFrame.from_records(records)


def score_branch(
    branch: str,
    n_boot: int = 1000,
    seed: int = 0,
    results_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Score ``results/<branch>.parquet`` and write ``reports/<branch>_scores.csv``."""
    df = read_results(branch, results_dir).to_pandas()
    scorer = get_scorer(branch) or score_frame
    scores = scorer(df, n_boot=n_boot, seed=seed)
    out_dir = reports_dir or (REPO_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{branch}_scores.csv"
    scores.to_csv(path, index=False)
    return scores, path


__all__ = ["pairwise_accuracy", "score_branch", "score_frame"]
