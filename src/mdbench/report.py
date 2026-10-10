"""Ladder plots from ``results/*.parquet``.

Per branch: absolute error against the reference (mean over rows, with the spread as an
error bar) versus wall-clock on a log axis, one marker per (level, method). This is the
accuracy-versus-cost curve whose flattening point is the product of the repo
(``docs/LADDER.md``). Rows without a reference are counted but cannot be plotted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mdbench import REPO_ROOT
from mdbench.schema import read_results, results_dir


def ladder_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows with a reference to one point per (level, method)."""
    d = df[df["reference"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["level", "method", "abs_err", "abs_err_sd", "wall", "n"])
    d["abs_err"] = (d["prediction"] - d["reference"]).abs()
    return (
        d.groupby(["level", "method"])
        .agg(
            abs_err=("abs_err", "mean"),
            abs_err_sd=("abs_err", "std"),
            wall=("wall_clock_s", "median"),
            n=("abs_err", "size"),
        )
        .reset_index()
        .fillna({"abs_err_sd": 0.0})
        .sort_values("level")
    )


def plot_branch(branch: str, out_dir: Path, directory: Path | None = None) -> Path | None:
    """Write ``reports/<branch>_ladder.png``; ``None`` if there is nothing to plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = read_results(branch, directory).to_pandas()
    pts = ladder_frame(df)
    if pts.empty:
        print(f"[report] branch {branch}: {len(df)} rows, none with a reference; no plot")
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for level, g in pts.groupby("level"):
        ax.errorbar(
            g["wall"].clip(lower=1e-3),
            g["abs_err"],
            yerr=g["abs_err_sd"],
            fmt="o",
            capsize=3,
            label=level,
        )
        for _, r in g.iterrows():
            ax.annotate(
                r["method"], (max(r["wall"], 1e-3), r["abs_err"]), fontsize=7, xytext=(4, 2),
                textcoords="offset points",
            )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("wall clock per calculation (s, log)")
    ax.set_ylabel("|prediction − reference|")
    ax.set_title(f"Branch {branch.upper()}: accuracy vs cost")
    ax.legend(title="level", fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{branch}_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def report_all(directory: Path | None = None, out_dir: Path | None = None) -> list[Path]:
    """Plot every ``results/<branch>.parquet``."""
    src = directory or results_dir()
    out = out_dir or (REPO_ROOT / "reports")
    paths = []
    for pq_file in sorted(src.glob("*.parquet")):
        p = plot_branch(pq_file.stem, out, src)
        if p is not None:
            paths.append(p)
    return paths


__all__ = ["ladder_frame", "plot_branch", "report_all"]
