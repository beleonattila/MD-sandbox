"""Branch A shared pieces: reference rows, the heavy-atom-count null, the ranking scorer.

Every prediction is a standard binding free energy in kcal/mol, one row per guest
(``system`` = guest id, e.g. ``cb7_s1_g23``). A *series* is the set of guests scored
together: the host (``cb7``) and each benchmark set within it (``cb7_s1``, ``cb7_s2``).

Scoring (docs/LADDER.md, docs/GROUND_TRUTH.md): Spearman rho first, pairwise ranking
accuracy second, RMSE and MUE third, each with a bootstrap 95% CI over guests, and the
L0 null's value beside every level.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mdbench.registry import register_plotter, register_scorer

KCAL_PER_KJ = 1.0 / 4.184
KB_KCAL_MOL_K = 0.0019872043  # Boltzmann constant, kcal/(mol K)
NULL_METHOD = "null_heavy_atom_count"


def entries(gt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Per-ligand entries of a ground-truth YAML (``guests`` for CB7, ``ligands`` for T4L)."""
    return gt["guests"] if "guests" in gt else gt["ligands"]


def reference_fields(gt: Mapping[str, Any], gid: str) -> dict[str, Any]:
    """``reference`` / ``reference_source`` for one ligand from the ground-truth YAML."""
    g = entries(gt)[gid]
    primary = gt["primary_sources"][g["source"]]["doi"]
    where = (f"CB7 Set {g['set']} table, ID {g['id']}" if "set" in g
             else f"Lysozyme L99A Set table, {g['name']}")  # fmt: skip
    return {
        "reference": float(g["dG_kcal_mol"]),
        "reference_source": f"doi:{gt['doi']} ({where}); primary doi:{primary}",
    }


def heavy_atom_count(smiles: str) -> int:
    """Heavy atoms of a SMILES string (RDKit)."""
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()


def loo_linear(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out least-squares line y = a + b x: (prediction, predictive sd) per point.

    For point i the line is fit to the others; the sd is the prediction-interval sd of
    that fit at x_i, ``s * sqrt(1 + 1/m + (x_i - mean)^2 / Sxx)`` with ``s`` the residual
    sd on m - 2 degrees of freedom. Needs at least four points.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 4:
        raise ValueError("leave-one-out regression needs at least 4 points")
    pred, sd = np.empty(n), np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        xs, ys = x[m], y[m]
        b, a = np.polyfit(xs, ys, 1)
        r = ys - (a + b * xs)
        s = math.sqrt(float(np.sum(r**2)) / (len(xs) - 2))
        sxx = float(np.sum((xs - xs.mean()) ** 2))
        pred[i] = a + b * x[i]
        sd[i] = s * math.sqrt(1 + 1 / len(xs) + (x[i] - xs.mean()) ** 2 / sxx) if sxx > 0 else s
    return pred, sd


def series_of(system: str) -> list[str]:
    """Series a ligand id belongs to: ``cb7_s1_g23`` -> ``['cb7', 'cb7_s1']``,
    ``t4l_p_xylene`` -> ``['t4l']``."""
    m = re.match(r"^(cb7)_(s\d+)_", system)
    if m:
        return [m.group(1), f"{m.group(1)}_{m.group(2)}"]
    return [system.split("_")[0]]


@register_scorer("a")
def score_a(df: Any, n_boot: int = 1000, seed: int = 0) -> Any:
    """Per (series, level, method): ranking and error metrics with CIs, null alongside."""
    import pandas as pd

    from mdbench.score import score_frame

    d = df[df["reference"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    out = []
    series = sorted({s for sys_ in d["system"] for s in series_of(sys_)})
    for ser in series:
        sub = d[d["system"].map(lambda s, ser=ser: ser in series_of(s))]
        sc = score_frame(sub, n_boot=n_boot, seed=seed)
        sc = sc[sc["scope"] == "across_systems"].drop(columns=["scope"]).copy()
        sc.insert(0, "series", ser)
        per_guest = sub.groupby(["level", "method", "system"])["wall_clock_s"].sum()
        sc["wall_clock_s_per_guest"] = [
            float(per_guest.loc[(lv, me)].median())
            for lv, me in zip(sc["level"], sc["method"], strict=True)
        ]
        null = sc[sc["method"] == NULL_METHOD]
        for k in ("spearman", "pairwise", "rmse"):
            sc[f"null_{k}"] = float(null[k].iloc[0]) if len(null) else float("nan")
        out.append(sc)
    return pd.concat(out, ignore_index=True)


@register_plotter("a")
def plot_a(df: Any, out_dir: Any) -> Any:
    """Spearman rho (95% CI) against median wall clock per ligand, one panel per series.

    The null (L0) is drawn as a horizontal band at its rho; the flattening point of each
    curve is the level above which rho stops rising (docs/LADDER.md).
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = score_a(df, n_boot=1000, seed=0)
    if sc.empty:
        return None
    hosts = [s for s in sorted(sc["series"].unique()) if "_" not in s]
    fig, axes = plt.subplots(1, len(hosts), figsize=(5.2 * len(hosts), 4.0), squeeze=False)
    for ax, ser in zip(axes[0], hosts, strict=True):
        g = sc[sc["series"] == ser].sort_values("level")
        null = g[g["method"] == NULL_METHOD]
        if len(null):
            n = null.iloc[0]
            ax.axhspan(n["spearman_lo"], n["spearman_hi"], color="0.85", label="L0 null 95% CI")
        for _, r in g.iterrows():
            x = max(float(r["wall_clock_s_per_guest"]), 1e-2)
            lo, hi = r["spearman"] - r["spearman_lo"], r["spearman_hi"] - r["spearman"]
            ax.errorbar(x, r["spearman"], yerr=[[lo], [hi]], fmt="o", capsize=3)
            ax.annotate(f"{r['level']} {r['method']}\nRMSE {r['rmse']:.1f}", (x, r["spearman"]),
                        fontsize=7, xytext=(4, 2), textcoords="offset points")  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("wall clock per ligand (s, log)")
        ax.set_ylabel("Spearman rho vs experiment")
        ax.set_title(f"{ser.upper()} (n = {int(g['n'].max())})")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = Path(out_dir) / "a_ranking_vs_cost.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


__all__ = [
    "KB_KCAL_MOL_K",
    "KCAL_PER_KJ",
    "NULL_METHOD",
    "entries",
    "heavy_atom_count",
    "loo_linear",
    "plot_a",
    "reference_fields",
    "score_a",
    "series_of",
]
