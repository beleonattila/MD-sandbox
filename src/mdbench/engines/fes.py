"""Free energy surfaces, basin populations, block analysis, reweighting, umbrella MBAR.

Conventions: angles in degrees in ``[-180, 180)`` for basins and histograms (that is how
basins are written in configs), energies in kJ/mol internally, ``kT`` in kJ/mol.
Conversions to kcal/mol happen only when a row or a figure is produced.

Basins are a named mapping of rectangles ``[phi_min, phi_max, psi_min, psi_max]``
(half-open on the max side); a basin may be a union of several rectangles. Points in no
basin get label ``-1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

KB_KJ_MOL_K: float = 0.00831446261815324
KJ_PER_KCAL: float = 4.184

Rect = Sequence[float]
Basins = Mapping[str, Sequence[Rect]]


def kT_kj(temperature_K: float) -> float:
    """Thermal energy in kJ/mol."""
    return KB_KJ_MOL_K * float(temperature_K)


def wrap180(x: np.ndarray | float) -> np.ndarray:
    """Wrap degrees into ``[-180, 180)``."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- basins


def validate_basins(basins: Basins) -> list[str]:
    """Check rectangle shapes and that no two basins overlap; returns basin names in order."""
    names = list(basins)
    if not names:
        raise ValueError("no basins defined")
    for name, rects in basins.items():
        if not rects:
            raise ValueError(f"basin {name!r} has no rectangles")
        for r in rects:
            if len(r) != 4 or not (r[0] < r[1] and r[2] < r[3]):
                raise ValueError(f"basin {name!r}: bad rectangle {r} (want phi0<phi1, psi0<psi1)")
    # Overlap check on a fine grid of cell centres.
    g = np.arange(-179.5, 180.0, 1.0)
    phi, psi = np.meshgrid(g, g, indexing="ij")
    hits = np.zeros(phi.shape, dtype=int)
    for rects in basins.values():
        hits += _in_rects(phi, psi, rects).astype(int)
    if np.any(hits > 1):
        raise ValueError("basins overlap")
    return names


def _in_rects(phi: np.ndarray, psi: np.ndarray, rects: Sequence[Rect]) -> np.ndarray:
    m = np.zeros(np.shape(phi), dtype=bool)
    for r in rects:
        m |= (phi >= r[0]) & (phi < r[1]) & (psi >= r[2]) & (psi < r[3])
    return m


def assign_basins(phi_deg: np.ndarray, psi_deg: np.ndarray, basins: Basins) -> np.ndarray:
    """Integer basin label per point (index into ``list(basins)``), ``-1`` if unassigned."""
    phi = wrap180(phi_deg)
    psi = wrap180(psi_deg)
    lab = np.full(phi.shape, -1, dtype=int)
    for i, rects in enumerate(basins.values()):
        lab[_in_rects(phi, psi, rects) & (lab < 0)] = i
    return lab


def populations(labels: np.ndarray, n_basins: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalised (weighted) basin populations; unassigned points count in the total."""
    lab = np.asarray(labels)
    w = np.ones(lab.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    tot = w.sum()
    if tot <= 0:
        raise ValueError("total weight is zero")
    return np.array([w[lab == i].sum() / tot for i in range(n_basins)])


def boltzmann_populations(free_energies_kj: np.ndarray, kT: float) -> np.ndarray:
    """``p_i ∝ exp(-F_i / kT)``; ``inf`` free energy gives zero population."""
    f = np.asarray(free_energies_kj, dtype=float)
    finite = np.isfinite(f)
    out = np.zeros_like(f)
    if not finite.any():
        raise ValueError("no finite free energies")
    z = np.exp(-(f[finite] - f[finite].min()) / kT)
    out[finite] = z / z.sum()
    return out


def delta_g_kj(pops: np.ndarray, ref: int, kT: float) -> np.ndarray:
    """``G_i - G_ref = -kT ln(p_i / p_ref)`` in kJ/mol (``inf`` when ``p_i`` is zero)."""
    p = np.asarray(pops, dtype=float)
    with np.errstate(divide="ignore"):
        return -kT * (np.log(p) - np.log(p[ref]))


def delta_g_err_kj(pops: np.ndarray, pop_err: np.ndarray, ref: int, kT: float) -> np.ndarray:
    """Delta-method error of :func:`delta_g_kj` (covariance neglected; conservative-ish)."""
    p = np.asarray(pops, dtype=float)
    e = np.asarray(pop_err, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (e / p) ** 2 + (e[ref] / p[ref]) ** 2
    out = kT * np.sqrt(rel)
    out[ref] = 0.0
    return out


# --------------------------------------------------------------------------- histograms


def histogram_fes(
    phi_deg: np.ndarray,
    psi_deg: np.ndarray,
    kT: float,
    n_bins: int = 60,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """2D free energy ``F = -kT ln p`` (kJ/mol, min shifted to 0; ``inf`` where empty).

    Returns ``(F[n_bins, n_bins], edges_deg)``; ``F[i, j]`` is phi-bin i, psi-bin j.
    """
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    h, _, _ = np.histogram2d(
        wrap180(phi_deg), wrap180(psi_deg), bins=[edges, edges], weights=weights
    )
    with np.errstate(divide="ignore"):
        f = -kT * np.log(h / h.sum())
    f -= np.min(f[np.isfinite(f)])
    return f, edges


def histogram_fes_1d(
    x_deg: np.ndarray, kT: float, n_bins: int = 72, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """1D ``F(x)`` in kJ/mol, min 0; returns ``(F, bin_centres_deg)``."""
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    h, _ = np.histogram(wrap180(x_deg), bins=edges, weights=weights)
    with np.errstate(divide="ignore"):
        f = -kT * np.log(h / h.sum())
    f -= np.min(f[np.isfinite(f)])
    return f, 0.5 * (edges[1:] + edges[:-1])


# --------------------------------------------------------------------------- blocks


@dataclass
class BlockResult:
    """Block-averaged populations: full-data value, SEM, and the SEM-vs-blocks curve."""

    pops: np.ndarray
    sem: np.ndarray
    n_blocks: list[int]
    sem_curve: np.ndarray  # (len(n_blocks), n_basins)


def block_populations(
    labels: np.ndarray,
    n_basins: int,
    weights: np.ndarray | None = None,
    n_blocks: Sequence[int] = (2, 3, 4, 5, 6, 8, 10, 15, 20),
    min_blocks: int = 4,
) -> BlockResult:
    """Block analysis of (weighted) basin populations along one time series.

    For each block count ``B`` the series is split into ``B`` contiguous blocks, the
    populations are computed per block, and ``SEM_B = std(ddof=1) / sqrt(B)``. The
    reported SEM is the maximum over ``B >= min_blocks`` (the plateau, conservatively):
    small B has noisy SEMs, large B underestimates when blocks are shorter than the
    correlation time, so the max is the honest single number.
    """
    lab = np.asarray(labels)
    w = np.ones(lab.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    pops = populations(lab, n_basins, w)
    counts = [int(b) for b in n_blocks if 2 <= int(b) <= len(lab)]
    curve = np.full((len(counts), n_basins), np.nan)
    for row, b in enumerate(counts):
        idx = np.array_split(np.arange(len(lab)), b)
        bp = []
        for ii in idx:
            if w[ii].sum() > 0:
                bp.append(populations(lab[ii], n_basins, w[ii]))
        if len(bp) >= 2:
            curve[row] = np.std(np.asarray(bp), axis=0, ddof=1) / np.sqrt(len(bp))
    use = [r for r, b in enumerate(counts) if b >= min_blocks and np.all(np.isfinite(curve[r]))]
    if not use:
        use = [r for r in range(len(counts)) if np.all(np.isfinite(curve[r]))]
    sem = np.nanmax(curve[use], axis=0) if use else np.full(n_basins, np.nan)
    return BlockResult(pops=pops, sem=sem, n_blocks=counts, sem_curve=curve)


def running_estimate(
    labels: np.ndarray,
    n_basins: int,
    weights: np.ndarray | None = None,
    n_points: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative populations at ``n_points`` increasing prefix lengths.

    Returns ``(frame_counts, pops[n_points, n_basins])`` — a time-resolved convergence
    diagnostic.
    """
    lab = np.asarray(labels)
    w = np.ones(lab.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    ends = np.unique(np.linspace(max(1, len(lab) // n_points), len(lab), n_points).astype(int))
    out = np.array(
        [
            populations(lab[:e], n_basins, w[:e]) if w[:e].sum() > 0 else np.full(n_basins, np.nan)
            for e in ends
        ]
    )
    return ends, out


# --------------------------------------------------------------------------- metad


def metad_weights(rbias_kj: np.ndarray, kT: float) -> np.ndarray:
    """Normalised reweighting factors ``exp(rbias / kT)`` (``rbias = V(s,t) - c(t)``)."""
    r = np.asarray(rbias_kj, dtype=float)
    w = np.exp((r - r.max()) / kT)
    return w / w.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


# --------------------------------------------------------------------------- umbrella


def umbrella_bias_kj(x_rad: np.ndarray, center_rad: float, k_kj_rad2: float) -> np.ndarray:
    """Harmonic periodic bias ``0.5 k d^2`` with minimum-image ``d``."""
    d = (np.asarray(x_rad, dtype=float) - center_rad + np.pi) % (2.0 * np.pi) - np.pi
    return 0.5 * float(k_kj_rad2) * d**2


@dataclass
class UmbrellaResult:
    """MBAR estimate from an umbrella set: populations, ΔG vs reference, F(x) profile."""

    pops: np.ndarray
    pop_err: np.ndarray
    dg_kj: np.ndarray
    dg_err_kj: np.ndarray
    profile_x_deg: np.ndarray
    profile_kj: np.ndarray
    n_used: list[int]
    g: list[float]
    overlap_min: float


def umbrella_mbar(
    x_by_window: Sequence[np.ndarray],
    labels_by_window: Sequence[np.ndarray],
    centers_rad: Sequence[float],
    k_kj_rad2: float,
    kT: float,
    n_basins: int,
    ref: int,
    profile_bins: int = 72,
    subsample: bool = True,
) -> UmbrellaResult:
    """MBAR (pymbar 4) over umbrella windows biased on a periodic coordinate ``x``.

    The unbiased Hamiltonian is common to all windows, so reduced potentials are bias
    only. Basin populations come from extra zero-sample states that are the unbiased
    state restricted to one basin (``u = 0`` inside, ``+inf`` outside), so ``f_b - f_0 =
    -ln p_b`` and ΔG uncertainties come straight from MBAR's covariance. Each window is
    subsampled by its statistical inefficiency (max over the bias-coordinate deviation
    and the basin indicators) so those asymptotic errors are meaningful.
    """
    import pymbar
    from pymbar import timeseries

    if not (len(x_by_window) == len(labels_by_window) == len(centers_rad)):
        raise ValueError("x, labels and centers must have one entry per window")
    xs, labs, n_used, gs = [], [], [], []
    for x, lab, c in zip(x_by_window, labels_by_window, centers_rad, strict=True):
        x = np.asarray(x, dtype=float)
        lab = np.asarray(lab)
        g = 1.0
        if subsample and len(x) > 10:
            series = [(x - c + np.pi) % (2 * np.pi) - np.pi]
            series += [(lab == b).astype(float) for b in range(n_basins)]
            for s in series:
                if np.ptp(s) > 0:
                    g = max(g, float(timeseries.statistical_inefficiency(s)))
            idx = np.unique(np.round(np.arange(0, len(x), g)).astype(int))
            idx = idx[idx < len(x)]
        else:
            idx = np.arange(len(x))
        xs.append(x[idx])
        labs.append(lab[idx])
        n_used.append(len(idx))
        gs.append(g)
    x_all = np.concatenate(xs)
    lab_all = np.concatenate(labs)
    n_win = len(centers_rad)
    # Only basins that were sampled get a restricted state (an empty one has p = 0 and an
    # undefined free energy); restricted states use a large finite penalty, not inf.
    present = [b for b in range(n_basins) if np.any(lab_all == b)]
    if ref not in present:
        raise ValueError("reference basin was never sampled")
    n_states = n_win + 1 + len(present)  # windows, unbiased, unbiased-restricted-to-basin
    u = np.zeros((n_states, len(x_all)))
    for k, c in enumerate(centers_rad):
        u[k] = umbrella_bias_kj(x_all, c, k_kj_rad2) / kT
    for j, b in enumerate(present):
        u[n_win + 1 + j] = np.where(lab_all == b, 0.0, 1.0e6)
    n_k = np.array([*n_used, 0, *([0] * len(present))])
    mbar = pymbar.MBAR(u, n_k, solver_protocol="robust")
    res = mbar.compute_free_energy_differences(compute_uncertainty=True)
    df, ddf = np.asarray(res["Delta_f"]), np.asarray(res["dDelta_f"])
    i0 = n_win
    col = {b: n_win + 1 + j for j, b in enumerate(present)}
    pops = np.zeros(n_basins)
    pop_err = np.zeros(n_basins)
    dg = np.full(n_basins, np.inf)
    dg_err = np.full(n_basins, np.nan)
    for b, c in col.items():
        pops[b] = np.exp(-df[i0, c])
        pop_err[b] = pops[b] * ddf[i0, c]
        dg[b] = kT * df[col[ref], c]
        dg_err[b] = kT * ddf[col[ref], c]
    dg_err[ref] = 0.0
    w0 = mbar.weights()[:, i0]
    prof, centres = histogram_fes_1d(np.degrees(x_all), kT, profile_bins, weights=w0)
    ov = mbar.compute_overlap()["matrix"][:n_win, :n_win]
    adj = [ov[k, (k + 1) % n_win] for k in range(n_win)]
    return UmbrellaResult(
        pops=pops,
        pop_err=pop_err,
        dg_kj=dg,
        dg_err_kj=dg_err,
        profile_x_deg=centres,
        profile_kj=prof,
        n_used=n_used,
        g=gs,
        overlap_min=float(np.min(adj)),
    )


def as_kcal(x_kj: Any) -> Any:
    """kJ/mol -> kcal/mol (arrays or scalars)."""
    return np.asarray(x_kj, dtype=float) / KJ_PER_KCAL


__all__ = [
    "KB_KJ_MOL_K",
    "KJ_PER_KCAL",
    "BlockResult",
    "UmbrellaResult",
    "as_kcal",
    "assign_basins",
    "block_populations",
    "boltzmann_populations",
    "delta_g_err_kj",
    "delta_g_kj",
    "effective_sample_size",
    "histogram_fes",
    "histogram_fes_1d",
    "kT_kj",
    "metad_weights",
    "populations",
    "running_estimate",
    "umbrella_bias_kj",
    "umbrella_mbar",
    "validate_basins",
    "wrap180",
]
