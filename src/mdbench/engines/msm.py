"""Markov state models with deeptime: discretisation, implied timescales, CK test.

Workflow: periodic features (cos/sin of each angle) -> k-means microstates -> reversible
MLE MSM per lag -> implied timescales; a Chapman-Kolmogorov test on user-defined crisp
macrostates (e.g. config-defined Ramachandran basins, rather than PCCA, so the test is
on the same states the populations are reported for); Bayesian MSM for population
uncertainty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def periodic_features(angles_rad: Sequence[np.ndarray]) -> np.ndarray:
    """Stack ``cos`` and ``sin`` of each angle series: ``(n_frames, 2 * n_angles)``."""
    cols = []
    for a in angles_rad:
        a = np.asarray(a, dtype=float)
        cols += [np.cos(a), np.sin(a)]
    return np.column_stack(cols)


def discretize(
    features: Sequence[np.ndarray], n_clusters: int, seed: int, max_iter: int = 200
) -> tuple[list[np.ndarray], np.ndarray]:
    """k-means (k-means++) on the concatenated features; returns dtrajs and centres."""
    from deeptime.clustering import KMeans

    data = np.concatenate([np.asarray(f, dtype=float) for f in features])
    km = KMeans(n_clusters=int(n_clusters), max_iter=max_iter, fixed_seed=int(seed))
    model = km.fit_fetch(data)
    dtrajs = [model.transform(np.asarray(f, dtype=float)).astype(int) for f in features]
    return dtrajs, np.asarray(model.cluster_centers)


def centres_to_angles_deg(centres: np.ndarray) -> np.ndarray:
    """Invert :func:`periodic_features` for cluster centres: ``(n_clusters, n_angles)`` deg."""
    c = np.asarray(centres, dtype=float)
    return np.degrees(np.arctan2(c[:, 1::2], c[:, 0::2]))


def estimate_msm(dtrajs: Sequence[np.ndarray], lag: int, reversible: bool = True):
    """Reversible MLE MSM on the largest connected set (deeptime ``MarkovStateModel``)."""
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    counts = (
        TransitionCountEstimator(lagtime=int(lag), count_mode="sliding")
        .fit_fetch(list(dtrajs))
        .submodel_largest()
    )
    return MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)


def implied_timescales(
    dtrajs: Sequence[np.ndarray], lags: Sequence[int], n_its: int = 3
) -> np.ndarray:
    """Implied timescales in frames, shape ``(len(lags), n_its)`` (NaN where unavailable)."""
    out = np.full((len(lags), n_its), np.nan)
    for i, lag in enumerate(lags):
        m = estimate_msm(dtrajs, lag)
        ts = np.asarray(m.timescales(min(n_its, m.n_states - 1)))
        out[i, : len(ts)] = ts
    return out


def _set_vectors(model, macro_of_micro: np.ndarray, n_macro: int) -> np.ndarray:
    """Indicator matrix (active states x macrostates) for a model's active set."""
    symbols = np.asarray(model.count_model.state_symbols)
    chi = np.zeros((len(symbols), n_macro))
    for i, s in enumerate(symbols):
        m = int(macro_of_micro[s])
        if m >= 0:
            chi[i, m] = 1.0
    return chi


def _coarse_transition(model, chi: np.ndarray, steps: int) -> np.ndarray:
    """``P_IJ = rho_I^T T^steps chi_J`` with ``rho_I`` the stationary dist restricted to I."""
    pi = np.asarray(model.stationary_distribution)
    t = np.linalg.matrix_power(np.asarray(model.transition_matrix), steps)
    out = np.full((chi.shape[1], chi.shape[1]), np.nan)
    for i in range(chi.shape[1]):
        rho = pi * chi[:, i]
        if rho.sum() <= 0:
            continue
        rho = rho / rho.sum()
        out[i] = rho @ t @ chi
    return out


@dataclass
class CKResult:
    """Chapman-Kolmogorov test on crisp macrostates."""

    lag_multiples: list[int]
    predicted: np.ndarray  # (len(multiples), n_macro, n_macro), from T(tau)^k
    estimated: np.ndarray  # same, from MSMs estimated at k*tau
    max_abs_dev: float


def ck_test(
    dtrajs: Sequence[np.ndarray],
    lag: int,
    macro_of_micro: np.ndarray,
    n_macro: int,
    multiples: Sequence[int] = (1, 2, 3, 4, 5),
) -> CKResult:
    """CK test: compare ``T(tau)^k`` to ``T(k tau)`` projected onto the macrostates."""
    base = estimate_msm(dtrajs, lag)
    chi0 = _set_vectors(base, macro_of_micro, n_macro)
    pred, est = [], []
    for k in multiples:
        pred.append(_coarse_transition(base, chi0, int(k)))
        mk = estimate_msm(dtrajs, int(k) * lag)
        est.append(_coarse_transition(mk, _set_vectors(mk, macro_of_micro, n_macro), 1))
    p, e = np.asarray(pred), np.asarray(est)
    return CKResult(
        lag_multiples=[int(k) for k in multiples],
        predicted=p,
        estimated=e,
        max_abs_dev=float(np.nanmax(np.abs(p - e))),
    )


def macro_populations(model, macro_of_micro: np.ndarray, n_macro: int) -> np.ndarray:
    """Stationary probability summed over the microstates of each macrostate."""
    return np.asarray(model.stationary_distribution) @ _set_vectors(model, macro_of_micro, n_macro)


def bayesian_macro_populations(
    dtrajs: Sequence[np.ndarray],
    lag: int,
    macro_of_micro: np.ndarray,
    n_macro: int,
    n_samples: int = 200,
) -> np.ndarray:
    """Posterior samples of macrostate populations, shape ``(n_samples, n_macro)``."""
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import BayesianMSM

    counts = (
        TransitionCountEstimator(lagtime=int(lag), count_mode="effective")
        .fit_fetch(list(dtrajs))
        .submodel_largest()
    )
    post = BayesianMSM(n_samples=int(n_samples), reversible=True).fit_fetch(counts)
    chi = np.zeros((counts.n_states, n_macro))
    for i, s in enumerate(np.asarray(counts.state_symbols)):
        m = int(macro_of_micro[s])
        if m >= 0:
            chi[i, m] = 1.0
    return np.array([np.asarray(s.stationary_distribution) @ chi for s in post.samples])


__all__ = [
    "CKResult",
    "bayesian_macro_populations",
    "centres_to_angles_deg",
    "ck_test",
    "discretize",
    "estimate_msm",
    "implied_timescales",
    "macro_populations",
    "periodic_features",
]
