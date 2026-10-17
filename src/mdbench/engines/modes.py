"""Collective-motion analysis: superposition, ANM, normal modes, PCA, and overlap scores.

Everything here is plain numpy on Cartesian coordinates of shape ``(n_atoms, 3)``. A
"mode" is a flattened ``3 * n_atoms`` vector; a set of modes is a ``(3n, k)`` matrix with
one mode per column.

Scores against an observed displacement ``d`` (e.g. closed minus open crystal, after
superposition), following Tama & Sanejouand, Protein Eng. 14:1-6 (2001),
doi:10.1093/protein/14.1.1:

- :func:`overlaps`: ``|v_k . d| / (|v_k| |d|)`` for every mode ``v_k``.
- :func:`cumulative_overlap`: ``|P_K d| / |d|``, the length of the projection of ``d`` onto
  the span of the first ``K`` modes. For orthonormal modes this is the usual
  ``sqrt(sum_k overlap_k^2)``; defined as a projection it stays in [0, 1] when the modes
  are not orthogonal (all-atom modes restricted to C-alpha atoms are not).

ANM (anisotropic network model): Atilgan et al., Biophys. J. 80:505-515 (2001),
doi:10.1016/S0006-3495(01)76033-X. Uniform springs of stiffness ``gamma`` between every
pair of nodes closer than ``cutoff``; the Hessian has exactly six zero modes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# 1 / (2 pi c) with c in cm/ps: converts sqrt(eigenvalue) in rad/ps to wavenumber in cm^-1.
RAD_PER_PS_TO_CM1 = 1.0 / (2.0 * np.pi * 2.99792458e-2)


# --------------------------------------------------------------------------- geometry


def kabsch(
    mobile: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal rotation of ``mobile`` onto ``target``: returns (R, mobile_centre, target_centre).

    The superposed coordinates are ``(mobile - mobile_centre) @ R.T + target_centre``.
    """
    p = np.asarray(mobile, float)
    q = np.asarray(target, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    pc, qc = w @ p, w @ q
    h = ((p - pc) * w[:, None]).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, pc, qc


def superpose(
    mobile: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """``mobile`` rotated and translated onto ``target`` (least squares)."""
    r, pc, qc = kabsch(mobile, target, weights)
    return (np.asarray(mobile, float) - pc) @ r.T + qc


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square deviation between two coordinate sets as given (no fitting)."""
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


def displacement(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Flattened ``end - start`` after superposing ``end`` onto ``start``."""
    return (superpose(end, start) - np.asarray(start, float)).ravel()


# --------------------------------------------------------------------------- Hessians


def anm_hessian(xyz: np.ndarray, cutoff: float, gamma: float = 1.0) -> np.ndarray:
    """ANM Hessian (3n, 3n) for nodes at ``xyz``; ``cutoff`` in the units of ``xyz``."""
    x = np.asarray(xyz, float)
    n = len(x)
    d = x[None, :, :] - x[:, None, :]
    r2 = np.sum(d**2, axis=-1)
    contact = (r2 < cutoff**2) & ~np.eye(n, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        blocks = -gamma * d[:, :, :, None] * d[:, :, None, :] / r2[:, :, None, None]
    blocks[~contact] = 0.0
    idx = np.arange(n)
    blocks[idx, idx] = -blocks.sum(axis=1)
    return blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


def rigid_body_basis(xyz: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal (3n, 6) basis of translations and rotations, mass-weighted if ``masses``.

    In mass-weighted coordinates ``q = sqrt(m) x`` the rigid-body vectors are
    ``sqrt(m_i) e_a`` and ``sqrt(m_i) (e_a x (r_i - r_com))``.
    """
    x = np.asarray(xyz, float)
    m = np.ones(len(x)) if masses is None else np.asarray(masses, float)
    sm = np.sqrt(m)
    r = x - (m @ x) / m.sum()
    cols = []
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        cols.append((sm[:, None] * e).ravel())
        cols.append((sm[:, None] * np.cross(e, r)).ravel())
    q, _ = np.linalg.qr(np.stack(cols, axis=1))
    return q


def internal_modes(
    hessian: np.ndarray, xyz: np.ndarray, masses: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of the (mass-weighted) Hessian with rigid body removed.

    With ``masses`` the Hessian is mass-weighted, ``H_mw = M^-1/2 H M^-1/2``. Rigid-body
    motion is projected out (``P H P``) and the six resulting null vectors are dropped,
    so the returned ``3n - 6`` modes are internal motions in ascending eigenvalue order;
    any negative eigenvalues (an imperfect minimum) are kept and come first. The
    eigenvectors are orthonormal in the (mass-weighted) space they were computed in; use
    :func:`cartesian_modes` to turn them into Cartesian displacement directions.
    """
    h = np.asarray(hessian, float)
    if masses is not None:
        inv = 1.0 / np.sqrt(np.repeat(np.asarray(masses, float), 3))
        h = h * inv[:, None] * inv[None, :]
    h = 0.5 * (h + h.T)
    rb = rigid_body_basis(xyz, masses)
    # P H P with P = I - B B^T, expanded so no dense (3n, 3n) projector is formed.
    hb = h @ rb
    hp = h - rb @ hb.T - hb @ rb.T + rb @ (rb.T @ hb) @ rb.T
    vals, vecs = np.linalg.eigh(0.5 * (hp + hp.T))
    # The six rigid-body vectors are exact null vectors of P H P; drop the six eigenvectors
    # with the largest rigid-body content rather than the six smallest eigenvalues, so a
    # negative internal eigenvalue is never mistaken for a rigid-body mode.
    rb_content = np.sum((rb.T @ vecs) ** 2, axis=0)
    keep = np.sort(np.argsort(rb_content)[: len(h) - 6])
    return vals[keep], vecs[:, keep]


def cartesian_modes(vecs: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Cartesian displacement directions ``M^-1/2 q`` (columns), each normalised to 1."""
    v = np.asarray(vecs, float)
    if masses is not None:
        v = v / np.sqrt(np.repeat(np.asarray(masses, float), 3))[:, None]
    return v / np.linalg.norm(v, axis=0, keepdims=True)


def restrict(modes: np.ndarray, atoms: Sequence[int]) -> np.ndarray:
    """Rows of ``modes`` (3n, k) belonging to ``atoms``; columns are not renormalised."""
    idx = np.ravel([[3 * i, 3 * i + 1, 3 * i + 2] for i in atoms])
    return np.asarray(modes)[idx]


def msf_from_modes(
    vals: np.ndarray, vecs: np.ndarray, masses: np.ndarray | None = None, kT: float = 1.0
) -> np.ndarray:
    """Per-atom mean-square fluctuation ``kT / m_i * sum_k |q_ik|^2 / lambda_k``.

    ``vals``/``vecs`` as returned by :func:`internal_modes`. Only positive eigenvalues
    contribute (a negative one has no harmonic fluctuation). Units: kT over the Hessian's
    units (times 1/mass when mass-weighted), i.e. length squared.
    """
    v = np.asarray(vecs, float)
    lam = np.asarray(vals, float)
    pos = lam > 0
    contrib = (v[:, pos] ** 2) / lam[pos]
    per_dof = kT * contrib.sum(axis=1)
    if masses is not None:
        per_dof = per_dof / np.repeat(np.asarray(masses, float), 3)
    return per_dof.reshape(-1, 3).sum(axis=1)


def wavenumbers_cm1(vals_mw: np.ndarray) -> np.ndarray:
    """Wavenumbers (cm^-1) from mass-weighted eigenvalues in kJ/(mol nm^2 amu) = ps^-2.

    Negative eigenvalues are returned as negative wavenumbers (imaginary frequencies).
    """
    v = np.asarray(vals_mw, float)
    return np.sign(v) * np.sqrt(np.abs(v)) * RAD_PER_PS_TO_CM1


# --------------------------------------------------------------------------- PCA


def pca(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of ``frames`` (n_frames, n_atoms, 3), already superposed.

    Returns (variances descending, orthonormal PCs as columns (3n, 3n), per-atom MSF).
    """
    x = np.asarray(frames, float).reshape(len(frames), -1)
    x = x - x.mean(axis=0)
    cov = x.T @ x / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    msf = np.diag(cov).reshape(-1, 3).sum(axis=1)
    return vals[order], vecs[:, order], msf


def iterative_superpose(frames: np.ndarray, reference: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """Superpose every frame onto ``reference``, then onto the running mean ``n_iter`` times."""
    f = np.asarray(frames, float)
    ref = np.asarray(reference, float)
    out = np.stack([superpose(x, ref) for x in f])
    for _ in range(int(n_iter)):
        mean = superpose(out.mean(axis=0), ref)
        out = np.stack([superpose(x, mean) for x in out])
    return out


# --------------------------------------------------------------------------- scores


def overlaps(modes: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """``|v_k . d| / (|v_k| |d|)`` for every column ``v_k`` of ``modes``."""
    m = np.asarray(modes, float)
    d = np.asarray(disp, float).ravel()
    return np.abs(m.T @ d) / (np.linalg.norm(m, axis=0) * np.linalg.norm(d))


def cumulative_overlap(modes: np.ndarray, disp: np.ndarray, k: int) -> float:
    """``|P_K d| / |d|`` for the span of the first ``k`` columns of ``modes``."""
    m = np.asarray(modes, float)[:, : int(k)]
    d = np.asarray(disp, float).ravel()
    q, _ = np.linalg.qr(m)
    return float(np.linalg.norm(q.T @ d) / np.linalg.norm(d))


def random_internal_subspace(xyz: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` random orthonormal directions (3n, k) orthogonal to rigid-body motion."""
    n3 = np.asarray(xyz).size
    rb = rigid_body_basis(xyz)
    g = rng.standard_normal((n3, int(k)))
    g -= rb @ (rb.T @ g)
    q, _ = np.linalg.qr(g)
    return q


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def fisher_se(r: float, n: int) -> float:
    """Standard error of a Pearson r from n pairs, by the Fisher z transform (delta method)."""
    return float((1.0 - r**2) / np.sqrt(max(1, n - 3)))


__all__ = [
    "RAD_PER_PS_TO_CM1",
    "anm_hessian",
    "cartesian_modes",
    "cumulative_overlap",
    "displacement",
    "fisher_se",
    "internal_modes",
    "iterative_superpose",
    "kabsch",
    "msf_from_modes",
    "overlaps",
    "pca",
    "pearson",
    "random_internal_subspace",
    "restrict",
    "rigid_body_basis",
    "rmsd",
    "superpose",
    "wavenumbers_cm1",
]
