"""Normal-mode engine: superposition, ANM, rigid-body projection, PCA, overlap scores."""

from __future__ import annotations

import numpy as np
import pytest

from mdbench.engines import modes as nm


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


@pytest.fixture
def chain() -> np.ndarray:
    """A 40-node random-walk chain, 3.8 A steps (a C-alpha trace stand-in)."""
    rng = np.random.default_rng(0)
    steps = rng.normal(size=(40, 3))
    steps *= 3.8 / np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps, axis=0)


def test_kabsch_recovers_rigid_motion(chain):
    moved = chain @ _rotation([1, 2, 3], 0.7).T + np.array([5.0, -2.0, 1.0])
    assert nm.rmsd(nm.superpose(moved, chain), chain) < 1e-10


def test_kabsch_never_reflects(chain):
    mirrored = chain * np.array([-1.0, 1.0, 1.0])
    r, _, _ = nm.kabsch(mirrored, chain)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_anm_has_six_zero_modes(chain):
    h = nm.anm_hessian(chain, cutoff=13.0)
    w = np.linalg.eigvalsh(h)
    assert np.allclose(w[:6], 0.0, atol=1e-8)
    assert w[6] > 1e-6
    assert np.allclose(h, h.T)


def test_internal_modes_are_orthogonal_to_rigid_body(chain):
    h = nm.anm_hessian(chain, cutoff=13.0)
    vals, vecs = nm.internal_modes(h, chain)
    assert vecs.shape == (3 * len(chain), 3 * len(chain) - 6)
    rb = nm.rigid_body_basis(chain)
    assert np.abs(rb.T @ vecs).max() < 1e-8
    assert np.all(np.diff(vals) >= -1e-12)
    assert vals[0] > 0


def test_mass_weighted_modes_keep_negative_eigenvalue(chain):
    rng = np.random.default_rng(1)
    h = nm.anm_hessian(chain, cutoff=13.0)
    vals0, vecs0 = nm.internal_modes(h, chain)
    v = vecs0[:, 0]
    h_neg = h - (vals0[0] + 0.5) * np.outer(v, v)  # push the softest mode below zero
    masses = rng.uniform(10, 16, len(chain))
    vals, _ = nm.internal_modes(h_neg, chain, masses)
    assert len(vals) == 3 * len(chain) - 6
    assert vals[0] < 0


def test_overlap_of_mode_with_itself_and_cumulative_completeness(chain):
    h = nm.anm_hessian(chain, cutoff=13.0)
    _, vecs = nm.internal_modes(h, chain)
    assert nm.overlaps(vecs[:, :5], 3.0 * vecs[:, 2])[2] == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    d = vecs @ rng.normal(size=vecs.shape[1])  # any internal displacement
    assert nm.cumulative_overlap(vecs, d, vecs.shape[1]) == pytest.approx(1.0)
    assert nm.cumulative_overlap(vecs, d, 5) < 1.0


def test_cumulative_overlap_is_bounded_for_non_orthogonal_modes():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(30, 4))
    m[:, 1] = m[:, 0] + 1e-3 * rng.normal(size=30)  # nearly parallel columns
    d = m[:, 0]
    assert sum(nm.overlaps(m, d) ** 2) > 1.0  # the naive sum would exceed 1
    assert nm.cumulative_overlap(m, d, 4) == pytest.approx(1.0)


def test_random_subspace_overlap_matches_expectation(chain):
    rng = np.random.default_rng(4)
    d = rng.normal(size=chain.size)
    d -= nm.rigid_body_basis(chain) @ (nm.rigid_body_basis(chain).T @ d)
    k, dof = 10, chain.size - 6
    co2 = [nm.cumulative_overlap(nm.random_internal_subspace(chain, k, rng), d, k) ** 2
           for _ in range(400)]  # fmt: skip
    assert np.mean(co2) == pytest.approx(k / dof, rel=0.15)


def test_msf_matches_inverse_hessian(chain):
    h = nm.anm_hessian(chain, cutoff=13.0)
    vals, vecs = nm.internal_modes(h, chain)
    pinv = np.linalg.pinv(h, rcond=1e-10)
    expected = np.array([np.trace(pinv[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]) for i in range(40)])
    assert np.allclose(nm.msf_from_modes(vals, vecs), expected, rtol=1e-6)


def test_pca_recovers_planted_direction(chain):
    rng = np.random.default_rng(5)
    h = nm.anm_hessian(chain, cutoff=13.0)
    _, vecs = nm.internal_modes(h, chain)
    direction = vecs[:, 3]
    amp = rng.normal(scale=2.0, size=500)
    frames = chain[None] + (amp[:, None] * direction[None]).reshape(500, -1, 3)
    frames += rng.normal(scale=0.05, size=frames.shape)
    var, pcs, msf = nm.pca(frames)
    assert nm.overlaps(pcs[:, :1], direction)[0] > 0.99
    assert var[0] > 10 * var[1]
    assert msf.shape == (40,)


def test_iterative_superpose_removes_random_rotations(chain):
    rng = np.random.default_rng(6)
    frames = np.stack(
        [chain @ _rotation(rng.normal(size=3), rng.uniform(0, 3)).T + rng.normal(size=3)
         for _ in range(10)]
    )  # fmt: skip
    fit = nm.iterative_superpose(frames, chain)
    assert np.abs(fit - fit[0]).max() < 1e-8


def test_wavenumbers_sign_and_scale():
    # 1 ps^-2 -> 1 rad/ps -> 5.3088 cm^-1
    assert nm.wavenumbers_cm1(np.array([1.0, -1.0])) == pytest.approx([5.3088, -5.3088], 1e-4)


def test_fisher_se():
    assert nm.fisher_se(0.0, 103) == pytest.approx(0.1)
    assert nm.fisher_se(1.0, 50) == 0.0
