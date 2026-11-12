"""In-between frames in internal coordinates: endpoints exact, bonds kept, rotors rigid."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from mdbench.branch_c import ic_interp as ic

MOTION_JSON = Path(__file__).resolve().parents[1] / "mdbench-motion.json"


def ethane_like(methyl_turn: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """X-C-C(H3) with the methyl turned by ``methyl_turn`` rad about the C-C axis.

    Atoms: 0 X (root neighbour), 1 C root, 2 Y (second root neighbour), 3 C methyl, 4-6 H.
    """
    tet = np.radians(109.47)
    xyz = np.zeros((7, 3))
    xyz[1] = 0, 0, 0
    xyz[0] = -1.5 * np.cos(np.pi - tet), 1.5 * np.sin(np.pi - tet), 0
    xyz[2] = -1.0 * np.cos(np.pi - tet), -1.0 * np.sin(np.pi - tet) * 0.5, 0.8
    xyz[3] = 1.53, 0, 0
    for k in range(3):
        a = methyl_turn + k * 2 * np.pi / 3
        xyz[4 + k] = (
            1.53 + 1.09 * np.cos(np.pi - tet),
            1.09 * np.sin(tet) * np.cos(a),
            1.09 * np.sin(tet) * np.sin(a),
        )
    return xyz, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (3, 6)]


def test_endpoints_reproduced() -> None:
    x0, bonds = ethane_like(0.0)
    x1, _ = ethane_like(0.7)
    x1 = x1 @ np.array(
        [[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]]
    ).T + [0.2, -0.1, 0.4]
    zm = ic.build_zmatrix(7, bonds, root=1, root_a=0, root_b=2)
    np.testing.assert_allclose(ic.interpolate(zm, x0, x1, 0.0), x0, atol=1e-9)
    np.testing.assert_allclose(ic.interpolate(zm, x0, x1, 1.0), x1, atol=1e-9)


@pytest.mark.parametrize("turn", [np.radians(55), np.radians(60), np.radians(65)])
def test_methyl_turns_as_rigid_rotor(turn: float) -> None:
    """A methyl turned ~60° must not split its H's (H-C-H stays tetrahedral mid-way)."""
    x0, bonds = ethane_like(0.0)
    x1, _ = ethane_like(turn)
    zm = ic.build_zmatrix(7, bonds, root=1, root_a=0, root_b=2)
    triples = ic.angle_triples(7, bonds)
    hch = [k for k, (i, c, j) in enumerate(triples) if c == 3 and i >= 4 and j >= 4]
    for u in (0.25, 0.5, 0.75):
        bl, an = ic.geometry(ic.interpolate(zm, x0, x1, u), np.array(bonds), triples)
        np.testing.assert_allclose(bl, ic.geometry(x0, np.array(bonds), triples)[0], atol=1e-9)
        np.testing.assert_allclose(an[hch], an[hch][0], atol=1e-6)
        assert abs(an[hch][0] - 109.47) < 0.5


def test_upsample_shape_and_disconnected_graph() -> None:
    x0, bonds = ethane_like(0.0)
    x1, _ = ethane_like(0.4)
    zm = ic.build_zmatrix(7, bonds, root=1, root_a=0, root_b=2)
    up = ic.upsample(zm, np.array([x0, x1, x0]), 4)
    assert up.shape == (9, 7, 3)
    np.testing.assert_allclose(up[4], x1, atol=1e-9)
    with pytest.raises(ValueError, match="disconnected"):
        ic.build_zmatrix(8, bonds, root=1, root_a=0, root_b=2)


@pytest.mark.skipif(not MOTION_JSON.exists(), reason="mdbench-motion.json not present")
def test_ala2_trajectory_geometry_stays_physical() -> None:
    """Real ala2 frames: bonds stay within the endpoint range, angles within a few degrees."""
    md = json.loads(MOTION_JSON.read_text())["md"]
    n_atoms, n_frames = len(md["atoms"]), md["n_frames"]
    xyz = (
        np.frombuffer(base64.b64decode(md["xyz_i16_centiA"]), "<i2").reshape(n_frames, n_atoms, 3)
        / 100.0
    )
    find = lambda n: next(i for i, a in enumerate(md["atoms"]) if a["r"] == "ALA" and a["n"] == n)  # noqa: E731
    bonds = [tuple(b) for b in md["bonds"]]
    zm = ic.build_zmatrix(n_atoms, bonds, root=find("CA"), root_a=find("N"), root_b=find("C"))
    bi, triples = np.array(bonds), ic.angle_triples(n_atoms, bonds)
    worst_b = worst_a = 0.0
    for f in range(0, 2000):  # first 2 ns; the full 10 ns gives max 7.0° (amide H-N-C)
        g0, g1 = ic.geometry(xyz[f], bi, triples), ic.geometry(xyz[f + 1], bi, triples)
        i0, i1 = ic.internals(zm, xyz[f]), ic.internals(zm, xyz[f + 1])
        for u in (0.25, 0.5, 0.75):
            bl, an = ic.geometry(ic.blend(zm, i0, i1, u), bi, triples)
            worst_b = max(worst_b, ic.excursion(bl, g0[0], g1[0]).max())
            worst_a = max(worst_a, ic.excursion(an, g0[1], g1[1]).max())
    assert worst_b < 1e-6
    assert worst_a < 8.0
