"""Fast tests for the ala2 motion export (bonds, restart dedup, quantisation, HILLS)."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from mdbench.branch_c import motion


def test_guess_bonds_methane_like() -> None:
    xyz = np.array([[0.0, 0, 0], [1.09, 0, 0], [0, 1.09, 0], [3.0, 0, 0], [3.0, 1.5, 0]])
    bonds = motion.guess_bonds(xyz, ["C", "H", "H", "C", "C"])
    # C-H at 1.09 bonded; H...H at 1.54 not; C-C at 1.5 bonded; C-C at 3.0 not.
    assert bonds == [(0, 1), (0, 2), (3, 4)]


def test_last_per_time_keeps_restart_rows() -> None:
    cols = {"time": np.array([0.0, 1, 2, 1, 2, 3]), "phi": np.array([9.0, 9, 9, 1, 2, 3])}
    out = motion.last_per_time(cols)
    assert out["time"].tolist() == [0, 1, 2, 3]
    assert out["phi"].tolist() == [9, 1, 2, 3]


def test_quantize_round_trip_and_range() -> None:
    xyz = np.array([[1.234, -4.567, 0.001]])
    q = motion.quantize(xyz)
    back = np.frombuffer(base64.b64decode(motion.b64(q, "<i2")), "<i2") / motion.COORD_SCALE
    assert back == pytest.approx(xyz.ravel(), abs=0.005)
    with pytest.raises(ValueError, match="int16"):
        motion.quantize(np.array([[400.0, 0, 0]]))


def test_metad_block_reads_hills_and_dedups_colvar(tmp_path: Path) -> None:
    (tmp_path / "HILLS").write_text(
        "#! FIELDS time phi psi sigma_phi sigma_psi height biasf\n"
        "1 -2.8 2.5 0.35 0.35 1.44 6\n"
        "2 -2.5 2.0 0.35 0.35 1.40 6\n"
    )
    (tmp_path / "COLVAR").write_text(
        "#! FIELDS time phi psi metad.bias metad.rbias\n"
        "0 -2.0 2.0 0 0\n1 -2.1 2.1 0 0\n"
        "#! FIELDS time phi psi metad.bias metad.rbias\n"
        "1 -2.2 2.2 0 0\n2 -2.3 2.3 0 0\n"
    )
    m = motion.metad_block(tmp_path)
    assert m["n_hills"] == 2 and m["sigma"] == 0.35 and m["biasf"] == 6.0
    t = np.frombuffer(base64.b64decode(m["cv_t"]), "<f4")
    phi = np.frombuffer(base64.b64decode(m["cv_phi"]), "<f4")
    assert t.tolist() == [0, 1, 2]
    assert phi[1] == pytest.approx(-2.2)
