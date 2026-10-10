"""Shared fixtures: a tiny water box and isolation of results/ and runs/."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point results/ at a temporary directory."""
    d = tmp_path / "results"
    monkeypatch.setenv("MDBENCH_RESULTS_DIR", str(d))
    return d


@pytest.fixture(scope="session")
def tiny_water():
    """~650-atom TIP3P box (1.9 nm), tight PME, switched LJ; built offline with Modeller.

    The tight Ewald tolerance is what makes the NVE drift test meaningful: at the
    OpenMM default (5e-4) PME force error alone gives ~0.08 kT/dof/ns on this box.
    """
    from mdbench.bench import build_water_box

    return build_water_box(
        650, ["amber14-all.xml", "amber14/tip3p.xml"], "tip3p", 0.8,
        ewald_error_tolerance=1e-5, switch_distance_nm=0.7,
    )  # fmt: skip
