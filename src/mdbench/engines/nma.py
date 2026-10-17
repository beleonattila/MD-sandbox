"""All-atom normal mode analysis with OpenMM: minimise, finite-difference Hessian.

The Hessian is built column by column from central differences of the forces,
``H[:, j] = -(F(x + h e_j) - F(x - h e_j)) / 2h``, which costs ``6 n`` force evaluations.
Finite differences of single-precision forces are dominated by rounding noise at any
useful ``h``, so the context should use double precision (CUDA ``Precision: double``).
Units: kJ/(mol nm^2).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def force_array(context: Any) -> np.ndarray:
    """Forces (n, 3) in kJ/(mol nm) for the context's current positions."""
    from openmm import unit

    st = context.getState(getForces=True)
    return st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def minimise_staged(context: Any, stages: list[dict[str, float]]) -> list[dict[str, float]]:
    """Run ``LocalEnergyMinimizer`` once per stage; returns energy/gradient after each.

    Each stage is ``{"tolerance_kj_mol_nm": ..., "max_iterations": ...}``. Callers swap
    contexts between precisions themselves; this only records where each stage ended.
    """
    import openmm
    from openmm import unit

    out = []
    for s in stages:
        t0 = time.perf_counter()
        openmm.LocalEnergyMinimizer.minimize(
            context, float(s["tolerance_kj_mol_nm"]), int(s["max_iterations"])
        )
        st = context.getState(getEnergy=True, getForces=True)
        g = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        out.append(
            {
                "tolerance_kj_mol_nm": float(s["tolerance_kj_mol_nm"]),
                "energy_kj_mol": st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                "grad_rms_kj_mol_nm": float(np.sqrt(np.mean(g**2))),
                "grad_max_kj_mol_nm": float(np.abs(g).max()),
                "wall_clock_s": time.perf_counter() - t0,
            }
        )
    return out


def fd_hessian(context: Any, positions_nm: np.ndarray, step_nm: float) -> tuple[np.ndarray, float]:
    """Central-difference Hessian (3n, 3n) at ``positions_nm``, symmetrised.

    Returns (Hessian, relative asymmetry before symmetrising), the asymmetry
    ``max|H - H^T| / max|H|`` being a check on the finite-difference noise. The
    context's positions are restored to ``positions_nm`` on return.
    """
    x0 = np.array(positions_nm, float)
    n3 = x0.size
    h = np.empty((n3, n3))
    flat = x0.ravel()
    for j in range(n3):
        xp = flat.copy()
        xp[j] += step_nm
        context.setPositions(xp.reshape(-1, 3))
        fp = force_array(context).ravel()
        xp[j] -= 2.0 * step_nm
        context.setPositions(xp.reshape(-1, 3))
        fm = force_array(context).ravel()
        h[:, j] = -(fp - fm) / (2.0 * step_nm)
    context.setPositions(x0)
    asym = float(np.max(np.abs(h - h.T)) / max(1e-30, np.max(np.abs(h))))
    return 0.5 * (h + h.T), asym


__all__ = ["fd_hessian", "force_array", "minimise_staged"]
