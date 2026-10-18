"""MM/GBSA rescoring of MD frames (branch A, L2).

Single-trajectory protocol: frames of the solvated complex are stripped to host + guest,
and each frame is scored as ``E(complex) - E(host) - E(guest)`` with the molecular
mechanics energy plus a generalized-Born solvation term (OBC2, Onufriev-Bashford-Case
2004) and OpenMM's ACE non-polar surface term. No conformational entropy term. The
receptor and ligand coordinates come from the same frame, so their internal energies
cancel exactly and the score is the GB-screened interaction plus desolvation. An optional
entropy term (:func:`interaction_entropy`) is estimated from the same frames.

Radii: Amber ``mbondi2`` (H 1.2 A, H on N 1.3 A, C 1.7, N 1.55, O 1.5, S 1.8, P 1.85,
F 1.5, Cl 1.7, Br 1.85, I 1.98) with the OBC2 screening factors used by Amber's
``igb=5`` (H 0.85, C 0.72, N 0.79, O 0.85, S 0.96, P 0.86, F 0.88; others 0.8). Solute
dielectric 1, solvent 78.5, no salt screening. These are the conventional MM/GBSA
defaults, chosen rather than tuned.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

MBONDI2_A = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.5, "S": 1.8, "P": 1.85, "F": 1.5,
             "Cl": 1.7, "Br": 1.85, "I": 1.98}  # fmt: skip
OBC_SCREEN = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "S": 0.96, "P": 0.86, "F": 0.88}
H_ON_N_RADIUS_A = 1.3


def gb_radii(elements: Sequence[str], bonded_to: Sequence[str | None]) -> np.ndarray:
    """mbondi2 radii in nm; ``bonded_to[i]`` is the heavy neighbour of hydrogen i."""
    out = []
    for el, nb in zip(elements, bonded_to, strict=True):
        r = H_ON_N_RADIUS_A if (el == "H" and nb == "N") else MBONDI2_A.get(el, 1.5)
        out.append(r / 10.0)
    return np.asarray(out)


def subset_system(system: Any, atoms: Sequence[int]) -> Any:
    """A new System holding only ``atoms`` (bonded terms and nonbonded parameters kept).

    Supports the forces Interchange emits for small molecules: harmonic bonds and angles,
    periodic torsions, and one NonbondedForce (exceptions kept when both atoms survive).
    Constraints are dropped; this is for single-point energies.
    """
    import openmm

    keep = {int(a): i for i, a in enumerate(atoms)}
    out = openmm.System()
    for a in atoms:
        out.addParticle(system.getParticleMass(int(a)))
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            g = openmm.HarmonicBondForce()
            for k in range(f.getNumBonds()):
                i, j, r, kk = f.getBondParameters(k)
                if i in keep and j in keep:
                    g.addBond(keep[i], keep[j], r, kk)
        elif isinstance(f, openmm.HarmonicAngleForce):
            g = openmm.HarmonicAngleForce()
            for k in range(f.getNumAngles()):
                i, j, m, th, kk = f.getAngleParameters(k)
                if i in keep and j in keep and m in keep:
                    g.addAngle(keep[i], keep[j], keep[m], th, kk)
        elif isinstance(f, openmm.PeriodicTorsionForce):
            g = openmm.PeriodicTorsionForce()
            for k in range(f.getNumTorsions()):
                i, j, p, q, n, ph, kk = f.getTorsionParameters(k)
                if all(x in keep for x in (i, j, p, q)):
                    g.addTorsion(keep[i], keep[j], keep[p], keep[q], n, ph, kk)
        elif isinstance(f, openmm.NonbondedForce):
            g = openmm.NonbondedForce()
            g.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            for a in atoms:
                g.addParticle(*f.getParticleParameters(int(a)))
            for k in range(f.getNumExceptions()):
                i, j, q, s, e = f.getExceptionParameters(k)
                if i in keep and j in keep:
                    g.addException(keep[i], keep[j], q, s, e)
        elif isinstance(f, openmm.CMMotionRemover):
            continue
        else:
            raise TypeError(f"subset_system: unsupported force {type(f).__name__}")
        g.setForceGroup(f.getForceGroup())
        out.addForce(g)
    return out


def add_gb(system: Any, elements: Sequence[str], bonded_to: Sequence[str | None]) -> Any:
    """Copy of a no-cutoff ``system`` with a GBSAOBCForce (OBC2 + ACE surface term)."""
    import openmm

    out = copy.deepcopy(system)
    nb = next(f for f in out.getForces() if isinstance(f, openmm.NonbondedForce))
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    gb = openmm.GBSAOBCForce()
    gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    for i, r in enumerate(gb_radii(elements, bonded_to)):
        q = nb.getParticleParameters(i)[0]
        gb.addParticle(q, float(r), OBC_SCREEN.get(elements[i], 0.8))
    out.addForce(gb)
    return out


class GBScorer:
    """E(complex) - E(host) - E(guest) in kJ/mol for host + guest coordinate frames.

    With ``solvation=False`` the GB/SA force is left out, which gives the gas-phase MM
    interaction energy that the interaction-entropy estimate is built from.
    """

    def __init__(
        self,
        vacuum_system: Any,
        elements: Sequence[str],
        bonded_to: Sequence[str | None],
        host: Sequence[int],
        guest: Sequence[int],
        platform: str = "CPU",
        solvation: bool = True,
    ) -> None:
        import openmm

        self.host, self.guest = list(host), list(guest)
        self.contexts = {}
        for name, atoms in (("complex", self.host + self.guest), ("host", self.host),
                            ("guest", self.guest)):  # fmt: skip
            sub = subset_system(vacuum_system, atoms)
            el = [elements[a] for a in atoms]
            bt = [bonded_to[a] for a in atoms]
            sysgb = add_gb(sub, el, bt) if solvation else _no_cutoff(sub)
            plat = openmm.Platform.getPlatformByName(platform)
            self.contexts[name] = (
                openmm.Context(sysgb, openmm.VerletIntegrator(0.001), plat),
                atoms,
            )

    def energy(self, name: str, xyz_nm: np.ndarray) -> float:
        """Potential energy (kJ/mol) of one component; ``xyz_nm`` covers host + guest."""
        from openmm import unit

        ctx, atoms = self.contexts[name]
        ctx.setPositions(xyz_nm[atoms])
        st = ctx.getState(getEnergy=True)
        return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def score(self, frames_nm: np.ndarray) -> np.ndarray:
        """Per-frame binding energy (kJ/mol); frames are (n, n_host + n_guest, 3) in nm."""
        out = np.empty(len(frames_nm))
        for k, x in enumerate(frames_nm):
            out[k] = self.energy("complex", x) - self.energy("host", x) - self.energy("guest", x)
        return out


def _no_cutoff(system: Any) -> Any:
    import openmm

    out = copy.deepcopy(system)
    for f in out.getForces():
        if isinstance(f, openmm.NonbondedForce):
            f.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    return out


def interaction_entropy(e_int_kcal: np.ndarray, kt_kcal: float) -> dict[str, float]:
    """-T dS of binding from the fluctuation of the gas-phase interaction energy.

    ``ie`` is the interaction-entropy estimate of Duan, He & Zhang (J. Am. Chem. Soc. 2016,
    doi:10.1021/jacs.6b02682), ``kT ln < exp(dE / kT) >`` with ``dE = E_int - <E_int>``;
    ``c2`` is its second-order cumulant, ``var(E_int) / 2kT`` (Sun et al. 2018), which is
    less dominated by single frames. Both are >= 0 (an entropic penalty), in kcal/mol.
    """
    e = np.asarray(e_int_kcal, float)
    de = (e - e.mean()) / kt_kcal
    m = de.max()
    ie = kt_kcal * (m + np.log(np.mean(np.exp(de - m))))
    c2 = e.var(ddof=1) / (2 * kt_kcal)
    return {"ie": float(ie), "c2": float(c2), "sd_kcal": float(e.std(ddof=1))}


__all__ = ["GBScorer", "add_gb", "gb_radii", "interaction_entropy", "subset_system"]
