"""AutoDock Vina docking of small, nearly rigid ligands (branch A, L1).

No Meeko/OpenBabel: PDBQT is written here from RDKit molecules with AutoDock 4 atom
types, which is all Vina's scoring function reads. Ligands are docked *rigidly* as an
ensemble of RDKit conformers (ETKDG), and the best score over the ensemble is kept. The
``TORSDOF`` line carries the ligand's rotatable-bond count, which is what Vina uses in its
torsional normalisation ``E / (1 + w * N_rot)``, so scores stay on the standard Vina scale
(the count is heavy-atom torsions, :func:`rotatable_bonds`).

Atom types (AutoDock 4): C aliphatic carbon, A aromatic carbon, N nitrogen that does not
accept H-bonds (amide, aniline, ammonium, pyridinium), NA acceptor nitrogen (pyridine-like
or neutral sp3 amine), OA oxygen, SA/S sulfur, HD polar hydrogen. Non-polar hydrogens are
dropped (Vina is a united-atom function).

References: Trott & Olson (2010) J. Comput. Chem. 31:455, doi:10.1002/jcc.21334; Eberhardt
et al. (2021) J. Chem. Inf. Model. 61:3891, doi:10.1021/acs.jcim.1c00203.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_amide_like_n(atom: Any) -> bool:
    """N bonded to a carbon that is double-bonded to O/N/S (amide, urea, amidine)."""
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() != "C":
            continue
        for b in nb.GetBonds():
            other = b.GetOtherAtom(nb)
            if b.GetBondTypeAsDouble() == 2.0 and other.GetSymbol() in ("O", "N", "S"):
                return True
    return False


def ad4_type(atom: Any) -> str:
    """AutoDock 4 type of an RDKit atom (molecule must have explicit hydrogens)."""
    sym = atom.GetSymbol()
    if sym == "H":
        heavy = atom.GetNeighbors()[0].GetSymbol() if atom.GetDegree() else "C"
        return "HD" if heavy in ("N", "O", "S") else "H"
    if sym == "C":
        return "A" if atom.GetIsAromatic() else "C"
    if sym == "N":
        if atom.GetFormalCharge() > 0:
            return "N"
        if atom.GetIsAromatic():
            return "NA" if atom.GetTotalNumHs() == 0 and atom.GetDegree() == 2 else "N"
        if _is_amide_like_n(atom):
            return "N"
        if any(nb.GetIsAromatic() for nb in atom.GetNeighbors()):
            return "N"  # aniline-type: lone pair conjugated into the ring
        return "NA"
    if sym == "O":
        return "OA"
    if sym == "S":
        return "SA" if atom.GetDegree() <= 2 else "S"
    return {"F": "F", "Cl": "Cl", "Br": "Br", "I": "I", "P": "P"}.get(sym, sym)


def _pdbqt_atom(i: int, name: str, resname: str, xyz: np.ndarray, typ: str) -> str:
    return (
        f"ATOM  {i:5d} {name:<4s} {resname:3s} A   1    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00    {0.0:+.3f} {typ:<2s}"
    )


def to_pdbqt(mol: Any, conf_id: int = -1, rigid_ligand: bool = False, torsdof: int = 0) -> str:
    """PDBQT text for an RDKit molecule with explicit H; non-polar H are dropped.

    ``rigid_ligand`` wraps the atoms in ROOT/ENDROOT with ``TORSDOF torsdof`` (a ligand
    with no active torsions); otherwise the output is a rigid receptor.
    """
    pos = mol.GetConformer(conf_id).GetPositions()
    lines, k = [], 0
    for a in mol.GetAtoms():
        t = ad4_type(a)
        if t == "H":
            continue
        k += 1
        name = f"{a.GetSymbol()}{k}"[:4]
        lines.append(_pdbqt_atom(k, name, "LIG" if rigid_ligand else "HST", pos[a.GetIdx()], t))
    if rigid_ligand:
        return "\n".join(["ROOT", *lines, "ENDROOT", f"TORSDOF {int(torsdof)}", ""])
    return "\n".join([*lines, ""])


# Aromatic ring carbons of standard residues (AutoDock type A).
AROMATIC_C = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"}, "HID": {"CG", "CD2", "CE1"},
    "HIE": {"CG", "CD2", "CE1"}, "HIP": {"CG", "CD2", "CE1"},
}  # fmt: skip


def protein_ad4_type(resname: str, name: str, element: str, n_h: int, heavy_neighbour: str) -> str:
    """AutoDock 4 type of a standard-residue atom from its names and attached hydrogens.

    ``n_h`` is the number of hydrogens bonded to a heavy atom; ``heavy_neighbour`` is the
    element a hydrogen is bonded to. Histidine ring nitrogens accept only when unprotonated.
    """
    if element == "H":
        return "HD" if heavy_neighbour in ("N", "O", "S") else "H"
    if element == "C":
        return "A" if name in AROMATIC_C.get(resname, ()) else "C"
    if element == "N":
        if resname in AROMATIC_C and resname != "TRP" and name in ("ND1", "NE2"):
            return "NA" if n_h == 0 else "N"
        return "N"
    if element == "O":
        return "OA"
    if element == "S":
        return "SA"
    return element


def protein_pdbqt(pdb_path: Any) -> str:
    """Rigid-receptor PDBQT of a protonated protein PDB (standard residues only)."""
    from openmm import app, unit

    pdb = app.PDBFile(str(pdb_path))
    xyz = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    neigh: dict[int, list[Any]] = {}
    for b in pdb.topology.bonds():
        neigh.setdefault(b[0].index, []).append(b[1])
        neigh.setdefault(b[1].index, []).append(b[0])
    lines, k = [], 0
    for a in pdb.topology.atoms():
        el = a.element.symbol
        nbs = neigh.get(a.index, [])
        n_h = sum(1 for x in nbs if x.element.symbol == "H")
        heavy = next((x.element.symbol for x in nbs if x.element.symbol != "H"), "C")
        t = protein_ad4_type(a.residue.name, a.name, el, n_h, heavy)
        if t == "H":
            continue
        k += 1
        lines.append(
            f"ATOM  {k:5d} {a.name[:4]:<4s} {a.residue.name:3s} A{int(a.residue.id):4d}    "
            f"{xyz[a.index][0]:8.3f}{xyz[a.index][1]:8.3f}{xyz[a.index][2]:8.3f}  1.00  0.00"
            f"    {0.0:+.3f} {t:<2s}"
        )
    return "\n".join([*lines, ""])


def rotatable_bonds(mol: Any) -> int:
    """Acyclic single bonds with at least two heavy neighbours on both ends.

    Heavy-atom torsions only: rotating a terminal CH3, NH2, NH3+ or OH moves hydrogens
    alone and is not counted, as in Vina's own torsion count for its normalisation.
    """

    def heavy_degree(a: Any) -> int:
        return sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1)

    n = 0
    for b in mol.GetBonds():
        if b.GetBondTypeAsDouble() != 1.0 or b.IsInRing():
            continue
        if heavy_degree(b.GetBeginAtom()) >= 2 and heavy_degree(b.GetEndAtom()) >= 2:
            n += 1
    return n


def conformers(mol: Any, n: int, seed: int) -> Any:
    """``n`` ETKDGv3 conformers, MMFF-relaxed, on a copy with explicit hydrogens."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=int(n), params=params))
    if not ids:
        raise RuntimeError("conformer embedding failed")
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    return m


def dock_rigid_ensemble(
    receptor_pdbqt: str,
    ligand: Any,
    center: np.ndarray,
    box_A: float,
    exhaustiveness: int,
    seed: int,
    n_poses: int = 1,
) -> dict[str, Any]:
    """Dock each conformer of ``ligand`` rigidly; returns best score and per-conformer scores."""
    from vina import Vina

    torsdof = rotatable_bonds(ligand)
    scores, best_pose = [], None
    v = Vina(sf_name="vina", seed=int(seed), verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
    v.compute_vina_maps(center=[float(c) for c in center], box_size=[float(box_A)] * 3)
    for cid in [c.GetId() for c in ligand.GetConformers()]:
        v.set_ligand_from_string(to_pdbqt(ligand, cid, rigid_ligand=True, torsdof=torsdof))
        v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
        e = float(v.energies(n_poses=1)[0][0])
        if not scores or e < min(scores):
            best_pose = v.poses(n_poses=1)
        scores.append(e)
    return {"best": min(scores), "scores": scores, "torsdof": torsdof, "pose": best_pose}


__all__ = [
    "ad4_type",
    "conformers",
    "dock_rigid_ensemble",
    "protein_ad4_type",
    "protein_pdbqt",
    "rotatable_bonds",
    "to_pdbqt",
]
