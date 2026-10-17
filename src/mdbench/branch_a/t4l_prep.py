"""T4 lysozyme L99A preparation for branch A (not a results-producing run)::

    python -m mdbench.branch_a.t4l_prep configs/a_t4l_prep.yml [ligand ...]

Per ligand (``data/experimental/t4l_l99a_binding_dG.yml``):

1. **Receptor.** The ligand's own crystal structure when it is on the WT* (C54T/C97A) +
   L99A background the affinities were measured on; otherwise the configured WT* entry
   for it, with the ligand pose moved in by superposing pocket C-alpha atoms.
2. **Ligand.** Heavy atoms from the entry's HETATM records (first altloc; the copy nearest
   residue 99 if there are several), bond orders from the table SMILES, hydrogens added
   with coordinates.
3. **Protein.** PDBFixer on chain A: heterogens and all crystal waters removed, missing
   atoms added, hydrogens by OpenMM Modeller at the experimental pH.
4. **Build.** OpenFF Sage 2.3.0 (ligand) + ff14SB port (protein), TIP3P, NaCl; serialised
   OpenMM systems for the complex, the free ligand and the vacuum complex, exactly as for
   CB7 (:func:`mdbench.branch_a.cb7_prep.build_guest`).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mdbench import get_code_version
from mdbench.branch_a.cb7_prep import build_guest
from mdbench.branch_c.adk import (
    _NoAliasDumper,
    parse_pdb_header,
    repo_path,
    sha256_file,
    write_pdb,
)
from mdbench.engines.modes import kabsch


def _atoms(path: Path, chain: str) -> list[dict[str, Any]]:
    """ATOM/HETATM records of one chain, first altloc only."""
    out = []
    for ln in path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")) or ln[21] != chain:
            continue
        if ln[16] not in (" ", "A"):
            continue
        out.append({
            "record": ln[:6].strip(), "name": ln[12:16].strip(), "resname": ln[17:20].strip(),
            "resseq": int(ln[22:26]), "xyz": np.array([float(ln[30:38]), float(ln[38:46]),
                                                        float(ln[46:54])]),
            "element": ln[76:78].strip() or ln[12:14].strip(), "line": ln,
        })  # fmt: skip
    return out


def ligand_heavy_atoms(path: Path, chain: str, resname: str) -> tuple[list[dict], dict]:
    """The ligand copy nearest ALA 99 CB (the L99A cavity); returns (atoms, record)."""
    atoms = _atoms(path, chain)
    groups: dict[int, list[dict]] = {}
    for a in atoms:
        if a["resname"] == resname and a["element"] != "H":
            groups.setdefault(a["resseq"], []).append(a)
    if not groups:
        raise ValueError(f"{path.name}: no {resname} in chain {chain}")
    cb = next(a["xyz"] for a in atoms if a["resseq"] == 99 and a["name"] == "CB")
    dist = {k: float(np.linalg.norm(np.mean([a["xyz"] for a in v], axis=0) - cb))
            for k, v in groups.items()}  # fmt: skip
    best = min(dist, key=dist.get)
    return groups[best], {"copies": len(groups), "chosen_resseq": best,
                          "distance_to_A99_CB_A": round(dist[best], 2)}  # fmt: skip


def pocket_superposition(src: Path, dst: Path, chain: str, centre: np.ndarray,
                         cutoff_A: float) -> tuple[np.ndarray, np.ndarray, dict]:  # fmt: skip
    """(R, t) mapping ``src`` coordinates onto ``dst`` by pocket C-alpha superposition."""
    a = {
        x["resseq"]: x["xyz"]
        for x in _atoms(src, chain)
        if x["name"] == "CA" and x["record"] == "ATOM"
    }
    b = {
        x["resseq"]: x["xyz"]
        for x in _atoms(dst, chain)
        if x["name"] == "CA" and x["record"] == "ATOM"
    }
    res = sorted(r for r in set(a) & set(b) if np.linalg.norm(a[r] - centre) < cutoff_A)
    pa, pb = np.array([a[r] for r in res]), np.array([b[r] for r in res])
    r_mat, ca, cb = kabsch(pa, pb)
    moved = (pa - ca) @ r_mat.T + cb
    rmsd = float(np.sqrt(np.mean(np.sum((moved - pb) ** 2, axis=1))))
    return r_mat, cb - ca @ r_mat.T, {"pocket_residues": res, "ca_rmsd_A": round(rmsd, 3)}


def ligand_rdkit(heavy: list[dict], smiles: str, xyz: np.ndarray) -> Any:
    """RDKit ligand with bond orders from ``smiles`` and hydrogens placed on the pose."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for a in heavy:
        rw.AddAtom(Chem.Atom(a["element"].capitalize()))
    conf = Chem.Conformer(len(heavy))
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    m = rw.GetMol()
    m.AddConformer(conf, assignId=True)
    from rdkit.Chem import rdDetermineBonds

    rdDetermineBonds.DetermineConnectivity(m)
    m = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), m)
    for a in m.GetAtoms():  # PDB heavy atoms carry no H: let valence fill them in
        a.SetNoImplicit(False)
        a.SetNumRadicalElectrons(0)
    Chem.SanitizeMol(m)
    m = Chem.AddHs(m, addCoords=True)
    if Chem.MolToSmiles(Chem.RemoveHs(m)) != Chem.MolToSmiles(Chem.MolFromSmiles(smiles)):
        raise RuntimeError(f"ligand perception failed for {smiles}")
    return m


def prepare_protein(raw: Path, chain: str, prep: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """PDBFixer + Modeller hydrogens on one chain, everything else removed; writes ``out``."""
    from openmm import Platform
    from openmm.app import ForceField, Modeller
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(raw))
    drop = set(prep["remove_hetero"])
    chains = list(fixer.topology.chains())
    keep = next(i for i, c in enumerate(chains)
                if c.id == chain and any(r.name not in drop for r in c.residues()))  # fmt: skip
    fixer.removeChains([i for i in range(len(chains)) if i != keep])
    fixer.findMissingResidues()
    n_res = len(list(fixer.topology.residues()))
    fixer.missingResidues = {k: v for k, v in fixer.missingResidues.items() if 0 < k[1] < n_res}
    missing_res = {str(k): v for k, v in fixer.missingResidues.items()}
    fixer.findNonstandardResidues()
    nonstd = [(r.name, r.id, rep) for r, rep in fixer.nonstandardResidues]
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    missing_atoms = {f"{r.name}{r.id}": [a.name for a in v] for r, v in fixer.missingAtoms.items()}
    fixer.addMissingAtoms(seed=0)
    mod = Modeller(fixer.topology, fixer.positions)
    used = mod.addHydrogens(ForceField("amber14-all.xml"), pH=float(prep["ph"]),
                            platform=Platform.getPlatformByName("Reference"))  # fmt: skip
    his = {f"{r.name}{r.id}": v for r, v in zip(mod.topology.residues(), used, strict=True)
           if r.name == "HIS" or (v or "").startswith("HI")}  # fmt: skip
    write_pdb(mod.topology, mod.positions, out)
    return {
        "missing_residues_modelled": missing_res,
        "nonstandard_replaced": nonstd,
        "missing_heavy_atoms_added": missing_atoms,
        "histidine_states": his,
        "n_residues": mod.topology.getNumResidues(),
        "n_atoms_with_H": mod.topology.getNumAtoms(),
    }


def run_prep(config_path: str | Path, only: list[str] | None = None) -> Path:
    """Prepare every ligand's complex; write the preparation record."""
    import openff.interchange
    import openff.toolkit
    import openmm
    import rdkit
    from openff.toolkit import Molecule, Topology
    from rdkit import Chem

    prep = yaml.safe_load(Path(config_path).read_text())["prep"]
    gt = yaml.safe_load(repo_path(prep["ground_truth"]).read_text())
    ligs = {k: v for k, v in gt["ligands"].items() if not only or k in only}
    raw_dir, out_root = repo_path(prep["raw_dir"]), repo_path(prep["out_dir"])
    chain = prep["chain"]
    rc = prep["receptor"]
    t0 = time.perf_counter()
    records: dict[str, Any] = {}
    for lid, lg in ligs.items():
        t1 = time.perf_counter()
        src = raw_dir / f"{lg['pdb']}.pdb"
        rec_entry = (
            lg["pdb"] if lg["pdb"] in rc["wt_star_entries"] else rc["fallback_entry"][lg["pdb"]]
        )
        dst = raw_dir / f"{rec_entry}.pdb"
        heavy, lrec = ligand_heavy_atoms(src, chain, lg["resname"])
        xyz = np.array([a["xyz"] for a in heavy])
        hdr = parse_pdb_header(src)
        rec: dict[str, Any] = {
            "source_entry": lg["pdb"], "receptor_entry": rec_entry,
            "source_header": {k: hdr[k] for k in ("id_code", "resolution_A", "ligands")},
            "source_sha256": sha256_file(src), "receptor_sha256": sha256_file(dst),
            "ligand_copy": lrec,
        }  # fmt: skip
        if rec_entry != lg["pdb"]:
            cut_a = 10.0 * float(prep["pocket_cutoff_nm"])
            r_mat, t, srec = pocket_superposition(src, dst, chain, xyz.mean(0), cut_a)
            xyz = xyz @ r_mat.T + t
            rec["pose_transfer"] = srec
        lig = ligand_rdkit(heavy, lg["smiles"], xyz)
        out = out_root / lid
        out.mkdir(parents=True, exist_ok=True)
        prot_pdb = out / "protein.pdb"
        rec["protein"] = prepare_protein(dst, chain, prep, prot_pdb)
        protein = Topology.from_pdb(str(prot_pdb)).molecule(0)
        ligand = Molecule.from_rdkit(lig, allow_undefined_stereo=True)
        pxyz = protein.conformers[0].m_as("nanometer")
        lc = lig.GetConformer().GetPositions().mean(axis=0) / 10.0
        dmin = float(
            np.min(
                np.linalg.norm(pxyz[:, None] - lig.GetConformer().GetPositions()[None] / 10, axis=2)
            )
        )
        rec["ligand_protein_min_distance_nm"] = round(dmin, 3)
        ca = [a.molecule_atom_index for a in protein.atoms
              if a.name == "CA" and np.linalg.norm(pxyz[a.molecule_atom_index] - lc)
              < float(prep["restraint_ca_cutoff_nm"])]  # fmt: skip
        rec["restraint_ca_atoms"] = len(ca)
        rec["ligand_smiles"] = Chem.MolToSmiles(Chem.RemoveHs(lig))
        rec["build"] = build_guest(lid, protein, ligand, prep, out, restraint_host=ca)
        rec["wall_clock_s"] = round(time.perf_counter() - t1, 1)
        records[lid] = rec
        print(f"[prep] {lid}: receptor {rec_entry}, complex {rec['build']['complex']['n_atoms']} "
              f"atoms, solvent {rec['build']['solvent']['n_atoms']}, "
              f"{rec['wall_clock_s']} s", flush=True)  # fmt: skip

    record = {
        "_note": "Generated by `python -m mdbench.branch_a.t4l_prep configs/a_t4l_prep.yml`. "
        "Do not edit by hand; edit the prep config and regenerate.",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": get_code_version(),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "tools": {
            "openmm": openmm.__version__,
            "pdbfixer": metadata.version("pdbfixer"),
            "openff-toolkit": openff.toolkit.__version__,
            "openff-interchange": openff.interchange.__version__,
            "openff-nagl": metadata.version("openff-nagl"),
            "rdkit": rdkit.__version__,
        },
        "inputs": prep,
        "decisions": {
            "receptor": "own crystal structure when on the WT* + L99A background of the "
            "measurements; toluene (4W53) pose moved into 181L and propylbenzene (4W55) into "
            "186L by pocket C-alpha superposition",
            "waters": "all crystal waters removed; the apolar cavity is dry when empty, and "
            "the ligand fills it in the complex. Bulk TIP3P by packmol.",
            "protonation": f"OpenMM Modeller at pH {prep['ph']} (fixed residue pKa values; "
            "histidine by H-bond analysis); no pKa calculation",
            "forcefield": "ff14SB (OpenFF port ff14sb_off_impropers_0.0.4) for the protein, "
            "Sage 2.3.0 with NAGL charges for the ligand, TIP3P, Joung-Cheatham ions",
            "salt": f"neutralise + NaCl at {prep['nacl_molar']} M for 50 mM sodium acetate",
            "restraint_anchor": "ABFE centre-of-mass restraint between the ligand heavy atoms "
            f"and protein C-alpha atoms within {prep['restraint_ca_cutoff_nm']} nm of the "
            "ligand centre",
        },
        "ligands": records,
    }
    out = repo_path(prep["record"])
    with out.open("w") as fh:
        fh.write("# T4 lysozyme L99A preparation record (generated; see _note).\n")
        yaml.dump(record, fh, Dumper=_NoAliasDumper, sort_keys=False, width=100)
    return out


def main(argv: list[str]) -> int:
    """``python -m mdbench.branch_a.t4l_prep <prep.yml> [ligand ...]``."""
    if not argv:
        print(__doc__)
        return 2
    print(f"[prep] record -> {run_prep(argv[0], argv[1:] or None)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
