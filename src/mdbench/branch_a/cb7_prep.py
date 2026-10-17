"""CB7 host-guest preparation for branch A (not a results-producing run)::

    python -m mdbench.branch_a.cb7_prep configs/a_cb7_prep.yml

1. **Fetch.** Download each guest's complex (AMBER prmtop/rst7) from the Mobley-Gilson
   benchmark repository at a pinned commit into ``fetch_dir`` (disposable).
2. **Extract and verify.** Take host and guest atoms, elements and bonds from the prmtop and
   coordinates from the rst7; assign bond orders from geometry and the total charge; check
   the guest's heavy-atom skeleton against the SMILES in the ground-truth file. Write
   ``systems/cb7/poses/<guest>.sdf`` (host then guest, bound pose).
3. **Build.** Parameterise host and guest with OpenFF Sage 2.3.0 (NAGL charges), solvate
   the complex and the free guest in TIP3P with NaCl, and write serialised OpenMM systems
   plus coordinates to ``systems/cb7/prepared/<guest>/``.

Everything downstream (L1 docking, L2 MM/GBSA, L3 ABFE) reads only the prepared files, so
all levels use identical parameters, and L3 can run in ``mdbench-fe`` without touching
the toolkit versions that made them.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mdbench import get_code_version
from mdbench.branch_c.adk import _NoAliasDumper, _rel, repo_path, sha256_file

# Prepared file names inside systems/cb7/prepared/<guest>/.
FILES = {
    "complex_system": "complex_system.xml.gz",
    "complex_pdb": "complex.pdb.gz",
    "solvent_system": "solvent_system.xml.gz",
    "solvent_pdb": "solvent.pdb.gz",
    "vacuum_system": "vacuum_system.xml.gz",  # host + guest, no solvent, no cutoff
    "vacuum_pdb": "vacuum.pdb",
    "indices": "indices.json",
}


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    """The CB7 ground-truth YAML (``data/experimental/cb7_binding_dG.yml``)."""
    return yaml.safe_load(repo_path(path).read_text())


# --------------------------------------------------------------------------- fetch


def fetch(prep: Mapping[str, Any], guests: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Download prmtop/rst7 for every guest (skips files already present); returns checksums."""
    src = prep["source"]
    out = repo_path(prep["fetch_dir"])
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, dict[str, str]] = {}
    for gid, g in guests.items():
        record[gid] = {}
        for ext in ("prmtop", "rst7"):
            dest = out / f"cb7-set{g['set']}-{g['id']}.{ext}"
            url = src["url"].format(commit=src["commit"], set=g["set"], id=g["id"], ext=ext)
            if not dest.exists():
                with urllib.request.urlopen(url, timeout=60) as resp:
                    dest.write_bytes(resp.read())
            record[gid][ext] = sha256_file(dest)
            record[gid][f"{ext}_url"] = url
    return record


# --------------------------------------------------------------------------- extract


def residue_to_rdkit(structure: Any, resname: str) -> Any:
    """RDKit molecule (with conformer, bond orders) for the one residue named ``resname``.

    Elements and bonds come from the prmtop; bond orders and formal charges are assigned
    from the geometry and the residue's total partial charge (rounded).
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    res = [r for r in structure.residues if r.name == resname]
    if len(res) != 1:
        raise ValueError(f"expected one residue {resname}, found {len(res)}")
    atoms = list(res[0].atoms)
    local = {a.idx: i for i, a in enumerate(atoms)}
    rw = Chem.RWMol()
    for a in atoms:
        at = Chem.Atom(int(a.atomic_number))
        at.SetNoImplicit(True)
        rw.AddAtom(at)
    for b in structure.bonds:
        i, j = b.atom1.idx, b.atom2.idx
        if i in local and j in local:
            rw.AddBond(local[i], local[j], Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        conf.SetAtomPosition(i, Point3D(float(a.xx), float(a.xy), float(a.xz)))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    charge = round(sum(a.charge for a in atoms))
    rdDetermineBonds.DetermineBondOrders(mol, charge=int(charge))
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    mol.SetProp("_Name", resname)
    return mol


def skeleton_smiles(mol_or_smiles: Any) -> str:
    """Canonical heavy-atom SMILES after neutralising charges: the identity check key."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else mol_or_smiles
    mol = rdMolStandardize.Uncharger().uncharge(Chem.RemoveHs(Chem.Mol(mol)))
    for a in mol.GetAtoms():
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def extract_pose(prmtop: Path, rst7: Path, host: str, guest: str) -> tuple[Any, Any, dict]:
    """(host_mol, guest_mol, record) from one benchmark complex."""
    import parmed

    s = parmed.load_file(str(prmtop), str(rst7))
    h = residue_to_rdkit(s, host)
    g = residue_to_rdkit(s, guest)
    counts: dict[str, int] = {}
    for r in s.residues:
        counts[r.name] = counts.get(r.name, 0) + 1
    rec = {
        "residues_in_source": counts,
        "source_box_A": [round(float(x), 3) for x in s.box[:3]] if s.box is not None else None,
    }
    return h, g, rec


def write_pose_sdf(path: Path, mols: list[Any]) -> None:
    """Host and guest (in that order) as a two-record SDF with formal charges."""
    from rdkit import Chem

    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as w:
        for m in mols:
            w.write(m)


def read_pose_sdf(path: str | Path) -> tuple[Any, Any]:
    """(host, guest) OpenFF Molecules with the bound-pose conformers."""
    from openff.toolkit import Molecule

    mols = Molecule.from_file(str(repo_path(path)), allow_undefined_stereo=True)
    if not isinstance(mols, list) or len(mols) != 2:
        raise ValueError(f"{path}: expected two molecules (host, guest)")
    return mols[0], mols[1]


# --------------------------------------------------------------------------- build


def _box_vectors_nm(top: Any) -> list[list[float]]:
    return np.asarray(top.box_vectors.m_as("nanometer"), float).round(4).tolist()


def _write_system(system: Any, path: Path) -> None:
    import openmm

    with gzip.open(path, "wt", compresslevel=6) as fh:
        fh.write(openmm.XmlSerializer.serialize(system))


def read_system(path: str | Path) -> Any:
    """Deserialise a (gzipped) OpenMM System XML."""
    import openmm

    path = repo_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return openmm.XmlSerializer.deserialize(fh.read())


def _write_pdb(topology: Any, positions: Any, path: Path) -> None:
    from mdbench.branch_c.adk import write_pdb

    write_pdb(topology, positions, path)


def _net_charge(system: Any) -> float:
    import openmm

    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    return round(
        sum(nb.getParticleParameters(i)[0]._value for i in range(system.getNumParticles())), 4
    )


def _ion_indices(top: Any, symbol: str) -> list[int]:
    return [i for i, a in enumerate(top.atoms) if a.molecule.n_atoms == 1 and a.symbol == symbol]


def gb_atom_table(molecules: list[Any]) -> tuple[list[str], list[str | None]]:
    """Element per atom and, for hydrogens, the element of the bonded heavy atom.

    Concatenated over ``molecules`` (OpenFF Molecules) in order: the MM/GBSA radii input.
    """
    elements, neighbour = [], []
    for m in molecules:
        for a in m.atoms:
            elements.append(a.symbol)
            neighbour.append(next(iter(a.bonded_atoms)).symbol if a.symbol == "H" else None)
    return elements, neighbour


def build_guest(
    gid: str,
    host: Any,
    guest: Any,
    prep: Mapping[str, Any],
    out_dir: Path,
    restraint_host: list[int] | None = None,
) -> dict[str, Any]:
    """Parameterise and solvate one complex and its free guest; returns the record.

    ``host`` may be a host molecule or a protein (OpenFF Molecule); ``prep["forcefield"]``
    is one offxml name or a list of them. ``restraint_host`` (host atom indices) anchors
    the ABFE centre-of-mass restraint; default is every host heavy atom.
    """
    from openff.interchange.components import _packmol as packmol
    from openff.toolkit import ForceField, Topology
    from openff.units import unit

    names = prep["forcefield"]
    ff = ForceField(*([names] if isinstance(names, str) else names))
    shapes = {
        "rhombic_dodecahedron": packmol.RHOMBIC_DODECAHEDRON,
        "cube": packmol.UNIT_CUBE,
    }
    kw = {
        "nacl_conc": float(prep["nacl_molar"]) * unit.molar,
        "padding": float(prep["padding_nm"]) * unit.nanometer,
        "box_shape": shapes[prep["box_shape"]],
        "tolerance": float(prep["packmol_tolerance_nm"]) * unit.nanometer,
        "target_density": float(prep["target_density_g_ml"]) * unit.gram / unit.milliliter,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"guest_formal_charge": int(guest.total_charge.m)}
    n_host, n_guest = host.n_atoms, guest.n_atoms

    legs = {
        "complex": Topology.from_molecules([host, guest]),
        "solvent": Topology.from_molecules([guest]),
    }
    for leg, top in legs.items():
        t0 = time.perf_counter()
        solv = packmol.solvate_topology(top, **kw)
        ic = ff.create_interchange(solv)
        system = ic.to_openmm_system(combine_nonbonded_forces=True)
        omm_top = ic.to_openmm_topology()
        _write_system(system, out_dir / FILES[f"{leg}_system"])
        _write_pdb(omm_top, ic.positions.to_openmm(), out_dir / FILES[f"{leg}_pdb"])
        n_solute = n_host + n_guest if leg == "complex" else n_guest
        rec[leg] = {
            "n_atoms": system.getNumParticles(),
            "n_waters": sum(1 for m in solv.molecules if m.hill_formula == "H2O"),
            "n_na": len(_ion_indices(solv, "Na")),
            "n_cl": len(_ion_indices(solv, "Cl")),
            "net_charge_e": _net_charge(system),
            "box_vectors_nm": _box_vectors_nm(solv),
            "solute_atoms": n_solute,
            "n_molecules": solv.n_molecules,
            "wall_clock_s": round(time.perf_counter() - t0, 1),
        }
        if abs(rec[leg]["net_charge_e"]) > 1e-3:
            raise RuntimeError(f"{gid} {leg}: solvated system is not neutral")
        if leg == "complex":
            idx = {
                "host": list(range(n_host)),
                "guest": list(range(n_host, n_host + n_guest)),
                "na": _ion_indices(solv, "Na"),
                "cl": _ion_indices(solv, "Cl"),
            }
        else:
            idx = {
                "guest": list(range(n_guest)),
                "na": _ion_indices(solv, "Na"),
                "cl": _ion_indices(solv, "Cl"),
            }
        rec[leg]["indices"] = idx

    # Host + guest in vacuum (no cutoff, no constraints change): the MM/GBSA reference.
    vac = Topology.from_molecules([host, guest])
    ic = ff.create_interchange(vac)
    vsys = ic.to_openmm_system(combine_nonbonded_forces=True)
    _write_system(vsys, out_dir / FILES["vacuum_system"])
    _write_pdb(ic.to_openmm_topology(), ic.positions.to_openmm(), out_dir / FILES["vacuum_pdb"])
    rec["vacuum"] = {"n_atoms": vsys.getNumParticles(), "net_charge_e": _net_charge(vsys)}

    elements, neighbour = gb_atom_table([host, guest])
    indices = {
        "host": list(range(n_host)),
        "guest": list(range(n_host, n_host + n_guest)),
        "complex": rec["complex"]["indices"],
        "solvent": rec["solvent"]["indices"],
        "guest_formal_charge": rec["guest_formal_charge"],
        "elements": elements,
        "h_neighbour": neighbour,
    }
    if restraint_host is not None:
        indices["complex"]["restraint_host"] = list(restraint_host)
    (out_dir / FILES["indices"]).write_text(json.dumps(indices, indent=1))
    for leg in ("complex", "solvent"):
        rec[leg].pop("indices")
    rec["files"] = {
        k: {"path": _rel(out_dir / v), "sha256": sha256_file(out_dir / v)} for k, v in FILES.items()
    }
    return rec


# --------------------------------------------------------------------------- driver


def run_prep(config_path: str | Path, only: list[str] | None = None) -> Path:
    """Fetch, extract, verify and build every guest; write the preparation record."""
    import openff.interchange
    import openff.toolkit
    import openmm
    import rdkit

    prep = yaml.safe_load(Path(config_path).read_text())["prep"]
    gt = load_ground_truth(prep["ground_truth"])
    guests = {k: v for k, v in gt["guests"].items() if not only or k in only}
    t0 = time.perf_counter()
    checksums = fetch(prep, guests)
    fetch_dir = repo_path(prep["fetch_dir"])
    poses_dir = repo_path(prep["poses_dir"])
    out_root = repo_path(prep["out_dir"])

    records: dict[str, Any] = {}
    for gid, g in guests.items():
        stem = fetch_dir / f"cb7-set{g['set']}-{g['id']}"
        h, gm, rec = extract_pose(
            stem.with_suffix(".prmtop"), stem.with_suffix(".rst7"),
            prep["host_resname"], prep["guest_resname"],
        )  # fmt: skip
        found, expected = skeleton_smiles(gm), skeleton_smiles(g["smiles"])
        if found != expected:
            raise RuntimeError(f"{gid}: guest in the complex is {found}, table says {expected}")
        from rdkit import Chem

        rec.update(
            identity={"expected_skeleton": expected, "found_skeleton": found, "ok": True},
            guest_smiles_as_simulated=Chem.MolToSmiles(Chem.RemoveHs(gm)),
            guest_formal_charge=Chem.GetFormalCharge(gm),
            host_formal_charge=Chem.GetFormalCharge(h),
            source=checksums[gid],
        )
        pose = poses_dir / f"{gid}.sdf"
        write_pose_sdf(pose, [h, gm])
        rec["pose_sdf"] = {"path": _rel(pose), "sha256": sha256_file(pose)}
        host, guest = read_pose_sdf(pose)
        rec["build"] = build_guest(gid, host, guest, prep, out_root / gid)
        records[gid] = rec
        print(f"[prep] {gid}: {rec['guest_smiles_as_simulated']} "
              f"q={rec['guest_formal_charge']:+d} complex={rec['build']['complex']['n_atoms']} "
              f"solvent={rec['build']['solvent']['n_atoms']} atoms", flush=True)  # fmt: skip

    record = {
        "_note": "Generated by `python -m mdbench.branch_a.cb7_prep configs/a_cb7_prep.yml`. "
        "Do not edit by hand; edit the prep config and regenerate.",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": get_code_version(),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "tools": {
            "openmm": openmm.__version__,
            "openff-toolkit": openff.toolkit.__version__,
            "openff-interchange": openff.interchange.__version__,
            "openff-nagl": metadata.version("openff-nagl"),
            "rdkit": rdkit.__version__,
            "parmed": metadata.version("parmed"),
            "packmol": "AmberTools/conda-forge build on PATH",
        },
        "inputs": prep,
        "decisions": {
            "poses": "bound poses from the benchmark authors' equilibrated GAFF complexes "
            "(MOE docking + 2 ns NVT + clustering); used as MD/ABFE starting points only. L1 "
            "docking does not see them.",
            "protonation": "as in the benchmark files (Marvin pKa at pH 4.74); amines "
            "protonated, S1-18 simulated as the neutral acid (file guest-18).",
            "forcefield": "OpenFF Sage 2.3.0 for host and guests (NAGL graph-network AM1-BCC "
            "charges), TIP3P water and Joung-Cheatham ions as bundled in Sage. The benchmark's "
            "own GAFF/RESP parameters are not used.",
            "salt": f"neutralising counterions + NaCl at {prep['nacl_molar']} M (SLTCAP) for "
            "the 50 mM sodium acetate buffer; acetate/acetic acid are not modelled.",
            "box": f"{prep['box_shape']}, {prep['padding_nm']} nm padding, packmol at "
            f"{prep['target_density_g_ml']} g/mL; density relaxes in NPT before sampling.",
            "nonbonded": "PME, 0.9 nm cutoff with 0.8 nm switch (Sage defaults), H-bond "
            "constraints and rigid water from the force field.",
        },
        "guests": records,
    }
    out = repo_path(prep["record"])
    with out.open("w") as fh:
        fh.write("# CB7 host-guest preparation record (generated; see _note).\n")
        yaml.dump(record, fh, Dumper=_NoAliasDumper, sort_keys=False, width=100)
    return out


def main(argv: list[str]) -> int:
    """``python -m mdbench.branch_a.cb7_prep <prep.yml> [guest ...]``."""
    if not argv:
        print(__doc__)
        return 2
    path = run_prep(argv[0], argv[1:] or None)
    print(f"[prep] record -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
