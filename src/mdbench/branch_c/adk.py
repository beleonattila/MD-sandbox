"""Branch C, adenylate kinase (E. coli AdK): preparation and the L0-L3 runners.

Preparation (not a results-producing run)::

    python -m mdbench.branch_c.adk prep configs/c_adk_prep.yml

reads the committed raw PDB files, verifies them against the expectations in the prep
config (organism, ligand, chains, resolution), runs PDBFixer + OpenMM Modeller and writes
``systems/adk/prepared/*`` plus the full preparation record ``systems/adk/prep.yml``.

Prediction targets (the ``system`` column of each row; one run emits several):

====================================  =====================================================
``adk_open_theta_nmp`` / ``_lid``     domain angle of the open basin (deg)
``adk_closed_theta_nmp`` / ``_lid``   domain angle of the closed basin (deg)
``adk_apo_theta_nmp_mean`` / ``_lid`` ensemble-mean domain angle of apo AdK (deg)
``adk_apo_p_closed``                  equilibrium fraction of apo AdK in the closed state
``adk_apo_dG_closed_open``            closed minus open free energy, kJ/mol (L0: energy)
====================================  =====================================================

"Closed" is the box ``theta_nmp < closed.theta_nmp_max and theta_lid < closed.theta_lid_max``
(config ``analysis.closed``), by default the midpoints between the 1AKE and 4AKE angles.

Runners (``method``):

- L0 ``boltzmann_endpoints_gbn2_min``: two-state Boltzmann over the minimised crystal
  endpoints in GBn2 implicit solvent (potential energy only, no entropy).
- L1 ``static_cv_crystal``: domain angles of the crystal structures.
- L2 ``unbiased_md`` (and ``unbiased_md_smoke``): explicit-solvent MD, histogram.
- L3 ``metad_wt_domain_angles``: well-tempered metadynamics on both angles, reweighted.
- L3 ``umbrella_theta_lid``: umbrella windows on theta_LID, MBAR.
- L3 ``msm_domain_angles``: MSM over a set of unbiased trajectories.
"""

from __future__ import annotations

import copy
import gc
import glob
import gzip
import hashlib
import io
import json
import logging
import math
import platform as _platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mdbench import REPO_ROOT, get_code_version
from mdbench.engines import domain_angles as da
from mdbench.registry import register

log = logging.getLogger(__name__)

KB_KJ_MOL_K = 0.0083144626  # Boltzmann constant, kJ/(mol K)
TARGETS: tuple[str, ...] = (
    "adk_open_theta_nmp",
    "adk_open_theta_lid",
    "adk_closed_theta_nmp",
    "adk_closed_theta_lid",
    "adk_apo_theta_nmp_mean",
    "adk_apo_theta_lid_mean",
    "adk_apo_p_closed",
    "adk_apo_dG_closed_open",
)


# --------------------------------------------------------------------------- utilities


def repo_path(p: str | Path) -> Path:
    """Resolve ``p`` relative to the repository root unless it is absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256_file(path: str | Path) -> str:
    """sha256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def jeffreys_sd(k: float, n: float) -> float:
    """Posterior sd of a proportion under a Jeffreys Beta(1/2, 1/2) prior.

    Used as a floor on population uncertainties: with ``n`` effectively independent
    samples and ``k`` hits it is never zero, even when no sample was closed.
    """
    a, b = k + 0.5, n - k + 0.5
    return float(math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0))))


def block_sem(x: np.ndarray, n_blocks: int, weights: np.ndarray | None = None) -> float:
    """Standard error of a (weighted) mean from ``n_blocks`` contiguous blocks."""
    x = np.asarray(x, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    nb = max(2, min(int(n_blocks), len(x)))
    means = []
    for xs, ws in zip(np.array_split(x, nb), np.array_split(w, nb), strict=True):
        if len(xs) and ws.sum() > 0:
            means.append(float(np.sum(ws * xs) / np.sum(ws)))
    if len(means) < 2:
        return float("nan")
    return float(np.std(means, ddof=1) / math.sqrt(len(means)))


def closed_indicator(theta_nmp: np.ndarray, theta_lid: np.ndarray, closed: Mapping) -> np.ndarray:
    """1.0 where (theta_nmp, theta_lid) [deg] is inside the closed box, else 0.0."""
    return (
        (np.asarray(theta_nmp) < float(closed["theta_nmp_max"]))
        & (np.asarray(theta_lid) < float(closed["theta_lid_max"]))
    ).astype(float)


def population_rows(
    theta_nmp: np.ndarray,
    theta_lid: np.ndarray,
    analysis: Mapping[str, Any],
    weights: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows for p_closed and the mean angles from (optionally weighted) samples.

    Uncertainty: block standard error over ``analysis.n_blocks`` contiguous blocks.
    For p_closed it is floored by the Jeffreys posterior sd with ``n = n_blocks`` (an
    effective-sample count that is conservative for correlated MD), so a run that never
    visits the closed state reports an honest non-zero error bar instead of 0.
    """
    nb = int(analysis.get("n_blocks", 5))
    w = np.ones(len(theta_nmp)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    ind = closed_indicator(theta_nmp, theta_lid, analysis["closed"])
    p = float(np.sum(w * ind))
    p_unc = block_sem(ind, nb, w)
    p_floor = jeffreys_sd(p * nb, nb)
    p_unc = p_floor if not math.isfinite(p_unc) else max(p_unc, p_floor)
    rows = [{"system": "adk_apo_p_closed", "prediction": p, "uncertainty": p_unc}]
    diag: dict[str, Any] = {"n_samples": len(theta_nmp), "n_blocks": nb, "p_closed": p}
    for name, x in (("theta_nmp", theta_nmp), ("theta_lid", theta_lid)):
        m = float(np.sum(w * x))
        u = block_sem(x, nb, w)
        if not math.isfinite(u):
            u = float(np.sqrt(np.sum(w * (x - m) ** 2)))
        rows.append({"system": f"adk_apo_{name}_mean", "prediction": m, "uncertainty": u})
        diag[f"{name}_mean"] = m
    # Time-resolved estimate (cumulative fractions) as the convergence diagnostic.
    fr = np.linspace(0.1, 1.0, 10)
    cum = []
    for f in fr:
        n = max(1, round(f * len(ind)))
        ww = w[:n] / w[:n].sum()
        cum.append(float(np.sum(ww * ind[:n])))
    diag["p_closed_cumulative"] = {"fraction": fr.round(2).tolist(), "p_closed": cum}
    return rows, diag


# --------------------------------------------------------------------------- preparation


def parse_pdb_header(path: str | Path) -> dict[str, Any]:
    """Facts from a PDB file header used to verify the entry against docs/SYSTEMS.md."""
    text = Path(path).read_text().splitlines()

    def spec(rec: str, key: str) -> str | None:
        for ln in text:
            if ln.startswith(rec) and f"{key}:" in ln:
                return ln[10:80].split(":", 1)[1].strip().rstrip(";").strip()
        return None

    res = None
    for ln in text:
        if ln.startswith("REMARK   2 RESOLUTION."):
            res = float(ln.split()[3])
    hetnam = sorted({ln[11:14].strip() for ln in text if ln.startswith("HETNAM")})
    chains: dict[str, set[int]] = {}
    for ln in text:
        if ln.startswith("ATOM"):
            chains.setdefault(ln[21], set()).add(int(ln[22:26]))
    seqres: dict[str, list[str]] = {}
    for ln in text:
        if ln.startswith("SEQRES"):
            seqres.setdefault(ln[11], []).extend(ln[19:].split())
    doi = next((ln[19:].strip() for ln in text if ln.startswith("JRNL        DOI")), None)
    taxid = spec("SOURCE", "ORGANISM_TAXID")
    return {
        "id_code": text[0][62:66].strip() if text and text[0].startswith("HEADER") else None,
        "header": text[0][10:50].strip() if text else None,
        "molecule": spec("COMPND", "MOLECULE"),
        "organism_scientific": spec("SOURCE", "ORGANISM_SCIENTIFIC"),
        "organism_taxid": int(taxid) if taxid else None,
        "method": next((ln[10:].strip() for ln in text if ln.startswith("EXPDTA")), None),
        "resolution_A": res,
        "ligands": hetnam,
        "chains": sorted(chains),
        "residues_per_chain": {c: len(r) for c, r in sorted(chains.items())},
        "residue_range": {c: [min(r), max(r)] for c, r in sorted(chains.items())},
        "seqres_identical_across_chains": len({tuple(s) for s in seqres.values()}) == 1,
        "seqres_length": {c: len(s) for c, s in sorted(seqres.items())},
        "missing_residue_records_REMARK465": sum(ln.startswith("REMARK 465") for ln in text),
        "altloc_atoms": sum(ln.startswith(("ATOM", "HETATM")) and ln[16] != " " for ln in text),
        "altloc_residues": sorted(
            {
                f"{ln[17:20]} {ln[21]}{int(ln[22:26])}"
                for ln in text
                if ln.startswith(("ATOM", "HETATM")) and ln[16] != " "
            }
        ),
        "crystal_waters": sum(ln.startswith("HETATM") and ln[17:20] == "HOH" for ln in text),
        "luzzati_esd_A": next(
            (
                ln.split(":")[1].strip()
                for ln in text
                if ln.startswith("REMARK   3") and "ESD FROM LUZZATI PLOT" in ln
            ),
            None,
        ),
        "cis_peptides": [ln[11:40].split() for ln in text if ln.startswith("CISPEP")],
        "primary_citation_doi_in_file": doi,
    }


def verify_entry(found: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Compare parsed header facts with expectations; returns {check: {expected, found, ok}}."""
    checks: dict[str, Any] = {}
    for key in ("organism_scientific", "organism_taxid", "molecule", "method"):
        checks[key] = {
            "expected": expected[key],
            "found": found[key],
            "ok": found[key] == expected[key],
        }
    checks["resolution_A"] = {
        "expected": expected["resolution_A"],
        "found": found["resolution_A"],
        "ok": found["resolution_A"] is not None
        and abs(found["resolution_A"] - float(expected["resolution_A"])) < 1e-6,
    }
    checks["ligands"] = {
        "expected": list(expected["ligands"]),
        "found": found["ligands"],
        "ok": sorted(expected["ligands"]) == found["ligands"],
    }
    checks["chains"] = {
        "expected": list(expected["chains"]),
        "found": found["chains"],
        "ok": list(expected["chains"]) == found["chains"],
    }
    nres = int(expected["residues_per_chain"])
    checks["residues_per_chain"] = {
        "expected": nres,
        "found": found["residues_per_chain"],
        "ok": all(v == nres for v in found["residues_per_chain"].values()),
    }
    checks["all_ok"] = all(v["ok"] for v in checks.values())
    return checks


def _variants(modeller_variants: Sequence[str | None], topology: Any) -> dict[str, str]:
    out = {}
    for res, var in zip(topology.residues(), modeller_variants, strict=True):
        if res.name in ("HIS", "HID", "HIE", "HIP") or (var and var.startswith("HI")):
            out[f"{res.name}{res.id}"] = str(var)
    return out


_AMINO_ACIDS = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
)  # fmt: skip


def prepare_chain(raw_pdb: Path, chain: str, prep: Mapping[str, Any]) -> tuple[Any, Any, dict]:
    """PDBFixer + Modeller.addHydrogens on one chain; returns (topology, positions, record)."""
    from openmm import Platform
    from openmm.app import ForceField, Modeller
    from pdbfixer import PDBFixer

    random.seed(int(prep["seed"]))
    np.random.seed(int(prep["seed"]))
    fixer = PDBFixer(filename=str(raw_pdb))
    chains = list(fixer.topology.chains())
    protein_chain_index = next(
        i
        for i, c in enumerate(chains)
        if c.id == chain and any(r.name not in ("HOH", "AP5") for r in c.residues())
    )
    drop = [i for i in range(len(chains)) if i != protein_chain_index]
    dropped: dict[str, int] = {}
    for i in drop:
        for r in chains[i].residues():
            if r.name in _AMINO_ACIDS:
                key = f"protein chain {chains[i].id} (residues)"
            else:
                key = f"{r.name} (chain {chains[i].id})"
            dropped[key] = dropped.get(key, 0) + 1
    fixer.removeChains(drop)
    before = sorted({r.name for r in fixer.topology.residues()})

    fixer.findMissingResidues()
    n_res = len(list(fixer.topology.residues()))
    terminal = {k: v for k, v in fixer.missingResidues.items() if k[1] in (0, n_res)}
    for k in terminal:
        del fixer.missingResidues[k]
    missing_residues = {
        f"chain{k[0]}:after_index{k[1]}": v for k, v in fixer.missingResidues.items()
    }
    fixer.findNonstandardResidues()
    nonstandard = [(r.name, r.id, rep) for r, rep in fixer.nonstandardResidues]
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=prep.get("crystal_waters") == "keep")
    after = sorted({r.name for r in fixer.topology.residues()})
    fixer.findMissingAtoms()
    missing_atoms = {
        f"{r.name}{r.id}": [a.name for a in atoms] for r, atoms in fixer.missingAtoms.items()
    }
    missing_terminals = {f"{r.name}{r.id}": list(t) for r, t in fixer.missingTerminals.items()}
    fixer.addMissingAtoms(seed=int(prep["seed"]))

    ff = ForceField(*prep["forcefield"])
    mod = Modeller(fixer.topology, fixer.positions)
    override = {str(k): str(v) for k, v in (prep.get("histidine_override") or {}).items()}
    variants = [override.get(str(r.id)) for r in mod.topology.residues()]
    plat = Platform.getPlatformByName(str(prep.get("hydrogen_platform", "CPU")))
    used = mod.addHydrogens(ff, pH=float(prep["ph"]), variants=variants, platform=plat)
    record = {
        "chain": chain,
        "residues_removed_with_other_chains": dict(sorted(dropped.items())),
        "residues_removed_as_heterogens": sorted(set(before) - set(after)),
        "missing_residues_modelled": missing_residues,
        "missing_terminal_residues_not_modelled": {
            f"after_index{k[1]}": v for k, v in terminal.items()
        },
        "nonstandard_residues_replaced": nonstandard,
        "missing_heavy_atoms_added": missing_atoms,
        "missing_terminal_atoms_added": missing_terminals,
        "histidine_states": _variants(used, mod.topology),
        "n_residues": mod.topology.getNumResidues(),
        "n_atoms_protein_with_H": mod.topology.getNumAtoms(),
    }
    return mod.topology, mod.positions, record


def net_charge(topology: Any, forcefield: Any) -> float:
    """Net charge (e) of ``topology`` under ``forcefield`` (NonbondedForce charges)."""
    import openmm

    system = forcefield.createSystem(topology)
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    q = sum(nb.getParticleParameters(i)[0]._value for i in range(nb.getNumParticles()))
    return round(float(q), 4)


def write_pdb(topology: Any, positions: Any, path: Path) -> None:
    """Write a PDB (gzip when ``path`` ends in .gz), keeping residue ids."""
    from openmm.app import PDBFile

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        # mtime=0 keeps the gzip header, and so the checksum, reproducible.
        with (
            gzip.GzipFile(path, "wb", mtime=0) as gz,
            io.TextIOWrapper(gz, encoding="ascii") as fh,
        ):
            PDBFile.writeFile(topology, positions, fh, keepIds=True)
    else:
        with path.open("w") as fh:
            PDBFile.writeFile(topology, positions, fh, keepIds=True)


def read_pdb(path: str | Path) -> Any:
    """``openmm.app.PDBFile`` from a (possibly gzipped) PDB."""
    from openmm.app import PDBFile

    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return PDBFile(fh)
    return PDBFile(str(path))


def _box_record(vectors: Any) -> dict[str, Any]:
    from openmm import unit

    v = np.array([[x.value_in_unit(unit.nanometer) for x in vec] for vec in vectors])
    return {
        "vectors_nm": v.round(4).tolist(),
        "volume_nm3": round(float(abs(np.linalg.det(v))), 2),
        "width_nm": round(float(v[0, 0]), 4),
    }


def run_prep(config_path: str | Path) -> Path:
    """Prepare every structure in the prep config and write the preparation record."""
    import openmm
    from openmm import unit
    from openmm.app import ForceField, Modeller

    cfg_raw = yaml.safe_load(Path(config_path).read_text())
    prep = cfg_raw["prep"]
    raw_dir, out_dir = repo_path(prep["raw_dir"]), repo_path(prep["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = ForceField(*prep["forcefield"])
    t0 = time.perf_counter()

    structures: dict[str, Any] = {}
    order = sorted(prep["structures"], key=lambda s: s != prep["box_from"])  # box_from first
    box_vectors = None
    variant_sets = {}
    for sid in order:
        meta = prep["structures"][sid]
        raw = raw_dir / f"{sid}.pdb"
        found = parse_pdb_header(raw)
        checks = verify_entry(found, meta["expected"])
        if not checks["all_ok"]:
            bad = {k: v for k, v in checks.items() if k != "all_ok" and not v["ok"]}
            raise RuntimeError(f"{sid} does not match its expected identity: {bad}")
        srec: dict[str, Any] = {
            "state": meta["state"],
            "role": meta["role"],
            "raw_file": _rel(raw),
            "raw_sha256": sha256_file(raw),
            "header": found,
            "verification": checks,
            "chains": {},
        }
        for ch in prep["chains"]:
            top, pos, rec = prepare_chain(raw, ch, prep)
            rec["net_charge_e"] = net_charge(top, ff)
            variant_sets[f"{sid}_{ch}"] = rec["histidine_states"]
            p_path = out_dir / f"{sid}_{ch}_protein.pdb"
            write_pdb(top, pos, p_path)
            rec["protein_pdb"] = _rel(p_path)
            rec["protein_pdb_sha256"] = sha256_file(p_path)
            if ch in prep["solvate_chains"]:
                random.seed(int(prep["seed"]))
                mod = Modeller(top, pos)
                kw: dict[str, Any] = {
                    "model": prep["water_model"],
                    "boxShape": prep["box_shape"],
                    "ionicStrength": float(prep["ionic_strength_molar"]) * unit.molar,
                    "positiveIon": prep["positive_ion"],
                    "negativeIon": prep["negative_ion"],
                    "neutralize": bool(prep["neutralize"]),
                }
                if box_vectors is None:
                    kw["padding"] = float(prep["padding_nm"]) * unit.nanometer
                else:
                    kw["boxVectors"] = box_vectors
                mod.addSolvent(ff, **kw)
                if box_vectors is None:
                    box_vectors = mod.topology.getPeriodicBoxVectors()
                names = [r.name for r in mod.topology.residues()]
                s_path = out_dir / f"{sid}_{ch}_solvated.pdb.gz"
                write_pdb(mod.topology, mod.positions, s_path)
                n_prot = rec["n_atoms_protein_with_H"]
                rec["solvated"] = {
                    "pdb": _rel(s_path),
                    "pdb_sha256": sha256_file(s_path),
                    "n_atoms_total": mod.topology.getNumAtoms(),
                    "n_atoms_protein": n_prot,
                    "n_waters": names.count("HOH"),
                    "n_na": names.count("NA"),
                    "n_cl": names.count("CL"),
                    "net_charge_e": net_charge(mod.topology, ff),
                    "box": _box_record(mod.topology.getPeriodicBoxVectors()),
                    "box_sized_from": prep["box_from"],
                }
            srec["chains"][ch] = rec
        structures[sid] = srec

    distinct = {json.dumps(v, sort_keys=True) for v in variant_sets.values()}
    his_values = [sorted(v.values()) for v in variant_sets.values()]
    if prep.get("require_identical_protonation") and len({tuple(h) for h in his_values}) > 1:
        raise RuntimeError(
            f"histidine states differ between structures; set histidine_override: {variant_sets}"
        )

    se = prep["storage_estimate"]
    n_frames = float(se["duration_ns"]) * 1000.0 / float(se["frame_interval_ps"])
    first = structures[prep["box_from"]]["chains"][prep["solvate_chains"][0]]
    n_prot = first["n_atoms_protein_with_H"]
    n_heavy = sum(
        1
        for ln in repo_path(first["protein_pdb"]).read_text().splitlines()
        if ln.startswith("ATOM") and ln[76:78].strip() != "H"
    )
    n_tot = first["solvated"]["n_atoms_total"]

    def mb(n: int) -> float:
        return round(n * 3 * 4 * n_frames / 1e6, 1)

    record = {
        "_note": "Generated by `python -m mdbench.branch_c.adk prep configs/c_adk_prep.yml`. "
        "Do not edit by hand; edit the prep config and regenerate.",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": get_code_version(),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "tools": {
            "pdbfixer": metadata.version("pdbfixer"),
            "openmm": openmm.__version__,
            "python": _platform.python_version(),
            "protonation_tool": "OpenMM Modeller.addHydrogens (called directly, as PDBFixer "
            "addMissingHydrogens does); standard residue pKa values at the given pH, histidine "
            "tautomer from hydrogen-bond analysis. No pKa calculation (no PROPKA/H++).",
        },
        "inputs": prep,
        "decisions": {
            "chain": "chain A of each entry is solvated for MD; chain B is prepared "
            "protein-only and used as an independent copy for L0/L1 uncertainty",
            "ligand": "Ap5A (AP5) removed from 1AKE: L0-L3 describe apo AdK; 1AKE then "
            "serves as the apo closed-state model. No ligand parameters are needed.",
            "crystal_waters": "all removed (none are buried catalytic waters relevant to the "
            "domain angles); bulk TIP3P added",
            "missing_loops": "none needed: both entries have complete chains 1-214 (no "
            "REMARK 465); PDBFixer found no missing residues or heavy atoms (per-chain "
            "records below)",
            "altlocs": "first alternate location kept (OpenMM PDBFile)",
            "protonation": f"pH {prep['ph']}; Asp/Glu deprotonated, Lys/Arg protonated, "
            "termini charged; histidine states listed per chain",
            "forcefield": "Amber ff14SB (amber14-all.xml) + TIP3P (amber14/tip3p.xml), "
            "OpenMM bundled; Na+/Cl- Joung-Cheatham parameters from amber14/tip3p.xml",
            "box": f"{prep['box_shape']}, padding {prep['padding_nm']} nm (OpenMM: width = "
            "bounding-sphere diameter + padding), sized on "
            f"{prep['box_from']} and reused for the other structure",
            "ions": f"neutralise + {prep['ionic_strength_molar']} M NaCl",
        },
        "histidine_states_all_structures_identical": len(distinct) == 1
        or len({tuple(h) for h in his_values}) == 1,
        "structures": structures,
        "atom_counts": {
            "protein_only_with_H": n_prot,
            "protein_only_heavy": n_heavy,
            "solvated_total": n_tot,
        },
        "storage_estimate": {
            "basis": f"{se['duration_ns']} ns at {se['frame_interval_ps']} ps/frame = "
            f"{int(n_frames)} frames; float32 bytes/frame = n_atoms*3*4 (docs/HARDWARE.md). "
            "XTC at precision 1000 compresses this ~3x; see the smoke-run measurement.",
            "protein_only_with_H_MB": mb(n_prot),
            "protein_only_heavy_MB": mb(n_heavy),
            "solvated_full_MB": mb(n_tot),
        },
    }
    out = repo_path(prep["record"])
    with out.open("w") as fh:
        fh.write("# Adenylate kinase preparation record (generated; see _note).\n")
        yaml.dump(record, fh, Dumper=_NoAliasDumper, sort_keys=False, width=100)
    return out


class _NoAliasDumper(yaml.SafeDumper):
    """SafeDumper that writes repeated objects in full (no &id anchors)."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


# --------------------------------------------------------------------------- shared helpers


def _md_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(cfg["md"]))


def prepared_universe(path: str | Path) -> Any:
    """MDAnalysis Universe of a prepared (possibly gzipped) PDB."""
    import MDAnalysis as mda

    return mda.Universe(str(repo_path(path)))


def protein_endpoints(cfg: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """``{state: {label: protein_pdb}}`` from ``cfg['adk']['endpoints']``."""
    return {s: dict(v) for s, v in cfg["adk"]["endpoints"].items()}


def build_explicit_system(cfg: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """(topology, system, positions, box_vectors) for the solvated start structure."""
    from openmm import app, unit

    sb = cfg["system_build"]
    pdb = read_pdb(repo_path(cfg["adk"]["solvated_pdb"]))
    ff = app.ForceField(*sb["forcefield"])
    kw: dict[str, Any] = {
        "nonbondedMethod": app.PME,
        "nonbondedCutoff": float(sb["nonbonded_cutoff_nm"]) * unit.nanometer,
        "constraints": getattr(app, str(sb["constraints"])),
        "rigidWater": bool(sb.get("rigid_water", True)),
        "ewaldErrorTolerance": float(sb["ewald_error_tolerance"]),
    }
    if sb.get("hydrogen_mass_amu"):
        kw["hydrogenMass"] = float(sb["hydrogen_mass_amu"]) * unit.amu
    if sb.get("switch_distance_nm"):
        kw["switchDistance"] = float(sb["switch_distance_nm"]) * unit.nanometer
    system = ff.createSystem(pdb.topology, **kw)
    return pdb.topology, system, pdb.positions, pdb.topology.getPeriodicBoxVectors()


def cv_groups_for(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Domain-angle atom groups (0-based) for a prepared PDB."""
    return da.group_indices(prepared_universe(path))


def read_colvar(path: str | Path) -> dict[str, np.ndarray]:
    """Read a PLUMED COLVAR file into ``{field: array}``."""
    lines = Path(path).read_text().splitlines()
    fields: list[str] = []
    data = []
    for ln in lines:
        if ln.startswith("#! FIELDS"):
            fields = ln.split()[2:]
        elif ln and not ln.startswith("#"):
            data.append([float(x) for x in ln.split()])
    arr = np.asarray(data, float).reshape(-1, len(fields))
    return {f: arr[:, i] for i, f in enumerate(fields)}


class LastState:
    """OpenMM reporter keeping the positions and box of the last reported step."""

    def __init__(self, interval: int) -> None:
        self._interval = int(interval)
        self.positions: Any = None
        self.box: Any = None

    def describeNextReport(self, simulation: Any) -> dict[str, Any]:
        steps = self._interval - simulation.currentStep % self._interval
        return {"steps": steps, "periodic": False, "include": ["positions"]}

    def report(self, simulation: Any, state: Any) -> None:
        self.positions = state.getPositions(asNumpy=True)
        self.box = state.getPeriodicBoxVectors()


# --------------------------------------------------------------------------- L0


@register(branch="c", system="adk", level="L0", method="boltzmann_endpoints_gbn2_min")
def adk_l0(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Two-state Boltzmann null over the minimised crystal endpoints (GBn2, ff14SB).

    For every endpoint copy (4AKE A/B = open, 1AKE A/B = closed) and every perturbation
    seed, coordinates get Gaussian noise of ``l0.perturb_sigma_nm`` and are minimised in
    GBn2 implicit solvent. E_state = mean minimised potential energy over its copies and
    seeds; dE = E_closed - E_open; p_closed = 1 / (1 + exp(dE / kT)). Mean angles are the
    p-weighted mixture of the minimised structures' angles.

    Uncertainty: bootstrap sd (``l0.n_boot``) over the minimisation replicates within
    each state. It measures how reproducible the null is, not how wrong it is: the model
    ignores conformational entropy and is expected to fail. That is its job as L0.
    """
    from openmm import app, unit

    from mdbench.engines.md import build_simulation, device_name

    p = cfg["l0"]
    md = _md_cfg(cfg)
    kT = KB_KJ_MOL_K * float(md["temperature_K"])
    ff = app.ForceField(*p["forcefield"])
    rng = np.random.default_rng(int(p["seed"]))
    gpu = "none"
    records = []
    for state, copies in protein_endpoints(cfg).items():
        for label, path in copies.items():
            pdb = read_pdb(repo_path(path))
            groups = cv_groups_for(path)
            system = ff.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=getattr(app, str(p["constraints"])),
            )
            x0 = np.array(pdb.positions.value_in_unit(unit.nanometer))
            sim = None
            for k in range(int(p["n_perturb"])):
                x = x0 + rng.normal(0.0, float(p["perturb_sigma_nm"]), x0.shape)
                if sim is None:
                    sim = build_simulation(pdb.topology, system, x * unit.nanometer, md)
                    if sim.context.getPlatform().getName() in ("CUDA", "OpenCL"):
                        gpu = device_name(sim.context)
                else:
                    sim.context.setPositions(x * unit.nanometer)
                sim.minimizeEnergy(
                    tolerance=float(p["minimize_tolerance_kj_mol_nm"])
                    * unit.kilojoule_per_mole
                    / unit.nanometer,
                    maxIterations=int(p["minimize_max_iterations"]),
                )
                st = sim.context.getState(getEnergy=True, getPositions=True)
                e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                xm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                ang = da.domain_angles_from_positions(xm, groups)
                rmsd = float(np.sqrt(np.mean(np.sum((xm - x0) ** 2, axis=1))))
                records.append(
                    {"state": state, "copy": label, "seed_index": k, "energy_kj_mol": e,
                     "rmsd_from_crystal_nm": rmsd, **ang}
                )  # fmt: skip
            del sim

    def estimate(recs: list[dict[str, Any]]) -> dict[str, float]:
        by = {s: [r for r in recs if r["state"] == s] for s in ("open", "closed")}
        e = {s: float(np.mean([r["energy_kj_mol"] for r in v])) for s, v in by.items()}
        de = e["closed"] - e["open"]
        pc = float(1.0 / (1.0 + math.exp(min(de / kT, 700.0))))
        out = {"dE": de, "p_closed": pc}
        for name in ("theta_nmp", "theta_lid"):
            t = {s: float(np.mean([r[name] for r in v])) for s, v in by.items()}
            out[name] = (1 - pc) * t["open"] + pc * t["closed"]
        return out

    point = estimate(records)
    by_state = {s: [r for r in records if r["state"] == s] for s in ("open", "closed")}
    boots = []
    for _ in range(int(p["n_boot"])):
        rs = []
        for v in by_state.values():
            rs.extend(v[i] for i in rng.integers(0, len(v), len(v)))
        boots.append(estimate(rs))
    sd = {k: float(np.std([b[k] for b in boots], ddof=1)) for k in point}
    # The logistic saturates for |dE| >> kT; then the bootstrap sd of p underflows. Use the
    # first-order propagation sd_p = p(1-p) sd_dE / kT, which is exact in that limit.
    pc = point["p_closed"]
    sd_p = max(sd["p_closed"], pc * (1 - pc) * sd["dE"] / kT)
    (run_dir / "l0_minimisations.json").write_text(
        json.dumps({"records": records, "point": point, "bootstrap_sd": sd}, indent=2)
    )
    rows = [
        {"system": "adk_apo_dG_closed_open", "prediction": point["dE"], "uncertainty": sd["dE"]},
        {"system": "adk_apo_p_closed", "prediction": pc, "uncertainty": sd_p},
        {"system": "adk_apo_theta_nmp_mean", "prediction": point["theta_nmp"],
         "uncertainty": sd["theta_nmp"]},
        {"system": "adk_apo_theta_lid_mean", "prediction": point["theta_lid"],
         "uncertainty": sd["theta_lid"]},
    ]  # fmt: skip
    for r in rows:
        r["gpu"] = gpu
    return rows


# --------------------------------------------------------------------------- L1


@register(branch="c", system="adk", level="L1", method="static_cv_crystal")
def adk_l1(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Domain angles of the crystal structures (no sampling, no energy).

    Prediction: mean over the crystallographic copies (chains A and B) of each entry.
    Uncertainty: sqrt(var_copies + var_coord), where var_copies is the sample variance
    between the copies in the asymmetric unit (an empirical measure of how well one
    crystal pins the angle) and var_coord propagates isotropic Gaussian coordinate error
    of ``l1.coordinate_sigma_A`` per Cartesian component by Monte Carlo
    (``l1.n_mc`` samples). There is no statistical sampling uncertainty at L1.

    The apo open crystal (4AKE) is also the single-structure prediction of the apo
    ensemble means (``adk_apo_theta_*_mean``).
    """
    p = cfg["l1"]
    rng = np.random.default_rng(int(p["seed"]))
    sigma_nm = float(p["coordinate_sigma_A"]) / 10.0
    per: dict[str, dict[str, Any]] = {}
    for state, copies in protein_endpoints(cfg).items():
        vals: dict[str, list[float]] = {"theta_nmp": [], "theta_lid": []}
        mc: dict[str, list[float]] = {"theta_nmp": [], "theta_lid": []}
        for path in copies.values():
            u = prepared_universe(path)
            groups = da.group_indices(u)
            x = u.atoms.positions / 10.0
            ang = da.domain_angles_from_positions(x, groups)
            for k in vals:
                vals[k].append(ang[k])
            samples = [
                da.domain_angles_from_positions(x + rng.normal(0, sigma_nm, x.shape), groups)
                for _ in range(int(p["n_mc"]))
            ]
            for k in mc:
                mc[k].append(float(np.std([s[k] for s in samples], ddof=1)))
        per[state] = {}
        for k in vals:
            v = np.array(vals[k])
            var_copies = float(np.var(v, ddof=1)) if len(v) > 1 else 0.0
            var_coord = float(np.mean(np.square(mc[k])))
            per[state][k] = {
                "per_copy": dict(zip(copies, v.tolist(), strict=True)),
                "mean": float(v.mean()),
                "sd_copies": math.sqrt(var_copies),
                "sd_coordinate_error": math.sqrt(var_coord),
                "uncertainty": math.sqrt(var_copies + var_coord),
            }
    (run_dir / "l1_static_cv.json").write_text(json.dumps(per, indent=2))
    rows = []
    for state in ("open", "closed"):
        for k in ("theta_nmp", "theta_lid"):
            d = per[state][k]
            rows.append(
                {
                    "system": f"adk_{state}_{k}",
                    "prediction": d["mean"],
                    "uncertainty": d["uncertainty"],
                }
            )
    apo = str(p["apo_state"])
    for k in ("theta_nmp", "theta_lid"):
        d = per[apo][k]
        rows.append(
            {
                "system": f"adk_apo_{k}_mean",
                "prediction": d["mean"],
                "uncertainty": d["uncertainty"],
            }
        )
    return rows


# --------------------------------------------------------------------------- L2 unbiased MD


def trajectory_angles(top: str | Path, traj: str | Path) -> np.ndarray:
    """(n_frames, 3) time [ps], theta_nmp, theta_lid [deg] from a solute XTC."""
    import MDAnalysis as mda

    return da.domain_angle_timeseries(mda.Universe(str(top), str(traj)))


@register(branch="c", system="adk", level="L2", method="unbiased_md")
@register(branch="c", system="adk", level="L2", method="unbiased_md_smoke")
def adk_l2_unbiased(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Explicit-solvent unbiased MD from one endpoint; histogram of the domain angles.

    Rows: p_closed and the mean angles (``population_rows``: block SEM, Jeffreys floor on
    p). One replicate per config (``replicate`` + ``md.seed``); the ladder aggregates
    replicates. The CV time series is written to ``cv_timeseries.csv`` so it survives
    deletion of the trajectory.
    """
    from mdbench.engines.md import run_md

    top, system, pos, box = build_explicit_system(cfg)
    res = run_md(top, system, pos, cfg["md"], run_dir, box_vectors=box)
    if not res.trajectory:
        raise RuntimeError("unbiased_md needs a trajectory (md.traj_interval_steps > 0)")
    cv = trajectory_angles(res.topology, res.trajectory)
    np.savetxt(run_dir / "cv_timeseries.csv", cv, delimiter=",",
               header="time_ps,theta_nmp_deg,theta_lid_deg", comments="")  # fmt: skip
    an = cfg["analysis"]
    keep = cv[:, 0] >= float(an.get("discard_ps", 0.0))
    rows, diag = population_rows(cv[keep, 1], cv[keep, 2], an)
    xtc = Path(res.trajectory)
    diag.update(
        md_result=json.loads((run_dir / "md_result.json").read_text()),
        xtc_bytes=xtc.stat().st_size,
        n_frames_total=len(cv),
        xtc_bytes_per_frame=xtc.stat().st_size / max(1, len(cv)),
    )
    (run_dir / "analysis.json").write_text(json.dumps(diag, indent=2))
    for r in rows:
        r["gpu"] = res.gpu
    return rows


# --------------------------------------------------------------------------- L3 metadynamics


def metad_script(groups: Mapping[str, Any], p: Mapping[str, Any], temperature_K: float) -> str:
    """PLUMED input: domain angles + well-tempered METAD (+ optional walls) + PRINT."""
    s = da.plumed_domain_angles(groups)
    sig = ",".join(str(math.radians(float(x))) for x in p["sigma_deg"])
    s += (
        f"metad: METAD ARG=theta_nmp,theta_lid SIGMA={sig} HEIGHT={p['height_kj_mol']} "
        f"PACE={p['pace_steps']} BIASFACTOR={p['biasfactor']} TEMP={temperature_K} "
        f"GRID_MIN=0,0 GRID_MAX=pi,pi GRID_BIN={p['grid_bin']},{p['grid_bin']} "
        f"CALC_RCT RCT_USTRIDE={p.get('rct_ustride', 10)} FILE=HILLS\n"
    )
    args = ["theta_nmp", "theta_lid", "metad.bias", "metad.rbias"]
    for i, w in enumerate(p.get("walls") or []):
        lab = f"wall{i}"
        kind = "UPPER_WALLS" if w["kind"] == "upper" else "LOWER_WALLS"
        s += (
            f"{lab}: {kind} ARG={w['cv']} AT={math.radians(float(w['at_deg']))} "
            f"KAPPA={w['kappa_kj_mol_rad2']}\n"
        )
        args.append(f"{lab}.bias")
    s += f"PRINT ARG={','.join(args)} STRIDE={p['print_stride_steps']} FILE=COLVAR\n"
    s += f"FLUSH STRIDE={p['print_stride_steps']}\n"
    return s


@register(branch="c", system="adk", level="L3", method="metad_wt_domain_angles")
def adk_l3_metad(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Well-tempered metadynamics on (theta_NMP, theta_LID), reweighted with c(t).

    Frame weights exp(rbias / kT) with rbias = V(s,t) - c(t) (PLUMED CALC_RCT; Tiwary &
    Parrinello reweighting). Uncertainty: weighted block SEM, Jeffreys floor on p_closed.
    Diagnostic: cumulative (time-resolved) p_closed in analysis.json.
    """
    from mdbench.engines.md import run_md

    top, system, pos, box = build_explicit_system(cfg)
    groups = cv_groups_for(cfg["adk"]["solvated_pdb"])
    md = cfg["md"]
    script = metad_script(groups, cfg["metad"], float(md["temperature_K"]))
    (run_dir / "plumed.dat").write_text(script)
    res = run_md(top, system, pos, md, run_dir, plumed_script=script, box_vectors=box)
    gc.collect()  # finalise the PLUMED object so COLVAR is complete
    col = read_colvar(run_dir / "COLVAR")
    kT = KB_KJ_MOL_K * float(md["temperature_K"])
    an = cfg["analysis"]
    t = col["time"]
    keep = t >= float(an.get("discard_ps", 0.0))
    rb = col["metad.rbias"][keep]
    w = np.exp((rb - rb.max()) / kT)
    nmp, lid = np.degrees(col["theta_nmp"][keep]), np.degrees(col["theta_lid"][keep])
    rows, diag = population_rows(nmp, lid, an, weights=w)
    diag["kish_n_eff"] = float(w.sum() ** 2 / np.sum(w**2))
    (run_dir / "analysis.json").write_text(json.dumps(diag, indent=2))
    for r in rows:
        r["gpu"] = res.gpu
    return rows


# --------------------------------------------------------------------------- L3 umbrella


def umbrella_script(groups: Mapping[str, Any], center_deg: float, p: Mapping[str, Any]) -> str:
    """PLUMED input: domain angles + harmonic RESTRAINT on theta_lid + PRINT."""
    s = da.plumed_domain_angles(groups)
    s += (
        f"res: RESTRAINT ARG={p['cv']} AT={math.radians(center_deg)} "
        f"KAPPA={p['kappa_kj_mol_rad2']}\n"
        f"PRINT ARG=theta_nmp,theta_lid,res.bias STRIDE={p['print_stride_steps']} FILE=COLVAR\n"
        f"FLUSH STRIDE={p['print_stride_steps']}\n"
    )
    return s


def mbar_unbiased_weights(
    samples_rad: Sequence[np.ndarray], centers_rad: Sequence[float], kappa: float, kT: float
) -> tuple[np.ndarray, Any]:
    """MBAR weights of the pooled samples in the unbiased state, plus the MBAR object.

    Harmonic bias u_k(x) = 0.5 * kappa * (x - c_k)^2 / kT (dimensionless).
    """
    import pymbar

    x = np.concatenate([np.asarray(s, float) for s in samples_rad])
    c = np.asarray(centers_rad, float)[:, None]
    u_kn = 0.5 * kappa * (x[None, :] - c) ** 2 / kT
    n_k = np.array([len(s) for s in samples_rad])
    mbar = pymbar.MBAR(u_kn, n_k)
    log_denom = np.log(n_k)[:, None] + mbar.f_k[:, None] - u_kn
    m = log_denom.max(axis=0)
    log_w = -(m + np.log(np.exp(log_denom - m).sum(axis=0)))
    w = np.exp(log_w - log_w.max())
    return w / w.sum(), mbar


@register(branch="c", system="adk", level="L3", method="umbrella_theta_lid")
def adk_l3_umbrella(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Umbrella sampling along theta_LID (theta_NMP free), MBAR reweighting.

    Windows run sequentially in ``w<ii>/``; each starts from the last frame of the
    previous one (``umbrella.chain_starts``) after ``md.equilibration_steps``. Each
    window's series is subsampled by its statistical inefficiency (pymbar). Uncertainty:
    bootstrap over the decorrelated samples within windows (``umbrella.n_boot``).
    Diagnostic: MBAR overlap matrix (minimum neighbour overlap) in analysis.json.
    """
    from pymbar import timeseries

    from mdbench.engines.md import run_md

    top, system0, pos, box = build_explicit_system(cfg)
    groups = cv_groups_for(cfg["adk"]["solvated_pdb"])
    md = _md_cfg(cfg)
    p = cfg["umbrella"]
    kT = KB_KJ_MOL_K * float(md["temperature_K"])
    centers = [float(c) for c in p["centers_deg"]]
    an = cfg["analysis"]
    gpu = "none"
    series = []
    for i, c in enumerate(centers):
        wd = run_dir / f"w{i:02d}"
        script = umbrella_script(groups, c, p)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "plumed.dat").write_text(script)
        last = LastState(int(md["n_steps"]))
        res = run_md(top, copy.deepcopy(system0), pos, md, wd, plumed_script=script,
                     box_vectors=box, extra_reporters=[last])  # fmt: skip
        gpu = res.gpu
        gc.collect()  # finalise the PLUMED object so COLVAR is complete
        if p.get("chain_starts", True) and last.positions is not None:
            pos, box = last.positions, last.box
        col = read_colvar(wd / "COLVAR")
        keep = col["time"] >= float(an.get("discard_ps", 0.0))
        lid, nmp = col["theta_lid"][keep], col["theta_nmp"][keep]
        g = timeseries.statistical_inefficiency(lid) if len(lid) > 2 else 1.0
        idx = timeseries.subsample_correlated_data(lid, g=g)
        series.append({"lid": lid[idx], "nmp": nmp[idx], "g": float(g)})

    def estimate(ser: list[dict[str, np.ndarray]]) -> tuple[dict[str, float], Any]:
        w, mbar = mbar_unbiased_weights(
            [s["lid"] for s in ser], [math.radians(c) for c in centers],
            float(p["kappa_kj_mol_rad2"]), kT,
        )  # fmt: skip
        lid = np.degrees(np.concatenate([s["lid"] for s in ser]))
        nmp = np.degrees(np.concatenate([s["nmp"] for s in ser]))
        ind = closed_indicator(nmp, lid, an["closed"])
        return {
            "p_closed": float(np.sum(w * ind)),
            "theta_nmp": float(np.sum(w * nmp)),
            "theta_lid": float(np.sum(w * lid)),
        }, mbar

    point, mbar = estimate(series)
    rng = np.random.default_rng(int(p.get("seed", 0)))
    boots = []
    for _ in range(int(p["n_boot"])):
        bs = []
        for s in series:
            j = rng.integers(0, len(s["lid"]), len(s["lid"]))
            bs.append({"lid": s["lid"][j], "nmp": s["nmp"][j]})
        boots.append(estimate(bs)[0])
    sd = {k: float(np.std([b[k] for b in boots], ddof=1)) for k in point}
    n_eff = sum(len(s["lid"]) for s in series)
    sd["p_closed"] = max(sd["p_closed"], jeffreys_sd(point["p_closed"] * n_eff, n_eff))
    ov = np.asarray(mbar.compute_overlap()["matrix"])
    diag = {
        "point": point,
        "bootstrap_sd": sd,
        "statistical_inefficiency": [s["g"] for s in series],
        "n_decorrelated": [len(s["lid"]) for s in series],
        "min_neighbour_overlap": float(min(ov[i, i + 1] for i in range(len(centers) - 1)))
        if len(centers) > 1
        else float("nan"),
        "f_k": np.asarray(mbar.f_k).tolist(),
    }
    (run_dir / "analysis.json").write_text(json.dumps(diag, indent=2))
    np.save(run_dir / "overlap_matrix.npy", ov)
    return [
        {"system": "adk_apo_p_closed", "prediction": point["p_closed"],
         "uncertainty": sd["p_closed"], "gpu": gpu},
        {"system": "adk_apo_theta_nmp_mean", "prediction": point["theta_nmp"],
         "uncertainty": sd["theta_nmp"], "gpu": gpu},
        {"system": "adk_apo_theta_lid_mean", "prediction": point["theta_lid"],
         "uncertainty": sd["theta_lid"], "gpu": gpu},
    ]  # fmt: skip


# --------------------------------------------------------------------------- L3 MSM


def msm_features(top: str | Path, traj: str | Path, feat: Mapping[str, Any]) -> np.ndarray:
    """Per-frame features: the two domain angles (rad), optionally CA-CA distances (nm)."""
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import self_distance_array

    u = mda.Universe(str(top), str(traj))
    groups = da.group_indices(u)
    ca = None
    if feat.get("ca_distance_stride"):
        ca = u.select_atoms("protein and name CA")[:: int(feat["ca_distance_stride"])]
    out = []
    for _ in u.trajectory:
        ang = da.domain_angles_from_positions(u.atoms.positions, groups)
        row = [math.radians(ang["theta_nmp"]), math.radians(ang["theta_lid"])]
        if ca is not None:
            row.extend((self_distance_array(ca.positions) / 10.0).tolist())
        out.append(row)
    return np.asarray(out, float)


def build_msm(
    features: Sequence[np.ndarray], angles_deg: Sequence[np.ndarray], m: Mapping[str, Any],
    closed: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """TICA -> k-means -> Bayesian MSM; returns populations, ITS and CK diagnostics."""
    from deeptime.clustering import KMeans
    from deeptime.decomposition import TICA
    from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM

    feats = [np.asarray(f, float) for f in features]
    if m.get("tica_dim"):
        tica = TICA(lagtime=int(m["tica_lag_frames"]), dim=int(m["tica_dim"])).fit(feats)
        tm = tica.fetch_model()
        feats = [tm.transform(f) for f in feats]
    km = KMeans(n_clusters=int(m["n_clusters"]), fixed_seed=int(m.get("seed", 1)))
    kmm = km.fit(np.concatenate(feats)).fetch_model()
    dtrajs = [kmm.transform(f).astype(int) for f in feats]
    ang = np.concatenate(angles_deg)
    dall = np.concatenate(dtrajs)
    ind = closed_indicator(ang[:, 0], ang[:, 1], closed)
    n = int(m["n_clusters"])
    frac_closed = np.array([ind[dall == i].mean() if np.any(dall == i) else 0.0 for i in range(n)])
    mean_ang = np.array(
        [ang[dall == i].mean(axis=0) if np.any(dall == i) else [np.nan, np.nan] for i in range(n)]
    )
    its = {}
    for lag in m["its_lags_frames"]:
        mle = MaximumLikelihoodMSM().fit(dtrajs, lagtime=int(lag)).fetch_model()
        k = min(int(m.get("n_its", 3)), mle.n_states - 1)
        its[int(lag)] = mle.timescales(k).tolist() if k > 0 else []
    lag = int(m["msm_lag_frames"])
    post = BayesianMSM(n_samples=int(m["n_bayes_samples"])).fit(dtrajs, lagtime=lag).fetch_model()
    pc, tn, tl = [], [], []
    for s in post.samples:
        sym = s.count_model.state_symbols
        pi = s.stationary_distribution
        pc.append(float(np.sum(pi * frac_closed[sym])))
        tn.append(float(np.nansum(pi * mean_ang[sym, 0])))
        tl.append(float(np.nansum(pi * mean_ang[sym, 1])))
    ck = None
    try:
        models = [
            MaximumLikelihoodMSM().fit(dtrajs, lagtime=lag * k).fetch_model()
            for k in range(1, int(m.get("ck_steps", 4)) + 1)
        ]
        cktest = models[0].ck_test(models, n_metastable_sets=int(m.get("ck_sets", 2)))
        ck = float(np.max(np.abs(cktest.predictions - cktest.estimates)))
    except Exception as err:  # CK needs enough connectivity; record why it failed
        ck = f"failed: {err}"
    return {
        "p_closed": (float(np.mean(pc)), float(np.std(pc, ddof=1))),
        "theta_nmp": (float(np.mean(tn)), float(np.std(tn, ddof=1))),
        "theta_lid": (float(np.mean(tl)), float(np.std(tl, ddof=1))),
        "implied_timescales_frames": its,
        "ck_max_abs_deviation": ck,
        "n_frames": len(dall),
    }


@register(branch="c", system="adk", level="L3", method="msm_domain_angles")
def adk_l3_msm(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """MSM over existing unbiased trajectories (no new MD).

    ``msm.trajectories`` are glob patterns (repo-relative) of ``traj.xtc`` files, each
    with ``topology.pdb`` beside it (the layout ``unbiased_md`` writes). Uncertainty: sd
    over the Bayesian MSM posterior. Diagnostics: implied timescales vs lag and the
    Chapman-Kolmogorov max deviation, in analysis.json.
    """
    m = cfg["msm"]
    paths = sorted({p for pat in m["trajectories"] for p in glob.glob(str(repo_path(pat)))})
    if len(paths) < int(m.get("min_trajectories", 2)):
        raise RuntimeError(f"MSM needs >= {m.get('min_trajectories', 2)} trajectories, got {paths}")
    feats, angs = [], []
    for tp in paths:
        top = Path(tp).with_name("topology.pdb")
        f = msm_features(top, tp, m.get("features") or {})
        feats.append(f)
        angs.append(np.degrees(f[:, :2]))
    out = build_msm(feats, angs, m, cfg["analysis"]["closed"])
    out["trajectories"] = [_rel(Path(p)) for p in paths]
    (run_dir / "analysis.json").write_text(json.dumps(out, indent=2, default=str))
    return [
        {"system": "adk_apo_p_closed", "prediction": out["p_closed"][0],
         "uncertainty": out["p_closed"][1]},
        {"system": "adk_apo_theta_nmp_mean", "prediction": out["theta_nmp"][0],
         "uncertainty": out["theta_nmp"][1]},
        {"system": "adk_apo_theta_lid_mean", "prediction": out["theta_lid"][0],
         "uncertainty": out["theta_lid"][1]},
    ]  # fmt: skip


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mdbench.branch_c.adk prep <prep.yml>``."""
    import argparse

    ap = argparse.ArgumentParser(prog="mdbench.branch_c.adk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prep", help="prepare AdK structures and write systems/adk/prep.yml")
    pr.add_argument("config")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        logging.basicConfig(level=logging.INFO)
        print(f"[prep] -> {run_prep(a.config)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
