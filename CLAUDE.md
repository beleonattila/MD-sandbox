# CLAUDE.md — mdbench

A method-ladder benchmark. Every scientific question in this repo is answered at
several levels of rigour (L0–L4, see `docs/LADDER.md`) against systems with
published experimental data. The output is not a prediction — it is a comparison
of methods against ground truth, including where extra rigour stops paying.

## Hardware ceiling

Single NVIDIA RTX 4050 Laptop GPU, **6 GB VRAM**, 24 GB system RAM, native Ubuntu.
Before proposing any method, state its VRAM footprint. Anything that needs more
than ~5 GB does not run here. Full-system ML potentials do not fit — ML/MM with a
ligand-sized ML region does.

## Environments

Never install into `base`. Three environments, activated per tier:

```bash
micromamba activate mdbench-md    # L1–L3: openmm, openmm-plumed, ambertools, openff, mdanalysis
micromamba activate mdbench-fe    # L3 alchemy only: openfe and its pinned ecosystem
micromamba activate mdbench-ml    # L4 only: torch, mace-torch, openmm-ml, xtb  (build last)
```

`mdbench-ml` is deferred until L4 work actually starts. Do not add torch to
`mdbench-md` — it drags in a second CUDA runtime and breaks the solve.

## Non-negotiable rules

1. **L0 before L3.** No branch may run an expensive level before its null baseline
   exists and has a recorded score. The null model is the thing to beat.
2. **One tidy row per calculation**, appended to `results/<branch>.parquet`:
   `system, branch, level, method, replicate, prediction, uncertainty, reference,
   reference_source, wall_clock_s, gpu, code_version, config_hash`.
   Engines emit this schema or they are not finished.
3. **`runs/` is disposable, `results/` is not.** Trajectories are regenerable from
   a config; parquet rows are the product. `runs/` is gitignored.
4. **Never quote precision beyond the experimental noise floor** (~0.4 kcal/mol for
   binding free energies). Report uncertainty on every prediction, always.
5. **TEM-1 work is scoped to reproducing published variant effects.** Do not build
   forward prediction or ranking of novel resistance mutations against clinical
   inhibitors. See `intent/A-binding-affinity.md` and `docs/SYSTEMS.md`.
6. **Ask before** starting a run estimated over 6 hours, or any action that spends
   money on rented compute.

## Commands

```bash
make env            # solve and create the environments from env/*.yml
make test           # pytest: energy conservation, schema, round-trips
make run CONFIG=configs/<name>.yml
make score BRANCH=<a|b|c|d>
make report         # ladder plots from results/*.parquet
make clean-runs     # delete trajectories, keep results/
```

## Style

Python 3.11+, ruff for lint and format, pytest. Type hints on public functions.
Configs are YAML; nothing that changes a result is hardcoded in a script.
Every run writes its resolved config next to its output.

## Version pins

Set in `env/*.yml`, `mdbench-md` solved 2026-10-10: OpenMM 8.4 · openmm-plumed 2.1 · PLUMED 2.9
· cuda-version 13.2 · openmm-ml 1.5 · OpenFE 1.12 · OpenFF Sage 2.3.0 · MACE-OFF24(M). Do not
upgrade mid-benchmark — a version change invalidates comparisons. Record `code_version` per row.

OpenMM is 8.4, not 8.5.x, because openmm-plumed 2.1 (newest) is only built for OpenMM 8.4 /
PLUMED 2.9. `cuda-version` must not exceed the driver's CUDA (`nvidia-smi`, currently 13.2):
a newer NVRTC fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The `mdbench-fe` and
`mdbench-ml` pins are not yet solved. Exact builds are in `env/mdbench-md.lock.txt`.
