# Hardware, storage and compute policy

## The machine

Intel i7-13650HX · 24 GB RAM · NVIDIA RTX 4050 Laptop, **6 GB VRAM** · 1 TB SSD.
Ubuntu installed natively, dual-booted beside Windows 11 (boot selector on F12).
No WSL, no container — see "Containers" below.

## Expected throughput

Calibrate these on day one with `make bench` and replace this table with measured
numbers. Estimates only:

| Workload | Expectation |
|---|---|
| Classical MD, ~25k atoms | 100–200 ns/day |
| Classical MD, 50–80k atoms | 50–100 ns/day |
| Host–guest ABFE (~3k atoms, 20 λ) | hours per guest |
| Protein–ligand ABFE (~25k atoms) | 1–3 days per ligand |
| RBFE edge, congeneric pair | 12–24 h |
| GFN2-xTB QM/MM umbrella sampling | 1–2 days per profile |
| MACE-OFF as ML region (ligand-sized) | fits in 6 GB |
| Full-system ML potential MD | does not fit — do not attempt |
| DFT QM/MM free energy | not locally; single-point corrections only |

The 6 GB is the binding constraint for anything neural, not for classical MD.
A solvated 50k-atom system is comfortable.

## Storage

The governing formula:

```
bytes/frame = n_atoms × 3 × 4      # float32 coordinates
```

25,000 atoms = 300 KB/frame. At one frame per 10 ps, 100 ns = 3 GB. At one frame
per 100 ps, the same run is 300 MB.

Two settings change total disk need by ~50×, and both default to the frugal option:

1. **Solute-only output.** Strip water unless the analysis needs it. ~10× saving.
   Water is required for solvation analysis and essentially nothing else here.
2. **Sparse output.** 100 ps between frames is sufficient for domain motion, RMSF,
   contacts and MSM featurisation. 10 ps is already oversampled for most analysis.
3. **XTC, not DCD or full-precision NetCDF.** Precision 1000 (0.001 nm) is lossless
   for every analysis in this repo and 3–4× smaller.

Budget for the Linux partition:

| Component | Size |
|---|---|
| Ubuntu + desktop + tools | 30 GB |
| Swap file | 8 GB |
| `mdbench-md` environment | 10 GB |
| `mdbench-fe` environment | 8 GB |
| `mdbench-ml` environment (deferred) | 15 GB |
| conda package cache | 15 GB |
| Reference data and model weights | 15 GB |
| Trajectory working set | 150 GB |
| Headroom (never exceed ~85% full) | 50 GB |
| **Total** | **~300 GB comfortable, 400 GB generous** |

Rough split of the trajectory working set: branch C is the consumer (an MSM wants
many trajectories; 4–10 µs of adenylate kinase, protein-only, is 25–60 GB). Branch A
is cheap **only if** per-λ-window trajectories are deleted once the reduced potential
matrix is extracted — keep them and 20 ligands is 40 GB instead of 2. Branch B's fast
folders are small systems. Branch D is 5–10 GB of umbrella windows.

SSD wear is not a concern: ~100 GB/week is ~5 TB/year against a 300–600 TBW rating.

## Retention policy

- `runs/` is gitignored and disposable. If a run can be regenerated from its config,
  its trajectory may be deleted at any time. The repo is built so this is always true.
- `results/*.parquet` is the product. Kilobytes. Never deleted, always committed.
- `make clean-runs` deletes trajectories older than the current working set.
- Extract and store the reduced potential matrix (or the CV time series) *before*
  deleting an alchemy or enhanced-sampling trajectory.

## Containers

None for now. Three separate micromamba environments give the isolation that matters
here; the real risk is dependency conflict between OpenFE's pins and the torch/CUDA
stack, which a container does not solve. Write the Dockerfile at the first cloud
burst, `FROM mambaorg/micromamba`, built off the already-solved lock files.

## Rented compute

Local first. The only workload that genuinely needs off-machine compute is the
branch A ABFE set.

- **Do not use Colab.** Compute units buy a budget, not reserved hardware, so the
  GPU can change between runs — which destroys the wall-clock column and any
  cost-per-accuracy comparison. The 24-hour ceiling and non-persistent filesystem
  are separately disqualifying for multi-day alchemy.
- **Rent a plain GPU VM by the hour** (Thunder Compute, RunPod, Vast.ai). ssh,
  tmux, persistent volume, checkpoint and resume.
- **Do not pay the datacentre-card premium.** Classical MD tracks memory bandwidth
  and clock, not tensor cores; a rented RTX 4090 often beats an A100 for ns/day at
  a third of the price. Reference points, Sept 2026: RTX A6000 ~$0.35/hr,
  A100 80GB ~$1.09/hr.
- **Budget:** the T4L ABFE set is ~20 ligands × ~12 h on an A6000 ≈ $85. Whole-year
  expectation $150–300.
- Record `gpu` on every row. A rented run and a local run are not interchangeable
  data points for the cost axis.

## HPC

Closed for now: DeiC national calls and LUMI access both require affiliation with a
Danish university. Worth revisiting if an affiliation materialises — that also
unlocks DTU local clusters and the DeiC sandbox, both NVIDIA and both free.

Note that LUMI-G is AMD MI250X, so it would mean a ROCm port of the whole stack.
It is not the prize it looks like for this repo.
