# mdbench

A method-ladder benchmark for molecular simulation. Four scientific questions, each
answered at five levels of rigour, against systems with published experimental data.

The point is not to predict anything new. It is to find out, on one machine with one
set of systems, **how much rigour a given question actually needs** — and to have run
every rung of the ladder personally rather than read about it.

For a visual summary of the branches, the ladder, costs and branch C progress, see the
[overview page](https://beleonattila.github.io/MD-sandbox/mdbench-overview.html)
(source: [`mdbench-overview.html`](mdbench-overview.html)). To watch the branch C
alanine dipeptide runs, open the
[motion page](https://beleonattila.github.io/MD-sandbox/mdbench-motion.html): 10 ns of
unbiased MD beside its Ramachandran path, and the metadynamics bias filling in
(source: [`mdbench-motion.html`](mdbench-motion.html), data regenerated with
`python -m mdbench export-motion`).

## The four branches

| | Question | First system |
|---|---|---|
| **A** | How well does this ligand bind, and is it better than that one? | CB7 host–guest, then T4 lysozyme L99A |
| **B** | How does this protein move? | Adenylate kinase |
| **C** | What conformations does it occupy, and with what probability? | Alanine dipeptide, then adenylate kinase |
| **D** | What makes one enzyme faster than another, against which substrates? | Ketosteroid isomerase, chorismate mutase |

Branch **C is built first** — cheapest, no safety surface, and it exercises the
machinery the other three depend on.

## The ladder

L0 null model · L1 single structure · L2 short MD plus endpoint estimator ·
L3 rigorous classical free energy · L4 beyond the classical force field.

L0 is mandatory and must be beaten. The product of this repo is the sentence, per
branch, naming the level at which the accuracy-versus-cost curve flattens.
Full definition in [`docs/LADDER.md`](docs/LADDER.md).

## What this repo does not do

- **De novo binder design is not a scored branch.** There is no ground truth without
  a wet lab; only retrospective comparison against published hit rates is honest.
- **It does not predict kcat for uncharacterised enzymes.** Branch D reproduces
  published structure–activity relationships. That is the achievable version.
- **It does not forward-predict novel resistance mutations.** TEM-1 work is scoped to
  reproducing published variant effects — see [`docs/SYSTEMS.md`](docs/SYSTEMS.md).

## Layout

```
CLAUDE.md          rules and pins for the coding agent
intent/            one file per branch: problem, scope, done-when
docs/              LADDER, HARDWARE, GROUND_TRUTH, SYSTEMS
systems/           per-system definitions, prep configs, reference values
src/mdbench/       the ladder engines, one module per level
configs/           one YAML per run; nothing result-changing lives in code
runs/              trajectories and logs — gitignored, disposable
results/           tidy parquet rows — the product, always committed
reports/           ladder plots
env/               three environment YAMLs and their lock files
```

## Getting started

```bash
make env
make test
make run CONFIG=configs/c_ala2_l0.yml
```

Hardware assumptions, storage budget and the rented-compute policy are in
[`docs/HARDWARE.md`](docs/HARDWARE.md). Everything is built for a single 6 GB laptop
GPU; the accuracy-versus-cost comparison only means something if the hardware is
held fixed, so any rented run records its GPU on the row.

## Status

Branch C is in progress. Alanine dipeptide has run through L0–L3. Adenylate kinase
is prepared, with L0–L1 run and the L2/L3 production runs configured but not started.
Experimental references are not yet attached, so nothing is scored against experiment.
Branch B is in progress: adenylate kinase has run through L0–L2 (ANM, all-atom normal
modes, 3 × 10 ns MD) and is scored against the crystal open→closed displacement; the
ten softest ANM modes already capture 97 % of it, and none of the three MD replicates
closed. Branches A and D are at the planning stage.
