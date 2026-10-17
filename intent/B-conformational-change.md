# Intent: branch B — conformational change

- **Author:** Attila Beleon
- **Status:** in progress — AdK L0–L2 run and scored (`src/mdbench/branch_b/adk.py`, `configs/b_adk_*.yml`); L3 waits on branch C's long runs
- **Created:** 2026-09-27
- **Ladder:** `docs/LADDER.md` · **Systems:** `docs/SYSTEMS.md`

## Problem

A structure is a single frame. Proteins move, and the motions that matter —
domain closure, loop gating, substrate access — happen on timescales that are
awkward relative to what a laptop GPU can reach. Knowing *which* methods recover a
known motion, and how much simulation is enough, is a practical skill that cannot be
acquired by reading.

## What this branch does

Takes systems whose conformational change is independently characterised, and asks
at each level whether the method recovers it: does the predicted motion match the
crystallographic endpoints, and do the derived observables match smFRET or NMR?

Adenylate kinase is the reference case — open and closed structures both solved,
motion well characterised, and small enough to simulate seriously here. Fast folders
(Trp-cage, chignolin) are the second case, because reversible folding is genuinely
reachable on this hardware and the melting temperature is measured.

Animations are a by-product of this branch, not its purpose. They are for
communication; the scored output is the observable comparison.

## Out of scope

- Folding of anything larger than a fast folder. Not reachable here.
- Kinetics requiring Markov state models — that is branch C, which shares the
  trajectories.
- Claiming that an unconverged trajectory "shows" a motion. A single anecdotal
  transition is not evidence; state how many independent runs saw it.

## Done when

1. L0 (anisotropic network model) has produced a predicted motion for ADK and been
   scored against the crystallographic endpoint difference — before any MD is run.
2. Unbiased MD at L2 has been run in multiple independent replicates, and the number
   that saw a transition is recorded honestly.
3. Derived observables are compared to published experimental distributions rather
   than to intuition.
4. Trajectories are stored protein-only in XTC, and the branch fits in 30 GB.

## Progress (2026-10-17)

AdK, open (4AKE) and closed (1AKE) crystal copies A and B. Scores are against the
C-alpha crystal displacement (overlap, reference 1.0) and the crystal C-alpha
B-factors (Pearson r, no reference: r = 1 is not achievable). Rows in
`results/b.parquet`, table in `reports/b_scores.csv`, plot in `reports/b_ladder.png`.

| Score | L0 null | L0 ANM | L1 all-atom NMA | L2 PCA, 3 × 10 ns |
|---|---|---|---|---|
| open→closed, cumulative overlap (10 modes) | 0.12 | 0.97 ± 0.01 | 0.97 ± 0.01 | 0.93 ± 0.01 |
| open→closed, best single mode | 0.08 | 0.79 ± 0.04 | 0.81 ± 0.02 | 0.65 ± 0.03 |
| closed→open, cumulative overlap | 0.12 | 0.74 ± 0.05 | 0.83 ± 0.03 | — |
| closed→open, best single mode | 0.08 | 0.55 | 0.66 ± 0.02 | — |
| open B-factor r | 0.75 | 0.80 ± 0.03 | 0.79 ± 0.04 | 0.79 ± 0.03 |
| closed B-factor r | 0.38 | 0.46 ± 0.13 | 0.30 ± 0.24 | — |
| wall clock | seconds | seconds | 82 min | 3.9 h |

- **The closing motion flattens at L0.** ANM (seconds) and all-atom NMA (82 min) both
  put 97 % of the open→closed displacement in their ten softest modes; 3 × 10 ns of
  explicit-solvent MD puts 93 % in its top ten PCs. Extra rigour buys nothing on this
  score for this system. This answers the first open question for the 10-ns scale only.
- **Direction matters.** Modes of the closed structure describe opening less well than
  modes of the open structure describe closing. All-atom NMA helps there (0.83 vs 0.74).
- **B-factors barely separate methods from the null.** The null (squared distance from
  the centroid) already gets r = 0.75 on the open structure. B-factor agreement is
  not evidence of a correct motion here.
- **Transitions: 0 of 3** open-start 10-ns replicates entered the closed box
  (θNMP < 58.3°, θLID < 125.9°, both at once); fraction 0.00 ± 0.15 (Jeffreys). Closest
  approach was 6.1 Å C-alpha RMSD to 1AKE. Replicate 2 touched each angle bound
  separately (θNMP 61.6° min, θLID 125.3° min) but never both together. No transition
  is claimed.
- **Caveats.** All-atom NMA of 1AKE_B kept one negative mode (−1.1 cm⁻¹) after
  double-precision minimisation (rms gradient ~0.05 kJ/mol/nm); it is kept in the
  score. The closed-B-factor score is unstable because the two closed copies disagree.
  Rows carry a `-dirty` code_version (uncommitted code).
- **Storage:** protein-only XTC, 62 MB for the whole branch so far.
- **Not done:** L3 (`configs/b_adk_l3_pooled.yml` pools branch C's 100-ns runs, which
  are not started: ~12 h each, needs approval); fast folders (Trp-cage, chignolin);
  smFRET/NMR distributions (done-when 3) still need sourced references.

## Open questions

- Does an anisotropic network model — seconds of CPU — capture most of the ADK
  motion that microseconds of MD reproduce? The literature suggests it does, and
  that would be the most useful negative result in the repo.
- How many independent replicates are needed before "we saw the transition" is a
  statement about the system rather than about luck?
