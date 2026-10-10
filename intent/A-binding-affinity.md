# Intent: branch A — binding affinity

- **Author:** Attila Beleon
- **Status:** draft
- **Created:** 2026-09-27
- **Ladder:** `docs/LADDER.md` · **Systems:** `docs/SYSTEMS.md`

## Problem

"This ligand binds better than that one" is the single most-used claim in
computational drug discovery, and it is produced by methods spanning six orders of
magnitude in cost. Docking scores are nearly free; ABFE costs a day per ligand.
Published comparisons exist, but reading them is not the same as having run the
ladder yourself on one machine and watched where the accuracy actually comes from.

## What this branch does

Predicts relative and absolute binding free energies for ligand sets with measured
affinities, at every level L0–L4, and reports accuracy against cost.

Order of work: CB7 host–guest first, because the alchemical machinery can be
debugged on a system that finishes in hours rather than days. Then T4L L99A, which
is the real benchmark. A congeneric RBFE series only if the first two land.

## Out of scope

- De novo design of new ligands. There is no ground truth without a wet lab.
- Forward prediction of novel resistance mutations against clinical inhibitors —
  see the scope limit in `docs/SYSTEMS.md`. TEM-1 inhibitor work here reproduces
  published variant effects and nothing further.
- Anything requiring more than 6 GB of VRAM.

## Done when

1. L0 through L3 have run on CB7 and T4L L99A, each with uncertainty, convergence
   diagnostics and wall-clock recorded.
2. Spearman, pairwise ranking accuracy and RMSE are reported for every level, with
   bootstrap confidence intervals and the null's score beside them.
3. The accuracy-versus-cost curve is plotted and the flattening point is named in
   one written sentence.
4. Per-λ trajectories are deleted after the reduced potential matrices are extracted,
   and the branch's disk footprint is under 40 GB.

## Open questions

- Does MM/GBSA (L2) beat docking (L1) on ranking for these sets, or only on absolute
  values? The literature is not unanimous and the answer may be set-dependent.
- Is the L2→L3 step worth 100× the compute on *ranking* specifically, as opposed to
  absolute accuracy? This is the question a working scientist actually faces.
- How much does the T4L cavity-water treatment move the result? Potentially larger
  than the method differences being measured.
