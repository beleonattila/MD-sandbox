# Intent: branch D — enzymatic properties

- **Author:** Attila Beleon
- **Status:** draft
- **Created:** 2026-09-27
- **Ladder:** `docs/LADDER.md` · **Systems:** `docs/SYSTEMS.md`

## Problem

"Which enzyme is faster, and why" is the question enzyme engineering actually asks,
and it is the least solved of the four. Ranking variants by kcat from structure is
not a solved problem and nothing in this repo will solve it. What *is* achievable is
reproducing published quantitative structure–activity relationships, and thereby
learning exactly which part of the problem the computation can and cannot reach.

## What this branch does

Reproduces two measured relationships rather than predicting new ones:

- **Ketosteroid isomerase** — active-site electric field against kcat across a mutant
  series, where a linear free-energy relationship has been measured. This is
  electrostatic preorganisation made directly computable, and it is the most
  tractable honest result available here.
- **Chorismate mutase** — a unimolecular pericyclic reaction with no covalent
  chemistry, the textbook QM/MM barrier, with published distal-mutation data.

L4 for this branch is one GFN2-xTB QM/MM profile. That is the ceiling on this
hardware and it is enough.

## Out of scope

- Predicting kcat for uncharacterised enzymes. Not achievable; claiming otherwise
  would be the most dishonest thing this repo could do.
- TEM-1 acylation work beyond reproducing published variant effects — see the scope
  limit in `docs/SYSTEMS.md`.
- Reactive ML potentials. Frontier work, not turnkey, and including them in v1
  guarantees v1 never ships.
- DFT-level QM/MM free energies. Single-point corrections only.

## Done when

1. The KSI field-versus-kcat relationship is reproduced, with the computed fields
   correlated against the published measurements and the correlation reported with
   a confidence interval.
2. A chorismate mutase PMF exists at L3 (classical) and L4 (xTB QM/MM), with the
   barrier compared to experiment and the gap explained rather than excused.
3. Where the distal-mutation effects are not reproduced, that is written up as a
   result — with which of the four ensemble mechanisms could account for it, and
   which the method used is structurally unable to see.

## Open questions and one live wire

- Does the classical force field get the KSI field right at all? It has no explicit
  polarisation, and this is precisely where that should hurt.
- **The dynamics-versus-electrostatics argument is contested.** Warshel has argued
  at length that dynamical contributions to catalysis are routinely overclaimed and
  that electrostatic preorganisation does the work. Write results in terms of
  populations and electrostatics; do not assert a dynamical explanation the data
  does not force. Stating "it appears to act through the ensemble — populations and
  electrostatics rather than contacts" is defensible; "it is a dynamics effect" is
  a claim this repo cannot support.
- Ranking distal mutations needs ΔΔG‡ to roughly 1 kcal/mol. That is at or past the
  state of the art. Expect L3 and L4 to both fail at it, and report the failure.
