# Intent: branch C — conformational distributions

- **Author:** Attila Beleon
- **Status:** draft — **build this branch first**
- **Created:** 2026-09-27
- **Ladder:** `docs/LADDER.md` · **Systems:** `docs/SYSTEMS.md`

## Problem

Function is a property of an ensemble, not of a structure. A crystal structure or
an AlphaFold prediction is approximately the mode of a distribution; the states at
5%, 1% and 0.1% occupancy are invisible in that representation and are frequently
the ones that matter — for catalysis, for binding by conformational selection, for
substrate access. Any method that reasons over a single structure is blind to this
by construction. This branch makes the distribution the object being computed.

## What this branch does

Computes free energy surfaces and state populations, and compares them to published
surfaces and to ensemble-averaged experimental observables.

Alanine dipeptide first, as a correctness check — the Ramachandran FES is known, it
takes minutes, and if it comes out wrong nothing above it can be trusted. Then
metadynamics and umbrella sampling on the ADK domain coordinate. Then an MSM over
the ADK trajectory set, sharing trajectories with branch B.

## Why this branch is first

It has no bio-safety surface, the lowest compute cost of the four, and it exercises
the machinery — PLUMED, collective variables, convergence diagnostics, the results
schema — that every other branch depends on. It is also the branch closest to the
argument in `Ensemble_Problem_Briefing`, and the one whose output is most legible to
someone who works on enzyme function.

## Out of scope

- Claiming a converged FES without a convergence diagnostic. Block analysis or
  time-resolved reconstruction, every time.
- Kinetics from an MSM without implied-timescale validation and a Chapman-Kolmogorov
  test.
- Dynamics-versus-electrostatics arguments about catalysis. This branch computes
  populations; interpreting them mechanistically is branch D's problem, and it is
  contested territory — see the note there.

## Done when

1. The alanine dipeptide FES matches published surfaces within the stated tolerance.
2. An ADK domain-coordinate FES exists from both metadynamics and umbrella sampling,
   and the two agree within their error bars — or the disagreement is explained.
3. An MSM with validated implied timescales gives state populations, compared to
   experimental distributions where they exist.
4. Every FES ships with its convergence diagnostic in the same figure.

## Open questions

- How much simulation does the ADK FES actually need before it stops moving? This is
  the number that determines whether the rest of the repo is feasible on this hardware.
- Do metadynamics and umbrella sampling agree here? When they do not, which is wrong
  is usually a CV-choice problem rather than a sampling problem.
