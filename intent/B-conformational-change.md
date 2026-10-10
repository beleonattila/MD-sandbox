# Intent: branch B — conformational change

- **Author:** Attila Beleon
- **Status:** draft
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

## Open questions

- Does an anisotropic network model — seconds of CPU — capture most of the ADK
  motion that microseconds of MD reproduce? The literature suggests it does, and
  that would be the most useful negative result in the repo.
- How many independent replicates are needed before "we saw the transition" is a
  statement about the system rather than about luck?
