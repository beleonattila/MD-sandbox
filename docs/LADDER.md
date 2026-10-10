# The ladder

Every branch answers its question at five levels. The levels are not "better and
better" — they are a controlled increase in cost and physical content, run so that
the *shape of the improvement* becomes visible. The interesting result of this repo
is where each curve flattens.

## Levels

**L0 — null model.** Deliberately embarrassing. Costs seconds, uses no simulation,
and must be beaten by everything above it. A surprising fraction of published
methods fail to beat their own null; that is the reason L0 is mandatory.

**L1 — single structure, no sampling.** One conformation, one energy evaluation or
one geometric analysis. Docking scores, static electrostatics, normal modes.
Cheap, interpretable, blind to everything dynamic.

**L2 — short MD plus an endpoint estimator.** Sampling enters, rigour does not.
MM/GBSA, RMSF, contact maps, near-attack-conformer counting. Known to be badly
calibrated for absolute numbers and often adequate for ranking — quantify that here.

**L3 — rigorous classical free energy.** Alchemical RBFE/ABFE with proper λ
schedules and MBAR; metadynamics or umbrella sampling with convergence diagnostics;
MSMs built from many trajectories. The expensive tier, and the one that should
actually work.

**L4 — beyond the classical force field.** ML/MM or QM/MM for the region where
the force field is wrong; ML-potential-corrected free energies. Expect this tier
to be thin for a long time, and expect it to buy less than it costs for most systems.

## Per branch

| Branch | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| **A** binding affinity | ligand heavy-atom count | docking score | MM/GBSA on short MD | ABFE / RBFE with MBAR | ML/MM correction on the ligand |
| **B** conformational change | anisotropic network model | normal modes from one structure | RMSF, contacts, PCA on short MD | long or parallel MD, reversible transitions | MACE-OFF on a small peptide |
| **C** distributions | Boltzmann over crystal endpoints | static CV values | histogram from unbiased MD | metadynamics / umbrella + MSM | ML-potential FES for alanine dipeptide |
| **D** enzymatic properties | kcat of nearest characterised homolog | active-site geometry | NAC populations, field at the bond | PMF along the reaction coordinate | QM/MM barrier with xTB |

## Required outputs at every level

A level is not complete without all four:

1. A prediction with an uncertainty. A number with no error bar is not a result.
2. A wall-clock cost recorded on the machine that produced it.
3. A convergence diagnostic appropriate to the method — block averaging, overlap
   matrices for alchemy, replica round-trips, implied timescales for MSMs.
4. A row in `results/<branch>.parquet` matching the schema in `CLAUDE.md`.

## Scoring

Rank correlation (Spearman) and pairwise ranking accuracy carry more weight than
RMSE. Most of the real use of these methods is "is A better than B", and a method
with a 2 kcal/mol systematic offset but correct ordering is more useful than the
reverse. Report all three, and report the null's score beside every level.

## The stopping rule

For each branch, plot accuracy against log wall-clock across L0–L4. Name the level
at which the curve flattens, and write it down in the branch's intent file. That
sentence — "for this system class, rigour above L*n* did not pay" — is the product
of this repo. A flat step is a finding, not a failure.
