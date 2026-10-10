# Alanine dipeptide: reference data status

No reference entry yet. Every `ala2.*` row in `results/c.parquet` has `reference` and
`reference_source` set to null.

Why: published ala2 Ramachandran populations depend on the force field, the water
model and how the basins are defined. As of 2026-10-10, no source has been checked
(DOI resolves, number found in the text) that reports basin populations for
ff14SB + TIP3P at 300 K with basins compatible with `configs/c_ala2_base.yml`.
Following `docs/GROUND_TRUTH.md`, an entry without a verified DOI does not go in.

What is reported in the meantime: whether the levels agree with each other. That
includes L2 against L3, metadynamics against umbrella sampling, and the MSM.

Candidates to verify before adding an entry (not yet checked, no numbers taken
from them):
- Force-field comparisons of ala2 basin populations in explicit water done with
  metadynamics (JCTC, about 2011).
- Zhang, Schweitzer-Stenner & Urbanc, JCTC 2020, "Do molecular dynamics force fields
  capture conformational dynamics of alanine in water?". This covers GAG/AAA
  tripeptides, not ala2, so it is at most a loose comparison.
