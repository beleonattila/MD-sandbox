# Ground truth

## The rule

**Adopt published reference data; do not curate your own.** Hand-assembling
"validated systems" from the literature is months of work done badly. Every
reference number in this repo comes from a named, citable source, and the row that
uses it records which one.

Every entry in `data/experimental/` carries: value, uncertainty, units, temperature,
buffer/pH where it matters, DOI, and the table or figure it came from. An entry
without a DOI does not go in.

## Sources by branch

| Branch | Reference data | Notes |
|---|---|---|
| A — binding affinity | SAMPL host–guest sets (ITC); T4 lysozyme L99A apolar ligand series; one congeneric series from the Wang/Merck FEP sets (TYK2 is the cheapest entry point) | T4L L99A is the canonical ABFE benchmark; decades of comparison to calibrate against |
| B — conformational change | Adenylate kinase open/closed crystal endpoints; smFRET distance distributions; fast-folder melting temperatures (Trp-cage, chignolin) | Reversible folding is genuinely reachable on this GPU for the fast folders |
| C — distributions | Alanine dipeptide Ramachandran FES (correctness check against published surfaces); ADK domain-angle distributions; NMR/smFRET observables | L0 here is a correctness test of the machinery, not a scientific result |
| D — enzymatic properties | Ketosteroid isomerase active-site electric field vs kcat across mutants (Boxer); chorismate mutase barriers and distal-mutation data | KSI gives a measured linear free-energy relationship — electrostatic preorganisation made directly computable |

## What has no ground truth

**De novo binder design does not belong in this repo as a scored branch.** There is
no ground truth without a wet lab; the only honest comparison is retrospective,
against published hit rates from other people's campaigns. Say so in the README
rather than implying otherwise.

## Leakage

Less acute here than for sequence models — a force field is not trained on the
specific complexes being tested — but it is not zero:

- **MLIPs are trained on datasets.** MACE-OFF is trained on SPICE. Check whether a
  test molecule's chemistry is represented before claiming transferability.
- **Force fields are parameterised against experiment.** Sage 2.3.0 is co-optimised
  to TIP3P and fit to QM plus condensed-phase properties. If a benchmark target was
  in the fitting set, the comparison is circular. Record this where known.
- **Published benchmark systems have been optimised against.** T4L L99A has been used
  to tune methods for twenty years. Good numbers there are weaker evidence than good
  numbers on something less-trodden. Note it; do not pretend otherwise.

## Metrics

Report all three, every time, with the null's score alongside:

- **Spearman ρ** — the primary metric. Ranking is what these methods are mostly used for.
- **Pairwise ranking accuracy** — fraction of pairs ordered correctly. Interpretable
  and robust to a systematic offset.
- **RMSE / MUE** — secondary. Useful for spotting a constant bias, misleading alone.

Bootstrap confidence intervals on all of them; with 20 ligands the error bars are
wide and pretending otherwise is the most common failure in this literature.

**The noise floor is ~0.4 kcal/mol** for binding free energies — roughly the
reproducibility of the experiments themselves. Never report or claim resolution
below it. A method that "achieves 0.3 kcal/mol RMSE" on 20 ligands has fit noise.

## Provenance in every row

`reference` and `reference_source` are mandatory columns. A prediction whose
reference cannot be traced to a DOI is not a benchmark result; it is a number.
