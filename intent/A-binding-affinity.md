# Intent: branch A — binding affinity

- **Author:** Attila Beleon
- **Status:** in progress — CB7 and T4L L99A L0–L2 run and scored, one CB7 L3 pilot (`src/mdbench/branch_a/`, `configs/a_*.yml`); the L3 campaigns need approval (>6 h)
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

## Progress (2026-10-17)

CB7 (15 guests, benchmarksets Sets 1 and 2, 298 K) and T4L L99A (7 apolar ligands, Morton
1995, 302 K). Sage 2.3.0 (NAGL charges) + ff14SB, TIP3P, 0.05 M NaCl. Rows in
`results/a.parquet`, table in `reports/a_scores.csv`, plot in `reports/a_ranking_vs_cost.png`.
Spearman rho with bootstrap 95 % CI; RMSE in kcal/mol; cost is median wall clock per ligand.

| Series | L0 null (heavy atoms) | L1 Vina | L2 MM/GBSA, 3 × 2 ns (T4L 3 × 1 ns) |
|---|---|---|---|
| CB7, n = 15 | 0.08 [−0.44, 0.64], RMSE 5.0 | 0.53 [−0.04, 0.90], RMSE 7.3 | 0.28 [−0.36, 0.70], RMSE 35.7 |
| CB7 Set 1, n = 8 | −0.50 | 0.71 | 0.21 |
| CB7 Set 2, n = 7 | −0.46 | −0.14 | 0.14 |
| T4L L99A, n = 7 | 0.36 [−0.65, 1.0], RMSE 0.67 | 0.50 [−0.59, 1.0], RMSE 0.68 | 0.75 [−0.18, 1.0], RMSE 17.9 |
| cost per ligand | ~0 s | 13–19 s | 23–25 min |

- **Flattening point (provisional, L0–L2 only):** on CB7, ranking rises from L0 to L1 and
  falls at L2, so the curve flattens at L1; on T4L L99A it keeps rising to L2. No level
  beats the null outside its 95 % CI on any series — with 7–15 ligands the intervals are
  too wide to separate methods. This is the measured answer to the first open question
  for these sets, not a general one.
- **MM/GBSA absolute values are meaningless here** (no entropy term): CB7 overbinds by
  ~30 kcal/mol, T4L by ~18. Only its ranking is compared. Worst CB7 case: g17
  (adamantane-1,3-diamine), −82.6 vs −11.3.
- **Adding an entropy term (`mmgbsa_obc2_ie`, 2026-10-18)** — interaction entropy from the
  same frames (second-order cumulant, chosen before scoring; `configs/a_*_l2_mmgbsa_ie.yml`),
  minutes of CPU. CB7: ρ 0.28 → 0.50 [−0.02, 0.82], RMSE 35.7 → 22.5, but within each set
  ranking falls (Set 1 0.21 → 0.12, Set 2 0.14 → 0.11): the gain is the offset between
  sets, not ranking inside one. The interaction-energy sd is 3–14 kcal/mol, above the
  ~3.6 kcal/mol beyond which the estimator is known to be unreliable (g3: −TΔS up to
  169 kcal/mol). T4L: sd 1.3–1.9 kcal/mol, −TΔS 1.5–3 kcal/mol, ρ unchanged at 0.75.
  This answers the first open question a little further: entropy does not rescue
  MM/GBSA ranking on CB7.
- **The T4L null is hard to beat on RMSE.** The measured range is 2 kcal/mol; heavy-atom
  count and Vina both sit at ~0.7 kcal/mol RMSE, near the noise floor.
- **L3 pilot, CB7 amantadine (g23):** −21.9 ± 0.9 vs −17.19 (4.6 h). **Not converged**:
  neighbour overlap 0.014 at the restraint→electrostatics boundary and 0.03–0.04 through
  electrostatics in both legs, zero replica round trips, halves differ by 1.5–2.1 kT, the
  forward estimate is still drifting. The quoted sd underestimates the error. The
  overbinding is in the known direction for fixed-charge cationic CB7 guests, and the
  experiment's 50 mM sodium acetate (Na⁺ competes at the portals) is not modelled.
  **Convergence check** (`configs/a_cb7_l3_abfe_g23_dense.yml`, 2026-10-18): same sampling,
  electrostatics windows at 0.1 plus 0.95 (complex 30 states, solvent 27), 2.7 h on an
  unshared GPU. −20.6 ± 0.6 vs −17.19. Neighbour overlap ≥ 0.14 everywhere, 3 (solvent)
  and 5 (complex) round trips; solvent halves agree to 0.2 kT, complex halves differ by
  1.7 kT because the first quarter is still relaxing (forward estimate 269.9 → 268.2 →
  267.7 → 267.6 kT). The windows moved the result by 1.4 kcal/mol; the remaining
  3.4 kcal/mol overbinding is not a sampling artefact of the schedule. Use this schedule
  for the campaign; a longer complex leg (or discarding the first quarter) would tighten
  the halves check.
- **Caveats.** CB7 g3 (bis-trimethylammonium adamantane) sits at the portal in all L2
  replicates (COM 0.55–0.73 nm from the host centre); it is scored as simulated.
  g18 is simulated as the neutral acid. Propylbenzene's pose is placed in 186L, not
  181L (clash). packmol placement is not deterministic, so rebuilding `systems/*/prepared`
  gives a different (equivalent) solvent box; the committed files are the ones used.
  Rows carry a `-dirty` code_version (uncommitted code).
- **Storage:** `runs/a` 211 MB (HREX `.nc` deleted after `u_kln.npz` extraction);
  prepared systems 19 MB.
- **Not done:** CB7 L3 for the other 14 guests (~2.7 h each with the denser schedule, ~38 h) and T4L L3
  (~15 h per ligand, `configs/a_t4l_l3_abfe_<ligand>.yml`) — both need approval; L4.
- **Campaign readiness (2026-10-18).** Per-guest L3 configs exist for all 14 remaining CB7
  guests and all 7 T4L ligands, with the dense schedule in the base configs (the first
  pilot pins its old schedule, so its row stays reproducible). A 30-iteration T4L benzene
  smoke test ran both legs, the pocket C-alpha restraint and MBAR end to end, and found one
  more GPU-memory problem: pymbar's JAX backend preallocated 75 % of the card at the first
  leg's analysis and held it (4.7 GB), leaving too little for the 35k-atom complex. MBAR
  now runs on the CPU (`engines/abfe.py`). Scorer bootstraps are seeded per set of
  systems, so a CI no longer moves when unrelated rows are added, and methods on the same
  ligands are resampled identically; the CB7 and T4L CIs above are from that scorer.
- **Open before the L3 campaign is scored:** `cb7_s1_g23` has two L3 rows (the 24-state
  pilot and the converged 30-state run) with the same level and method, and the scorer
  averages rows per system. Whether to exclude the pilot by config hash, give it its own
  method name, or keep the average is a decision about superseded rows, so it is left open.

## Open questions

- Does MM/GBSA (L2) beat docking (L1) on ranking for these sets, or only on absolute
  values? The literature is not unanimous and the answer may be set-dependent.
- Is the L2→L3 step worth 100× the compute on *ranking* specifically, as opposed to
  absolute accuracy? This is the question a working scientist actually faces.
- How much does the T4L cavity-water treatment move the result? Potentially larger
  than the method differences being measured.
