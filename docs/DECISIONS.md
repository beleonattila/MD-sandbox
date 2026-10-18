# Decision log

Decisions already made are listed first, then the questions still open. Development
paused on 2026-10-18 for the user to study the results and do background research.
None of the open questions may be acted on until the user answers it.

## Made

| date | branch | decision | why |
|---|---|---|---|
| 2026-10-17 | A | CB7 first, then T4L L99A. Sage 2.3.0 (NAGL) + ff14SB, TIP3P, 0.05 M NaCl. Each system is parameterised once and reused by every level. | Debug the alchemical machinery on a system that finishes in hours, and keep the parameters fixed across the ladder. |
| 2026-10-17 | A | L0 null is the heavy-atom count, leave-one-out. L1 is Vina rigid-ensemble docking. L2 is single-trajectory MM/GBSA (OBC2, 3 MD replicates). L3 is ABFE with openmmtools HREX + MBAR, a COM restraint, an analytic standard-state term and co-alchemical counterions. | These are the cheapest credible method at each rung. |
| 2026-10-17 | A | Created a separate `mdbench-fe` env (OpenFE 1.12, openmmtools 0.26). Vina 1.2.7 is the only package added to `mdbench-md`. | Keeps the md solve and its pins intact. |
| 2026-10-17 | A | Ran the L3 amantadine pilot (24/21 states). It was not converged (electrostatics overlap ~0.03). | It was a pilot before any campaign. |
| 2026-10-18 | A | Dense electrostatics schedule (steps of 0.1, plus 0.95; complex 30 states, solvent 27) is the default. The pilot config keeps its old schedule pinned. | It converged amantadine (overlap ≥ 0.14, 3/5 round trips), giving −20.6 ± 0.6 against −17.19 measured. The remaining overbinding is not a schedule artefact. |
| 2026-10-18 | A | The openmmtools energy context cache is capped at one context and emptied after each leg. | Under NPT, `minimize` built one GPU context per state and ran out of memory. |
| 2026-10-18 | A | pymbar's JAX backend is forced onto the CPU (`JAX_PLATFORMS=cpu`, `engines/abfe.py`). | It preallocated 4.7 GB of VRAM, which broke the T4L complex leg. The peak is now 0.6 GB. |
| 2026-10-18 | A | Scorer bootstrap is seeded per set of systems. | A CI no longer moves when unrelated rows are added, and methods on the same ligands share resamples. CIs shifted by ≤ 0.09. |
| 2026-10-18 | A | MM/GBSA entropy variant `mmgbsa_obc2_ie` uses the second-order cumulant, chosen before scoring. The exponential form is recorded too. | Fixing the choice before scoring avoids estimator shopping. Result: CB7 ρ goes from 0.28 to 0.50, but only through set separation, and the fluctuations are beyond the reliable range. T4L is unchanged. |
| 2026-10-18 | A | The CB7 and T4L L3 campaigns are not started. | Both exceed 6 h, and the user paused to research. |

Branch B and C decisions are recorded in their intent files and commit messages
(24edc79, 4f52958, 756bbe0).

## Open — the user's call

### Branch A
1. **CB7 L3 campaign.** Run the 14 remaining guests at about 38 h, or with a longer
   complex leg at 50–55 h?
2. **T4L L3.** About 15 h per ligand, roughly 105 h for all seven ligands. Run it, run
   a subset, or skip it?
3. **The superseded amantadine pilot row.** `cb7_s1_g23` has two L3 rows (config
   hashes 515dab492b02 for the pilot and ce71233791aa for the dense run), and the
   scorer averages them. Exclude the pilot by config hash, give it its own method
   name, or keep the average? This must be settled before L3 is scored.
4. **The CB7 overbinding.** Amantadine overbinds by about 3.4 kcal/mol even when
   converged. Is that the force field (a known CB7 issue in the literature), and does
   it change the plan? This is background research, not a run.

### Branch B
5. **AdK L3 (`b_adk_l3_pooled.yml`).** It depends on branch C's 100 ns runs, about
   12 h each, which need approval.

### Branch C (deferred since 2026-10-10)
6. Does the 0.4 kcal/mol noise floor apply to conformational dG?
7. Should the results schema gain `observable` and `units` columns?
8. May rows in `results/c.parquet` be replaced? MSM uncertainty is about 3× too small,
   and the φ-only umbrella runs are not converged for αR.
9. A DOI-verified ala2 reference is needed before branch C can be scored.
10. AdK production runs: L2 about 3.3 days, metad about 2 days, umbrella about 2.2 days,
    MSM about 26 days.
