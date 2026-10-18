# Runbook — commands that produced the committed results

These are the commands that produced every row in `results/*.parquet`, checked on
2026-10-18. On that date each row's `config_hash` was reproduced from the current
config of the same name, so rerunning a config gives the same hash.

## Setup

Use micromamba from a non-interactive shell. `make`, `ruff` and `pytest` exist only inside
the environments.

```bash
MM="$HOME/.local/bin/micromamba run -r $HOME/micromamba"
$MM -n mdbench-md make test      # 181 passed, 2 skipped (2026-10-18)
$MM -n mdbench-fe make test      # 56 passed
$MM -n mdbench-md make lint
```

`mdbench-md` runs L0–L2 and branches B and C. `mdbench-fe` runs only the L3 ABFE configs.
For ad-hoc Python outside `make`, set `PYTHONPATH=src`.

## Branch A — binding affinity (`results/a.parquet`, 90 rows)

```bash
# Preparation: run once, emits no rows. Writes systems/{cb7,t4l}/prepared and prep.yml.
$MM -n mdbench-md env PYTHONPATH=src python -m mdbench.branch_a.cb7_prep configs/a_cb7_prep.yml
$MM -n mdbench-md env PYTHONPATH=src python -m mdbench.branch_a.t4l_prep configs/a_t4l_prep.yml

# L0–L2, same pattern for t4l. The _ie configs reuse the L2 frames, so run them after L2.
for c in a_cb7_l0_null a_cb7_l1_vina a_cb7_l2_mmgbsa a_cb7_l2_mmgbsa_ie; do
  $MM -n mdbench-md make run CONFIG=configs/$c.yml; done

# L3 ABFE on CB7 amantadine (g23), the only L3 rows so far
$MM -n mdbench-fe make run CONFIG=configs/a_cb7_l3_abfe_g23.yml        # pilot, 24/21 states, not converged
$MM -n mdbench-fe make run CONFIG=configs/a_cb7_l3_abfe_g23_dense.yml  # 30/27 states, converged, 2.7 h

# Scores and plots
$MM -n mdbench-md make score BRANCH=a    # writes reports/a_scores.csv
$MM -n mdbench-md make report            # writes reports/a_ladder.png and a_ranking_vs_cost.png
```

| config | rows | wall clock |
|---|---|---|
| `a_cb7_l0_null`, `a_t4l_l0_null` | 15, 7 | ~0 s |
| `a_cb7_l1_vina`, `a_t4l_l1_vina` | 15, 7 | 13–19 s per ligand |
| `a_cb7_l2_mmgbsa`, `a_t4l_l2_mmgbsa` | 15, 7 | 23–25 min per ligand (GPU) |
| `a_cb7_l2_mmgbsa_ie`, `a_t4l_l2_mmgbsa_ie` | 15, 7 | minutes total (CPU, no MD) |
| `a_cb7_l3_abfe_g23`, `a_cb7_l3_abfe_g23_dense` | 1, 1 | ~2.7 h each (GPU, ~0.5 GB VRAM) |

### Configured, not yet run (each needs approval, over 6 h)

```bash
# CB7: 14 remaining guests, ~2.7 h each, ~38 h in total
for c in configs/a_cb7_l3_abfe_s*.yml; do $MM -n mdbench-fe make run CONFIG=$c; done
# T4L: 7 ligands, ~15 h each (measured in the 2026-10-18 smoke test), ~105 h in total
for c in configs/a_t4l_l3_abfe_*.yml; do $MM -n mdbench-fe make run CONFIG=$c; done
```

### Test runs that do not touch `results/`

Put a temporary config at `configs/_tmp_<name>.yml`. It has to sit in `configs/`
because `extends` resolves relative to that directory. Set `output.root` in it to a
scratch directory, and run with `MDBENCH_RESULTS_DIR=<scratch>`. Delete the temporary
config afterwards. The T4L smoke test was run this way with a short iteration count.

## Branch B — conformational change, AdK (`results/b.parquet`, 34 rows)

```bash
for c in b_adk_l0_null b_adk_l0_anm b_adk_l1_nma b_adk_l2_pca b_adk_l2_pca_r1 b_adk_l2_pca_r2 b_adk_l2_pooled; do
  $MM -n mdbench-md make run CONFIG=configs/$c.yml; done
$MM -n mdbench-md make score BRANCH=b
```

Not run: `b_adk_l3_pooled.yml`. It pools branch C's 100 ns AdK runs, which have not
been started.

## Branch C — conformational distributions (`results/c.parquet`, 42 rows)

```bash
for c in c_ala2_l0 c_ala2_l1 c_ala2_l2 c_ala2_l3_metad c_ala2_l3_umbrella c_ala2_l3_msm; do
  $MM -n mdbench-md make run CONFIG=configs/$c.yml; done
$MM -n mdbench-md env PYTHONPATH=src python -m mdbench.branch_c.adk prep configs/c_adk_prep.yml
for c in c_adk_l0 c_adk_l1; do $MM -n mdbench-md make run CONFIG=configs/$c.yml; done
$MM -n mdbench-md env PYTHONPATH=src python -m mdbench export-motion \
    runs/c/ala2/L2-md_histogram-r0-b3c7163251e5/rep_2 runs/c/ala2/L3-wtmetad_reweight-r0-0cfdc6ec0a11/rep_<n> \
    -o mdbench-motion.json   # viewer data for mdbench-motion.html
```

Branch C cannot be scored yet: every row has `reference: null`. The AdK L2 and L3
production runs (`c_adk_l2_unbiased*`, `c_adk_l3_*`) are configured but not started
(see `docs/DECISIONS.md`).

## Pages

The repo copy of the overview page is `mdbench-overview.html`, served by GitHub Pages
from `main`. Its embedded `const DATA=` holds branch A's scores, regenerated from
`reports/a_scores.csv`, so update it after rescoring. `mdbench-motion.html` holds the
user's own uncommitted edits.
