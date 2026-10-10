# Experimental reference data

Adopted, not curated (`docs/GROUND_TRUTH.md`). Every number used as a `reference` in a
results row must come from an entry here, and every entry must be traceable to a
published source. **An entry without a verified DOI does not go in.** If a value cannot
be verified, leave it out and leave `reference` / `reference_source` null on the row; a
wrong DOI is worse than a missing one.

## Layout

One YAML file per system and observable, named `<system>_<observable>.yml`
(e.g. `ala2_ramachandran_fes.yml`). Large tabulated data (a published FES grid, a
distance distribution) goes beside it as CSV, referenced from the YAML by `data_file`.

## Required fields (per entry)

| Field | Meaning |
|---|---|
| `system` | System id as used in configs and results rows (e.g. `ala2`, `adk`) |
| `observable` | What was measured (e.g. `population_alphaR`, `domain_angle_mean`) |
| `value` | The reported number (or `data_file` for a tabulated distribution) |
| `uncertainty` | Reported uncertainty, and `uncertainty_kind` (`sd`, `sem`, `ci95`, `range`, `not_reported`) |
| `units` | Units of `value` and `uncertainty` |
| `temperature_K` | Temperature of the measurement or reference calculation |
| `buffer` / `pH` | Buffer, ionic strength and pH where they matter; `null` with a reason otherwise |
| `doi` | DOI of the source, verified to resolve to the paper the number is taken from |
| `location` | Table or figure (and panel/row) the number was read from |
| `method` | Experimental technique (ITC, smFRET, NMR, ...) or, for a reference simulation, its method and force field |
| `notes` | Anything that limits the comparison: construct, mutant, fitting-set overlap (leakage), digitised from a figure, etc. |
| `added_by` / `verified_on` | Who added it and the date the DOI and value were checked against the source |

`reference_source` on a results row is the entry's `doi` (plus `location` when one paper
supplies several values).

## Example (structure only, not a real value)

```yaml
system: example
observable: example_population
value: 0.0
uncertainty: 0.0
uncertainty_kind: sd
units: fraction
temperature_K: 298.0
buffer: null
pH: null
doi: 10.xxxx/xxxxx        # must be verified; placeholder here
location: Table 1, row 3
method: NMR
notes: placeholder showing the schema only
added_by: name
verified_on: 2026-01-01
```
