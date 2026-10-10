# Systems

Chosen for overlap: a system that serves several branches is worth more than a
system that serves one, because the preparation is reused and the comparison
across branches is controlled.

| System | A | B | C | D | Size | Why |
|---|:-:|:-:|:-:|:-:|---|---|
| TEM-1 β-lactamase | ✓ | ✓ | ✓ | ✓ | ~260 res | Serves all four. Inhibitor binding, Ω-loop distal variants, acylation chemistry, a very large published mutational record |
| T4 lysozyme L99A | ✓ | ✓ | ✓ | — | 164 res | The ABFE benchmark. The engineered cavity also breathes, giving a B/C system for free |
| Adenylate kinase | — | ✓ | ✓ | ~ | 214 res | The domain-motion reference. Its rate-limiting step *is* the conformational change — the cleanest demonstration that an enzymatic property need not be chemistry |
| Alanine dipeptide | — | — | ✓ | — | 22 atoms | Correctness check. If the Ramachandran FES is wrong, everything above it is wrong |
| CB7 host–guest | ✓ | — | — | — | ~3k atoms | ABFE that finishes in hours, so the alchemy machinery can be debugged before it costs days |

## Structures

Verify every PDB ID against the PDB before use — the ones marked VERIFY are from
memory and must be checked:

- **TEM-1**: 1BTL (apo, 1.8 Å).
- **Adenylate kinase**: 4AKE (open, apo), 1AKE (closed, with Ap5A).
- **T4 lysozyme L99A**: 181L (with benzene). Apo L99A — VERIFY.
- **Ketosteroid isomerase** (branch D): *P. putida* KSI — VERIFY the ID and the
  specific Boxer-series construct.
- **Chorismate mutase** (branch D): *B. subtilis* — VERIFY (2CHT is the likely ID).
- **Alanine dipeptide**: built, not downloaded.
- **CB7**: from the SAMPL challenge inputs, not the PDB.

## Preparation

Preparation is a method choice and must be recorded, not buried. Every system's
`prep.yml` states protonation assignment and pH, the protonation tool and version,
which crystallographic waters were kept, box padding and ion concentration, and any
missing loop that was modelled and how.

Two that bite in this set specifically:

- **TEM-1 E166 and the Ω-loop** — protonation of the catalytic glutamate changes the
  mechanism you are simulating. Choose explicitly and document the choice.
- **T4L L99A cavity waters** — whether the engineered cavity is treated as dry or
  hydrated changes binding free energies materially. State it; consider running both.

## Scope limit on TEM-1

TEM-1 stays in the repo, and its scope is **reproduction of published variant
effects and inhibitor binding**. The repo does not build forward prediction or
ranking of novel resistance mutations against clinical inhibitors. The scientific
value here is method validation against a well-characterised system, and that value
is entirely retrospective. Intent files and configs should reflect that in their
wording, because the wording is what a reader — human or model — goes on.
