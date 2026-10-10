# Adenylate kinase: experimental reference data (none adopted yet)

No AdK entry has been added to `data/experimental/`. Every AdK row in `results/c.parquet` therefore
has `reference` and `reference_source` null.

What would qualify, once a DOI and the exact table or figure have been checked against the source:

- apo E. coli AdK closed-state population, or an open/closed equilibrium constant, from NMR or
  smFRET, measured on the wild-type construct at a stated temperature and buffer;
- smFRET distance distributions between labelled sites, compared through a forward model, not
  directly with the domain angles.

The crystal domain angles (L1) are structural descriptors of 4AKE and 1AKE. They are not
ensemble measurements and must not be used as the reference for `adk_apo_*` targets.
