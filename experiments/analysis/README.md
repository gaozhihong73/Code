# Paper result tables

`build_supplementary_tables.py` constructs the frozen, read-only result layer
for E1, E2, and E3. It does not train models, rerun screening, or overwrite any
source artifact.

Run from the repository root:

```bash
uv run python -B experiments/analysis/build_supplementary_tables.py
```

The script writes unrounded CSV files to
`outputs/paper_experiments/tables/` and a source/output hash manifest to
`outputs/paper_experiments/manifests/supplementary_tables_metadata.json`.

Key rules:

- The independent simulation repetition is the uncertainty unit.
- Errors use denormalized phenotype values and the training-only P95-P5 scale.
- Latent truth is used for formal curve and inflection recovery; observed
  targets are exported separately.
- E2 LD-aware recall uses training-individual dosage correlations only.
- Intermediate E2 screening scores were not saved, so no ranking is inferred.
  `storage_position` in `screened_snps.csv` is explicitly not a rank.
- E3 timing and resource fields are excluded by design. The E1/E2 resource
  table records training-loop time only; screening time and historical peak
  memory remain unavailable rather than being estimated.

The generated tables are the only supported inputs for later manuscript and
supplementary plotting scripts. Plotting code must not parse training logs.
