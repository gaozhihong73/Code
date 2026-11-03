# Paper plotting

Supplementary figures are generated from the frozen tables under
`outputs/paper_experiments/tables/`; plotting code does not parse training
logs or rerun any model.

Generate the current supplementary set:

```bash
uv run python -B experiments/plotting/plot_supplementary.py --figure all
```

Use `--figure s01`, `s02`, `s03`, or `s04` to rebuild one figure. Each run
exports a figure-specific CSV, JSON metadata, and a 300 dpi PNG. SVG, PDF,
and TIFF exports are deferred until the layout is approved.

To refresh metadata for already-exported supplementary figures without writing
any figure files, use `--metadata-only`. Plotting metadata records provenance
paths and commands only; it does not perform content-digest validation.

PNG-only review drafts for the full-factor E1/E2 views and E3 metric bars are
generated with:

```bash
uv run python -B experiments/plotting/plot_experiment_drafts.py --figure all
```

During review, this entry point writes only PNG plus figure CSV and metadata.
SVG, PDF, and TIFF are intentionally deferred until the layout is approved.
