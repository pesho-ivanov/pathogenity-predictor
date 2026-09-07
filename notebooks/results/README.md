# Generated results

Files produced by project notebooks live here; external inputs belong in the project-root [data/](../../data/) directory.

## Q0 artifacts

[Q0](../Q0.ipynb) regenerates these Git-ignored files under `notebooks/results/q0/`:

- `summary.json`: input provenance, quality counts, explicit filters, and limitations.
- `cohort.csv.gz`: proposed unique SNVs, source IDs, all usable gene IDs, labels,
  review annotations, and illustrative coordinate groups; **not predictor features**.
- `filter_counts.csv`: sequential removals, including duplicate collapse.
- `inconsistent_variant_keys.csv`: duplicate keys with inconsistent annotations.
- Label, review-status, variant-type, chromosome, consequence, and gene-count CSVs.
- `environment.json`: package versions, Python/platform information, and module hash.

There is no split assignment. Future modeling must freeze groups and splits before
fitting anything, and establish an appropriate untouched evaluation set. Coordinate
groups must be reconsidered with the actual sequence context and all related genes.
