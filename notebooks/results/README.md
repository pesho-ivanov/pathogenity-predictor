# Generated results

Generated reports, manifests, caches and models live here. External inputs and
Q1's two shared VCF partitions belong in the project-root [data/](../../data/) directory.

## Scope of existing results

The project now targets **missense variants only**. The current Q0–Q2 outputs and
pilot VCFs retain the earlier broad-SNV cohort; their counts and scores are not
missense-only results. Their code and frozen artifacts have not been rebuilt.
Q8 retains its earlier broader survey as background; splicing and regulatory
experiments are outside the current scope. New experiments require frozen
missense-only inputs and corresponding baseline results.

## Q0 artifacts

[Q0](../Q0-clinvar-summary.ipynb) regenerates these Git-ignored files under `notebooks/results/q0/`:

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

## Q1 artifacts: splitting

[Q1](../Q1-clinvar-split.ipynb) writes Git-ignored data preparation outputs to `notebooks/results/q1/`:

- `protocol.json`: pinned inputs, sampling/grouping rules, split implementation
  hash, artifact checksums and hashes of both shared VCFs. Frozen before fitting.
- `full_cohort_groups.csv.gz`: full eligible-cohort component and split assignments,
  retaining bridges through variants outside the pilot; no labels.
- `split_manifest.csv`: ordered pilot variants, genes, source IDs, intervals,
  context hashes and frozen splits; no labels.
- `sequences.csv.gz`, `sequence_exclusions.csv`: DNA-only feature inputs and exclusions.
- `train_labels.csv`, `validation_labels.csv`: inspectable outcome exports;
  downstream experiments read labels from the canonical VCF files in `data/`.
- `leakage_checks.json`: executable relationship, interval and sequence audits.
- `split_counts.csv`, `grouping.png`, `split_sizes.png`: explanatory diagram and split sizes.
- `environment.json`: CPU runtime, package versions, seed and preparation code hash.

Q1 additionally writes `data/clinvar-train-pilot.vcf` and `data/clinvar-test-pilot.vcf`.
The latter is the validation partition; there is no separate test stage.
The recorded Q2 experiment consumes these fixed partitions and verifies their hashes.

## Q2 artifacts: prediction

[Q2](../Q2-evo2-classifier.ipynb) writes model outputs to `notebooks/results/q2/`:

- `protocol.json`: model/checkpoint, feature construction, selection and evaluation
  settings, with hashes of the Q1 data artifacts it consumes.
- `features/<fingerprint>/*.npz`, `feature_manifest.json`: resumable DNA-only features,
  zero-shot scores and cache hashes bound to the producing experiment.
- `benchmark.json`, `compute.json`, `model_load.log`: synthetic order/FP8 scaling
  checks, compute usage, environment and checkpoint loading details.
- `*_classifier.npz`, `selection.json`: scaler means/scales, coefficients,
  intercepts, validation scores and selected C; fitting uses training data only.
- `validation_report_started.json`: binds the validation report to its experiment.
- `validation_predictions.csv`, `validation_report.json`, `validation_curves.png`,
  `auroc_intervals.png`: development results, paired component bootstrap intervals
  and plots. Validation also selects C; the intervals do not correct selection bias.

## Q8 artifacts: tool survey

[Q8](../Q8-tools-survey.ipynb) renders the reviewed catalog in
`notebooks/src/q8_catalog.json` offline and writes to `notebooks/results/q8/`:

- `survey.json`, `tool_comparison.csv`: dated tool comparisons and source URLs.
- `tool_landscape.png`, `baseline_shortlist.png`: documented targets and proposed baselines.
- `provenance.json`: catalog/code hashes, dependencies, artifact checksums and
  executable catalog consistency checks.

This is an authored literature survey. It does not access the ClinVar partitions,
run predictors or measure accuracy. Any later benchmark must use Q1's missense-only fixed VCFs
and audit the selected tools' actual training provenance and sequence contexts.

## Archived three-way experiment

`archive/three_way/` preserves the preceding train/validation/test experiment's
Q1 data, Q2 model outputs, evaluation lock, source modules and executed notebooks.
It is separate from the current two-way workflow. Its previously evaluated test
variants now belong to validation; they are not an untouched holdout.
