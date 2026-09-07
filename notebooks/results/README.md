# Generated results

Generated reports, manifests, caches and models live here. External inputs and
Q1's two shared VCF partitions belong in the project-root [data/](../../data/) directory.

## Comparison exports

The root [comparison.ipynb](../../comparison.ipynb) aggregates Q2, Q8 and Q9 and
lists unevaluated Q8 competitors. It recomputes AUROC and average precision from
verified validation predictions, with 1,000 component-bootstrap replicates and
seed 42. Each method's own coverage is shown; direct comparisons use the intersection
of scored variants. Invalid/stale inputs, missing results and blocked experiments
have explicit statuses. Archived broad-SNV results never enter this comparison.

`comparison/summary.json`, `methods.csv` and `metrics.png` are generated alongside
the root notebook. The summary records source hashes, Q1 identity, coverage, metrics,
intervals and source errors. `watch_status.json` and `watch.log` report the local
refresh service. Notebook saves and relevant result-file changes trigger an atomic
refresh after a short quiet period. Each refresh executes the comparison's display
cell in a fresh kernel and saves its execution count and outputs. `display.json`
supplies the computed display data to that cell. Source notebooks are never executed;
editing displayed numbers or prose does not change the underlying predictions.
Run the root notebook once, or use `python -m notebooks.src.comparison_watch --start`,
to enable it for the workspace session. Use `--stop` to stop it. Reload the comparison
in Jupyter after external updates. A refresh failure is recorded in the watcher log.

Future result notebooks can write `results/qN/comparison_predictions.csv` with
`variant_key`, `label` (0 benign / 1 pathogenic), and a column per method. Include
every frozen validation variant exactly once, using empty scores for missing
coverage; an optional `component` column must match Q1. Export
`results/qN/comparison_results.json` **last**, with this structure:

```json
{
  "q1_protocol_sha256": "SHA-256 of results/q1/protocol.json",
  "predictions_sha256": "SHA-256 of comparison_predictions.csv",
  "methods": {"score_column": "Method display name"},
  "limitations": "Selection, external training and calibration caveats"
}
```

The watcher discovers these exports automatically. Match the exact Q1 validation
membership and ClinVar labels. Scores must increase with pathogenicity; zero-shot
log likelihood scores need not be probabilities. Each `qN:score_column` ID must be
unique, including the built-in Q2/Q8/Q9 methods. Missing data is never treated as benign.

## Scope of existing results

Q1's pilot VCFs and manifests now contain **missense variants only**. Q0 now audits the downloaded **6 July 2026** snapshot. Q1/Q2 preserve the
**5 September 2026** snapshot and assignments used to build the existing pilot.
Q2's missense feature extraction, fitting and validation are complete, and its
notebook is saved with all cells executed and outputs retained. The earlier broad-SNV experiment is
preserved in `archive/broad_snv_before_missense/`.
Q8 now surveys six missense-specific tools and evaluates AlphaMissense on the
fixed pilot. Its earlier broader survey, notebook and implementation are preserved
in `archive/q8_before_missense_baseline/`.

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
- `full_cohort_groups.csv.gz`: full Q0 SNV cohort component and split assignments,
  retaining nonmissense and unsampled bridges, with `MC` annotations for auditing; no labels.
- `split_manifest.csv`: ordered missense pilot variants, genes, source IDs, intervals,
  `MC` annotations, context hashes and frozen splits; no labels.
- `filter_counts.csv`: Q0 filtering stages plus the exact missense consequence filter.
- `sequences.csv.gz`, `sequence_exclusions.csv`: DNA-only feature inputs and exclusions.
- `train_labels.csv`, `validation_labels.csv`: inspectable outcome exports;
  downstream experiments read labels from the canonical VCF files in `data/`.
- `leakage_checks.json`: executable relationship, interval, sequence and missense
  eligibility audits, including preservation of all earlier split assignments.
- `split_counts.csv`, `grouping.png`, `split_sizes.png`: explanatory diagram and split sizes.
- `environment.json`: CPU runtime, package versions, seed and preparation code hash.

Q1 additionally writes `data/clinvar-train-pilot.vcf` and `data/clinvar-test-pilot.vcf`.
The latter is the validation partition; there is no separate test stage.
Q2 consumes these fixed missense partitions and verifies their hashes.

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

## Q8 artifacts: missense survey and AlphaMissense reference

[Q8](../Q8-existing-tools.ipynb) renders the reviewed catalog in
`notebooks/src/q8_catalog.json` offline, then downloads verified AlphaMissense scores
when absent and benchmarks the fixed pilot. It writes to `notebooks/results/q8/`:

- `survey.json`, `tool_comparison.csv`: dated tool comparisons and source URLs.
- `tool_landscape.png`, `baseline_shortlist.png`: missense methods and the selected reference.
- `provenance.json`: catalog/code hashes, dependencies, artifact checksums and
  executable catalog consistency checks.

- `baseline_protocol.json`: pinned score source, exact-allele matching, maximum
  transcript score aggregation and evaluation settings recorded before reading outcomes.
- `pilot_scores.csv`: one row per frozen pilot variant, continuous score or missing
  status, annotation/transcript counts, minimum score and score range; no labels.
- `matched_annotations.csv`: every matching transcript/protein annotation and score.
- `pilot_evaluation.csv`: scores joined to unchanged partitions, components and
  ClinVar outcomes; used for evaluation only, never as predictor input.
- `coverage.csv`, `coverage.png`: scored/missing counts by partition and class.
- `validation_metrics.json`, `validation_curves.png`: AUROC, average precision,
  component-bootstrap 95% intervals (1,000 replicates, seed 42), and ROC/PR curves.
- `baseline_provenance.json`: input/code/output hashes, original archive header,
  licenses, dependencies, compute and rerun Q1 leakage checks.

ClinVar remains the ground truth. AlphaMissense uses ClinVar calibration, whose
exact overlap is unresolved. Metrics apply to covered validation variants and are
development results; neither independent clinical validation nor a tool ranking.
Missing scores are never imputed as benign. No Q2 models are fitted by Q8.

## Archived three-way experiment

`archive/three_way/` preserves the preceding train/validation/test experiment's
Q1 data, Q2 model outputs, evaluation lock, source modules and executed notebooks.
It is separate from the current two-way workflow. Its previously evaluated test
variants now belong to validation; they are not an untouched holdout.

## Q0 snapshot transition

`archive/before_q0_july_download/` preserves the earlier Q0 results and notebook,
plus the Q1/Q2 protocols before adding automatic downloads. Q0 now reads the
July snapshot; Q1/Q2 keep their September input, pilot contents and assignments.
