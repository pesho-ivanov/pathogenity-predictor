# Generated results

Generated reports, manifests, caches and models live here. External inputs and
Q1's shared VCF partitions belong in the project-root [data/](../../data/) directory.

## Comparison exports

The root [README comparison](../../README.md#method-comparison) aggregates Q2, Q8, Q9 and Q10 and
lists unevaluated Q8 competitors. It recomputes AUROC and average precision from
verified validation predictions, with 1,000 component-bootstrap replicates and
seed 42. Each method's own coverage is shown; direct comparisons use the intersection
of scored variants. Invalid/stale inputs, missing results and blocked experiments
are explained in the expandable details. Archived broad-SNV results never enter this comparison.

`comparison/summary.json` and `methods.csv` are generated alongside
the README section. The summary records source hashes, Q1 identity, coverage, metrics,
intervals and source errors. `watch_status.json` and `watch.log` report the local
refresh service. Notebook saves and relevant result-file changes trigger an atomic
refresh after a short quiet period. Only the section between `<!-- comparison:start -->`
and `<!-- comparison:end -->` is replaced; surrounding README edits are preserved.
The README presents evaluation tables without a chart of methods on the Y-axis.
Commit and push the README to publish updates.
Source notebooks are never executed by the watcher; editing displayed numbers or
prose does not change the underlying predictions.
Use `.venv/bin/python -m notebooks.src.comparison_watch --start` from the repository
root to enable refresh for the workspace session. Use `--stop` to stop it, or omit
both flags for a one-time refresh. Restart the watcher after a workspace restart.
A refresh failure is recorded in the watcher log.

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
Q8 now surveys six missense-specific tools and evaluates AlphaMissense, REVEL,
SIFT4G, PolyPhen-2 HumVar and EVE on
the fixed pilot. Its earlier broader survey, notebook and implementation are preserved
in `archive/q8_before_missense_baseline/`; the completed AlphaMissense-only notebook,
implementation and results are preserved in `archive/q8_before_revel/`. The completed
two-predictor run is preserved in `archive/q8_before_remaining_tools/`.

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

## Q1 artifacts: full missense partitions

[Q1](../Q1-clinvar-split.ipynb) executes [q1_full.py](../src/q1_full.py) to prepare
all eligible missense variants. Full-dataset artifacts live under `q1/full/`:

- `protocol.json`: parent pilot identity, unchanged filters and assignments,
  sequence rules, code hash, artifact hashes and full VCF checksums.
- `full_cohort_groups.csv.gz`: the broader eligible SNV grouping cohort, including
  any groups joined by identical full-cohort DNA; all split assignments preserved.
- `split_manifest.csv`, `sequences.csv.gz`: every retained missense variant's
  membership and related group, plus DNA-only reference/alternate inputs.
- `filter_counts.csv`, `sequence_exclusions.csv`: complete eligibility and sequence
  accounting with no sampling cap.
- `train_labels.csv`, `validation_labels.csv`: ordered outcomes, independently
  read from the complete exported VCF records and rechecked against Q0's filters.
- `leakage_checks.json`: full-scale overlap checks, complete candidate accounting,
  unchanged earlier assignments and unchanged pilot DNA.
- `split_counts.csv`, `split_sizes.png`, `environment.json`: counts, class balance,
  split figure and CPU/dependency information.

Q1 writes `data/clinvar-train.vcf` (47,230 variants) and `data/clinvar-test.vcf`
(18,040 validation variants). These are the fixed full inputs for new experiments.
The source snapshot is unchanged from the September pilot; all 65,270 eligible
missense SNVs pass the sequence checks. The 70/30 target remains approximate
because whole groups keep their earlier assignments. The test-named file includes
previously evaluated pilot variants and is development validation.

Existing model outputs and the README comparison retain their verified pilot
cohort. Full-dataset model experiments require their own fitting and result
provenance; the old metrics are not relabeled. The earlier Q1 notebook and parent
protocol are preserved in `archive/before_full_missense/`.

## Q1 artifacts: preserved pilot

The preserved [pilot implementation](../src/q1.py), also used to reconstruct the
full workflow's parent cohort, writes to `notebooks/results/q1/`:

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

The pilot implementation writes `data/clinvar-train-pilot.vcf` and `data/clinvar-test-pilot.vcf`.
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
- `validation_predictions.csv`, `validation_report.json`, `validation_curves.png`:
  development results, paired component bootstrap intervals and ROC/PR curves.
  Validation also selects C; the intervals do not correct selection bias.

The notebook uses [q2_display.py](../src/q2_display.py) to omit repetitive cache-progress
messages and render validation results, running the complete frozen Q2 workflows.
It displays numeric intervals and ROC/PR curves without the horizontal method
comparison plot. This keeps `q2.py` and the recorded experiment identities intact.
The earlier notebook and removed plots are preserved in
`archive/before_removing_method_axis_plots/`.

## Q8 artifacts: missense survey and external predictors

[Q8](../Q8-existing-tools.ipynb) renders the reviewed catalog in
`notebooks/src/q8_catalog.json` offline, then acquires pinned published scores
when absent and benchmarks the fixed pilot. The reference AlphaMissense workflow
writes directly to `notebooks/results/q8/`:

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

The additional [REVEL workflow](../src/q8_revel.py) writes the same artifact names
under `notebooks/results/q8/revel/`. `pilot_scores.csv` and `pilot_evaluation.csv`
use the score column `revel`; `matched_annotations.csv` retains all source fields,
including both genome-build coordinates and semicolon-separated transcript IDs.
Only exact `grch38_pos`/REF/ALT matches receive scores. Missing mappings never use
`hg19_pos`; transcript aggregation uses the maximum score fixed before reading labels.
Provenance records the pinned v1.3 archive identity, full scan counts, rows without
GRCh38 positions, Q1 audits, source hashes and the shared evaluation implementation.
The completion manifest is written last. The README comparison verifies REVEL and
AlphaMissense independently and compares their continuous scores on shared variants.
REVEL's HGMD and constituent-tool training overlap with ClinVar remains unresolved.

The [remaining-tool workflow](../src/q8_remaining.py) writes the same evaluation
artifacts under `q8/sift4g/`, `q8/polyphen2/` and `q8/eve/`. Each `score` column
increases with predicted pathogenicity: SIFT4G uses `1 - raw_score`, PolyPhen-2
uses HumVar, and EVE uses continuous scores without uncertainty-category filtering.
Aggregate by maximum across exact GRCh38 allele matches and score-list entries.
`matched_annotations.csv` retains raw and oriented scores, source-row and score-list
indices, and transcript/protein mappings. `pilot_scores.csv` distinguishes missing
tool scores from absent exact-allele mappings; no missing value is imputed.

Each completion manifest binds the results to the pinned dbNSFP4.9a extraction,
acquisition code, scoring code, shared evaluator, Q1 inputs and local leakage
audits. ClinVar fields are excluded from the extracted predictor input. Training
overlap is unresolved for PolyPhen-2; evolutionary sequence exposure and coverage
remain limitations for SIFT4G and EVE. The README independently verifies each tool
and recomputes a common scored validation subset across available methods.

`q8/primateai3d/access_status.json` records the missing licensed data access and
official instructions. It produces no predictions or performance estimate;
the original PrimateAI model is not substituted for PrimateAI-3D.

## Q9 artifacts: Evo2 1B fine-tuning

[Q9](../Q9-evo2-1b.ipynb) writes to `notebooks/results/q9/`:

- `input_checks.json`, `protocol.json`: exact missense partitions, leakage audits,
  settings and source/environment/checkpoint identities, frozen before fitting.
- `environment/`: reproducible setup commands, dependency inventory and execution logs.
- `checkpoint_source.json`, `converted_checkpoint.json`, `base_checkpoint_zarr/`:
  source hashes and the converted BioNeMo checkpoint. Zarr avoids the pinned
  Megatron writer's incompatibility with this machine's PyTorch version.
- `conversion_audit.json`, `parity_nemo.npz`: BioNeMo tensor and forward checks.
  The earlier Vortex comparison is a historical diagnostic in
  `archive/before_bionemo_comparison/`; it does not gate BioNeMo training.
- `preflight.json`, `smoke_adapter.pt`, `readiness.json`:
  gradient/update, frozen-weight and reload checks, resource measurements and
  explicit training blockers. Readiness and supervised training both select Hyena
  block 23; attention block 24 stays frozen. Eight updates with a fixed head check
  the backbone's effect, then a separate step checks the classifier head.
  Adapters record their block index. The smoke adapter is diagnostic, not a fitted predictor.
  `readiness.png` displays the checks before performance results are available.
- `investigation/`: training-only activation/gradient traces, original and
  BF16-adapted checkpoint probes, fixed-head update tests, plots, raw measurements
  and a hash-checked manifest. Probes compare training block 23 with blocks 23–24;
  they use synthetic targets, restore weights, and provide no pathogenicity accuracy estimate.
- Only after all gates pass: `frozen_features.*`, `baseline_selection.json`,
  `training_settings.json`, `history.json`, `best_adapter.pt`, `last_adapter.pt`,
  `validation_predictions.npz`, `metrics.json` and performance plots.

Failed compatibility checks stop model fitting. No performance result is inferred
from a successful installation or forward pass. Completed caches are hash-checked;
changed identities require preserving the earlier run. Interrupted training resumes
from a completed epoch, including FP32 optimizer state and validation-selection history;
an incomplete epoch is repeated with the same seed. Checkpoint progress is saved
atomically. Previous subprocess logs are retained under `environment/logs/`, and a
file lock prevents two notebook runs from training simultaneously.

`resume_migration.json`, when present, documents the reviewed recovery of the earlier
epoch-1 run after adding resume support. It binds the archived protocol and file hashes
to the repaired workflow and rejects changes to model, features, splits or training
settings. The original checkpoints and source files remain in `archive/`.

## Q10 artifacts: frozen Evo2 7B

[Q10](../Q10-evo2-7b.ipynb) writes to `notebooks/results/q10/`:

- `input_checks.json`, `protocol.json`: Q1 membership and leakage audits, fixed
  classifier/feature settings, code hashes and runtime/checkpoint identities.
- `checkpoint_source.json`, `converted_checkpoint.json`, `base_checkpoint_zarr/`:
  hash-pinned Savanna 7B source and its BioNeMo conversion; original weights are
  external inputs under `data/evo2-savanna-7b/`.
- `environment/`: shared BioNeMo dependency inventory, conversion/extraction
  commands and logs; earlier logs are retained on rerun.
- `preflight.json`: every loaded parameter checked against the original source,
  training-only A/B/A repeatability and strand-invariance checks. These checks
  do not establish upstream numerical parity or absence of pretraining overlap.
- `features/*.npz`, `features/*.json`, `feature_manifest.json`: resumable batches
  of 100 variants, DNA-only 8,192-dimensional features, hashes and compute usage.
- `classifier.npz`, `selection.json`: training-only scaler and balanced logistic
  regression, four validation C candidates and the selected model. The 7B fit
  permits 30,000 LBFGS iterations; training-only checks found that C=10 required
  6,553, exceeding Q9's original 3,000-iteration limit.
- `validation_predictions.csv`, `metrics.json`, `validation_curves.png`: paired
  7B/1B development comparisons with component-bootstrap intervals.
- `comparison_predictions.csv`, `comparison_results.json`: the 7B result for
  the README, bound to source code, protocol and result hashes.

The backbone remains frozen. Q9 supplies the verified 1B reference predictions;
its original checkpoint is BF16-sensitive, so a difference cannot be attributed
solely to size. No variants are moved or resampled, and no untouched test is used.

`feature_reuse.json` records the convergence repair. The original feature-producing
sources and protocol remain in `archive/before_convergence_fix/`; cached batches
retain that original identity and their original hashes. Reuse verifies unchanged
DNA, checkpoint, runtime and backbone code, and permits only the recorded classifier
iteration-limit change. It does not relabel old features as newly computed results.

## Archived three-way experiment

`archive/three_way/` preserves the preceding train/validation/test experiment's
Q1 data, Q2 model outputs, evaluation lock, source modules and executed notebooks.
It is separate from the current two-way workflow. Its previously evaluated test
variants now belong to validation; they are not an untouched holdout.

## Q0 snapshot transition

`archive/before_q0_july_download/` preserves the earlier Q0 results and notebook,
plus the Q1/Q2 protocols before adding automatic downloads. Q0 now reads the
July snapshot; Q1/Q2 keep their September input, pilot contents and assignments.
