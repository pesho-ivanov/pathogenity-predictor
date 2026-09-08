# Generated results

Generated reports, manifests, caches and models live here. External inputs and
Q1's shared VCF partitions belong in the project-root [data/](../../data/) directory.

## Comparison exports

The root [README comparison](../../README.md#method-comparison) presents Q2, Q8,
Q11, Q14 and Q16 in one table, including unevaluated Q8 competitors.
It recomputes AUROC and average precision from
verified validation predictions, with 1,000 component-bootstrap replicates and
seed 42. Each method's own coverage is shown; shared-variant analyses remain in
the notebooks and preserved result records. Invalid/stale inputs, missing results and blocked experiments
are explained in the expandable details. Archived broad-SNV and September pilot
results never enter the current July full-cohort comparison.

`comparison/published/results.json` retains measurements from completed source
notebooks when their ignored prediction exports are absent in a checkout. These
small published records are versioned alongside the notebooks. Each record pins
the executed notebook checksum and the exact output containing its metrics;
cohort matching verifies the VCFs, split membership, labels and DNA checksums.
A regenerated parent protocol is accepted only when these inputs remain identical.
Valid local predictions take precedence. Invalid or partially present local
exports remain explicit errors and never fall back to published scores.

Completed LoRA confirmations also retain the original exploration and full-run
status records as immutable `comparison/published/*-status-*.json` files. Each
status binds its measured duration to an executed notebook checksum; the two
durations reproduce the published combined runtime. Publication replaces only
that experiment's LoRA/control records and preserves competitor and shared-subset
measurements.

`comparison/published/prior-readme.md` preserves the source for previously
published Q2/Q8 shared-subset results and rounded timings. Without the original
per-variant exports, that subset cannot be extended to LoRA. The README labels
the methods' individual coverage and links to the preserved shared comparison;
LoRA is reported on full validation.
Notebook edits invalidate a retained record until its evidence is reverified;
this mechanism preserves completed measurements without executing notebooks or
inventing missing predictions.

`comparison/summary.json` and `methods.csv` are generated alongside
the README section. The summary records source hashes, Q1 identity, coverage, metrics,
intervals and source errors. `watch_status.json` and `watch.log` report the local
refresh service. Notebook saves and relevant result-file changes trigger an atomic
refresh after a short quiet period. The formatter updates the marked main comparison
and Q16 summary, and removes secondary AUROC/AP tables. Data, labels, methods,
settings and surrounding prose are preserved. The README presents one performance
table without a chart of methods on the Y-axis.
The runtime column uses recorded timings only for methods whose cohort and result
checks pass. Its measured stages appear in the expandable details; missing,
invalid or historical timings appear as `—`. Durations use seconds, minutes or
hours and are also exported as `runtime_seconds` and `runtime_scope` in the
comparison summary and CSV. Shared-subset metrics do not imply a separate timed run.
Published-score lookup times exclude upstream model training. Evo2 timings cover
the recorded inference/extraction and fitting stages, with setup and evaluation
excluded where stated; hardware, downloads and cache reuse differ across workflows.
Commit and push the README to publish updates.
Source notebooks are never executed by the watcher; editing displayed numbers or
prose does not change the underlying predictions.
Use `.venv/bin/python -m notebooks.src.readme_comparison` for a one-time refresh
that verifies Q16 alongside the earlier results. The presentation code is separate
from the checksum-pinned experiment implementations; it does not modify their
notebooks, source hashes or published evidence. Historical experiment runners may
write the older layout; run this formatter after publishing their results.

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
  "q1_protocol_sha256": "SHA-256 of results/q1/full/protocol.json",
  "predictions_sha256": "SHA-256 of comparison_predictions.csv",
  "methods": {"score_column": "Method display name"},
  "runtimes": {"score_column": {"seconds": 123.4, "scope": "Inference only; excludes downloads and model loading"}},
  "limitations": "Selection, external training and calibration caveats"
}
```

The watcher discovers these exports automatically and prefers the full Q1 protocol.
`runtimes` is optional and keyed by the same score columns as `methods`; provide
finite, nonnegative measured seconds and a description of the included stages.
For existing 7B exports, the collector follows result checksums to the recorded
scoring or feature manifests without changing the producing experiment.
An invalid or incomplete full dataset never falls back to pilot metrics. The
comparison records the ClinVar date and cohort scope, and watches both full VCFs.
Match the exact Q1 validation
membership and ClinVar labels. Scores must increase with pathogenicity; zero-shot
log likelihood scores need not be probabilities. Each `qN:score_column` ID must be
unique, including the built-in Q2/Q8/Q9 methods. Missing data is never treated as benign.

## Scope of existing results

Q0 and Q1 now use **6 July 2026** ClinVar. Q1 regenerates missense-only full and
pilot partitions from that source, preserving the assignments of variants shared
with the September eligible cohort. New experiments use the full July protocol.

The completed Q2/Q8/Q9/Q10 notebooks and result directories still record their
**5 September 2026 pilot** experiments. Before the input migration, their exact
notebooks, code, protocols, model artifacts and VCF inputs were preserved in
`archive/before_shared_july_snapshot/`. They are historical results and are excluded
from the current comparison. A July evaluation requires a new frozen experiment;
old fitted models or metrics must not be relabeled as July results. The method
artifact descriptions below also document these preserved September runs.

Earlier broad-SNV results remain in `archive/broad_snv_before_missense/`; preceding
Q8 survey and tool runs remain in their existing archive directories.

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

Q1 writes `data/clinvar-train.vcf` (**46,888 variants**) and
`data/clinvar-test.vcf` (**17,927 validation variants**). All
**64,815 eligible July missense SNVs** pass the sequence checks.

The source is **6 July 2026**, shared with Q0. The 70/30 target remains approximate
because variants shared with September preserve their earlier group assignments.
The test-named file includes previously evaluated variants and is development
validation. The README comparison uses this full July cohort and rejects old
pilot metrics. New model experiments require their own fitting and provenance.

## Q1 artifacts: July pilot and grouping cohort

The [pilot implementation](../src/q1.py), used to reconstruct the full workflow's
parent grouping cohort, now also reads July and writes to `notebooks/results/q1/`.
The September pilot implementation and outputs are preserved in
`archive/before_shared_july_snapshot/`.

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
- Optional `split_counts.csv`, `grouping.png`, `split_sizes.png`: pilot summaries
  produced by the pilot display helpers; Q1 now displays the full cohort instead.
- `environment.json`: CPU runtime, package versions, seed and preparation code hash.

The pilot implementation writes `data/clinvar-train-pilot.vcf` and `data/clinvar-test-pilot.vcf`.
The latter is the validation partition; there is no separate test stage.
The archived Q2 1B experiment used the September pilot; the current 7B extension
uses the full July partitions.

## Q2 artifacts: archived 1B prediction

The earlier 1B Q2 workflow wrote to `notebooks/results/q2/`. Its completed
September artifacts now remain under `archive/before_shared_july_snapshot/`:

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

## Q2 artifacts: Evo2 7B zero-shot

[Q2](../Q2-evo2-classifier.ipynb) runs [q2_7b.py](../src/q2_7b.py) and writes to `q2/7b/`:

- `protocol.json`: full Q1 identity, pinned 7B checkpoint and Evo2 source,
  Vortex/FP8 runtime, fixed scoring rules and producing source hashes.
- `preflight.json`: training-only A/B/A repeatability, strand invariance,
  frozen parameters and unchanged FP8 scales. Batch size 1 matches the original
  Q2 scoring; larger training-probe batches changed likelihoods and were not used.
- `scores/*.npz`, `scores/*.json`, `score_manifest.json`: resumable 100-variant
  batches with reference/alternate strand-mean likelihoods, negative differences,
  exact keys, hashes and timings. All 17,927 July validation variants are scored.
- `validation_predictions.csv`, `metrics.json`, `validation_curves.png`:
  full-validation outcomes, AUROC/AP with component-bootstrap intervals, and
  ROC/precision–recall plots. No September outcomes or scores enter this result.
- `inference.log`, `model_load.log`: resumable progress and checkpoint loading.

`q2/comparison_predictions.csv` and `q2/comparison_results.json` export the complete
current validation cohort to the README, binding it to the full split protocol,
predictions and producing code. This extension fits no classifier, scaler,
threshold or hyperparameter. The earlier 1B classifier experiment remains linked
from Q2 as a historical result on its separately archived September pilot.

## Q8 artifacts: missense survey and external predictors

[Q8](../Q8-existing-tools.ipynb) executes [q8_full.py](../src/q8_full.py) on
all **17,927 July validation variants** from the complete frozen Q1 partitions.
It reruns Q1's full eligibility and sequence/group leakage checks, then looks up
published AlphaMissense, REVEL, SIFT4G, PolyPhen-2 HumVar and EVE scores. Clinical
labels are read from the full validation VCF only after lookup. No model fitting,
recalibration, threshold tuning or variant sampling occurs.

Current artifacts live under `q8/full/`. Its root contains the shared frozen
`protocol.json`, copied and verified `leakage_checks.json`, and the overlaid
`validation_curves.png`. AlphaMissense writes its per-method files there; REVEL,
SIFT4G, PolyPhen-2 and EVE write under `revel/`, `sift4g/`, `polyphen2/` and `eve/`:

- `baseline_protocol.json`: tool, pinned release, score definition, full Q1
  protocol/VCF checksums, shared full Q8 protocol hash and producing source hashes.
- `validation_scores.csv`: every full validation key, score or missing status,
  annotation counts and score ranges; no clinical labels.
- `matched_annotations.csv`: every matching source annotation, including raw and
  oriented score entries for dbNSFP tools and transcript/protein mapping fields.
- `validation_predictions.csv`: all validation keys joined to Q1 components and
  ClinVar labels after scoring; the README's verified prediction source.
- `coverage.csv`: scored/missing counts by ClinVar class, keeping the complete
  validation denominator. Missing scores are never imputed as benign.
- `validation_metrics.json`: AUROC and average precision on scored variants,
  with 1,000 component-bootstrap replicates, seed 42 and percentile 95% intervals.
- `baseline_provenance.json`: completion marker written last, binding acquisition,
  source code, Q1 inputs, audit checks and every result checksum. It records the
  measured CPU lookup/evaluation duration; shared dbNSFP acquisition is excluded
  from the three dbNSFP methods' individual timings.

The full dbNSFP extraction is kept separately in
`data/dbnsfp4.9a/full-validation/`, reusing the verified raw `blocks/` cache.
`q8/full/primateai3d/access_status.json` records the licensed-data blocker against
this full cohort. No PrimateAI-3D predictions or substitute PrimateAI scores are made.

The README accepts only complete matching validation exports, preserves missing
scores, and recomputes metrics on the intersection scored by every available
method. Clinical training/calibration and evolutionary-data overlap remain
unresolved; these are development results, not independent clinical validation.

Historical September pilot notebooks, implementations, `pilot_scores.csv`,
`pilot_evaluation.csv`, survey exports and results remain under
`archive/before_shared_july_snapshot/`. The notebook present before this refresh is
also copied to `archive/q8_before_full_validation/`. The dated method survey stays
available in `notebooks/src/q8_catalog.json` and the README methodology table.

To regenerate Q8 automatically, run
`.venv/bin/python -m notebooks.src.refresh_q8` from the repository root. The runner
executes every code cell with a fresh Gamow kernel, retains all outputs/counts,
and replaces the saved notebook only after successful execution.

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

## Q11 artifacts: one-hour partial-epoch Evo2 7B LoRA

[Q11](../Q11-evo2-lora.ipynb) uses the frozen July training pool and full validation,
with a user-authorized partial epoch, and stores its own
protocols, checkpoint conversion and results under `q11/`. Original source
weights are shared under `data/evo2-savanna-7b/`; the pinned BioNeMo runtime is
shared with Q9. Historical Q9/Q10 pilot protocols and results are not consumed.

- `input_checks.json`, `protocol.json`, `environment/`, `checkpoint_source.json`,
  `converted_checkpoint.json`: frozen full inputs, code/configuration identities,
  dependency versions, pinned upstream sources and checksummed conversion.
- `preflight.json`: tensor conversion audit, training-only repeatability,
  strand/gradient/update checks and measured parameter counts/memory.
- `calibration.json`: training-only batched throughput, feature equivalence and
  reserved time for full validation/reporting. The earlier frozen feature batches,
  configuration and source code are preserved under `archive/before_one_hour_*/`.
- `last_checkpoint.pt`: resumable adapter/head weights, exact target paths,
  rank/alpha/initialization settings, FP32 masters, Adam moments, fixed identity scaling,
  shuffle position, RNG states and accumulated history. Saved
  atomically every 32 optimizer steps and at the final partial-epoch boundary; an interruption repeats
  work since the last saved boundary. `final_adapter.pt` omits the optimizer for
  inference. Both contain a SHA-256-prefixed PyTorch payload; use
  `q11_backend.load_checkpoint` to verify and read them. The `q11-lora-v2-partial` format
  rejects earlier whole-block/full-epoch checkpoints and saves no original backbone tensors.
- `training_history.json`, `training_seen.csv`, `metrics.json`,
  `comparison_predictions.csv`, `comparison_results.json`, `validation_curves.png`:
  actual training coverage, stop reason, whole-component intervals, compute and
  complete validation exports. Reload verification uses the fixed first 64 validation
  variants. `run_status.json` includes full notebook execution time and whether it
  finished within one hour.
- `diagnostics/execution-failed-*.ipynb`: actual partial execution with errors
  retained when a run fails; these are diagnostics, not completed notebooks.
  `run_status.json` records failure or completion. Earlier successful notebook
  versions are preserved in `notebook_history/` before replacement.
- `diagnostics/final_block_updates.json`: training-only numerical investigation
  of the failed block-31 gate, including activation scales, gradients, weight
  changes, fixed-head score changes and restoration checks for blocks 31 and 30.
  `archive/block31_preflight/` and `archive/before_lora_*/` preserve the failed
  protocol, sources and diagnostics. Reproducing this historical probe requires
  its archived implementation and configuration.
- `diagnostics/original_numerics/`: fresh traces of all 32 blocks and tail
  submodules, source-to-loaded tensor audit, residual-addition precision replay,
  one synthetic-target step comparing FP32 masters with deployed BF16 weights,
  and frozen/head/restoration checks. The completed
  [original Q11 investigation](../Q11-gradient-diagnostics.ipynb) reruns these
  probes from the archived configuration without reading validation labels.
  Regenerate and execute it with
  `.venv/bin/python -m notebooks.src.q11_numerics --notebook`.
- `diagnostics/lora_quality/`: checkpoint and implementation identity, sampled
  trained/disabled-adapter predictions, reference/difference head contributions,
  raw difference magnitudes, seeded seen/unseen training probes, paired
  component-bootstrap differences and plots. The
  [LoRA performance investigation](../Q11-lora-diagnostics.ipynb) freezes a
  uniform, seed-42 sample of 2,048 validation keys before inference, checks
  their saved predictions against a fresh forward pass, then disables
  adapter outputs while holding the trained head fixed. The smaller diagnostic
  sample was requested to reduce runtime; the full 17,927-variant benchmark
  remains unchanged. Completed training probes are verified and reused from
  `archive/lora_quality_full_attempt_20260907T232136Z/`, which also preserves
  the interrupted full-pass log, partial notebook, protocol and source.
  This intervention is
  not a separately fitted frozen-head baseline. It changes no original Q11
  weights, metrics or cohort. Generate and execute it with
  `.venv/bin/python -m notebooks.src.q11_lora_diagnostics --notebook`.

Run `.venv/bin/python -m notebooks.src.refresh_q11` to generate, execute and save
the notebook from a fresh Gamow kernel. The runner publishes only after all cells
succeed and then refreshes the README. Inputs or sources that differ from an
existing frozen protocol must be archived before a new experiment.

## Q12 artifacts: magnitude-aware LoRA exploration

[Q12](../Q12-lora-improvement.ipynb) uses the completed Q11 input/runtime and
the user-authorized same 24,576 training variants and 2,048-variant validation
sample. It starts from the original backbone and zero-output adapters. Run
`.venv/bin/python -m notebooks.src.refresh_q12` to execute and save the notebook.

- `q12/protocol.json`, `input_checks.json`: fixed membership, Q1/Q11 identities,
  producing source hashes, feature/optimizer settings and leakage checks.
- `preflight.json`, `calibration.json`: training-only numerical/update checks,
  exact restoration, measured throughput and final-validation/report reserve.
- `features/*.npz`, `feature_manifest.json`: resumable, checksummed frozen
  unit-reference/unit-difference vectors and two log magnitude features. The
  three classifier candidates share this extraction; the control branch uses
  the same cached features during continuation.
- `heads.pt`, `head_results.json`: training-only standardization, all three
  20-epoch heads, training losses, sampled validation scores and selected design.
- `last_checkpoint.pt`: adapter/head and control states, both optimizers, FP32
  masters, exact batch cursor and RNG states, saved every 32 matched updates.
  Resume retains the original start/deadline in `execution.json`.
- `best_lora.pt`: best monitored LoRA and its control from the same update count.
  `best_control.pt` retains the strongest control across monitored steps.
  `selected_model.pt` promotes LoRA only for at least +0.005 AUROC over that
  strongest control with no AP decrease. All Q12 tensor files have a SHA-256
  prefix and `q12-magnitude-v1` payload; read them with `q12_backend.load_state`.
- `training_history.json`, `training_membership.csv`, `validation_predictions.csv`,
  `metrics.json`, `validation_curves.png`: matched progress, actual stopping
  point, sampled predictions, component-bootstrap intervals and paired effects.
  Selection and repeated validation make these development results.
- `explore.log`, `run_status.json`, `failed-*.ipynb`: actual execution progress
  and any errors. The runner saves a completed notebook only after all code
  cells execute from a fresh kernel with outputs/counts retained.

Exploration writes no benchmark comparison export. The separate command
`.venv/bin/python -m notebooks.src.refresh_q12 --full-validation` executes and
saves `Q12-lora-validation.ipynb`, evaluates the selected model on all 17,927
variants and writes `q12/full/`. Only after successful notebook execution does
it publish `q12/comparison_predictions.csv` and `comparison_results.json`.
The comparison collector rejects sampled Q12 exports, incomplete full results,
and missing reload verification. Full validation remains development data.

## Q13/Q14 artifacts: controlled adapter experiments

The follow-up workflows retain Q12's exact 24,576 training and 2,048 validation
variants and verify its frozen feature cache before reuse. Run
`.venv/bin/python -m notebooks.src.refresh_q13` for the regularized-head and
learning-rate comparison, followed by
`.venv/bin/python -m notebooks.src.refresh_q14` for adapters on blocks 29 and 30
with the strongest prior frozen classifier held fixed.

[The completed Q14 notebook](../Q14-layer-adapters.ipynb) took **43.0 minutes**
and selected learning rate **1e-4, step 512**. On the **2,048-variant development
sample**, LoRA scored **0.845 AUROC / 0.748 AP**, compared with **0.832 / 0.725**
for the strongest inherited frozen classifier. Paired gains were **+0.0130 AUROC
[0.0050, 0.0206]** and **+0.0234 AP [0.0088, 0.0380]**, using 95% whole-component
bootstrap intervals. The frozen backbone, classifier and scaler remained
unchanged, checkpoint reload checks passed, and the sampled promotion rule
passed. Both learning-rate trials completed 512 updates on the same 16,384
training variants. The sampled result stays separate from the completed
full-validation benchmark below. The reported exploration duration
excludes inherited feature extraction, earlier searches and full confirmation.

- `q13/` and `q14/` each contain a frozen `protocol.json`, input checks,
  parent/checkpoint hashes and an exact `source_snapshot/` of the implementation.
- `heads.pt` and `head_results.json` record the training-only scaler and head.
  Q13 retains every regularization candidate, its actual convergence status and
  FP64-to-FP32 deployment checks. Q14 loads the strongest completed frozen
  control and verifies unchanged classifier weights and scaling throughout.
- Each learning-rate directory retains `best_lora.pt`, `best_control.pt`,
  `last_checkpoint.pt` and `history.json`. Checkpoints include adapter tensors,
  heads, scaler, optimizer masters, RNG and batch cursor. They are inspection
  records; these runners archive and restart rather than resume interrupted runs.
  Use the corresponding backend's `load_state` to verify the SHA-256-prefixed
  `q13-controlled-v1` or `q14-controlled-v1` tensor payload.
- `training_history.json`, `validation_predictions.csv` and `metrics.json`
  retain both matched and strongest frozen controls, sampled component-bootstrap
  intervals, the promotion decision and numerical integrity checks.
- `run_status.json` binds a successful fresh-kernel execution to the saved
  notebook checksum and measured duration. Real failures, partial notebooks,
  logs and prior sources remain in `archive/`; outputs are never fabricated or
  substituted for executing a notebook.

Sampled exploration does not publish benchmark rows. A promoted adapter can
undergo a separate, unfitted confirmation with
`.venv/bin/python -m notebooks.src.refresh_lora_validation --question q13`
(or `q14`). This executes all 17,927 validation variants for LoRA and its frozen
controls, reproduces the original selection predictions and reloads the saved
models. It reports the full cohort, variants outside the selection sample and
components absent from that sample. These remain development results.

Only after saving the fully executed validation notebook does the runner export
both LoRA and its strongest frozen control and refresh the README. Both source
notebooks, full predictions, parent checkpoint and completion records are
checksum-bound. A failed statistical confirmation remains a reportable completed
result; sampled or incomplete results cannot enter the full-cohort comparison.
Runtime separates current exploration and full confirmation from inherited
feature extraction, head fitting and earlier searches. The one-hour target
applies to each exploration, with full confirmation timed separately.

[The completed Q14 full-validation notebook](../Q14-lora-validation.ipynb)
took **34.7 minutes** and passed the prespecified confirmation rule on **17,927
variants**. LoRA scored **0.852 AUROC / 0.770 AP** versus **0.839 / 0.749** for the
strongest frozen classifier, inherited from Q12. Paired gains were **+0.0131 AUROC
[0.0080, 0.0180]** and **+0.0206 AP [0.0123, 0.0287]**. Gains also held outside
the 2,048-variant selection sample. On the 3,246 variants in components absent
from selection, the AUROC and AP gain intervals included zero; this remains
development validation, not an untouched test. Both full-cohort models are
published in the README comparison with verified notebook and artifact evidence.

`q14/full/` retains the frozen confirmation protocol, exact selection membership,
full predictions, subset metrics, paired intervals, checkpoint provenance and
fresh reload checks. The saved notebook reproduces all 2,048 original selection
predictions and passes both 64-variant checkpoint reloads. Exploration plus
full confirmation took **77.7 minutes**; inherited feature extraction, head
fitting and earlier searches are additional. Only the 43.0-minute exploration
was subject to the one-hour budget.

Q15 contingency implementation passed CPU tests; GPU training and replay
equivalence remain unverified. No Q15 experiment has run, and no completed
Q15 notebook or benchmark result is claimed.

## Archived three-way experiment

`archive/three_way/` preserves the preceding train/validation/test experiment's
Q1 data, Q2 model outputs, evaluation lock, source modules and executed notebooks.
It is separate from the current two-way workflow. Its previously evaluated test
variants now belong to validation; they are not an untouched holdout.

## Q0 snapshot transition

`archive/before_q0_july_download/` preserves the earlier Q0 results and notebook,
plus the Q1/Q2 protocols before adding automatic downloads. Q0 now reads the
July snapshot. The later shared-input migration also moved Q1 to July and
preserved the September experiment records in `archive/before_shared_july_snapshot/`.

## Shared July snapshot migration

`archive/before_shared_july_snapshot/` preserves the pre-migration repository
layout for Q0/Q1 and the completed September Q2/Q8/Q9/Q10 experiments, including
four generated VCFs, source modules, notebooks and the former README comparison.
Large immutable Q9/Q10 artifacts are hardlinked locally to avoid copying model
weights; do not edit them in place. `archive_manifest.json` records the source
commit and storage policy. `split_inheritance.json` records assignment inheritance;
`migration_audit.json` records the verified final July membership and snapshot checks.
The uncompressed September `data/clinvar.vcf` is removed after the July rebuild.

<!-- q16-results:start -->
## Q16 artifacts: longer LoRA continuation

[Q16](../Q16-lora-continuation.ipynb) continues Q14 adapters, keeps the classifier/scaler fixed, selects a candidate on the existing 2,048-variant sample, and compares it with Q14 and the frozen classifier on all 17,927 validation variants. Full validation never changes the sample decision.

`q16/` contains the frozen protocol/input checks, training history and exposure records, selected and best-continuation checkpoints, full `validation_predictions.csv`, `metrics.json`, `training_curves.png`, and the original `run_status.json`. Actual failed attempts remain preserved.

`publication_status.json`, written after publication, records the actual end-to-end completion or publication failure against the original deadline. It is separate from the hash-bound notebook completion status and is not inserted into the earlier result artifact registry.

`comparison/published/q16-results.json` retains the full result and displayed provenance with a checksum-pinned executed notebook and byte-for-byte copy of its measured run status. Run `.venv/bin/python -m notebooks.src.readme_comparison` to verify the evidence and include Q16 in the main table. A partial or corrupt local Q16 result is shown as unavailable, never as cached numbers. The frozen experiment publisher is retained to reproduce the original run; current README formatting lives in `src/readme_comparison.py`.

The 2,048-variant sample selects checkpoints; full validation reports the frozen selection. The full partition and its subsets have been evaluated before and are development data, not an untouched test set. Component-bootstrap intervals do not correct repeated selection or establish clinical validity. Pretraining, homology and external-data overlap remain unresolved. Single-variant BF16 scoring is numerically sensitive; Q16 uses its verified batch-32 workflow.
<!-- q16-results:end -->
