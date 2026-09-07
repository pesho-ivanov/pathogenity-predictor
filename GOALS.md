# Project goals

Build a small, simple, clean project that predicts missense variant pathogenicity
using ClinVar data. A small classifier on frozen Evo2 representations is a
candidate; use the simplest approach justified by validation results.

## Variant scope

- Restrict predictor development, evaluation and research questions to missense
  variants. Keep the current single-nucleotide substitution restriction.
- Use ClinVar's `MC` missense annotation (`SO:0001583`) to define eligibility;
  missing annotations and records without that consequence are excluded.
  Retain all gene associations for leakage checks, including when a variant has
  multiple consequence annotations. Eligibility annotations are not predictor inputs.
- Keep the original ClinVar input intact. Full-file counts may provide context,
  but modeling cohorts and reported performance must refer to missense variants.
- Preserve earlier broad-SNV runs as historical records; do not describe their
  counts or scores as missense-only results. Before new experiments, define and
  freeze missense-only inputs with executable eligibility checks. Keep existing
  variants in their assigned splits and preserve the earlier manifests and results.
- Focus tool comparisons on missense predictors and broad scorers applied to
  missense variants. Standalone splicing and regulatory prediction are out of scope.

## Reproducibility and implementation

- Keep dependencies, abstractions, and model complexity minimal.
- Keep one shared `requirements.txt` at the project root for all notebooks.
- Store external inputs in `data/`. Q1 also writes the shared experiment inputs
  `data/clinvar-train-pilot.vcf` and `data/clinvar-test-pilot.vcf` there. Store other generated
  files under `notebooks/results/`, organized by research question, except the
  small README comparison plot in `assets/comparison.png`.
- All subsequent experiments must use Q1's missense-only, fixed VCF partitions: `clinvar-train-pilot.vcf`
  for training and `clinvar-test-pilot.vcf` for validation. Keep their membership and
  checksums fixed; do not resample or move variants between the files.
- Put reusable implementation in clear Python files with explicit inputs and
  outputs. Make data preparation, feature construction, split assignment,
  training, and evaluation easy to inspect.
- Record data sources and snapshot versions, label definitions and filters,
  reference genome, model/checkpoint versions, dependencies, random seeds, and
  compute requirements.
- Provide an executable workflow covering data acquisition and preparation,
  training or inference, evaluation, and every reported result. No hidden manual
  steps; notebooks must run from a fresh kernel in cell order.

## Validation and leakage prevention

- Prioritize credible evaluation over higher scores. Freeze training and
  validation before fitting any learned preprocessing or model. The current
  workflow has no separate test stage; the filename `clinvar-test-pilot.vcf` denotes
  validation, not an untouched final test set.
- Keep duplicate variants, alternate alleles at the same locus, overlapping
  sequence contexts, and related groups within one split. Keep genes disjoint
  when evaluating generalization to unseen genes.
- Fit learned preprocessing, feature selection, and model parameters on training
  data only. Use validation data for model, hyperparameter, and threshold choices.
  Report these as development results; final performance claims require a separate
  untouched holdout. Recheck group and sequence separation for each experiment,
  especially when changing context length; stop if the fixed split is incompatible.
- Exclude labels and label-derived information from predictor inputs. Audit
  annotation, pretrained-model, and external-data provenance for circularity and
  contamination; document unresolved risks.
- Include executable leakage checks and inspectable split manifests in the
  notebook workflow. Fail on detected overlap or contamination, and explain what
  the checks do and do not establish.

## Results and presentation

- Present explanations and results as Jupyter notebooks (`.ipynb`) in `notebooks/`.
- Keep the aggregate comparison in the root `README.md`. Refresh its marked
  section and plot when source notebooks or result exports change; compare only
  matching frozen cohorts and show missing/blocked methods explicitly.
- Name notebooks with their question number and one or two descriptive keywords,
  separated by hyphens, for example `Q0-clinvar-summary.ipynb`.
- Title research questions in `README.md` as `Q0`, `Q1`, and so on, followed by
  the question text. Each question must correspond to a notebook, linked from
  its README entry and identified by the same question label.
- Briefly describe every notebook in a `README.md` file, with a link to the
  notebook and a short explanation of its purpose.
- Keep notebooks short and low in code: import Python modules and use a small
  number of clear calls to execute the complete workflow. Keep important settings,
  inputs, outputs, and validation checks visible and the implementation traceable.
- Keep notebook implementation and test Python files in `notebooks/src/`.
  Notebook code cells should contain only imports and short calls into those files.
- Save every notebook after all nonempty code cells have executed in order from
  a fresh kernel, with execution counts and outputs retained. Remove unused empty
  cells, resolve errors, and apply this requirement to automatic notebook updates.
- Use brief descriptions and favor informative images and plots.
- Use tables sparingly; preferably hide detailed tables in expandable dropdowns.
- End every notebook with a short conclusion that directly answers its research
  question, supported by the main result and any essential uncertainty or limitation.
- Report evaluation results, uncertainty, and practical limitations concisely.
  Clearly distinguish a reproducible research prototype from a clinically
  validated predictor.

## Completion criteria

A fresh environment can reproduce data preparation, frozen splits, leakage
checks, model fitting or inference, and final evaluation by following the
notebooks in `notebooks/`. Those notebooks explain the approach and present the
results with concise text and useful visuals, backed by small, readable Python
modules.
