# Project goals

Build a small, simple, clean project that predicts genetic variant pathogenicity
using ClinVar data. A small classifier on frozen Evo2 representations is a
candidate; use the simplest approach justified by validation results.

## Reproducibility and implementation

- Keep dependencies, abstractions, and model complexity minimal.
- Keep one shared `requirements.txt` at the project root for all notebooks.
- Reserve `data/` for external input data. Store files produced by notebooks
  under `notebooks/results/`, organized by research question (for example, `notebooks/results/q0/`).
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

- Prioritize credible evaluation over higher scores. Freeze train, validation,
  and test splits before fitting any learned preprocessing or model.
- Keep duplicate variants, alternate alleles at the same locus, overlapping
  sequence contexts, and related groups within one split. Keep genes disjoint
  when evaluating generalization to unseen genes.
- Fit learned preprocessing, feature selection, and model parameters on training
  data only. Use validation data for model, hyperparameter, and threshold choices.
  Reserve the test set for final evaluation; if used for development, replace it
  with an untouched holdout.
- Exclude labels and label-derived information from predictor inputs. Audit
  annotation, pretrained-model, and external-data provenance for circularity and
  contamination; document unresolved risks.
- Include executable leakage checks and inspectable split manifests in the
  notebook workflow. Fail on detected overlap or contamination, and explain what
  the checks do and do not establish.

## Results and presentation

- Present explanations and results as Jupyter notebooks (`.ipynb`) in `notebooks/`.
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
- Use brief descriptions and favor informative images and plots.
- Use tables sparingly; preferably hide detailed tables in expandable dropdowns.
- Report evaluation results, uncertainty, and practical limitations concisely.
  Clearly distinguish a reproducible research prototype from a clinically
  validated predictor.

## Completion criteria

A fresh environment can reproduce data preparation, frozen splits, leakage
checks, model fitting or inference, and final evaluation by following the
notebooks in `notebooks/`. Those notebooks explain the approach and present the
results with concise text and useful visuals, backed by small, readable Python
modules.
