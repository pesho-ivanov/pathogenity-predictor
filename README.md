# Gamow predictor

A small research project for predicting genetic variant pathogenicity from
ClinVar data. See [GOALS.md](GOALS.md) for the project requirements.

The project is at the setup stage. No ClinVar predictor has been trained or
validated yet.

Clone with the Evo2 source included:

```bash
git clone --recurse-submodules git@github.com:pesho-ivanov/gamow-predictor.git
```

## Research questions

These questions support the [project goals](GOALS.md): ClinVar pathogenicity
prediction, potentially using frozen Evo2 representations. Evo2 already has
published variant-effect results, so the strongest contribution would address
generalization, reliability, and when its representations add value.
([Evo2 paper](https://www.nature.com/articles/s41586-026-10176-5))

### Q0. What does the ClinVar dataset contain, and which variants are suitable for reliable evaluation?

Explore variant types, genes, pathogenicity labels, review status, missing
annotations, conflicting classifications, and duplicate or related records.
Use plots to reveal class imbalance and potential leakage, then define an
initial dataset and filtering criteria.

### Q1. Can a small classifier on frozen Evo2 representations outperform zero-shot Evo2 scoring on previously unseen genes?

Compare logistic regression on reference/alternate representations with zero-shot
scores and simple sequence or consequence baselines, using gene-disjoint
evaluation. This directly tests whether a lightweight supervised model adds
transferable information.

### Q2. How much does apparent predictive performance depend on similarities between training and test data?

Compare conventional random splits with progressively stricter locus-,
sequence-context-, and gene-separated splits. Treat random splits as a diagnostic
benchmark; quantify how much performance survives credible leakage controls.

### Q3. Does longer sequence context improve pathogenicity prediction, and for which variant classes?

Vary context length while holding the checkpoint, classifier, and evaluation
variants fixed. Compare missense, splice-associated, and noncoding variants where
sample sizes permit, measuring both predictive gains and computational cost.

### Q4. Is it better to train on fewer strongly supported ClinVar labels or more labels with weaker supporting evidence?

Compare training sets filtered by review status, including size-matched
comparisons, against a fixed, strongly reviewed holdout. ClinVar's review status
captures review processes and agreement, making this a useful test of label
selection.
([ClinVar documentation](https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/))

### Q5. Can the predictor identify when its own predictions are unreliable?

Evaluate calibration and whether withholding low-confidence predictions reduces
errors on unseen genes and different variant classes. Report the relationship
between retained coverage and error rate, alongside discrimination metrics.

### Q6. Can a model trained on an older ClinVar snapshot predict subsequently resolved variants of uncertain significance?

Freeze training labels at an earlier release and evaluate variants that later
receive clear classifications. This requires historical snapshots and careful
provenance checks, but would test usefulness beyond reproducing existing labels.

### Suggested priorities

Start with question **Q0** to understand the data and define the initial cohort.
Then prioritize questions **Q1 and Q2**, with **Q3–Q5** as focused supporting experiments.
Question **Q6** is a valuable extension once the core workflow and historical data
are available.

## Notebooks

`notebooks/` is reserved for project notebooks. Explanations and results belong
in `results/`; briefly document new notebooks here.

## Data

The local ClinVar VCF is excluded from Git. See [data/README.md](data/README.md)
for its location and preparation command.

## Machine configuration

Observed on 2026-09-07 in the current container/VM. These are the resources
visible to this environment, not minimum project requirements.

- **GPU:** 1 NVIDIA L40S, 46,068 MiB VRAM; NVIDIA driver 565.57.01.
- **CPU:** AMD EPYC 9254; 8 logical CPUs exposed (4 cores, 2 threads per core),
  x86_64, KVM virtualization.
- **Memory:** 144 GiB RAM; no swap.
- **Storage:** project resides on a 124 GiB overlay filesystem.
- **OS:** Ubuntu 24.04.2 LTS; Linux kernel 5.15.0-126-generic.
- **Python:** 3.12.3 at `/usr/bin/python`.
- **GPU software:** CUDA toolkit 12.9 (`nvcc` 12.9.41), PyTorch CUDA build 12.9,
  cuDNN 9.9.0.

The system Python is externally managed (PEP 668): direct `pip install` commands
are blocked. Use a virtual environment and its Jupyter kernel for package
changes.
