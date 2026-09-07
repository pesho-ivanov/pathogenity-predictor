# Pesho's Pathogenicity Predictor

A small research project for predicting genetic variant pathogenicity from
ClinVar data. See [GOALS.md](GOALS.md) for the project requirements.

This is exploratory research, not clinical validation.

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

### [Q0. What does the ClinVar dataset contain, and which variants are suitable for reliable evaluation?](notebooks/Q0-clinvar-summary.ipynb)

Explore variant types, genes, pathogenicity labels, review status, missing
annotations, conflicting classifications, and duplicate or related records.
Use plots to reveal class imbalance and potential leakage, then define an
initial dataset and filtering criteria.

### [Q1. How should ClinVar variants be split for reliable evaluation on previously unseen genes?](notebooks/Q1-clinvar-split.ipynb)

Build the 5,000-variant pilot from Q0's cohort. Explain how genes, loci, source
IDs and overlapping or identical sequence contexts connect variants into groups;
assign whole groups to training and validation. Visualize the split sizes, verify
separation and export `data/clinvar-train-pilot.vcf` and `data/clinvar-test-pilot.vcf` as the
fixed inputs for all later experiments. The latter file contains validation data.

### [Q2. Can a small classifier on frozen Evo2 representations outperform zero-shot Evo2 scoring on previously unseen genes?](notebooks/Q2-evo2-classifier.ipynb)

Compare logistic regression on frozen Evo2 1B representations with zero-shot
scores and a DNA-only sequence baseline: 5,000 SNVs, 1,024-base contexts,
gene-disjoint splits, and paired component bootstrap intervals. Genes, source
IDs, loci and overlapping contexts are grouped across the full cohort before
sampling; identical pilot contexts, including reverse complements, also stay
together. Clinical annotations are excluded from predictor features.

Use Q1's training VCF for fitting and its validation VCF for model selection and
comparison. A separate untouched holdout is required for final performance claims.

### Q3. How much does apparent predictive performance depend on similarities between training and test data?

Compare conventional random splits with progressively stricter locus-,
sequence-context-, and gene-separated splits. Treat random splits as a diagnostic
benchmark; quantify how much performance survives credible leakage controls.

### Q4. Does longer sequence context improve pathogenicity prediction, and for which variant classes?

Vary context length while holding the checkpoint, classifier, and evaluation
variants fixed. Compare missense, splice-associated, and noncoding variants where
sample sizes permit, measuring both predictive gains and computational cost.

### Q5. Is it better to train on fewer strongly supported ClinVar labels or more labels with weaker supporting evidence?

Compare training sets filtered by review status, including size-matched
comparisons, against a fixed, strongly reviewed holdout. ClinVar's review status
captures review processes and agreement, making this a useful test of label
selection.
([ClinVar documentation](https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/))

### Q6. Can the predictor identify when its own predictions are unreliable?

Evaluate calibration and whether withholding low-confidence predictions reduces
errors on unseen genes and different variant classes. Report the relationship
between retained coverage and error rate, alongside discrimination metrics.

### Q7. Can a model trained on an older ClinVar snapshot predict subsequently resolved variants of uncertain significance?

Freeze training labels at an earlier release and evaluate variants that later
receive clear classifications. This requires historical snapshots and careful
provenance checks, but would test usefulness beyond reproducing existing labels.

### [Q8. What in silico tools currently exist for predicting genetic variant pathogenicity, and which are suitable baselines for this project?](notebooks/Q8-tools-survey.ipynb)

Survey tools for missense, splicing, regulatory effects and broad variant scoring.
Compare their inputs, availability, licensing, compute and clinical training or
calibration exposure. Use a dated, cited overview to identify practical baselines
for evaluation on Q1's fixed partitions.

## Notebooks

Research notebooks and their explanations live in `notebooks/`; generated files
live in `notebooks/results/`.
Implementation lives in [q0.py](notebooks/src/q0.py), [q1.py](notebooks/src/q1.py)
and [q2.py](notebooks/src/q2.py). The tool survey uses [q8.py](notebooks/src/q8.py)
and a [bundled source catalog](notebooks/src/q8_catalog.json).

## Data

The local ClinVar VCF is excluded from Git. See [data/README.md](data/README.md)
for its location and preparation command. Q1 also writes the two shared VCF
partitions to `data/`; other generated files live under `notebooks/results/`.
Generated artifacts are described in [results documentation](notebooks/results/README.md).

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
