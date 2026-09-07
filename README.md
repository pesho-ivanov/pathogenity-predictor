# Gamow predictor

A small research project for predicting genetic variant pathogenicity from
ClinVar data. See [GOALS.md](GOALS.md) for the project requirements.

Q0 explores the local ClinVar data and proposes a conservative SNV cohort.
No ClinVar predictor has been trained or validated yet.

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

Question titles link to available notebooks. Q1–Q6 are planned; their notebooks
have not been created yet.

### [Q0. What does the ClinVar dataset contain, and which variants are suitable for reliable evaluation?](notebooks/Q0.ipynb)

Explore variant types, genes, pathogenicity labels, review status, missing
annotations, conflicting classifications, and duplicate or related records.
Use plots to reveal class imbalance and potential leakage, then define an
initial dataset and filtering criteria.

All labels are explored; this is development data, not an untouched test set.

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

Research notebooks and their explanations live in `notebooks/`; generated files
live in `notebooks/results/`. Q0 is linked
above; Q1–Q6 have not been implemented yet. Reusable Q0 logic is in [q0.py](notebooks/src/q0.py).

### Running Q0

Use Python 3.12 and the local input described in [data/README.md](data/README.md).
The notebook scans the full file on CPU, without GPU use or model downloads.
On the machine documented below, execution took about two minutes with 6.9 GiB
peak process memory. Allow 16 GB of RAM for headroom. There is no random sampling.

Create an isolated environment from the project root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name gamow-q0 --display-name "Gamow Q0"
.venv/bin/jupyter lab notebooks/Q0.ipynb
```

Select the **Gamow Q0** kernel and run all cells. Alternatively, execute from a
fresh kernel at the command line:

```bash
.venv/bin/jupyter execute notebooks/Q0.ipynb --kernel_name=gamow-q0 --inplace --timeout=1200
.venv/bin/python -m unittest discover -s notebooks/src -v
```

Configuration and workflow code live in `notebooks/src/q0.py`; notebook cells
contain only an import and short section calls.

The notebook verifies the input checksum, exports audit summaries and a
provisional cohort under `notebooks/results/q0/`, and keeps detailed tables in dropdowns.
It does not assign train/validation/test splits or certify that leakage is absent.

## Data

The local ClinVar VCF is excluded from Git. See [data/README.md](data/README.md)
for its location and preparation command. `data/` contains external inputs only.
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
