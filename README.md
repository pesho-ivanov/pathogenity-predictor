# Gamow predictor

A small research project for predicting genetic variant pathogenicity from
ClinVar data. See [GOALS.md](GOALS.md) for the project requirements.

The project is at the setup stage. No ClinVar predictor has been trained or
validated yet.

Clone with the Evo2 source included:

```bash
git clone --recurse-submodules git@github.com:pesho-ivanov/gamow-predictor.git
```

## Notebooks

- [BRCA1 execution snapshot](results/brca1_zero_shot_vep.executed.ipynb): partial
  execution of the upstream Evo2 1B example using experimental BRCA1 function
  scores. Stopped during variant scoring; no final plots or AUROC were produced.
  The install command failed because the Python environment was externally
  managed; inference used packages already installed. This snapshot used
  `evo2/notebooks/brca1/` as its working directory.
- [Upstream BRCA1 example](evo2/notebooks/brca1/brca1_zero_shot_vep.ipynb): zero-shot
  variant-effect scoring against experimental BRCA1 function measurements.
- [Upstream exon classifier](evo2/notebooks/exon_classifier/exon_classifier.ipynb):
  exon classification example using Evo2.
- [Upstream generation example](evo2/notebooks/generation/generation_notebook.ipynb):
  DNA sequence generation with Evo2.
- [Upstream sparse autoencoder example](evo2/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb):
  exploration of Evo2 representations with a sparse autoencoder.

`notebooks/` is reserved for project notebooks. Explanations and results belong
in `results/`; briefly document new notebooks here.

## Data

The local ClinVar VCF is excluded from Git. See [data/README.md](data/README.md)
for its location and preparation command.
