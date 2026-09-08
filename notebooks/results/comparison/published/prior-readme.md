# Pathogenicity Prediction

## Problem statement

Predict whether a single-nucleotide **missense variant** is **benign (0)** or
**pathogenic (1)** using ClinVar labels, merging “likely benign” and “likely
pathogenic” into their respective classes. Uncertain, conflicting, other, and
missing labels are excluded from training and evaluation.

### Data

Q0 and Q1 share **`data/clinvar_20260706.vcf` (6 July 2026)** and automatically
download the verified dated archive when needed. The obsolete `data/clinvar.vcf`
has been removed. Completed Q2/Q8/Q9/Q10 September pilot runs remain historical;
the current comparison requires results from the full July dataset.
See [data/README.md](data/README.md) for filenames and checksum checks. Q1 also writes the two shared VCF
partitions to `data/`; other generated files live under `notebooks/results/`.
Generated artifacts are described in [results documentation](notebooks/results/README.md).

<!-- comparison:start -->
## Method comparison

Compare methods on Q1’s current frozen missense validation set. Only results matching its snapshot and complete cohort are included.

**17,927 missense validation variants · ClinVar 2026-07-06 · full cohort**

| Method | Notebook | Paper | Scored / validation | AUROC [95% CI] | Average precision [95% CI] | Runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Evo2 7B base frozen head (BioNeMo, BF16) | [Q10](notebooks/Q10-evo2-7b.ipynb) | — | — | — | — | — |
| Evo2 7B base zero-shot (Vortex, FP8) | [Q2](notebooks/Q2-evo2-classifier.ipynb) | — | 17,927 / 17,927 | 0.837 [0.824, 0.849] | 0.724 [0.686, 0.755] | 2.3 h |
| Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp) | Q11 | — | — | — | — | — |
| <hr> | <hr> | <hr> | <hr> | <hr> | <hr> | <hr> |
| SIFT4G | [Q8](notebooks/Q8-existing-tools.ipynb) | [Vaser et al. (2016)](https://doi.org/10.1038/nprot.2015.123) | 16,917 / 17,927 | 0.878 [0.865, 0.890] | 0.772 [0.734, 0.804] | 13.0 s |
| PolyPhen-2 | [Q8](notebooks/Q8-existing-tools.ipynb) | [Adzhubei et al. (2010)](https://doi.org/10.1038/nmeth0410-248) | 16,430 / 17,927 | 0.894 [0.883, 0.905] | 0.822 [0.781, 0.852] | 12.4 s |
| REVEL | [Q8](notebooks/Q8-existing-tools.ipynb) | [Ioannidis et al. (2016)](https://doi.org/10.1016/j.ajhg.2016.08.016) | 17,720 / 17,927 | 0.974 [0.970, 0.978] | 0.958 [0.945, 0.968] | 1.8 min |
| AlphaMissense | [Q8](notebooks/Q8-existing-tools.ipynb) | [Cheng et al. (2023)](https://doi.org/10.1126/science.adg7492) | 16,884 / 17,927 | 0.962 [0.956, 0.967] | 0.940 [0.923, 0.951] | 1.2 min |
| EVE | [Q8](notebooks/Q8-existing-tools.ipynb) | [Frazer et al. (2021)](https://doi.org/10.1038/s41586-021-04043-8) | 9,284 / 17,927 | 0.909 [0.889, 0.925] | 0.916 [0.889, 0.935] | 5.5 s |
| PrimateAI-3D (licensed) | [Q8](notebooks/Q8-existing-tools.ipynb) | [Gao et al. (2023)](https://doi.org/10.1126/science.abn8197) | — | — | — | — |

Runtime covers the recorded stages listed in the details below; hardware and caching differ between workflows. “—” means no verified timing is available for the current cohort.

**Direct comparison on the same variants**

| Method | Shared variants | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- | --- |
| Evo2 7B base zero-shot (Vortex, FP8) (Q2) | 8817 | 0.815 [0.797, 0.832] | 0.824 [0.791, 0.855] |
| <hr> | <hr> | <hr> | <hr> |
| SIFT4G (Q8) | 8817 | 0.887 [0.872, 0.901] | 0.881 [0.853, 0.906] |
| PolyPhen-2 (Q8) | 8817 | 0.900 [0.887, 0.913] | 0.903 [0.879, 0.925] |
| REVEL (Q8) | 8817 | 0.972 [0.966, 0.978] | 0.976 [0.967, 0.984] |
| AlphaMissense (Q8) | 8817 | 0.961 [0.953, 0.968] | 0.967 [0.957, 0.975] |
| EVE (Q8) | 8817 | 0.911 [0.893, 0.928] | 0.921 [0.899, 0.940] |

<details>
<summary>Provenance, missing results and limitations</summary>

| Method | Details |
| --- | --- |
| Evo2 7B base frozen head (BioNeMo, BF16) (Q10) | Export cohort is stale |
| Evo2 7B base zero-shot (Vortex, FP8) (Q2) | Zero-shot inference on the complete current full validation cohort. No parameters or thresholds fitted. Archived September results are not mixed with this run. Pretraining, homology and annotation overlap remain unresolved; validation is development data, not an untouched final test. Runtime: Validation scoring batches summed across runs; excludes downloads, model loading and evaluation. |
| Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp) (Q11) | The partial-epoch experiment with full validation has not completed. |
| <hr> | <hr> |
| SIFT4G (Q8) | dbNSFP4.9a; 1 minus the minimum raw SIFT4G score. Evolutionary sequence exposure is unaudited. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| PolyPhen-2 (Q8) | HumVar model from dbNSFP4.9a; maximum raw score. Known disease training variants may overlap ClinVar. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| REVEL (Q8) | HGMD and constituent-tool training overlap with ClinVar unresolved; maximum exact-allele score across transcript annotations. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training. |
| AlphaMissense (Q8) | ClinVar calibration overlap unresolved; maximum matching transcript score. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training. |
| EVE (Q8) | dbNSFP4.9a continuous EVE score; maximum across matches, no confidence-category filtering. Limited protein/position coverage. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| PrimateAI-3D (Q8) | PrimateAI-3D has not been run: Illumina requires a signed license agreement and supplies the score/model download link by email. No approved link or licensed score file was provided. The original PrimateAI scores in dbNSFP are a different model and are not substituted. |

```json
{
  "cohort": {
    "protocol_sha256": "79c57bccf81867775b19c4ea272b4de9743c36b782242b4641be49185aca167c",
    "vcf_exports": {
      "clinvar-test.vcf": "ae019c877240364c5ab02884a68ed93f203efad1650137a8be83369fe59a8d7a",
      "clinvar-train.vcf": "aa7d1e941ba06cd8ccc7531ae5946b4e851b7d321c531fc583d76fdd64bbdd86"
    },
    "validation_variants": 17927,
    "clinvar_date": "2026-07-06",
    "scope": "full"
  },
  "source_errors": {
    "Q2": "Q2 cohort is stale",
    "Q9": "Q9 cohort is stale",
    "Q10": "Export cohort is stale"
  },
  "bootstrap": {
    "repetitions": 1000,
    "seed": 42
  },
  "generated_utc": "2026-09-07T21:29:44.301325+00:00"
}
```

</details>

**Conclusion.** Use the common-subset comparison to assess methods; coverage remains a separate limitation.

These are development results: validation participates in model selection. AlphaMissense calibration and REVEL/PolyPhen-2 training overlap with ClinVar remain unresolved. The 95% intervals resample whole Q1 components and do not correct selection bias or establish clinical validity.
<!-- comparison:end -->

<details>
<summary>Methodological choices of published predictors</summary>

These six predictors use different evidence and learning strategies. A multiple
sequence alignment (MSA) lines up related protein sequences to reveal conserved
positions and coordinated changes. Clinical-label use below describes the original
methods; Q8 uses ClinVar labels to evaluate the available predictions.

| Method | Model type | Biological evidence | Training signal | Clinical-label use | Score source in this project |
| --- | --- | --- | --- | --- | --- |
| [SIFT4G](https://doi.org/10.1038/nprot.2015.123) | Conservation-based substitution scoring | Protein MSAs; amino-acid conservation | Amino-acid frequencies in related proteins | No clinical-label fitting in core scoring | [dbNSFP4.9a](data/README.md#q8-sift4g-polyphen-2-and-eve-scores); `1 - SIFT4G_score` |
| [PolyPhen-2 HumVar](https://genetics.bwh.harvard.edu/wiki/%21pph2/overview) | Naïve Bayes classifier | Protein sequence conservation and structural features | [HumVar](https://genetics.bwh.harvard.edu/downloads/pph2/training/): disease/function variants versus common neutral human variants | Disease/function labels in training; ClinVar overlap unresolved | [dbNSFP4.9a](data/README.md#q8-sift4g-polyphen-2-and-eve-scores); `Polyphen2_HVAR_score` |
| [REVEL](https://doi.org/10.1016/j.ajhg.2016.08.016) | Random forest ensemble | Scores from 13 prediction/conservation tools | HGMD disease variants versus rare presumed-neutral variants | Disease labels in training; constituent tools bring their own training histories; ClinVar overlap unresolved | [REVEL v1.3](data/README.md#q8-revel-scores); continuous REVEL score |
| [AlphaMissense](https://doi.org/10.1126/science.adg7492) | AlphaFold-derived neural network | Protein MSAs and learned structural representations | Human/primate population variation, building on AlphaFold pretraining | Core model avoids clinical labels; [released scores and thresholds use ClinVar calibration](https://www.ebi.ac.uk/training/online/courses/alphafold/classifying-the-effects-of-missense-variants-using-alphamissense/understanding-pathogenicity-scores-from-alphamissense/) | [2023 GRCh38 release](data/README.md#q8-alphamissense-scores); `am_pathogenicity` |
| [EVE](https://github.com/OATML-Markslab/EVE) | Protein-family variational autoencoder, then Gaussian mixture scoring | Protein MSAs; dependencies between amino-acid positions | Unsupervised learning of natural protein sequences | No clinical labels required for fitting/scoring; ClinVar used for evaluation | [dbNSFP4.9a](data/README.md#q8-sift4g-polyphen-2-and-eve-scores); continuous `EVE_score`, without confidence-category filtering |
| [PrimateAI-3D](https://www.illumina.com/science/genomics-research/articles/primateai-3d.html) | 3D convolutional neural network | Protein 3D structures and MSAs | Common human/primate variation; auxiliary sequence/structure learning tasks | No clinical annotations for core training; ClinVar used for benchmarking | [Licensed scores](data/README.md#q8-primateai-3d-access); unavailable to this project |

The score sources describe Q8's published-score lookups; current-cohort result
availability is shown above. Training without clinical labels does not establish
benchmark independence: calibration and overlap in variants, proteins or
evolutionary sequences still require auditing.

</details>

## Research questions

### [Q0. Which ClinVar missense variants are usable?](notebooks/Q0-clinvar-summary.ipynb)

### [Q1. How should variants be split?](notebooks/Q1-clinvar-split.ipynb)

### [Q2. Can Evo2 classifiers beat zero-shot scoring?](notebooks/Q2-evo2-classifier.ipynb)

### [Q8. How do existing predictors compare?](notebooks/Q8-existing-tools.ipynb)

### [Q9. Does Evo2 1B fine-tuning help?](notebooks/Q9-evo2-1b.ipynb)

### [Q10. Does frozen Evo2 7B outperform 1B?](notebooks/Q10-evo2-7b.ipynb)

### [Q11. How well does a short LoRA run perform?](notebooks/Q11-evo2-lora.ipynb)

## Setup

Use Linux x86-64 with Python 3.12, virtual-environment support, Git and curl.
Quick CPU setup:

```bash
git clone --recurse-submodules https://github.com/pesho-ivanov/pathogenity-predictor.git
cd pathogenity-predictor
python3.12 scripts/setup.py --profile cpu --test
.venv/bin/jupyter lab
```

For a GPU machine, use `nvcr.io/nvidia/pytorch:25.04-py3` and run
`python scripts/setup.py --profile gpu --test` inside the container.

Q11 adds a budgeted partial-epoch, single-GPU rank-8 LoRA run. It uses 512-base
paired alleles, batches of 32, at most 768 updates and 30 training minutes, and
complete validation. Every original backbone weight stays frozen. Run
`.venv/bin/python -m notebooks.src.refresh_q11` to prepare inputs and execute the
notebook. The 45–55-minute target applies to the prepared H100 with cached dependencies
and weights, with a 60-minute outer budget; initial installation/downloads are additional.
Training uses BF16 with
FP32 optimizer masters, fixed per-example normalization and resumable checkpoints.
The runner reserves time for full validation and publishes only on success.

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

## Unsuccessful attempts

### Evo2 1B: numerical sensitivity and no fine-tuning gain

NVIDIA's [BioNeMo model compatibility table](https://docs.nvidia.com/bionemo-framework/2.7.1/main/developer-guide/bionemo-evo2/bionemo-evo2-Overview/index.html#available-models-in-ngc)
documents low BF16 accuracy for the original 1B checkpoint. The
[fine-tuning tutorial's motivation](https://docs.nvidia.com/bionemo-recipes/latest/main/examples/bionemo-evo2/examples/fine-tuning-tutorial/index.html#background-and-motivation)
reports near-random BRCA1 zero-shot AUC without FP8. NVIDIA supplies
`evo2/1b-8k-bf16:1.0`, a checkpoint adapted for BF16. Q9's main comparison used
the original checkpoint; the adapted version was tested only in numerical
diagnostics. Q9's original-checkpoint BF16 results are consistent with this
documented limitation, which we consider a likely contributor to the poor
performance.

The [Q9 experiments](notebooks/Q9-evo2-1b.ipynb) exposed numerical problems with
the original 1B checkpoint. Vortex and BioNeMo produced different outputs;
their RMS normalization differed, although this was not established as the sole
cause. RMS (root mean square) measures the magnitude of the activations.

In BioNeMo/BF16, Hyena block 23 produced activations around **10¹⁶ RMS**, while
final attention block 24 contributed only **10⁻⁶ RMS**. Its contribution vanished
in BF16 addition. Training block 23 allowed effective backbone updates, but the
September pilot selected **epoch 0**: fine-tuning did not improve validation
performance over the frozen classifier. The activation imbalance remains
unexplained. [Diagnostic results](notebooks/results/q9/investigation/summary.json)
and [backend comparison](notebooks/results/q9/archive/display_error/parity.json).

### Q11: ineffective final attention block

The original Q11 attempt stopped **before training** because final attention
block **31** was numerically inactive in the tested checkpoint/configuration.
FP32 optimizer master weights alone did not produce effective BF16 updates.
Block **30** has demonstrated usable updates, making it a better-supported
training target; validation must establish whether those updates improve
prediction.

<details>
<summary>Q11 diagnostic findings</summary>

- **Vanishing contribution:** block 30 activations were about **1.2 × 10¹¹ RMS**,
  versus **0.006 RMS** for block 31's attention contribution. BF16 addition left
  every output element unchanged.
- **Tiny gradients:** attention-weight gradients were around **10⁻¹⁹** and MLP
  gradients around **10⁻²⁵**. AdamW's epsilon suppressed the updates further;
  gradient clipping reduced them roughly **33×** because it included the fixed
  classifier's gradients.
- **Master weights were insufficient:** a fresh probe changed **88 FP32 master
  elements**, but **zero deployed BF16 elements**. Predictions remained identical.
- **Checkpoint audit:** all **325 loaded tensors** matched the original
  checkpoint. The tiny final-block weights were already present there; optimizer
  wiring and the autograd connection were correct. Why pretraining produced this
  imbalance remains unresolved.

</details>
