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
| Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp) | [Q11](notebooks/Q11-evo2-lora.ipynb) | — | 17,927 / 17,927 | 0.693 [0.675, 0.711] | 0.487 [0.435, 0.536] | 42.6 min |
| Evo2 7B Q14 LoRA (blocks 29 and 30, rank 8, 512 bp) | [Q14](notebooks/Q14-lora-validation.ipynb) | — | 17,927 / 17,927 | 0.852 [0.838, 0.865] | 0.770 [0.736, 0.797] | 1.3 h |
| Evo2 7B Q14 strongest frozen classifier (magnitude features, 512 bp) | [Q14](notebooks/Q14-lora-validation.ipynb) | — | 17,927 / 17,927 | 0.839 [0.824, 0.852] | 0.749 [0.716, 0.778] | 1.3 h |
| Evo2 7B base zero-shot (Vortex, FP8) | [Q2](notebooks/Q2-evo2-classifier.ipynb) | — | 17,927 / 17,927 | 0.837 [0.824, 0.849] | 0.724 [0.686, 0.755] | 2.3 h |
| <hr> | <hr> | <hr> | <hr> | <hr> | <hr> | <hr> |
| SIFT4G | [Q8](notebooks/Q8-existing-tools.ipynb) | [Vaser et al. (2016)](https://doi.org/10.1038/nprot.2015.123) | 16,917 / 17,927 | 0.878 [0.865, 0.890] | 0.772 [0.734, 0.804] | 13.0 s |
| PolyPhen-2 | [Q8](notebooks/Q8-existing-tools.ipynb) | [Adzhubei et al. (2010)](https://doi.org/10.1038/nmeth0410-248) | 16,430 / 17,927 | 0.894 [0.883, 0.905] | 0.822 [0.781, 0.852] | 12.4 s |
| REVEL | [Q8](notebooks/Q8-existing-tools.ipynb) | [Ioannidis et al. (2016)](https://doi.org/10.1016/j.ajhg.2016.08.016) | 17,720 / 17,927 | 0.974 [0.970, 0.978] | 0.958 [0.945, 0.968] | 1.8 min |
| AlphaMissense | [Q8](notebooks/Q8-existing-tools.ipynb) | [Cheng et al. (2023)](https://doi.org/10.1126/science.adg7492) | 16,884 / 17,927 | 0.962 [0.956, 0.967] | 0.940 [0.923, 0.951] | 1.2 min |
| EVE | [Q8](notebooks/Q8-existing-tools.ipynb) | [Frazer et al. (2021)](https://doi.org/10.1038/s41586-021-04043-8) | 9,284 / 17,927 | 0.909 [0.889, 0.925] | 0.916 [0.889, 0.935] | 5.5 s |
| PrimateAI-3D (licensed) | [Q8](notebooks/Q8-existing-tools.ipynb) | [Gao et al. (2023)](https://doi.org/10.1126/science.abn8197) | — | — | — | — |

Runtime covers the recorded stages listed in the details below; hardware and caching differ between workflows. “—” means no verified timing is available for the current cohort.

Completed notebook measurements are preserved when local prediction exports are absent. Their source notebooks and exact cohort checksums are verified; locally available predictions take precedence.

**Direct comparison on the same variants**

| Method | Shared variants | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- | --- |
| Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp) (Q11) | 17927 | 0.693 [0.675, 0.711] | 0.487 [0.435, 0.536] |
| Evo2 7B Q14 LoRA (blocks 29 and 30, rank 8, 512 bp) (Q14) | 17927 | 0.852 [0.838, 0.865] | 0.770 [0.736, 0.797] |
| Evo2 7B Q14 strongest frozen classifier (magnitude features, 512 bp) (Q14) | 17927 | 0.839 [0.824, 0.852] | 0.749 [0.716, 0.778] |

**Published comparison on 8,817 shared variants (Q2/Q8; excludes LoRA)**

| Method | Shared variants | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- | --- |
| Evo2 7B base zero-shot (Vortex, FP8) (Q2) | 8817 | 0.815 [0.797, 0.832] | 0.824 [0.791, 0.855] |
| <hr> | <hr> | <hr> | <hr> |
| SIFT4G (Q8) | 8817 | 0.887 [0.872, 0.901] | 0.881 [0.853, 0.906] |
| PolyPhen-2 (Q8) | 8817 | 0.900 [0.887, 0.913] | 0.903 [0.879, 0.925] |
| REVEL (Q8) | 8817 | 0.972 [0.966, 0.978] | 0.976 [0.967, 0.984] |
| AlphaMissense (Q8) | 8817 | 0.961 [0.953, 0.968] | 0.967 [0.957, 0.975] |
| EVE (Q8) | 8817 | 0.911 [0.893, 0.928] | 0.921 [0.899, 0.940] |

These shared-subset results come from the completed Q2/Q8 comparison. Computing LoRA on that subset requires the original per-variant exports.

<details>
<summary>Provenance, missing results and limitations</summary>

| Method | Details |
| --- | --- |
| Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp) (Q11) | User-authorized partial-epoch training for a one-hour run; full frozen validation is retained. No fitted frozen baseline or improvement claim; these are development results. Pretraining sequence exposure, homology and shared-patient overlap remain unresolved. The final attention block is numerically inactive in this BF16 configuration; LoRA targets the preceding Hyena mixer. No clinical validity or independent final-test performance is established. Runtime: Fresh notebook start through input checks, cached setup verification, preflight, partial-epoch training, full validation, 64-variant reload check and bootstrap; excludes initial environment/model acquisition and final notebook/README export. |
| Evo2 7B Q14 LoRA (blocks 29 and 30, rank 8, 512 bp) (Q14) | Prespecified full-validation confirmation rule met. Full frozen July missense development validation, following repeated selection on its 2,048-variant subset. Variants outside that subset and components absent from it were not used by those recent selection runs, but this repository has previously evaluated the full validation partition. Neither subset is an untouched final test set. Component-bootstrap intervals do not remove selection bias. Pretraining exposure, homology and shared-patient overlap remain unresolved; this research prototype has no clinical validation. Runtime: Current Q14 exploration (43.0 min, including all its search arms) plus separate full validation (34.7 min), shared by LoRA and controls. Excludes inherited Q12 feature extraction/fitting and earlier experiments or attempts. The one-hour target applies only to exploration. |
| Evo2 7B Q14 strongest frozen classifier (magnitude features, 512 bp) (Q14) | Prespecified full-validation confirmation rule met. Full frozen July missense development validation, following repeated selection on its 2,048-variant subset. Variants outside that subset and components absent from it were not used by those recent selection runs, but this repository has previously evaluated the full validation partition. Neither subset is an untouched final test set. Component-bootstrap intervals do not remove selection bias. Pretraining exposure, homology and shared-patient overlap remain unresolved; this research prototype has no clinical validation. Runtime: Current Q14 exploration (43.0 min, including all its search arms) plus separate full validation (34.7 min), shared by LoRA and controls. Excludes inherited Q12 feature extraction/fitting and earlier experiments or attempts. The one-hour target applies only to exploration. |
| Evo2 7B base zero-shot (Vortex, FP8) (Q2) | Zero-shot inference on the complete current full validation cohort. No parameters or thresholds fitted. Archived September results are not mixed with this run. Pretraining, homology and annotation overlap remain unresolved; validation is development data, not an untouched final test. Published executed-notebook result; per-variant export is absent locally. Runtime: Validation scoring batches summed across runs; excludes downloads, model loading and evaluation. Published rounded duration. |
| <hr> | <hr> |
| SIFT4G (Q8) | dbNSFP4.9a; 1 minus the minimum raw SIFT4G score. Evolutionary sequence exposure is unaudited. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| PolyPhen-2 (Q8) | HumVar model from dbNSFP4.9a; maximum raw score. Known disease training variants may overlap ClinVar. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| REVEL (Q8) | HGMD and constituent-tool training overlap with ClinVar unresolved; maximum exact-allele score across transcript annotations. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training. |
| AlphaMissense (Q8) | ClinVar calibration overlap unresolved; maximum matching transcript score. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training. |
| EVE (Q8) | dbNSFP4.9a continuous EVE score; maximum across matches, no confidence-category filtering. Limited protein/position coverage. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| PrimateAI-3D (Q8) | Requires licensed data; no predictions for the current cohort. |

```json
{
  "cohort": {
    "protocol_sha256": "931efbfed86417617fb10649a1e61833f0836a9b210a4723d6ac9797a97a8f28",
    "vcf_exports": {
      "clinvar-test.vcf": "ae019c877240364c5ab02884a68ed93f203efad1650137a8be83369fe59a8d7a",
      "clinvar-train.vcf": "aa7d1e941ba06cd8ccc7531ae5946b4e851b7d321c531fc583d76fdd64bbdd86"
    },
    "validation_variants": 17927,
    "clinvar_date": "2026-07-06",
    "scope": "full"
  },
  "source_errors": {},
  "bootstrap": {
    "repetitions": 1000,
    "seed": 42
  },
  "published_methods": [
    "q2:zero_shot_7b",
    "q8:AlphaMissense",
    "q8:REVEL",
    "q8:SIFT4G",
    "q8:PolyPhen-2",
    "q8:EVE"
  ],
  "published_bootstrap": {
    "repetitions": 1000,
    "seed": 42
  },
  "generated_utc": "2026-09-08T03:42:10.852912+00:00"
}
```

</details>

**Conclusion.** Per-method results retain their reported coverage. The published shared-subset comparison covers Q2/Q8; LoRA has complete validation metrics.

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

## Q12 exploratory results

[Q12 improved fine-tuning](notebooks/Q12-lora-improvement.ipynb) completed in
**53.7 minutes**, using 24,576 training variants and the fixed **2,048-variant
validation sample**.

| Model | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- |
| Q12 frozen-backbone classifier — selected | **0.832 [0.812, 0.853]** | **0.725 [0.681, 0.771]** |
| Q12 LoRA with improved features and head fitting | 0.831 [0.811, 0.853] | 0.724 [0.681, 0.768] |

The previous [Q11 LoRA](notebooks/Q11-lora-diagnostics.ipynb) scored
**0.684 AUROC / 0.449 AP** on this same sample. Q12 retained the frozen classifier:
LoRA minus its matched control was **−0.0008 AUROC [−0.0029, 0.0016]**.

These are development results used to select features and checkpoints. The 95%
intervals resample whole Q1 components and do not correct selection bias.
These sampled results are separate from the full-cohort comparison above.
Q12's retained frozen classifier was subsequently evaluated on all **17,927
validation variants** as the control in [Q14 full validation](notebooks/Q14-lora-validation.ipynb).

## Q13 exploratory results

[Q13 optimization](notebooks/Q13-lora-optimization.ipynb) completed in **41.3
minutes** on the same 2,048-variant validation sample. It tested converged
regularized heads, separate head/adapter gradient clipping and two smaller
adapter learning rates.

Head fitting used 24,576 training variants; each adapter trial processed 16,384
of them, with its best checkpoint selected by validation.

| Model | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- |
| Best Q13 LoRA: learning rate 3e-5, step 256 | 0.830 [0.809, 0.851] | 0.722 [0.677, 0.767] |
| Q13 matched frozen control | 0.829 [0.808, 0.850] | 0.722 [0.676, 0.766] |

The paired AUROC gain was **+0.0006 [−0.0003, +0.0016]**. Both trials declined
between steps 256 and 512. Q13 did **not** improve on the strongest Q12 frozen
classifier (**0.832 AUROC / 0.725 AP**), which Q13 retained. These sampled
development results do not replace the full-cohort comparison.

## Q14 exploratory results

[Q14 layer adapters](notebooks/Q14-layer-adapters.ipynb) completed in **43.0
minutes** and improved both metrics on the same **2,048-variant development
sample**. It held the strongest Q12 frozen classifier and its scaler fixed
while training rank-8 adapters in blocks 29 and 30. Each of the two learning-rate
trials processed the same 16,384 variants from the 24,576-variant training pool.

| Model | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- |
| Q14 LoRA: learning rate 1e-4, step 512 — selected | **0.845 [0.825, 0.865]** | **0.748 [0.702, 0.791]** |
| Strongest frozen classifier, inherited from Q12 | 0.832 [0.812, 0.853] | 0.725 [0.681, 0.771] |

The paired gains were **+0.0130 AUROC [0.0050, 0.0206]** and **+0.0234 AP
[0.0088, 0.0380]**. The adapter passed the prespecified sampled promotion rule:
at least +0.005 AUROC over the strongest frozen control with no AP decrease.
Checkpoint reload, unchanged backbone, fixed head and fixed scaler checks passed.

These intervals resample whole Q1 components and do not correct repeated
development selection. The separate full-cohort confirmation is reported below;
sampled results do not enter the full-cohort comparison.
The 43-minute duration covers this exploration and excludes inherited feature
extraction, earlier experiments and the separate full confirmation.

## Q14 full-cohort confirmation

[Q14 full validation](notebooks/Q14-lora-validation.ipynb) confirmed the adapter
gain on **all 17,927 validation variants**. LoRA scored **0.852 AUROC / 0.770 AP**,
compared with **0.839 / 0.749** for the strongest frozen classifier. The paired
gains were **+0.0131 AUROC [0.0080, 0.0180]** and **+0.0206 AP [0.0123, 0.0287]**.
The complete metrics and intervals appear in the comparison above.

On the **15,879 variants outside the selection sample**, gains remained
**+0.0131 AUROC [0.0076, 0.0179]** and **+0.0203 AP [0.0120, 0.0278]**. The
prespecified confirmation rule passed. The smaller set of **3,246 variants in
components absent from selection** had positive point gains, but its intervals
included zero: **+0.0082 AUROC [−0.0007, 0.0168]** and **+0.0162 AP
[−0.0054, 0.0382]**. These remain development results, without an untouched test
set or correction for repeated model selection.

Full validation reproduced all 2,048 original selection predictions, verified
fresh 64-variant reloads of LoRA and the frozen control, and confirmed unchanged
backbone weights. Its fully executed notebook took **34.7 minutes**, separately
from the **43.0-minute** exploration (**77.7 minutes combined**). This excludes
inherited feature extraction, head fitting and earlier experiments; the one-hour
budget applied to the exploration.

## Research questions

### [Q0. Which ClinVar missense variants are usable?](notebooks/Q0-clinvar-summary.ipynb)

### [Q1. How should variants be split?](notebooks/Q1-clinvar-split.ipynb)

### [Q2. Can Evo2 classifiers beat zero-shot scoring?](notebooks/Q2-evo2-classifier.ipynb)

### [Q8. How do existing predictors compare?](notebooks/Q8-existing-tools.ipynb)

### [Q9. Does Evo2 1B fine-tuning help?](notebooks/Q9-evo2-1b.ipynb)

### [Q10. Does frozen Evo2 7B outperform 1B?](notebooks/Q10-evo2-7b.ipynb)

### [Q11. How well does a short LoRA run perform?](notebooks/Q11-evo2-lora.ipynb)

[Performance investigation](notebooks/Q11-lora-diagnostics.ipynb): adapter ablation
on a seeded 2,048-variant diagnostic sample, feature contributions and training probes.
On that sample, raw allele-difference magnitude scores AUROC/AP **0.777/0.637**,
versus **0.684/0.449** for the classifier; its per-example normalization removes
this magnitude. Disabling adapters with the trained head fixed changes AUROC by
only **−0.002** (trained-minus-disabled 95% interval **[−0.004, 0.008]**).
These post-hoc diagnostics identify feature construction and head fitting as
priorities for the next experiment; the full-cohort benchmark is reported above.

### [Q12. Can better features and head fitting improve LoRA?](notebooks/Q12-lora-improvement.ipynb)

Compare three feature designs, including allele-difference magnitude, before
matched LoRA and frozen-backbone classifier training. This one-hour exploration
uses Q11's 24,576 training variants and fixed 2,048-variant validation sample.
Sampled development results stay separate from the full-cohort comparison.

See the [Q12 results table](#q12-exploratory-results) for the completed run.
Adding magnitude features raised the initial head comparison from **0.661 / 0.481**
to **0.819 / 0.706** AUROC/AP, before continuation.

Run `.venv/bin/python -m notebooks.src.refresh_q12` after Q11 and its performance
investigation. The selected model can be evaluated separately on all validation
variants with `.venv/bin/python -m notebooks.src.refresh_q12 --full-validation`.

### [Q13. Can better optimization produce an adapter-specific gain?](notebooks/Q13-lora-optimization.ipynb)

Fit regularized linear heads on verified frozen features, then compare LoRA and
matched head-only continuation with independent gradient clipping, warmup and
cosine learning-rate decay. Nonconverged head candidates remain recorded but
cannot be selected. The completed run found no meaningful adapter advantage;
see the [Q13 results](#q13-exploratory-results).

Run `.venv/bin/python -m notebooks.src.refresh_q13` after Q12. The runner archives
earlier attempts and executes every notebook cell from a fresh kernel. The
initial failed FP32 head-fitting attempt and its real error are preserved;
the completed retry fits in FP64 and verifies its FP32 deployment.

### [Q14. Can adapting two active mixer blocks improve Evo2 fine-tuning?](notebooks/Q14-layer-adapters.ipynb)

Hold the strongest completed frozen classifier fixed and train adapters on
blocks 29 and 30. The completed two-rate experiment found an adapter gain on
the development sample, then passed [full-cohort confirmation](notebooks/Q14-lora-validation.ipynb);
see the [Q14 results](#q14-exploratory-results).

Run `.venv/bin/python -m notebooks.src.refresh_q14` after Q13. The runner
archives previous attempts and executes the notebook from a fresh kernel.
Evaluate the promoted model separately on all validation variants with
`.venv/bin/python -m notebooks.src.refresh_lora_validation --question q14`.

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
