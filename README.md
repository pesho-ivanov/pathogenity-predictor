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
| Evo2 1B base zero-shot (Vortex, FP8) | [Q2](notebooks/Q2-evo2-classifier.ipynb) | — | — | — | — | — |
| SIFT4G | [Q8](notebooks/Q8-existing-tools.ipynb) | [Vaser et al. (2016)](https://doi.org/10.1038/nprot.2015.123) | 16,917 / 17,927 | 0.878 [0.865, 0.890] | 0.772 [0.734, 0.804] | 13.0 s |
| PolyPhen-2 | [Q8](notebooks/Q8-existing-tools.ipynb) | [Adzhubei et al. (2010)](https://doi.org/10.1038/nmeth0410-248) | 16,430 / 17,927 | 0.894 [0.883, 0.905] | 0.822 [0.781, 0.852] | 12.4 s |
| REVEL | [Q8](notebooks/Q8-existing-tools.ipynb) | [Ioannidis et al. (2016)](https://doi.org/10.1016/j.ajhg.2016.08.016) | 17,720 / 17,927 | 0.974 [0.970, 0.978] | 0.958 [0.945, 0.968] | 1.8 min |
| AlphaMissense | [Q8](notebooks/Q8-existing-tools.ipynb) | [Cheng et al. (2023)](https://doi.org/10.1126/science.adg7492) | 16,884 / 17,927 | 0.962 [0.956, 0.967] | 0.940 [0.923, 0.951] | 1.2 min |
| EVE | [Q8](notebooks/Q8-existing-tools.ipynb) | [Frazer et al. (2021)](https://doi.org/10.1038/s41586-021-04043-8) | 9,284 / 17,927 | 0.909 [0.889, 0.925] | 0.916 [0.889, 0.935] | 5.5 s |
| Evo2 1B base frozen head (BioNeMo, BF16) | [Q9](notebooks/Q9-evo2-1b.ipynb) | — | — | — | — | — |
| Sequence baseline (Evo2 1B experiment) | [Q9](notebooks/Q9-evo2-1b.ipynb) | — | — | — | — | — |
| Evo2 7B base frozen head (BioNeMo, BF16) | [Q10](notebooks/Q10-evo2-7b.ipynb) | — | — | — | — | — |
| PrimateAI-3D (licensed) | [Q8](notebooks/Q8-existing-tools.ipynb) | [Gao et al. (2023)](https://doi.org/10.1126/science.abn8197) | — | — | — | — |

Runtime covers the recorded stages listed in the details below; hardware and caching differ between workflows. “—” means no verified timing is available for the current cohort.

**Direct comparison on the same variants**

| Method | Shared variants | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- | --- |
| SIFT4G (Q8) | 8817 | 0.887 [0.872, 0.901] | 0.881 [0.853, 0.906] |
| PolyPhen-2 (Q8) | 8817 | 0.900 [0.887, 0.913] | 0.903 [0.879, 0.925] |
| REVEL (Q8) | 8817 | 0.972 [0.966, 0.978] | 0.976 [0.967, 0.984] |
| AlphaMissense (Q8) | 8817 | 0.961 [0.953, 0.968] | 0.967 [0.957, 0.975] |
| EVE (Q8) | 8817 | 0.911 [0.893, 0.928] | 0.921 [0.899, 0.940] |

<details>
<summary>Provenance, missing results and limitations</summary>

| Method | Details |
| --- | --- |
| Evo2 1B base zero-shot (Vortex, FP8) (Q2) | Q2 cohort is stale |
| SIFT4G (Q8) | dbNSFP4.9a; 1 minus the minimum raw SIFT4G score. Evolutionary sequence exposure is unaudited. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| PolyPhen-2 (Q8) | HumVar model from dbNSFP4.9a; maximum raw score. Known disease training variants may overlap ClinVar. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| REVEL (Q8) | HGMD and constituent-tool training overlap with ClinVar unresolved; maximum exact-allele score across transcript annotations. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training. |
| AlphaMissense (Q8) | ClinVar calibration overlap unresolved; maximum matching transcript score. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training. |
| EVE (Q8) | dbNSFP4.9a continuous EVE score; maximum across matches, no confidence-category filtering. Limited protein/position coverage. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training. |
| Evo2 1B base frozen head (BioNeMo, BF16) (Q9) | Q9 cohort is stale |
| Sequence baseline (Evo2 1B experiment) (Q9) | Q9 cohort is stale |
| Evo2 7B base frozen head (BioNeMo, BF16) (Q10) | Export cohort is stale |
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
  "generated_utc": "2026-09-07T20:31:45.583530+00:00"
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

These questions support the [project goals](GOALS.md): ClinVar missense pathogenicity
prediction, using frozen Evo2 representations or light fine-tuning. Evo2 already has
published variant-effect results, so the strongest contribution would address
generalization, reliability, and when its representations add value.
([Evo2 paper](https://www.nature.com/articles/s41586-026-10176-5))

### [Q0. What does ClinVar contain about missense variants, and which are suitable for reliable evaluation?](notebooks/Q0-clinvar-summary.ipynb)

Explore missense annotations, genes, pathogenicity labels, review status, missing
annotations, conflicting classifications, and duplicate or related records.
Use plots to reveal class imbalance and potential leakage, then define a
missense-only dataset and filtering criteria.

### [Q1. How should ClinVar missense variants be split for reliable evaluation on previously unseen genes?](notebooks/Q1-clinvar-split.ipynb)

Prepare all eligible missense SNVs from **6 July 2026**, using the same
`data/clinvar_20260706.vcf` input as Q0, the existing quality filters and exact
ClinVar `MC` annotation `SO:0001583`. Preserve earlier splits for variants shared
with September; new groups follow the seed-42, 70/30 assignment.
The full dataset contains **64,815 variants: 46,888 training and 17,927 validation**.
Check genes, loci, source IDs and overlapping or identical sequence contexts, then
export [clinvar-train.vcf](data/clinvar-train.vcf) and
[clinvar-test.vcf](data/clinvar-test.vcf) as the full inputs for new experiments.
The latter file contains validation data. September inputs and completed results
are preserved in `notebooks/results/archive/before_shared_july_snapshot/`.

### [Q2. Can a small classifier on frozen Evo2 representations outperform zero-shot Evo2 scoring for missense variants in previously unseen genes?](notebooks/Q2-evo2-classifier.ipynb)

Compare logistic regression on frozen Evo2 1B representations with zero-shot
scores and a DNA-only sequence baseline: missense SNVs, 1,024-base contexts,
gene-disjoint splits, and paired component bootstrap intervals. Genes, source
IDs, loci and overlapping contexts are grouped across the full cohort before
sampling; identical pilot contexts, including reverse complements, also stay
together. Clinical annotations are excluded from predictor features.

The 1B classifier comparison is preserved with the archived September pilot.
The Evo2 7B base zero-shot extension uses Vortex/FP8 to score all **17,927 July
validation variants**, without fitting a classifier or choosing a threshold.
Its new README row uses the complete current cohort; historical scores are kept
separate. A separate untouched holdout is required for final performance claims.

### Q3. How much does apparent missense predictive performance depend on similarities between training and test data?

Compare conventional random splits with progressively stricter locus-,
sequence-context-, and gene-separated splits. Treat random splits as a diagnostic
benchmark; quantify how much performance survives credible leakage controls.

### Q4. Does longer sequence context improve missense pathogenicity prediction?

Vary context length while holding the checkpoint, classifier, and evaluation
missense variants fixed. Measure predictive gains and computational cost, with
gene-level comparisons where sample sizes permit.

### Q5. Is it better to train on fewer strongly supported ClinVar missense labels or more labels with weaker supporting evidence?

Compare training sets filtered by review status, including size-matched
comparisons, against a fixed, strongly reviewed holdout. ClinVar's review status
captures review processes and agreement, making this a useful test of label
selection.
([ClinVar documentation](https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/))

### Q6. Can the predictor identify when its missense predictions are unreliable?

Evaluate calibration and whether withholding low-confidence predictions reduces
errors on unseen genes. Report the relationship between retained coverage and
error rate, alongside discrimination metrics.

### Q7. Can a model trained on an older ClinVar snapshot predict subsequently resolved missense variants of uncertain significance?

Freeze training labels at an earlier release and evaluate variants that later
receive clear classifications. This requires historical snapshots and careful
provenance checks, but would test usefulness beyond reproducing existing labels.

### [Q8. Which existing missense predictors are practical baselines, and how do they perform on the full validation dataset?](notebooks/Q8-existing-tools.ipynb)

Compare six missense predictors and evaluate AlphaMissense, REVEL, SIFT4G,
PolyPhen-2 HumVar and EVE using pinned published scores on all **17,927** variants
in Q1's full GRCh38 validation VCF from **6 July 2026**. Preserve missing scores,
report coverage and performance with component-bootstrap intervals, and compare
available methods on the same scored variants. Historical pilot results remain archived;
PrimateAI-3D requires an approved licensed download. ClinVar remains the ground
truth; external calibration and training overlap limit independence claims.

### [Q9. Can light fine-tuning of Evo2 1B base improve missense pathogenicity prediction compared with frozen representations and zero-shot scoring?](notebooks/Q9-evo2-1b.ipynb)

Train Hyena block 23 alongside a classification head, keeping attention block 24 frozen,
using Q1's fixed pilot partitions, 1,024-base sequence contexts and the
Evo2 1B base BioNeMo checkpoint.
Compare validation performance, training time and GPU memory with frozen and
zero-shot baselines produced by the same BioNeMo model, plus a sequence baseline.
Use the [pinned BioNeMo tutorial](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/ca16c2acf9bf813d020b6d1e2d4e1240cfef6a69/docs/docs/user-guide/examples/bionemo-evo2/fine-tuning-tutorial.ipynb)
as the training scaffold, adding selective weight updates and supervised missense classification.

### [Q10. Does a frozen Evo2 7B classifier improve on the 1B BioNeMo configuration?](notebooks/Q10-evo2-7b.ipynb)

Extract frozen Evo2 7B base features with BioNeMo in BF16 and fit the same
training-only scaler and balanced logistic regression used by the 1B baseline.
Keep Q1's missense partitions, 1,024-base contexts and validation selection fixed.
Report paired component-bootstrap comparisons and compute usage. The original
1B checkpoint is BF16-sensitive, so differences do not isolate model size alone.

## Setup

See [SETUP.md](SETUP.md) for a fresh CPU installation, the NVIDIA GPU container,
notebook execution and transferring existing data and experiments. Quick CPU setup:

```bash
git clone --recurse-submodules https://github.com/pesho-ivanov/pathogenity-predictor.git
cd pathogenity-predictor
python3.12 scripts/setup.py --profile cpu --test
.venv/bin/jupyter lab
```

For a GPU machine, use `nvcr.io/nvidia/pytorch:25.04-py3` and run
`python scripts/setup.py --profile gpu --test` inside the container.

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

## Next steps

Use the frozen July partitions throughout development: **46,888 training and
17,927 validation variants**. Work in this order, keeping completed results and
their protocols reproducible.

1. **Complete the full-cohort Evo2 baselines (Q2, Q9, Q10).** Finish zero-shot
   scoring and train the sequence baseline and frozen 1B/7B classifiers on the
   full training partition. Fit preprocessing on training data only. Publish
   current-cohort predictions through fully executed notebooks and refresh the
   comparison above; preserve the archived pilot runs.
2. **Explain coverage and audit independence (Q8).** Break down missing published
   scores by gene, class and annotation/matching failure. Investigate overlap
   with predictor training and calibration data, including related proteins and
   pretraining sequences, and document what remains unknown. Add PrimateAI-3D
   when an approved licensed score file becomes available.
3. **Measure whether Evo2 adds value (Q2, Q8–Q10).** Compare completed methods on
   the same variants, reporting paired component-bootstrap intervals for
   differences in AUROC and average precision. Keep coverage visible alongside
   these comparisons, and report runtime and memory with their measurement
   scope. Use the results to choose the simplest justified approach.
4. **Test targeted improvements (Q4, Q9).** Once the baselines are complete,
   evaluate longer context or light fine-tuning where the results justify the
   added cost. Change one methodological choice at a time and retain the fixed
   partitions. Recheck sequence and group separation before each experiment;
   stop if a context change makes the split incompatible.
5. **Evaluate confidence and abstention (Q6).** Fit calibration using
   training-only folds, select operating thresholds on development validation,
   and report calibration error and error rates at different retained coverage
   levels. Include uncertainty across held-out gene groups.
6. **Prepare an untouched final evaluation (Q7).** Define and freeze a separate
   temporal holdout, with variant, gene and sequence separation checks, before
   inspecting its outcomes. Lock the selected model, calibration and thresholds
   before scoring it. The current `clinvar-test.vcf` remains development
   validation; external training overlap must still be reported.
