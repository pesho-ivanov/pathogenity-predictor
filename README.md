# Pesho's Pathogenicity Predictor

A small research project for predicting the pathogenicity of **missense variants**
using ClinVar data.

The evaluation tables use the preserved **5,000-variant pilot**. Q1 provides
separately frozen full datasets for new experiments.

<!-- comparison:start -->
## Method comparison

Compare methods on Q1’s frozen missense **pilot** validation set. These scores do not evaluate the full VCF exports.

**1,342 missense validation variants · ClinVar labels**

| Method | Notebook | Paper | Scored / validation | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Evo2 1B base zero-shot (Vortex, FP8) | [Q2](notebooks/Q2-evo2-classifier.ipynb) | — | 1,342 / 1,342 | 0.866 [0.840, 0.890] | 0.769 [0.713, 0.817] |
| SIFT4G | [Q8](notebooks/Q8-existing-tools.ipynb) | [Vaser et al. (2016)](https://doi.org/10.1038/nprot.2015.123) | 1,270 / 1,342 | 0.897 [0.877, 0.915] | 0.798 [0.748, 0.840] |
| PolyPhen-2 | [Q8](notebooks/Q8-existing-tools.ipynb) | [Adzhubei et al. (2010)](https://doi.org/10.1038/nmeth0410-248) | 1,229 / 1,342 | 0.903 [0.884, 0.921] | 0.827 [0.775, 0.872] |
| REVEL | [Q8](notebooks/Q8-existing-tools.ipynb) | [Ioannidis et al. (2016)](https://doi.org/10.1016/j.ajhg.2016.08.016) | 1,330 / 1,342 | 0.978 [0.970, 0.985] | 0.960 [0.941, 0.975] |
| AlphaMissense | [Q8](notebooks/Q8-existing-tools.ipynb) | [Cheng et al. (2023)](https://doi.org/10.1126/science.adg7492) | 1,276 / 1,342 | 0.967 [0.958, 0.976] | 0.943 [0.922, 0.960] |
| EVE | [Q8](notebooks/Q8-existing-tools.ipynb) | [Frazer et al. (2021)](https://doi.org/10.1038/s41586-021-04043-8) | 705 / 1,342 | 0.912 [0.888, 0.934] | 0.906 [0.869, 0.937] |
| Evo2 1B base frozen head (BioNeMo, BF16) | [Q9](notebooks/Q9-evo2-1b.ipynb) | — | 1,342 / 1,342 | 0.654 [0.614, 0.690] | 0.552 [0.475, 0.618] |
| Sequence baseline (Evo2 1B experiment) | [Q9](notebooks/Q9-evo2-1b.ipynb) | — | 1,342 / 1,342 | 0.563 [0.529, 0.598] | 0.390 [0.330, 0.455] |
| Evo2 7B base frozen head (BioNeMo, BF16) | [Q10](notebooks/Q10-evo2-7b.ipynb) | — | 1,342 / 1,342 | 0.881 [0.856, 0.903] | 0.803 [0.753, 0.843] |
| PrimateAI-3D (licensed) | [Q8](notebooks/Q8-existing-tools.ipynb) | [Gao et al. (2023)](https://doi.org/10.1126/science.abn8197) | — | — | — |

**Direct comparison on the same variants**

| Method | Paper | Shared variants | AUROC [95% CI] | Average precision [95% CI] |
| --- | --- | --- | --- | --- |
| Evo2 1B base zero-shot (Vortex, FP8) (Q2) | — | 672 | 0.851 [0.816, 0.885] | 0.846 [0.798, 0.889] |
| SIFT4G (Q8) | [Vaser et al. (2016)](https://doi.org/10.1038/nprot.2015.123) | 672 | 0.901 [0.875, 0.925] | 0.882 [0.832, 0.920] |
| PolyPhen-2 (Q8) | [Adzhubei et al. (2010)](https://doi.org/10.1038/nmeth0410-248) | 672 | 0.911 [0.888, 0.932] | 0.906 [0.863, 0.938] |
| REVEL (Q8) | [Ioannidis et al. (2016)](https://doi.org/10.1016/j.ajhg.2016.08.016) | 672 | 0.971 [0.959, 0.982] | 0.970 [0.950, 0.983] |
| AlphaMissense (Q8) | [Cheng et al. (2023)](https://doi.org/10.1126/science.adg7492) | 672 | 0.966 [0.952, 0.978] | 0.967 [0.950, 0.980] |
| EVE (Q8) | [Frazer et al. (2021)](https://doi.org/10.1038/s41586-021-04043-8) | 672 | 0.911 [0.889, 0.934] | 0.909 [0.872, 0.938] |
| Evo2 1B base frozen head (BioNeMo, BF16) (Q9) | — | 672 | 0.651 [0.604, 0.698] | 0.686 [0.615, 0.753] |
| Sequence baseline (Evo2 1B experiment) (Q9) | — | 672 | 0.577 [0.530, 0.626] | 0.580 [0.499, 0.655] |
| Evo2 7B base frozen head (BioNeMo, BF16) (Q10) | — | 672 | 0.879 [0.850, 0.909] | 0.888 [0.853, 0.919] |

<details>
<summary>Provenance, missing results and limitations</summary>

| Method | Details |
| --- | --- |
| Evo2 1B base zero-shot (Vortex, FP8) (Q2) | Development result; validation participates in model selection. |
| SIFT4G (Q8) | dbNSFP4.9a; 1 minus the minimum raw SIFT4G score. Evolutionary sequence exposure is unaudited. |
| PolyPhen-2 (Q8) | HumVar model from dbNSFP4.9a; maximum raw score. Known disease training variants may overlap ClinVar. |
| REVEL (Q8) | HGMD and constituent-tool training overlap with ClinVar unresolved; maximum exact-allele score across transcript annotations. |
| AlphaMissense (Q8) | ClinVar calibration overlap unresolved; maximum matching transcript score. |
| EVE (Q8) | dbNSFP4.9a continuous EVE score; maximum across matches, no confidence-category filtering. Limited protein/position coverage. |
| Evo2 1B base frozen head (BioNeMo, BF16) (Q9) | Development result; validation participates in model selection. |
| Sequence baseline (Evo2 1B experiment) (Q9) | Development result; validation participates in model selection. |
| Evo2 7B base frozen head (BioNeMo, BF16) (Q10) | Validation selects C; pretraining and homology overlap remain unresolved. The 1B baseline uses the original BF16-sensitive checkpoint, so this is a comparison of configurations, not an isolated model-size effect. |
| PrimateAI-3D (Q8) | PrimateAI-3D has not been run: Illumina requires a signed license agreement and supplies the score/model download link by email. No approved link or licensed score file was provided. The original PrimateAI scores in dbNSFP are a different model and are not substituted. |

```json
{
  "cohort": {
    "protocol_sha256": "0b1b21eb6fb829884d1f8bd09c0314a34ae2acddb018f239d6346f7933566409",
    "vcf_exports": {
      "clinvar-test-pilot.vcf": "1007a3b5229199203c1174e78d430ad0195b9f33ea258af1829077223e72f4eb",
      "clinvar-train-pilot.vcf": "db7bf9cd1b743e2a9f7045a7053f268b9f568dd9726347af286e224c1b572ea6"
    },
    "validation_variants": 1342
  },
  "source_errors": {},
  "bootstrap": {
    "repetitions": 1000,
    "seed": 42
  },
  "generated_utc": "2026-09-07T18:50:04.520678+00:00"
}
```

</details>

**Conclusion.** Use the common-subset comparison to assess methods; coverage remains a separate limitation.

These are development results: validation participates in model selection. AlphaMissense calibration and REVEL/PolyPhen-2 training overlap with ClinVar remain unresolved. The 95% intervals resample whole Q1 components and do not correct selection bias or establish clinical validity.
<!-- comparison:end -->

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

Prepare all 65,270 eligible missense SNVs from the September snapshot, using the
pilot's quality filters and exact ClinVar `MC` annotation `SO:0001583`. Preserve
the original 70/30 group assignment: 47,230 training and 18,040 validation variants.
Check genes, loci, source IDs and overlapping or identical sequence contexts, then
export [clinvar-train.vcf](data/clinvar-train.vcf) and
[clinvar-test.vcf](data/clinvar-test.vcf) as the full inputs for new experiments.
The latter file contains validation data. Earlier pilot inputs remain available
to reproduce their recorded evaluations.

### [Q2. Can a small classifier on frozen Evo2 representations outperform zero-shot Evo2 scoring for missense variants in previously unseen genes?](notebooks/Q2-evo2-classifier.ipynb)

Compare logistic regression on frozen Evo2 1B representations with zero-shot
scores and a DNA-only sequence baseline: missense SNVs, 1,024-base contexts,
gene-disjoint splits, and paired component bootstrap intervals. Genes, source
IDs, loci and overlapping contexts are grouped across the full cohort before
sampling; identical pilot contexts, including reverse complements, also stay
together. Clinical annotations are excluded from predictor features.

The recorded Q2 experiment uses Q1's fixed pilot VCFs; new full-dataset experiments
use the full exports. A separate untouched holdout is required for final performance claims.

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

### [Q8. What in silico tools currently exist for predicting missense variant pathogenicity, and which are suitable baselines for this project?](notebooks/Q8-existing-tools.ipynb)

Compare six missense predictors and evaluate AlphaMissense, REVEL, SIFT4G,
PolyPhen-2 HumVar and EVE using pinned published scores on Q1's fixed GRCh38 pilot.
Report coverage and validation performance with component-bootstrap intervals;
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

## Data

Q0 automatically downloads the dated **6 July 2026** ClinVar snapshot into `data/`
when missing. Q1/Q2 retain their separately pinned **5 September 2026** pilot input.
See [data/README.md](data/README.md) for filenames and checksum checks. Q1 also writes the two shared VCF
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
