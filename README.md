# Pathogenicity Prediction

## Problem statement

This is a **binary classification** task: predict whether a single-nucleotide
**missense variant** is **benign (0)** or **pathogenic (1)** using ClinVar labels,
merging “likely benign” and “likely pathogenic” into their respective classes.
Uncertain, conflicting, other, and missing labels are excluded from training
and evaluation.

The model outputs a continuous pathogenicity score. Applying a threshold converts
that score into a binary prediction; we currently evaluate the scores using
AUROC and average precision.

### Basic classification approaches

- **Majority-class baseline:** always predict the most common training label.
- **Single-score threshold:** predict pathogenic when one score exceeds a cutoff.
- **Logistic regression:** learn a weighted combination of features and map it
  through a sigmoid to a probability.
- **Shallow decision tree:** learn a small set of feature-based decision rules.
- **Naive Bayes:** combine feature evidence assuming conditional independence.
- **k-nearest neighbors:** use the labels of the most similar training examples.

Our Evo2 predictor uses a linear logistic classifier on sequence representations.
Q14 and Q16 hold that classifier fixed while LoRA adapts the representations. The other
approaches above describe basic alternatives; they are not all implemented or
evaluated in this project. No clinical probability calibration or classification
cutoff has been validated for these LoRA runs.

See the [Q16 results](#q16-does-longer-fine-tuning-improve-on-q14) for the larger
fine-tuning run and its full validation comparison.

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

**17,927 missense validation variants · ClinVar 2026-07-06 · full cohort**

| Method | Notebook | Paper | Scored / validation | AUROC [95% CI] | Average precision [95% CI] | Runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Evo2 7B Q16 continued LoRA (blocks 29 and 30, rank 8, 512 bp) | [Q16](notebooks/Q16-lora-continuation.ipynb) | — | 17,927 / 17,927 | 0.872 [0.859, 0.884] | 0.801 [0.769, 0.828] | 3.7 h |
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

Coverage differs among tools: each AUROC and average precision uses the scored variants shown. These rows do not establish a ranking on identical variants. “—” indicates an unavailable result or timing.

Runtime covers the recorded stages described below; hardware and caching differ. The Q16 row includes preparation through completion of its saved notebook. Sampled checkpoint-selection results and shared-variant analyses remain in the linked notebooks and [preserved comparison records](notebooks/results/comparison/published/results.json).

<details>
<summary>Provenance, missing results and limitations</summary>

**Evo2 7B Q16 continued LoRA (blocks 29 and 30, rank 8, 512 bp) (Q16).** The 2,048-variant sample selects checkpoints; full validation reports the frozen selection. The full partition and its subsets have been evaluated before and are development data, not an untouched test set. Component-bootstrap intervals do not correct repeated selection or establish clinical validity. Pretraining, homology and external-data overlap remain unresolved. Single-variant BF16 scoring is numerically sensitive; Q16 uses its verified batch-32 workflow. Runtime: Original request through the saved notebook, including preparation, the failed preflight, training, all three full-validation passes, reload checks and confidence intervals. Excludes inherited Q12/Q14 fitting and initial model acquisition.

**Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp) (Q11).** User-authorized partial-epoch training for a one-hour run; full frozen validation is retained. No fitted frozen baseline or improvement claim; these are development results. Pretraining sequence exposure, homology and shared-patient overlap remain unresolved. The final attention block is numerically inactive in this BF16 configuration; LoRA targets the preceding Hyena mixer. No clinical validity or independent final-test performance is established. Runtime: Fresh notebook start through input checks, cached setup verification, preflight, partial-epoch training, full validation, 64-variant reload check and bootstrap; excludes initial environment/model acquisition and final notebook/README export.

**Evo2 7B Q14 LoRA (blocks 29 and 30, rank 8, 512 bp) (Q14).** Prespecified full-validation confirmation rule met. Full frozen July missense development validation, following repeated selection on its 2,048-variant subset. Variants outside that subset and components absent from it were not used by those recent selection runs, but this repository has previously evaluated the full validation partition. Neither subset is an untouched final test set. Component-bootstrap intervals do not remove selection bias. Pretraining exposure, homology and shared-patient overlap remain unresolved; this research prototype has no clinical validation. Runtime: Current Q14 exploration (43.0 min, including all its search arms) plus separate full validation (34.7 min), shared by LoRA and controls. Excludes inherited Q12 feature extraction/fitting and earlier experiments or attempts. The one-hour target applies only to exploration.

**Evo2 7B Q14 strongest frozen classifier (magnitude features, 512 bp) (Q14).** Prespecified full-validation confirmation rule met. Full frozen July missense development validation, following repeated selection on its 2,048-variant subset. Variants outside that subset and components absent from it were not used by those recent selection runs, but this repository has previously evaluated the full validation partition. Neither subset is an untouched final test set. Component-bootstrap intervals do not remove selection bias. Pretraining exposure, homology and shared-patient overlap remain unresolved; this research prototype has no clinical validation. Runtime: Current Q14 exploration (43.0 min, including all its search arms) plus separate full validation (34.7 min), shared by LoRA and controls. Excludes inherited Q12 feature extraction/fitting and earlier experiments or attempts. The one-hour target applies only to exploration.

**Evo2 7B base zero-shot (Vortex, FP8) (Q2).** Zero-shot inference on the complete current full validation cohort. No parameters or thresholds fitted. Archived September results are not mixed with this run. Pretraining, homology and annotation overlap remain unresolved; validation is development data, not an untouched final test. Published executed-notebook result; per-variant export is absent locally. Runtime: Validation scoring batches summed across runs; excludes downloads, model loading and evaluation. Published rounded duration.

**SIFT4G (Q8).** dbNSFP4.9a; 1 minus the minimum raw SIFT4G score. Evolutionary sequence exposure is unaudited. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training.

**PolyPhen-2 (Q8).** HumVar model from dbNSFP4.9a; maximum raw score. Known disease training variants may overlap ClinVar. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training.

**REVEL (Q8).** HGMD and constituent-tool training overlap with ClinVar unresolved; maximum exact-allele score across transcript annotations. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training.

**AlphaMissense (Q8).** ClinVar calibration overlap unresolved; maximum matching transcript score. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training.

**EVE (Q8).** dbNSFP4.9a continuous EVE score; maximum across matches, no confidence-category filtering. Limited protein/position coverage. Published executed-notebook result; per-variant export is absent locally. Runtime: CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training.

**PrimateAI-3D (licensed) (Q8).** Requires licensed data; no predictions for the current cohort.

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
  "errors": {},
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
  "generated_utc": "2026-09-08T21:32:52.873215+00:00"
}
```

</details>

These are development results: validation participates in model selection. The 95% intervals resample whole Q1 components and do not correct selection bias. Pretraining exposure and external-tool training/calibration overlap with ClinVar remain unresolved; clinical validity has not been established.
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

<!-- q16-results:start -->
## Q16. Does longer fine-tuning improve on Q14?

[Q16 continued LoRA](notebooks/Q16-lora-continuation.ipynb) trained on all **46,888 variants**, processing **132,592 examples** in **4,145 updates**. It continued Q14’s adapters at learning rate **3e-5**, with the classifier and scaler fixed. Full metrics and coverage appear in the [main comparison](#method-comparison).

The tested continuation improved full-validation AUROC without reducing AP versus Q14. Continuation minus Q14: AUROC **+0.0204 [+0.0160, +0.0249]**, AP **+0.0316 [+0.0250, +0.0384]**. The sample-based decision retains **Q16 continued LoRA**. The best nonzero-update continuation is chosen by sample AUROC, then AP; it replaces Q14 only if sample AUROC strictly increases and AP does not decrease. Full results do not change that decision. The 2,048-variant sample selects checkpoints; full validation reports the frozen selection. The full partition and its subsets have been evaluated before and are development data, not an untouched test set. Component-bootstrap intervals do not correct repeated selection or establish clinical validity. Pretraining, homology and external-data overlap remain unresolved. Single-variant BF16 scoring is numerically sensitive; Q16 uses its verified batch-32 workflow.

The 2,048-variant selection sample scored **0.869 AUROC / 0.777 average precision**. Fresh notebook execution took **202.4 minutes**; the runtime reported in the main table is **220.0 minutes** through notebook completion, with its scope recorded in the comparison details.

The [preserved Q16 record](notebooks/results/comparison/published/q16-results.json) pins the executed notebook, displayed metrics, source hashes, cohort and measured runtime.
<!-- q16-results:end -->

## Experiment details

The answers below describe the completed **Q14 Evo2 7B experiment** and the
preceding Q11–Q13 attempts. **All reported performance is development validation;
there is no untouched test set.** AP means **average precision across
thresholds**, not precision at a selected cutoff.

<details>
<summary>Variant counts, ClinVar release, labels and split construction</summary>

These are the counts after eligibility filtering. The exploratory rows are
subsets of the full partitions and must not be added to them.

| Cohort | Total | Pathogenic / likely pathogenic | Benign / likely benign |
| --- | ---: | ---: | ---: |
| Full eligible modeling cohort | **64,815** | **24,696** | **40,119** |
| Full training partition | 46,888 | 18,173 | 28,715 |
| Full validation partition | 17,927 | 6,523 | 11,404 |
| Q14 registered training subset | 24,576 | 9,616 | 14,960 |
| Actually processed by each Q14 adapter trial | **16,384** | **6,467** | **9,917** |
| Validation sample used for model selection | 2,048 | 688 | 1,360 |

Both adapter trials processed the **same 16,384 unique variants**, once per
trial. The inherited classifier and scaler used the larger 24,576-variant
training subset. Counts were verified against Q1's frozen label files, Q14's
membership lists and the saved training-order hashes.

**Release and assembly.** The input is ClinVar **6 July 2026**,
`clinvar_20260706.vcf`, on **GRCh38**. Sequence contexts come from the pinned UCSC
`hg38.fa.gz` reference. File dates, checksums and every retained REF allele are
checked. [Input sources and checksums](data/README.md).

**Variant types.** Only missense **SNVs** are modeled: one A/C/G/T REF base and
one different A/C/G/T ALT base, on chromosomes 1–22, X or Y. Indels and multi-base
substitutions are excluded. Eligibility requires ClinVar's exact `MC` missense
term `SO:0001583`, at least **two review stars**, an accepted label, no germline
conflict evidence and usable gene identifiers. Consistent duplicates are
collapsed; inconsistent duplicates are excluded. Truncated or non-ACGT sequence
windows are excluded, and a reference mismatch stops execution.

**Binary labels.** Only these exact `CLNSIG` strings are accepted:

| Integer label | Accepted classifications |
| --- | --- |
| **0: benign** | `Benign`, `Likely_benign`, `Benign/Likely_benign` |
| **1: pathogenic** | `Pathogenic`, `Likely_pathogenic`, `Pathogenic/Likely_pathogenic` |

Uncertain, conflicting, missing and other classifications are excluded; labels
are not inferred from free text. Eligibility annotations and labels are not
predictor features. [Label mapping and filters](notebooks/src/q0.py) ·
[Missense eligibility](notebooks/src/q1.py).

**Splitting.** The original assignment is label-independent, with a **70/30
training/validation target** and seed **42**. Connected components join shared
genes, variation IDs, allele IDs, variant keys, loci and transitive overlaps of
the original **1,024-base** contexts. Qualifying nonmissense SNVs remain in this
graph as bridges; only missense SNVs enter modeling. Identical or
reverse-complement contexts are also joined.

The initial component anchor is its minimum variant key. Assignment uses
`int(SHA256("42:split:" + anchor), 16) / 2**256`: values below 0.70 go to
training. Inherited assignments are preserved, including recorded overrides;
the full cohort is not rebalanced or newly stratified. A later sequence merge
that would connect opposite splits stops execution instead of moving variants.
The realized variant counts therefore differ from exactly 70/30.

**Gene/locus separation.** Audits found zero cross-split overlap in gene IDs,
loci, components, variation IDs and allele IDs. Identical or reverse-complement
512-base allele contexts also did not cross splits. Homology between different
genes, shared-patient overlap and pretraining sequence exposure remain unresolved.

Q12–Q14 reuse the same seed-42 training subset and uniform 2,048-variant
validation sample without replacement. `clinvar-train.vcf` is training;
**`clinvar-test.vcf` is development validation despite its filename**.
[Grouping and original assignment](notebooks/src/q1.py) ·
[Frozen full-cohort protocol](notebooks/src/q1_full.py) ·
[Q14 membership checks](notebooks/src/q14.py).

</details>

<details>
<summary>Evo2 checkpoint, sequence window and exact score formula</summary>

The checkpoint is **`arcinstitute/savanna_evo2_7b_base`**, revision
`eb0a7478e5f3c291f31e2b3d9ec14fc067f9982a`, file
`savanna_evo2_7b_base.pt`. Its SHA-256 is
`ed8d264c14fea3c6305b475122e068e09009cbdabc05e1653764147c1b294cd4`.
The measured loaded backbone contains **6,481,649,408 parameters**, with
**32 layers** and hidden width **4,096**. Q14 uses the BioNeMo/NeMo conversion
in **BF16**, with FP8 disabled. All 325 audited converted tensors matched the
source; native Savanna/Vortex forward-output parity remains unresolved.
[Model configuration](notebooks/src/q10.py) ·
[Loading and conversion audit](notebooks/src/q11_backend.py).

Each allele uses **512 nucleotides**: 256 before the mutation, the mutated
position and 255 afterward. The reference/alternate pair is canonicalized
against its reverse complement **before cropping** from the frozen 1,024-base
contexts. Q14 uses one canonical orientation, producing two sequences per
variant. [Canonical crop](notebooks/src/q11.py).

For the Q14 classifier, let $h_r$ and $h_a$ be the 4,096-dimensional means of
the last decoder block's activations over all 512 positions, **before final
normalization**. Pooling is computed in FP32. Define:

$$
d=h_a-h_r,\qquad
R(v)=\sqrt{\frac{1}{4096}\sum_j v_j^2},\qquad
U(v)=\frac{v}{\max(R(v),10^{-6})},\qquad
M(v)=\max(R(v),10^{-12}).
$$

The **8,194 features** retain both direction and mutation magnitude:

$$
\phi=\left[U(h_r),\ U(d),\ \ln M(d),
\ln\max\left(\frac{M(d)}{M(h_r)},10^{-12}\right)\right].
$$

The reported score is the classifier's raw logit:

$$
z=w^\top\left(\frac{\phi-\mu}{s}\right)+b.
$$

The mean $\mu$ and population standard deviation $s$ are fitted on training
features only, with each standard deviation floored at $10^{-6}$.
The classifier and scaler are inherited from Q12 and remain fixed during Q14.
**Higher $z$ means more pathogenic.** AUROC and AP use $z$; a sigmoid would
preserve ranking, but its output is not established as a calibrated clinical
probability. [Feature formula](notebooks/src/q12_backend.py) ·
[Training-only standardization](notebooks/src/q12.py).

The separate **Q2 zero-shot row** uses a different score:

$$
S_{\mathrm{zero-shot}}=\bar{\ell}(x_r)-\bar{\ell}(x_a),\qquad
\ell(x)=\frac{1}{L-1}\sum_{t=2}^{L}\ln p(x_t\mid x_{<t}),\quad L=1024.
$$

Here $\bar{\ell}$ averages the forward and reverse-complement sequence
likelihoods, without a beginning-of-sequence token. Higher scores again mean more
pathogenic: the alternate sequence is less likely than the reference.
[Q2 zero-shot definition](notebooks/src/q2_7b.py).

</details>

<details>
<summary>LoRA settings, trainable parameters, optimization, GPU time and cost</summary>

Q14 uses **rank 8**, **alpha 16** (scaling $\alpha/r=2$), **dropout 0**,
Xavier-normal A matrices and zero B matrices. Targets use zero-based layer
indices:

```text
decoder.layers.29.mixer.dense_projection
decoder.layers.29.mixer.dense
decoder.layers.30.mixer.dense_projection
decoder.layers.30.mixer.dense
```

There are **393,216 trainable adapter parameters** across eight matrices. The
**6,481,649,408 backbone parameters** and **8,195 classifier parameters** are
frozen. Classifier and scaler preservation, actual adapter updates and
checkpoint reload predictions were checked.
[Adapter implementation](notebooks/src/q14_adapters.py).

| Setting | Q14 value |
| --- | --- |
| Loss | Class-balanced binary cross entropy with logits |
| Class weights | $24,576/(2N_c)$; benign ≈0.821390, pathogenic ≈1.277870 |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon $10^{-8}$ |
| Adapter weight decay | 0.01 |
| Peak learning rates tested | $3\times10^{-5}$ and **$10^{-4}$, selected** |
| Schedule | 32-step linear warmup, then cosine decay to 10% of peak |
| Gradient clipping | Adapter gradient norm at 1.0 |
| Batch size | **32 variants**, or **64 allele sequences** |
| Gradient accumulation | One batch per optimizer update |
| Updates | **512 per trial**; selected checkpoint at step 512 |
| Validation checkpoints | Steps 0, 256 and 512 |
| Seed | 42 |
| Precision | BF16 backbone/adapters; FP32 pooling, classifier and optimizer masters |

For a batch of $B=32$ variants, the loss is
$\frac{1}{B}\sum_i c_{y_i}[\log(1+\exp(z_i))-y_i z_i]$, evaluated with the
stable BCE-with-logits implementation. The class weights use the registered
training subset: $N_0=14,960$, $N_1=9,616$.
Only adapters enter the optimizer; the inherited classifier has learning rate
zero. The two trials start from the same fixed Q12 classifier and fresh
zero-output adapters and use identical batch order.
[Frozen configuration](notebooks/src/q14.py) ·
[Loss, optimizer and schedule](notebooks/src/q14_backend.py).

**Hardware and timing.** The run used **one NVIDIA H100 PCIe, 80 GB**.
The fully executed Q14 exploration took **43.0 minutes**, including both
learning-rate trials, sampled validation and integrity checks. Full confirmation
took **34.7 minutes** separately: **77.7 minutes combined**. Initial setup,
inherited feature extraction/head fitting and earlier experiments are additional.
The one-hour target applied only to exploration.

**Cost.** Actual billing was not recorded. At an hourly instance rate $R$,
these elapsed times correspond to approximately **$0.717R$** for exploration
and **$1.295R$** including confirmation, excluding earlier work and other
charges. These are rate-based estimates, not a recorded invoice.
[Completed runtime records](notebooks/results/comparison/published/results.json).

</details>

<details>
<summary>Best completed 7B results and failed attempts</summary>

The [main comparison](#method-comparison) reports completed full-cohort results.
The linked source notebooks retain the individual checkpoint measurements,
sampled trials and negative outcomes. Paired effects are reported in
[Q14 confirmation](#q14-full-cohort-confirmation) and the
[Q16 continuation](#q16-does-longer-fine-tuning-improve-on-q14).
There are **no untouched-test results**.

Q12's best frozen classifier scored **0.831970 AUROC / 0.724542 AP** on the
selection sample. Q13's own best matched control scored **0.829448 / 0.721583**,
so its small adapter gain did not beat the inherited Q12 classifier. Q14 retained
that strongest Q12 classifier as its fixed control. Sampled gains and full-cohort
gains must be compared within their respective cohorts.
[Q11](notebooks/Q11-evo2-lora.ipynb) ·
[Q12](notebooks/Q12-lora-improvement.ipynb) ·
[Q13](notebooks/Q13-lora-optimization.ipynb) ·
[Q14 exploration](notebooks/Q14-layer-adapters.ipynb) ·
[Q14 full validation](notebooks/Q14-lora-validation.ipynb).

- The early **Q11 block-31 attempt** stopped at preflight because diagnostic
  updates produced no effective BF16 prediction changes. It has no valid
  supervised performance score. See the [executed numerical
  investigation](notebooks/Q11-gradient-diagnostics.ipynb) and
  [failure discussion](#q11-ineffective-final-attention-block).
- The initial **Q13 FP32 head-fitting attempt** stopped before adapter training
  because no candidate met its convergence rule. The preserved retry fitted
  heads in FP64; only L2 strength 0.01 converged and qualified.
- **Q12 and Q13 completed negative adapter experiments**; those results are
  distinct from execution failures.
- **Q15 remains unrun**: its contingency implementation passed CPU tests, but
  GPU training and replay equivalence are unverified. It has no performance result.

</details>

<details>
<summary>Sources of the comparison AUROCs around 0.88–0.97</summary>

These are **Q8's measurements against this project's ClinVar validation labels**,
using published predictor outputs. They are not performance figures copied from
the method papers. Exact **GRCh38 chromosome, position, REF and ALT** are matched,
with labels joined afterward. Q8 fits no predictor or threshold.

| Predictor | Published prediction source | Score transformation / aggregation |
| --- | --- | --- |
| SIFT4G | dbNSFP4.9a, `SIFT4G_score` | 1 minus the minimum raw score |
| PolyPhen-2 HumVar | dbNSFP4.9a, `Polyphen2_HVAR_score` | Maximum raw score |
| REVEL | [v1.3, 3 May 2021](https://zenodo.org/records/7072866) | Maximum matching transcript score, using GRCh38 positions |
| AlphaMissense | [2023 GRCh38 release](https://github.com/google-deepmind/alphamissense), pinned GCS generation `1691073413649109` | Maximum matching `am_pathogenicity` |
| EVE | dbNSFP4.9a, continuous `EVE_score` | Maximum raw score, without confidence-category filtering |

The main [comparison table](#method-comparison) reports their AUROC/AP and exact
coverage: SIFT4G **16,917**, PolyPhen-2 **16,430**, REVEL **17,720**,
AlphaMissense **16,884**, and EVE **9,284**, out of 17,927 validation variants.
Missing scores remain missing, so these per-method cohorts differ.
[Score sources and checksums](data/README.md) ·
[Executed Q8 measurements](notebooks/Q8-existing-tools.ipynb) ·
[Portable comparison evidence](notebooks/results/comparison/published/results.json).

Q8's original per-variant caches are currently absent; its preserved measurements
are rounded executed-notebook results with verified provenance. The preserved
**8,817-variant shared comparison excludes LoRA**. Training/calibration overlap
with ClinVar, homologous sequences and pretraining exposure remain unresolved.
These comparisons do not establish independent clinical validity or identify
the cause of the performance gap.

</details>

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

<!-- q16-question:start -->
### [Q16. Does longer fine-tuning improve on Q14?](notebooks/Q16-lora-continuation.ipynb)

Continue Q14’s successful adapters at learning rate **3e-5**, using all **46,888 training variants** for up to **three epochs** within the original four-hour deadline. Keep the classifier and scaler fixed. Select on the existing 2,048-variant sample, then score all 17,927 validation variants with the continuation, Q14 parent and frozen classifier; preserve negative outcomes and paired component intervals. See the [Q16 results](#q16-does-longer-fine-tuning-improve-on-q14).

Run `.venv/bin/python -m notebooks.src.refresh_q16` with the prepared **Lida** kernel. The runner preserves prior attempts and saves only a fully executed notebook.
<!-- q16-question:end -->

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

After an experiment publishes its results, refresh the main comparison with
`.venv/bin/python -m notebooks.src.readme_comparison`. This verifies the saved
evidence and keeps one performance table, including Q16.

## Machine configuration

Historical environment snapshot observed on **2026-09-07**. These were the
resources visible then, not minimum project requirements. **Q14 ran on a
separate H100 configuration**, as recorded in [Experiment details](#experiment-details).

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
