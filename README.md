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

### Data

Q0 and Q1 share **`data/clinvar_20260706.vcf` (6 July 2026)** and automatically
download the verified dated archive when needed. The comparison uses the full
July missense dataset.
See [data/README.md](data/README.md) for filenames and checksum checks. Q1 also writes the two shared VCF
partitions to `data/`; other generated files live under `notebooks/results/`.
Generated artifacts are described in [results documentation](notebooks/results/README.md).

## Q16 and published predictors

| Aspect | Ours: [Q16 Evo2 LoRA](notebooks/Q16-lora-continuation.ipynb) | AlphaMissense | PrimateAI-3D | EVE | REVEL | SIFT4G | PolyPhen-2 HumVar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper year | — (project notebook) | [2023](https://doi.org/10.1126/science.adg7492) | [2023](https://doi.org/10.1126/science.abn8197) | [2021](https://doi.org/10.1038/s41586-021-04043-8) | [2016](https://doi.org/10.1016/j.ajhg.2016.08.016) | [2016 (online 2015)](https://doi.org/10.1038/nprot.2015.123) | [2010](https://doi.org/10.1038/nmeth0410-248) |
| Core approach | DNA foundation model with two adapted layers and a fixed linear classifier | AlphaFold-derived neural network | 3D convolutional neural network | Protein-family variational autoencoder, then Gaussian mixture scoring | Random forest ensemble | Conservation-based substitution scoring | Naïve Bayes classifier |
| Number of trained parameters | **393,216** LoRA parameters updated in Q16; pretrained backbone and 8,195-parameter classifier frozen | Not verified | Not verified | [Varies by protein/alignment length](https://github.com/OATML-Markslab/EVE/blob/master/EVE/VAE_encoder.py) | Not quantified (random forest) | No trained neural weights; conservation scoring | Not quantified (Naïve Bayes) |
| Main information used | Reference and alternate DNA sequences around the variant (512 bases each) | Protein sequence, evolutionary conservation and learned structural context | 3D protein structure and evolutionary variation across primates | Evolutionary variation in aligned sequences of related proteins | Scores from 13 existing functional-effect and conservation tools | Amino-acid conservation across related proteins | Protein sequence conservation and structural features |
| Biological information | Reference/alternate 512-base DNA windows; evolutionary information learned during pretraining | Protein sequence alignments, evolutionary conservation and learned structural representations | 3D protein structure and multiple-sequence alignments | Aligned sequences of related proteins; dependencies between amino-acid positions | Functional-effect and conservation scores from 13 tools | Protein multiple-sequence alignments; amino-acid conservation | Protein sequence conservation and structural features |
| Task-specific training evidence | 46,888 ClinVar variants, after Evo2 pretraining | Human/primate population variation, building on AlphaFold pretraining | Millions of tolerated human/primate variants; auxiliary sequence/structure learning tasks | Unsupervised learning of natural protein sequences | HGMD disease variants versus rare presumed-neutral variants | Amino-acid frequencies in related proteins | [HumVar](https://genetics.bwh.harvard.edu/downloads/pph2/training/): disease/function variants versus common neutral human variants |
| Clinical-label use | ClinVar labels for classifier/adapter fitting, development selection and evaluation; no untouched final test | Core model avoids clinical labels; [released scores and thresholds use ClinVar calibration](https://www.ebi.ac.uk/training/online/courses/alphafold/classifying-the-effects-of-missense-variants-using-alphamissense/understanding-pathogenicity-scores-from-alphamissense/) | No clinical annotations for core training; ClinVar used for benchmarking | No clinical labels required for fitting/scoring; ClinVar used for evaluation | Disease labels in training; constituent tools bring their own training histories; ClinVar overlap unresolved | No clinical-label fitting in core scoring | Disease/function labels in training; ClinVar overlap unresolved |
| AUROC [95% CI] | 0.872 [0.859–0.884] | 0.962 [0.956–0.967] | Not evaluated locally | 0.909 [0.889–0.925] | 0.974 [0.970–0.978] | 0.878 [0.865–0.890] | 0.894 [0.883–0.905] |
| Average precision [95% CI] | 0.801 [0.769–0.828] | 0.940 [0.923–0.951] | Not evaluated locally | 0.916 [0.889–0.935] | 0.958 [0.945–0.968] | 0.772 [0.734–0.804] | 0.822 [0.781–0.852] |
| Coverage of our 17,927 variants | 17,927 — 100% | 16,884 — 94.2% | Unavailable; licensed data needed | 9,284 — 51.8% | 17,720 — 98.8% | 16,917 — 94.4% | 16,430 — 91.6% |
| Scored / validation | 17,927 / 17,927 | 16,884 / 17,927 | — | 9,284 / 17,927 | 17,720 / 17,927 | 16,917 / 17,927 | 16,430 / 17,927 |
| Validation evidence | Repeatedly used development partition; no untouched final test | Published genetic and functional benchmarks | Published population, patient-cohort and functional benchmarks | Published clinical-label and functional-assay comparisons | Published independent test sets | — | — |
| Practical use | GPU model execution; verified batch-32 workflow | Precomputed score lookup | Licensed score access | Precomputed lookup for covered proteins | Precomputed score lookup | — | — |
| Compute evidence in our project | 3.67 h total experiment including training, validation and reporting; 27.27 GiB peak GPU allocation; verified batch-32 inference; separate training/inference timings unavailable | 1.2 min lookup/evaluation; not trained locally | No measurement; not trained locally | 5.5 s aggregation/evaluation; not trained locally | 1.8 min lookup/evaluation; not trained locally | 13.0 s aggregation/evaluation; not trained locally | 12.4 s aggregation/evaluation; not trained locally |
| Main strength | Complete local coverage, inspectable adaptation and explicit leakage checks | Strong performance with protein-specific context | Broad tolerated-variation training and external evaluation | Clinical-label-independent evolutionary modeling | Strongest reported discrimination in our comparison | — | — |
| Main limitation here | Lower discrimination; no calibrated clinical probabilities; batch sensitivity | Missing scores and unresolved ClinVar calibration overlap | Cannot assess performance on our cohort | Substantial missing coverage | Possible disease-training and constituent-model overlap | — | — |
| Evaluated method and sources | [Evo2 7B Q16 continued LoRA (blocks 29 and 30, rank 8, 512 bp)](notebooks/Q16-lora-continuation.ipynb) | [AlphaMissense](notebooks/Q8-existing-tools.ipynb) ([paper](https://doi.org/10.1126/science.adg7492)) | [PrimateAI-3D (licensed)](notebooks/Q8-existing-tools.ipynb) ([paper](https://doi.org/10.1126/science.abn8197)) | [EVE](notebooks/Q8-existing-tools.ipynb) ([paper](https://doi.org/10.1038/s41586-021-04043-8)) | [REVEL](notebooks/Q8-existing-tools.ipynb) ([paper](https://doi.org/10.1016/j.ajhg.2016.08.016)) | [SIFT4G](notebooks/Q8-existing-tools.ipynb) ([paper](https://doi.org/10.1038/nprot.2015.123)) | [PolyPhen-2](notebooks/Q8-existing-tools.ipynb) ([paper](https://doi.org/10.1038/nmeth0410-248)) |
| Score source in this project | [Q16 continued LoRA checkpoint](notebooks/Q16-lora-continuation.ipynb); local GPU inference | [2023 GRCh38 release](data/README.md#q8-alphamissense-scores); `am_pathogenicity` | [Licensed scores](data/README.md#q8-primateai-3d-access); unavailable to this project | [dbNSFP4.9a](data/README.md#q8-sift4g-polyphen-2-and-eve-scores); continuous `EVE_score`, without confidence-category filtering | [REVEL v1.3](data/README.md#q8-revel-scores); continuous REVEL score | [dbNSFP4.9a](data/README.md#q8-sift4g-polyphen-2-and-eve-scores); `1 - SIFT4G_score` | [dbNSFP4.9a](data/README.md#q8-sift4g-polyphen-2-and-eve-scores); `Polyphen2_HVAR_score` |

## Experiment details

The details below describe the completed **Q16 Evo2 7B experiment**. **All reported performance is development validation;
there is no untouched test set.** AP means **average precision across
thresholds**, not precision at a selected cutoff.

<details>
<summary>Variant counts, ClinVar release, labels and split construction</summary>

These are the counts after eligibility filtering. The selection sample is a
subset of development validation.

| Cohort | Total | Pathogenic / likely pathogenic | Benign / likely benign |
| --- | ---: | ---: | ---: |
| Full eligible modeling cohort | **64,815** | **24,696** | **40,119** |
| Full training partition | 46,888 | 18,173 | 28,715 |
| Full validation partition | 17,927 | 6,523 | 11,404 |
| Validation sample used for model selection | 2,048 | 688 | 1,360 |

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

Q16 trains on all **46,888 training variants** and uses a fixed **2,048-variant
validation sample** for checkpoint selection. `clinvar-train.vcf` is training;
**`clinvar-test.vcf` is development validation despite its filename**.
[Grouping and original assignment](notebooks/src/q1.py) ·
[Frozen full-cohort protocol](notebooks/src/q1_full.py) ·
[Q16 membership checks](notebooks/src/q16.py).

</details>

<details>
<summary>Evo2 checkpoint, sequence window and exact score formula</summary>

The checkpoint is **`arcinstitute/savanna_evo2_7b_base`**, revision
`eb0a7478e5f3c291f31e2b3d9ec14fc067f9982a`, file
`savanna_evo2_7b_base.pt`. Its SHA-256 is
`ed8d264c14fea3c6305b475122e068e09009cbdabc05e1653764147c1b294cd4`.
The measured loaded backbone contains **6,481,649,408 parameters**, with
**32 layers** and hidden width **4,096**. Q16 uses the BioNeMo/NeMo conversion
in **BF16**, with FP8 disabled. All 325 audited converted tensors matched the
source; native Savanna/Vortex forward-output parity remains unresolved.
[Q16 model loading and checks](notebooks/src/q16_backend.py).

Each allele uses **512 nucleotides**: 256 before the mutation, the mutated
position and 255 afterward. The reference/alternate pair is canonicalized
against its reverse complement **before cropping** from the frozen 1,024-base
contexts. Q16 uses one canonical orientation, producing two sequences per
variant. [Q16 input preparation](notebooks/src/q16.py).

For the Q16 classifier, let $h_r$ and $h_a$ be the 4,096-dimensional means of
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
The classifier and scaler were fitted on a 24,576-variant training subset
and remain fixed during Q16.
**Higher $z$ means more pathogenic.** AUROC and AP use $z$; a sigmoid would
preserve ranking, but its output is not established as a calibrated clinical
probability. [Q16 scoring workflow](notebooks/src/q16_backend.py).

</details>

<details>
<summary>Q16 adaptation, training and checkpoint selection</summary>

Q16 continues trained **rank-8 LoRA adapters in blocks 29 and 30**, with
**alpha 16** and **dropout 0**. Only **393,216 adapter parameters** update;
the backbone, **8,195-parameter classifier** and training-fitted scaler stay fixed.

Training uses **AdamW**, peak learning rate **3e-5**, a **64-step warmup** followed
by cosine decay to 10% of peak, weight decay **0.01**, gradient norm limit **1**,
seed **42**, and batches of **32 variants**. Class-balanced binary cross entropy
uses weights computed from all training labels. The backbone and adapters use
BF16, with FP32 pooling, classifier and optimizer masters.

The completed run processed **132,592 examples** in **4,145 optimizer updates**,
covering all **46,888 training variants**. The retained checkpoint is at update
**3,584**, chosen using the fixed **2,048-variant development sample**; its
reported metrics use all **17,927 validation variants**. Backbone, classifier
and scaler preservation and checkpoint reload predictions were verified.

The run used an **NVIDIA H100 with 80 GB memory**. The comparison reports
**3.67 hours** through saved-notebook completion and **27.27 GiB** peak GPU
allocation. [Executed Q16 notebook](notebooks/Q16-lora-continuation.ipynb) ·
[Frozen configuration](notebooks/src/q16.py) ·
[GPU workflow](notebooks/src/q16_backend.py) ·
[Completed Q16 record](notebooks/results/comparison/published/q16-results.json).

</details>

<details>
<summary>Sources of the comparison AUROCs around 0.88–0.97</summary>

These are **Q8's measurements against this project's ClinVar validation labels**,
using published predictor outputs. They are not performance figures copied from
the method papers. Exact **GRCh38 chromosome, position, REF and ALT** are matched,
with labels joined afterward. Q8 fits no predictor or threshold.

| Predictor | Paper year | Published prediction source | Score transformation / aggregation |
| --- | --- | --- | --- |
| SIFT4G | [2016 (online 2015)](https://doi.org/10.1038/nprot.2015.123) | dbNSFP4.9a, `SIFT4G_score` | 1 minus the minimum raw score |
| PolyPhen-2 HumVar | [2010](https://doi.org/10.1038/nmeth0410-248) | dbNSFP4.9a, `Polyphen2_HVAR_score` | Maximum raw score |
| REVEL | [2016](https://doi.org/10.1016/j.ajhg.2016.08.016) | [v1.3, 3 May 2021](https://zenodo.org/records/7072866) | Maximum matching transcript score, using GRCh38 positions |
| AlphaMissense | [2023](https://doi.org/10.1126/science.adg7492) | [2023 GRCh38 release](https://github.com/google-deepmind/alphamissense), pinned GCS generation `1691073413649109` | Maximum matching `am_pathogenicity` |
| EVE | [2021](https://doi.org/10.1038/s41586-021-04043-8) | dbNSFP4.9a, continuous `EVE_score` | Maximum raw score, without confidence-category filtering |

The main [comparison table](#q16-and-published-predictors) reports their AUROC/AP and exact
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
