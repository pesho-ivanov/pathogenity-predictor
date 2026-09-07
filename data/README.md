# Data

Run [Q0](../notebooks/Q0-clinvar-summary.ipynb) or
[Q1](../notebooks/Q1-clinvar-split.ipynb) to download the **6 July 2026**
GRCh38 ClinVar snapshot automatically when its local input is missing:

- Archive: `data/clinvar_20260706.vcf.gz` (192,290,992 bytes).
- Decompressed input: `data/clinvar_20260706.vcf` (1,926,862,516 bytes).
- Source: [dated NCBI archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260706.vcf.gz).
- [Published archive MD5](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260706.vcf.gz.md5): `f78d25d49e17a070957a127e409f87b9`.
- Decompressed SHA-256: `95ef7cef2b32bc5ac2edae06b27ca24442bb0b50e7e5113026129abdddefe664`.

Both notebooks reuse the same existing verified VCF. If only the archive exists,
they verify and decompress it. Otherwise they download the dated archive, verify
it, and decompress it. Temporary files become final files only after validation;
a checksum mismatch or download failure stops execution. The VCF header is checked
for `fileDate=2026-07-06` and `reference=GRCh38`. No moving latest-release URL is used.

The obsolete `data/clinvar.vcf` September input has been removed. Both notebooks
use only the dated July path; all downloaded data files remain excluded from Git.

## Shared experiment partitions

[Q1](../notebooks/Q1-clinvar-split.ipynb) prepares the **full missense cohort** from
the same **6 July 2026** source as Q0. It applies Q0's quality and label filters
(at least two review stars, unambiguous B/P labels, usable genes, nonconflicting
single A/C/G/T substitutions on chromosomes 1–22, X and Y), then requires exact
`SO:0001583` in ClinVar's `MC` field. All gene associations remain available for
grouping, including records with multiple consequences.

- `clinvar-train.vcf`: **46,888 training variants** (72.34%).
- `clinvar-test.vcf`: **17,927 validation variants** (27.66%).

All **64,815 eligible missense variants** pass the sequence checks; there are no sequence exclusions.

The full dataset has no sampling cap or class balancing. Variants shared with the
September eligible cohort retain their splits; 14 changed component anchors have
fixed, label-independent overrides. New groups use the original seed-42, 70/30
assignment. Unequal group sizes make the actual proportions approximate. July
eligibility and labels are taken from July records; absent variants are not imported
from September. The complete assignment is checksum-pinned before modeling.

Grouping includes the broader Q0-eligible SNV cohort as relationship bridges.
Every eligible 1,024-base reference and alternate context is checked, including
identical sequences and reverse complements. A cross-split relationship or REF
mismatch stops preparation. Truncated or non-ACGT contexts are explicitly excluded.

The [full protocol](../notebooks/results/q1/full/protocol.json) freezes both file
checksums; the [full manifest](../notebooks/results/q1/full/split_manifest.csv)
records every variant and group. The [full-data module](../notebooks/src/q1_full.py)
provides `prepare()`, `verify_protocol()` and `load_partition_labels()` for new
experiments. Predictor inputs use only the DNA allowlist; clinical annotations
supply eligibility, outcomes and audit metadata. VCF headers and complete records
are preserved in source order, and the July ClinVar input remains unchanged.

The regenerated July pilot contains **3,654 training** variants in
`clinvar-train-pilot.vcf` and **1,346 validation** variants in
`clinvar-test-pilot.vcf`. Its protocol and grouping cohort live under
`notebooks/results/q1/`. These are July inputs, distinct from the archived pilot.

The original September pilot (**3,658 training / 1,342 validation**) and full
partitions, protocols, implementations and completed Q2/Q8/Q9/Q10 results are
preserved under `notebooks/results/archive/before_shared_july_snapshot/`, with the
original directory structure. `split_inheritance.json` records the migration's
assignment rule and source checksums. The raw September VCF is not retained there;
its dated download URL and checksums remain in the archived implementation.

September scores do not evaluate the July datasets. New experiments must freeze
against `results/q1/full/protocol.json`, refit training-dependent preprocessing and
models, and evaluate the full July validation cohort. The README rejects predictions
bound to another snapshot or cohort. Both datasets are for development; the
`test` filename denotes validation, and final performance claims require an
untouched holdout.

These generated partition VCFs are the exception to keeping only external inputs
in `data/`. Other generated artifacts belong under `notebooks/results/`.

## GRCh38 reference for Q1

[Q1](../notebooks/Q1-clinvar-split.ipynb) automatically downloads the initial UCSC
[hg38.fa.gz](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz)
to `data/hg38.fa.gz` (983,659,424 bytes). It checks the
[published MD5](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/md5sum.txt)
`1c9dcaddfa41027f17cd8f7a82c7293b` and the pinned SHA-256:

```text
c1dd87068c254eb53d944f71e51d1311964fce8de24d6fc0effc9c61c01527d4
```

The notebook uppercases softmasked bases, uses chromosomes 1–22, X and Y,
checks every eligible ClinVar REF against the genome, and excludes truncated or
non-ACGT windows without replenishing the sample. A REF mismatch stops execution.

The external Evo2 checkpoint uses the standard Hugging Face cache, outside
generated notebook results: `arcinstitute/evo2_1b_base`, revision
`2279e1df422c991037470302360edd40d0d2ea1e`. [Q2](../notebooks/Q2-evo2-classifier.ipynb) verifies its SHA-256
`8ffba7d0e6445a8f2c92d9ff1c4e772c7f73ca9179f5e0c699b8b0ca1b966f64`.

Q2's zero-shot 7B extension uses the external Hugging Face cache in the same way:
[`arcinstitute/evo2_7b_base`](https://huggingface.co/arcinstitute/evo2_7b_base/tree/074097e9dc788e8bfe045d6495b9f6153a7c6bfc),
revision `074097e9dc788e8bfe045d6495b9f6153a7c6bfc`, file `evo2_7b_base.pt`
(13,006,429,947 bytes), SHA-256
`d8a0e775a5d849921b8725837c6a3cbc71fa15e712f4189a2ed52ef955aad29b`.
It uses Vortex/FP8 and the full July `clinvar-test.vcf` validation partition.
The full training partition supplies only inference-check examples; no model or
threshold is fitted. Its README row uses all 17,927 current validation variants.
This is a separate checkpoint format and inference implementation from Q10's
archived BioNeMo frozen classifier.

## Q8 AlphaMissense scores

Q8's current [full-validation workflow](../notebooks/src/q8_full.py) verifies both
full Q1 VCFs and evaluates every one of the **17,927** July validation variants.
Only genomic keys enter lookup; ClinVar labels are joined afterwards. It reuses
the pinned sources and aggregation rules below, with new outputs under
`notebooks/results/q8/full/`. Historical pilot results remain in the dated archive.

[Q8](../notebooks/Q8-existing-tools.ipynb) downloads the published 2023 GRCh38 scores
to `data/alphamissense/AlphaMissense_hg38.tsv.gz` when absent. It pins GCS generation
`1691073413649109`, checks the archive size (642,961,469 bytes) and published MD5
`9fd167735f16a1b87da6eb3e4c25fcb5`, and atomically promotes verified downloads.
Existing corrupt archives fail verification; valid files are reused offline.
The [source metadata](https://storage.googleapis.com/storage/v1/b/dm_alphamissense/o/AlphaMissense_hg38.tsv.gz?generation=1691073413649109)
and [download](https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz?generation=1691073413649109)
are public; no model weights, GPU, credentials or variant uploads are required.

The original archive header says CC BY-NC-SA 4.0; the later
[official README](https://github.com/google-deepmind/alphamissense/blob/fe2dc845f93310abd6c1b0e8955d7a96c2144d66/README.md)
says predictions are CC BY 4.0. Q8 retains both statements in provenance and leaves
the downloaded archive unchanged. Cite Cheng et al. (2023), Science,
[doi:10.1126/science.adg7492](https://doi.org/10.1126/science.adg7492).
The archive stays outside Git; current matched scores and evaluation outputs live
in `notebooks/results/q8/full/`. Run Q1 before Q8 to reproduce the full frozen inputs.

## Q8 REVEL scores

[Q8](../notebooks/Q8-existing-tools.ipynb) also downloads REVEL **v1.3, 3 May 2021**
from [Zenodo record 7072866](https://zenodo.org/records/7072866) to
`data/revel/revel-v1.3_all_chromosomes.zip` when absent. It checks the published
MD5 `3ea2bc33e6b5455fc7e9899da863b5fe` and size **667,102,707 bytes** before use.
The compressed CSV is streamed without extracting its approximately 6.5 GB contents.
Existing valid archives are reused offline; corrupt or partial downloads fail.

Use `grch38_pos` for exact REF/ALT matching; missing GRCh38 positions remain
unmapped, with no GRCh37 fallback or additional liftover. Scores are aggregated by
maximum across matching transcript annotations, independently of ClinVar labels.
Every match and missing score is retained under `notebooks/results/q8/full/revel/`.

The [official download page](https://sites.google.com/site/revelgenomics/downloads)
specifies non-commercial use; Zenodo metadata separately lists ODC-ODbL. Both
statements are recorded in provenance. Cite Ioannidis et al. (2016), AJHG,
[doi:10.1016/j.ajhg.2016.08.016](https://doi.org/10.1016/j.ajhg.2016.08.016).
REVEL's HGMD and constituent-model training overlap with this ClinVar cohort remains
unresolved. The archive is excluded from Git; there are no variant uploads.

## Q8 SIFT4G, PolyPhen-2 and EVE scores

[Q8](../notebooks/Q8-existing-tools.ipynb) downloads indexed regions from the public
[GRCh38 dbNSFP4.9a release](https://grr.iossifovlab.com/hg38/scores/dbNSFP4.9a/index.html)
into `data/dbnsfp4.9a/`. This is the GRR distribution of the 2024 academic release;
the benchmark uses `SIFT4G_score`, `Polyphen2_HVAR_score` and `EVE_score`.
The [acquisition module](../notebooks/src/q8_dbnsfp.py) downloads the Tabix index,
release README and resource metadata automatically, verifying published MD5s.
It reads only indexed byte ranges from the 39,507,803,250-byte BGZF table and caches
the downloaded blocks. The first pilot extraction needs several GB and may take
tens of minutes; no full database download, model weights or GPU are needed.

Every range must match the pinned ETag `1f3e8e14b401b05cf61b25decdaa8692`, GCS
generation `1781155602928795`, total length and exact byte interval. SHA-256 checks
protect cached blocks; Biopython verifies BGZF CRCs. The full database MD5 is
**not** recomputed. `pilot_annotations.tsv.gz` retains exact pilot alleles,
protein/transcript mapping fields and raw score lists, excluding clinical fields.
`acquisition.json`, written last, binds this extraction to the ordered pilot keys,
source version, producing code and table checksum. Later runs reuse it offline;
an identity or checksum mismatch stops execution. Interrupted range downloads can
resume from the block cache. All files in this directory are excluded from Git.

The full-validation extraction uses `full-validation/validation_annotations.tsv.gz`
and its own `full-validation/acquisition.json`, bound to every ordered July
validation key. It shares the verified `blocks/` cache with earlier extractions;
the pilot table and manifest are preserved. The three current result folders are
`notebooks/results/q8/full/sift4g/`, `polyphen2/` and `eve/` under the same parent.

SIFT4G is oriented as `1 - score`; PolyPhen-2 uses **HumVar**, and EVE uses its
continuous score without confidence-category filtering. Take the maximum oriented
score across exact-allele annotations; leave missing values unscored. The academic
dbNSFP branch and upstream tool restrictions apply. Source documentation and
tool-specific training/exposure limitations are retained with each result.

## Q8 PrimateAI-3D access

The [official repository](https://github.com/Illumina/PrimateAI-3D) requires a signed
license agreement and supplies the score/model download link by email. No approved
link or licensed score file was available for this run. Q8 records this blocker in
`notebooks/results/q8/full/primateai3d/access_status.json`, with no predictions or metrics.
Provide an approved download URL or a local path to licensed data to complete this
benchmark. dbNSFP's original **PrimateAI** scores are a different model and are not
used as a substitute for PrimateAI-3D.

## Q9 Evo2 1B BioNeMo inputs

[Q9](../notebooks/Q9-evo2-1b.ipynb) automatically fetches pinned external
BioNeMo, NeMo and Megatron sources into `data/bionemo-recipes/`. Its
`data/evo2-savanna-1b/savanna_evo2_1b_base.pt` checkpoint comes from
`arcinstitute/savanna_evo2_1b_base`, revision
`7217626d9f843e1830a5de1f5209c046570b6856`, SHA-256
`7bb731473c99db72aba34e7b0df443f2f422e12b0669bea6d78415dc87057a08`.
Converted weights, logs, caches and experiment outputs belong under
`notebooks/results/q9/`. Q9 consumes the same fixed missense VCFs as Q2.

The gradient investigation also downloads NVIDIA's public
`evo2-1b-8k-bf16-nemo2`, version `1.0`, into `data/evo2-bf16-1b/`.
Its `nemo2_evo2_1b_8k_bf16.tar.gz` archive has SHA-256
`ea4a3f5c9c26d5edc10bdc85165c090ad0ff23ac2670d4f61244f5f0d9d5e817`.
The archive and extracted external weights are retained; diagnostic outputs and
download provenance are under `notebooks/results/q9/investigation/`.

## Q10 Evo2 7B checkpoint

[Q10](../notebooks/Q10-evo2-7b.ipynb) downloads
`evo2-savanna-7b/savanna_evo2_7b_base.pt` from
[`arcinstitute/savanna_evo2_7b_base`](https://huggingface.co/arcinstitute/savanna_evo2_7b_base/tree/eb0a7478e5f3c291f31e2b3d9ec14fc067f9982a),
revision `eb0a7478e5f3c291f31e2b3d9ec14fc067f9982a`.
The 15,759,263,071-byte file is verified against SHA-256
`ed8d264c14fea3c6305b475122e068e09009cbdabc05e1653764147c1b294cd4`.
It reuses the pinned BioNeMo sources above; converted weights and all generated
features, classifiers and evaluations live under `notebooks/results/q10/`.
