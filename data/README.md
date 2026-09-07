# Data

Run [Q0](../notebooks/Q0-clinvar-summary.ipynb) to download the **6 July 2026**
GRCh38 ClinVar snapshot automatically when its local input is missing:

- Archive: `data/clinvar_20260706.vcf.gz` (192,290,992 bytes).
- Decompressed input: `data/clinvar_20260706.vcf` (1,926,862,516 bytes).
- Source: [dated NCBI archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260706.vcf.gz).
- [Published archive MD5](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260706.vcf.gz.md5): `f78d25d49e17a070957a127e409f87b9`.
- Decompressed SHA-256: `95ef7cef2b32bc5ac2edae06b27ca24442bb0b50e7e5113026129abdddefe664`.

Q0 reuses an existing verified VCF. If only the archive exists, it verifies and
decompresses that archive. Otherwise it downloads the dated archive, verifies it,
and decompresses it. Temporary files become final files only after validation;
a checksum mismatch or download failure stops execution. The VCF header is checked
for `fileDate=2026-07-06` and `reference=GRCh38`. No moving latest-release URL is used.

The existing Q1/Q2 missense pilot is separately pinned to **5 September 2026**.
Its source remains `data/clinvar.vcf` with SHA-256
`0524586dcf9e8c8f1fe7742450b0555ac55d04a6e9a262f61db1d15f113e622a`.
If absent, Q1 downloads its own dated archive to `data/clinvar_20260905.vcf.gz`
and verifies it before decompression. Q0's July download preserves this input and
the current pilot partitions. All these data files are excluded from Git.

## Shared experiment partitions

The [Q1](../notebooks/Q1-clinvar-split.ipynb) workflow produces a **missense-only pilot**.
It applies Q0's quality and label filters, then requires the exact `SO:0001583`
identifier in ClinVar's `MC` field. Missing annotations and other consequences
are excluded; records with missense plus additional consequences remain eligible,
with all gene associations retained for grouping.

- `clinvar-train-pilot.vcf`: training variants, used to fit preprocessing and models.
- `clinvar-test-pilot.vcf`: validation variants, used to select settings and compare
  models. Despite its filename, this is **not a separate final test set**.

These files contain a 5,000-variant sample of the 65,270 eligible missense SNVs.
Sampling uses a fixed hash with seed 42. Whole related groups have a 70/30 target;
all earlier full-cohort split assignments are preserved and checked by a pinned
fingerprint. Grouping still includes nonmissense variants as relationship bridges.
Each export preserves the original VCF headers and complete records in source
order. Clinical annotations supply outcomes and audit information only; predictor
features use DNA. The source VCF remains unchanged.

[Q1's protocol](../notebooks/results/q1/protocol.json) freezes both file checksums;
its [manifest](../notebooks/results/q1/split_manifest.csv) records membership and
groups. Subsequent experiments must use these exact files without repartitioning.
Changed or incomplete exports fail verification. Q2 reads its labels directly from
the VCFs and aligns them to the manifest. The partitions are development data;
final performance claims require an untouched holdout.

The earlier broad-SNV VCFs, manifests, source code, notebooks and Q2 results are
preserved in `notebooks/results/archive/broad_snv_before_missense/`. Q2's old
features and model results are archived; missense model results require rerunning Q2.

These two generated VCFs are the exception to keeping only external inputs here.
All VCFs remain excluded from Git.

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
checks every sampled ClinVar REF against the genome, and excludes truncated or
non-ACGT windows without replenishing the sample. A REF mismatch stops execution.

The external Evo2 checkpoint uses the standard Hugging Face cache, outside
generated notebook results: `arcinstitute/evo2_1b_base`, revision
`2279e1df422c991037470302360edd40d0d2ea1e`. [Q2](../notebooks/Q2-evo2-classifier.ipynb) verifies its SHA-256
`8ffba7d0e6445a8f2c92d9ff1c4e772c7f73ca9179f5e0c699b8b0ca1b966f64`.

## Q8 AlphaMissense scores

[Q8](../notebooks/Q8-tools-survey.ipynb) downloads the published 2023 GRCh38 scores
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
The archive stays outside Git; matched scores and evaluation outputs live in
`notebooks/results/q8/`. Run Q1 before Q8 to reproduce the frozen pilot inputs.
