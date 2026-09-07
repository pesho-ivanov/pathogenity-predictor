# Data

`clinvar.vcf` is the uncompressed ClinVar VCF used for this project. The file is
approximately 1.94 GB and is excluded from Git.

In the current environment it was prepared from the existing local archive:

```bash
gzip -dc /root/data/clinvar.vcf.gz > data/clinvar.vcf
```

Run this command from the project root after making that archive available.
The original archive is preserved. The VCF header declares `source=ClinVar`,
`reference=GRCh38`, and `fileDate=2026-09-05`. Its SHA-256 is:

```text
0524586dcf9e8c8f1fe7742450b0555ac55d04a6e9a262f61db1d15f113e622a
```

This identifies the local input; the original download URL and chain of custody
have not been independently verified. The notebook checks the hash, assembly,
and date and fails if they differ. If only the local archive is available, it
decompresses it automatically. It never silently fetches a newer ClinVar release.
On another machine, supply the same VCF at `data/clinvar.vcf`, or set `ARCHIVE`
in [q0.py](../notebooks/src/q0.py) to a copy of the same compressed input.

## Shared experiment partitions

**Scope transition:** the existing files below contain a broader SNV pilot and
have not been rebuilt for the project's missense-only scope. They are historical
inputs, not ready-to-use missense-only datasets. Before new experiments, freeze
missense-only inputs and their checksums, preserving existing split assignments
and the earlier files and manifests.

The recorded [Q1](../notebooks/Q1-clinvar-split.ipynb) workflow produces:

- `clinvar-train-pilot.vcf`: training variants, used to fit preprocessing and models.
- `clinvar-test-pilot.vcf`: validation variants, used to select settings and compare
  models. Despite its filename, this is **not a separate final test set**.

These files contain Q1's eligible 5,000-variant pilot, not the entire ClinVar VCF.
Whole related groups are assigned by a fixed hash with seed 42 and a 70/30 target.
Each export preserves the original VCF headers and complete records in source
order. Clinical annotations supply outcomes and audit information only; predictor
features use DNA. The source VCF remains unchanged.

[Q1's protocol](../notebooks/results/q1/protocol.json) freezes both file checksums;
its [manifest](../notebooks/results/q1/split_manifest.csv) records membership and
groups. Reproducing the earlier experiment requires these exact files, without repartitioning.
Changed or incomplete exports fail verification. Q2 reads its labels directly from
the VCFs and aligns them to the manifest. The partitions are development data;
final performance claims require an untouched holdout.

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
