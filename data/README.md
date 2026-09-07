# Local data

`clinvar.vcf` is the uncompressed ClinVar VCF used for this project. The file is
approximately 1.94 GB and is excluded from Git.

In the current environment it was prepared from the existing local archive:

```bash
gzip -dc /root/data/clinvar.vcf.gz > data/clinvar.vcf
```

Run this command from the project root after making that archive available.
The original archive is preserved. Data acquisition provenance, the exact
ClinVar release, and label filtering still need to be documented before model
development.
