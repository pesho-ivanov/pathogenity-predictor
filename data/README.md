# External data

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
On another machine, supply the same VCF at `data/clinvar.vcf`, or set the notebook's
archive path to a copy of the same compressed input.
