"""Build Q8, execute every cell from a fresh kernel, and save only on success.

Run from the repository root: .venv/bin/python -m notebooks.src.refresh_q8
"""

from pathlib import Path
import json
import shutil

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


class LoggedClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        value = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return value


def refresh():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(cells=[
        md("""# Q8. Which existing missense predictors are practical baselines, and how do they perform on the full validation dataset?

Evaluate **AlphaMissense, REVEL, SIFT4G, PolyPhen-2 HumVar and EVE** on every variant in Q1's frozen full `clinvar-test.vcf` from **6 July 2026**. This is development **validation**, with no separate untouched test set. PrimateAI-3D remains blocked on licensed data access.

[Implementation](src/q8_full.py) · [Method survey and sources](src/q8_catalog.json) · [Dependencies](../requirements.txt) · [README comparison](../README.md#method-comparison).

Run [Q1](Q1-clinvar-split.ipynb) first, then execute this notebook in cell order with the **Gamow** kernel. CPU only; no local model fitting, recalibration or variant uploads. Published score archives and indexed dbNSFP blocks are downloaded automatically when missing, verified and cached in `data/`. Acquisition can take tens of minutes. Historical September pilot notebooks, inputs and results remain in `results/archive/before_shared_july_snapshot/`."""),
        code('from src import q8_full'),
        md("""## Frozen full inputs and leakage checks

Verify both full VCF checksums and rerun Q1's missense eligibility, gene, locus, group and sequence-context separation checks. Preserve every assigned split. Only validation chromosome/position/REF/ALT keys enter the score lookups; ClinVar outcomes are loaded afterwards from the full validation VCF.

The protocol fixes all five sources and score definitions before lookup. No sampling cap, balancing, threshold tuning or benign imputation is applied. Local leakage checks do not establish independence from external clinical calibration, model training or evolutionary sequences."""),
        code('cohort = q8_full.prepare()'),
        md("""## AlphaMissense

Use the pinned **2023 GRCh38** archive. Match chromosome, position, REF and ALT exactly, removing only the `chr` prefix; take the maximum continuous `am_pathogenicity` across matching transcript annotations. Keep unmatched alleles unscored.

The core model uses population variation; released scores and thresholds use [ClinVar calibration](https://www.ebi.ac.uk/training/online/courses/alphafold/classifying-the-effects-of-missense-variants-using-alphamissense/understanding-pathogenicity-scores-from-alphamissense/). Calibration overlap is unresolved. [Published source and checksums](../data/README.md#q8-alphamissense-scores)."""),
        code("reports = {'AlphaMissense': q8_full.run_method('AlphaMissense', cohort)}"),
        md("""## REVEL

Use **v1.3**, exact GRCh38 `grch38_pos`/REF/ALT matches, and the maximum continuous score across matching annotations. Missing GRCh38 positions have no GRCh37 fallback. REVEL's HGMD training variants and constituent tools may overlap ClinVar; no independence claim is made. [Published source and checksums](../data/README.md#q8-revel-scores)."""),
        code("reports['REVEL'] = q8_full.run_method('REVEL', cohort)"),
        md("""## SIFT4G, PolyPhen-2 HumVar and EVE

Retrieve exact validation alleles from **dbNSFP4.9a** using its Tabix index and verified HTTP byte ranges. Full-validation annotations have their own manifest; raw block caches are shared with earlier lookups. Metadata MD5s, data ETag/generation, byte ranges, block SHA-256 checksums and BGZF CRCs are checked. The full 39.5 GB database MD5 is not recomputed.

- **SIFT4G:** `1 - SIFT4G_score`.
- **PolyPhen-2:** `Polyphen2_HVAR_score` (HumVar).
- **EVE:** continuous `EVE_score`, without confidence-category filtering.

Take the maximum damaging-direction score across matching annotations; preserve every missing score and exact-allele mismatch. SIFT4G/EVE evolutionary-data overlap and PolyPhen-2 clinical-training overlap are unaudited. [Acquisition and sources](../data/README.md#q8-sift4g-polyphen-2-and-eve-scores)."""),
        code('reports.update(q8_full.run_remaining(cohort))'),
        md("""## PrimateAI-3D access

The [official release](https://github.com/Illumina/PrimateAI-3D) requires a signed license agreement and an emailed download link. No licensed score file or approved link is available. Record this blocker for the current full cohort; dbNSFP's original PrimateAI scores cannot substitute for PrimateAI-3D."""),
        code('primate_access = q8_full.report_primate_access(cohort)'),
        md("""## Validation performance and coverage

Each method's AUROC and average precision use its scored validation subset. Percentile **95% intervals** resample whole Q1 components **1,000 times**, with **seed 42**; members stay together before restricting to scored variants. Missing predictions are retained in coverage denominators. Intervals do not correct external-data overlap or development-selection bias.

Each method writes raw scores, matched annotations, labeled validation predictions, coverage, metrics and completion provenance under [results/q8/full/](results/q8/full/). [Artifact descriptions](results/README.md#q8-artifacts-missense-survey-and-external-predictors)."""),
        code('q8_full.show_results(reports)'),
        md("""## Conclusion

Refresh the README from verified exports, then compare methods on the same variants scored by every currently available method. Full-cohort results remain separate from archived pilot evaluations."""),
        code('comparison_result = q8_full.conclude(reports, primate_access)'),
    ], metadata={'kernelspec': {'display_name': 'Gamow', 'language': 'python', 'name': 'gamow'}})
    path = ROOT / 'notebooks/Q8-existing-tools.ipynb'
    archive = ROOT / 'notebooks/results/archive/q8_before_full_validation'
    archive.mkdir(parents=True, exist_ok=True)
    if path.exists() and not (archive / path.name).exists():
        shutil.copy2(path, archive / path.name)
    client = LoggedClient(notebook, timeout=None, kernel_name='gamow', allow_errors=False,
                          resources={'metadata': {'path': str(ROOT / 'notebooks')}})
    client.execute()
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if ([cell.execution_count for cell in cells] != list(range(1, len(cells) + 1))
            or any(not cell.source.strip() for cell in cells)
            or any(output.output_type == 'error' for cell in cells for output in cell.outputs)):
        raise RuntimeError('Q8 did not complete every code cell in order')
    summary = json.loads((ROOT / 'notebooks/results/comparison/summary.json').read_text())
    expected = {'q8:' + name for name in ['AlphaMissense', 'REVEL', 'SIFT4G', 'PolyPhen-2', 'EVE']}
    completed = {row['id'] for row in summary['methods'] if row['id'] in expected
                 and row['status'] == 'Available' and row.get('total') == summary['cohort']['validation_variants']}
    if completed != expected:
        raise RuntimeError('The README did not accept all five full-validation result exports')
    temporary = path.with_suffix('.ipynb.partial')
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print(f'Saved fully executed notebook: {path}', flush=True)


if __name__ == '__main__':
    refresh()
