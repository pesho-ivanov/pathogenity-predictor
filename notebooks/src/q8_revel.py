"""REVEL v1.3 lookup and evaluation on the unchanged Q1 missense pilot.

The lookup receives genomic keys only. Outcomes are joined afterwards by the
shared Q8 evaluator; no classifier, aggregation rule or threshold is fitted.
"""

import csv
import importlib.metadata
import io
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import pandas as pd

from . import q1, q8_baseline as baseline

OUTPUT = q1.ROOT / 'notebooks/results/q8/revel'
ARCHIVE = q1.ROOT / 'data/revel/revel-v1.3_all_chromosomes.zip'
SOURCE = {
    'tool': 'REVEL', 'release': '1.3', 'release_date': '2021-05-03', 'assembly': 'GRCh38',
    'url': 'https://zenodo.org/api/records/7072866/files/revel-v1.3_all_chromosomes.zip/content',
    'record': '7072866', 'doi': '10.5281/zenodo.7072866',
    'bytes': 667102707, 'md5': '3ea2bc33e6b5455fc7e9899da863b5fe',
    'metadata_url': 'https://zenodo.org/api/records/7072866',
    'documentation': 'https://sites.google.com/site/revelgenomics/downloads',
    'citation': 'Ioannidis et al. (2016), AJHG, doi:10.1016/j.ajhg.2016.08.016',
    'license_note': 'Official download page: non-commercial use; contact authors for other uses. '
                    'Zenodo metadata separately lists ODC-ODbL. Preserve both source statements.',
}
POLICY = {
    **baseline.POLICY,
    'matching': 'Exact GRCh38 chromosome, 1-based grch38_pos, REF and ALT; remove chr prefix only. '
                'Skip missing GRCh38 positions; never substitute hg19_pos, lift over, or complement alleles.',
    'aggregation': 'Maximum continuous REVEL score over all matching allele/amino-acid/transcript rows; '
                   'keep all matching rows, transcript IDs, minimum score and score range.',
    'exposure': 'REVEL learned from HGMD disease variants and presumed neutral population variants. '
                'Its 13 constituent tools have their own training histories. Exact overlap with this '
                'ClinVar pilot is unresolved; no claim of independent clinical validation.',
}
COLUMNS = ['chr', 'hg19_pos', 'grch38_pos', 'ref', 'alt', 'aaref', 'aaalt', 'REVEL', 'Ensembl_transcriptid']
MEMBER = 'revel_with_transcript_ids'


def lookup_scores(path, variant_keys):
    """Stream the zipped CSV without extracting it or using ClinVar annotations."""
    keys = list(variant_keys)
    wanted = set(keys)
    if len(keys) != len(wanted):
        raise ValueError('Lookup keys must be unique')
    for key in keys:
        parts = key.split(':')
        if (len(parts) != 5 or parts[0] != 'GRCh38' or not parts[2].isdigit()
                or int(parts[2]) < 1 or parts[3] not in list('ACGT')
                or parts[4] not in list('ACGT') or parts[3] == parts[4]):
            raise ValueError(f'Invalid GRCh38 SNV key: {key}')
    matches, rows, missing_positions = [], 0, 0
    with zipfile.ZipFile(path) as archive:
        if MEMBER not in archive.namelist():
            raise ValueError('Missing REVEL score member in the archive')
        with archive.open(MEMBER) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            if next(reader, None) != COLUMNS:
                raise ValueError('Unexpected REVEL columns')
            for fields in reader:
                rows += 1
                if len(fields) != len(COLUMNS):
                    raise ValueError(f'Malformed REVEL row {rows}')
                chrom, _, pos, ref, alt = fields[:5]
                if pos == '.':
                    missing_positions += 1
                else:
                    if not pos.isdigit() or int(pos) < 1:
                        raise ValueError(f'Invalid REVEL GRCh38 position on row {rows}')
                    key = f'GRCh38:{chrom.removeprefix("chr")}:{pos}:{ref}:{alt}'
                    if key in wanted:
                        score = float(fields[7])
                        if not np.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError(f'Invalid REVEL score for {key}')
                        matches.append([key, *fields[:7], score, fields[8]])
                if rows % 20_000_000 == 0:
                    print(f'  Read {rows:,} REVEL rows; retained {len(matches):,} annotations.', flush=True)
    annotations = pd.DataFrame(matches, columns=['variant_key', *COLUMNS])
    grouped = annotations.groupby('variant_key').agg(
        revel=('REVEL', 'max'), score_min=('REVEL', 'min'), n_annotations=('REVEL', 'size'))
    transcripts = annotations[['variant_key', 'Ensembl_transcriptid']].copy()
    transcripts['Ensembl_transcriptid'] = transcripts.Ensembl_transcriptid.str.split(';')
    transcripts = transcripts.explode('Ensembl_transcriptid')
    transcripts = transcripts[transcripts.Ensembl_transcriptid.notna() &
                              ~transcripts.Ensembl_transcriptid.isin(['', '.'])]
    grouped['n_transcripts'] = transcripts.groupby('variant_key').Ensembl_transcriptid.nunique()
    scores = pd.DataFrame({'variant_key': keys}).join(grouped, on='variant_key', validate='one_to_one')
    for column in ['n_annotations', 'n_transcripts']:
        scores[column] = scores[column].fillna(0).astype(int)
    scores['score_range'] = scores.revel - scores.score_min
    scores['status'] = np.where(scores.revel.notna(), 'scored', 'no_exact_allele_match')
    return scores, annotations, {'source_rows_scanned': rows, 'rows_without_grch38_position': missing_positions,
                                'archive_member': MEMBER, 'archive_columns': COLUMNS}


def run_baseline():
    started = time.monotonic()
    pilot, protocol, checks = baseline.load_pilot()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {
        'source': SOURCE, 'policy': POLICY,
        'q1_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
        'implementation_sha256': q1.digest_file(__file__),
        'shared_evaluation_sha256': q1.digest_file(baseline.__file__),
    }
    baseline.write_json(OUTPUT / 'baseline_protocol.json', identity)
    archive = baseline.ensure_scores(ARCHIVE, SOURCE)
    scores, annotations, scan = lookup_scores(archive, pilot.variant_key)
    scores.to_csv(OUTPUT / 'pilot_scores.csv', index=False)
    annotations.to_csv(OUTPUT / 'matched_annotations.csv', index=False)
    table, coverage, report = baseline.evaluate(pilot, scores, score_column='revel')
    table.to_csv(OUTPUT / 'pilot_evaluation.csv', index=False)
    coverage.to_csv(OUTPUT / 'coverage.csv', index=False)
    baseline.write_json(OUTPUT / 'validation_metrics.json', report)
    baseline.show_results(table, coverage, report, tool='REVEL', score_column='revel', output=OUTPUT)
    q1.verify_protocol()
    files = ['baseline_protocol.json', 'pilot_scores.csv', 'matched_annotations.csv',
             'pilot_evaluation.csv', 'coverage.csv', 'validation_metrics.json', 'coverage.png', 'validation_curves.png']
    provenance = {
        **identity, **scan, 'archive_sha256': q1.digest_file(archive),
        'q1_vcf_sha256': protocol['vcf_exports'], 'local_split_checks': checks,
        'precomputed_scores_used': True, 'models_fitted': False,
        'network': 'Required only when the verified archive is absent; no variant upload.',
        'compute': 'CPU lookup and evaluation; no GPU, fitting or protein annotation service.',
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scikit-learn', 'matplotlib']},
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'artifacts': {name: q1.digest_file(OUTPUT / name) for name in files},
    }
    # This manifest is the completion marker consumed by the comparison watcher.
    baseline.write_json(OUTPUT / 'baseline_provenance.json', provenance)
    print('Q1 hashes unchanged; REVEL predictions and evaluation saved to results/q8/revel/.')
    return report


def show_conclusion(alphamissense, revel):
    from IPython.display import Markdown, display
    lines = []
    for name, report in [('AlphaMissense', alphamissense), ('REVEL', revel)]:
        metrics = []
        for key, title in [('auroc', 'AUROC'), ('average_precision', 'average precision')]:
            metric = report['metrics'].get(key)
            if metric:
                ci = metric['ci95']
                interval = f' (95% CI {ci[0]:.3f}–{ci[1]:.3f})' if ci else ''
                metrics.append(f'{title} **{metric["value"]:.3f}**{interval}')
        measured = '; '.join(metrics) if metrics else 'Insufficient scored classes for ranking metrics'
        lines.append(f'**{name}:** {report["covered"]:,}/{report["total"]:,} validation variants scored '
                     f'({report["covered"] / report["total"]:.1%}); {report["missing"]:,} missing. {measured}.')
    display(Markdown('\n\n'.join(lines) + '\n\n'
        'Both predictors can be applied to this pilot by reproducible CPU lookup. **ClinVar remains the ground truth.** '
        'These metrics use each tool’s covered variants; use the [README comparison](../README.md#method-comparison) '
        'for results on the same variants. Intervals resample Q1 components. '
        'AlphaMissense calibration and REVEL/constituent training overlap with ClinVar remain unresolved; '
        'these are development results, with no claim about clinical validity or missing variants.'))
