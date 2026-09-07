"""Pinned AlphaMissense lookup and development evaluation on Q1's frozen pilot.

Lookup accepts genomic keys only. ClinVar outcomes are loaded after scoring;
no predictor, aggregation rule, calibration or threshold is fitted here.
"""

import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from . import q1

OUTPUT = q1.ROOT / 'notebooks/results/q8'
ARCHIVE = q1.ROOT / 'data/alphamissense/AlphaMissense_hg38.tsv.gz'
SOURCE = {
    'tool': 'AlphaMissense', 'release': '2023', 'assembly': 'GRCh38',
    'url': 'https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz?generation=1691073413649109',
    'generation': '1691073413649109', 'updated': '2023-08-03T14:36:53.693Z',
    'bytes': 642961469, 'md5': '9fd167735f16a1b87da6eb3e4c25fcb5',
    'metadata_url': 'https://storage.googleapis.com/storage/v1/b/dm_alphamissense/o/AlphaMissense_hg38.tsv.gz?generation=1691073413649109',
    'documentation': 'https://github.com/google-deepmind/alphamissense/blob/fe2dc845f93310abd6c1b0e8955d7a96c2144d66/README.md',
    'citation': 'Cheng et al. (2023), Science, doi:10.1126/science.adg7492',
    'license_note': 'Archive header: CC BY-NC-SA 4.0; later official README: predictions CC BY 4.0. Both statements retained; original archive unchanged.',
}
POLICY = {
    'matching': 'Exact GRCh38 chromosome, 1-based position, REF and ALT; remove chr prefix only; no liftover or reverse complementation.',
    'aggregation': 'Maximum continuous am_pathogenicity across all matching transcript annotations in the pinned hg38 release; keep all annotations.',
    'missing': 'No exact allele match remains unscored; never impute benign scores.',
    'labels': 'Canonical Q1 pilot VCF ClinVar labels: benign=0, pathogenic=1; loaded after lookup.',
    'evaluation': 'Validation only, on covered variants; report all-variant coverage by split and class. No separate test set.',
    'bootstrap': 'Resample Q1 validation components with replacement; retain every member, then restrict to scored variants; percentile 95% intervals.',
    'bootstrap_repetitions': 1000, 'seed': 42,
    'exposure': 'AlphaMissense released scores and thresholds use ClinVar calibration. Exact calibration membership is not audited; this is not ClinVar-independent ground truth.',
    'scope': 'Q1 audits certify local fixed splits and 1024-base contexts only, not independence of external protein/evolutionary inputs or calibration data.',
}
COLUMNS = ['#CHROM', 'POS', 'REF', 'ALT', 'genome', 'uniprot_id',
           'transcript_id', 'protein_variant', 'am_pathogenicity', 'am_class']
ANNOTATIONS = ['variant_key', 'uniprot_id', 'transcript_id', 'protein_variant',
               'am_pathogenicity', 'am_class']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ensure_scores(path=ARCHIVE, source=None):
    """Download atomically when absent; fail on corrupt existing/downloaded bytes."""
    source = SOURCE if source is None else source
    path = Path(path)

    def verify(candidate):
        if candidate.stat().st_size != source['bytes']:
            raise ValueError('AlphaMissense archive size mismatch')
        if q1.digest_file(candidate, 'md5') != source['md5']:
            raise ValueError('AlphaMissense archive checksum mismatch')

    if path.exists():
        verify(path)
        print('Reusing verified AlphaMissense archive.', flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    print(f'Downloading AlphaMissense ({source["bytes"] / 1e6:.0f} MB) to {path}.', flush=True)
    try:
        with urllib.request.urlopen(source['url'], timeout=60) as response, partial.open('wb') as output:
            total, reported = 0, 0
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                total += len(block)
                if total - reported >= 100_000_000:
                    print(f'  {total / 1e6:.0f} MB downloaded', flush=True)
                    reported = total
        verify(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def load_pilot():
    """Verify Q1 hashes and rerun its full-cohort, sequence and missense audits."""
    if not (q1.OUTPUT / 'protocol.json').exists():
        raise FileNotFoundError('Run Q1-clinvar-split.ipynb first to reproduce the frozen missense pilot.')
    protocol = q1.verify_protocol()
    pilot = q1.read_csv(q1.OUTPUT / 'split_manifest.csv')
    full = q1.read_csv(q1.OUTPUT / 'full_cohort_groups.csv.gz')
    dna = q1.read_csv(q1.OUTPUT / 'sequences.csv.gz')
    checks = q1.audit_splits(full, pilot, dna)
    checks.update(q1.audit_missense_scope(full, pilot))
    if len(pilot) != 5000 or pilot.split.value_counts().to_dict() != {'train': 3658, 'validation': 1342}:
        raise ValueError('Expected the frozen 5000-variant Q1 missense pilot')
    return pilot, protocol, checks


def lookup_scores(path, variant_keys):
    """Stream the gzip release once, retaining only exact target allele matches."""
    keys = list(variant_keys)
    wanted = set(keys)
    if len(keys) != len(wanted):
        raise ValueError('Lookup keys must be unique')
    for key in keys:
        parts = key.split(':')
        if (len(parts) != 5 or parts[0] != 'GRCh38' or not parts[2].isdigit()
                or int(parts[2]) < 1 or parts[3] not in 'ACGT' or len(parts[3]) != 1
                or parts[4] not in 'ACGT' or len(parts[4]) != 1 or parts[3] == parts[4]):
            raise ValueError(f'Invalid GRCh38 SNV key: {key}')
    annotations, comments = [], []
    rows = 0
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.startswith('#CHROM\t'):
                if line.rstrip('\r\n').split('\t') != COLUMNS:
                    raise ValueError('Unexpected AlphaMissense columns')
                break
            if not line.startswith('#'):
                raise ValueError('Missing AlphaMissense column header')
            comments.append(line.rstrip('\r\n'))
        else:
            raise ValueError('Missing AlphaMissense column header')
        for line in stream:
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != len(COLUMNS) or fields[4] != 'hg38':
                raise ValueError('Malformed row or non-hg38 AlphaMissense data')
            rows += 1
            chrom, pos, ref, alt = fields[:4]
            key = f'GRCh38:{chrom.removeprefix("chr")}:{pos}:{ref}:{alt}'
            if key in wanted:
                score = float(fields[8])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'Invalid AlphaMissense score for {key}')
                annotations.append([key, *fields[5:8], score, fields[9]])
            if rows % 20_000_000 == 0:
                print(f'  Read {rows:,} score rows; retained {len(annotations):,} annotations.', flush=True)
    annotation_frame = pd.DataFrame(annotations, columns=ANNOTATIONS)
    grouped = annotation_frame.groupby('variant_key').agg(
        am_pathogenicity=('am_pathogenicity', 'max'),
        score_min=('am_pathogenicity', 'min'),
        n_annotations=('transcript_id', 'size'), n_transcripts=('transcript_id', 'nunique'))
    scores = pd.DataFrame({'variant_key': keys}).join(grouped, on='variant_key', validate='one_to_one')
    for column in ['n_annotations', 'n_transcripts']:
        scores[column] = scores[column].fillna(0).astype(int)
    scores['score_range'] = scores.am_pathogenicity - scores.score_min
    scores['status'] = np.where(scores.am_pathogenicity.notna(), 'scored', 'no_exact_allele_match')
    return scores, annotation_frame, {'source_rows_scanned': rows, 'archive_header': comments}


def metric_summary(validation, repetitions=1000, seed=42):
    """Component bootstrap, including uncovered members before coverage selection."""
    labels = validation.label.to_numpy(dtype=int)
    scores = validation.am_pathogenicity.to_numpy(dtype=float)
    covered = np.isfinite(scores)
    if not set(labels).issubset({0, 1}) or (covered & ((scores < 0) | (scores > 1))).any():
        raise ValueError('Invalid labels or scores')
    result = {'total': len(validation), 'covered': int(covered.sum()),
              'missing': int((~covered).sum()), 'components': int(validation.component.nunique()),
              'metrics': {}, 'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
              'bootstrap_valid_replicates': 0, 'status': 'both_classes_required'}
    if len(np.unique(labels[covered])) < 2:
        return result
    metrics = {'auroc': roc_auc_score, 'average_precision': average_precision_score}
    members = list(validation.groupby('component', sort=True).indices.values())
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in metrics}
    for _ in range(repetitions):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        idx = idx[covered[idx]]
        if len(np.unique(labels[idx])) < 2:
            continue
        for name, metric in metrics.items():
            draws[name].append(float(metric(labels[idx], scores[idx])))
    result['bootstrap_valid_replicates'] = len(draws['auroc'])
    result['status'] = 'ok'
    for name, metric in metrics.items():
        result['metrics'][name] = {
            'value': float(metric(labels[covered], scores[covered])),
            'ci95': np.quantile(draws[name], [.025, .975]).tolist() if len(draws[name]) >= 2 else None,
        }
    return result


def evaluate(pilot, scores):
    """Load outcomes only after genomic lookup, preserving manifest order."""
    if scores.variant_key.tolist() != pilot.variant_key.tolist():
        raise ValueError('Scores must cover each ordered pilot key exactly once')
    table = pilot[['variant_key', 'split', 'component']].merge(scores, on='variant_key', validate='one_to_one', sort=False)
    table['label'] = -1
    for split in q1.SPLITS:
        mask = table.split.eq(split)
        table.loc[mask, 'label'] = q1.load_partition_labels(split, table.loc[mask, 'variant_key'].tolist())
    coverage = []
    for split in q1.SPLITS:
        for label, name in [(None, 'all'), (0, 'benign'), (1, 'pathogenic')]:
            selected = table[table.split.eq(split)]
            if label is not None:
                selected = selected[selected.label.eq(label)]
            covered = int(selected.am_pathogenicity.notna().sum())
            coverage.append({'split': split, 'class': name, 'total': len(selected),
                             'covered': covered, 'missing': len(selected) - covered,
                             'coverage_fraction': covered / len(selected) if len(selected) else None})
    report = metric_summary(table[table.split.eq('validation')], POLICY['bootstrap_repetitions'], POLICY['seed'])
    return table, pd.DataFrame(coverage), report


def run_baseline():
    start = time.monotonic()
    pilot, protocol, checks = load_pilot()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {'source': SOURCE, 'policy': POLICY,
                'q1_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
                'implementation_sha256': q1.digest_file(__file__)}
    # Record the source, mapping and evaluation decisions before reading outcomes.
    write_json(OUTPUT / 'baseline_protocol.json', identity)
    archive = ensure_scores()
    scores, annotations, scan = lookup_scores(archive, pilot.variant_key)
    scores.to_csv(OUTPUT / 'pilot_scores.csv', index=False)
    annotations.to_csv(OUTPUT / 'matched_annotations.csv', index=False)
    table, coverage, report = evaluate(pilot, scores)
    table.to_csv(OUTPUT / 'pilot_evaluation.csv', index=False)
    coverage.to_csv(OUTPUT / 'coverage.csv', index=False)
    write_json(OUTPUT / 'validation_metrics.json', report)
    show_results(table, coverage, report)
    q1.verify_protocol()
    files = ['baseline_protocol.json', 'pilot_scores.csv', 'matched_annotations.csv',
             'pilot_evaluation.csv', 'coverage.csv', 'validation_metrics.json',
             'coverage.png', 'validation_curves.png']
    provenance = {
        **identity, **scan, 'archive_sha256': q1.digest_file(archive),
        'q1_vcf_sha256': protocol['vcf_exports'], 'local_split_checks': checks,
        'variants_read': True, 'precomputed_scores_used': True, 'models_fitted': False,
        'network': 'Required only when the verified archive is absent; no variant upload.',
        'compute': 'CPU lookup and evaluation; no model weights or GPU required.',
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scikit-learn', 'matplotlib']},
        'elapsed_seconds': round(time.monotonic() - start, 2),
        'artifacts': {name: q1.digest_file(OUTPUT / name) for name in files},
    }
    write_json(OUTPUT / 'baseline_provenance.json', provenance)
    print('Q1 hashes unchanged; scores, coverage, validation metrics and provenance saved to results/q8/.')
    return report


def show_results(table, coverage, report):
    from .q8 import details
    fig, ax = plt.subplots(figsize=(7, 3), layout='constrained')
    overall = coverage[coverage['class'].eq('all')]
    ax.barh(overall.split, overall.covered, color='#28866b', label='Scored')
    ax.barh(overall.split, overall.missing, left=overall.covered, color='#c9cfd4', label='Missing')
    for i, row in enumerate(overall.itertuples()):
        ax.text(20, i, f'{row.covered:,}/{row.total:,} ({row.coverage_fraction:.1%})', va='center', color='white')
    ax.set(xlabel='Frozen pilot variants', title='AlphaMissense exact-allele coverage')
    ax.legend(loc='lower right')
    fig.savefig(OUTPUT / 'coverage.png', dpi=160)
    plt.show()
    plt.close(fig)
    details('Coverage by partition and ClinVar class', coverage.to_html(index=False, border=0))
    validation = table[table.split.eq('validation') & table.am_pathogenicity.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    if report['status'] == 'ok':
        labels, scores = validation.label, validation.am_pathogenicity
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, color='#28866b')
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[1].plot(recall, precision, color='#28866b')
        axes[1].axhline(labels.mean(), ls='--', color='gray', label='Covered prevalence')
        axes[1].legend()
    else:
        for ax in axes:
            ax.text(.5, .5, 'Metrics unavailable: both scored classes required', ha='center', wrap=True)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Validation ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Validation precision–recall')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    plt.close(fig)


def show_conclusion(report):
    from IPython.display import Markdown, display
    if report['status'] == 'ok':
        values = []
        for key, name in [('auroc', 'AUROC'), ('average_precision', 'average precision')]:
            metric = report['metrics'][key]
            interval = metric['ci95']
            ci = f' (95% CI {interval[0]:.3f}–{interval[1]:.3f})' if interval else ''
            values.append(f'{name} **{metric["value"]:.3f}**{ci}')
        result = '; '.join(values) + '.'
    else:
        result = 'Ranking metrics are unavailable because the scored subset lacks both classes.'
    display(Markdown(
        f'**AlphaMissense is the selected reference predictor; ClinVar remains the ground truth.** '
        f'It covers **{report["covered"]:,}/{report["total"]:,} validation variants** '
        f'({report["covered"] / report["total"]:.1%}); {report["missing"]:,} remain unscored. '
        f'{result}\n\n'
        'Intervals resample Q1 components and describe the covered development cohort. '
        'AlphaMissense uses ClinVar calibration, whose exact overlap remains unresolved. '
        'These results do not establish clinical validity, performance on missing variants, '
        'or superiority to other tools. No Q2 model was retrained or compared against archived broad-SNV results.'))
