"""CPU-only ClinVar exploration. Annotations here are audit data, not features."""

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import time
from urllib.parse import unquote

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


LABELS = {
    'Benign': 'benign', 'Likely_benign': 'benign',
    'Benign/Likely_benign': 'benign', 'Pathogenic': 'pathogenic',
    'Likely_pathogenic': 'pathogenic',
    'Pathogenic/Likely_pathogenic': 'pathogenic',
}
# Exact VCF spellings; unknown values remain unknown, never inferred from text.
REVIEW_STARS = {
    'practice_guideline': 4,
    'reviewed_by_expert_panel': 3,
    'criteria_provided,_multiple_submitters,_no_conflicts': 2,
    'criteria_provided,_single_submitter': 1,
    'criteria_provided,_conflicting_classifications': 1,
    'criteria_provided,_conflicting_interpretations': 1,  # older releases
    'no_assertion_criteria_provided': 0,
    'no_classification_provided': 0,
    'no_assertion_provided': 0,
    'no_classification_for_the_individual_variant': 0,
    'no_interpretation_for_the_single_variant': 0,
}
CHROMS = [str(i) for i in range(1, 23)] + ['X', 'Y']
INFO_FIELDS = ['ALLELEID', 'CLNSIG', 'CLNREVSTAT', 'CLNSIGCONF', 'GENEINFO',
               'MC', 'CLNVC', 'CLNSIGINCL', 'ONC', 'SCI', 'ORIGIN', 'CLNSIGSCV']
KEY = ['chrom', 'pos', 'ref', 'alt']
COLORS = ['#31688e', '#35b779', '#e69f00', '#cc6677', '#8172b3']

# Notebook configuration. Paths are relative to this module, not the kernel cwd.
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'data/clinvar.vcf'
ARCHIVE = Path('/root/data/clinvar.vcf.gz')
OUTPUT = ROOT / 'notebooks/results/q0'
SHA256 = '0524586dcf9e8c8f1fe7742450b0555ac55d04a6e9a262f61db1d15f113e622a'
FILE_DATE = '2026-09-05'
REFERENCE = 'GRCh38'
MIN_REVIEW_STARS = 2
WINDOW_BP = 8192  # Illustrative coordinate windows, not a model-context choice.


def ensure_input(path, archive):
    """Use the supplied local file, or atomically decompress the local archive."""
    path, archive = Path(path), Path(archive)
    if path.exists():
        return path
    if not archive.is_file():
        raise FileNotFoundError(f'Provide the pinned VCF at {path} or its archive at {archive}; '
                                'no newer release will be downloaded automatically.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    try:
        with gzip.open(archive, 'rb') as source, temporary.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def parse_record(line, line_number):
    """Parse the eight site columns and selected raw INFO values without expansion."""
    fields = line.rstrip('\r\n').split('\t')
    if len(fields) != 8:
        raise ValueError(f'Line {line_number}: expected eight site-only VCF columns')
    chrom, pos, variation_id, ref, alt, qual, filt, raw_info = fields
    if not pos.isdigit() or int(pos) < 1 or not chrom or not ref or not alt:
        raise ValueError(f'Line {line_number}: invalid coordinate or empty allele')
    info = {}
    for item in ([] if raw_info == '.' else raw_info.split(';')):
        key, separator, value = item.partition('=')
        if not key or key in info:
            raise ValueError(f'Line {line_number}: empty or duplicate INFO key')
        info[key] = value if separator else 'true'
    # Compare ALL INFO annotations except source identifiers; key, QUAL and FILTER
    # are checked too. Preserve source IDs separately when collapsing duplicates.
    identity = {'ALLELEID', 'CLNSIGSCV', 'ONCSCV', 'SCISCV'}
    signature = [qual, filt, sorted((k, v) for k, v in info.items() if k not in identity)]
    return [chrom, int(pos), variation_id, ref, alt, filt] + [
        '' if info.get(k, '.') == '.' else info.get(k, '') for k in INFO_FIELDS
    ] + [hashlib.sha256(json.dumps(signature).encode()).digest(),
         hashlib.sha256(line.rstrip('\r\n').encode()).digest()]


def gene_ids(raw):
    """Keep all unique positive NCBI Gene IDs, not just the first named gene."""
    ids = []
    for pair in raw.split('|') if raw else []:
        symbol, separator, identifier = pair.rpartition(':')
        if separator and symbol and identifier.isdigit() and int(identifier) > 0:
            ids.append(str(int(identifier)))
    return '|'.join(sorted(set(ids), key=int))


def read_clinvar(path, expected_sha256=None, expected_date=None, expected_reference='GRCh38'):
    """One full-file scan; checksum covers original bytes, including the header."""
    start = time.monotonic()
    digest, rows, metadata = hashlib.sha256(), [], {}
    header_seen = False
    with Path(path).open('rb') as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            line = raw.decode('utf-8')
            if line.startswith('##'):
                if header_seen:
                    raise ValueError('Metadata after the VCF column header')
                key, _, value = line[2:].strip().partition('=')
                if key in {'fileformat', 'fileDate', 'source', 'reference'}:
                    metadata[key] = value
                continue
            if line.startswith('#CHROM'):
                if header_seen or line.strip().split('\t') != [
                    '#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'
                ]:
                    raise ValueError('Invalid or repeated VCF column header')
                header_seen = True
                continue
            if not header_seen:
                raise ValueError('Missing VCF column header')
            rows.append(parse_record(line, line_number))
            if len(rows) % 500_000 == 0:
                print(f'Read {len(rows):,} records', flush=True)
    if not rows or metadata.get('source') != 'ClinVar' or not metadata.get('fileformat', '').startswith('VCFv4.'):
        raise ValueError('Expected a nonempty ClinVar VCF v4 file')
    if metadata.get('reference') != expected_reference:
        raise ValueError(f'Unexpected assembly: {metadata.get("reference")}')
    if expected_date and metadata.get('fileDate') != expected_date:
        raise ValueError('VCF fileDate differs from the pinned input')
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        raise ValueError('Input SHA-256 mismatch; review the data source before proceeding')
    columns = ['chrom', 'pos', 'variation_id', 'ref', 'alt', 'filter'] + INFO_FIELDS + ['annotation_hash', 'record_hash']
    frame = pd.DataFrame(rows, columns=columns)
    del rows
    for name in ['chrom', 'filter'] + INFO_FIELDS[:-1]:
        frame[name] = frame[name].astype('category')
    frame['label'] = frame.CLNSIG.map(LABELS).astype('object').fillna('excluded')
    frame['stars'] = frame.CLNREVSTAT.map(REVIEW_STARS).astype('float64').fillna(-1).astype('int8')
    frame['gene_ids'] = frame.GENEINFO.map(gene_ids).astype('object')
    frame['is_snv'] = frame.ref.isin(list('ACGT')) & frame.alt.isin(list('ACGT')) & frame.ref.ne(frame.alt)
    frame['conflict'] = (
        frame.CLNSIGCONF.ne('') | frame.CLNSIG.str.contains('conflict', case=False, regex=False)
        | frame.CLNREVSTAT.str.contains('conflict', case=False, regex=False)
          & ~frame.CLNREVSTAT.str.contains('no_conflicts', regex=False)
    )
    metadata.update(sha256=digest.hexdigest(), bytes=Path(path).stat().st_size,
                    records=len(frame), parse_seconds=round(time.monotonic() - start, 2),
                    input_name=Path(path).name, acquisition='Existing local archive; original download unverified')
    return frame, metadata


def annotation_counts(series, separator, transform=lambda value: value):
    """Count each annotation at most once per record, avoiding a large explode."""
    counts = Counter()
    for raw, count in series.value_counts().items():
        if raw:
            for value in set(transform(v) for v in str(raw).split(separator)):
                if value:
                    counts[value] += int(count)
    return pd.Series(counts, dtype='int64').sort_values(ascending=False)


def audit_data(frame):
    duplicates = frame.loc[frame.duplicated(KEY, keep=False)]
    signatures = duplicates.groupby(KEY, observed=True).annotation_hash.nunique()
    inconsistent = signatures[signatures > 1].reset_index()[KEY]
    unique = frame.drop_duplicates(KEY)
    locus_sizes = unique.groupby(['chrom', 'pos'], observed=True).size()
    metrics = {
        'Records': len(frame),
        'Unique variant keys': len(unique),
        'Exact duplicate rows beyond first': int(frame.record_hash.duplicated().sum()),
        'Repeated variant rows beyond first': int(frame.duplicated(KEY).sum()),
        'Inconsistent duplicate variant keys': len(inconsistent),
        'Repeated Variation IDs beyond first': int(frame.loc[~frame.variation_id.isin(['', '.']), 'variation_id'].duplicated().sum()),
        'Repeated Allele IDs beyond first': int(frame.loc[frame.ALLELEID.ne(''), 'ALLELEID'].duplicated().sum()),
        'Loci with multiple distinct alleles': int((locus_sizes > 1).sum()),
        'Records with multiple gene IDs': int(frame.gene_ids.str.contains('|', regex=False).sum()),
        'Records with no usable gene ID': int(frame.gene_ids.eq('').sum()),
        'Records with germline conflict evidence': int(frame.conflict.sum()),
        'Records with oncogenicity classification': int(frame.ONC.ne('').sum()),
        'Records with somatic impact classification': int(frame.SCI.ne('').sum()),
    }
    missing = {name: int(frame[name].eq('').sum()) for name in
               ['CLNSIG', 'CLNREVSTAT', 'GENEINFO', 'MC', 'CLNVC', 'ORIGIN']}
    unknown_reviews = frame.loc[frame.stars.eq(-1), 'CLNREVSTAT'].value_counts()
    unknown_reviews = unknown_reviews[unknown_reviews > 0]
    return {'metrics': metrics, 'missing': missing, 'inconsistent_keys': inconsistent,
            'unknown_reviews': unknown_reviews}


def context_groups(frame, window_bp=8192):
    """Connected components of overlapping half-open SNV windows, per chromosome.

    Endpoints are coordinate proxies: no FASTA or chromosome-length lookup.
    Touching windows do not overlap. Transitive overlaps join the same group.
    """
    if not isinstance(window_bp, int) or window_bp < 2 or window_bp % 2:
        raise ValueError('window_bp must be a positive even integer >= 2')
    loci = frame.loc[frame.is_snv, ['chrom', 'pos']].drop_duplicates().copy()
    loci['chrom'] = loci.chrom.astype(str)
    loci = loci.sort_values(['chrom', 'pos']).reset_index(drop=True)
    new_group = loci.chrom.ne(loci.chrom.shift()) | loci.pos.diff().ge(window_bp)
    anchors = loci.pos.where(new_group).ffill().astype('int64')
    loci['context_group'] = loci.chrom + ':' + anchors.astype(str)
    return loci


def propose_cohort(frame, audit, min_stars=2):
    """Ordered exclusions; no split assignment or model selection occurs here."""
    if min_stars not in (2, 3, 4):
        raise ValueError('This conservative cohort requires two or more review stars')
    bad = pd.MultiIndex.from_frame(audit['inconsistent_keys'])
    masks = [
        ('No inconsistent duplicate annotations', ~pd.MultiIndex.from_frame(frame[KEY]).isin(bad)),
        ('Chromosomes 1–22, X, Y', frame.chrom.isin(CHROMS)),
        ('Single A/C/G/T substitution', frame.is_snv),
        ('No germline conflict evidence', ~frame.conflict),
        ('Accepted benign/pathogenic label', frame.label.ne('excluded')),
        (f'At least {min_stars} review stars', frame.stars.ge(min_stars)),
        ('At least one usable gene ID', frame.gene_ids.ne('')),
    ]
    retained = np.ones(len(frame), dtype=bool)
    stages = [('All records', len(frame), 0)]
    for name, mask in masks:
        removed = retained & ~np.asarray(mask)
        retained &= np.asarray(mask)
        stages.append((name, int(retained.sum()), int(removed.sum())))
    candidates = frame.loc[retained].copy()
    # Aggregate all contributing IDs on repeated keys before taking one row.
    ids = ['variation_id', 'ALLELEID', 'CLNSIGSCV']
    repeated = candidates.duplicated(KEY, keep=False)
    if repeated.any():
        grouped = candidates.loc[repeated].groupby(KEY, observed=True)[ids].agg(
            lambda values: '|'.join(sorted({v for raw in values for v in str(raw).split('|') if v and v != '.'}))
        )
        cohort = candidates.drop_duplicates(KEY).set_index(KEY)
        for name in ids:
            cohort[name] = cohort[name].astype('object')
            cohort.loc[grouped.index, name] = grouped[name]
        cohort = cohort.reset_index()
    else:
        cohort = candidates
    stages.append(('Collapse consistent duplicate keys', len(cohort), len(candidates) - len(cohort)))
    funnel = pd.DataFrame(stages, columns=['stage', 'retained', 'removed'])
    if int(funnel.removed.sum()) + len(cohort) != len(frame):
        raise AssertionError('Filtering counts do not reconcile')
    assert_cohort(cohort, audit['inconsistent_keys'], min_stars)
    return cohort, funnel


def assert_cohort(cohort, inconsistent_keys, min_stars=2):
    """Fail on duplicate keys or violated eligibility rules; this is NOT a split audit."""
    checks = {
        'unique variant keys': not cohort.duplicated(KEY).any(),
        'eligible chromosomes': cohort.chrom.isin(CHROMS).all(),
        'SNVs': (cohort.ref.isin(list('ACGT')) & cohort.alt.isin(list('ACGT')) & cohort.ref.ne(cohort.alt)).all(),
        'positive coordinates': cohort.pos.gt(0).all(),
        'no germline conflicts': (~cohort.conflict).all(),
        'accepted raw labels': cohort.CLNSIG.isin(LABELS).all(),
        'correct derived labels': cohort.CLNSIG.map(LABELS).astype('object').eq(cohort.label).all(),
        'review threshold': cohort.CLNREVSTAT.map(REVIEW_STARS).astype(float).ge(min_stars).all(),
        'usable genes': cohort.GENEINFO.map(gene_ids).astype('object').ne('').all(),
        'no inconsistent duplicate keys': not pd.MultiIndex.from_frame(cohort[KEY]).isin(pd.MultiIndex.from_frame(inconsistent_keys)).any(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError('Cohort audit failed: ' + ', '.join(failed))
    return {name: bool(ok) for name, ok in checks.items()}


def details(title, value):
    """Show long tables or provenance only inside an HTML disclosure."""
    from IPython.display import HTML, display
    if isinstance(value, pd.Series):
        value = value.rename('count').to_frame()
    body = value.to_html(escape=True) if isinstance(value, pd.DataFrame) else '<pre>' + html.escape(json.dumps(value, indent=2)) + '</pre>'
    display(HTML(f'<details><summary>{html.escape(title)}</summary>{body}</details>'))


def _bar(ax, series, title, limit=12, color=COLORS[0]):
    series = series[series > 0].sort_values(ascending=False).head(limit).iloc[::-1]
    ax.barh([unquote(str(s)).replace('_', ' ') for s in series.index], series.values, color=color)
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_xlabel('Records' if 'gene' not in title.lower() else 'Record–gene associations')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: (
        f'{value / 1_000_000:g}M' if abs(value) >= 1_000_000
        else f'{value / 1_000:g}k' if abs(value) >= 1_000 else f'{value:g}'
    )))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)


def plot_composition(frame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    _bar(axes[0, 0], frame.CLNVC.astype('object').replace('', 'Missing').value_counts(), 'Variant types')
    chrom = frame.chrom.value_counts().reindex(CHROMS + ['MT']).fillna(0)
    axes[0, 1].bar(chrom.index, chrom.values, color=COLORS[0])
    axes[0, 1].set(title='Chromosome coverage', ylabel='Records')
    axes[0, 1].tick_params(axis='x', rotation=90)
    _bar(axes[1, 0], annotation_counts(frame.MC, ',', lambda v: v.split('|')[-1]), 'Molecular consequences (top 12)')
    _bar(axes[1, 1], annotation_counts(frame.GENEINFO, '|'), 'Most represented genes')
    plt.show()


def plot_classifications(frame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    _bar(axes[0], frame.CLNSIG.astype('object').replace('', 'Missing').value_counts(), 'Raw germline classifications')
    groups = frame.label.where(frame.label.ne('excluded'), 'Other / uncertain / missing')
    counts = pd.crosstab(frame.stars, groups).reindex(range(-1, 5), fill_value=0)
    counts.index = ['Unknown / missing', '0', '1', '2', '3', '4']
    counts.plot.barh(stacked=True, ax=axes[1], color=[COLORS[2], COLORS[1], COLORS[3]])
    axes[1].set(title='Review tiers and germline label groups', xlabel='Records', ylabel='Review stars')
    axes[1].legend(title='', loc='lower right', fontsize=8)
    plt.show()


def plot_quality(frame, audit):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    missing = pd.Series(audit['missing']).sort_values()
    axes[0].barh(missing.index, missing / len(frame) * 100, color=COLORS[2])
    axes[0].set(title='Missing annotations', xlabel='Percent of records')
    metrics = audit['metrics']
    _bar(axes[1], pd.Series({
        'Germline conflict evidence': metrics['Records with germline conflict evidence'],
        'No usable gene ID': metrics['Records with no usable gene ID'],
        'Multiple gene IDs': metrics['Records with multiple gene IDs'],
        'Repeated variant rows': metrics['Repeated variant rows beyond first'],
    }), 'Quality and grouping flags', color=COLORS[2])
    plt.show()


def plot_grouping(frame, loci, title='All SNVs', window_bp=8192):
    sizes = loci.groupby('context_group').size()
    genes = annotation_counts(frame.loc[frame.is_snv, 'gene_ids'], '|')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    axes[0].plot(np.arange(1, len(genes) + 1), genes.cumsum() / max(1, genes.sum()) * 100, color=COLORS[0])
    axes[0].set(xlabel='Genes ranked by representation', ylabel='Cumulative % of SNV–gene associations', title=f'{title}: gene concentration')
    if len(sizes):
        axes[1].hist(sizes, bins=np.unique(np.geomspace(1, max(2, sizes.max()), 30)), color=COLORS[4])
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
    axes[1].set(xlabel='Distinct loci per connected group', ylabel='Groups (log scale)', title=f'Overlapping {window_bp:,}-base windows')
    plt.show()
    return {'distinct_snv_loci': len(loci), 'context_groups': len(sizes),
            'largest_context_group_loci': int(sizes.max()) if len(sizes) else 0,
            'loci_in_overlapping_groups': int(sizes[sizes > 1].sum())}


def plot_cohort(cohort, funnel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    axes[0].barh(funnel.stage.iloc[::-1], funnel.retained.iloc[::-1], color=COLORS[0])
    axes[0].set(title='Candidate cohort: sequential filtering', xlabel='Records retained')
    counts = cohort.label.value_counts().reindex(['benign', 'pathogenic'], fill_value=0)
    bars = axes[1].bar(counts.index, counts.values, color=[COLORS[1], COLORS[3]])
    axes[1].bar_label(bars, fmt='{:,.0f}', padding=3)
    axes[1].set(title='Candidate class balance', ylabel='Unique SNVs')
    axes[1].margins(y=.15)
    plt.show()


def export_results(frame, cohort, funnel, audit, provenance, output_dir, window_bp=8192, min_stars=2):
    """Export audit metadata, NOT predictor features or a frozen split manifest."""
    assert_cohort(cohort, audit['inconsistent_keys'], min_stars)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = KEY + ['variation_id', 'ALLELEID', 'CLNSIGSCV', 'GENEINFO', 'gene_ids',
                     'CLNSIG', 'CLNREVSTAT', 'stars', 'label', 'MC']
    manifest = cohort[columns].copy()
    manifest['chrom'] = manifest.chrom.astype(str)
    manifest = manifest.merge(context_groups(cohort, window_bp), on=['chrom', 'pos'], validate='many_to_one')
    manifest['variant_key'] = provenance['reference'] + ':' + manifest[KEY].astype(str).agg(':'.join, axis=1)
    manifest['locus_id'] = manifest.chrom + ':' + manifest.pos.astype(str)
    manifest['assembly'] = provenance['reference']
    manifest['cohort_status'] = 'proposed_not_split'
    if len(manifest) != len(cohort) or manifest.context_group.isna().any():
        raise AssertionError('Missing or duplicated context-group assignments')
    manifest = manifest.sort_values(KEY)
    manifest.to_csv(output_dir / 'cohort.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    funnel.to_csv(output_dir / 'filter_counts.csv', index=False)
    audit['inconsistent_keys'].to_csv(output_dir / 'inconsistent_variant_keys.csv', index=False)
    tables = {'raw_labels': frame.CLNSIG.value_counts(), 'review_status': frame.CLNREVSTAT.value_counts(),
              'variant_types': frame.CLNVC.value_counts(), 'chromosomes': frame.chrom.value_counts(),
              'consequences': annotation_counts(frame.MC, ','), 'gene_coverage': annotation_counts(frame.GENEINFO, '|')}
    for name, counts in tables.items():
        counts[counts > 0].rename_axis('value').rename('records').to_csv(output_dir / f'{name}.csv')
    summary = {
        'input': provenance, 'audit': audit['metrics'], 'missing_annotations': audit['missing'],
        'unknown_review_statuses': {str(k): int(v) for k, v in audit['unknown_reviews'].items()},
        'cohort_records': len(cohort), 'cohort_labels': {k: int(v) for k, v in cohort.label.value_counts().items()},
        'cohort_gene_ids': len(annotation_counts(cohort.gene_ids, '|')),
        'configuration': {'min_review_stars': min_stars, 'accepted_labels': LABELS, 'chromosomes': CHROMS,
                          'window_bp': window_bp, 'variant_type': 'single A/C/G/T substitution', 'random_seed': None,
                          'gene_requirement': 'at least one positive NCBI Gene ID',
                          'duplicates': 'exclude inconsistent annotations; merge consistent keys and source identifiers'},
        'limitations': ['Full dataset explored; no untouched test set established.',
                        'No split assignment, learned preprocessing, model fitting, or feature extraction.',
                        'Coordinate windows are proxies; reference bases and chromosome ends not validated.',
                        'Duplicate checks compare representations in this VCF, not normalized indels.',
                        'Source identifiers, multi-gene links and context groups must be grouped jointly before freezing splits.',
                        'Gene families, sequence homology, patient relationships, annotation circularity and pretrained contamination not resolved.',
                        'Annotations and labels in this manifest are audit metadata, not predictor inputs.'],
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    environment = {name: importlib.metadata.version(name) for name in
                   ['numpy', 'pandas', 'matplotlib', 'ipython', 'ipykernel', 'nbformat', 'nbclient']}
    environment.update(python=platform.python_version(), platform=platform.platform(),
                       recorded_utc=datetime.now(timezone.utc).isoformat(),
                       peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
                       implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(f'Exported {len(cohort):,} proposed variants across {summary["cohort_gene_ids"]:,} gene IDs to {output_dir}')
    return summary


# Section calls used by Q0.ipynb. Inputs and results stay explicit between cells.
def load_data():
    started = time.monotonic()
    variants, provenance = read_clinvar(
        ensure_input(INPUT, ARCHIVE), expected_sha256=SHA256,
        expected_date=FILE_DATE, expected_reference=REFERENCE,
    )
    print(f'{len(variants):,} records · {provenance["reference"]} · {provenance["fileDate"]}')
    details('Input provenance', provenance)
    details('Run configuration', {
        'input': str(INPUT), 'archive': str(ARCHIVE), 'output': str(OUTPUT),
        'min_review_stars': MIN_REVIEW_STARS, 'illustrative_window_bp': WINDOW_BP,
    })
    return variants, provenance, started


def show_classifications(variants):
    plot_classifications(variants)
    details('Exact germline label mapping', LABELS)
    details('Exact review-status mapping', REVIEW_STARS)


def show_quality(variants):
    audit = audit_data(variants)
    plot_quality(variants, audit)
    details('Quality and classification-availability counts', audit['metrics'])
    details('Unknown or missing review statuses', audit['unknown_reviews'])
    return audit


def show_grouping(variants):
    loci = context_groups(variants, WINDOW_BP)
    grouping = plot_grouping(variants, loci, window_bp=WINDOW_BP)
    details('SNV coordinate-group diagnostics', grouping)


def show_cohort(variants, audit):
    cohort, funnel = propose_cohort(variants, audit, MIN_REVIEW_STARS)
    plot_cohort(cohort, funnel)
    details('Sequential exclusions (rows, then unique variants)', funnel)
    details('Executable cohort checks', assert_cohort(cohort, audit['inconsistent_keys'], MIN_REVIEW_STARS))
    details('Candidate preview: audit annotations, not features',
            cohort[KEY + ['CLNSIG', 'CLNREVSTAT', 'gene_ids']].head(10))
    return cohort, funnel


def save_results(variants, cohort, funnel, audit, provenance, started):
    summary = export_results(
        variants, cohort, funnel, audit, provenance, OUTPUT,
        window_bp=WINDOW_BP, min_stars=MIN_REVIEW_STARS,
    )
    print(f'Completed in {(time.monotonic() - started) / 60:.1f} minutes (CPU).')
    details('Cohort class counts', summary['cohort_labels'])
    details('Limits of this audit', summary['limitations'])
    return summary
