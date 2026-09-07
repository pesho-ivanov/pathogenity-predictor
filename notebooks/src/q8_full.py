"""Published missense predictors on every variant in Q1's full validation VCF.

Reuse the pilot's exact-allele lookups and score definitions, but bind all new
artifacts to the complete current cohort. No predictor is fitted or recalibrated.
"""

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import q0, q1, q1_full, q8_baseline as baseline, q8_dbnsfp as db
from . import q8_remaining as remaining, q8_revel as revel

OUTPUT = baseline.OUTPUT / 'full'
DB_INPUT = db.DATA / 'full-validation'
METHODS = {
    'AlphaMissense': {'folder': '', 'column': 'am_pathogenicity', 'source': baseline.SOURCE,
                      'policy': baseline.POLICY},
    'REVEL': {'folder': 'revel', 'column': 'revel', 'source': revel.SOURCE, 'policy': revel.POLICY},
    **{method['tool']: {'folder': slug, 'column': 'score', 'source': db.SOURCE,
                       'policy': remaining.POLICY, 'method': method}
       for slug, method in remaining.METHODS.items()},
}
POLICY = {
    'scope': 'All variants in the current full clinvar-test.vcf; development validation',
    'inputs': 'Both frozen VCFs verified; only validation genomic keys enter score lookup',
    'fitting': 'None; fixed published models and aggregation rules; no recalibration or threshold selection',
    'labels': 'Full Q1 validation VCF ClinVar labels loaded after lookup: benign=0, pathogenic=1',
    'missing': 'Retain every validation variant; absent/empty scores remain missing, never imputed',
    'bootstrap': baseline.POLICY['bootstrap'], 'bootstrap_repetitions': 1000, 'seed': 42,
}
for _spec in METHODS.values():
    _spec['policy'] = {**_spec['policy'], **POLICY}


def sources():
    paths = [Path(module.__file__) for module in
             [q0, q1, q1_full, baseline, db, remaining, revel]]
    paths += [Path(__file__), q1.ROOT / 'requirements.txt']
    return {str(path.relative_to(q1.ROOT)): q1.digest_file(path) for path in paths}


def prepare():
    """Rerun full split/sequence audits and freeze all choices before score lookup."""
    full, _ = q1_full.prepare()
    validation = full.loc[full.split.eq('validation'), ['variant_key', 'split', 'component']].reset_index(drop=True)
    parent = q1_full.verify_protocol()
    protocol = {
        'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json'),
        'q1_vcf_sha256': parent['vcf_exports'], 'clinvar_date': q1.CLINVAR_DATE,
        'scope': 'full_validation', 'validation_variants': len(validation),
        'validation_keys_sha256': q1.fingerprint(validation.variant_key.tolist()),
        'policy': POLICY, 'methods': METHODS, 'sources': sources(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    q1.write_json(OUTPUT / 'protocol.json', protocol, frozen=True)
    checks = q1.read_json(q1_full.OUTPUT / 'leakage_checks.json')
    q1.write_json(OUTPUT / 'leakage_checks.json', checks, frozen=True)
    print(f"ClinVar {q1.CLINVAR_DATE}: all {len(validation):,} full validation variants; "
          f"{len(checks)} split/sequence audit groups passed. No training or sampling.", flush=True)
    return {'validation': validation, 'protocol': protocol}


def verify(cohort):
    parent = q1_full.verify_protocol()
    protocol = q1.read_json(OUTPUT / 'protocol.json')
    if (protocol != cohort['protocol'] or protocol['sources'] != sources()
            or protocol['q1_protocol_sha256'] != q1.digest_file(q1_full.OUTPUT / 'protocol.json')
            or protocol['q1_vcf_sha256'] != parent['vcf_exports']
            or protocol['validation_keys_sha256'] != q1.fingerprint(cohort['validation'].variant_key.tolist())):
        raise ValueError('Full Q8 experiment identity or validation membership changed')
    if (list(cohort['validation'].columns) != ['variant_key', 'split', 'component']
            or not cohort['validation'].split.eq('validation').all()):
        raise ValueError('Q8 scoring requires the full validation partition without clinical annotations')
    return protocol


def acquire_dbnsfp(cohort):
    verify(cohort)
    return db.fetch_annotations(cohort['validation'].variant_key, directory=DB_INPUT,
                                cache_directory=db.DATA / 'blocks', table_name='validation_annotations.tsv.gz')


def evaluate(validation, scores, column):
    """Join outcomes only after lookup, requiring complete exact validation order."""
    if (scores.variant_key.tolist() != validation.variant_key.tolist()
            or scores.variant_key.duplicated().any() or not validation.split.eq('validation').all()):
        raise ValueError('Scores must retain every full validation key in frozen order')
    table = validation.merge(scores, on='variant_key', validate='one_to_one', sort=False)
    table['label'] = q1_full.load_partition_labels('validation', validation.variant_key.tolist())
    report = baseline.metric_summary(table, POLICY['bootstrap_repetitions'], POLICY['seed'], column)
    coverage = []
    for label, name in [(None, 'all'), (0, 'benign'), (1, 'pathogenic')]:
        selected = table if label is None else table[table.label.eq(label)]
        covered = int(np.isfinite(selected[column]).sum())
        coverage.append({'split': 'validation', 'class': name, 'total': len(selected),
                         'covered': covered, 'missing': len(selected) - covered,
                         'coverage_fraction': covered / len(selected) if len(selected) else None})
    return table, pd.DataFrame(coverage), report


def run_method(tool, cohort, raw=None, acquisition=None):
    """Execute exact lookup, evaluation and provenance for one fixed predictor."""
    protocol = verify(cohort)
    spec = METHODS[tool]
    output = OUTPUT / spec['folder']
    output.mkdir(parents=True, exist_ok=True)
    identity = {'scope': 'full_validation', 'tool': tool, 'method': spec,
                'q1_protocol_sha256': protocol['q1_protocol_sha256'],
                'q1_vcf_sha256': protocol['q1_vcf_sha256'], 'sources': sources(),
                'full_q8_protocol_sha256': q1.digest_file(OUTPUT / 'protocol.json'),
                'policy': {**spec['policy'], **POLICY}}
    q1.write_json(output / 'baseline_protocol.json', identity, frozen=True)
    completion = output / 'baseline_provenance.json'
    if completion.exists():
        previous = q1.read_json(completion)
        if any(previous[key] != value for key, value in identity.items()):
            raise ValueError('Preserve the previous full Q8 run before changing its identity')
        completion.unlink()  # The new run publishes a completion marker only after all checks.
    started = time.monotonic()
    keys = cohort['validation'].variant_key
    if tool in {'AlphaMissense', 'REVEL'}:
        module = baseline if tool == 'AlphaMissense' else revel
        archive = baseline.ensure_scores(module.ARCHIVE, module.SOURCE)
        scores, annotations, acquisition = module.lookup_scores(archive, keys)
        acquisition = {**acquisition, 'archive_sha256': q1.digest_file(archive)}
    else:
        if raw is None or acquisition is None:
            raise ValueError('Acquire the full-cohort dbNSFP annotations before scoring')
        expected_raw, expected = db.fetch_annotations(keys, directory=DB_INPUT, cache_directory=db.DATA / 'blocks',
                                                      table_name='validation_annotations.tsv.gz')
        if acquisition != expected or not raw.equals(expected_raw):
            raise ValueError('dbNSFP acquisition provenance differs from the full validation extraction')
        scores, annotations = remaining.score_alleles(raw, keys, spec['method'])
    scores.to_csv(output / 'validation_scores.csv', index=False)
    annotations.to_csv(output / 'matched_annotations.csv', index=False)
    table, coverage, report = evaluate(cohort['validation'], scores, spec['column'])
    table.to_csv(output / 'validation_predictions.csv', index=False)
    coverage.to_csv(output / 'coverage.csv', index=False)
    baseline.write_json(output / 'validation_metrics.json', report)
    verify(cohort)
    files = ['baseline_protocol.json', 'validation_scores.csv', 'matched_annotations.csv',
             'validation_predictions.csv', 'coverage.csv', 'validation_metrics.json']
    elapsed = round(time.monotonic() - started, 2)
    baseline.write_json(completion, {
        **identity, 'acquisition': acquisition, 'models_fitted': False, 'precomputed_scores_used': True,
        'local_split_checks_sha256': q1.digest_file(OUTPUT / 'leakage_checks.json'),
        'elapsed_seconds': elapsed,
        'runtime_scope': ('CPU validation lookup and evaluation, including archive verification/downloads; excludes Q1 audits and upstream model training.'
                          if tool in {'AlphaMissense', 'REVEL'} else
                          'CPU validation score aggregation and evaluation; excludes shared dbNSFP acquisition, Q1 audits and upstream model training.'),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scikit-learn', 'biopython']},
        'artifacts': {name: q1.digest_file(output / name) for name in files},
    })
    print(f'{tool}: {report["covered"]:,}/{report["total"]:,} validation variants scored; {elapsed:.1f} seconds.', flush=True)
    return report


def report_primate_access(cohort):
    from IPython.display import Markdown, display
    protocol = verify(cohort)
    output = OUTPUT / 'primateai3d'
    output.mkdir(parents=True, exist_ok=True)
    status = {'status': 'blocked', 'reason': remaining.PRIMATE_REASON,
              'source': remaining.PRIMATE_URL, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'predictions_produced': False, 'q1_protocol_sha256': protocol['q1_protocol_sha256']}
    baseline.write_json(output / 'access_status.json', status)
    display(Markdown('**PrimateAI-3D remains blocked:** ' + status['reason']))
    return status


def run_remaining(cohort):
    raw, acquisition = acquire_dbnsfp(cohort)
    return {tool: run_method(tool, cohort, raw, acquisition)
            for tool in ['SIFT4G', 'PolyPhen-2', 'EVE']}


def show_results(reports):
    """Show numeric coverage/intervals and overlaid validation curves."""
    from IPython.display import Markdown, display
    from sklearn.metrics import precision_recall_curve, roc_curve
    from . import comparison
    rows = [{'Method': tool, 'Scored / validation': f'{r["covered"]:,} / {r["total"]:,}',
             'AUROC [95% CI]': comparison.format_metric(r, 'auroc'),
             'Average precision [95% CI]': comparison.format_metric(r, 'average_precision')}
            for tool, r in reports.items()]
    display(Markdown(comparison.markdown_table(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for tool in reports:
        spec = METHODS[tool]
        table = pd.read_csv(OUTPUT / spec['folder'] / 'validation_predictions.csv')
        covered = table[np.isfinite(table[spec['column']])]
        if covered.label.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(covered.label, covered[spec['column']])
        precision, recall, _ = precision_recall_curve(covered.label, covered[spec['column']])
        axes[0].plot(fpr, tpr, label=tool)
        axes[1].plot(recall, precision, label=tool)
    axes[0].plot([0, 1], [0, 1], ':', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Validation ROC: each scored subset')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall: each scored subset')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    plt.close(fig)


def conclude(reports, access):
    from IPython.display import Markdown, display
    from . import comparison
    result, _ = comparison.refresh()
    shared = {f'q8:{tool}': result['common'].get(f'q8:{tool}') for tool in reports}
    if all(shared.values()):
        rows = [{'Method': tool, 'Shared variants': shared[f'q8:{tool}']['total'],
                 'AUROC [95% CI]': comparison.format_metric(shared[f'q8:{tool}'], 'auroc'),
                 'Average precision [95% CI]': comparison.format_metric(shared[f'q8:{tool}'], 'average_precision')}
                for tool in reports]
        display(Markdown('**Comparison on the same variants as the README**\n\n' + comparison.markdown_table(rows)))
    ranges = [report['covered'] / report['total'] for report in reports.values()]
    display(Markdown(
        f'**Conclusion.** Five published predictors provide scores for **{min(ranges):.1%}–{max(ranges):.1%}** '
        f'of the **{next(iter(reports.values()))["total"]:,}** full validation variants. '
        'The shared-subset table supports direct comparison; missing-score coverage remains a limitation. '
        + ('PrimateAI-3D is blocked on licensed data access. ' if access['status'] == 'blocked' else '')
        + 'These are development results against ClinVar, with unresolved clinical-training/calibration '
        'and evolutionary-data overlap. They do not establish independent clinical validity.'))
    return result
