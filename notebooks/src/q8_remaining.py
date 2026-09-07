"""Benchmark SIFT4G, PolyPhen-2 HumVar and EVE; report PrimateAI-3D access."""

from datetime import datetime, timezone
import importlib.metadata
import re
import time

import numpy as np
import pandas as pd

from . import comparison as c, q1, q8_baseline as baseline, q8_dbnsfp as db

METHODS = {
    'sift4g': {'tool': 'SIFT4G', 'column': 'SIFT4G_score', 'invert': True,
               'exposure': 'Evolutionary conservation; no clinical-label fitting in core scoring. External sequence overlap is unaudited.'},
    'polyphen2': {'tool': 'PolyPhen-2', 'column': 'Polyphen2_HVAR_score', 'invert': False,
                 'exposure': 'HumVar selected before evaluation. Known disease training variants may overlap ClinVar; exact overlap is unresolved.'},
    'eve': {'tool': 'EVE', 'column': 'EVE_score', 'invert': False,
            'exposure': 'Evolutionary MSA model, without clinical-label fitting. Protein/position coverage is limited; no EVE category or uncertainty filtering.'},
}
POLICY = {
    **baseline.POLICY,
    'matching': 'Exact GRCh38 chromosome, 1-based position, REF and ALT in dbNSFP4.9a; no liftover or allele complementation.',
    'aggregation': 'Maximum damaging-direction score across exact-allele rows and score-list entries. '
                   'SIFT4G uses 1-raw_score; PolyPhen-2 HumVar and EVE retain raw scores. '
                   'Preserve raw scores, mapping fields and score-list indices. Missing scores remain missing.',
    'exposure': 'Tool-specific training and external sequence exposure are recorded separately; ClinVar remains the reference labels.',
}
PRIMATE_REASON = ('PrimateAI-3D has not been run: Illumina requires a signed license agreement and supplies '
                  'the score/model download link by email. No approved link or licensed score file was provided. '
                  'The original PrimateAI scores in dbNSFP are a different model and are not substituted.')
PRIMATE_URL = 'https://github.com/Illumina/PrimateAI-3D/tree/a3c210bcd0880c72b307b5fba0a2a17bddda59cd'


def score_alleles(raw, variant_keys, method):
    """Aggregate numeric score lists without using labels or clinical annotations."""
    keys = list(variant_keys)
    c.require(len(set(keys)) == len(keys), 'Duplicate pilot keys')
    c.require(set(raw.variant_key).issubset(keys), 'Unexpected alleles in score extraction')
    annotations = []
    for source_row, row in enumerate(raw.to_dict('records')):
        key = f'GRCh38:{row["#chr"]}:{row["pos(1-based)"]}:{row["ref"]}:{row["alt"]}'
        c.require(row['variant_key'] == key, 'dbNSFP allele fields differ from the requested key')
        values = re.split('[;,]', row[method['column']])
        for index, token in enumerate(values):
            if token in ['', '.']:
                original = oriented = np.nan
            else:
                original = float(token)
                c.require(np.isfinite(original) and 0 <= original <= 1, 'Invalid external score')
                oriented = 1 - original if method['invert'] else original
            keep = {name: row[name] for name in db.FIELDS[:10]}
            annotations.append({'variant_key': key, **keep, 'source_row': source_row, 'score_index': index,
                                'raw_score': original, 'score': oriented})
    columns = ['variant_key', *db.FIELDS[:10], 'source_row', 'score_index', 'raw_score', 'score']
    frame = pd.DataFrame(annotations, columns=columns)
    grouped = frame.groupby('variant_key').agg(score=('score', 'max'), score_min=('score', 'min'),
                    n_scores=('score', 'count'), n_source_rows=('source_row', 'nunique'))
    scores = pd.DataFrame({'variant_key': keys}).join(grouped, on='variant_key', validate='one_to_one')
    for column in ['n_scores', 'n_source_rows']:
        scores[column] = scores[column].fillna(0).astype(int)
    scores['score_range'] = scores.score - scores.score_min
    scores['status'] = np.where(scores.score.notna(), 'scored',
                               np.where(scores.n_source_rows.gt(0), 'score_missing', 'no_exact_allele_match'))
    return scores, frame


def run_baselines():
    started = time.monotonic()
    pilot, parent, checks = baseline.load_pilot()
    identity = {'source': db.SOURCE, 'policy': POLICY,
                'q1_protocol_sha256': c.digest(q1.OUTPUT / 'protocol.json'),
                'implementation_sha256': c.digest(__file__),
                'shared_evaluation_sha256': c.digest(baseline.__file__),
                'acquisition_implementation_sha256': c.digest(db.__file__)}
    # Freeze all model/aggregation choices before acquiring scores or loading outcomes.
    for slug, method in METHODS.items():
        output = baseline.OUTPUT / slug
        output.mkdir(parents=True, exist_ok=True)
        baseline.write_json(output / 'baseline_protocol.json', {**identity, 'method': method})
    raw, acquisition = db.fetch_annotations(pilot.variant_key)
    reports = {}
    for slug, method in METHODS.items():
        output = baseline.OUTPUT / slug
        scores, annotations = score_alleles(raw, pilot.variant_key, method)
        scores.to_csv(output / 'pilot_scores.csv', index=False)
        annotations.to_csv(output / 'matched_annotations.csv', index=False)
        table, coverage, report = baseline.evaluate(pilot, scores, score_column='score')
        table.to_csv(output / 'pilot_evaluation.csv', index=False)
        coverage.to_csv(output / 'coverage.csv', index=False)
        baseline.write_json(output / 'validation_metrics.json', report)
        baseline.show_results(table, coverage, report, tool=method['tool'], score_column='score', output=output)
        q1.verify_protocol()
        files = ['baseline_protocol.json', 'pilot_scores.csv', 'matched_annotations.csv', 'pilot_evaluation.csv',
                 'coverage.csv', 'validation_metrics.json', 'coverage.png', 'validation_curves.png']
        provenance = {**identity, 'method': method, 'acquisition': acquisition,
                      'q1_vcf_sha256': parent['vcf_exports'], 'local_split_checks': checks,
                      'models_fitted': False, 'precomputed_scores_used': True,
                      'network': 'HTTP byte-range downloads only; no variant or label uploads.',
                      'packages': {p: importlib.metadata.version(p) for p in
                                   ['biopython', 'numpy', 'pandas', 'scikit-learn', 'matplotlib']},
                      'elapsed_seconds': round(time.monotonic() - started, 2),
                      'artifacts': {name: c.digest(output / name) for name in files}}
        baseline.write_json(output / 'baseline_provenance.json', provenance)
        reports[method['tool']] = report
        print(f'{method["tool"]}: {report["covered"]}/{report["total"]} validation variants scored.', flush=True)
    return reports


def report_primate_access():
    """Record the observed access blocker without producing invented predictions."""
    from IPython.display import Markdown, display
    output = baseline.OUTPUT / 'primateai3d'
    output.mkdir(parents=True, exist_ok=True)
    status = {'status': 'blocked', 'reason': PRIMATE_REASON, 'source': PRIMATE_URL,
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'predictions_produced': False,
              'q1_protocol_sha256': c.digest(q1.OUTPUT / 'protocol.json')}
    baseline.write_json(output / 'access_status.json', status)
    display(Markdown('**PrimateAI-3D: access blocked.** ' + PRIMATE_REASON + f' [Official access instructions]({PRIMATE_URL}).'))
    return status


def show_conclusion(alphamissense, revel, remaining, primate_access):
    from IPython.display import Markdown, display
    reports = {'AlphaMissense': alphamissense, 'REVEL': revel, **remaining}
    rows = [{'Method': name, 'Scored / validation': f'{r["covered"]:,} / {r["total"]:,}',
             'AUROC [95% CI]': c.format_metric(r, 'auroc'),
             'Average precision [95% CI]': c.format_metric(r, 'average_precision')} for name, r in reports.items()]
    display(Markdown(c.markdown_table(rows) + '\n\n'
        'Five predictors have reproducible pilot scores; PrimateAI-3D remains blocked on licensed data access. '
        '**ClinVar remains the reference labels.** Metrics above use each method’s scored validation subset; '
        'the [README comparison](../README.md#method-comparison) recomputes metrics on the common scored subset. '
        'Coverage is a separate limitation. Intervals resample Q1 components, and external training/calibration '
        'overlap remains unaudited. These are development results.'))
