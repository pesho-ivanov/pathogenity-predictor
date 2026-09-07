"""Q1 full missense partitions, extending the frozen pilot without reassignment.

The q1 module prepares a July pilot and a complete relationship graph.
This workflow reuses that verified cohort, removes the sampling cap, checks
every eligible DNA context, and publishes separately frozen full-dataset inputs.
"""

from contextlib import ExitStack
import hashlib
import importlib.metadata
import platform
import time

import matplotlib.pyplot as plt
import pandas as pd

from . import q0, q1

OUTPUT = q1.OUTPUT / 'full'
VCF_FILES = {'train': q1.ROOT / 'data/clinvar-train.vcf',
             'validation': q1.ROOT / 'data/clinvar-test.vcf'}


def config():
    return {
        'parent_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
        'implementation_sha256': q1.digest_file(__file__),
        'eligibility': q1.protocol_config()['eligibility'],
        'sampling': 'All eligible missense SNVs; no sample cap or class balancing',
        'seed': q1.SEED, 'train_target': .70, 'validation_target': .30,
        'assignment': 'Preserve every frozen full-cohort assignment; never rebalance or rehash',
        'context_bp': q1.CONTEXT, 'assembly': 'GRCh38',
        'clinvar_date': q1.CLINVAR_DATE, 'clinvar_sha256': q1.CLINVAR_SHA256,
        'reference_sha256': q1.REFERENCE_SHA256,
        'grouping': 'Full Q0 SNV cohort including nonmissense bridges; join all retained identical DNA contexts including reverse complements',
        'sequence_exclusions': 'Truncated or non-ACGT windows, without replacement; REF mismatch is fatal',
        'vcf_roles': {path.name: split for split, path in VCF_FILES.items()},
        'evaluation_scope': 'Development: clinvar-test.vcf is validation, including previously evaluated pilot variants; no untouched test set',
    }


def expand_groups(full, dna):
    """Merge any newly discovered identical contexts without changing any split."""
    full = full.copy()
    graph = q1.Components(len(full))
    seen = {}
    for i, component in enumerate(full.component):
        if component in seen:
            graph.join(i, seen[component])
        else:
            seen[component] = i
    q1.merge_identical_contexts(graph, full, dna)
    full['component'] = graph.names(full.variant_key)
    if full.groupby('component').split.nunique().gt(1).any():
        raise AssertionError('Identical DNA connects frozen train and validation groups; stop without moving variants')
    return full


def audit(full, retained, dna, exclusions):
    """Check complete coverage, inherited membership, and full-scale leakage."""
    checks = q1.audit_splits(full, retained, dna)
    checks = {name.replace('full:', 'grouping cohort:').replace('pilot:', 'full missense:'): value
              for name, value in checks.items()}
    scope = q1.audit_missense_scope(full, retained)
    checks.update({name.replace('pilot:', 'full missense:'): value for name, value in scope.items()})
    candidates = set(full.loc[full.MC.map(q1.has_missense), 'variant_key'])
    kept, excluded = set(retained.variant_key), set(exclusions.variant_key)
    if (kept & excluded or kept | excluded != candidates or exclusions.variant_key.duplicated().any()
            or set(exclusions.reason) - {'truncated_window', 'non_ACGT_window'}):
        raise AssertionError('Full missense coverage must equal every candidate minus recorded sequence exclusions')
    pilot = q1.read_csv(q1.OUTPUT / 'split_manifest.csv')
    if not set(pilot.variant_key) <= kept:
        raise AssertionError('An existing pilot variant was lost')
    inherited = retained.set_index('variant_key').loc[pilot.variant_key]
    for column in ['split', 'ref_context_hash', 'alt_context_hash']:
        if inherited[column].tolist() != pilot[column].tolist():
            raise AssertionError(f'Pilot {column} changed')
    checks['all eligible missense variants retained or explicitly excluded for sequence quality'] = True
    checks['all pilot variants preserve their split and exact reference/alternate DNA'] = True
    return checks


def export_vcfs(manifest, source=q1.CLINVAR_INPUT, expected_sha256=q1.CLINVAR_SHA256):
    """Publish source-order, byte-preserved VCF records only after all checks pass."""
    if manifest.variant_key.duplicated().any() or set(manifest.split) != set(q1.SPLITS):
        raise ValueError('VCF export requires unique variants and both splits')
    assignment = dict(zip(manifest.variant_key, manifest.split))
    temporary = {split: path.with_name(path.name + '.partial') for split, path in VCF_FILES.items()}
    seen, header_seen, digest = set(), False, hashlib.sha256()
    try:
        with ExitStack() as stack:
            for path in temporary.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            outputs = {split: stack.enter_context(path.open('wb')) for split, path in temporary.items()}
            with source.open('rb') as stream:
                for raw in stream:
                    digest.update(raw)
                    if raw.startswith(b'#'):
                        if header_seen:
                            raise ValueError('Unexpected VCF header after column header')
                        if raw.startswith(b'#CHROM'):
                            if raw.rstrip(b'\r\n').split(b'\t') != [
                                    b'#CHROM', b'POS', b'ID', b'REF', b'ALT', b'QUAL', b'FILTER', b'INFO']:
                                raise ValueError('Invalid VCF column header')
                            header_seen = True
                        for output in outputs.values():
                            output.write(raw)
                        continue
                    if not header_seen:
                        raise ValueError('Missing VCF column header')
                    fields = raw.rstrip(b'\r\n').split(b'\t')
                    if len(fields) != 8:
                        raise ValueError('Invalid VCF record')
                    key = 'GRCh38:' + b':'.join(fields[i] for i in [0, 1, 3, 4]).decode()
                    if key in assignment:
                        info = dict(item.split('=', 1) for item in fields[7].decode().split(';') if '=' in item)
                        if not q1.has_missense(info.get('MC')):
                            raise ValueError(f'Selected variant lacks missense annotation: {key}')
                        if key in seen:
                            raise ValueError(f'Duplicate selected VCF variant: {key}')
                        seen.add(key)
                        outputs[assignment[key]].write(raw)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Source VCF checksum mismatch')
        if not header_seen or seen != set(assignment):
            raise ValueError('VCF export does not cover the exact split manifest')
        hashes = {VCF_FILES[split].name: q1.digest_file(path) for split, path in temporary.items()}
        # Reapply the original quality/label filters to the actual exported records.
        labels = {split: read_labels(path, hashes[VCF_FILES[split].name], manifest, split)
                  for split, path in temporary.items()}
        for path in VCF_FILES.values():
            if path.exists():
                q1.verify_file(path, hashes[path.name])
        for split, path in VCF_FILES.items():
            if not path.exists():
                temporary[split].replace(path)
        return hashes, labels
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def read_labels(path, expected_hash, manifest, split):
    frame, _ = q0.read_clinvar(path, expected_hash, q1.CLINVAR_DATE)
    eligible, _ = q0.propose_cohort(frame, q0.audit_data(frame))
    if len(eligible) != len(frame) or not eligible.MC.astype('object').map(q1.has_missense).all():
        raise AssertionError('Exported VCF violates the original quality, label or missense filters')
    frame['variant_key'] = 'GRCh38:' + frame[['chrom', 'pos', 'ref', 'alt']].astype(str).agg(':'.join, axis=1)
    keys = manifest.loc[manifest.split.eq(split), 'variant_key'].tolist()
    if frame.variant_key.duplicated().any() or set(frame.variant_key) != set(keys):
        raise AssertionError('VCF membership differs from the full split manifest')
    labels = frame.set_index('variant_key').loc[keys, 'label'].map({'benign': 0, 'pathogenic': 1})
    if labels.isna().any() or set(labels) != {0, 1}:
        raise AssertionError('Both unambiguous classes are required; do not search for a different split')
    return labels.rename('label').reset_index()


def verify_protocol():
    q1.verify_protocol()
    protocol = q1.read_json(OUTPUT / 'protocol.json')
    if protocol['config'] != config():
        raise RuntimeError('Frozen full-dataset configuration changed; preserve the recorded experiment')
    for name, digest in protocol['artifacts'].items():
        q1.verify_file(OUTPUT / name, digest)
    if set(protocol['vcf_exports']) != {path.name for path in VCF_FILES.values()}:
        raise ValueError('Protocol must identify both full VCF exports')
    for path in VCF_FILES.values():
        q1.verify_file(path, protocol['vcf_exports'][path.name])
    return protocol


def load_partition_labels(split, keys):
    """Full-dataset counterpart of q1.load_partition_labels, with strict alignment."""
    if split not in q1.SPLITS:
        raise ValueError('Only train and validation are defined')
    protocol = verify_protocol()
    manifest = q1.read_csv(OUTPUT / 'split_manifest.csv')
    expected = manifest.loc[manifest.split.eq(split), 'variant_key'].tolist()
    if list(keys) != expected:
        raise ValueError('Requested label order differs from the full frozen split')
    path = VCF_FILES[split]
    return read_labels(path, protocol['vcf_exports'][path.name], manifest, split).label.to_numpy(dtype=int)


def prepare():
    """Reproduce the parent cohort, expand every eligible variant, audit and freeze."""
    started = time.monotonic()
    print(f'Verify ClinVar {q1.CLINVAR_DATE}, shared with Q0, and inherited group assignments.', flush=True)
    q1.prepare()  # Downloads and reconstructs the parent from scratch if absent.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT / 'protocol.json').exists():
        full = q1.read_csv(q1.OUTPUT / 'full_cohort_groups.csv.gz')
        candidates = full.loc[full.MC.map(q1.has_missense)].reset_index(drop=True)
        print(f'Extract reference/alternate DNA for all {len(candidates):,} eligible missense variants.', flush=True)
        dna, exclusions = q1.extract_contexts(candidates, q1.REFERENCE)
        full = expand_groups(full, dna)
        retained = full.set_index('variant_key').loc[dna.variant_key].reset_index()
        retained['ref_context_hash'] = dna.ref_sequence.map(q1.context_hash)
        retained['alt_context_hash'] = dna.alt_sequence.map(q1.context_hash)
        checks = audit(full, retained, dna, exclusions)
        print('Leakage checks passed; export and recheck the complete VCF records.', flush=True)
        hashes, labels = export_vcfs(retained)
        full.to_csv(OUTPUT / 'full_cohort_groups.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        retained.to_csv(OUTPUT / 'split_manifest.csv', index=False)
        dna.to_csv(OUTPUT / 'sequences.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        exclusions.to_csv(OUTPUT / 'sequence_exclusions.csv', index=False)
        funnel = q1.read_csv(q1.OUTPUT / 'filter_counts.csv')
        funnel.loc[len(funnel)] = ['Complete A/C/G/T reference and alternate contexts', len(retained), len(exclusions)]
        funnel.to_csv(OUTPUT / 'filter_counts.csv', index=False)
        for split, target in labels.items():
            target.to_csv(OUTPUT / f'{split}_labels.csv', index=False)
        q1.write_json(OUTPUT / 'leakage_checks.json', checks)
        files = ['full_cohort_groups.csv.gz', 'split_manifest.csv', 'sequences.csv.gz',
                 'sequence_exclusions.csv', 'filter_counts.csv', 'leakage_checks.json',
                 'train_labels.csv', 'validation_labels.csv']
        q1.write_json(OUTPUT / 'protocol.json', {
            'config': config(), 'vcf_exports': hashes,
            'artifacts': {name: q1.digest_file(OUTPUT / name) for name in files},
        }, frozen=True)
    verify_protocol()
    full = q1.read_csv(OUTPUT / 'full_cohort_groups.csv.gz')
    retained = q1.read_csv(OUTPUT / 'split_manifest.csv')
    dna = q1.read_csv(OUTPUT / 'sequences.csv.gz')
    exclusions = q1.read_csv(OUTPUT / 'sequence_exclusions.csv')
    checks = audit(full, retained, dna, exclusions)
    if checks != q1.read_json(OUTPUT / 'leakage_checks.json'):
        raise AssertionError('Recorded full-dataset checks differ from rerun checks')
    q1.write_json(OUTPUT / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'biopython', 'matplotlib']},
        'compute': 'CPU only; no model inference or fitting', 'seed': q1.SEED,
        'seconds_this_run': time.monotonic() - started,
    })
    print(f'{len(retained):,} full-dataset variants; {len(exclusions):,} sequence exclusions; {len(checks)} audit groups passed.')
    return retained, dna


def settings():
    print(f'All eligible missense SNVs · ClinVar {q1.CLINVAR_DATE}, shared with Q0 · no sampling cap')
    print('1,024-base contexts · seed 42 · original 70/30 group assignment · CPU only')
    print('data/clinvar-train.vcf = training; data/clinvar-test.vcf = validation.')


def show_splits(retained):
    protocol = verify_protocol()
    if not retained.equals(q1.read_csv(OUTPUT / 'split_manifest.csv')):
        raise ValueError('Displayed data must match the frozen full manifest')
    counts = retained.groupby('split').agg(variants=('variant_key', 'size'), components=('component', 'nunique')).reindex(q1.SPLITS)
    for split in q1.SPLITS:
        labels = load_partition_labels(split, retained.loc[retained.split.eq(split), 'variant_key'])
        counts.loc[split, 'B'] = int((labels == 0).sum())
        counts.loc[split, 'P'] = int((labels == 1).sum())
        counts.loc[split, 'genes'] = len(set('|'.join(retained.loc[retained.split.eq(split), 'gene_ids']).split('|')))
    counts = counts.astype(int)
    counts['percent'] = (counts.variants / len(retained) * 100).round(2)
    counts['vcf'] = [VCF_FILES[split].name for split in q1.SPLITS]
    counts.to_csv(OUTPUT / 'split_counts.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    for ax, column, title in zip(axes, ['variants', 'genes'], ['Full missense variants', 'Distinct genes · no overlap']):
        bars = ax.bar(q1.SPLITS, counts[column], color=['#31688e', '#e69f00'])
        ax.bar_label(bars, fmt='{:,.0f}', padding=3)
        ax.set(title=title, ylabel='Count')
        ax.margins(y=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUTPUT / 'split_sizes.png', dpi=160)
    plt.show()
    q0.details('Counts, class balance and proportions', counts)
    q0.details('Eligibility filters and sequence exclusions', q1.read_csv(OUTPUT / 'filter_counts.csv'))
    q0.details('Passed leakage checks', q1.read_json(OUTPUT / 'leakage_checks.json'))
    q0.details('Frozen full-dataset protocol', protocol)


def show_conclusion():
    from IPython.display import Markdown, display
    verify_protocol()
    counts = q1.read_csv(OUTPUT / 'split_manifest.csv').split.value_counts()
    total = int(counts.sum())
    display(Markdown(
        f'**Conclusion.** The full missense dataset contains **{counts["train"]:,} training '
        f'({counts["train"] / total:.2%})** and **{counts["validation"]:,} validation '
        f'({counts["validation"] / total:.2%})** variants. Genes and checked DNA contexts stay '
        'within one split, and all July pilot assignments are preserved. '
        'Variants shared with the archived September cohort keep their splits. '
        'Whole groups make the original 70/30 target approximate. '
        'The test-named file remains development validation; homology, shared patients '
        'and pretraining overlap are not ruled out.'))


def run_checks():
    import io
    import unittest
    from . import test_q1_full
    q1.run_checks()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_q1_full))
    if not result.wasSuccessful():
        raise AssertionError(stream.getvalue())
    print(f'{result.testsRun} additional full-dataset integrity tests passed.')
