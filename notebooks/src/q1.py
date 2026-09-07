"""Q1: reproducible ClinVar missense sampling and frozen related-variant splits.

Clinical labels determine Q0 eligibility, but never sampling or split assignment.
This module runs on CPU and does not import the modeling or GPU stack.
"""

import gzip
from contextlib import ExitStack
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import q0

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'notebooks/results/q1'
# Q0 and Q1 share one dated source and one verified download cache.
CLINVAR_INPUT = q0.INPUT
CLINVAR_ARCHIVE = q0.ARCHIVE
CLINVAR_URL = q0.URL
CLINVAR_ARCHIVE_MD5 = q0.ARCHIVE_MD5
CLINVAR_SHA256 = q0.SHA256
CLINVAR_DATE = q0.FILE_DATE
REFERENCE = ROOT / 'data/hg38.fa.gz'
REFERENCE_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
REFERENCE_MD5 = '1c9dcaddfa41027f17cd8f7a82c7293b'
REFERENCE_SHA256 = 'c1dd87068c254eb53d944f71e51d1311964fce8de24d6fc0effc9c61c01527d4'
SEED = 42
N_VARIANTS = 5000
CONTEXT = 1024
SPLITS = ['train', 'validation']
VCF_FILES = {'train': ROOT / 'data/clinvar-train-pilot.vcf',
             'validation': ROOT / 'data/clinvar-test-pilot.vcf'}
DNA_COLUMNS = ['variant_key', 'ref_sequence', 'alt_sequence']
BASES = 'ACGT'
MISSENSE_SO = 'SO:0001583'
# Frozen July assignments; all 341,718 variants shared with September retain
# their split. Changed component anchors need the 14 label-independent overrides
# below. The migration audit and prior manifests are preserved in results/archive.
FULL_ASSIGNMENTS_SHA256 = '31d028f56340493402d797e12a242d58b86ac29ee8ba771b6816a131c2ca2f0c'
PREVIOUS_ASSIGNMENTS_SHA256 = '89888d17823a691eaae3af09b88102461a0f177a0c29b8fa221dac24b2916eed'
INHERITED_COMPONENT_SPLITS = {
    'GRCh38:10:102918570:C:T': 'train',
    'GRCh38:11:65779458:C:A': 'train',
    'GRCh38:12:6935780:G:A': 'train',
    'GRCh38:14:23522287:G:A': 'validation',
    'GRCh38:14:75669413:T:C': 'validation',
    'GRCh38:15:33473445:C:T': 'validation',
    'GRCh38:19:35546813:G:A': 'validation',
    'GRCh38:21:44235375:G:A': 'train',
    'GRCh38:22:22548334:A:G': 'train',
    'GRCh38:2:159349937:T:C': 'validation',
    'GRCh38:2:213056961:T:C': 'validation',
    'GRCh38:4:106095574:C:A': 'train',
    'GRCh38:6:31865727:G:A': 'validation',
    'GRCh38:6:31935916:C:T': 'train',
}


def has_missense(raw):
    """Accept an exact missense SO identifier among comma-separated MC terms."""
    return isinstance(raw, str) and any(
        term.partition('|')[0] == MISSENSE_SO for term in raw.split(','))


def audit_missense_scope(full, pilot):
    if not pilot.MC.astype('object').map(has_missense).all():
        raise AssertionError('Pilot contains a variant without an exact missense annotation')
    if fingerprint(sorted(zip(full.variant_key, full.split))) != FULL_ASSIGNMENTS_SHA256:
        raise AssertionError('Existing full-cohort split assignments changed')
    return {'pilot: every variant has an exact MC missense annotation': True,
            'full: frozen July assignments preserve every shared September variant split': True}


def digest_file(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()



def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()



def read_json(path):
    return json.loads(Path(path).read_text())



def write_json(path, value, frozen=False):
    path = Path(path)
    if frozen and path.exists():
        if read_json(path) != value:
            raise RuntimeError(f'Frozen artifact differs: {path}. Preserve the recorded experiment.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)



def verify_file(path, expected, algorithm='sha256'):
    if digest_file(path, algorithm) != expected:
        raise ValueError(f'{algorithm} mismatch: {path}')



def acquire_reference():
    """Fetch the initial UCSC GRCh38 FASTA, never a moving latest release."""
    if not REFERENCE.exists():
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        temporary = REFERENCE.with_suffix('.gz.partial')
        subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--continue-at', '-',
                        '--output', str(temporary), REFERENCE_URL], check=True)
        verify_file(temporary, REFERENCE_MD5, 'md5')
        verify_file(temporary, REFERENCE_SHA256)
        temporary.replace(REFERENCE)
    verify_file(REFERENCE, REFERENCE_MD5, 'md5')
    verify_file(REFERENCE, REFERENCE_SHA256)
    return REFERENCE



class Components:
    """Union-find; all edges are formed without labels."""

    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]

    def names(self, keys):
        roots = [self.find(i) for i in range(len(keys))]
        anchors = {}
        for root, key in zip(roots, keys):
            anchors[root] = min(anchors.get(root, key), key)
        return [anchors[root] for root in roots]



def full_components(manifest, context=CONTEXT):
    """Join genes, source IDs, loci, and transitive overlapping half-open windows.

    Run on the FULL eligible cohort, before sampling; an unselected bridging
    variant must still connect its neighbors. Different chromosomes never overlap.
    """
    graph = Components(len(manifest))
    for column in ['gene_ids', 'variation_id', 'ALLELEID', 'variant_key', 'locus_id']:
        seen = {}
        for i, raw in enumerate(manifest[column]):
            for value in str(raw).split('|'):
                if value and value != '.':
                    if value in seen:
                        graph.join(i, seen[value])
                    else:
                        seen[value] = i
    ordered = manifest.reset_index(drop=True).sort_values(['chrom', 'pos'])
    previous = None
    for i, chrom, pos in ordered[['chrom', 'pos']].itertuples():
        if previous is not None:
            j, last_chrom, last_pos = previous
            if chrom == last_chrom and pos - last_pos < context:
                graph.join(i, j)
        previous = (i, chrom, pos)
    return graph



def seeded_hash(key, purpose):
    return hashlib.sha256(f'{SEED}:{purpose}:{key}'.encode()).hexdigest()



def sample_indices(manifest, count=N_VARIANTS):
    if manifest.variant_key.duplicated().any():
        raise ValueError('Duplicate variant keys before sampling')
    return sorted(range(len(manifest)), key=lambda i: seeded_hash(manifest.variant_key.iloc[i], 'sample'))[:count]



def assign_split(component):
    fraction = int(seeded_hash(component, 'split'), 16) / 2**256
    return INHERITED_COMPONENT_SPLITS.get(component, 'train' if fraction < .70 else 'validation')



def reverse_complement(sequence):
    return sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]



def context_hash(sequence):
    return hashlib.sha256(min(sequence, reverse_complement(sequence)).encode()).hexdigest()



def mutate_context(chromosome, pos, ref, alt, context=CONTEXT):
    """Return (reference, alternate, exclusion); REF mismatch is always fatal."""
    if not 0 < pos <= len(chromosome) or chromosome[pos - 1] != ref:
        raise ValueError(f'Genome REF mismatch at position {pos}: expected {ref}')
    if ref not in BASES or alt not in BASES or ref == alt:
        raise ValueError('Expected one A/C/G/T substitution')
    start = pos - 1 - context // 2
    sequence = chromosome[max(0, start):start + context]
    if start < 0 or len(sequence) != context:
        return None, None, 'truncated_window'
    if set(sequence) - set(BASES):
        return None, None, 'non_ACGT_window'
    alternate = sequence[:context // 2] + alt + sequence[context // 2 + 1:]
    return sequence, alternate, ''



def extract_contexts(selected, fasta):
    from Bio.SeqIO.FastaIO import SimpleFastaParser
    sequences, exclusions, observed = [], [], set()
    by_chromosome = {str(chrom): rows for chrom, rows in selected.groupby('chrom')}
    with gzip.open(fasta, 'rt') as stream:
        for title, chromosome in SimpleFastaParser(stream):
            name = title.split()[0]
            chrom = name.removeprefix('chr')
            if name != 'chr' + chrom or chrom not in by_chromosome:
                continue
            if chrom in observed:
                raise ValueError('Duplicate chromosome in FASTA')
            observed.add(chrom)
            chromosome = chromosome.upper()
            for row in by_chromosome[chrom].itertuples():
                ref, alt, reason = mutate_context(chromosome, row.pos, row.ref, row.alt)
                if reason:
                    exclusions.append([row.variant_key, reason])
                else:
                    sequences.append([row.variant_key, ref, alt])
            if observed == set(by_chromosome):
                break
    if observed != set(by_chromosome):
        raise ValueError(f'Missing FASTA chromosomes: {set(by_chromosome) - observed}')
    dna = pd.DataFrame(sequences, columns=DNA_COLUMNS).set_index('variant_key')
    order = [key for key in selected.variant_key if key in dna.index]
    return dna.loc[order].reset_index(), pd.DataFrame(exclusions, columns=['variant_key', 'reason'])



def merge_identical_contexts(graph, full_manifest, sequences):
    positions = dict(zip(full_manifest.variant_key, range(len(full_manifest))))
    seen = {}
    for row in sequences.itertuples(index=False):
        index = positions[row.variant_key]
        for sequence in [row.ref_sequence, row.alt_sequence]:
            key = context_hash(sequence)
            if key in seen:
                graph.join(index, seen[key])
            else:
                seen[key] = index



def audit_splits(full_manifest, pilot, sequences):
    """Independent cross-split checks; all recorded cohort bridges remain visible."""
    checks = {}
    for name, frame in [('full', full_manifest), ('pilot', pilot)]:
        if frame.variant_key.duplicated().any() or set(frame.split) - set(SPLITS):
            raise AssertionError(f'Invalid {name} manifest')
        for column in ['component', 'gene_ids', 'variation_id', 'ALLELEID', 'variant_key', 'locus_id']:
            seen = {}
            for raw, split in zip(frame[column], frame.split):
                for key in str(raw).split('|'):
                    if key and key != '.':
                        if key in seen and seen[key] != split:
                            raise AssertionError(f'Cross-split {column}: {key}')
                        seen[key] = split
        previous = None
        for row in frame.sort_values(['chrom', 'pos']).itertuples():
            if row.start != row.pos - 1 - CONTEXT // 2 or row.end != row.start + CONTEXT:
                raise AssertionError('Incorrect sequence interval')
            if previous and row.chrom == previous.chrom and row.start < previous.end and row.split != previous.split:
                raise AssertionError('Overlapping sequence windows cross splits')
            previous = row
        checks[f'{name}: genes, IDs, loci, components and overlapping windows disjoint'] = True
    if not pilot.variant_key.isin(full_manifest.variant_key).all():
        raise AssertionError('Pilot contains a variant absent from the full cohort')
    expected = full_manifest.set_index('variant_key').loc[pilot.variant_key]
    for column in ['split', 'component', 'gene_ids', 'variation_id', 'ALLELEID', 'locus_id', 'start', 'end']:
        if expected[column].tolist() != pilot[column].tolist():
            raise AssertionError(f'Pilot does not preserve full-cohort {column}')
    if list(sequences.columns) != DNA_COLUMNS or sequences.variant_key.tolist() != pilot.variant_key.tolist():
        raise AssertionError('Sequence inputs must contain only the ordered DNA allowlist')
    seen = {}
    for row, dna in zip(pilot.itertuples(), sequences.itertuples()):
        ref, alt = dna.ref_sequence, dna.alt_sequence
        if (len(ref) != CONTEXT or len(alt) != CONTEXT or set(ref + alt) - set(BASES)
                or ref[CONTEXT // 2] != row.ref or alt[CONTEXT // 2] != row.alt
                or [i for i in range(CONTEXT) if ref[i] != alt[i]] != [CONTEXT // 2]):
            raise AssertionError('Invalid reference/alternate pair')
        for sequence, column in [(ref, 'ref_context_hash'), (alt, 'alt_context_hash')]:
            key = context_hash(sequence)
            if key != getattr(row, column):
                raise AssertionError('Sequence hash differs from frozen manifest')
            if key in seen and seen[key] != row.split:
                raise AssertionError('Identical or reverse-complement context crosses splits')
            seen[key] = row.split
    checks['pilot: identical reference/alternate contexts including reverse complements disjoint'] = True
    checks['pilot: DNA allowlist, exact single substitutions, and full-cohort membership'] = True
    return checks



def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype={key: str for key in
                       ['chrom', 'gene_ids', 'variation_id', 'ALLELEID']})


def export_vcfs(pilot, source=None, expected_sha256=None):
    """Copy original headers and selected records, in source order, to two VCFs.

    Validate the full source checksum and exact membership before publishing.
    Clinical INFO fields remain available as outcomes, never predictor inputs.
    Existing exports must be byte-identical; mismatches are never overwritten.
    """
    source = CLINVAR_INPUT if source is None else Path(source)
    expected_sha256 = CLINVAR_SHA256 if expected_sha256 is None else expected_sha256
    if pilot.variant_key.duplicated().any() or set(pilot.split) != set(SPLITS):
        raise ValueError('VCF export requires unique variants and both splits')
    assignment = dict(zip(pilot.variant_key, pilot.split))
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
                    fields = raw.split(b'\t', 5)
                    if len(fields) < 6:
                        raise ValueError('Invalid VCF record')
                    key = 'GRCh38:' + b':'.join(fields[i] for i in [0, 1, 3, 4]).decode()
                    if key in assignment:
                        info = raw.decode().rstrip('\r\n').split('\t')[7]
                        consequence = next((item[3:] for item in info.split(';') if item.startswith('MC=')), '')
                        if not has_missense(consequence):
                            raise ValueError(f'Selected VCF variant lacks missense annotation: {key}')
                        if key in seen:
                            raise ValueError(f'Duplicate selected VCF variant: {key}')
                        seen.add(key)
                        outputs[assignment[key]].write(raw)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Source VCF checksum mismatch')
        if not header_seen or seen != set(assignment):
            raise ValueError('VCF export does not cover the exact split manifest')
        hashes = {VCF_FILES[split].name: digest_file(path) for split, path in temporary.items()}
        # Check both destinations before replacing either file.
        for path in VCF_FILES.values():
            if path.exists():
                verify_file(path, hashes[path.name])
        for split, path in VCF_FILES.items():
            if not path.exists():
                temporary[split].replace(path)
        return hashes
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def load_partition_labels(split, keys):
    """Read outcomes from the canonical VCF; align to the frozen DNA order."""
    if split not in SPLITS:
        raise ValueError('Unknown split; only train and validation are defined')
    protocol = verify_protocol()
    path = VCF_FILES[split]
    frame, _ = q0.read_clinvar(path, protocol['vcf_exports'][path.name], CLINVAR_DATE)
    if not frame.MC.astype('object').map(has_missense).all():
        raise ValueError('Pilot VCF must contain only missense variants')
    frame['variant_key'] = 'GRCh38:' + frame[['chrom', 'pos', 'ref', 'alt']].astype(str).agg(':'.join, axis=1)
    expected = read_csv(OUTPUT / 'split_manifest.csv')
    expected_keys = expected.loc[expected.split.eq(split), 'variant_key'].tolist()
    if frame.variant_key.duplicated().any() or set(frame.variant_key) != set(expected_keys) or list(keys) != expected_keys:
        raise ValueError('VCF membership or requested label order differs from the frozen split')
    labels = frame.set_index('variant_key').loc[keys, 'label'].map({'benign': 0, 'pathogenic': 1})
    if labels.isna().any() or set(labels) != {0, 1}:
        raise ValueError('Both unambiguous classes are required')
    return labels.to_numpy(dtype=int)



def protocol_config():
    return {'seed': SEED, 'sample_size_before_sequence_exclusions': N_VARIANTS,
            'context_bp': CONTEXT, 'assembly': 'GRCh38',
            'eligibility': 'Q0 filters plus exact MC SO:0001583; retain all gene associations',
            'sampling': 'lowest SHA256(42:sample:variant_key) among eligible missense variants, no labels',
            'full_cohort_assignment_sha256': FULL_ASSIGNMENTS_SHA256,
            'split': 'Preserve shared September assignments; new groups use SHA256(42:split:min_full_component_variant_key), 70/30, no labels',
            'previous_assignment_sha256': PREVIOUS_ASSIGNMENTS_SHA256,
            'inherited_component_splits': INHERITED_COMPONENT_SPLITS,
            'vcf_roles': {path.name: split for split, path in VCF_FILES.items()},
            'evaluation_scope': 'Development only: clinvar-test-pilot.vcf is validation; no separate untouched test set',
            'grouping': 'full Q0 SNV cohort including nonmissense bridges; genes, source IDs, loci, overlapping windows; pilot identical DNA incl RC',
            'clinvar_sha256': CLINVAR_SHA256, 'clinvar_date': CLINVAR_DATE,
            'clinvar_url': CLINVAR_URL, 'clinvar_archive_md5': CLINVAR_ARCHIVE_MD5,
            'q0_implementation_sha256': digest_file(q0.__file__),
            'reference_url': REFERENCE_URL, 'reference_md5': REFERENCE_MD5, 'reference_sha256': REFERENCE_SHA256}


def verify_protocol():
    protocol = read_json(OUTPUT / 'protocol.json')
    if protocol['config'] != protocol_config():
        raise RuntimeError('Frozen split configuration changed; do not reassign evaluated variants.')
    if protocol['implementation_sha256'] != digest_file(__file__):
        raise RuntimeError('Split implementation changed; preserve the recorded dataset provenance.')
    for name, digest in protocol['artifacts'].items():
        verify_file(OUTPUT / name, digest)
    if set(protocol['vcf_exports']) != {path.name for path in VCF_FILES.values()}:
        raise ValueError('Protocol must identify both canonical VCF exports')
    for path in VCF_FILES.values():
        verify_file(path, protocol['vcf_exports'][path.name])
    return protocol


def prepare():
    """Acquire/verify inputs, construct and freeze manifests, then audit them."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    q0.ensure_input(CLINVAR_INPUT, CLINVAR_ARCHIVE, url=CLINVAR_URL,
                    archive_md5=CLINVAR_ARCHIVE_MD5, expected_sha256=CLINVAR_SHA256)
    acquire_reference()
    config = protocol_config()
    started = time.monotonic()
    protocol_path = OUTPUT / 'protocol.json'
    if protocol_path.exists():
        protocol = verify_protocol()
    else:
        # Rebuild from the verified raw VCF, without trusting a mutable Q0 cache.
        print('Preparing the full Q0 cohort from the verified VCF…', flush=True)
        frame, provenance = q0.read_clinvar(CLINVAR_INPUT, CLINVAR_SHA256, CLINVAR_DATE)
        audit = q0.audit_data(frame)
        cohort, funnel = q0.propose_cohort(frame, audit)
        missense_count = int(cohort.MC.astype('object').map(has_missense).sum())
        funnel.loc[len(funnel)] = ['Exact MC SO:0001583 missense annotation', missense_count, len(cohort) - missense_count]
        columns = ['chrom', 'pos', 'ref', 'alt', 'variation_id', 'ALLELEID', 'gene_ids', 'MC']
        full = cohort[columns].copy().reset_index(drop=True)
        for name in ['chrom', 'variation_id', 'ALLELEID', 'gene_ids']:
            full[name] = full[name].astype(str)
        full['variant_key'] = 'GRCh38:' + full[['chrom', 'pos', 'ref', 'alt']].astype(str).agg(':'.join, axis=1)
        labels = dict(zip(full.variant_key, cohort.label.map({'benign': 0, 'pathogenic': 1}).astype(int)))
        del frame, cohort, audit
        full['locus_id'] = full.chrom + ':' + full.pos.astype(str)
        full['start'] = full.pos - 1 - CONTEXT // 2
        full['end'] = full.start + CONTEXT
        graph = full_components(full)
        candidates = full.loc[full.MC.astype('object').map(has_missense)].reset_index(drop=True)
        selected = candidates.iloc[sample_indices(candidates)].copy()
        dna, exclusions = extract_contexts(selected, REFERENCE)
        merge_identical_contexts(graph, full, dna)
        full['component'] = graph.names(full.variant_key)
        full['split'] = full.component.map(assign_split)
        pilot = full.set_index('variant_key').loc[dna.variant_key].reset_index()
        pilot['ref_context_hash'] = dna.ref_sequence.map(context_hash)
        pilot['alt_context_hash'] = dna.alt_sequence.map(context_hash)
        audit_splits(full, pilot, dna)
        audit_missense_scope(full, pilot)
        # A fixed split can fail feasibility; labels NEVER cause a new seed or assignment.
        for split in SPLITS:
            keys = pilot.loc[pilot.split.eq(split), 'variant_key']
            target = pd.DataFrame({'variant_key': keys, 'label': keys.map(labels)})
            if set(target.label) != {0, 1}:
                raise ValueError(f'{split} lacks both classes. Stop; do not search for a favorable split.')
            target.to_csv(OUTPUT / f'{split}_labels.csv', index=False)
        full.to_csv(OUTPUT / 'full_cohort_groups.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        pilot.to_csv(OUTPUT / 'split_manifest.csv', index=False)
        dna.to_csv(OUTPUT / 'sequences.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        exclusions.to_csv(OUTPUT / 'sequence_exclusions.csv', index=False)
        funnel.to_csv(OUTPUT / 'filter_counts.csv', index=False)
        files = ['full_cohort_groups.csv.gz', 'split_manifest.csv', 'sequences.csv.gz',
                 'sequence_exclusions.csv', 'filter_counts.csv'] + [f'{split}_labels.csv' for split in SPLITS]
        protocol = {'config': config, 'input': provenance,
                    'implementation_sha256': digest_file(__file__),
                    'artifacts': {name: digest_file(OUTPUT / name) for name in files},
                    'vcf_exports': export_vcfs(pilot)}
        write_json(protocol_path, protocol, frozen=True)
    full = read_csv(OUTPUT / 'full_cohort_groups.csv.gz')
    pilot = read_csv(OUTPUT / 'split_manifest.csv')
    dna = read_csv(OUTPUT / 'sequences.csv.gz')
    checks = audit_splits(full, pilot, dna)
    checks.update(audit_missense_scope(full, pilot))
    write_json(OUTPUT / 'leakage_checks.json', checks)
    write_json(OUTPUT / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'biopython', 'matplotlib']},
        'implementation_sha256': digest_file(__file__), 'seed': SEED,
        'seconds_this_run': time.monotonic() - started, 'compute': 'CPU only',
    })
    print(f'{len(pilot):,} sampled variants retained; {len(checks)} leakage check groups passed.')
    return pilot, dna


def settings():
    print('Q0 cohort → exact MC missense filter → 5,000-variant pilot → frozen training / validation groups')
    print('1,024-base windows · seed 42 · approximately 70% / 30% · CPU only')
    print('data/clinvar-train-pilot.vcf = training; data/clinvar-test-pilot.vcf = validation (no separate test stage).')
    q0.details('Pinned inputs and split rules', protocol_config())


def show_grouping():
    """Illustrate why an unsampled bridge must still connect sampled variants."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 3.1), layout='constrained')
    ax.set(xlim=(-.6, 6.9), ylim=(-1.1, 1.5))
    ax.axis('off')
    for x, name, genes, sampled in [(0, 'A', 'Gene 1', True), (1.8, 'B', 'Genes 1 + 2', False),
                                     (3.6, 'C', 'Gene 2', True)]:
        ax.scatter(x, .4, s=1900, color='#31688e' if sampled else '#dddddd', zorder=3)
        ax.text(x, .4, name, ha='center', va='center', color='white' if sampled else '#333333', fontsize=16)
        ax.text(x, -.12, genes, ha='center', fontsize=10)
    ax.plot([0, 3.6], [.4, .4], color='#31688e', lw=3, zorder=1)
    ax.text(1.8, -.55, 'B can be absent from the pilot; A and C still stay together.', ha='center', fontsize=10)
    ax.annotate('', xy=(5.15, .4), xytext=(4.05, .4), arrowprops={'arrowstyle': '->', 'lw': 2})
    ax.add_patch(FancyBboxPatch((5.2, -.05), 1.25, .9, boxstyle='round,pad=.2',
                               facecolor='#e4f2ec', edgecolor='#35b779'))
    ax.text(5.825, .4, 'One group\nOne split', ha='center', va='center', fontsize=12)
    ax.set_title('Illustration: group the full cohort before sampling', loc='left', fontweight='bold')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / 'grouping.png', dpi=160)
    plt.show()


def show_splits(pilot):
    """Display frozen split sizes and executable audit results."""
    protocol = verify_protocol()
    frozen = read_csv(OUTPUT / 'split_manifest.csv')
    if not pilot.equals(frozen):
        raise ValueError('Displayed pilot must match the frozen manifest')
    counts = pilot.groupby('split').agg(variants=('variant_key', 'size'), components=('component', 'nunique')).reindex(SPLITS)
    genes = {split: set('|'.join(pilot.loc[pilot.split.eq(split), 'gene_ids']).split('|')) for split in SPLITS}
    if any(genes[a] & genes[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise AssertionError('Gene overlap between splits')
    counts['genes'] = [len(genes[split]) for split in SPLITS]
    counts['percent'] = (counts.variants / len(pilot) * 100).round(2)
    counts['purpose'] = ['Fit scaling and classifiers', 'Choose settings and compare models']
    counts['vcf'] = [VCF_FILES[split].name for split in SPLITS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    colors = ['#31688e', '#e69f00']
    for ax, column, title in zip(axes, ['variants', 'genes'], ['Pilot variants', 'Distinct genes · no overlap']):
        bars = ax.bar(SPLITS, counts[column], color=colors)
        ax.bar_label(bars, fmt='{:,.0f}', padding=3)
        ax.set(title=title, ylabel='Count')
        ax.margins(y=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUTPUT / 'split_sizes.png', dpi=160)
    plt.show()
    counts.to_csv(OUTPUT / 'split_counts.csv')
    print('Whole groups are assigned by a fixed hash. Different group sizes make the proportions approximate.')
    q0.details('Counts, proportions and split purposes', counts)
    q0.details('Sequential eligibility filters and retained counts', read_csv(OUTPUT / 'filter_counts.csv'))
    q0.details('Passed checks and their scope', read_json(OUTPUT / 'leakage_checks.json'))
    q0.details('Pilot manifest preview · full CSV: results/q1/split_manifest.csv',
               pilot[['variant_key', 'gene_ids', 'start', 'end', 'component', 'split']].head(15))
    q0.details('Frozen split protocol', protocol)


def run_checks():
    import unittest
    from . import test_q1
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_q1)
    result = unittest.TextTestRunner(stream=stream).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(stream.getvalue())
    print(f'{result.testsRun} split and sequence integrity tests passed.')
