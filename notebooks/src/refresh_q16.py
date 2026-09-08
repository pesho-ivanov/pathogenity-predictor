"""Execute the larger Q16 continuation and full validation in a fresh Lida kernel."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import subprocess
import time

import nbformat
from jupyter_client.kernelspec import KernelSpecManager

from . import q1, q16, q16_report, q9_environment
from .refresh_q11 import LoggedClient, validate_execution


def build_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return nbformat.v4.new_notebook(cells=[
        md('# Q16. Does a larger continuation improve the successful Evo2 adapter?\n\n'
           'Continue the [Q14 winner](Q14-layer-adapters.ipynb) on **all 46,888 training variants**, '
           'for at most **three additional epochs**. This is a single registered learning-rate run. '
           'Its **four-hour cumulative budget includes preparation, training, full validation, '
           'checkpoint verification and reporting**. A measured validation reserve stops training '
           'early if needed. Historical model acquisition and Q12/Q14 fitting are prerequisites.\n\n'
           'Use **Lida** on the prepared H100 80 GB and run all cells in order. '
           '[Frozen protocol](src/q16.py) · [GPU workflow](src/q16_backend.py) · '
           '[Reporting](src/q16_report.py) · [Setup](../README.md#setup).'),
        code('from src import q16, q16_report'),
        md('## Freeze inputs and the cumulative deadline\n\n'
           'Reuse the checksum-pinned **6 July 2026 ClinVar, GRCh38, missense SNV** cohort. '
           'Keep all **46,888 training** and **17,927 validation** variants in their inherited '
           'gene/component/locus/sequence-disjoint partitions. Recheck eligibility, VCF checksums, '
           'labels and canonical 512-base allele-context separation. Validation is development '
           'data; `clinvar-test.vcf` does not denote an untouched test set.\n\n'
           'Retain Q14’s exact **2,048 validation keys** for checkpoint selection. Full validation '
           'does not select a checkpoint or refit parameters. Sources, parent checkpoints, exact '
           'memberships and the original request deadline are frozen before training. Resumption '
           'preserves that deadline. No validation labels enter gradients or preprocessing.'),
        code('protocol = q16.prepare()'),
        md('## Continue the adapter, then validate all three models\n\n'
           'Load Q14’s selected nonzero adapters in **blocks 29 and 30**, rank **8**, alpha **16**, '
           'dropout **0**. Keep the original 7B backbone, **8,195-parameter classifier** and '
           'training-fitted scaler fixed. The 8,194 features retain reference/difference direction '
           'and mutation magnitude. Only **393,216 adapter parameters** update.\n\n'
           'Warm restart **AdamW** with fresh optimizer state and FP32 masters initialized from '
           'the saved BF16 adapters. Use peak learning rate **3e-5**, **64-step warmup**, cosine '
           'decay to 10%, weight decay **0.01**, gradient norm limit **1**, seed **42**, and '
           'batch **32 variants**. Shuffle all training variants independently each epoch and '
           'retain the final eight-variant batch: at most **4,398 updates / 140,664 examples**. '
           'Balanced BCE weights are computed from the full training labels.\n\n'
           'Compute actual Q14 starting predictions, monitor every **512 updates** and at '
           'stopping, and save optimizer/master/RNG/cursor state every **32 updates**. Keep the '
           'best newly trained checkpoint by sampled AUROC, then AP. Retain it only if AUROC '
           'strictly improves over Q14 without an AP decrease; otherwise retain Q14. Preserve '
           'the best continuation even when it loses.\n\n'
           'Reserve at least **65 minutes plus reporting time**, adjusted upward by measured '
           'inference speed. Run fresh full inference for the best continuation, Q14 parent, '
           'and original frozen control regardless of the selection outcome. Destroy and reload '
           'learned tensors to verify 64 predictions, reproduce inherited predictions, and verify '
           'unchanged backbone, classifier and scaler.\n\n'
           'The first preflight stopped before training because **batch-one BF16 scoring** '
           'differed from batched scoring; its real failed notebook and sources are archived. '
           'The revised preflight measures that sensitivity and enforces the original **2e-4** '
           'tolerance on operational batch comparisons, permutations, repeatability and reverse '
           'complements. Single-variant numerical equivalence is not claimed.'),
        code('result = q16.run_experiment()'),
        md('## Full development results and uncertainty\n\n'
           'Report AUROC/AP with **1,000 whole-component bootstrap resamples**, seed **42**, '
           'and paired continuation-minus-Q14 and continuation-minus-frozen effects. Also show '
           'the **15,879 variants outside selection** and **3,246 variants in components absent '
           'from selection**. Neither subset is an untouched holdout, and intervals do not '
           'correct repeated model selection. Pretraining exposure, homology and shared-patient '
           'overlap remain unresolved. Preserve negative results and actual timing.'),
        code('q16_report.show_results()'),
        md('## Conclusion\n\n'
           'The conclusion below uses the complete predictions and states whether continuation '
           'improved on Q14. After this notebook finishes and is saved with all outputs, its '
           'verified result is added to the README beside the existing comparisons.'),
        code('q16_report.show_conclusion()'),
    ], metadata={'kernelspec': {'display_name': 'Lida', 'language': 'python', 'name': 'lida'},
                 'language_info': {'name': 'python', 'version': '3.12'}})


def ensure_kernel():
    manager = KernelSpecManager()
    if 'lida' not in manager.find_kernel_specs():
        subprocess.run([str(q9_environment.PYTHON), '-m', 'ipykernel', 'install', '--user',
                        '--name', 'lida', '--display-name', 'Lida'], check=True)
    spec = manager.get_kernel_spec('lida')
    q16.require(spec.argv[0] == str(q9_environment.PYTHON),
                'Lida must use the prepared BioNeMo Python environment')


def archive_attempt():
    kept = {'archive', 'notebook_history', 'notebook.lock', 'run.lock'}
    paths = [path for path in q16.OUTPUT.iterdir() if path.name not in kept]
    if paths:
        archive = q16.OUTPUT / 'archive' / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
        archive.mkdir(parents=True)
        for path in paths:
            shutil.move(str(path), archive / path.name)
        print(f'Preserved previous attempt: {archive}', flush=True)


def refresh(started_utc=None, deadline_utc=None, resume=False):
    q16.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (q16.OUTPUT / 'notebook.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (q16.OUTPUT / 'run.lock').open('a') as run_lock:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not resume:
                archive_attempt()
                start = time.time() if started_utc is None else float(started_utc)
                end = start + q16.CONFIG['total_seconds'] if deadline_utc is None else float(deadline_utc)
                q1.write_json(q16.OUTPUT / 'execution.json', {'started_utc': start, 'deadline_utc': end}, frozen=True)
            else:
                q16.require(started_utc is None and deadline_utc is None, 'Resume cannot reset the deadline')
                q16.require(not (q16.OUTPUT / 'metrics.json').exists(), 'Cannot resume a completed run')
                q16.verify_protocol()
        os.environ['Q16_RESUME'] = '1' if resume else '0'
        started = time.time()
        q1.write_json(q16.OUTPUT / 'run_status.json', {'status': 'running', 'question': 'q16',
            'notebook_started_utc': started, **q1.read_json(q16.OUTPUT / 'execution.json')})
        notebook = build_notebook()
        try:
            ensure_kernel()
            LoggedClient(notebook, timeout=None, kernel_name='lida', allow_errors=False,
                resources={'metadata': {'path': str(q16.ROOT / 'notebooks')}}).execute()
            validate_execution(notebook)
            q16.verified_results()
        except BaseException as error:
            failed = q16.OUTPUT / f'failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb'
            nbformat.write(notebook, failed)
            q1.write_json(q16.OUTPUT / 'run_status.json', {'status': 'failed', 'question': 'q16',
                'reason': str(getattr(error, 'evalue', error))[-3000:],
                'notebook': str(failed.relative_to(q16.ROOT)),
                'deadline_utc': q16.deadline()})
            print(f'Actual failed notebook preserved: {failed}', flush=True)
            raise
        if q16.NOTEBOOK.exists():
            history = q16.OUTPUT / 'notebook_history'
            history.mkdir(exist_ok=True)
            shutil.copy2(q16.NOTEBOOK, history / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb')
        temporary = q16.NOTEBOOK.with_suffix('.ipynb.partial')
        nbformat.write(notebook, temporary)
        temporary.replace(q16.NOTEBOOK)
        elapsed = time.time() - started
        execution = q1.read_json(q16.OUTPUT / 'execution.json')
        q1.write_json(q16.OUTPUT / 'run_status.json', {'status': 'complete', 'question': 'q16',
            'notebook_execution_seconds': elapsed, 'notebook_sha256': q1.digest_file(q16.NOTEBOOK),
            'request_elapsed_seconds': time.time() - execution['started_utc'],
            'within_four_hours': time.time() <= execution['deadline_utc'], **execution})
        try:
            q16_report.publish()
        except BaseException as error:
            q1.write_json(q16.OUTPUT / 'publication_status.json', {'status': 'failed',
                'reason': str(error), 'finished_utc': time.time(),
                'request_elapsed_seconds': time.time() - execution['started_utc']})
            raise
        finished = time.time()
        q1.write_json(q16.OUTPUT / 'publication_status.json', {'status': 'complete',
            'finished_utc': finished, 'request_elapsed_seconds': finished - execution['started_utc'],
            'within_four_hours': finished <= execution['deadline_utc'],
            'published_record_sha256': q1.digest_file(q16.ROOT / 'notebooks/results/comparison/published/q16-results.json')})
        print(f'Saved fully executed {q16.NOTEBOOK}; fresh notebook {elapsed / 60:.1f} minutes.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--started-utc', type=float)
    parser.add_argument('--deadline-utc', type=float)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    refresh(args.started_utc, args.deadline_utc, args.resume)
