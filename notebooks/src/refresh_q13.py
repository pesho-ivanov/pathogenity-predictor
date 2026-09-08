"""Archive previous Q13 attempts and execute a complete notebook in a fresh kernel."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import time

import nbformat

from . import q1, q13
from .refresh_q11 import LoggedClient, validate_execution


def build_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return nbformat.v4.new_notebook(cells=[
        md('# Q13. Can better head convergence and gentler updates make Evo2 LoRA useful?\n\n'
           'A one-hour experiment on the prepared H100: optimize a magnitude-aware linear classifier '
           'on verified frozen features, then compare two gentler LoRA settings against matched frozen '
           'controls. [Q12](Q12-lora-improvement.ipynb) found no adapter benefit; classifier improvement '
           'alone does not establish successful backbone adaptation.\n\n'
           'Use **Gamow** and run every cell in order. Prerequisites are completed Q11/Q12 artifacts, '
           'the checksum-pinned Savanna Evo2 7B base checkpoint and the prepared BioNeMo runtime. '
           '[Protocol and reporting](src/q13.py) · [Head optimization](src/q13_heads.py) · '
           '[GPU workflow](src/q13_backend.py) · [Setup](../README.md#setup).'),
        code('from src import q13'),
        md('## Fixed inputs and feature provenance\n\n'
           'Retain Q1’s frozen 6 July 2026 ClinVar missense partitions, all inherited eligibility and '
           'gene/component/sequence separation checks, and exactly Q12’s authorized **24,576 training '
           'variants** and uniform seed-42 **2,048 validation variants**. The full partitions contain '
           '46,888 and 17,927 variants respectively. Validation remains a development set.\n\n'
           'Reuse only checksum-verified Q12 frozen feature caches: 512-base canonical paired strand, '
           'final-block mean pooling, unit-RMS reference and allele-difference vectors, plus log '
           'difference RMS and log relative RMS (**8,194 features**). Retain Q12 training-only '
           'standardization. Freeze sources, parent results, cache manifest and exact memberships before '
           'fitting; validation labels never enter gradients or preprocessing.'),
        code('protocol = q13.prepare()'),
        md('## Fit the head before adapting the backbone\n\n'
           'Optimize a linear classifier using balanced training-subset binary cross entropy plus '
           'explicit L2 weight penalty (unpenalized intercept), with strengths **1e-4, 1e-3 and 1e-2**. '
           'Fit in FP64 for **1,000 iterations** followed by one continuation of up to **2,000 iterations** '
           '(**3,000 total**), requiring fitted full gradient norm **at most 1e-6**. Verify FP32 deployment '
           'logits and report the separate deployed gradient residual for every candidate. Exclude unconverged candidates; '
           'select by sampled AUROC, then AP, then stronger L2. This fitting stage runs again in a fresh notebook; the '
           'reused feature cache is a verified experimental input.\n\n'
           'Initialize both adapter arms from the same selected fitted head, the original frozen '
           'backbone and zero-output rank-8/alpha-16 LoRA in the same block-30 projections as Q12. '
           'Try adapter peak learning rates **3e-5 and 1e-5**, with **32-step warmup** and cosine decay '
           'to **10%**. Head learning rate **1e-5**, adapter weight decay **0.01**, seed **42**, '
           'batch **32**, at most **512 updates per arm**. Keep BF16 backbone/adapters and FP32 head '
           'and optimizer master weights.'),
        md('## Matched controls, stopping and integrity\n\n'
           'Each arm has a cloned frozen-backbone head receiving the same examples, batch order, '
           'update count, class weights, explicit head L2 penalty and head schedule. Clip adapter '
           'and head gradient norms separately at **1.0**, using identical head clipping in the control. '
           'Record separate gradient norms and actual adapter/head weight and prediction changes. '
           'Retain **step 0**, then monitor after **256** and **512** updates, or when the deadline '
           'requires final validation and reporting. Save optimizer/master/RNG/cursor state every '
           '**32 updates**, with the original run deadline recorded. These states are retained for '
           'inspection; automatic checkpoint resumption is not implemented. A fresh runner archives '
           'the preceding attempt and starts a new experiment from zero-output adapters.\n\n'
           'Use one GPU process. Reserve a sampled evaluation, a **64-variant** checkpoint reload '
           'check and three minutes for reporting under the one-hour budget; interrupted and completed '
           'attempts remain inspectable. Verify finite updates and unchanged original backbone tensors. '
           'Erase learned tensors and reload the selected checkpoint to reproduce predictions.\n\n'
           'Choose LoRA only when it improves AUROC by **at least 0.005** without decreasing AP '
           'against the strongest frozen classifier, including Q12’s control and the initial fitted '
           'head. Retain the best LoRA checkpoint together with the control at its exact update count '
           'for paired inference. A gain from classifier fitting is reported separately.'),
        code('result = q13.run_experiment()'),
        md('## Sampled development results\n\n'
           'Report AUROC/AP with **1,000 whole-component bootstrap draws**, seed **42**, and the paired '
           'LoRA-minus-control effect. Intervals do not correct repeated validation, regularization, '
           'learning-rate or checkpoint selection. These results cannot establish performance on '
           'an untouched holdout and must stay separate from the full-cohort README benchmark.'),
        code('q13.show_results()'),
        md('## Conclusion\n\nThe conclusion is computed from the completed run and its frozen promotion rule.'),
        code('q13.show_conclusion()'),
    ], metadata={'kernelspec': {'display_name': 'Gamow', 'language': 'python', 'name': 'gamow'},
                 'language_info': {'name': 'python', 'version': '3.12'}})


def archive_attempt():
    """Preserve previous artifacts; a new notebook never short-circuits training."""
    retained = {'archive', 'notebook_history', 'notebook.lock', 'run.lock'}
    paths = [path for path in q13.OUTPUT.iterdir() if path.name not in retained]
    if not paths:
        return
    directory = q13.OUTPUT / 'archive' / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    directory.mkdir(parents=True)
    for path in paths:
        shutil.move(str(path), directory / path.name)
    print(f'Preserved previous attempt: {directory.relative_to(q13.ROOT)}', flush=True)


def refresh():
    q13.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (q13.OUTPUT / 'notebook.lock').open('a') as notebook_lock:
        fcntl.flock(notebook_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check before moving artifacts that a separately launched backend is idle.
        with (q13.OUTPUT / 'run.lock').open('a') as run_lock:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            archive_attempt()
        started = time.time()
        os.environ['Q13_RUN_STARTED_UTC'] = str(started)
        q1.write_json(q13.OUTPUT / 'run_status.json', {'status': 'running', 'started_utc': started})
        notebook = build_notebook()
        path = q13.ROOT / 'notebooks/Q13-lora-optimization.ipynb'
        try:
            LoggedClient(notebook, timeout=None, kernel_name='gamow', allow_errors=False,
                         resources={'metadata': {'path': str(q13.ROOT / 'notebooks')}}).execute()
            validate_execution(notebook)
            q13.verified_results()
        except BaseException as error:
            failed = q13.OUTPUT / f'failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb'
            nbformat.write(notebook, failed)
            lines = str(getattr(error, 'evalue', error)).splitlines()
            reason = next((line.strip() for line in reversed(lines) if line.strip()), type(error).__name__)
            q1.write_json(q13.OUTPUT / 'run_status.json', {
                'status': 'failed', 'reason': reason, 'notebook': str(failed.relative_to(q13.ROOT))})
            print(f'Actual failed execution preserved at {failed}; completed notebook not replaced.', flush=True)
            raise
        if path.exists():
            history = q13.OUTPUT / 'notebook_history'
            history.mkdir(exist_ok=True)
            shutil.copy2(path, history / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb')
        temporary = path.with_suffix('.ipynb.partial')
        nbformat.write(notebook, temporary)
        temporary.replace(path)
        elapsed = time.time() - started
        q1.write_json(q13.OUTPUT / 'run_status.json', {
            'status': 'complete', 'notebook_execution_seconds': elapsed,
            'within_one_hour': elapsed <= q13.CONFIG['total_seconds'],
            'notebook_sha256': q1.digest_file(path),
        })
        print(f'Saved fully executed {path}; fresh notebook elapsed {elapsed / 60:.1f} minutes.', flush=True)


if __name__ == '__main__':
    refresh()
