"""Archive previous Q14 attempts and execute a complete notebook in a fresh kernel."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import time

import nbformat

from . import q1, q14
from .refresh_q11 import LoggedClient, validate_execution


def build_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return nbformat.v4.new_notebook(cells=[
        md('# Q14. Can adapting two active mixer blocks improve Evo2 fine-tuning?\n\n'
           'Test additional adapter capacity by extending block-30 LoRA to **blocks 29 and 30**. '
           'Initialize from the strongest frozen classifier completed by [Q12](Q12-lora-improvement.ipynb) '
           'or [Q13](Q13-lora-optimization.ipynb), hold that classifier fixed, and retain the same '
           'magnitude-aware features and matched frozen controls. Two learning-rate arms test this hypothesis '
           'under a one-hour budget on the prepared H100.\n\n'
           'Use **Gamow** and run every cell in order. Prerequisites are completed Q11/Q12/Q13 artifacts, '
           'the checksum-pinned Savanna Evo2 7B base checkpoint and the prepared BioNeMo runtime. '
           '[Protocol and reporting](src/q14.py) · [Adapter placement](src/q14_adapters.py) · '
           '[GPU workflow](src/q14_backend.py) · [Setup](../README.md#setup).'),
        code('from src import q14'),
        md('## Fixed inputs and feature provenance\n\n'
           'Retain Q1’s frozen 6 July 2026 ClinVar missense partitions, all inherited eligibility and '
           'gene/component/sequence separation checks, and exactly Q12’s authorized **24,576 training '
           'variants** and uniform seed-42 **2,048 validation variants**. The full partitions contain '
           '46,888 and 17,927 variants respectively. Validation remains a development set.\n\n'
           'Reuse only checksum-verified Q12 frozen feature caches: 512-base canonical paired strand, '
           'final-block mean pooling, unit-RMS reference and allele-difference vectors, plus log '
           'difference RMS and log relative RMS (**8,194 features**). Retain the selected checkpoint’s '
           'training-only standardization. Freeze sources, parent results, cache manifest and exact memberships before '
           'training; validation labels never enter gradients or preprocessing. Preserve the exact '
           'implementation in each attempt’s source snapshot.'),
        code('protocol = q14.prepare()'),
        md('## Fix the strongest classifier and extend adapter placement\n\n'
           'Compare the completed **best frozen controls** from Q12 and Q13 by sampled AUROC, then AP; '
           'retain Q12 on an exact tie. Freeze the winning source, full checkpoint path and checksum '
           'before training. Load its **control head and mean/scale**, using Q12’s verified training '
           'class weights. Q14 does not refit the classifier or require a new convergence test. '
           'The parent’s selected adapter head is never substituted for its frozen control.\n\n'
           'Initialize both arms from that same head and the original frozen backbone. Attach '
           'distinct rank-8/alpha-16 adapters to **dense_projection** and **dense** in each of '
           '**blocks 29 and 30**: four target modules, Xavier A, zero B, dropout **0**. '
           'Verify unique adapter tensors, exact target coverage, zero-output equivalence and actual '
           'updates before interpreting scores. Keep the original backbone weights frozen.\n\n'
           'Try adapter peak learning rates **3e-5 and 1e-4**, with **32-step warmup** and cosine decay '
           'to **10%**. Head learning rate **0**, adapter weight decay **0.01**, seed **42**, '
           'batch **32**, at most **512 updates per arm**. Keep BF16 backbone/adapters and FP32 head '
           'and optimizer master weights.'),
        md('## Matched controls, stopping and integrity\n\n'
           'Both arms begin with the same fixed classifier and receive the same training examples '
           'and batch order. Optimize adapters with balanced binary cross entropy only; explicit '
           'head L2 strength is **0** because classifier weights remain fixed. Clip adapter gradient '
           'norms at **1.0**. Classifier parameters have gradients disabled and never enter an optimizer. '
           'The matched control applies the identical fixed classifier to original '
           'frozen features at every checkpoint. Record actual adapter changes and check that classifier '
           'weights and the scaler remain unchanged, and reproduce the inherited control scores before '
           'training, so any prediction improvement must come from the adapters. '
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
           'against the strongest frozen classifier, including Q12’s and Q13’s controls and the initial fitted '
           'head. Retain the best LoRA checkpoint together with the control at its exact update count '
           'for paired inference. This experiment tests adapter placement and learning rate without '
           'claiming that additional capacity necessarily helps.'),
        code('result = q14.run_experiment()'),
        md('## Sampled development results\n\n'
           'Report AUROC/AP with **1,000 whole-component bootstrap draws**, seed **42**, and the paired '
           'LoRA-minus-control effect. Intervals do not correct repeated validation, regularization, '
           'learning-rate or checkpoint selection. These results cannot establish performance on '
           'an untouched holdout and must stay separate from the full-cohort README benchmark.'),
        code('q14.show_results()'),
        md('## Conclusion\n\nThe conclusion is computed from the completed run and its frozen promotion rule.'),
        code('q14.show_conclusion()'),
    ], metadata={'kernelspec': {'display_name': 'Gamow', 'language': 'python', 'name': 'gamow'},
                 'language_info': {'name': 'python', 'version': '3.12'}})


def archive_attempt():
    """Preserve previous artifacts; a new notebook never short-circuits training."""
    retained = {'archive', 'notebook_history', 'notebook.lock', 'run.lock'}
    paths = [path for path in q14.OUTPUT.iterdir() if path.name not in retained]
    if not paths:
        return
    directory = q14.OUTPUT / 'archive' / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    directory.mkdir(parents=True)
    for path in paths:
        shutil.move(str(path), directory / path.name)
    print(f'Preserved previous attempt: {directory.relative_to(q14.ROOT)}', flush=True)


def refresh():
    q14.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (q14.OUTPUT / 'notebook.lock').open('a') as notebook_lock:
        fcntl.flock(notebook_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check before moving artifacts that a separately launched backend is idle.
        with (q14.OUTPUT / 'run.lock').open('a') as run_lock:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            archive_attempt()
        started = time.time()
        os.environ['Q14_RUN_STARTED_UTC'] = str(started)
        q1.write_json(q14.OUTPUT / 'run_status.json', {'status': 'running', 'started_utc': started})
        notebook = build_notebook()
        path = q14.ROOT / 'notebooks/Q14-layer-adapters.ipynb'
        try:
            LoggedClient(notebook, timeout=None, kernel_name='gamow', allow_errors=False,
                         resources={'metadata': {'path': str(q14.ROOT / 'notebooks')}}).execute()
            validate_execution(notebook)
            q14.verified_results()
        except BaseException as error:
            failed = q14.OUTPUT / f'failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb'
            nbformat.write(notebook, failed)
            lines = str(getattr(error, 'evalue', error)).splitlines()
            reason = next((line.strip() for line in reversed(lines) if line.strip()), type(error).__name__)
            q1.write_json(q14.OUTPUT / 'run_status.json', {
                'status': 'failed', 'reason': reason, 'notebook': str(failed.relative_to(q14.ROOT))})
            print(f'Actual failed execution preserved at {failed}; completed notebook not replaced.', flush=True)
            raise
        if path.exists():
            history = q14.OUTPUT / 'notebook_history'
            history.mkdir(exist_ok=True)
            shutil.copy2(path, history / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb')
        temporary = path.with_suffix('.ipynb.partial')
        nbformat.write(notebook, temporary)
        temporary.replace(path)
        elapsed = time.time() - started
        q1.write_json(q14.OUTPUT / 'run_status.json', {
            'status': 'complete', 'notebook_execution_seconds': elapsed,
            'within_one_hour': elapsed <= q14.CONFIG['total_seconds'],
            'notebook_sha256': q1.digest_file(path),
        })
        print(f'Saved fully executed {path}; fresh notebook elapsed {elapsed / 60:.1f} minutes.', flush=True)


if __name__ == '__main__':
    refresh()
