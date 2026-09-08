"""Archive previous Q15 attempts and execute a complete notebook in a fresh kernel."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import time

import nbformat

from . import q1, q15
from .refresh_q11 import LoggedClient, validate_execution


def build_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return nbformat.v4.new_notebook(cells=[
        md('# Q15. Can cached-prefix replay and ranking loss improve Evo2 adapters?\n\nCache the frozen backbone prefix so two adapter objectives can revisit the same training examples within a one-hour budget. Compare **weighted binary cross entropy** with **within-batch pairwise logistic ranking loss**, using a fixed inherited classifier. The completed [Q14 experiment](Q14-layer-adapters.ipynb) determines the starting frozen control and adapter learning rate before Q15 begins.\n\nUse **Gamow** and run every cell in order. Prerequisites: completed Q11–Q14 artifacts, the pinned Savanna Evo2 7B base checkpoint and prepared H100/BioNeMo runtime. [Protocol](src/q15.py) · [Replay](src/q15_replay.py) · [Pooling](src/q15_pooling.py) · [Objectives](src/q15_objective.py) · [GPU workflow](src/q15_backend.py) · [Setup](../README.md#setup).'),
        code('from src import q15'),
        md('## Frozen inputs and provenance\n\nRetain Q1’s July 2026 missense partitions and inherited eligibility, gene/component and sequence-separation checks. Use exactly the previously authorized **24,576 of 46,888 training variants** and seed-42 **2,048 of 17,927 validation variants**. Freeze memberships, parent identities, source snapshots, cache provenance and initialization checkpoint before training.\n\nKeep 512-base canonical paired strands and the same **8,194** Q12 features: normalized reference and difference vectors plus log difference/relative magnitude. Reuse verified Q12 frozen features for the control and training-only preprocessing. Labels never enter feature construction or validation gradients.'),
        code('protocol = q15.prepare()'),
        md('## Fixed initialization and paired objectives\n\nChoose the strongest frozen control recorded by Q14, including inherited Q12/Q13 controls, by AUROC then AP; prefer Q14’s own control on exact ties. Freeze the exact control-head checkpoint and its scaler. Read and freeze the **actual peak learning rate** from Q14’s best adapter trial history. Both Q15 arms start with this same fixed classifier and fresh **rank-8/alpha-16** adapters in **blocks 29 and 30** (dense_projection and dense; Xavier A, zero B, dropout 0). Parent adapters are not reused.\n\nThe BCE arm uses verified Q12 training-class weights. The ranking arm minimizes mean logistic loss over all positive–negative pairs **within each batch**. A one-class ranking batch skips the optimizer and is explicitly recorded. Both arms receive the same attempted batches, seed **42**, batch **32**, **64-step warmup**, cosine decay to **10%**, adapter decay **0.01**, and adapter-gradient clipping **1.0**. Classifier learning rate is **0**; classifier parameters never enter an optimizer, and the inherited scaler stays fixed.'),
        md('## Cached replay, matched progress and integrity\n\nCache BF16 activations before the first adapted block on the CPU. Keep the complete **2,048-variant validation cache (about 16 GiB)** and one **4,096-variant training chunk (about 32 GiB)** at a time; sizes exclude model and transfer overhead. Replay the trainable tail for **three passes per chunk**, then replace the chunk. The maximum is **2,304 attempted batches per arm**, or **73,728 example exposures** across **24,576 unique variants**. Report actual unique variants, repeated exposures, optimizer updates and ranking skips separately.\n\nBefore training, compare replay against full GPU forward features and adapter gradients at zero and nonzero adapter weights. Fail on disagreement. Calibrate prefix extraction, paired tail updates and cached validation. Check the deadline before cache allocation/extraction and paired training attempts; reserve both final sample evaluations, fresh full-forward **64-variant** reload checks and reporting. Model loading, cache construction and transfers count within the **one-hour** budget. Stop both arms at the same attempted cursor and report partial coverage when needed.\n\nMonitor both objectives and the fixed control initially, after each training chunk (**384 attempted batches**), and after a final partial chunk. Preserve optimizer/master/RNG and cursor states for inspection; the fresh runner archives preceding attempts without automatic resumption. Verify unchanged original backbone weights, classifier and scaler, then erase and reload learned tensors and reproduce predictions using full forwards. Promote only an adapter with **at least 0.005 AUROC gain** and **no AP decrease** against the strongest freshly reproduced or inherited frozen control. Head-only gains do not count.'),
        code('result = q15.run_experiment()'),
        md('## Sampled development results\n\nReport BCE and ranking results against the fixed control with **1,000 whole-component bootstrap draws**, seed **42**, including the paired best-adapter effect. These intervals do not correct repeated validation, objective or checkpoint selection. No untouched test stage exists, and sampled results stay separate from the full-cohort benchmark.'),
        code('q15.show_results()'),
        md('## Conclusion\n\nComputed from the completed run, actual coverage and frozen promotion rule. Full validation and benchmark export are not part of this exploratory runner.'),
        code('q15.show_conclusion()'),
    ], metadata={'kernelspec': {'display_name': 'Gamow', 'language': 'python', 'name': 'gamow'},
                 'language_info': {'name': 'python', 'version': '3.12'}})


def archive_attempt():
    """Preserve previous artifacts; a new notebook never short-circuits training."""
    retained = {'archive', 'notebook_history', 'notebook.lock', 'run.lock'}
    paths = [path for path in q15.OUTPUT.iterdir() if path.name not in retained]
    if not paths:
        return
    directory = q15.OUTPUT / 'archive' / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    directory.mkdir(parents=True)
    for path in paths:
        shutil.move(str(path), directory / path.name)
    print(f'Preserved previous attempt: {directory.relative_to(q15.ROOT)}', flush=True)


def refresh():
    q15.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (q15.OUTPUT / 'notebook.lock').open('a') as notebook_lock:
        fcntl.flock(notebook_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check before moving artifacts that a separately launched backend is idle.
        with (q15.OUTPUT / 'run.lock').open('a') as run_lock:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            archive_attempt()
        started = time.time()
        os.environ['Q15_RUN_STARTED_UTC'] = str(started)
        q1.write_json(q15.OUTPUT / 'run_status.json', {'status': 'running', 'started_utc': started})
        notebook = build_notebook()
        path = q15.ROOT / 'notebooks/Q15-prefix-ranking.ipynb'
        try:
            LoggedClient(notebook, timeout=None, kernel_name='gamow', allow_errors=False,
                         resources={'metadata': {'path': str(q15.ROOT / 'notebooks')}}).execute()
            validate_execution(notebook)
            q15.verified_results()
        except BaseException as error:
            failed = q15.OUTPUT / f'failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb'
            nbformat.write(notebook, failed)
            lines = str(getattr(error, 'evalue', error)).splitlines()
            reason = next((line.strip() for line in reversed(lines) if line.strip()), type(error).__name__)
            q1.write_json(q15.OUTPUT / 'run_status.json', {
                'status': 'failed', 'reason': reason, 'notebook': str(failed.relative_to(q15.ROOT))})
            print(f'Actual failed execution preserved at {failed}; completed notebook not replaced.', flush=True)
            raise
        if path.exists():
            history = q15.OUTPUT / 'notebook_history'
            history.mkdir(exist_ok=True)
            shutil.copy2(path, history / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb')
        temporary = path.with_suffix('.ipynb.partial')
        nbformat.write(notebook, temporary)
        temporary.replace(path)
        elapsed = time.time() - started
        q1.write_json(q15.OUTPUT / 'run_status.json', {
            'status': 'complete', 'notebook_execution_seconds': elapsed,
            'within_one_hour': elapsed <= q15.CONFIG['total_seconds'],
            'notebook_sha256': q1.digest_file(path),
        })
        print(f'Saved fully executed {path}; fresh notebook elapsed {elapsed / 60:.1f} minutes.', flush=True)


if __name__ == '__main__':
    refresh()
