"""Execute full development validation of a promoted Q13/Q14/Q15 LoRA from a fresh kernel."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import time

import nbformat

from . import q1, lora_validation as validation
from .refresh_q11 import LoggedClient, validate_execution


def build_notebook(question):
    validation.parent(question)
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    exploration = {'q13': 'Q13-lora-optimization.ipynb', 'q14': 'Q14-layer-adapters.ipynb',
                   'q15': 'Q15-prefix-ranking.ipynb'}[question]
    return nbformat.v4.new_notebook(cells=[
        md(f'# {question.upper()}. Does the selected LoRA improvement hold across full validation?\n\n'
           f'Freeze the adapter promoted by [{question.upper()} exploration]({exploration}) and evaluate '
           'all **17,927** July missense validation variants against its matched control and the '
           'strongest frozen control retained during exploration. No training, refitting or new '
           'checkpoint selection occurs here. This separate validation run is outside the '
           'exploratory one-hour budget.\n\n'
           'Use **Gamow**, the prepared H100/BioNeMo runtime, and completed parent artifacts. '
           '[Protocol and reporting](src/lora_validation.py) · [GPU workflow](src/lora_validation_backend.py) · '
           '[Setup](../README.md#setup).'),
        code('from src import lora_validation'),
        md('## Freeze the selected models, full cohort and confirmation rule\n\n'
           'Require a completed parent experiment that promoted LoRA by its declared rule. Verify '
           'its source identity, checkpoint and result hashes. Recheck Q1’s frozen full missense '
           'partitions, labels and inherited gene/component/sequence separation. Preserve the exact '
           'sources, validation order, original selection keys and selection components before scoring.\n\n'
           'Confirmation requires all of: full-cohort LoRA **AUROC gain at least 0.005**, **no AP '
           'decrease**, and a paired AUROC **95% interval lower bound above zero** against the '
           'strongest frozen control; plus **positive AUROC gain and no AP decrease outside the '
           '2,048 selection variants**. Report a failure of this rule as a negative confirmation result.'),
        code(f"protocol = lora_validation.prepare('{question}')"),
        md('## Complete inference and checkpoint integrity\n\n'
           'Score every frozen validation variant with the selected adapter, its control from the '
           'same training update count, and the strongest retained frozen classifier. Keep model, '
           'features, standardization and classifier parameters fixed. Validate exact prediction '
           'membership and labels. Erase and reload learned tensors to reproduce **64 predictions** '
           'and verify unchanged original backbone weights.\n\n'
           'Use actual inference in a fresh kernel. Previous complete and failed validation attempts '
           'are archived, including source snapshots and real failed notebooks; the runner never '
           'uses saved result files as a substitute for this execution.'),
        code(f"result = lora_validation.run_validation('{question}')"),
        md('## Full and prespecified subset results\n\n'
           'Report AUROC/AP and paired LoRA-minus-strongest-control effects using **1,000 '
           'whole-component bootstrap resamples**, seed **42**. Repeat reporting for variants '
           'outside the selection sample and, separately, for variants belonging to components '
           'absent from that sample. The latter is descriptive and does not change the confirmation rule.\n\n'
           'These are **development results**. The sample guided repeated model selection, and '
           'the repository has evaluated the broader validation cohort before. Neither subset '
           'is an untouched final test set; intervals do not remove selection bias.'),
        code(f"lora_validation.show_results('{question}')"),
        md('## Conclusion\n\nComputed from complete saved predictions and the frozen confirmation rule. '
           'After this fresh notebook is saved, the runner exports full-cohort LoRA and strongest-control '
           'scores and refreshes the README comparison. A negative confirmation result remains visible.'),
        code(f"lora_validation.show_conclusion('{question}')"),
    ], metadata={'kernelspec': {'display_name': 'Gamow', 'language': 'python', 'name': 'gamow'},
                 'language_info': {'name': 'python', 'version': '3.12'}})


def archive_attempt(question):
    directory = validation.output(question)
    kept = {'archive', 'notebook_history', 'notebook.lock', 'run.lock'}
    paths = [path for path in directory.iterdir() if path.name not in kept]
    if not paths:
        return
    target = directory / 'archive' / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    target.mkdir(parents=True)
    for path in paths:
        shutil.move(str(path), target / path.name)
    print(f'Preserved preceding full-validation attempt: {target.relative_to(validation.ROOT)}', flush=True)


def refresh(question):
    directory = validation.output(question)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'notebook.lock').open('a') as notebook_lock:
        fcntl.flock(notebook_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / 'run.lock').open('a') as run_lock:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            archive_attempt(question)
        started = time.time()
        os.environ['LORA_VALIDATION_STARTED_UTC'] = str(started)
        q1.write_json(directory / 'run_status.json', {'status': 'running', 'started_utc': started,
                                                     'question': question})
        notebook = build_notebook(question)
        path = validation.ROOT / f'notebooks/{question.upper()}-lora-validation.ipynb'
        try:
            LoggedClient(notebook, timeout=None, kernel_name='gamow', allow_errors=False,
                         resources={'metadata': {'path': str(validation.ROOT / 'notebooks')}}).execute()
            validate_execution(notebook)
            validation.verified_results(question)
        except BaseException as error:
            failed = directory / f'failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb'
            nbformat.write(notebook, failed)
            lines = str(getattr(error, 'evalue', error)).splitlines()
            reason = next((line.strip() for line in reversed(lines) if line.strip()), type(error).__name__)
            q1.write_json(directory / 'run_status.json', {'status': 'failed', 'reason': reason,
                'question': question, 'notebook': str(failed.relative_to(validation.ROOT))})
            print(f'Actual failed validation preserved at {failed}; completed notebook not replaced.', flush=True)
            raise
        if path.exists():
            history = directory / 'notebook_history'
            history.mkdir(exist_ok=True)
            shutil.copy2(path, history / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb')
        temporary = path.with_suffix('.ipynb.partial')
        nbformat.write(notebook, temporary)
        temporary.replace(path)
        elapsed = time.time() - started
        q1.write_json(directory / 'run_status.json', {'status': 'complete', 'question': question,
            'notebook_execution_seconds': elapsed, 'notebook_sha256': q1.digest_file(path)})
        validation.export_full_comparison(question)
        from . import comparison
        comparison.publish_full_lora(question)
        comparison.refresh()
        print(f'Saved fully executed {path}; separate validation elapsed {elapsed / 60:.1f} minutes.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', choices=['q13', 'q14', 'q15'], required=True)
    refresh(parser.parse_args().question)
