"""Generate Q11, execute with a fresh kernel and publish only after success.

Run: .venv/bin/python -m notebooks.src.refresh_q11
Failures preserve the actual partial execution under results/q11/diagnostics/.
"""

from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


class LoggedClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        value = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return value


def build_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return nbformat.v4.new_notebook(cells=[
        md("# Q11. How well does a one-hour Evo2 7B LoRA run predict missense pathogenicity?\n\nTrain **rank-8 LoRA in the final Hyena mixer (block 30)** and a linear classifier under a one-hour budget on the prepared H100. Keep every original backbone weight frozen. This user-authorized **partial epoch** uses a seeded shuffle of Q1's complete training pool and evaluates its complete frozen validation partition.\n\n[Workflow](src/q11.py) · [GPU implementation](src/q11_backend.py) · [Dependencies](../requirements.txt) · [Setup](../README.md#setup).\n\nUse **Gamow** and run all cells in order. Inputs, pinned model and BioNeMo environment are prepared automatically when missing; initial downloads/installation are outside the one-hour target. Expected hardware: one 80 GB H100, about 128 GB host RAM. Artifacts stay in `results/q11/`; external inputs stay in `data/`."),
        code('from src import q11'),
        md('## Frozen inputs and shorter-context audits\n\nKeep all **46,888 training and 17,927 validation variants** from the frozen 6 July 2026 ClinVar missense cohort. `clinvar-test.vcf` is development validation, without an untouched test stage. Verify source/VCF checksums, labels, inherited assignments, genes, loci, components and original sequence separation before fitting.\n\nEach GRCh38 allele begins with its frozen 1,024-base context. Select the lexicographically smaller forward/reverse-complement **pair**, then crop **512 bases around the substitution**, with 256 bases before it. This gives one reproducible, strand-invariant orientation for both alleles. Check every reference and alternate crop for identical/reverse-complement sequences across splits. Crops remain inside the audited genomic intervals; stop on overlap.\n\nPool final-block activations over positions, concatenate reference and alternate-minus-reference vectors, and normalize each half by its own per-example RMS (epsilon 1e-6). The resulting 8,192 features contain DNA only; normalization has no learned statistics.'),
        code('inputs = q11.prepare_inputs()'),
        md("## Pinned model and budget\n\nUse Q10's pinned `arcinstitute/savanna_evo2_7b_base` checkpoint and BioNeMo conversion. [Arc's training guidance](https://github.com/ArcInstitute/evo2#training-and-finetuning) and the [pinned BioNeMo tutorial](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/ca16c2acf9bf813d020b6d1e2d4e1240cfef6a69/docs/docs/user-guide/examples/bionemo-evo2/fine-tuning-tutorial.ipynb) document the backend.\n\nAttach pinned NeMo LoRA wrappers only to `decoder.layers.30.mixer.dense_projection` and `decoder.layers.30.mixer.dense`: rank **8**, alpha **16**, dropout **0**, Xavier A and zero B. All original tensors, including attention block 31, remain frozen. Earlier archived probes found block 31 numerically inactive in this BF16 configuration; the new adapter checks must pass independently.\n\nSeed **42**; batches of **32 variants**; at most **768 updates / 24,576 variants**. Stop earlier at an optimizer boundary after **30 minutes** of training or when needed to reserve full validation and reporting time. Estimate validation duration from training-only throughput probes, add a 30% margin and four minutes for reporting. No baseline feature extraction or hyperparameter search.\n\nUse BF16 backbone/adapters, FP32 pooling/head, FP32 AdamW master weights and accumulation, learning rates **1e-4**, weight decay **0.01**, gradient clipping **1.0**, and training-pool balanced class weights. Initialize the head with seeded PyTorch defaults and train it jointly with the adapters."),
        code('protocol = q11.establish_environment()'),
        md("## Preflight, partial epoch and full validation\n\nTraining-only probes verify source-to-conversion tensors, A/B/A repeatability, strand invariance, batched feature equivalence, zero-adapter equivalence, finite gradients, actual adapter/head updates and unchanged original backbone weights. Probe updates are restored. When B starts at zero, A's first gradient is zero; all adapter matrices must receive nonzero gradients across later probe steps.\n\nSave atomic, checksummed resume checkpoints every 32 updates and at the final optimizer boundary. They preserve adapter/head weights, exact configuration, FP32 masters, Adam moments, RNG state and the deterministic shuffle cursor. Record actual training keys and epoch fraction. Resume never resets cumulative training compute. Changed sources/settings invalidate reuse; the earlier full-epoch workflow and computed features are archived.\n\nPredict all **17,927 validation variants once**. Then overwrite learned tensors, reload the final checkpoint and verify the first **64 validation predictions**. Validation labels never enter optimization. A failure produces a genuine diagnostic notebook, without publishing an unfinished result."),
        code('result = q11.run_experiment()'),
        md('## Development validation\n\nReport AUROC, average precision, and 95% percentile intervals from **1,000 whole-component bootstrap resamples, seed 42**. Show training coverage, stopping reason, runtime and memory. These intervals do not correct development-selection bias. This experiment does not fit a frozen baseline and cannot establish that LoRA improves one.\n\nPretraining sequence exposure, homology and shared-patient overlap remain unresolved. This is a research prototype, without clinical validation.'),
        code('q11.show_results()'),
        code('q11.export_comparison()'),
        md('## Conclusion\n\nComputed from the completed partial-epoch run and full validation.'),
        code('q11.show_conclusion()'),
    ], metadata={'kernelspec': {'display_name': 'Gamow', 'language': 'python', 'name': 'gamow'},
                 'language_info': {'name': 'python', 'version': '3.12'}})


def validate_execution(notebook):
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    if (not cells or any(not c.source.strip() for c in cells)
            or [c.execution_count for c in cells] != list(range(1, len(cells) + 1))
            or any(o.output_type == 'error' for c in cells for o in c.outputs)):
        raise RuntimeError('Q11 must finish every code cell in order with outputs retained')


def refresh():
    from . import comparison, q11
    import os, time
    started = time.time()
    os.environ['Q11_RUN_STARTED_UTC'] = str(started)
    q11.OUTPUT.mkdir(parents=True, exist_ok=True)
    q11.q1.write_json(q11.OUTPUT / 'run_status.json', {'status': 'running', 'started_utc': started})
    notebook = build_notebook()
    path = ROOT / 'notebooks/Q11-evo2-lora.ipynb'
    client = LoggedClient(notebook, timeout=None, kernel_name='gamow', allow_errors=False,
                          resources={'metadata': {'path': str(ROOT / 'notebooks')}})
    try:
        client.execute()
        validate_execution(notebook)
        q11.verified_results()
    except BaseException as error:
        directory = q11.OUTPUT / 'diagnostics'
        directory.mkdir(parents=True, exist_ok=True)
        failed = directory / f'execution-failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb'
        nbformat.write(notebook, failed)
        # The full exception stays in the diagnostic notebook; the README gets
        # its final concrete error, without a copied ANSI traceback.
        lines = str(getattr(error, 'evalue', error)).splitlines()
        reason = next((line.strip() for line in reversed(lines) if line.strip()), type(error).__name__)
        q11.q1.write_json(q11.OUTPUT / 'run_status.json', {
            'status': 'blocked', 'reason': reason, 'diagnostic': str(failed.relative_to(ROOT))})
        print(f'Execution failed; actual cells/errors preserved at {failed}. No completed notebook published.', flush=True)
        raise
    if path.exists():
        archive = q11.OUTPUT / 'notebook_history'
        archive.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(path, archive / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb')
    temporary = path.with_suffix('.ipynb.partial')
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    comparison.refresh()
    elapsed = time.time() - started
    q11.q1.write_json(q11.OUTPUT / 'run_status.json', {'status': 'complete', 'wall_seconds': elapsed,
        'within_one_hour': elapsed <= q11.CONFIG['total_seconds']})
    print(f'Saved fully executed notebook: {path}', flush=True)


if __name__ == '__main__':
    refresh()
