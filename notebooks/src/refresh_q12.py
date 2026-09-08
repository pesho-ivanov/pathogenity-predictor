"""Execute Q12 from a fresh kernel; full validation is an explicit separate mode."""

from datetime import datetime, timezone
import os
import shutil
import time

import nbformat

from . import q1, q12
from .refresh_q11 import LoggedClient, validate_execution


def build_notebook(full=False):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    if full:
        cells=[md('# Q12. How does the selected model perform on complete validation?\n\n'
            'Evaluate the model selected by [Q12 exploration](Q12-lora-improvement.ipynb) on all 17,927 frozen July '
            'missense validation variants. This is a separate run outside the exploratory one-hour budget. '
            'The validation sample already participated in model selection; this is development evaluation, without an untouched test. '
            '[Implementation](src/q12.py) · [GPU workflow](src/q12_backend.py).'),
            code('from src import q12'),code('result = q12.run_full_validation()'),
            code('q12.show_results(full=True)'),code('q12.show_conclusion(full=True)')]
    else:
        cells=[md('# Q12. Can better features and head fitting improve Evo2 LoRA?\n\n'
            'A one-hour exploratory experiment on the prepared H100. Preserve variant-effect magnitude, fit the classifier '
            'before LoRA, and compare continued LoRA training against a matched frozen-backbone classifier. '
            'Prerequisites are the completed [Q11 model](Q11-evo2-lora.ipynb), '
            '[Q11 diagnostic sample](Q11-lora-diagnostics.ipynb), and their verified inputs/runtime. '
            '[Implementation](src/q12.py) · [GPU workflow](src/q12_backend.py) · [Setup](../README.md#setup).'),
            code('from src import q12'),
            md('## Frozen inputs and authorized exploratory scope\n\n'
               'Retain Q1’s complete July missense partitions and all original leakage checks. Use exactly the same '
               '**24,576 of 46,888 training variants** seen by Q11. Monitoring uses the existing uniform seed-42 '
               '**2,048 of 17,927 validation variants**, across 823 Q1 components. Freeze membership and source hashes '
               'before fitting. These subsets are explicitly authorized for exploration; no samples change the full benchmark. '
               'Only paired DNA sequences enter feature construction.'),
            code('protocol = q12.prepare()'),
            md('## Features, classifier fitting and matched continuation\n\n'
               'Keep the original pinned 7B backbone, 512-base canonical paired strand and final-block mean pooling. '
               'Cache frozen features once. Compare three heads: (1) normalized reference plus allele difference; '
               '(2) those features plus log difference RMS and log relative difference RMS; (3) normalized difference '
               'plus the two log magnitudes, omitting reference. RMS floor **1e-12**; fit feature means/stds only on '
               'training rows (std floor **1e-6**) and freeze them during adaptation.\n\n'
               'Each FP32 linear head gets **20 epochs**, batch **512**, AdamW **1e-3**, weight decay **0.01**, '
               'balanced training-subset class weights, clipping **1.0**, seed **42**. Select by sampled AUROC, then '
               'average precision, then fewer features. No validation data fit preprocessing or gradients.\n\n'
               'Start rank-8/alpha-16 LoRA with zero B matrices in the same two block-30 projections as Q11. '
               'Clone the selected head into a frozen-backbone control. Both branches receive identical batches and '
               'update counts; LoRA uses live features and the control uses cached features. Batch **32**, at most '
               '**768 updates**, adapter learning rate **1e-4**, head learning rate **3e-5**, BF16 adapters/backbone '
               'with FP32 optimizer masters/head and original weights frozen.'),
            md('## Monitoring, stopping and integrity\n\n'
               'Evaluate initially, every **256 updates**, and at stopping. Stop after **two checks** without at least '
               '**0.002 AUROC** progress or when the one-hour budget needs its final validation/report reserve. '
               'Calibrate inference speed using training rows; reserve one sample evaluation and reload check with '
               'a **30% margin**, plus **three minutes** for reporting. Save matched optimizer/RNG/cursor state every '
               '32 updates; resuming retains the original deadline.\n\n'
               'Keep the best LoRA checkpoint together with its control from the same update count. Also retain the '
               'best frozen control across monitored steps. Select LoRA only if it exceeds the best control AUROC '
               'by **at least 0.005** without reducing AP. Otherwise select the control. The paired effect uses the '
               'control from the same update count as the best LoRA checkpoint.\n\n'
               'Training-only probes check repeatability, strand/batch equivalence, zero-adapter equivalence, finite '
               'gradients and actual adapter/head updates, then restore their changes. Verify unchanged original '
               'backbone tensors. Erase learned tensors and reload the selected checkpoint to reproduce 64 scores.'),
            code('result = q12.run_experiment()'),
            md('## Sampled development results\n\n'
               'AUROC and average precision use 1,000 whole-component bootstrap resamples, seed 42. These intervals '
               'do not correct feature, checkpoint or stopping selection bias. The figure shows matched continuation '
               'progress and precision–recall curves. A negative result is a valid outcome.'),
            code('q12.show_results()'),
            md('## Conclusion\n\n'
               'Full validation is separate: `.venv/bin/python -m notebooks.src.refresh_q12 --full-validation`. '
               'That entrypoint executes another notebook on all 17,927 variants before exporting a benchmark row. '
               'This exploratory notebook contributes a README research entry, not sampled benchmark scores.'),
            code('q12.show_conclusion()')]
    return nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Gamow','language':'python','name':'gamow'},'language_info':{'name':'python','version':'3.12'}})


def refresh(full=False):
    from . import comparison
    started=time.time();os.environ['Q12_RUN_STARTED_UTC']=str(started)
    q12.OUTPUT.mkdir(parents=True,exist_ok=True)
    notebook=build_notebook(full)
    name='Q12-lora-validation.ipynb' if full else 'Q12-lora-improvement.ipynb'
    path=q12.ROOT/'notebooks'/name
    directory=q12.OUTPUT/'full' if full else q12.OUTPUT
    directory.mkdir(exist_ok=True)
    q1.write_json(directory/'run_status.json',{'status':'running','started_utc':started})
    try:
        LoggedClient(notebook,timeout=None,kernel_name='gamow',allow_errors=False,
                     resources={'metadata':{'path':str(q12.ROOT/'notebooks')}}).execute()
        validate_execution(notebook);q12.verified_results(full)
    except BaseException as error:
        failed=directory/f'failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb'
        nbformat.write(notebook,failed)
        q1.write_json(directory/'run_status.json',{'status':'failed','reason':str(error),'notebook':str(failed.relative_to(q12.ROOT))})
        raise
    if path.exists():
        archive=directory/'notebook_history';archive.mkdir(exist_ok=True)
        shutil.copy2(path,archive/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.ipynb')
    temporary=path.with_suffix('.ipynb.partial');nbformat.write(notebook,temporary);temporary.replace(path)
    if full:q12.export_full_comparison(q12.verified_results(full=True))
    comparison.refresh()
    elapsed=time.time()-started
    q1.write_json(directory/'run_status.json',{'status':'complete','notebook_execution_seconds':elapsed,
        'within_one_hour':elapsed<=q12.CONFIG['total_seconds'] if not full else None})
    print(f'Saved fully executed {path}; fresh notebook elapsed {elapsed/60:.1f} minutes.',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--full-validation',action='store_true')
    refresh(parser.parse_args().full_validation)
