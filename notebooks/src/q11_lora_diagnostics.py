"""Read-only, seeded-sample diagnosis of Q11's completed LoRA checkpoint.

Run with --notebook to execute and save the investigation from a fresh kernel.
Neither the original Q11 sources, fitted weights nor metrics are rewritten.
"""

from datetime import datetime, timezone
import os
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from . import q0, q1, q11, q9_environment

OUTPUT = q11.OUTPUT / 'diagnostics/lora_quality'
TRAINING_PROBES = q11.OUTPUT / 'archive/lora_quality_full_attempt_20260907T232136Z'
CONFIG = {'seed': 42, 'training_probe_per_group': 1024, 'validation_probe_variants': 2048,
          'validation_selection': 'Uniform without replacement, seed 42, selected before inference; diagnostic only',
          'bootstrap_repetitions': 1000,
          'interventions': ['trained', 'zero_adapters_same_trained_head'],
          'magnitude_score_direction': 'Larger raw or relative allele change predicts pathogenicity; no validation-selected sign flip'}


def metrics(labels, scores):
    return {'auroc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores))}


def feature_statistics(frame, column):
    values = frame[column]
    correlation = values.corr(frame.score) if values.nunique() > 1 and frame.score.nunique() > 1 else np.nan
    return {'quantiles': np.quantile(values, [0, .01, .5, .99, 1]).tolist(),
            'pearson_with_full_score': float(correlation) if np.isfinite(correlation) else None}


def protocol():
    q11.verified_results()
    value = {'q11_identity': q11.verify_protocol(), 'configuration': CONFIG,
            'checkpoint_sha256': q1.digest_file(q11.OUTPUT/'final_adapter.pt'),
            'predictions_sha256': q1.digest_file(q11.OUTPUT/'comparison_predictions.csv'),
            'implementation_sha256': q1.digest_file(__file__)}
    prior = q1.read_json(TRAINING_PROBES/'protocol.json')
    if any(prior[key] != value[key] for key in ['q11_identity', 'checkpoint_sha256', 'predictions_sha256']):
        raise ValueError('Completed training probes belong to another model')
    q1.verify_file(TRAINING_PROBES/'q11_lora_diagnostics.py', prior['implementation_sha256'])
    value['completed_training_probes'] = {
        'directory': str(TRAINING_PROBES.relative_to(q11.ROOT)),
        'artifacts': {name: q1.digest_file(TRAINING_PROBES/name) for name in
                      ['protocol.json', 'q11_lora_diagnostics.py', 'seen_training.csv', 'unseen_training.csv']}}
    saved = pd.read_csv(q11.OUTPUT/'comparison_predictions.csv')
    chosen = np.sort(np.random.default_rng(CONFIG['seed']).choice(
        len(saved), size=min(CONFIG['validation_probe_variants'], len(saved)), replace=False))
    value['validation_variant_keys'] = saved.iloc[chosen].variant_key.tolist()
    return value


def prediction_summary(frame):
    labels, scores = frame.label.to_numpy(), frame.score.to_numpy()
    order = np.argsort(-scores, kind='stable')
    return {'variants': len(frame), 'pathogenic': int(labels.sum()), 'prevalence': float(labels.mean()),
            **metrics(labels, scores), 'precision_at_logit_zero': float(precision_score(labels, scores >= 0)),
            'recall_at_logit_zero': float(recall_score(labels, scores >= 0)),
            'top_fraction_precision': {str(f): float(labels[order[:max(1, int(len(frame)*f))]].mean())
                                       for f in [.01, .05, .1, .2]},
            'score_range': [float(scores.min()), float(scores.max())]}


def paired_bootstrap(frame, other):
    labels, groups = frame.label.to_numpy(), frame.component.to_numpy()
    a, b = frame.score.to_numpy(), other.score.to_numpy()
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(CONFIG['seed'])
    draws = {'auroc': [], 'average_precision': []}
    for _ in range(CONFIG['bootstrap_repetitions']):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if len(np.unique(labels[index])) < 2:
            continue
        ma, mb = metrics(labels[index], a[index]), metrics(labels[index], b[index])
        for key in draws:
            draws[key].append(ma[key]-mb[key])
    require = len(draws['auroc']) >= .9 * CONFIG['bootstrap_repetitions']
    if not require:
        raise ValueError('Insufficient paired component-bootstrap draws')
    ma, mb = metrics(labels, a), metrics(labels, b)
    return {key: {'trained_minus_disabled': ma[key]-mb[key],
                  'ci95': np.quantile(values, [.025, .975]).tolist()} for key, values in draws.items()}


def gpu():
    import torch
    from . import q11_backend as backend
    identity = q1.fingerprint(protocol())
    if q1.fingerprint(q1.read_json(OUTPUT/'protocol.json')) != identity:
        raise ValueError('Diagnostic protocol changed')
    start = time.perf_counter()
    _, manifest, dna, indexes, labels, _ = q11.verify_inputs()
    model, tokenizer = backend.load_backbone()
    try:
        adapters = backend.attach_lora(model)
        head = torch.nn.Linear(q11.CONFIG['feature_dimension'], 1, device='cuda', dtype=torch.float32)
        mean = torch.zeros(q11.CONFIG['feature_dimension'], device='cuda')
        scale = torch.ones_like(mean)
        state = backend.load_checkpoint(q11.OUTPUT/'final_adapter.pt', q11.verify_protocol())
        backend.restore_checkpoint(state, adapters, head, None, mean, scale)
        frozen_before = backend.parameter_hash(model, frozen_only=True)

        def collect(rows, name):
            blocks = []
            size = q11.CONFIG['microbatch_variants']
            with torch.no_grad():
                for offset in range(0, len(rows), size):
                    batch = rows.iloc[offset:offset+size]
                    sequences = [s for pair in backend.pairs_from(batch) for s in q11.crop_pair(*pair)]
                    values = backend.encode_batch(model, tokenizer, sequences)
                    reference, difference = values[0::2], values[1::2]-values[0::2]
                    features = backend.normalize_pair(reference, difference)
                    half = reference.shape[-1]
                    ref_score = features[:, :half] @ head.weight[0, :half]
                    delta_score = features[:, half:] @ head.weight[0, half:]
                    score = head(features).flatten()
                    if not torch.allclose(ref_score+delta_score+head.bias[0], score, atol=2e-5, rtol=2e-5):
                        raise ValueError('Classifier decomposition failed')
                    ref_rms = reference.double().square().mean(-1).sqrt()
                    delta_rms = difference.double().square().mean(-1).sqrt()
                    values = torch.stack([score, ref_score, delta_score, ref_rms, delta_rms,
                                          delta_rms/ref_rms.clamp_min(1e-12),
                                          (difference == 0).float().mean(-1),
                                          features[:, half:].double().square().mean(-1).sqrt()], dim=1)
                    if not torch.isfinite(values).all():
                        raise ValueError('Non-finite diagnostic features')
                    frame = pd.DataFrame(values.cpu().numpy(), columns=[
                        'score', 'reference_head_score', 'difference_head_score', 'reference_rms',
                        'difference_rms', 'relative_difference_rms', 'zero_difference_fraction', 'normalized_difference_rms'])
                    frame.insert(0, 'variant_key', batch.variant_key.to_numpy())
                    blocks.append(frame)
                    if offset // size % 16 == 0 or offset+len(batch) == len(rows):
                        print(f'{name}: {offset+len(batch)}/{len(rows)}', flush=True)
            result = pd.concat(blocks, ignore_index=True)
            result.to_csv(OUTPUT/f'{name}.csv', index=False)
            return result

        history = q1.read_json(q11.OUTPUT/'training_history.json')
        order = np.asarray(history['order'])
        train = dna.iloc[indexes['train']]
        selected = {}
        for name, pool in [('seen_training', order[:history['offset']]),
                           ('unseen_training', order[history['offset']:])]:
            probe = pd.read_csv(TRAINING_PROBES/f'{name}.csv')
            positions = pd.Series(np.arange(len(train)), index=train.variant_key).loc[probe.variant_key].to_numpy()
            if (not set(positions).issubset(set(pool))
                    or not np.array_equal(probe.label, labels['train'][positions])
                    or not np.array_equal(probe.component, manifest.loc[manifest.split.eq('train'), 'component'].to_numpy()[positions])):
                raise ValueError('Completed training probe membership or labels changed')
            probe.to_csv(OUTPUT/f'{name}.csv', index=False)
            selected[name] = prediction_summary(probe)
            print(f'{name} previously completed diagnostic, verified: {selected[name]}', flush=True)

        keys = q1.read_json(OUTPUT/'protocol.json')['validation_variant_keys']
        validation = dna.iloc[indexes['validation']].set_index('variant_key').loc[keys].reset_index()
        learned = collect(validation, 'trained_validation')
        saved = pd.read_csv(q11.OUTPUT/'comparison_predictions.csv').set_index('variant_key').loc[keys].reset_index()
        if (learned.variant_key.tolist() != saved.variant_key.tolist()
                or not np.allclose(learned.score, saved.fine_tuned, atol=2e-5, rtol=2e-5)):
            raise ValueError('Reloaded trained model differs from the published predictions')
        learned['label'] = saved.label.to_numpy()
        learned['component'] = saved.component.to_numpy()
        learned.to_csv(OUTPUT/'trained_validation.csv', index=False)
        print('Trained diagnostic-sample metrics:', prediction_summary(learned), flush=True)
        # Intervention: remove adapter output while retaining the jointly fitted head.
        with torch.no_grad():
            for adapter in adapters.values():
                adapter.linear_out.weight.zero_()
        disabled = collect(validation, 'disabled_validation')
        disabled['label'] = saved.label.to_numpy()
        disabled['component'] = saved.component.to_numpy()
        disabled.to_csv(OUTPUT/'disabled_validation.csv', index=False)
        backend.restore_checkpoint(state, adapters, head, None, mean, scale)
        if backend.parameter_hash(model, frozen_only=True) != frozen_before:
            raise ValueError('Original backbone changed during diagnostics')

        report = {'identity': identity, 'status': 'complete', 'configuration': CONFIG,
                  'scope': 'User-authorized seeded validation diagnostic sample; full benchmark remains unchanged',
                  'full_validation': prediction_summary(pd.read_csv(q11.OUTPUT/'comparison_predictions.csv').rename(columns={'fine_tuned':'score'})),
                  'trained': prediction_summary(learned), 'disabled': prediction_summary(disabled),
                  'paired_differences': paired_bootstrap(learned, disabled), 'training_probes': selected,
                  'component_scores': {mode: {column: metrics(frame.label, frame[column]) for column in
                       ['reference_head_score', 'difference_head_score', 'difference_rms', 'relative_difference_rms']}
                       for mode, frame in [('trained', learned), ('disabled', disabled)]},
                  'feature_summary': {mode: {column: feature_statistics(frame, column)
                       for column in ['reference_head_score','difference_head_score','reference_rms',
                                      'difference_rms','relative_difference_rms','zero_difference_fraction','normalized_difference_rms']}
                       for mode, frame in [('trained',learned),('disabled',disabled)]},
                  'head_weight_norms': {'reference': float(head.weight[0,:4096].norm()),
                                        'difference': float(head.weight[0,4096:].norm()), 'bias': float(head.bias[0])},
                  'sampled_published_predictions_reproduced': True, 'original_backbone_unchanged': True,
                  'seconds': time.perf_counter()-start,
                  'limitations': 'Post-hoc development diagnostics on a fixed seeded sample of 2,048 validation variants; these do not replace the original 17,927-variant benchmark. Disabled adapters retain the jointly trained head; this is not a fitted frozen-head baseline. Previously completed training probes are verified and reused from the archived full-pass attempt. Training probes are diagnostic samples, not full-cohort performance estimates. The 512-base representation, single strand, normalization, head and backend all differ from Q2; the Q2–Q11 gap cannot be assigned to LoRA alone.',
                  'artifacts': {p.name:q1.digest_file(p) for p in OUTPUT.glob('*.csv')}}
        q1.write_json(OUTPUT/'report.json', report, frozen=True)
        print('Diagnostic evaluation complete:', report['paired_differences'], flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = protocol()
    q1.write_json(OUTPUT/'protocol.json', identity, frozen=True)
    if not (OUTPUT/'report.json').exists():
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        with (OUTPUT/'run.log').open('w') as log:
            command = [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.q11_lora_diagnostics', '--gpu']
            print('Seeded 2,048-variant diagnostic progress: results/q11/diagnostics/lora_quality/run.log', flush=True)
            subprocess.run(command, cwd=q11.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = q1.read_json(OUTPUT/'report.json')
    if report['status'] != 'complete' or report['identity'] != q1.fingerprint(identity):
        raise ValueError('Incomplete or mismatched diagnostic results')
    for name, checksum in report['artifacts'].items():
        q1.verify_file(OUTPUT/name, checksum)
    return report


def show(report):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    learned = pd.read_csv(OUTPUT/'trained_validation.csv')
    disabled = pd.read_csv(OUTPUT/'disabled_validation.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for frame, label in [(learned,'Trained LoRA'),(disabled,'Adapters disabled; same head')]:
        precision, recall, _ = precision_recall_curve(frame.label, frame.score)
        axes[0].plot(recall, precision, label=label)
    axes[0].axhline(learned.label.mean(), color='gray', linestyle='--', label='Positive-label prevalence')
    axes[0].set(xlabel='Recall', ylabel='Precision')
    axes[0].legend(fontsize=8)
    ordered = learned.sort_values('score')
    bins = np.array_split(np.arange(len(ordered)), 10)
    axes[1].plot([ordered.iloc[b].score.mean() for b in bins],
                 [ordered.iloc[b].label.mean() for b in bins], marker='o')
    axes[1].set(xlabel='Mean logit in score decile', ylabel='Observed pathogenic fraction')
    fig.savefig(OUTPUT/'diagnostic_curves.png', dpi=160)
    plt.show()
    q0.details('Measured interventions, feature branches and training probes', report)


def conclude(report):
    from IPython.display import Markdown, display
    a,b=report['trained'],report['disabled']
    d=report['paired_differences']['auroc']
    branches=report['component_scores']['trained']
    full = report['full_validation']
    display(Markdown(f"**Conclusion.** The original full-validation AUROC/AP remain **{full['auroc']:.3f}/{full['average_precision']:.3f}**. "
        f"On the fixed 2,048-variant diagnostic sample, trained AUROC/AP are **{a['auroc']:.3f}/{a['average_precision']:.3f}**, reproducing saved scores. "
        f"Disabling adapters with the trained head fixed gives **{b['auroc']:.3f}/{b['average_precision']:.3f}** "
        f"(trained-minus-disabled AUROC {d['trained_minus_disabled']:+.3f}, 95% component-bootstrap interval "
        f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]). The reference and difference head contributions have "
        f"AUROC **{branches['reference_head_score']['auroc']:.3f}** and **{branches['difference_head_score']['auroc']:.3f}**, respectively. "
        f"Larger raw allele-difference magnitude scores AUROC **{branches['difference_rms']['auroc']:.3f}**; "
        f"the classifier's per-example normalization discards this magnitude. "
        f"At logit zero, precision/recall are **{a['precision_at_logit_zero']:.3f}/{a['recall_at_logit_zero']:.3f}**; "
        f"AP is a ranking summary, with positive prevalence **{a['prevalence']:.3f}**. "
        f"{report['limitations']}"))


def notebook():
    import nbformat
    from nbclient import NotebookClient
    from .refresh_q11 import validate_execution
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    value = nbformat.v4.new_notebook(cells=[
        md('# Q11. Why are the partial-epoch LoRA scores limited?\n\n'
           'Investigate the completed run without modifying its fitted checkpoint or original notebook. '
           'Use a uniform, seeded sample of 2,048 frozen validation variants for two inference passes: trained adapters '
           'and zero adapter outputs with the jointly fitted head held fixed. This user-authorized speed adjustment '
           'leaves the original 17,927-variant benchmark unchanged. Previously completed seeded probes use 1,024 '
           'already-seen and 1,024 unvisited training rows; their source, checkpoint identity, checksums and membership '
           'are verified before reuse. These are diagnostic samples, not full-cohort results. '
           '[Implementation](src/q11_lora_diagnostics.py) · [Original experiment](Q11-evo2-lora.ipynb).'),
        code('from src import q11_lora_diagnostics as diagnostic'),
        md('## Checks and interventions\n\n'
           'Verify the original source/checkpoint/prediction checksums, freeze the seed-42 sample keys before inference, '
           'and reproduce every saved score on that sample. '
           'Decompose the linear head into reference and allele-difference contributions. Measure raw and relative '
           'difference magnitudes before the per-example RMS normalization. Larger magnitude is predeclared as the '
           'positive direction; no validation-selected sign flip. Restore adapter weights after the intervention and '
           'verify unchanged original backbone tensors. No new parameters or thresholds are fitted. '
           'Expect about 6–8 minutes on the prepared H100.'),
        code('report = diagnostic.run()'),
        md('## Results\n\n'
           'AP is average precision over the ranking, distinct from precision at a chosen threshold. '
           'The classifier uses balanced training class weights, so its raw sigmoid is not established as a calibrated risk. '
           'Paired differences use 1,000 whole-component bootstrap draws, seed 42. '
           'Branch and magnitude scores are post-hoc diagnostics and do not provide an independent model-selection holdout.'),
        code('diagnostic.show(report)'),
        md('## Interpretation\n\n'
           'The original run used 52.4% of training variants, 512-base single-strand paired inputs and a randomly initialized '
           'joint head, without fitting a frozen-head baseline. Several choices differ from Q2. '
           '[Arc recommends intermediate embeddings over final embeddings](https://github.com/ArcInstitute/evo2#embeddings); '
           'that supports testing the feature layer, but does not prove the cause of this run’s gap. '
           'A head-only control is also part of [NVIDIA’s Evo2 LoRA example](https://docs.nvidia.com/bionemo-recipes/latest/main/recipes/recipes/evo2_megatron/index.html#lora-fine-tuning).'),
        code('diagnostic.conclude(report)'),
    ],metadata={'kernelspec':{'display_name':'Gamow','language':'python','name':'gamow'}})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    try:
        NotebookClient(value,timeout=None,kernel_name='gamow',allow_errors=False,
                       resources={'metadata':{'path':str(q11.ROOT/'notebooks')}}).execute()
        validate_execution(value)
    except BaseException:
        nbformat.write(value,OUTPUT/f'failed-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.ipynb')
        raise
    path=q11.ROOT/'notebooks/Q11-lora-diagnostics.ipynb'
    temporary=path.with_suffix('.ipynb.partial')
    nbformat.write(value,temporary)
    temporary.replace(path)
    print('Saved fully executed diagnostic notebook:',path,flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--notebook',action='store_true')
    args=parser.parse_args()
    if args.gpu:
        gpu()
    elif args.notebook:
        notebook()
    else:
        parser.error('Choose --gpu or --notebook')
