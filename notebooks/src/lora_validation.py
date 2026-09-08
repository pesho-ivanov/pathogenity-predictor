"""Complete-cohort development confirmation of a promoted Q13, Q14 or Q15 adapter."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from . import q0, q1, q1_full, q11, q12, q13, q14, q9_environment

ROOT = q11.ROOT
CONFIG = {
    'seed': 42, 'bootstrap_repetitions': 1000, 'reload_variants': 64,
    'microbatch_variants': 32, 'validation_variants': 17927,
    'confirmation_auroc': .005,
    'confirmation': 'Full validation: LoRA AUROC gain >=0.005, AP gain >=0 and paired AUROC interval lower bound >0 against the strongest frozen control. Outside the 2048 selection variants: AUROC gain >0 and AP gain >=0. No additional fitting or model selection.',
}
LIMITATIONS = (
    'Full frozen July missense development validation, following repeated selection '
    'on its 2,048-variant subset. Variants outside that subset and components absent '
    'from it were not used by those recent selection runs, but this repository has '
    'previously evaluated the full validation partition. Neither subset is an '
    'untouched final test set. Component-bootstrap intervals do not remove selection '
    'bias. Pretraining exposure, homology and shared-patient overlap remain unresolved; '
    'this research prototype has no clinical validation.'
)
require = q12.require
metric = q12.metric
better = q12.better


def parent(question):
    require(question in {'q13', 'q14', 'q15'}, 'Full LoRA validation question must be q13, q14 or q15')
    if question == 'q15':
        from . import q15
        return q15
    return {'q13': q13, 'q14': q14}[question]


def output(question):
    return parent(question).OUTPUT / 'full'


def sources(question):
    names = ['lora_validation.py', 'lora_validation_backend.py', 'refresh_lora_validation.py', 'refresh_q11.py', 'comparison.py']
    return parent(question).sources() | {
        f'notebooks/src/{name}': q1.digest_file(ROOT / 'notebooks/src' / name) for name in names}


def _full_inputs():
    protocol, manifest, dna, indexes, labels, checks = q11.verify_inputs()
    rows = dna.iloc[indexes['validation']].reset_index(drop=True)
    groups = manifest.set_index('variant_key').loc[rows.variant_key, 'component'].to_numpy()
    require(len(rows) == CONFIG['validation_variants'] and not rows.variant_key.duplicated().any(),
            'Complete frozen validation must contain all 17,927 unique variants')
    return protocol, rows, labels['validation'], groups, checks


def prepare(question):
    """Freeze the selected adapter and all confirmation rules before full scoring."""
    module = parent(question)
    result = module.verified_results()
    require(result['promotion_passed'] is True and result['selected_model'] == 'lora',
            'Full adapter confirmation requires a promoted parent LoRA result')
    protocol, rows, labels, groups, checks = _full_inputs()
    inherited = q1.read_json(module.OUTPUT / 'protocol.json')
    keys = inherited['validation_variant_keys']
    require(len(keys) == 2048 and len(set(keys)) == len(keys) and set(keys) <= set(rows.variant_key),
            'Parent selection sample is not the frozen 2,048-variant validation subset')
    indexed_groups = dict(zip(rows.variant_key, groups))
    selection_groups = [indexed_groups[key] for key in keys]
    value = {
        'configuration': CONFIG, 'question': question, 'sources': sources(question),
        'parent_identity': module.verify_protocol(),
        'parent_metrics_sha256': q1.digest_file(module.OUTPUT / 'metrics.json'),
        'parent_selected_model_sha256': q1.digest_file(module.OUTPUT / 'selected_model.pt'),
        'parent_protocol_sha256': q1.digest_file(module.OUTPUT / 'protocol.json'),
        'q1_identity': q1.fingerprint(protocol),
        'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json'),
        'vcf_exports': protocol['vcf_exports'],
        'validation_variant_keys': rows.variant_key.tolist(),
        'validation_variants': len(rows), 'labels_sha256': q1.fingerprint(labels.tolist()),
        'components_sha256': q1.fingerprint(groups.tolist()),
        'selection_variant_keys': keys, 'selection_components': sorted(set(selection_groups)),
        'selection_variant_components': selection_groups,
        'outside_selection_variants': int((~rows.variant_key.isin(keys)).sum()),
        'unseen_component_variants': int((~np.isin(groups, selection_groups)).sum()),
        'limitations': LIMITATIONS,
    }
    directory = output(question)
    directory.mkdir(parents=True, exist_ok=True)
    q1.write_json(directory / 'protocol.json', value, frozen=True)
    q1.write_json(directory / 'input_checks.json', checks, frozen=True)
    for relative, sha in value['sources'].items():
        source = ROOT / relative
        target = directory / 'source_snapshot' / relative
        q1.verify_file(source, sha)
        if target.exists():
            q1.verify_file(target, sha)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            q1.verify_file(target, sha)
    print(f"Frozen {question.upper()} LoRA confirmation on all {len(rows):,} validation variants; "
          f"{value['outside_selection_variants']:,} outside selection and "
          f"{value['unseen_component_variants']:,} in components absent from selection.", flush=True)
    q0.details('Frozen full-validation scope, confirmation rule and provenance', {
        key: value[key] for key in ['configuration', 'question', 'parent_identity',
            'parent_metrics_sha256', 'parent_selected_model_sha256', 'q1_identity',
            'validation_variants', 'outside_selection_variants', 'unseen_component_variants', 'limitations']})
    q0.details('Inherited full-cohort eligibility and leakage checks', checks)
    return value


def verify_protocol(question):
    module = parent(question)
    value = q1.read_json(output(question) / 'protocol.json')
    require(value['question'] == question and value['configuration'] == CONFIG
            and value['sources'] == sources(question), 'Full-validation configuration or sources changed')
    require(value['parent_identity'] == module.verify_protocol(), 'Parent source identity changed')
    for name, key in [('metrics.json', 'parent_metrics_sha256'),
                      ('selected_model.pt', 'parent_selected_model_sha256'),
                      ('protocol.json', 'parent_protocol_sha256')]:
        q1.verify_file(module.OUTPUT / name, value[key])
    q1.verify_file(q1_full.OUTPUT / 'protocol.json', value['q1_protocol_sha256'])
    full = q1_full.verify_protocol()
    require(value['q1_identity'] == q1.fingerprint(full) and value['vcf_exports'] == full['vcf_exports'],
            'Frozen full-cohort validation identity changed')
    inherited = q1.read_json(module.OUTPUT / 'protocol.json')
    require(value['selection_variant_keys'] == inherited['validation_variant_keys'],
            'Original selection membership changed')
    return q1.fingerprint(value)


def inputs(question):
    verify_protocol(question)
    _, rows, labels, groups, checks = _full_inputs()
    protocol = q1.read_json(output(question) / 'protocol.json')
    require(rows.variant_key.tolist() == protocol['validation_variant_keys'], 'Full validation order changed')
    require(q1.fingerprint(labels.tolist()) == protocol['labels_sha256'], 'Full validation labels changed')
    require(q1.fingerprint(groups.tolist()) == protocol['components_sha256'], 'Full validation components changed')
    return rows, labels, groups, checks


def intervals(labels, predictions, groups):
    """Shared component resamples, with the paired effect against the strongest control."""
    require({'lora', 'matched_control', 'strongest_control'} <= set(predictions),
            'Confirmation requires the selected LoRA and both frozen controls')
    renamed = {key: value for key, value in predictions.items()
               if key not in {'matched_control', 'strongest_control'}}
    renamed['original_matched_control'] = predictions['matched_control']
    renamed['matched_control'] = predictions['strongest_control']
    metrics, paired = q12.intervals(labels, renamed, groups, CONFIG['bootstrap_repetitions'])
    metrics['strongest_control'] = metrics.pop('matched_control')
    metrics['matched_control'] = metrics.pop('original_matched_control')
    return metrics, paired


def confirmation_passes(result):
    """Apply the frozen full-cohort and outside-selection confirmation gate."""
    full = result['metrics']
    paired = result['paired_lora_minus_strongest_control']
    outside = result['subsets'].get('outside_selection', {})
    if 'metrics' not in outside or outside.get('status') == 'not_available':
        return False
    other = outside['metrics']
    return bool(
        full['lora']['auroc']['value'] - full['strongest_control']['auroc']['value']
        >= CONFIG['confirmation_auroc'] - 1e-12
        and full['lora']['average_precision']['value'] >= full['strongest_control']['average_precision']['value']
        and paired['auroc']['ci95'][0] > 0
        and other['lora']['auroc']['value'] > other['strongest_control']['auroc']['value']
        and other['lora']['average_precision']['value'] >= other['strongest_control']['average_precision']['value']
    )


def run_validation(question):
    """Score the frozen models in an actual GPU run, outside exploration's budget."""
    verify_protocol(question)
    directory = output(question)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (directory / 'metrics.json').exists(),
                'Full validation already completed; use refresh_lora_validation to archive and rerun')
        execution = directory / 'execution.json'
        if not execution.exists():
            q1.write_json(execution, {'started_utc': float(os.environ.get('LORA_VALIDATION_STARTED_UTC', time.time())),
                                     'question': question}, frozen=True)
        log = directory / 'validation.log'
        if log.exists():
            archive = directory / 'logs'
            archive.mkdir(exist_ok=True)
            log.rename(archive / f'validation-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
        args = [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.lora_validation_backend', question]
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        print(f'Full validation: {log.relative_to(ROOT)}', flush=True)
        with log.open('w') as stream:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     pass_fds=(lock.fileno(),))
            try:
                status = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        require(status == 0, 'Full LoRA validation failed:\n' + '\n'.join(log.read_text(errors='replace').splitlines()[-30:]))
    return verified_results(question)


def verified_results(question):
    directory = output(question)
    result = q1.read_json(directory / 'metrics.json')
    require(result.get('status') == 'complete' and result.get('scope') == 'full_validation'
            and result.get('identity') == verify_protocol(question), 'Incomplete or stale full-validation results')
    require(result.get('validation_variants') == CONFIG['validation_variants'], 'Incomplete validation coverage')
    require(result.get('reload_verified') is True and result.get('frozen_unchanged') is True,
            'Full-validation reload or frozen-backbone checks failed')
    require('validation_predictions.csv' in result.get('artifacts', {}), 'Verified full predictions are missing')
    for name, sha in result['artifacts'].items():
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()), 'Validation artifact path escapes its directory')
        q1.verify_file(path, sha)
    frame = pd.read_csv(directory / 'validation_predictions.csv')
    protocol = q1.read_json(directory / 'protocol.json')
    require(frame.variant_key.tolist() == protocol['validation_variant_keys']
            and not frame.variant_key.duplicated().any(), 'Predictions do not cover the exact full validation order')
    require(q1.fingerprint(frame.label.astype(int).tolist()) == protocol['labels_sha256']
            and q1.fingerprint(frame.component.tolist()) == protocol['components_sha256'],
            'Prediction labels or components differ from frozen validation')
    require({'lora', 'matched_control', 'strongest_control'} <= set(result['metrics']),
            'Full-validation comparison is incomplete')
    for name in ['lora', 'matched_control', 'strongest_control']:
        values = metric(frame.label.to_numpy(), frame[name].to_numpy())
        require(all(np.isclose(values[key], result['metrics'][name][key]['value'], atol=1e-12, rtol=0.)
                    for key in values), 'Reported full metrics do not reproduce saved predictions')
    expected = {'outside_selection': ~frame.variant_key.isin(protocol['selection_variant_keys']).to_numpy(),
                'unseen_components': ~frame.component.isin(protocol['selection_components']).to_numpy()}
    for name, mask in expected.items():
        subset = result['subsets'][name]
        require(subset['variants'] == int(mask.sum()), f'Incorrect {name} cohort size')
        if 'metrics' in subset:
            for model in ['lora', 'matched_control', 'strongest_control']:
                values = metric(frame.label.to_numpy()[mask], frame[model].to_numpy()[mask])
                require(all(np.isclose(values[key], subset['metrics'][model][key]['value'], atol=1e-12, rtol=0.)
                            for key in values), f'Incorrect {name} reported metrics')
        else:
            require(subset.get('status') == 'not_available'
                    and len(np.unique(frame.label.to_numpy()[mask])) < 2,
                    f'{name} results cannot be omitted when both classes exist')
    require(bool(result['confirmation_passed']) == confirmation_passes(result),
            'Full-validation confirmation disagrees with the frozen rule')
    return result


def export_full_comparison(question):
    """Export both fixed models only after complete inference and notebook execution."""
    from . import comparison

    result = verified_results(question)
    directory, module = output(question), parent(question)
    protocol = q1.read_json(directory / 'protocol.json')
    require(result.get('original_selection_scores_reproduced') is True,
            'Full benchmark publication requires reproduction of the original selection scores')
    evidence = comparison.full_lora_notebook_evidence(ROOT, question, result)
    parent_result = q1.read_json(module.OUTPUT / 'metrics.json')
    exploration = comparison.result_notebook_evidence(ROOT,
        'notebooks/' + comparison.LORA_EXPLORATION_NOTEBOOKS[question], parent_result)
    parent_status = q1.read_json(module.OUTPUT / 'run_status.json')
    require(parent_status.get('status') == 'complete'
            and parent_status.get('notebook_sha256') == exploration['sha256'],
            'Full benchmark publication requires the completed exploration notebook')
    status = q1.read_json(directory / 'run_status.json')
    require(status.get('status') == 'complete' and status.get('question') == question
            and status.get('notebook_sha256') == evidence['sha256'],
            'Full benchmark publication requires the saved completed notebook and its completion record')
    path = module.OUTPUT / 'comparison_predictions.csv'
    temporary = path.with_name(path.name + '.partial')
    shutil.copyfile(directory / 'validation_predictions.csv', temporary)
    temporary.replace(path)
    files = {'full/metrics.json', 'full/protocol.json', 'full/run_status.json', 'full/validation_predictions.csv',
             'protocol.json', 'metrics.json', 'selected_model.pt', 'run_status.json'}
    files.update('full/' + name for name in result['artifacts'])
    exploration_seconds = parent_status['notebook_execution_seconds']
    full_seconds = status['notebook_execution_seconds']
    require(all(isinstance(value, (int, float)) and np.isfinite(value) and value >= 0
                for value in [exploration_seconds, full_seconds]), 'Completed notebook runtime is invalid')
    outcome = 'met' if result['confirmation_passed'] else 'not met'
    spec = {
        'scope': 'full_validation', 'require_complete': True,
        'q1_protocol_sha256': protocol['q1_protocol_sha256'], 'vcf_exports': protocol['vcf_exports'],
        'methods': comparison.LORA_FULL_METHODS[question], 'sources': protocol['sources'],
        'predictions_sha256': q1.digest_file(path), 'full_identity': result['identity'],
        'parent_identity': protocol['parent_identity'], 'notebook': evidence, 'exploration_notebook': exploration,
        'artifacts': {name: q1.digest_file(module.OUTPUT / name) for name in sorted(files)},
        'confirmation_passed': result['confirmation_passed'],
        'runtime_stages': {'exploration_seconds': exploration_seconds, 'full_validation_seconds': full_seconds},
        'runtimes': {name: {'seconds': exploration_seconds + full_seconds,
            'scope': f'Current {question.upper()} exploration ({exploration_seconds/60:.1f} min, including all its search arms) '
                     f'plus separate full validation ({full_seconds/60:.1f} min), shared by LoRA and controls. '
                     'Excludes inherited Q12 feature extraction/fitting and earlier experiments or attempts. '
                     'The one-hour target applies only to exploration.'}
            for name in comparison.LORA_FULL_METHODS[question]},
        'limitations': f'Prespecified full-validation confirmation rule {outcome}. ' + LIMITATIONS,
    }
    # A new complete run replaces the export; refresh archived its preceding full
    # evidence, and the executed notebook remains checksum-bound to these results.
    q1.write_json(module.OUTPUT / 'comparison_results.json', spec)
    return spec


def show_results(question):
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display
    from sklearn.metrics import precision_recall_curve

    result = verified_results(question)
    frame = pd.read_csv(output(question) / 'validation_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    for name in ['lora', 'matched_control', 'strongest_control']:
        precision, recall, _ = precision_recall_curve(frame.label, frame[name])
        axes[0].plot(recall, precision, label=name.replace('_', ' '))
    axes[0].axhline(frame.label.mean(), color='gray', linestyle=':', label='Prevalence')
    axes[0].set(xlabel='Recall', ylabel='Precision', title='Complete development validation')
    axes[0].legend(fontsize=8)
    cohorts = [('Full', result)] + [(label, result['subsets'][key]) for label, key in
                                    [('Outside selection', 'outside_selection'), ('Unseen components', 'unseen_components')]]
    for position, (_, cohort) in enumerate(cohorts):
        if 'paired_lora_minus_strongest_control' not in cohort:
            continue
        value = cohort['paired_lora_minus_strongest_control']['auroc']
        low, high = value['ci95']
        axes[1].vlines(position, low, high, color='tab:blue')
        axes[1].plot(position, value['value'], marker='o', color='tab:blue')
    axes[1].axhline(0, color='gray', linestyle=':')
    axes[1].set(xticks=range(len(cohorts)), xticklabels=[name for name, _ in cohorts],
                ylabel='LoRA minus strongest control AUROC', title='Paired component intervals')
    axes[1].tick_params(axis='x', labelsize=8)
    fig.savefig(output(question) / 'validation_curves.png', dpi=160)
    plt.show()
    table = ['| Model | Full AUROC (95% interval) | Full AP (95% interval) |', '| --- | ---: | ---: |']
    for name in ['lora', 'matched_control', 'strongest_control']:
        values = result['metrics'][name]
        text = [f"{values[key]['value']:.3f} [{values[key]['ci95'][0]:.3f}, {values[key]['ci95'][1]:.3f}]"
                for key in ['auroc', 'average_precision']]
        table.append(f"| {name.replace('_', ' ')} | {text[0]} | {text[1]} |")
    display(Markdown('\n'.join(table)))
    q0.details('Full results, prespecified subsets, integrity checks and provenance', result)


def show_conclusion(question):
    from IPython.display import Markdown, display

    result = verified_results(question)
    selected = result['metrics']['lora']
    paired = result['paired_lora_minus_strongest_control']['auroc']
    decision = ('met' if result['confirmation_passed'] else 'did not meet')
    message = (
        f"**Conclusion.** The frozen {question.upper()} adapter **{decision}** the prespecified "
        f"full-validation confirmation rule. LoRA achieved **{selected['auroc']['value']:.3f} AUROC** "
        f"and **{selected['average_precision']['value']:.3f} AP** on **{result['validation_variants']:,} variants**. "
        f"Its AUROC gain against the strongest frozen control was **{paired['value']:+.4f}** "
        f"(95% component interval **[{paired['ci95'][0]:+.4f}, {paired['ci95'][1]:+.4f}]**). "
        'Confirmation requires full AUROC gain at least 0.005, no AP decrease and a positive lower '
        'AUROC interval bound, plus positive AUROC gain with no AP decrease outside the selection sample. '
        f"The separate validation run took **{result['seconds']/60:.1f} minutes**. " + LIMITATIONS
    )
    display(Markdown(message))
