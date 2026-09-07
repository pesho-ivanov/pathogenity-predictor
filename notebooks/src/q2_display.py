"""Q2 notebook presentation, separate from the frozen calculation implementation.

The recorded Q2/Q9/Q10 experiments hash q2.py. Adjust rendering and cache-progress
messages while running the complete workflows, retaining all provenance checks.
"""

import re
from unittest.mock import patch

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from . import q0, q2


def extract_features(sequences):
    """Run feature extraction without the repeated cache-progress stamps."""
    def print_without_cache_progress(*args, **kwargs):
        if (len(args) == 1 and isinstance(args[0], str)
                and re.fullmatch(r'[\d,]+/[\d,]+ variants cached \(\d+\.\d min this run\)', args[0])):
            return
        print(*args, **kwargs)

    with patch.object(q2, 'print', print_without_cache_progress, create=True):
        return q2.extract_features(sequences)


def report_validation():
    """Run Q2 validation and display curves and numeric intervals."""
    with patch.object(q2, 'show_results', show_results):
        return q2.report_validation()


def show_results(result):
    table = q2.read_csv(q2.OUTPUT / 'validation_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    names = {'evo': 'Evo2 + logistic regression', 'zero_shot': 'Zero-shot Evo2',
             'sequence': 'Sequence + logistic regression'}
    for key, title in names.items():
        fpr, tpr, _ = roc_curve(table.label, table[key])
        precision, recall, _ = precision_recall_curve(table.label, table[key])
        metric = result['metrics'][key]
        axes[0].plot(fpr, tpr, label=f"{title}: {metric['auroc']:.3f}")
        axes[1].plot(recall, precision, label=f"{title}: {metric['average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], ':', color='gray')
    axes[1].axhline(result['validation_prevalence'], ls=':', color='gray', label='Prevalence')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Validation ROC · AUROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Validation PR · average precision')
    for ax in axes:
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.savefig(q2.OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    plt.close(fig)
    delta = result['evo_minus_zero_shot_auroc']
    low, high = result['bootstrap']['evo_minus_zero_shot_95ci']
    conclusion = 'higher than' if low > 0 else 'lower than' if high < 0 else 'not clearly different from'
    print(f'Evo2 classifier AUROC is {conclusion} zero-shot in this pilot: Δ={delta:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}].')
    print('Validation also selects C. These development metrics and intervals are not independent final test estimates.')
    q0.details('Validation metrics and paired component bootstrap', result)
