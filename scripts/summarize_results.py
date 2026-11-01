"""Add a measured discussion and class-level diagnostics using saved predictions only."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.work' / 'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def summarize():
    results = ROOT / 'results'
    comparison = pd.read_csv(results / 'comparison.csv')
    assert len(comparison) == 6 and (comparison.epochs == 30).all()
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
    class_rows = []
    for col, dataset in enumerate(('C16', 'C17', 'BRACS')):
        for row, method in enumerate(('baseline', 'label_smoothing')):
            folder = results / f'{dataset}_{method}'
            predictions = pd.read_csv(folder / 'test_predictions.csv')
            classes = np.arange(predictions.filter(regex=r'^p_\d+$').shape[1])
            precision, recall, f1, support = precision_recall_fscore_support(
                predictions.label, predictions.prediction, labels=classes, zero_division=0)
            for i in classes:
                class_rows.append(dict(dataset=dataset, method=method, class_id=int(i),
                                       precision=precision[i], recall=recall[i], f1=f1[i], support=int(support[i])))
            counts = confusion_matrix(predictions.label, predictions.prediction, labels=classes)
            pd.DataFrame(counts, index=classes, columns=classes).to_csv(folder / 'confusion_counts.csv')
            normalized = counts / np.maximum(counts.sum(1, keepdims=True), 1)
            ax = axes[row, col]
            artist = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
            for i in classes:
                for j in classes:
                    ax.text(j, i, f'{normalized[i, j]:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if normalized[i, j] > 0.5 else 'black')
            ax.set(xticks=classes, yticks=classes, xlabel='Predicted class ID', ylabel='True class ID',
                   title=f'{dataset}: {method}')
    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=0.7, label='Fraction within true class')
    fig.savefig(results / 'confusion_matrices.png', dpi=160)
    plt.close(fig)
    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(results / 'per_class_metrics.csv', index=False)

    table = ['| Dataset | Method | Best epoch | Test AUROC | Test macro-F1 | Accuracy (%) |',
             '|---|---|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        table.append(f'| {row.dataset} | {row.method} | {row.best_epoch} | {row.auc:.4f} | '
                     f'{row.f1:.4f} | {row.acc:.2f} |')
    by_method = comparison.set_index(['dataset', 'method'])
    c17 = per_class[per_class.dataset == 'C17'].set_index(['method', 'class_id'])
    details = []
    for dataset in ('C16', 'C17', 'BRACS'):
        baseline, extension = (by_method.loc[(dataset, method)] for method in ('baseline', 'label_smoothing'))
        delta = extension[['auc', 'f1', 'loss', 'brier', 'ece']] - baseline[['auc', 'f1', 'loss', 'brier', 'ece']]
        details.append(f'**{dataset}.** Label smoothing changed AUROC by {delta.auc:+.4f} and macro-F1 '
                       f'by {delta.f1:+.4f}; NLL changed by {delta.loss:+.4f}, Brier score by '
                       f'{delta.brier:+.4f}, and ECE by {delta.ece:+.4f}. Lower values are better for '
                       'the last three metrics.')
    details.append(
        '**C17 class-level tradeoff.** The baseline recalls for classes 1 and 2 were '
        f'{c17.loc[("baseline", 1), "recall"]:.3f} and {c17.loc[("baseline", 2), "recall"]:.3f}; '
        'the smoothed model recalls were '
        f'{c17.loc[("label_smoothing", 1), "recall"]:.3f} and '
        f'{c17.loc[("label_smoothing", 2), "recall"]:.3f}. '
        'These classes have only 16 and 18 test slides, respectively, versus 121 in class 0. '
        'Improved overall accuracy or calibration therefore does not imply improved recognition of every '
        'class. Class IDs follow the supplied HDF5 labels. The normalized confusion matrices below and '
        '`results/per_class_metrics.csv` expose this tradeoff.')
    details.append(
        '**Interpretation and limits.** This fixed-epsilon experiment shows dataset-dependent tradeoffs, '
        'not a general improvement from label smoothing. It does not consistently satisfy the hypothesis '
        'of better probability quality with competitive macro-F1. Smoothing may reduce overconfidence '
        'but also suppress separation of rare classes or cause underconfidence. Those are possible '
        'explanations, not causal conclusions. Baseline and extension sometimes select different epochs, '
        'which is another part of the observed effect. Both C16 models select epoch 3 because the small '
        'validation set reaches the maximum selection score then; later training is not selected merely '
        'for lower NLL. Only one seed is run, and the fixed, slide-level C17 validation split shares '
        '27 patients with training (none with test). Repeated seeds, patient-grouped validation, and '
        'validation-only selection of epsilon would be appropriate follow-up work. No test-driven '
        'hyperparameter changes were made in this comparison.')
    source = ('## Measured results and discussion\n\nThis section records the saved execution supplied '
              'with this notebook. Regenerate it with `scripts/summarize_results.py` after changing '
              'the experiments.\n\nAll six runs completed the configured 30 epochs '
              '(180 epochs total), using the supplied splits and seed 42. Test metrics below come from '
              'the validation-selected checkpoints.\n\n' + '\n'.join(table) + '\n\n' +
              '\n\n'.join(details) + '\n\n![Per-class confusion matrices](results/confusion_matrices.png)')
    path = ROOT / 'ECE1512_ProjectA_Student.ipynb'
    nb = nbformat.read(path, as_version=4)
    nb.cells = [cell for cell in nb.cells if cell.get('id') != 'measured-discussion']
    nb.cells.append(nbformat.v4.new_markdown_cell(source, id='measured-discussion'))
    nb.cells[0].source = '\n'.join(
        'The model, data loaders, training/evaluation pipeline, report answers, and research extension '
        'are completed below. Dataset and metric helpers remain in the provided modules.'
        if line.startswith('Please complete every cell marked') else line
        for line in nb.cells[0].source.split('\n'))
    nbformat.write(nb, path)
    print('Saved per-class metrics, confusion matrices, and the notebook discussion.')


if __name__ == '__main__':
    summarize()
