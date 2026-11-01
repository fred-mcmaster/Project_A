"""Independently verify all six reported runs against their saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import nbformat

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def verify():
    comparison = pd.read_csv(RESULTS / 'comparison.csv')
    assert len(comparison) == 6
    assert set(zip(comparison.dataset, comparison.method)) == {
        (dataset, method) for dataset in ('C16', 'C17', 'BRACS')
        for method in ('baseline', 'label_smoothing')}
    for row in comparison.itertuples():
        folder = ROOT / row.run_dir
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        history = pd.read_csv(folder / 'history.csv')
        predictions = pd.read_csv(folder / 'test_predictions.csv')
        split = pd.read_csv(RESULTS / row.dataset / 'splits.csv')
        assert len(history) == row.epochs == checkpoint['epochs_completed'] == 30
        best = (history.val_auc + history.val_f1).idxmax()
        assert best + 1 == row.best_epoch == checkpoint['best_epoch'] + 1
        assert predictions.slide_id.tolist() == split.loc[split.split == 'test', 'slide_id'].tolist()
        assert predictions.label.tolist() == split.loc[split.split == 'test', 'label'].tolist()
        p = predictions.filter(regex=r'^p_\d+$').to_numpy()
        y = predictions.label.to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        np.testing.assert_array_equal(p.argmax(1), predictions.prediction)
        independent = dict(acc=100 * accuracy_score(y, p.argmax(1)),
                           f1=f1_score(y, p.argmax(1), average='macro'),
                           auc=roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else
                           roc_auc_score(y, p, average='macro', multi_class='ovr'),
                           # Remove float32 softmax roundoff before sklearn's float64 sum check.
                           loss=log_loss(y, p / p.sum(axis=1, keepdims=True)),
                           brier=np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))
        confidence = p.max(1)
        correct = p.argmax(1) == y
        bins = np.minimum((confidence * 15).astype(int), 14)
        independent['ece'] = sum(np.mean(bins == i) * abs(correct[bins == i].mean() -
                                    confidence[bins == i].mean())
                                 for i in range(15) if np.any(bins == i))
        recorded = json.loads((folder / 'test_metrics.json').read_text())
        assert recorded['signature'] == checkpoint['signature']
        for metric, value in independent.items():
            np.testing.assert_allclose(getattr(row, metric), value, rtol=2e-5, atol=1e-5,
                                       err_msg=f'{row.dataset}/{row.method}/{metric}')
            np.testing.assert_allclose(recorded['metrics'][metric], value, rtol=2e-5, atol=1e-5)
        print(f'{row.dataset}/{row.method}: 30 epochs, selected epoch {row.best_epoch}, '
              f'{len(y)} test slides, all metrics independently verified.')

    nb = nbformat.read(ROOT / 'ECE1512_ProjectA_Student.ipynb', as_version=4)
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == 'error' for cell in code for output in cell.outputs)
    assert all('TODO(' not in cell.source for cell in code)
    print(f'All {len(code)} code cells executed; no errors or incomplete implementations.')
    for slide_id in ('test_067', 'test_068'):
        with np.load(RESULTS / 'C16_baseline' / f'{slide_id}_attention.npz') as attention:
            assert attention['coords'].shape == (len(attention['attention']), 2)
            assert np.isfinite(attention['attention']).all()
            assert (attention['attention'] >= 0).all()
            np.testing.assert_allclose(attention['attention'].sum(), 1, atol=1e-5)
            np.testing.assert_allclose(attention['probabilities'].sum(), 1, atol=1e-5)
    overlays = [cell for cell in code if cell.source.startswith('attention_visualizations = {}')]
    assert len(overlays) == 1
    assert sum('image/png' in output.get('data', {}) for output in overlays[0].outputs) == 2
    print('Both WSI overlays and their raw attention arrays are present and valid.')


if __name__ == '__main__':
    verify()
