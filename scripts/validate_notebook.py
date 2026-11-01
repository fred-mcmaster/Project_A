"""Check the notebook's actual model/training definitions and supplied real datasets."""
import ast
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['MPLCONFIGDIR'] = str(ROOT / '.work' / 'matplotlib')
import nbformat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from utils import Struct, load_config, classification_metrics

torch.set_num_threads(4)
nb = nbformat.read(ROOT / 'ECE1512_ProjectA_Student.ipynb', as_version=4)
nbformat.validate(nb)
namespace = {}
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'TODO'
           for node in ast.walk(tree)):
        raise AssertionError('Unimplemented TODO call remains.')
    if cell.source.startswith(('import os, sys', 'class Classifier1FC', 'def train_one_epoch',
                               'def calibration_metrics')):
        exec(compile(tree, '<notebook>', 'exec'), namespace)
namespace['classification_metrics'] = classification_metrics
ABMIL = namespace['ABMIL']
train_one_epoch = namespace['train_one_epoch']
evaluate = namespace['evaluate']


class NotebookChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.conf = Struct(D_feat=4, D_inner=12, D_attn=6, n_class=3, droprate=0.0)
        self.model = ABMIL(self.conf)
        self.samples = [dict(input=torch.eye(4)[i].repeat(i + 2, 1), label=i, slide_id=str(i))
                        for i in range(3)]
        self.loader = DataLoader(self.samples, batch_size=2, collate_fn=mil_collate_fn)

    def test_padding_and_permutation(self):
        self.model.eval()
        batch = mil_collate_fn(self.samples)
        logits, attention = self.model(batch['input'], mask=batch['mask'], return_attention=True)
        torch.testing.assert_close(attention.sum(-1), torch.ones(3, 1))
        self.assertTrue((attention.squeeze(1)[~batch['mask']] == 0).all())
        for i, sample in enumerate(self.samples):
            torch.testing.assert_close(logits[i:i + 1], self.model(sample['input'].unsqueeze(0)))
        batch['input'][~batch['mask']] = float('nan')
        torch.testing.assert_close(logits, self.model(batch['input'], mask=batch['mask']))
        order = torch.randperm(batch['input'].shape[1])
        torch.testing.assert_close(logits, self.model(batch['input'][:, order], mask=batch['mask'][:, order]))
        with self.assertRaises(ValueError):
            self.model(batch['input'], mask=torch.zeros_like(batch['mask']))

    def test_training_learns_and_evaluation_is_read_only(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.03)
        before = evaluate(self.model, criterion, self.loader, 'cpu', self.conf)
        for _ in range(60):
            train_one_epoch(self.model, criterion, self.loader, optimizer, 'cpu')
        self.assertTrue(self.model.training)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))
        saved = {k: v.clone() for k, v in self.model.state_dict().items()}
        after, predictions = evaluate(self.model, criterion, self.loader, 'cpu', self.conf, True)
        self.assertFalse(self.model.training)
        self.assertLess(after['loss'], before['loss'] * 0.1)
        self.assertEqual(after['acc'], 100.0)
        self.assertEqual(list(predictions.slide_id), ['0', '1', '2'])
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(saved[k], v)
        expected = -np.log(predictions.filter(regex=r'^p_\d+$').to_numpy()[np.arange(3), np.arange(3)]).mean()
        self.assertAlmostEqual(after['loss'], expected, places=6)
        with self.assertRaises(ValueError):
            train_one_epoch(self.model, criterion, [], optimizer, 'cpu')
        with self.assertRaises(ValueError):
            evaluate(self.model, criterion, [], 'cpu', self.conf)

    def test_smoothing_and_calibration(self):
        logits = torch.tensor([[2., -1., 0.], [0.3, 0.4, 0.5]])
        labels = torch.tensor([0, 2])
        target = 0.9 * torch.nn.functional.one_hot(labels, 3) + 0.1 / 3
        expected = -(target * logits.log_softmax(1)).sum(1).mean()
        torch.testing.assert_close(nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels), expected)
        probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
        metrics = namespace['calibration_metrics'](probabilities, torch.tensor([0, 0]))
        self.assertAlmostEqual(metrics['ece'], 0.4, places=6)
        self.assertAlmostEqual(metrics['brier'], 0.4, places=6)

    def test_real_datasets_and_forward_backward(self):
        for config_path in sorted((ROOT / 'config').glob('*.yml')):
            conf = load_config(config_path)
            datasets = build_HDF5_feat_dataset(conf.data_dir, conf)
            sets = [set(ds.data_names) for ds in datasets]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertTrue(all(len(ds) > 0 for ds in datasets))
            second = build_HDF5_feat_dataset(conf.data_dir, conf)
            self.assertEqual([ds.data_names for ds in datasets], [ds.data_names for ds in second])
            conf.D_feat = datasets[0].feature_dim
            model = ABMIL(conf)
            batch = mil_collate_fn([datasets[0][0], datasets[0][1]])
            logits = model(batch['input'], mask=batch['mask'])
            loss = nn.CrossEntropyLoss()(logits, batch['label'])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            print(f'{conf.dataset}: splits {[len(ds) for ds in datasets]}, '
                  f'batch {tuple(batch["input"].shape)}, loss {loss.item():.4f}', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
