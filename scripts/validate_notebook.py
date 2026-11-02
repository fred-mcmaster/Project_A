"""Check the educational notebook's baseline model and training definitions."""
import ast
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.work' / 'matplotlib'))
import nbformat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import mil_collate_fn
from research_extensions.experiment_utils import prepare_dataset
from utils import Struct, classification_metrics

torch.set_num_threads(4)
nb = nbformat.read(ROOT / 'ECE1512_ProjectA_Student.ipynb', as_version=4)
nbformat.validate(nb)
namespace = dict(torch=torch, nn=nn, F=torch.nn.functional,
                 classification_metrics=classification_metrics)
names = {'Classifier1FC', 'DimReduction', 'AttentionGated', 'ABMIL',
         'train_one_epoch', 'evaluate'}
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    # Notebook installation and magic commands are not Python definitions.
    source = '\n'.join(line for line in cell.source.splitlines()
                       if not line.lstrip().startswith(('!', '%')))
    tree = ast.parse(source)
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == 'TODO' for node in ast.walk(tree)):
        raise AssertionError('Unimplemented TODO call remains.')
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if definitions:
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook>', 'exec'), namespace)
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
        # The educational model uses full single-slide bags without padding masks.
        self.loader = DataLoader(self.samples, batch_size=1, collate_fn=mil_collate_fn)

    def test_attention_and_permutation(self):
        self.model.eval()
        for sample in self.samples:
            bag = sample['input'].unsqueeze(0)
            logits, attention = self.model(bag, return_attention=True)
            torch.testing.assert_close(attention.sum(-1), torch.ones(1, 1))
            self.assertTrue((attention >= 0).all())
            order = torch.randperm(bag.shape[1])
            torch.testing.assert_close(logits, self.model(bag[:, order]))

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
        after = evaluate(self.model, criterion, self.loader, 'cpu', self.conf)
        self.assertFalse(self.model.training)
        self.assertLess(after['loss'], before['loss'] * 0.1)
        self.assertEqual(after['acc'], 100.0)
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(saved[k], v)
        with torch.no_grad():
            probabilities = torch.cat([self.model(s['input'].unsqueeze(0)).softmax(1)
                                       for s in self.samples]).numpy()
        expected = -np.log(probabilities[np.arange(3), np.arange(3)]).mean()
        self.assertAlmostEqual(after['loss'], expected, places=6)

    def test_real_datasets_and_forward_backward(self):
        import yaml
        for config in (ROOT / 'config').glob('*.yml'):
            feature = Path(yaml.safe_load(config.read_text())['data_dir'])
            if not (feature if feature.is_absolute() else ROOT / feature).is_file():
                self.skipTest('Download all three feature files for real-data integration.')
        for dataset in ('C16', 'C17', 'BRACS'):
            conf, datasets = prepare_dataset(ROOT, dataset, 42)
            groups = [set(ds.data_names) for ds in datasets]
            self.assertTrue(all(groups))
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            model = ABMIL(conf)
            batch = mil_collate_fn([datasets[0][0]])
            loss = nn.CrossEntropyLoss()(model(batch['input']), batch['label'])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
