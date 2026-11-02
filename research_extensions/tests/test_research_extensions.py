"""Checks of the exported model, selected objective, and experiment controls."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import Struct, seed_everything
from dataloader import mil_collate_fn
from research_extensions.abmil_model import ABMIL
from research_extensions.training_engine import make_loaders, train_one_epoch, evaluate
from research_extensions.label_smoothing_loss import make_training_loss, make_evaluation_loss
from research_extensions.experiment_utils import prepare_dataset, selected_epoch, config_from_checkpoint
from research_extensions.evaluate_extension import metrics_from_predictions, read_run, main as evaluate_main

ROOT=Path(os.environ.get('ECE1512_PROJECT_DIR',Path(__file__).resolve().parents[2]))
torch.set_num_threads(4)


class ResearchExtensionTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.conf=Struct(D_feat=4,D_inner=12,D_attn=6,n_class=3,droprate=0.0,B=2,n_worker=0,pin_memory=False,seed=42)
        self.samples=[dict(input=torch.eye(4)[i].repeat(i+2,1),label=i,slide_id=str(i)) for i in range(3)]
        self.loader=DataLoader(self.samples,batch_size=2,collate_fn=mil_collate_fn)

    def test_exported_definitions_match_notebook(self):
        nb=json.loads((ROOT/'ECE1512_ProjectA_Student.ipynb').read_text(encoding='utf-8'))
        notebook_nodes={}
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                for node in ast.parse(''.join(cell['source'])).body:
                    if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                        notebook_nodes[node.name]=ast.dump(node,include_attributes=False)
        package=Path(__file__).resolve().parents[1]
        for filename in ('abmil_model.py','training_engine.py'):
            for node in ast.parse((package/filename).read_text(encoding='utf-8-sig')).body:
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                    self.assertEqual(ast.dump(node,include_attributes=False),notebook_nodes[node.name],node.name)

    def test_smoothing_matches_soft_targets_and_gradients(self):
        for classes in (2,4,7):
            logits=torch.randn(classes,classes,requires_grad=True)
            labels=torch.arange(classes)
            target=.9*torch.nn.functional.one_hot(labels,classes)+.1/classes
            manual=-(target*logits.log_softmax(1)).sum(1).mean()
            actual=make_training_loss('label_smoothing',.1)(logits,labels)
            torch.testing.assert_close(actual,manual)
            torch.testing.assert_close(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(manual,logits)[0])
            torch.testing.assert_close(make_training_loss('label_smoothing',0)(logits,labels),make_training_loss('baseline')(logits,labels))
        for epsilon in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): make_training_loss('label_smoothing',epsilon)
        with self.assertRaises(ValueError): make_training_loss('unknown')

    def test_padding_and_permutation(self):
        model=ABMIL(self.conf).eval()
        batch=mil_collate_fn(self.samples)
        logits,attention=model(batch['input'],mask=batch['mask'],return_attention=True)
        torch.testing.assert_close(attention.sum(-1),torch.ones(3,1))
        self.assertTrue((attention.squeeze(1)[~batch['mask']]==0).all())
        batch['input'][~batch['mask']]=float('nan')
        torch.testing.assert_close(logits,model(batch['input'],mask=batch['mask']))
        perm=torch.randperm(batch['input'].shape[1])
        torch.testing.assert_close(logits,model(batch['input'][:,perm],mask=batch['mask'][:,perm]))
        with self.assertRaises(ValueError): model(batch['input'],mask=torch.zeros_like(batch['mask']))

    def test_smoothed_training_learns_and_evaluation_is_read_only(self):
        model=ABMIL(self.conf)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.03)
        before=evaluate(model,make_evaluation_loss(),self.loader,'cpu',self.conf)
        for _ in range(60):
            train_one_epoch(model,make_training_loss('label_smoothing'),self.loader,optimizer,'cpu')
        weights={name:value.clone() for name,value in model.state_dict().items()}
        after,pred=evaluate(model,make_evaluation_loss(),self.loader,'cpu',self.conf,True)
        self.assertLess(after['loss'],before['loss']*.25)
        self.assertEqual(after['acc'],100)
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for name,value in model.state_dict().items(): torch.testing.assert_close(value,weights[name])
        manual=-np.log(pred.filter(regex=r'^p_\d+$').to_numpy()[np.arange(3),np.arange(3)]).mean()
        self.assertAlmostEqual(after['loss'],manual,places=6)

    def test_paired_initialization_and_slide_order(self):
        states,orders=[],[]
        for method in ('baseline','label_smoothing'):
            seed_everything(42)
            loaders=make_loaders(self.samples,self.samples,self.samples,self.conf)
            model=ABMIL(self.conf)
            make_training_loss(method)
            states.append({name:value.clone() for name,value in model.state_dict().items()})
            orders.append([slide for batch in loaders[0] for slide in batch['slide_id']])
        self.assertEqual(orders[0],orders[1])
        for name in states[0]: torch.testing.assert_close(states[0][name],states[1][name])

    def test_selection_uses_validation_score_and_earliest_tie(self):
        history=pd.DataFrame(dict(epoch=[0,1,2],val_auc=[.7,.9,.9],val_f1=[.5,.8,.8],val_loss=[.1,.6,.4]))
        self.assertEqual(selected_epoch(history),1)

    def test_probability_metrics(self):
        frame=pd.DataFrame(dict(label=[0,1,1],prediction=[0,1,0],p_0=[.8,.4,.9],p_1=[.2,.6,.1]))
        metrics=metrics_from_predictions(frame,2)
        self.assertAlmostEqual(metrics['acc'],200/3)
        self.assertAlmostEqual(metrics['auc'],.5)
        self.assertAlmostEqual(metrics['f1'],2/3)
        self.assertAlmostEqual(metrics['ece'],.5)
        self.assertAlmostEqual(metrics['brier'],2.02/3)
        self.assertAlmostEqual(metrics['loss'],-np.log([.8,.6,.1]).mean())

    def test_real_feature_bags_all_datasets(self):
        import yaml
        for config in (ROOT/'config').glob('*.yml'):
            feature = Path(yaml.safe_load(config.read_text())['data_dir'])
            if not (feature if feature.is_absolute() else ROOT/feature).is_file():
                self.skipTest('Download all three feature files to run the real-data integration check.')
        for dataset in ('C16','C17','BRACS'):
            conf,datasets=prepare_dataset(ROOT,dataset,42)
            batch=mil_collate_fn([datasets[0][0],datasets[0][1]])
            model=ABMIL(conf)
            logits=model(batch['input'],mask=batch['mask'])
            loss=make_training_loss('label_smoothing')(logits,batch['label'])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_paths_relocate_without_changing_model_settings(self):
        checkpoint = dict(config=dict(dataset='CAMELYON17',data_dir='/old/machine/features.h5',
            dataset_csv_dir='/old/machine/metadata',camelyon17_csv='/old/machine/meta.csv',
            D_feat=384,D_inner=256,D_attn=128,n_class=4,seed=42,lr=1e-4))
        conf = config_from_checkpoint(checkpoint, ROOT)
        self.assertEqual(conf.data_dir, str((ROOT/'feature/BRACS/C17/patch_feats_pretrain_medical_ssl.h5').resolve()))
        self.assertEqual(conf.dataset_csv_dir, str((ROOT/'dataset_csv').resolve()))
        self.assertFalse(hasattr(conf, 'camelyon17_csv'))
        self.assertEqual((conf.D_feat,conf.n_class,conf.seed,conf.lr),(384,4,42,1e-4))
        self.assertEqual(checkpoint['config']['data_dir'], '/old/machine/features.h5')

    def test_bundled_artifacts_match_checkpoints_and_results_table(self):
        comparison=pd.read_csv(ROOT/'research_extensions/evaluation_results/extension_comparison.csv')
        self.assertEqual(len(comparison),6)
        for row in comparison.itertuples():
            run,saved,history,splits,conf,checkpoint=read_run(ROOT,
                ROOT/'research_extensions/evaluation_results',row.dataset,row.method,ROOT/'checkpoints')
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(len(history),30)
            self.assertEqual(saved['best_epoch']+1,row.best_epoch)
            self.assertEqual(conf.seed,row.seed)
            predictions=pd.read_csv(run/'test_predictions.csv')
            test=splits[splits.split.eq('test')]
            self.assertEqual(predictions.slide_id.tolist(),test.slide_id.tolist())
            self.assertEqual(predictions.label.tolist(),test.label.tolist())
            metrics=metrics_from_predictions(predictions,int(conf.n_class))
            recorded=json.loads((run/'test_metrics.json').read_text())
            for key,value in metrics.items():
                np.testing.assert_allclose(value,getattr(row,key),rtol=2e-5,atol=1e-5)
                np.testing.assert_allclose(value,recorded[key],rtol=2e-5,atol=1e-5)

    def test_single_row_evaluation_requires_no_feature_access(self):
        with tempfile.TemporaryDirectory() as output, patch(
                'research_extensions.evaluate_extension.build_HDF5_feat_dataset',
                side_effect=AssertionError('Saved-prediction verification must not read features.')):
            evaluate_main(['--project-dir',str(ROOT),'--datasets','C16','--methods','baseline',
                           '--device','cpu','--output-dir',output])
            table=pd.read_csv(Path(output)/'extension_comparison.csv')
            self.assertEqual(len(table),1)
            self.assertEqual(table.iloc[0]['method'],'baseline')
            self.assertEqual(len(pd.read_csv(Path(output)/'extension_deltas.csv')),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
