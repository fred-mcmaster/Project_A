"""Compare frozen baseline and label-smoothing checkpoints on all three datasets.

Defaults to the six original notebook runs in results/. Existing predictions are
verified and reused. --recompute reruns inference with the exported ABMIL model
and checks the predictions against the saved files; it never reselects an epoch.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ece1512_research_matplotlib'))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, precision_recall_fscore_support, confusion_matrix
from dataloader import build_HDF5_feat_dataset
from research_extensions.abmil_model import ABMIL
from research_extensions.training_engine import make_loaders, evaluate
from research_extensions.label_smoothing_loss import make_evaluation_loss
from research_extensions.experiment_utils import (
    CONFIG_FILES, DATASET_NAMES, SPLIT_NAMES, config_from_checkpoint, resolve_path,
    selected_epoch, sha256_file,
)


def metrics_from_predictions(frame, n_classes, n_bins=15):
    columns = [f'p_{c}' for c in range(n_classes)]
    p = frame[columns].to_numpy(dtype=np.float64)
    y = frame.label.to_numpy(dtype=np.int64)
    if len(p) == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-6).any():
        raise ValueError('Probabilities must be nonempty, finite, and between zero and one.')
    if (y < 0).any() or (y >= n_classes).any():
        raise ValueError('Labels are outside the configured class range.')
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    predicted = p.argmax(1)
    np.testing.assert_array_equal(predicted, frame.prediction)
    confidence, correct = p.max(1), predicted == y
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = sum((bins == i).mean() * abs(correct[bins == i].mean() - confidence[bins == i].mean())
              for i in range(n_bins) if np.any(bins == i))
    return dict(auc=roc_auc_score(y, p[:, 1]) if n_classes == 2 else
                roc_auc_score(y, p, average='macro', multi_class='ovr'),
                acc=100 * accuracy_score(y, predicted),
                f1=f1_score(y, predicted, labels=np.arange(n_classes), average='macro', zero_division=0),
                loss=log_loss(y, p / p.sum(1, keepdims=True), labels=np.arange(n_classes)),
                brier=float(np.square(p - np.eye(n_classes)[y]).sum(1).mean()), ece=float(ece))


def read_run(root, runs_dir, dataset, method):
    run_dir = runs_dir / f'{dataset}_{method}'
    checkpoint_path = run_dir / 'best.pt'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    history = pd.read_csv(run_dir / 'history.csv')
    split_table = pd.read_csv(runs_dir / dataset / 'splits.csv')
    if len(history) != int(saved['epochs_completed']):
        raise ValueError(f'Incomplete history: {run_dir}')
    np.testing.assert_array_equal(history.epoch, np.arange(len(history)))
    if selected_epoch(history) != int(saved['best_epoch']):
        raise ValueError(f'Checkpoint is not the validation-selected epoch: {run_dir}')
    if saved.get('method', method) != method:
        raise ValueError(f'Checkpoint method does not match directory: {run_dir}')
    conf = config_from_checkpoint(saved, root)
    if conf.dataset != DATASET_NAMES[dataset]:
        raise ValueError('Dataset identifier does not match the checkpoint configuration.')
    groups = [set(split_table.loc[split_table.split.eq(split), 'slide_id']) for split in SPLIT_NAMES]
    if not all(groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Saved train/validation/test splits must be nonempty and disjoint.')
    if split_table.slide_id.duplicated().any():
        raise ValueError('Duplicate slide IDs in saved splits.')
    return run_dir, saved, history, split_table, conf


def check_pair(pair):
    left, right = pair['baseline'], pair['label_smoothing']
    for field in ('dataset','data_dir','D_feat','D_inner','D_attn','n_class','B','n_shot','seed','lr','wd','droprate'):
        if getattr(left['conf'], field) != getattr(right['conf'], field):
            raise ValueError(f'Uncontrolled baseline/extension difference: {field}')
    if left['saved']['epochs_completed'] != right['saved']['epochs_completed']:
        raise ValueError('Paired runs must have the same epoch budget.')
    pd.testing.assert_frame_equal(left['splits'], right['splits'])


def plots(output, records, datasets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'baseline':'#176B9E', 'label_smoothing':'#D66B27'}
    fig, axes = plt.subplots(2, len(datasets), figsize=(5*len(datasets),8), squeeze=False)
    for col, dataset in enumerate(datasets):
        axes[1,col].plot([0,1],[0,1],'--',color='gray',label='ideal')
        for method in ('baseline','label_smoothing'):
            record = records[(dataset,method)]
            history, pred = record['history'], record['predictions']
            axes[0,col].plot(history.epoch+1,history.val_loss,label=method,color=colors[method])
            p = pred.filter(regex=r'^p_\d+$').to_numpy()
            confidence, correct = p.max(1), p.argmax(1) == pred.label.to_numpy()
            bins = np.minimum((confidence*15).astype(int),14)
            rows = [dict(bin=i,mean_confidence=confidence[bins==i].mean(),accuracy=correct[bins==i].mean(),count=int((bins==i).sum()))
                    for i in range(15) if (bins==i).any()]
            binned = pd.DataFrame(rows)
            binned.to_csv(output/f'{dataset}_{method}'/'reliability_bins.csv',index=False)
            axes[1,col].plot(binned.mean_confidence,binned.accuracy,marker='o',label=method,color=colors[method])
        axes[0,col].set(title=f'{dataset}: validation NLL',xlabel='Epoch',ylabel='Hard-target CE')
        axes[1,col].set(title=f'{dataset}: test reliability',xlabel='Mean confidence',ylabel='Empirical accuracy',xlim=(0,1),ylim=(0,1))
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'extension_learning_and_calibration.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(2,len(datasets),figsize=(5*len(datasets),9),squeeze=False,layout='constrained')
    for col,dataset in enumerate(datasets):
        for row,method in enumerate(('baseline','label_smoothing')):
            record=records[(dataset,method)]
            pred,n_class=record['predictions'],int(record['conf'].n_class)
            count=confusion_matrix(pred.label,pred.prediction,labels=np.arange(n_class))
            normalized=count/np.maximum(count.sum(1,keepdims=True),1)
            ax=axes[row,col]
            image=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
            for i in range(n_class):
                for j in range(n_class):
                    ax.text(j,i,f'{normalized[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if normalized[i,j]>.5 else 'black')
            ax.set(title=f'{dataset}: {method}',xlabel='Predicted class',ylabel='True class',xticks=range(n_class),yticks=range(n_class))
            pd.DataFrame(count).to_csv(output/f'{dataset}_{method}'/'confusion_counts.csv',index=False)
    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.7,label='Fraction of true class')
    fig.savefig(output/'extension_confusion_matrices.png',dpi=180)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runs-dir',type=Path,default=Path('results'))
    parser.add_argument('--output-dir',type=Path,default=Path('research_extensions/evaluation_results'))
    parser.add_argument('--datasets',nargs='+',choices=list(CONFIG_FILES),default=list(CONFIG_FILES))
    parser.add_argument('--recompute',action='store_true',help='Reproduce checkpoint inference and compare to any saved predictions.')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args(argv)
    root=args.project_dir.resolve()
    runs_dir=resolve_path(root,args.runs_dir).resolve()
    output=resolve_path(root,args.output_dir).resolve()
    if output == runs_dir:
        parser.error('Use a separate output directory to preserve source experiments.')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    output.mkdir(parents=True,exist_ok=True)
    records,per_class,provenance={},[],[]
    rows=[]
    for dataset in args.datasets:
        pair={}
        for method in ('baseline','label_smoothing'):
            run_dir,saved,history,split_table,conf=read_run(root,runs_dir,dataset,method)
            pair[method]=dict(saved=saved,history=history,splits=split_table,conf=conf,run_dir=run_dir)
        check_pair(pair)
        for method,record in pair.items():
            run_dir,saved,conf=record['run_dir'],record['saved'],record['conf']
            original_path=run_dir/'test_predictions.csv'
            original=pd.read_csv(original_path) if original_path.exists() else None
            maximum_difference=None
            if args.recompute or original is None:
                datasets=build_HDF5_feat_dataset(conf.data_dir,conf)
                for split,ds in zip(SPLIT_NAMES,datasets):
                    expected=record['splits'].loc[record['splits'].split.eq(split),'slide_id'].tolist()
                    if list(ds.data_names)!=expected:
                        raise ValueError(f'Current split differs from saved experiment: {dataset}/{split}')
                loader=make_loaders(*datasets,conf)[2]
                model=ABMIL(conf).to(device)
                model.load_state_dict(saved['model_state'],strict=True)
                _,predictions=evaluate(model,make_evaluation_loss(),loader,device,conf,return_predictions=True)
                del model,loader,datasets
                mode='checkpoint_inference'
                if original is not None:
                    pd.testing.assert_frame_equal(predictions[['slide_id','label','prediction']],original[['slide_id','label','prediction']])
                    fresh=predictions.filter(regex=r'^p_\d+$').to_numpy()
                    old=original.filter(regex=r'^p_\d+$').to_numpy()
                    np.testing.assert_allclose(fresh,old,rtol=2e-5,atol=2e-6)
                    maximum_difference=float(np.abs(fresh-old).max())
            else:
                predictions=original
                mode='verified_saved_predictions'
            expected=record['splits'][record['splits'].split.eq('test')]
            if predictions.slide_id.tolist()!=expected.slide_id.tolist() or predictions.label.tolist()!=expected.label.tolist():
                raise ValueError('Test predictions do not match saved test IDs/labels.')
            metrics=metrics_from_predictions(predictions,int(conf.n_class))
            recorded_path=run_dir/'test_metrics.json'
            if recorded_path.exists():
                recorded=json.loads(recorded_path.read_text())
                if recorded['signature']!=saved['signature']:
                    raise ValueError('Saved test metrics and checkpoint signatures differ.')
                for key,value in metrics.items():
                    np.testing.assert_allclose(value,recorded['metrics'][key],rtol=2e-5,atol=1e-5)
            target=output/f'{dataset}_{method}'
            target.mkdir(exist_ok=True)
            predictions.to_csv(target/'test_predictions.csv',index=False)
            record['history'].to_csv(target/'history.csv',index=False)
            record['splits'].to_csv(target/'splits.csv',index=False)
            (target/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            epsilon=float(saved.get('label_smoothing',0.0 if method=='baseline' else 0.1))
            rows.append(dict(dataset=dataset,method=method,epochs=int(saved['epochs_completed']),
                             best_epoch=int(saved['best_epoch'])+1,seed=int(conf.seed),label_smoothing=epsilon,**metrics))
            precision,recall,f1,support=precision_recall_fscore_support(predictions.label,predictions.prediction,
                labels=np.arange(int(conf.n_class)),zero_division=0)
            for cls in range(int(conf.n_class)):
                per_class.append(dict(dataset=dataset,method=method,class_id=cls,precision=float(precision[cls]),
                                      recall=float(recall[cls]),f1=float(f1[cls]),support=int(support[cls])))
            record['predictions']=predictions
            records[(dataset,method)]=record
            provenance.append(dict(dataset=dataset,method=method,mode=mode,source_checkpoint=str(run_dir/'best.pt'),
                source_checkpoint_sha256=sha256_file(run_dir/'best.pt'),training_signature=saved['signature'],
                legacy_notebook_checkpoint=('method' not in saved),label_smoothing=epsilon,
                saved_prediction_max_abs_difference=maximum_difference,
                test_count=len(predictions),selected_epoch=int(saved['best_epoch'])+1,epochs_completed=int(saved['epochs_completed'])))
            print(f'{dataset}/{method}: AUC={metrics["auc"]:.4f} F1={metrics["f1"]:.4f} '
                  f'accuracy={metrics["acc"]:.2f}% | {mode}',flush=True)
    comparison=pd.DataFrame(rows)
    comparison.to_csv(output/'extension_comparison.csv',index=False)
    pd.DataFrame(per_class).to_csv(output/'extension_per_class_metrics.csv',index=False)
    differences=[]
    indexed=comparison.set_index(['dataset','method'])
    for dataset in args.datasets:
        left,right=indexed.loc[(dataset,'baseline')],indexed.loc[(dataset,'label_smoothing')]
        differences.append(dict(dataset=dataset,**{key:float(right[key]-left[key]) for key in ('auc','f1','acc','loss','brier','ece')}))
    pd.DataFrame(differences).to_csv(output/'extension_deltas.csv',index=False)
    plots(output,records,args.datasets)
    payload=dict(protocol='Fixed checkpoints selected by validation AUROC + F1; no test-based tuning.',
                 ece_bins=15,brier_definition='Mean over slides of sum over classes of squared probability error.',
                 torch_version=str(torch.__version__),device=str(device),
                 gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                 implementation_sha256={p.name:sha256_file(p) for p in Path(__file__).resolve().parent.glob('*.py')},runs=provenance)
    (output/'evaluation_provenance.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(comparison.to_string(index=False))
    print(f'Evaluation artifacts saved to {output}',flush=True)


if __name__=='__main__':
    main()

