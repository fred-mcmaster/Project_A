"""Train the ABMIL baseline with validation-only checkpoint selection.

Run from the project folder:
    python -m research_extensions.train_extension
Use a separate --output-dir for debugging or new hyperparameters.
Final test evaluation is handled by evaluate_extension.py after model selection.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from utils import seed_everything
from research_extensions.abmil_model import ABMIL
from research_extensions.training_engine import make_loaders, train_one_epoch, evaluate
from research_extensions.experiment_utils import (
    CONFIG_FILES, load_settings, resolve_path, prepare_dataset, split_records,
    experiment_manifest, save_checkpoint,
)


def fit(model, training_loss, optimizer, train_loader, val_loader, conf, epochs, device, run_dir, manifest, overwrite=False):
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / 'best.pt'
    if checkpoint_path.exists() and not overwrite:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved.get('signature') == manifest['signature'] and saved.get('epochs_completed') == epochs:
            print(f'Reusing completed {run_dir.name}: {epochs} epochs, best epoch {saved["best_epoch"]+1}', flush=True)
            return saved
        raise FileExistsError(f'{run_dir} contains a different experiment. Choose a new --output-dir or use --overwrite.')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    best_score, best_epoch, best_state = float('-inf'), -1, None
    history = []
    eval_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        started = time.monotonic()
        train_metrics = train_one_epoch(model, training_loss, train_loader, optimizer, device)
        val_metrics = evaluate(model, eval_loss, val_loader, device, conf)
        score = val_metrics['auc'] + val_metrics['f1']
        if not np.isfinite(score):
            raise FloatingPointError('Non-finite validation selection score.')
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
        row = dict(epoch=epoch, train_loss=train_metrics['loss'], train_acc=train_metrics['acc'],
                   **{f'val_{key}':value for key,value in val_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        print(f'{run_dir.name} {epoch+1:02d}/{epochs} | train loss={row["train_loss"]:.4f} | '
              f'val AUC={row["val_auc"]:.4f} F1={row["val_f1"]:.4f} | best={best_epoch+1} | '
              f'{time.monotonic()-started:.1f}s', flush=True)
    if best_state is None:
        raise ValueError('At least one epoch is required.')
    payload = dict(model_state=best_state, best_epoch=best_epoch, best_score=best_score,
                   epochs_completed=epochs, history=history, config=vars(conf),
                   signature=manifest['signature'], method=manifest['method'])
    save_checkpoint(checkpoint_path, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('experiment_config.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/research_extension_runs'))
    parser.add_argument('--datasets', nargs='+', choices=list(CONFIG_FILES))
    parser.add_argument('--methods', nargs='+', choices=['baseline'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--overwrite', action='store_true', help='Replace a generated run with changed settings.')
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    epochs = args.epochs if args.epochs is not None else int(settings['epochs'])
    seed = args.seed if args.seed is not None else int(settings['seed'])
    if epochs <= 0:
        parser.error('--epochs must be positive.')
    datasets_to_run = args.datasets or settings['datasets']
    methods = args.methods or settings['methods']
    root = args.project_dir.resolve()
    output = resolve_path(root, args.output_dir).resolve()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    for dataset in datasets_to_run:
        conf, datasets = prepare_dataset(root, dataset, seed)
        conf.train_epoch = epochs
        table = split_records(conf, datasets)
        for method in methods:
            manifest = experiment_manifest(conf, datasets, method, epochs, device)
            # Reset model initialization and the independent sampler seed per run.
            seed_everything(seed)
            loaders = make_loaders(*datasets, conf)
            model = ABMIL(conf).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))
            fit(model, torch.nn.CrossEntropyLoss(), optimizer, loaders[0], loaders[1],
                conf, epochs, device, output / f'{dataset}_{method}', manifest, args.overwrite)
            del model, optimizer, loaders
        split_dir = output / dataset
        split_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(split_dir / 'splits.csv', index=False)
    print(f'Training/model selection complete. Checkpoints: {output}', flush=True)
    print('Run evaluate_extension.py with --runs-dir pointing to this folder for final test evaluation.')


if __name__ == '__main__':
    main()
