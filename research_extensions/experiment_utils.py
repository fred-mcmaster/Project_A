"""Configuration, split recording, checkpoint selection, and reproducibility."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import h5py
import pandas as pd
import torch
from utils import load_config, infer_feature_dim, Struct
from dataloader import build_HDF5_feat_dataset, _map_label

CONFIG_FILES = {'C16':'camelyon16_medical_ssl_config.yml',
                'C17':'camelyon17_medical_ssl_config.yml',
                'BRACS':'bracs_medical_ssl_config.yml'}
DATASET_NAMES = {'C16':'CAMELYON16', 'C17':'CAMELYON17', 'BRACS':'BRACS'}
SPLIT_NAMES = ('train', 'validation', 'test')

def resolve_path(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path

def load_settings(path):
    settings = json.loads(Path(path).read_text(encoding='utf-8'))
    if not settings['datasets'] or not set(settings['datasets']) <= set(CONFIG_FILES):
        raise ValueError('datasets must be a nonempty subset of C16, C17, BRACS.')
    if not settings['methods'] or not set(settings['methods']) <= {'baseline','label_smoothing'}:
        raise ValueError('methods must be baseline and/or label_smoothing.')
    if len(set(settings['datasets'])) != len(settings['datasets']) or len(set(settings['methods'])) != len(settings['methods']):
        raise ValueError('Duplicate datasets/methods are not allowed.')
    if not 0 <= float(settings['label_smoothing']) <= 1:
        raise ValueError('label_smoothing must be between 0 and 1.')
    if int(settings['epochs']) <= 0:
        raise ValueError('epochs must be positive.')
    return settings

def resolve_conf_paths(conf, root):
    conf.data_dir = str(resolve_path(root, conf.data_dir).resolve())
    conf.dataset_csv_dir = str(resolve_path(root, getattr(conf, 'dataset_csv_dir', 'dataset_csv')).resolve())
    for name in ('camelyon17_csv', 'bracs_csv'):
        if getattr(conf, name, None):
            setattr(conf, name, str(resolve_path(root, getattr(conf, name)).resolve()))
    return conf

def prepare_dataset(root, dataset, seed=None):
    conf = resolve_conf_paths(load_config(Path(root) / 'config' / CONFIG_FILES[dataset]), root)
    if seed is not None:
        conf.seed = int(seed)
    datasets = build_HDF5_feat_dataset(conf.data_dir, conf)
    conf.D_feat = infer_feature_dim(datasets[0])
    return conf, datasets

def config_from_checkpoint(checkpoint, root):
    # Keep the trained architecture/protocol, but use this checkout's data paths.
    # This also relocates fresh CLI checkpoints that stored absolute paths.
    conf = Struct(**checkpoint['config'])
    dataset = next(key for key, name in DATASET_NAMES.items() if name == conf.dataset)
    current = load_config(Path(root) / 'config' / CONFIG_FILES[dataset])
    for name in ('data_dir', 'dataset_csv_dir', 'camelyon17_csv', 'bracs_csv'):
        conf.__dict__.pop(name, None)
        if hasattr(current, name):
            setattr(conf, name, getattr(current, name))
    return resolve_conf_paths(conf, root)


def environment_info(device):
    packages = ('torch', 'torchmetrics', 'h5py', 'numpy', 'pandas', 'scikit-learn',
                'PyYAML', 'matplotlib', 'requests')
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in packages},
                device=str(device), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None)

def split_records(conf, datasets):
    groups = [set(ds.data_names) for ds in datasets]
    if not all(groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Splits must be nonempty and disjoint at slide level.')
    rows = []
    with h5py.File(conf.data_dir, 'r') as h5:
        for split, ds in zip(SPLIT_NAMES, datasets):
            for slide_id in ds.data_names:
                group = h5[slide_id]
                label = _map_label(int(group.attrs['label']), conf.dataset, conf.n_class)
                if not 0 <= label < int(conf.n_class):
                    raise ValueError(f'Invalid label for {slide_id}: {label}')
                rows.append(dict(split=split, slide_id=slide_id, label=label, patches=int(group['feat'].shape[0])))
    return pd.DataFrame(rows)

def selected_epoch(history):
    scores = history['val_auc'] + history['val_f1']
    if history.empty or not scores.map(lambda x: float('-inf') < x < float('inf')).all():
        raise ValueError('Validation selection requires finite scores and a nonempty history.')
    # idxmax returns the first maximum, preserving the notebook tie rule.
    return int(history.loc[scores.idxmax(), 'epoch'])

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def experiment_manifest(conf, datasets, method, epsilon, epochs, device):
    feature = Path(conf.data_dir)
    module_dir = Path(__file__).resolve().parent
    code_files = ('abmil_model.py','training_engine.py','label_smoothing_loss.py',
                  'experiment_utils.py','train_extension.py')
    metadata = dict(config=vars(conf).copy(), method=method,
                    label_smoothing=0.0 if method == 'baseline' else float(epsilon),
                    epochs=int(epochs), device=str(device), torch_version=str(torch.__version__),
                    selection='validation macro-AUROC + macro-F1; earliest tie',
                    feature_size=feature.stat().st_size, feature_mtime_ns=feature.stat().st_mtime_ns,
                    splits={name:list(ds.data_names) for name,ds in zip(SPLIT_NAMES,datasets)},
                    code_sha256={name:sha256_file(module_dir/name) for name in code_files})
    root = module_dir.parent
    metadata['shared_code_sha256'] = {name: sha256_file(root / name) for name in ('dataloader.py', 'utils.py')}
    metadata['environment'] = environment_info(device)
    metadata['metadata_sha256'] = {
        name: sha256_file(path) for name in ('camelyon17.csv', 'bracs.csv')
        if (path := Path(getattr(conf, name[:-4] + '_csv', Path(conf.dataset_csv_dir) / name))).is_file()
    }
    metadata['signature'] = hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    return metadata

def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
