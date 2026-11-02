"""Check feature schema, labels, and ordered splits against the reported experiments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import pandas as pd
from research_extensions.experiment_utils import prepare_dataset, split_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=ROOT)
    parser.add_argument('--datasets', nargs='+', choices=['C16', 'C17', 'BRACS'], default=['C16', 'C17', 'BRACS'])
    args = parser.parse_args(argv)
    root = args.project_dir.resolve()
    manifest = json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf-8'))
    for dataset in args.datasets:
        conf, datasets = prepare_dataset(root, dataset, 42)
        expected = manifest['datasets'][dataset]
        if int(conf.n_class) != expected['classes'] or int(conf.D_feat) != expected['feature_dim']:
            raise ValueError(f'{dataset}: class count or feature dimension differs from the reported protocol.')
        with h5py.File(conf.data_dir, 'r') as h5:
            if len(h5) != expected['slides']:
                raise ValueError(f'{dataset}: expected {expected["slides"]} slides, found {len(h5)}.')
            for slide_id, group in h5.items():
                shape = group['feat'].shape
                if len(shape) != 2 or shape[0] == 0 or shape[1] != expected['feature_dim']:
                    raise ValueError(f'{dataset}/{slide_id}: invalid feature shape {shape}.')
        current = split_records(conf, datasets)
        reference_dir = ROOT / 'research_extensions' / 'evaluation_results'
        for method in ('baseline', 'label_smoothing'):
            reference = pd.read_csv(reference_dir / f'{dataset}_{method}' / 'splits.csv')
            pd.testing.assert_frame_equal(current, reference, check_dtype=False)
        print(f'{dataset}: schema, labels, patch counts and ordered splits verified; '
              f'train/validation/test = {[len(ds) for ds in datasets]}', flush=True)
    print('For byte-level feature verification, also run scripts/download_features.py --verify-only.')


if __name__ == '__main__':
    main()
