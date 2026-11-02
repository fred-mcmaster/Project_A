# ABMIL baseline and research extension plan

This folder contains the reproducible gated ABMIL baseline for C16, C17, and BRACS. **Top-k attention pooling is selected for implementation.** The code and bundled results currently cover the baseline only. The Word proposal also retains diversity-regularized two-head attention as the second research idea.

## Selected idea: top-k attention pooling

For each slide, score every valid patch with the existing gated attention network. Select the k highest attention logits, normalize only those logits with softmax, and form a weighted sum of the selected projected features. Exclude padding and use at most the slide's valid patch count. With all patches selected, pooling must match the baseline.

Keep ordinary cross-entropy and the baseline training protocol. Define the k candidates before running extension experiments, select k using validation only, and evaluate the frozen selected model on the test split. Concentrating evidence may help when many patches are uninformative; it may also discard distributed tissue patterns. No improvement or runtime gain has been measured yet. All patches still need feature extraction and attention scoring.

## Files

| File | Purpose |
|---|---|
| `abmil_model.py` | Masked gated ABMIL baseline, compatible with the three bundled checkpoints. |
| `training_engine.py` | Loaders, optimization, hard-target evaluation, and calibration metrics. |
| `experiment_utils.py` | Configuration, ordered splits, checkpoint selection, and provenance. |
| `experiment_config.json` | Three datasets, baseline method, 30 epochs, seed 42. |
| `train_extension.py` | Baseline training and validation-selected checkpoint export. |
| `evaluate_extension.py` | Frozen checkpoint evaluation, saved-prediction verification, and plots. |
| `tests/test_research_extensions.py` | Baseline behavior, reproducibility, and artifact checks. |
| `evaluation_results/` | Results from the three completed 30-epoch baseline experiments. |

## Verify the bundled baseline results

Run from the repository root after installing `requirements.txt`:

```bash
python -m research_extensions.evaluate_extension --device cpu
```

This verifies checkpoint hashes, ordered split membership, validation-selected epochs, and metrics using the tracked predictions without downloading features. Bundled weights are in `checkpoints/<dataset>_baseline/best.pt`. Outputs go to `results/pretrained_evaluation/`.

To rerun inference after downloading the features:

```bash
python -m research_extensions.evaluate_extension --recompute --device cuda
```

The evaluator compares recomputed probabilities with the saved predictions and records checkpoint hashes and inference differences in `evaluation_provenance.json`. Inference does not reselect epochs.

## Train and evaluate fresh baselines

```bash
python -m research_extensions.train_extension --epochs 30 --seed 42 --device cuda
python -m research_extensions.evaluate_extension --runs-dir results/research_extension_runs --recompute --device cuda --output-dir results/retrained_evaluation
```

Training defaults to all three datasets and `baseline`. It selects the checkpoint with maximum validation macro-AUROC + macro-F1, keeping the earliest epoch on a tie. Test data are evaluated only by the second command. Use `--device cpu` if needed.

Completed matching runs are reused. Changed configurations or code require a new `--output-dir` or explicit `--overwrite`. Each run saves its configuration, implementation hashes, ordered splits, checkpoint, and history. A short pipeline check uses a separate directory:

```bash
python -m research_extensions.train_extension --datasets C16 --epochs 1 --device cpu --output-dir results/baseline_smoke
python -m research_extensions.evaluate_extension --datasets C16 --runs-dir results/baseline_smoke --device cpu --output-dir results/baseline_smoke_evaluation
```

## Evaluation artifacts

- `extension_comparison.csv`: three baseline rows with selected epochs and classification/calibration metrics.
- `extension_per_class_metrics.csv`: precision, recall, F1, and support.
- `extension_learning_and_calibration.png`: validation NLL and test reliability plots.
- `extension_confusion_matrices.png`: baseline class-level errors.
- `evaluation_provenance.json`: checkpoint hashes, verification mode, and implementation hashes.
- `<dataset>_baseline/`: predictions, metrics, histories, ordered splits, reliability bins, and confusion counts.

The shared output filenames retain the `extension_` prefix for the future extension workflow. They currently contain baseline evidence only. Lower NLL, Brier score, and ECE is better. These are single-seed runs; C17's supplied train/validation split shares 27 patients, while test patients and centers remain separate.

## Checks

```bash
python -m unittest discover -s research_extensions/tests -v
```

Checks cover baseline agreement with the notebook on unpadded bags, masking, permutation invariance, learning, read-only evaluation, reproducible initialization/order, validation-only checkpoint selection, probability metrics, checkpoint relocation, and bundled artifacts. Real-feature integration runs when all three feature files are present.
