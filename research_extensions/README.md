# Research extensions: ABMIL with label smoothing

This folder packages the selected research idea independently of the notebook.
The accompanying Word report explains **two proposals** and reports the implemented
label-smoothing experiment on C16, C17, and BRACS. Diversity-regularized two-head
attention is a proposal only; it is not claimed as implemented or evaluated.

## Files

| File | Purpose |
|---|---|
| `abmil_model.py` | Masked gated ABMIL; compatible with the original six checkpoints. |
| `label_smoothing_loss.py` | Selected objective: uniform label smoothing, epsilon = 0.1. |
| `training_engine.py` | Shared loader, training, hard-target evaluation, and calibration routines. |
| `experiment_utils.py` | Dataset/configuration loading, split recording, selection, and provenance. |
| `experiment_config.json` | Dataset list, methods, 30 epochs, seed 42, and smoothing strength. |
| `train_extension.py` | Train baseline and extension with paired initialization and slide order. |
| `evaluate_extension.py` | Evaluate frozen checkpoints, verify metrics, and export comparisons/plots. |
| `tests/test_research_extensions.py` | Eight checks of mathematical correctness and the experiment controls. |
| `evaluation_results/` | Verified results from all six original 30-epoch experiments. |

## Setup

Run the commands below from the project folder containing
`ECE1512_ProjectA_Student.ipynb`. Use the existing `.venv` interpreter, or install
`requirements.txt` in another Python environment. The existing YAML files specify
feature paths and dataset-specific settings. No feature downloads are needed.

## Evaluate the completed experiments

The original trained checkpoints remain in `results/C16_baseline`,
`results/C16_label_smoothing`, and the equivalent C17/BRACS directories. These
are six completed 30-epoch experiments, not one-epoch demonstration results.

```powershell
.\.venv\Scripts\python.exe -m research_extensions.evaluate_extension
```

This verifies the saved validation-selected epochs and independently calculates
accuracy, macro-AUROC, macro-F1, hard-target NLL, Brier score, and 15-bin top-label
ECE from saved test predictions. If a run has no saved test predictions, its
frozen checkpoint is evaluated. Outputs go to `research_extensions/evaluation_results/`.

To reproduce inference and verify the exported model against the saved predictions:

```powershell
.\.venv\Scripts\python.exe -m research_extensions.evaluate_extension --recompute
```

Inference does not reselect checkpoints or tune hyperparameters. The delivered
results were checked using this option. `evaluation_provenance.json` records
checkpoint hashes, inference mode, and maximum probability differences.
For old notebook checkpoints without a smoothing field, the evaluator records
the notebook's fixed epsilon of 0.1; their folder names and stored test signatures
identify the original baseline/extension runs.

## Train fresh experiments

```powershell
.\.venv\Scripts\python.exe -m research_extensions.train_extension
.\.venv\Scripts\python.exe -m research_extensions.evaluate_extension --runs-dir results/research_extension_runs --output-dir research_extensions/fresh_evaluation_results
```

Training defaults to all three datasets and both methods. New checkpoints and
histories are saved under `results/research_extension_runs/`, separate from the
original experiments. The training command selects a checkpoint using validation
macro-AUROC + macro-F1 and does not evaluate the test set. Ties keep the earliest
epoch. The evaluation command is the final test step after selection.

Completed matching experiments are reused. If settings or implementation files
change, use a new `--output-dir` or explicitly pass `--overwrite` to replace the
generated runs. The full configuration, split IDs, feature-file metadata, and
implementation hashes are recorded in each run's `manifest.json`.

A short debugging run uses a separate folder:

```powershell
.\.venv\Scripts\python.exe -m research_extensions.train_extension --datasets C16 --epochs 1 --output-dir results/research_extension_debug
```

Debugging results are not the reported 30-epoch experiments. Additional options
include `--methods`, `--seed`, `--epsilon`, `--device`, and `--project-dir`.

## Experimental controls

- Only the training objective changes: ordinary CE versus CE with epsilon = 0.1.
- Architecture, full feature bags, training slides, splits, seed, optimizer,
  learning rate, weight decay, batch size, and epoch budget are paired.
- Both methods reset the model seed and use the same independently seeded
  training sampler. Validation/test data are not shuffled.
- Validation and test always use **ordinary cross-entropy**, so loss is comparable
  NLL. The smoothed training objective is not reported as test NLL.
- C16 uses two classes, C17 four, and BRACS all seven original classes.
- The provided C17 train/validation split shares patients; test patients and
  centers are separate. This limitation is retained and discussed in the report.
- The single-seed comparison does not establish statistical significance.

## Output files

- `extension_comparison.csv`: six method/dataset rows with selected epochs and metrics.
- `extension_deltas.csv`: label smoothing minus baseline; accuracy is in percentage points.
- `extension_per_class_metrics.csv`: precision, recall, F1, and support for every class.
- `extension_learning_and_calibration.png`: validation NLL and test reliability plots.
- `extension_confusion_matrices.png`: class-level errors for both methods.
- `evaluation_provenance.json`: source checkpoint hashes and inference verification.
- `<dataset>_<method>/`: test probabilities, metrics, training history, split IDs,
  reliability bins, and confusion counts.

Lower NLL, Brier score, and ECE is better. The observed changes are mixed: C16
calibration worsens; C17 calibration improves but minority-class recall and
macro-F1 fall; BRACS shows small calibration/F1 gains with a small AUROC decrease.
No test-driven hyperparameter changes were made to improve these outcomes.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s research_extensions/tests -v
```

Checks cover exact model/engine equivalence to the notebook, the smoothed-target
formula and gradients for 2/4/7 classes, padding and permutation behavior, learning
on a small controlled task, read-only evaluation, paired initialization/order,
validation-only selection with tie handling, known probability metrics, and
forward/backward passes on real bags from all three datasets.

The supplied notebook and its highlighted answers remain the guided educational
version; these files provide the standalone implementation and reproducible CLI.
