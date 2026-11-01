# ECE1512 Project A - PyTorch MIL on WSI features

This is the **starter package** for Project A. The project uses Attention-Based Multiple Instance Learning (ABMIL) for whole-slide histopathology classification. After completing Project A, you should understand the standard PyTorch supervised-learning pipeline and be able to construct a basic PyTorch training pipeline on your own.

## 1. Download the pre-extracted features

The pre-extracted histopathology features are hosted on Zenodo:

https://zenodo.org/records/23130543


Download the HDF5 feature files and place them in the designated `feature` folder.

```text
feature/
  C16/patch_feats_pretrain_medical_ssl.h5
  C17/patch_feats_pretrain_medical_ssl.h5
  BRACS/patch_feats_pretrain_medical_ssl.h5
```

If you store the HDF5 feature files elsewhere, edit only `data_dir` in the selected YAML file.

## 2. Coding

Open:

```text
ECE1512_ProjectA_Student.ipynb
```

The notebook now includes the completed masked ABMIL model, training and evaluation
pipeline, report answers, and research extension. Run its cells in order. Step 9
compares ordinary cross-entropy with label smoothing on C16, C17, and BRACS using
the configured 30 epochs for each of the six experiments.

For this local feature layout, the C16 and C17 YAML files point to
`feature/BRACS/C16/` and `feature/BRACS/C17/`. BRACS uses `feature/BRACS/`.
The existing HDF5 files are used directly.

## 3. What is provided

- `dataloader.py` - HDF5 reading, dataset-specific splitting, few-shot selection, and padded MIL collation.
- `utils.py` - configuration loading, seeding, feature-dimension checks, and AUROC/F1 metric helpers.
- `config/` - templates for CAMELYON16, CAMELYON17, and BRACS.
- `dataset_csv/` - metadata required by the CAMELYON17 and BRACS split logic.
- `ECE1512_ProjectA_Student.ipynb` - guided PyTorch implementation, baseline experiment, and ablation workspace.

The completed ABMIL model and training loop are implemented directly in the notebook.

## 4. Expected HDF5 layout

Each top-level HDF5 key is one slide:

```text
slide_id/
  feat    [num_patches, feature_dim]
  coords  [num_patches, 2]   # optional for training
attrs:
  label   integer class label
```

Different slides contain different numbers of patches. `mil_collate_fn` pads a batch and returns a boolean `mask` that marks valid patches.

## 5. Colab/local setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

In Colab, place the project folder in Google Drive or upload it to the runtime, set `PROJECT_DIR` in the notebook, and select `C16`, `C17`, or `BRACS`.

For an initial debugging run, use only a few epochs. Once your pipeline passes the sanity checks, run the full baseline and keep the same split, seed, and evaluation protocol for your ablations.

## 6. Running and checking the completed notebook

In a local notebook editor, select `.venv/Scripts/python.exe` as the Python kernel.
The project-local environment contains the missing dependencies and uses the
existing CUDA-enabled PyTorch installation. To execute and save all notebook
outputs from PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebook.py
```

Completed experiments are reused when the notebook code, configuration, feature
file, split membership, and epoch count match. Set `REUSE_COMPLETED_RUNS = False`
in the notebook to force fresh training. Results are written to `results/`:

- `comparison.csv` and `extension_deltas.csv`: baseline/extension test results.
- `<dataset>_<method>/best.pt`: validation-selected weights, configuration, and history.
- `<dataset>_<method>/history.csv`, `test_metrics.json`, and `test_predictions.csv`.
- `<dataset>/splits.csv`: exact slide membership, labels, and patch counts.
- `learning_and_calibration.png`: validation curves and test reliability plots.
- `per_class_metrics.csv` and `confusion_matrices.png`: class-level test diagnostics.
- `environment.json`: versions and compute device used by the experiments.

The notebook contains the written explanations, proposals, measured results,
and discussion. Its optional WSI overlays use the files in `slide/`;
`scripts/download_slides.py` downloads the three assets specified by Step 8.

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe scripts/summarize_results.py
.\.venv\Scripts\python.exe scripts/verify_results.py
```

The first command checks masking, permutation invariance, optimization,
evaluation, label smoothing, and real-data forward/backward passes. After all
six experiments finish, the second refreshes the notebook's recorded discussion
and class-level figures. The third independently checks the reported metrics
and selected epochs against saved predictions and histories, and verifies the
two WSI overlays.
