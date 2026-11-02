# ECE1512 Project A: reproducible ABMIL experiments

This project classifies whole-slide histopathology images using **gated attention-based multiple instance learning (ABMIL)** on fixed, pre-extracted medical-SSL patch features. It provides a reproducible PyTorch baseline and selects **top-k attention pooling** as the next research extension to implement.

The completed baseline covers CAMELYON16 (C16), CAMELYON17 (C17), and BRACS: **30 epochs per dataset, seed 42**. The [baseline summary](<ECE1512 ProjectA Baseline Evaluation Summary.docx>) includes measured results and the nine notebook report questions and answers. The [research proposal](ECE1512_ProjectA_Research_Extension_Proposals_and_Evaluation.docx) describes selected top-k attention pooling and the alternative diversity-regularized two-head idea. Both extensions are pending implementation and evaluation. The image encoder is fixed; training updates the MIL aggregator and classifier.

## 1. What is included

| Artifact | Location |
|---|---|
| Pinned training/evaluation dependencies | [`requirements.txt`](requirements.txt) |
| Optional notebook/WSI dependencies | [`requirements-notebook.txt`](requirements-notebook.txt) |
| Training CLI and shared model/engine | [`research_extensions/`](research_extensions/) |
| Evaluation CLI | [`research_extensions/evaluate_extension.py`](research_extensions/evaluate_extension.py) |
| Three original validation-selected baseline checkpoints (~2 MB total) | [`checkpoints/`](checkpoints/), included directly in Git |
| Checkpoint hashes and original compute environment | [`checkpoints/manifest.json`](checkpoints/manifest.json), [`checkpoints/environment.json`](checkpoints/environment.json) |
| Per-slide test probabilities, histories, ordered splits, metrics, figures | [`research_extensions/evaluation_results/`](research_extensions/evaluation_results/) |
| Dataset checksums and immutable download revision | [`data_manifest.json`](data_manifest.json) |
| Dataset-specific settings and experiment settings | [`config/`](config/), [`experiment_config.json`](research_extensions/experiment_config.json) |
| Educational notebook | [`ECE1512_ProjectA_Student.ipynb`](ECE1512_ProjectA_Student.ipynb) |

Feature files and raw slides are downloaded separately. Generated runs go under ignored `results/`; the checkpoints and reference results above are versioned. The notebook is supplied with outputs cleared; the CLI is the primary reproduction path.

## 2. Clone and install

Use **64-bit Python 3.12** (the recorded version is **3.12.7**). Run all commands from the repository root.

```bash
git clone https://github.com/fred-mcmaster/Project_A.git
cd Project_A
```

Create and activate an isolated environment using your platform's commands.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Choose **one** PyTorch installation below, then install the remaining pinned dependencies. The original experiments used Windows, PyTorch **2.13.0+cu126**, CUDA 12.6, and an **NVIDIA GeForce RTX 3080 Ti**. CUDA requires a compatible NVIDIA driver. See the [PyTorch installation documentation](https://pytorch.org/get-started/locally/).

```bash
# Original CUDA build, for Windows/Linux with an NVIDIA GPU:
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

# Or CPU build, for Windows/Linux without a compatible GPU:
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Or macOS (the supplied CLI uses CPU on macOS):
python -m pip install torch==2.13.0
```

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The direct package versions are pinned to the recorded environment. Platform-specific transitive dependencies are resolved by pip. CPU inference and training are supported but can be substantially slower. The original GPU has 12 GB VRAM; minimum memory and wall-clock runtime have not been benchmarked. Full variable-length bags are used.

## 3. Dataset source and download

The course's pre-extracted **WSI Feature** dataset is published by Linfeng Ye at [Zenodo, DOI 10.5281/zenodo.23130543](https://zenodo.org/records/23130543), with a [Hugging Face copy](https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature/tree/6df7ef4b42b7ecd11633722a392a1efc6d80bcc7). The underlying datasets are [CAMELYON16](https://camelyon16.grand-challenge.org/), [CAMELYON17](https://camelyon17.grand-challenge.org/), and [BRACS](https://www.bracs.icar.cnr.it/). Use the supplied features for this experiment; newly extracted features are a different input representation.

Download all three HDF5 files (**15.54 GB total**, plus environment/output storage):

```bash
python scripts/download_features.py
python scripts/verify_data.py
```

The downloader uses the immutable Hugging Face revision in `data_manifest.json`, streams one file at a time, and checks byte size and SHA256 before accepting it. Existing files are checked and reused. To download just one dataset, append `--datasets C16`, `--datasets C17`, or `--datasets BRACS`. To inspect URLs without downloading, use `--dry-run`. Interrupted `.part` files are restarted on the next download attempt.

The destination layout intentionally matches the existing YAML files and original checkpoints:

```text
feature/
  BRACS/
    patch_feats_pretrain_medical_ssl.h5       # BRACS
    C16/patch_feats_pretrain_medical_ssl.h5   # CAMELYON16
    C17/patch_feats_pretrain_medical_ssl.h5   # CAMELYON17
```

For a manual Zenodo download, obtain `C16.zip`, `C17.zip`, and `BRACS.zip`, extract them, and place each `patch_feats_pretrain_medical_ssl.h5` at the corresponding path above. Check the files after extraction:

```bash
python scripts/download_features.py --verify-only
python scripts/verify_data.py
```

If storing features elsewhere, edit only `data_dir` in the relevant `config/*.yml` and run `verify_data.py`; training and checkpoint evaluation both use these checkout-local paths. `download_features.py` always uses the destinations recorded in `data_manifest.json`; its `--verify-only` option checks that default layout.

Each top-level HDF5 key is a slide ID, containing `feat [num_patches, 384]`, optional `coords [num_patches, 2]`, and an integer `label` attribute on the slide group. Labels are read from HDF5. The versioned [metadata CSVs](dataset_csv/README.md) supply the C17 centers/patient IDs and BRACS split assignments. They are already included; no access to the original author's local metadata path is needed.

| Dataset | Classes | Train | Validation | Test | Split protocol |
|---|---:|---:|---:|---:|---|
| C16 | 2 | 242 | 27 | 129 | IDs containing `test` held out; stratified 10% validation of remaining slides, seed 42 |
| C17 | 4 | 269 | 30 | 200 | Centers 0-2 for development, centers 3-4 for test; stratified 10% slide-level validation, seed 42 |
| BRACS | 7 | 389 | 62 | 86 | Versioned CSV train/validation/test assignments; original seven classes |

`verify_data.py` checks every slide's feature shape, label, patch count, and **ordered** split membership against the reference exports. C17's split is a course experiment partition of the available challenge training slides; it is not the official challenge train/test partition. Its training and validation sets share 27 patients; test patients and centers remain separate.

## 4. Verify or evaluate the pretrained models

The following command reproduces all reported metrics from the **tracked test probabilities**, verifies each checkpoint's SHA256, and checks that the selected epoch matches the validation history. It requires **no downloaded features and no retraining**:

```bash
python scripts/verify_results.py --device cpu
```

To actually run inference using all three bundled baseline checkpoints, first download the features, then run:

```bash
python -m research_extensions.evaluate_extension --recompute --device cuda
```

Use `--device cpu` if needed. Outputs go to `results/pretrained_evaluation/`, preserving the versioned reference artifacts. The evaluator verifies ordered test IDs/labels and compares recomputed probabilities with the saved predictions (`rtol=2e-5`, `atol=2e-6`). It also independently recomputes the metrics (`rtol=2e-5`, `atol=1e-5`). `evaluation_provenance.json` records actual checkpoint hashes, execution mode, device, and prediction differences.

## 5. Train and evaluate the baselines

After dataset verification, run:

```bash
python -m research_extensions.train_extension --epochs 30 --seed 42 --device cuda --output-dir results/retrained
python -m research_extensions.evaluate_extension --runs-dir results/retrained --recompute --device cuda --output-dir results/retrained_evaluation
```

The first command trains all three baseline experiments and selects checkpoints using **validation macro-AUROC + macro-F1**, keeping the earliest epoch on a tie. It never evaluates the test split. The second command performs final test inference on those selected checkpoints. Each run resets model initialization and the independent training-sampler seed.

Training writes `<dataset>_<method>/best.pt`, `history.csv`, `manifest.json`, and `<dataset>/splits.csv`. Evaluation writes `extension_comparison.csv`, per-class metrics, calibration/confusion figures, and per-run test probabilities/metrics. Checkpoints include model weights, selected epoch, completed epoch count, configuration, training signature, and history.

Completed experiments with matching signatures are reused. To force a fresh run, choose a new training output directory and pass that same directory as evaluation's `--runs-dir`, or explicitly use `--overwrite`. A changed code/configuration signature requires a new directory or `--overwrite`. These local cache signatures include feature file timestamps and paths; they are not intended to match across machines. Checkpoint hashes identify the original supplied models.

For a short pipeline check (not a reported result):

```bash
python -m research_extensions.train_extension --datasets C16 --epochs 1 --device cuda --output-dir results/smoke
python -m research_extensions.evaluate_extension --datasets C16 --runs-dir results/smoke --recompute --device cuda --output-dir results/smoke_evaluation
```

## 6. Hyperparameters and experiment configuration

| Setting | Value used for every reported run |
|---|---|
| Features | Fixed medical-SSL features, 384 dimensions; every patch in each bag |
| Projection | Linear(384, 256), ReLU |
| Attention | One gated head: tanh and sigmoid branches, 256 -> 128 -> scalar, masked patch softmax |
| Classifier | Linear(256, C); C = 2 / 4 / 7 for C16 / C17 / BRACS |
| Dropout | 0.0 |
| Training objective | Ordinary hard-target cross-entropy |
| Optimizer | AdamW; learning rate = 0.0001; weight decay = 0.00001; default betas = (0.9, 0.999), eps = 1e-8 |
| Schedule | Constant learning rate; no warm-up, scheduler, early stopping, or augmentation |
| Epochs / batch size | 30 / 1 slide |
| Training subset | All training slides (`n_shot=0`) |
| Seed | 42 for Python, NumPy, PyTorch, and the training sampler |
| Data loading | 0 workers; training shuffled, validation/test ordered; CPU threads capped at 4 |
| Model selection | Maximum validation macro-AUROC + macro-F1, earliest tie |
| Validation/test loss | Ordinary hard-target cross-entropy |

Dataset-specific values live in `config/*.yml`; datasets, methods, epochs, and seed also live in `research_extensions/experiment_config.json`. Currently only `baseline` is implemented. CLI flags override the experiment JSON; the CLI infers feature dimension from the HDF5 file. `warmup_epoch` and `n_res` are inherited starter fields and are **unused**.

## 7. Results and exact commands for every row

These baseline values match the baseline report and [`extension_comparison.csv`](research_extensions/evaluation_results/extension_comparison.csv). All three baseline runs completed 30 epochs; selected epochs below are **one-based**. AUROC is macro one-vs-rest (binary C16 equals positive-class AUROC); F1 is macro-F1. Accuracy is a percentage. NLL is mean hard-target CE, Brier is the mean sum of squared class-probability errors, and ECE uses 15 equal-width top-label confidence bins. Lower NLL/Brier/ECE is better.

| Row | Dataset | Method | Selected epoch | Accuracy (%) | AUROC | Macro-F1 | NLL | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [R1](#r1-c16-baseline) | C16 | Baseline CE | 3 | 88.37 | 0.9342 | 0.8668 | 0.2804 | 0.1628 | 0.0564 |
| [R2](#r2-c17-baseline) | C17 | Baseline CE | 22 | 78.50 | 0.8243 | 0.5339 | 0.7347 | 0.3253 | 0.1148 |
| [R3](#r3-bracs-baseline) | BRACS | Baseline CE | 6 | 39.53 | 0.8043 | 0.3494 | 1.4266 | 0.7183 | 0.2547 |

For each row below: the **first command** reproduces the reported checkpoint's inference, and the **second and third commands** independently retrain and evaluate that row. Run these after setup/download. For verification without features, omit `--recompute` from the first command. For CPU execution, replace `--device cuda` with `--device cpu`.

### R1: C16 baseline

```bash
python -m research_extensions.evaluate_extension --datasets C16 --methods baseline --recompute --device cuda --output-dir results/pretrained_R1
python -m research_extensions.train_extension --datasets C16 --methods baseline --epochs 30 --seed 42 --device cuda --output-dir results/retrained
python -m research_extensions.evaluate_extension --datasets C16 --methods baseline --runs-dir results/retrained --recompute --device cuda --output-dir results/retrained_R1
```

### R2: C17 baseline

```bash
python -m research_extensions.evaluate_extension --datasets C17 --methods baseline --recompute --device cuda --output-dir results/pretrained_R2
python -m research_extensions.train_extension --datasets C17 --methods baseline --epochs 30 --seed 42 --device cuda --output-dir results/retrained
python -m research_extensions.evaluate_extension --datasets C17 --methods baseline --runs-dir results/retrained --recompute --device cuda --output-dir results/retrained_R2
```

### R3: BRACS baseline

```bash
python -m research_extensions.evaluate_extension --datasets BRACS --methods baseline --recompute --device cuda --output-dir results/pretrained_R3
python -m research_extensions.train_extension --datasets BRACS --methods baseline --epochs 30 --seed 42 --device cuda --output-dir results/retrained
python -m research_extensions.evaluate_extension --datasets BRACS --methods baseline --runs-dir results/retrained --recompute --device cuda --output-dir results/retrained_R3
```

The supplied checkpoints and saved probabilities support checking the reported values to the displayed precision. Fresh retraining can differ across hardware, drivers, or dependency builds despite fixed seeds; bitwise-identical training across platforms is not guaranteed. These are **single-seed** comparisons, with no confidence intervals or claim of statistical significance.

## Selected research idea: top-k attention pooling

Keep the gated patch scorer, projection, classifier, and ordinary cross-entropy. For a bag of N valid patches, choose the k highest attention logits, apply softmax over those selected logits, and pool only their projected features. Padding must be excluded, and k must be capped at the number of valid patches. Selecting all patches must reproduce the baseline.

The hypothesis is that concentrating on the most informative patches can reduce background dilution. A small k may discard distributed tissue evidence, so improvement is an experimental question. Define the candidate k values before extension experiments and select k using validation only. Preserve the baseline features, splits, optimizer, epoch budget, initialization, sampler seed, and validation checkpoint rule. Report macro-AUROC, macro-F1, accuracy, per-class recall, and probability quality on all three datasets. Top-k pooling still scores every patch; reduced feature extraction cost is not assumed.

**Status:** selected for implementation; no top-k code, checkpoints, or measured results are included yet. Diversity-regularized two-head attention remains the second proposal.

## 8. Validation and optional notebook/WSI workflow

```bash
python -m unittest discover -s research_extensions/tests -v
```

The suite checks baseline outputs against the notebook on unpadded bags, padding and permutation behavior, controlled learning, reproducible seeds/order, validation-only selection, known probability metrics, checkpoint relocation, and the bundled results. Real-feature forward/backward integration checks run when all features are present; otherwise that test is explicitly skipped. Saved-artifact tests run without feature downloads.

For the educational notebook and optional raw-WSI attention overlays:

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "Project A"
python scripts/validate_notebook.py
python scripts/execute_notebook.py
```

Alternatively select the new environment's interpreter in your notebook editor and run the cells in order. Set `PROJECT_DIR` if working outside the repository or in Colab. The notebook uses its own baseline workflow and does not automatically reuse `checkpoints/`. CLI runs and results are sufficient for reproducing the tables.

The optional overlays need approximately **4.95 GB** of raw slide/XML assets:

```bash
python scripts/download_slides.py
```

Run the notebook again after downloading to generate the overlays. OpenSlide may require a supported native library on your platform; see the [OpenSlide Python installation instructions](https://openslide.org/api/python/).

The reports' historical references to `results/<dataset>_<method>/best.pt` correspond to the byte-identical files now included under `checkpoints/<dataset>_<method>/best.pt`. Historical histories, predictions, and split files are included under `research_extensions/evaluation_results/`.

## References

- Ilse, Tomczak, and Welling. [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html), ICML 2018.
- Ye, Linfeng. [WSI Feature](https://doi.org/10.5281/zenodo.23130543), course feature release, 2026.
