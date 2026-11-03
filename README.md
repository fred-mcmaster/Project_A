# ECE1512 Project A: ABMIL and Top-K attention pooling (20%)

This project classifies whole-slide histopathology images using gated attention-based multiple instance learning (ABMIL) on fixed, pre-extracted medical-SSL patch features. The research extension uses **ratio-based Top-K attention pooling**, retaining the highest-scoring **20% of patches in each slide** (`conf.topk_ratio = 0.20`). The image encoder is fixed; training updates the MIL aggregator and classifier.

The saved notebook outputs are the source of the results below. All three baseline notebooks contain completed 30-epoch runs. The C16 and BRACS Top-K notebooks also contain completed 30-epoch runs; **the local C17 Top-K notebook has no saved outputs**, so its 20% result is unavailable. The earlier fixed-128 experiments are a different pooling configuration and do not supply results for the current ratio.

## Included notebooks and code

| Dataset | ABMIL baseline | Top-K 20% |
|---|---|---|
| C16 | [Student C16](ECE1512_ProjectA_Student_C16.ipynb) | [Top-K C16](ECE1512_ProjectA_TopK_C16.ipynb) |
| C17 | [Student C17](ECE1512_ProjectA_Student_C17.ipynb) | [Top-K C17](ECE1512_ProjectA_TopK_C17.ipynb) |
| BRACS | [Student BRACS](ECE1512_ProjectA_Student_BRACS.ipynb) | [Top-K BRACS](ECE1512_ProjectA_TopK_BRACS.ipynb) |

Shared data loading and metric helpers are in [`dataloader.py`](dataloader.py) and [`utils.py`](utils.py). Dataset settings are in [`config/`](config/), split metadata is in [`dataset_csv/`](dataset_csv/README.md), and optional C16 attention overlays use [`wsi_visualization.py`](wsi_visualization.py). The measured outcomes and their interpretation are in [`observed_results.md`](observed_results.md).

Each Top-K notebook trains its selected dataset independently. Baseline training is in the separate Student notebooks.

## Setup

Use 64-bit Python 3.12 and run from the repository root. The saved C16 and BRACS Top-K outputs report PyTorch **2.13.0+cu126**, CUDA available, and an **NVIDIA GeForce RTX 3080 Ti**.

```bash
git clone https://github.com/fred-mcmaster/Project_A.git
cd Project_A
```

Create and activate an environment. Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

For the CUDA build recorded in the notebook outputs, install PyTorch first:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

For CPU execution, use the CPU build instead on Windows/Linux:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

On macOS, use `python -m pip install torch==2.13.0`. Then install the shared dependencies and notebook kernel:

```bash
python -m pip install -r requirements.txt
python -m pip install ipykernel
python -m pip check
```

Select this environment as the notebook kernel in your editor. The notebooks choose CUDA when available and CPU otherwise. The first notebook cell also provides a convenience dependency installation command; `requirements.txt` supplies the pinned training dependencies.

## Feature data and splits

Download the course's pre-extracted WSI features from [Zenodo](https://zenodo.org/records/23130543) or the [Hugging Face dataset](https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature). Extract the HDF5 files into the layout expected by the YAML configurations:

```text
feature/
  BRACS/
    patch_feats_pretrain_medical_ssl.h5       # BRACS
    C16/patch_feats_pretrain_medical_ssl.h5   # CAMELYON16
    C17/patch_feats_pretrain_medical_ssl.h5   # CAMELYON17
```

If features are stored elsewhere, update `data_dir` in the corresponding `config/*.yml`. Each HDF5 slide group provides `feat [num_patches, 384]`, an integer `label` attribute, and optional patch coordinates. The loader reads labels from HDF5 and uses the included metadata CSVs for C17 centers and BRACS split assignments.

| Dataset | Classes | Train | Validation | Test | Split protocol |
|---|---:|---:|---:|---:|---|
| C16 | 2 | 242 | 27 | 129 | IDs containing `test` held out; stratified 10% validation of remaining slides, seed 42 |
| C17 | 4 | 269 | 30 | 200 | Centers 0-2 for development, centers 3-4 for test; stratified 10% slide-level validation, seed 42 |
| BRACS | 7 | 389 | 62 | 86 | Included CSV train/validation/test assignments; original seven classes |

The C17 partition is an experiment split of available challenge training slides. Validation is split by slide rather than by patient; it should not be interpreted as patient-independent validation.

## Run the experiments

1. Open the dataset-specific notebook and select the configured Python environment.
2. Check `PROJECT_DIR`, `DATASET`, and the feature path printed in Step 0. The Top-K notebooks set `conf.use_topk = True` and `conf.topk_ratio = 0.20` after loading the YAML configuration.
3. Restart the kernel and run cells in order through Step 7. The batch preview and model sanity cells consume random numbers, so execution order matters for reproducing the saved run.
4. Keep `NUM_EPOCHS = int(conf.train_epoch)` (30). Step 6 selects the checkpoint using validation macro-AUROC + macro-F1, retaining the earliest exact tie. Step 7 restores that state for held-out test evaluation.
5. Save the notebook with its outputs. Step 8 prints attention normalization and selected-patch counts. The final WSI visualization cell is optional.

`best_model_state` and the epoch history are held in notebook memory; these notebooks do not automatically export checkpoints, CSV histories, test probabilities, or pairing manifests. Saved cell outputs record the reported measurements. An independent rerun requires training again unless a checkpoint has been saved separately.

## Pooling and training configuration

For a slide with `N` patches, the current implementation computes:

```python
k = max(1, int(N * conf.topk_ratio))  # ratio = 0.20
```

It scores every patch with the gated attention network, retains the `k` largest attention logits, masks all other logits to negative infinity, and applies softmax over the retained set. The weighted sum of projected features forms the slide representation. Thus `k` varies with slide size and uses floor rounding, with at least one patch retained. A ratio of 1.0 retains every patch and recovers ordinary attention pooling.

| Setting | Notebook value |
|---|---|
| Features | Fixed medical-SSL, 384 dimensions; all patches scored |
| Projection | Linear(384, 256), ReLU |
| Attention | Gated scorer, hidden dimension 128, one attention head |
| Pooling | Baseline: all patches; Top-K: highest-scoring 20%, softmax renormalized |
| Classifier | Linear(256, C), with C = 2 / 4 / 7 |
| Dropout | 0.0 |
| Loss | Ordinary hard-target cross-entropy |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.00001 |
| Epochs / batch size | 30 / 1 slide |
| Training subset | All training slides (`n_shot=0`) |
| Seed | 42 |
| Data loading | 0 workers; training shuffled, validation/test ordered |
| Model selection | Maximum validation macro-AUROC + macro-F1, earliest tie |
| Schedule | Constant learning rate; no scheduler or early stopping |

`warmup_epoch` and `n_res` are inherited configuration fields and are unused. Keep batch size 1 for this notebook implementation: the model does not consume the padding mask returned by the collate helper, so variable-length padded batches would require additional masking support. Scoring and projection still process the full bag; the experiment does not establish lower feature extraction or full-bag scoring cost.

## Results from saved notebook outputs

Values below are copied from **Step 7, final held-out test evaluation** (cell index 30) in the linked notebooks. Selected epochs are one-based; all five runs with saved results completed 30 epochs. Accuracy is a percentage, AUROC and F1 are macro averages, and test loss is mean cross-entropy (NLL; lower is better).

| Dataset | Method | Selected epoch | Accuracy (%) | Macro-AUROC | Macro-F1 | Test loss |
|---|---|---:|---:|---:|---:|---:|
| C16 | ABMIL | 2 | 82.9457 | 0.8707 | 0.8159 | 0.4165 |
| C16 | Top-K 20% | 2 | 86.8217 | 0.9395 | 0.8617 | 0.3064 |
| C17 | ABMIL | 17 | 79.0000 | 0.8367 | 0.5258 | 0.7025 |
| C17 | Top-K 20% | - | - | - | - | - |
| BRACS | ABMIL | 10 | 40.6977 | 0.7922 | 0.3439 | 1.6316 |
| BRACS | Top-K 20% | 10 | 38.3721 | 0.7812 | 0.3208 | 1.6879 |

The C17 Top-K row is unavailable because the local notebook has no saved outputs. No three-dataset conclusion can be drawn for the current ratio.

Relative to the saved notebook baselines, Top-K 20% improves C16 accuracy by **3.88 percentage points**, AUROC by **0.0688**, and macro-F1 by **0.0458**, with test loss decreasing by **0.1101**. BRACS accuracy decreases by **2.33 percentage points**, AUROC by **0.0110**, and macro-F1 by **0.0231**, with test loss increasing by **0.0563**. These are single-seed observations. Per-class recall, paired error changes, Brier scores, and ECE are not available from these current saved outputs. See the [observed-results discussion](observed_results.md) for the training-curve evidence and limitations.

## Optional C16 attention overlays

For the final C16 visualization cell, install the optional WSI dependencies:

```bash
python -m pip install openslide-python openslide-bin
```

Place `test_067.tif`, `test_068.tif`, and `test_068.xml` in `slide/` under the project root. The helper restores the selected checkpoint and plots slide-level attention overlays and zooms. Colors represent within-slide attention percentiles; the displayed `P(tumor)` is a slide-level prediction. The default 512-pixel level-0 patch footprint is an assumption because the HDF5 files do not contain patch-size metadata.

## Reference

Ilse, Tomczak, and Welling. [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html), ICML 2018.
