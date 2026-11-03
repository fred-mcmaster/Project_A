# Observed results: notebook ABMIL versus Top-K attention pooling (20%)

The current Top-K notebooks use **`conf.topk_ratio = 0.20`**. For each slide with `N` patches, they retain `k = max(1, int(0.20 * N))` highest-scoring attention logits, mask the others to negative infinity, and renormalize softmax over the selected set. The selected count therefore scales with slide size, using floor rounding and a minimum of one patch. Earlier fixed-128 results describe a different experiment.

Results below come directly from **Step 7, final held-out test evaluation** (cell index 30). All three Top-K notebooks contain 30-epoch logs and final test outputs. The three separate Student notebooks supply the baseline measurements; the Top-K notebooks train only the extension.

| Dataset | Baseline source | Top-K source |
|---|---|---|
| C16 | [Student C16](ECE1512_ProjectA_Student_C16.ipynb) | [Top-K C16](ECE1512_ProjectA_TopK_C16.ipynb) |
| C17 | [Student C17](ECE1512_ProjectA_Student_C17.ipynb) | [Top-K C17](ECE1512_ProjectA_TopK_C17.ipynb) |
| BRACS | [Student BRACS](ECE1512_ProjectA_Student_BRACS.ipynb) | [Top-K BRACS](ECE1512_ProjectA_TopK_BRACS.ipynb) |

The notebooks use the same pre-extracted medical-SSL features, dataset split settings, projection, gated scorer, classifier, ordinary cross-entropy, and AdamW (learning rate 0.0001, weight decay 0.00001). Configuration uses seed 42, batch size 1, and 30 epochs, with no scheduler or early stopping. Each checkpoint maximizes validation macro-AUROC + macro-F1, retaining the earliest exact tie; test evaluation follows checkpoint selection. These saved outputs do not include hashes verifying identical initial weights or training slide orders across runs. Preview and sanity cells consume random numbers, so a shared seed alone is insufficient to prove pairing across different execution orders.

| Dataset | Method | Selected epoch | Accuracy (%) | Macro-AUROC | Macro-F1 | Test loss (CE/NLL) |
|---|---|---:|---:|---:|---:|---:|
| C16 | ABMIL | 2 | 82.9457 | 0.8707 | 0.8159 | 0.4165 |
| C16 | Top-K 20% | 2 | 86.8217 | 0.9395 | 0.8617 | 0.3064 |
| C17 | ABMIL | 17 | 79.0000 | 0.8367 | 0.5258 | 0.7025 |
| C17 | Top-K 20% | 5 | 80.5000 | 0.8663 | 0.4876 | 0.5164 |
| BRACS | ABMIL | 10 | 40.6977 | 0.7922 | 0.3439 | 1.6316 |
| BRACS | Top-K 20% | 10 | 38.3721 | 0.7812 | 0.3208 | 1.6879 |

Selected epochs are one-based; all six runs completed the full 30 epochs. Accuracy is a percentage. AUROC is macro one-vs-rest and F1 is macro-F1, as computed by `classification_metrics` in `utils.py`. Test loss is mean hard-target cross-entropy, equivalent to NLL for the corresponding softmax probabilities; lower is better.

Top-K minus ABMIL, calculated from the displayed notebook values:

| Dataset | Accuracy change (percentage points) | Macro-AUROC change | Macro-F1 change | Test loss change |
|---|---:|---:|---:|---:|
| C16 | +3.88 | +0.0688 | +0.0458 | -0.1101 |
| C17 | +1.50 | +0.0296 | -0.0382 | -0.1861 |
| BRACS | -2.33 | -0.0110 | -0.0231 | +0.0563 |

**C16.** Top-K 20% improves all four reported test metrics. Accuracy rises from 82.9457% to 86.8217%, corresponding to 107 versus 112 correct predictions among 129 slides, a net gain of five. AUROC rises from 0.8707 to 0.9395 and macro-F1 from 0.8159 to 0.8617, while test loss falls from 0.4165 to 0.3064. Without per-slide predictions, the net gain does not identify how many baseline errors were corrected or how many new errors were introduced.

Both C16 methods select epoch 2. The Top-K log reaches validation AUROC 1.000 and macro-F1 1.000 at that epoch, with validation loss 0.0936. Later perfect scores do not replace this checkpoint because the rule keeps the earliest exact tie. By epoch 30, the logged training and validation losses are 0.0001 and 0.0000 to four decimal places, while final test performance is measured at the selected epoch. The 27-slide validation split provides limited evidence about generalization despite its perfect selection metrics. The test improvement is consistent with less dilution by uninformative patches, but tissue-level causation is not established.

**C17.** Top-K 20% selects epoch 5 and reports 80.5000% accuracy, 0.8663 macro-AUROC, 0.4876 macro-F1, and 0.5164 test loss on 200 test slides. The baseline selects epoch 17 and reports 79.0000%, 0.8367, 0.5258, and 0.7025, respectively. Accuracy therefore increases by 1.50 percentage points, corresponding to 158 versus 161 correct predictions, a net gain of three. AUROC improves by 0.0296 and test loss falls by 0.1861, while macro-F1 decreases by 0.0382. This is a mixed result: improved accuracy, ranking, and loss coexist with worse macro-F1. AUROC measures ranking across thresholds, while macro-F1 uses argmax predictions and gives each class equal weight; higher accuracy does not establish uniform class-level improvement. The saved outputs do not identify which classes account for the F1 decline.

At the selected C17 Top-K epoch, the log reports training loss 0.4248, validation loss 0.4318, validation AUROC 0.904, and validation macro-F1 0.675. By epoch 30, training loss falls to 0.0326 while validation loss rises to 0.6581, consistent with overfitting after selection. These validation figures are rounded log values. The supplied development split uses slide-level validation across centers 0-2; patients can appear in both training and validation. Centers 3-4 form the held-out test split, so this setup should not be described as patient-independent validation.

**BRACS.** Top-K 20% decreases all three classification metrics and increases test loss. Accuracy falls from 40.6977% to 38.3721%, corresponding to 35 versus 33 correct predictions among 86 slides, a net loss of two. AUROC falls from 0.7922 to 0.7812, macro-F1 from 0.3439 to 0.3208, and loss rises from 1.6316 to 1.6879. This is a negative result for the current 20% setting on the seven-class task. It provides no evidence of an aggregate improvement on BRACS.

Both BRACS methods select epoch 10. At that epoch, the Top-K log reports validation loss 1.5896, AUROC 0.831, and macro-F1 0.403; baseline reports 1.5017, 0.835, and 0.415. These validation figures are rounded log values. Top-K training loss falls from 0.6264 at epoch 10 to 0.0921 at epoch 30 while validation loss rises to 3.2728, consistent with overfitting after the selected checkpoint. The baseline also shows a growing gap, ending at training loss 0.1133 and validation loss 2.8493. Excluding lower-scoring patches may discard useful distributed evidence, but the saved aggregate metrics do not establish why BRACS degrades or which classes account for it.

The saved attention inspection cells confirm the ratio on individual test slides:

| Dataset | Example slide | Total patches | Selected patches | Selected fraction |
|---|---|---:|---:|---:|
| C16 | `test_001` | 17,294 | 3,458 | 19.9954% |
| C17 | `patient_060_node_0` | 7,171 | 1,434 | 19.9972% |
| BRACS | `BRACS_1003691` | 6,708 | 1,341 | 19.9911% |

These counts equal `max(1, int(0.20 * N))`, and attention sums are approximately 1.0. They describe individual slides rather than dataset medians. Every patch is still projected and scored before selection; these results do not demonstrate cheaper feature extraction or removal of full-bag attention scoring. Batch size 1 avoids padding in the reported runs; the notebook model does not apply the collate helper's mask for variable-length padded batches.

The C16 notebook also contains qualitative attention overlays for `test_068` and `test_067`. Overlay colors show within-slide attention percentiles, and the default 512-pixel level-0 patch footprint is an assumption. These figures do not provide quantitative localization accuracy or establish that high attention identifies tumor tissue correctly.

The available evidence supports a C16 improvement, mixed C17 results, and a BRACS decline at this ratio. These are single-seed observations at one ratio; they provide no confidence intervals, statistical-significance claim, or evidence that 20% is optimal. Ratio sensitivity should be assessed with validation data before final test evaluation. Further seeds and patient-grouped validation would address additional uncertainties.

The current notebooks report only AUROC, accuracy, macro-F1, and cross-entropy loss. They do not provide the current per-class recalls, paired error counts, Brier scores, ECE, per-slide prediction exports, pairing manifests, or metric-recomputation artifacts. Consequently, the earlier fixed-128 figures and claims based on those artifacts have been removed from this current-ratio discussion. The selected model state and history remain in notebook memory unless saved separately; the stored cell outputs are the evidence summarized here.
