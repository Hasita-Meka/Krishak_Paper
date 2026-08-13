# Krishak — Rice Disease Classification Pipeline

A full end-to-end deep learning pipeline for classifying **10 rice leaf classes (nine diseases plus healthy)** using:
- **Baseline CNNs** (5 architectures with full diagnostics)
- **QEFS** — Quantum-Inspired Evolutionary Feature Selection
- **Genetic NAS** — Genetic Algorithm Neural Architecture Search
- **Continual Learning** — 5 methods (GDumb, DER++, X-DER, A-GEM, ER-ACE)
- **Fairness & Explainability** — SHAP, Grad-CAM, Integrated Gradients, per-class fairness metrics

Companion code for the paper *"Krishak: Deep Continual Learning Approaches for Paddy Disease Classification"* (Meka, Mishra, Sumathi D, Chauhan, Funde — submitted to Information Processing in Agriculture).

---

## Dataset

**Source:** [Paddy Doctor — Paddy Disease Classification (Kaggle)](https://www.kaggle.com/competitions/paddy-disease-classification)

**10 classes:** `bacterial_leaf_blight`, `bacterial_leaf_streak`, `bacterial_panicle_blight`, `blast`, `brown_spot`, `dead_heart`, `downy_mildew`, `hispa`, `normal`, `tungro`

**Expected folder structure after download:**

```
Krishak/
├── Dataset/
│   ├── train.csv
│   ├── train_images/
│   │   ├── bacterial_leaf_blight/
│   │   ├── blast/
│   │   └── ... (10 class folders)
│   └── test_images/
```

---

## Requirements

- Python 3.10+
- CUDA 12.x (NVIDIA GPU strongly recommended — tested on RTX 5080 16 GB)
- Conda or virtualenv

### Install dependencies

```bash
# Option A — pip
pip install -r requirements.txt

# Option B — Conda (recommended)
conda create -n krishak python=3.10 -y
conda activate krishak
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Full Pipeline

Run phases **in order**. Each phase saves checkpoints and results to `outputs/` so you can resume if interrupted.

### Phase 1 — Data Cleaning

Removes corrupt/duplicate images, validates file integrity, saves `Dataset/cleaned_train.csv`.

```bash
python 01_data_cleaning.py
```

### Phase 2 — Data Balancing

Balances all 10 classes to 1,700 images each via augmentation, saves `Dataset/balanced_train.csv` and augmented images to `Dataset/aug_images/`.

```bash
python 02_data_balancing.py
```

### Phase 3 — Baseline CNN Training

Trains 5 CNN architectures: **DenseNet-161, GoogLeNet, MobileNetV3-Large, ResNet-50, ShuffleNetV2**.

Each model produces: train/val/test accuracy, F1-score, confusion matrix, Grad-CAM heatmaps, SHAP explanations, confidence intervals, 5-fold cross-validation, memory profile, inference time.

```bash
python 03_train_cnn_models.py
```

**Key hyperparameters** (edit `config.py` to change):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_EPOCHS` | 60 | Training epochs |
| `BATCH_SIZE` | 32 | Batch size |
| `LEARNING_RATE` | 1e-3 | Initial LR (CosineAnnealing scheduler) |
| `PATIENCE` | 12 | Early stopping patience |
| `CV_FOLDS` | 5 | Cross-validation folds |

### Phase 4 — QEFS (Quantum-Inspired Evolutionary Feature Selection)

Applies a Hybrid Firefly–Swallow Evolutionary Algorithm (HFSEA) to select the most discriminative features from 512-dim quantum projections on top of each CNN backbone.

```bash
python 04_qefs.py
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QEFS_FEATURE_DIM` | 512 | Quantum projection dimension |
| `QEFS_POP_SIZE` | 30 | Firefly population size |
| `QEFS_MAX_ITER` | 50 | HFSEA iterations |
| `QEFS_FINETUNE_EPOCHS` | 20 | Quantum fine-tune epochs |

### Phase 5 — Genetic NAS

Uses a Genetic Algorithm to search for the optimal classifier head architecture on top of QEFS-enhanced backbones. Each candidate is evaluated on validation accuracy.

```bash
python 05_genetic_nas.py
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GNAS_POP_SIZE` | 20 | GA population size |
| `GNAS_GENERATIONS` | 15 | Number of GA generations |
| `GNAS_EVAL_EPOCHS` | 10 | Epochs per candidate evaluation |
| `GNAS_CROSSOVER_RATE` | 0.8 | Crossover probability |
| `GNAS_MUTATION_RATE` | 0.15 | Mutation probability |

### Phase 6 — Continual Learning

Splits 10 classes into 5 sequential tasks (2 classes each) and trains 5 continual learning methods on the GNAS-optimised backbone. Measures Average Accuracy (AA), Backward Transfer (BWT), and Forward Transfer (FWT).

```bash
python 06_continual_learning.py
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CL_MODEL_NAME` | `mobilenet_v3_large` | Backbone for all CL methods (best result, frozen-backbone strategy) |
| `CL_N_TASKS` | 5 | Number of sequential tasks |
| `CL_EPOCHS_PER_TASK` | 50 | Epochs trained per task |
| `CL_BUFFER_SIZE` | 2000 | Replay buffer size |
| `CL_LEARNING_RATE` | 5e-4 | Lower LR for CL fine-tuning |
| `CL_FREEZE_BACKBONE` | `True` | Freeze backbone during CL to prevent catastrophic forgetting |

### Phase 7 — Comparison Plots

Generates comprehensive cross-phase comparison plots across all methods and models.

```bash
python 07_comparison_plots.py
```

### Phase 8 — Fairness & Explainability

Loads all trained models (Baseline, QEFS, GNAS, CL) and generates:

- **Fairness** (per class treated as a "group"): Equal Opportunity, Equalized Odds, Demographic Parity, Disparate Impact Ratio, Statistical Parity Difference, Theil Index, ROC/PR curves, calibration, cross-model fairness heatmaps.
- **Explainability (XAI):** SHAP (GradientExplainer), Grad-CAM / Grad-CAM++ / EigenCAM / Score-CAM, Integrated Gradients, saliency maps, occlusion sensitivity, t-SNE/UMAP embeddings, LIME.
- Re-generates any missing plots from phases 3–7 and produces the grand accuracy leaderboard with confidence intervals.

```bash
python 08_fairness_explainability.py
```

Optional extras: `pip install shap lime scikit-learn umap-learn captum`

---

## Pipeline Chain

```
Dataset
   │
   ▼
01_data_cleaning.py        → Dataset/cleaned_train.csv
   │
   ▼
02_data_balancing.py       → Dataset/balanced_train.csv
   │
   ▼
03_train_cnn_models.py     → outputs/models/{model}_best.pth
   │                          outputs/results/{model}.json
   ▼
04_qefs.py                 → outputs/models/qefs_{model}_best.pth
   │                          outputs/results/qefs_{model}.json
   ▼
05_genetic_nas.py          → outputs/models/gnas_{model}_best.pth
   │                          outputs/results/gnas_{model}.json
   ▼
06_continual_learning.py   → outputs/results/cl_{method}.json
   │
   ▼
07_comparison_plots.py     → outputs/plots/*.png
   │
   ▼
08_fairness_explainability.py → outputs/results/fairness_report.json
                                 outputs/plots/xai/*.png, outputs/plots/fairness/*.png
```

---

## Outputs

All results are saved under `outputs/`:

```
outputs/
├── models/          # .pth checkpoints (baseline, QEFS, GNAS, CL) — NOT tracked in git, see below
├── results/         # .json/.csv metrics per model/method — tracked in git (~1.5 MB)
├── plots/           # PNG figures (training curves, confusion matrices,
│                    #   SHAP, Grad-CAM, feature importance, CL curves, fairness, etc.)
│                    #   — NOT tracked in git (~210 MB); regenerate via 07/08 scripts
└── logs/            # Training logs — NOT tracked in git
```

**Pretrained checkpoints:** `outputs/models/*.pth` (~2.4 GB total) are excluded from this repository via `.gitignore` because of GitHub's file-size limits. To reproduce results without retraining, either run the pipeline end-to-end (Phases 3–6 regenerate every checkpoint) or contact the author for the trained weights.

---

## Validation Test

After training all phases, run the validation suite to confirm everything loaded correctly — it dynamically checks synthetic + real checkpoint loading across the QEFS→GNAS→CL chain, script syntax for every pipeline stage, and integrity of existing `.pth` checkpoints, printing a pass/fail count at the end:

```bash
python test_pipeline.py
```

---

## Project Structure

```
Krishak/
├── config.py                  # All hyperparameters and paths
├── utils.py                   # Shared dataset, training loops, metrics, EarlyStopping
├── 01_data_cleaning.py
├── 02_data_balancing.py
├── 03_train_cnn_models.py
├── 04_qefs.py
├── 05_genetic_nas.py
├── 06_continual_learning.py
├── 07_comparison_plots.py
├── 08_fairness_explainability.py
├── quick_test_pipeline.py     # End-to-end smoke test (reduced settings)
├── test_pipeline.py           # Post-training validation suite
├── requirements.txt
├── run_pipeline.sh            # Bash script to run all phases sequentially
├── LICENSE
├── README.md
└── Dataset/
    ├── train.csv
    ├── train_images/
    └── test_images/
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{meka_krishak,
  title   = {Krishak: Deep Continual Learning Approaches for Paddy Disease Classification},
  author  = {Meka, Hasita Chowdary and Mishra, Sanket and Sumathi, D and Chauhan, Hardik and Funde, Nitesh A.},
  journal = {Information Processing in Agriculture},
  note    = {Manuscript submitted}
}
```

Update this entry with the final venue/DOI once the paper is published.

---

## License

Released under the [MIT License](LICENSE). The dataset itself is subject to the terms of the [Paddy Doctor Kaggle competition](https://www.kaggle.com/competitions/paddy-disease-classification), not this repository's license.
