#!/usr/bin/env bash
# =============================================================================
#  Krishak — Full Pipeline Runner
#  Run from the project root folder:  bash run_pipeline.sh
# =============================================================================

set -e

PYTHON=${PYTHON:-python}
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

echo ""
echo "============================================================"
echo "  Krishak Pipeline"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: $($PYTHON --version)"
echo "============================================================"

# ── 0. Install dependencies ───────────────────────────────────────────────
echo ""
echo "[0] Installing dependencies …"
$PYTHON -m pip install -q -r requirements.txt
# For CUDA 12.x + RTX 5080, install torch separately if needed:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ── 1. Data Cleaning ──────────────────────────────────────────────────────
echo ""
echo "[1] Data Cleaning …"
$PYTHON 01_data_cleaning.py 2>&1 | tee "$LOG_DIR/01_cleaning.log"
echo "    ✓ Done. Cleaned CSV: Dataset/cleaned_train.csv"

# ── 2. Data Balancing ─────────────────────────────────────────────────────
echo ""
echo "[2] Data Balancing …"
$PYTHON 02_data_balancing.py 2>&1 | tee "$LOG_DIR/02_balancing.log"
echo "    ✓ Done. Balanced CSV: Dataset/balanced_train.csv"

# ── 3. Train Baseline CNN Models ──────────────────────────────────────────
echo ""
echo "[3] Training 5 CNN Models …"
echo "    (DenseNet161, GoogLeNet, MobileNet_v3_large, ResNet50, ShuffleNet_v2)"
$PYTHON 03_train_cnn_models.py 2>&1 | tee "$LOG_DIR/03_cnn.log"
echo "    ✓ Done. Models: outputs/models/"

# ── 4. QEFS ───────────────────────────────────────────────────────────────
echo ""
echo "[4] QEFS — Quantum-Inspired Evolutionary Feature Selection …"
$PYTHON 04_qefs.py 2>&1 | tee "$LOG_DIR/04_qefs.log"
echo "    ✓ Done. Results: outputs/results/qefs_*.json"

# ── 5. Genetic NAS ────────────────────────────────────────────────────────
echo ""
echo "[5] Genetic NAS …"
$PYTHON 05_genetic_nas.py 2>&1 | tee "$LOG_DIR/05_gnas.log"
echo "    ✓ Done. Results: outputs/results/gnas_*.json"

# ── 6. Continual Learning ─────────────────────────────────────────────────
echo ""
echo "[6] Continual Learning (GDumb, DER++, X-DER, A-GEM, ER-ACE) …"
$PYTHON 06_continual_learning.py 2>&1 | tee "$LOG_DIR/06_cl.log"
echo "    ✓ Done. Results: outputs/results/cl_*.json"

# ── 7. Comparison Plots ───────────────────────────────────────────────────
echo ""
echo "[7] Generating Comparison Plots …"
$PYTHON 07_comparison_plots.py 2>&1 | tee "$LOG_DIR/07_plots.log"
echo "    ✓ Done. Plots: outputs/plots/"

# ── 8. Fairness & Explainability ──────────────────────────────────────────
echo ""
echo "[8] Fairness Analysis & Explainability (SHAP, Grad-CAM, XAI) …"
$PYTHON 08_fairness_explainability.py 2>&1 | tee "$LOG_DIR/08_fairness_xai.log"
echo "    ✓ Done. Results: outputs/results/fairness_report.json"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Models:  outputs/models/"
echo "  Results: outputs/results/"
echo "  Plots:   outputs/plots/"
echo "  Logs:    outputs/logs/"
echo "  Leaderboard: outputs/results/leaderboard.csv"
echo "============================================================"
echo ""
