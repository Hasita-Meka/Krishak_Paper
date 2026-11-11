"""
config.py — Shared configuration for the Krishak pipeline.
All scripts import from here to ensure consistency.
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.environ.get("KRISHAK_DATASET_DIR", os.path.join(BASE_DIR, "Dataset"))
TRAIN_CSV       = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMAGES    = os.path.join(DATASET_DIR, "train_images")
TEST_IMAGES     = os.path.join(DATASET_DIR, "test_images")

CLEANED_CSV     = os.path.join(DATASET_DIR, "cleaned_train.csv")
BALANCED_CSV    = os.path.join(DATASET_DIR, "balanced_train.csv")

# Revision runs write to outputs/<RUN_TAG>/ so the original submission results
# in outputs/ are never overwritten. Set KRISHAK_RUN_TAG=v1 to use the old folder.
RUN_TAG         = os.environ.get("KRISHAK_RUN_TAG", "v2_leakfree")
_OUT_ROOT       = os.environ.get("KRISHAK_OUTPUT_ROOT", os.path.join(BASE_DIR, "outputs"))
OUTPUT_DIR      = _OUT_ROOT if RUN_TAG == "v1" else os.path.join(_OUT_ROOT, RUN_TAG)
MODELS_DIR      = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR        = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGE_SIZE  = 224          # Input resolution for all models
# SEED drives training randomness (weights init, batch order, augmentation, searches)
# and can be changed per run for the multi-seed study (R1, R5 #15).
# SPLIT_SEED fixes the train/val/test partition so every seed sees the same split.
SEED        = int(os.environ.get("KRISHAK_SEED", 42))
SPLIT_SEED  = 42
# Nested CV (R1, R5 #2): when KRISHAK_OUTER_FOLD=k (1..CV_FOLDS), the held-out
# "test" set becomes outer fold k of the original photographs and the whole
# pipeline (backbone, QIL, HFSEA, GNAS, head) is fitted on the other folds only.
OUTER_FOLD  = int(os.environ["KRISHAK_OUTER_FOLD"]) if os.environ.get("KRISHAK_OUTER_FOLD") else None
VAL_SPLIT   = 0.15         # 15 % of training data → validation
TEST_SPLIT  = 0.10         # 10 % of training data → held-out test

# ── Training hyperparameters ───────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_WORKERS     = 8
PIN_MEMORY      = True
NUM_EPOCHS      = 60
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
MOMENTUM        = 0.9

# Scheduler
LR_SCHEDULER    = "cosine"   # "cosine" | "step" | "plateau"
T_MAX           = NUM_EPOCHS
ETA_MIN         = 1e-6

# Regularisation
DROPOUT         = 0.4
LABEL_SMOOTHING = 0.1

# Early stopping
PATIENCE        = 12
MIN_DELTA       = 1e-4

# Mixed precision
USE_AMP         = True       # FP16 via torch.cuda.amp

# ── Model names ────────────────────────────────────────────────────────────
ALL_MODEL_NAMES = [
    "densenet161",
    "googlenet",
    "mobilenet_v3_large",
    "resnet50",
    "shufflenet_v2",
]
# KRISHAK_MODELS=mobilenet_v3_large,resnet50 restricts a run to those backbones
MODEL_NAMES = ([m.strip() for m in os.environ["KRISHAK_MODELS"].split(",") if m.strip()]
               if os.environ.get("KRISHAK_MODELS") else list(ALL_MODEL_NAMES))

# ── QEFS hyperparameters ───────────────────────────────────────────────────
QEFS_FEATURE_DIM    = 512        # dimension after quantum projection
QEFS_POP_SIZE       = 30         # HFSEA population size
QEFS_MAX_ITER       = 50         # HFSEA iterations
QEFS_ALPHA          = 0.5        # absorption coefficient (firefly)
QEFS_GAMMA          = 1.0        # light absorption coefficient
QEFS_BETA0          = 1.0        # attractiveness at r=0
QEFS_DELTA          = 0.97       # random walk step decay (swallow)
QEFS_FINETUNE_EPOCHS = 20        # quantum fine-tune epochs after HFSEA

# ── Genetic NAS hyperparameters ────────────────────────────────────────────
GNAS_POP_SIZE       = 20         # population size
GNAS_GENERATIONS    = 15         # number of generations
GNAS_CROSSOVER_RATE = 0.8
GNAS_MUTATION_RATE  = 0.15
GNAS_TOURNAMENT_K   = 3          # tournament selection size
GNAS_EVAL_EPOCHS    = 10         # epochs per candidate evaluation

# ── Continual Learning ─────────────────────────────────────────────────────
CL_N_TASKS          = 5                    # split 10 classes into 5 tasks (2 classes each)
CL_BUFFER_SIZE      = int(os.environ.get("KRISHAK_CL_BUFFER", 2000))  # replay buffer size; sweep for R1 / R5 #5
CL_EPOCHS_PER_TASK  = 50                   # more epochs per task for stronger convergence
CL_MODEL_NAME       = os.environ.get("KRISHAK_CL_MODEL", "mobilenet_v3_large")
CL_LEARNING_RATE    = 5e-4                 # lower LR for CL fine-tuning (prevents instability)
# Backbone during CL (R5 #3): "1" frozen (paper setting), "partial" = last ~20% of
# backbone parameter tensors trainable, "0" = whole backbone trainable.
CL_FREEZE_MODE      = os.environ.get("KRISHAK_CL_FREEZE", "1")
CL_FREEZE_BACKBONE  = CL_FREEZE_MODE != "0"
# Task order (R1, R5 #4): 0 = original alphabetical pairing; k > 0 = random class
# permutation drawn with seed k, then paired into CL_N_TASKS tasks of 2 classes.
CL_TASK_ORDER       = int(os.environ.get("KRISHAK_CL_ORDER", 0))

# ── Cross-validation ───────────────────────────────────────────────────────
CV_FOLDS            = 5
SKIP_CV             = os.environ.get("KRISHAK_SKIP_CV") == "1"   # nested / multi-seed runs skip inner CV
CV_FOLD_EPOCHS      = int(os.environ.get("KRISHAK_CV_EPOCHS", 15))  # epochs per CV fold; set 15/30/60 for the R5 sensitivity check

# ── SHAP ───────────────────────────────────────────────────────────────────
SHAP_BACKGROUND     = 50         # background samples for DeepSHAP
SHAP_TEST_SAMPLES   = 20

# ── Hardware ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only print once from the main process (not from each DataLoader worker)
import multiprocessing as _mp
if _mp.current_process().name == "MainProcess":
    print(f"[config] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[config] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[config] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Smoke-test mode (KRISHAK_SMOKE=1): tiny CPU run to check the code paths ──
SMOKE = os.environ.get("KRISHAK_SMOKE") == "1"
if SMOKE:
    NUM_EPOCHS = 1; T_MAX = 1; PATIENCE = 2; CV_FOLD_EPOCHS = 1
    QEFS_POP_SIZE = 4; QEFS_MAX_ITER = 1; QEFS_FINETUNE_EPOCHS = 1
    GNAS_POP_SIZE = 3; GNAS_GENERATIONS = 1; GNAS_EVAL_EPOCHS = 1
    CL_EPOCHS_PER_TASK = 1; BATCH_SIZE = 8; NUM_WORKERS = 0
    SHAP_BACKGROUND = 4; SHAP_TEST_SAMPLES = 2
if not torch.cuda.is_available():
    # CPU-only (smoke tests): disable AMP and make CUDA memory probes no-ops
    USE_AMP = False; PIN_MEMORY = False
    for _fn in ("reset_peak_memory_stats", "empty_cache", "synchronize"):
        setattr(torch.cuda, _fn, lambda *a, **k: None)
    for _fn in ("max_memory_allocated", "memory_allocated", "max_memory_reserved", "memory_reserved"):
        setattr(torch.cuda, _fn, lambda *a, **k: 0)

