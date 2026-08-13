"""
05_genetic_nas.py — Genetic Neural Architecture Search on QEFS Backbones
=========================================================================
Applies a Genetic Algorithm (GA) to search for the optimal classification
head architecture on top of the trained QEFS (Quantum-Enhanced Feature
Space) backbones, one model at a time.

Pipeline:
  1. Load QEFS backbone from qefs_{model}_best.pth  (falls back to
     baseline CNN if QEFS checkpoint is absent)
  2. Run GA over SearchableHead hyperparameters:
       n_layers ∈ {1,2,3}, hidden_dim ∈ {128,256,512},
       dropout ∈ {0.2,0.3,0.4,0.5}, activation ∈ {relu,gelu,silu},
       use_bn, use_skip, lr_mult ∈ {0.5,1.0,2.0}
  3. Full fine-tune of the best architecture found
  4. Comprehensive evaluation:
       train/val/test accuracy, class-wise accuracies, F1-score,
       confusion matrix, feature-importance heat-map (GradCAM),
       classification report, accuracy vs loss curves, inference time,
       training time, confidence intervals, 5-fold cross-validation,
       memory profiling, SHAP explainability plots

Outputs (per model):
  outputs/models/gnas_{model}_best.pth
  outputs/results/gnas_{model}.json
  outputs/plots/gnas_{model}_*.png

Run:
  python 05_genetic_nas.py

Hardware note: designed for AMD Ryzen 9 9955HX3D + NVIDIA RTX 5080 (16 GB
VRAM, 64 GB RAM) with CUDA + AMP enabled.
"""

# ── Stdlib ──────────────────────────────────────────────────────────────────
import os, sys, json, time, copy, random, warnings, gc, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tracemalloc, psutil

# ── Global font-size settings (labels must be readable in print) ─────────────
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.titlesize": 18,
    "axes.titlepad":    10,
    "xtick.major.pad":   6,
    "ytick.major.pad":   6,
})

import seaborn as sns  # ensure seaborn is imported for theme
sns.set_theme(style="whitegrid", font_scale=1.15)
sns.set_palette("colorblind")


# Unified color palette across all scripts
KRISHAK_COLORS = {
    "Baseline":  "#1f77b4",   # blue
    "QEFS":      "#ff7f0e",   # orange
    "GNAS":      "#2ca02c",   # green
    "CL":        "#d62728",   # red
    "Fairness":  "#9467bd",   # purple
    "XAI":       "#8c564b",   # brown
}
KRISHAK_PALETTE = list(KRISHAK_COLORS.values())
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# ── Project imports ──────────────────────────────────────────────────────────
from config import *
from utils import (
    set_seed, get_model, get_feature_extractor,
    PlantDiseaseDataset, split_dataframe, compute_class_weights,
    get_train_transform, get_val_transform,
    train_one_epoch, evaluate, EarlyStopping,
    compute_full_metrics, bootstrap_confidence_interval,
    profile_memory, compute_and_save_shap, grad_cam_heatmap,
    plot_training_curves, plot_confusion_matrix, plot_class_accuracies,
    save_results, load_results, full_train,
)

set_seed()

# ════════════════════════════════════════════════════════════════════════════
# Raw CNN feature dimensions (output of get_feature_extractor)
# ════════════════════════════════════════════════════════════════════════════
_cnn_feat_dims: Dict[str, int] = {
    "densenet161":         2208,
    "googlenet":           1024,
    "mobilenet_v3_large":  1280,
    "resnet50":            2048,
    "shufflenet_v2":       1024,
}


# ════════════════════════════════════════════════════════════════════════════
# QEFS Building Blocks
# (mirrored here so this script is self-contained; also used by 04_qefs.py)
# ════════════════════════════════════════════════════════════════════════════
class QuantumInspiredLayer(nn.Module):
    """
    Quantum-Inspired transformation layer.
    Maps raw CNN features → compressed QEFS feature space via
    a learned entanglement projection (Linear → BN → ReLU).

    Checkpoint key prefix: q_layer
    Sub-keys: ent_fc, bn, act
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ent_fc = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.ent_fc(x)))


class QCNNHead(nn.Module):
    """
    Full QEFS model: CNN backbone + QuantumInspiredLayer + classifier.
    Checkpoint key prefix: backbone (CNN), q_layer, dropout, head
    Used by 04_qefs.py; reconstructed here for weight loading.
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, q_dim: int):
        super().__init__()
        self.backbone = backbone
        self.q_layer  = QuantumInspiredLayer(feat_dim, q_dim)
        self.dropout  = nn.Dropout(DROPOUT)
        self.head     = nn.Linear(q_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        q = self.q_layer(f)
        return self.head(self.dropout(q))


def get_backbone_dim(model_name: str) -> int:
    """Return raw CNN feature dimension for a given model."""
    return _cnn_feat_dims[model_name]


# ════════════════════════════════════════════════════════════════════════════
# QEFS Feature Extractor (backbone for GNAS)
# ════════════════════════════════════════════════════════════════════════════
class QEFSFeatureExtractor(nn.Module):
    """
    Wraps a trained QEFS backbone to return QEFS_FEATURE_DIM-dim features.
    Checkpoint key prefix in GNASModel: backbone
      backbone.backbone.*  → CNN feature extractor
      backbone.q_layer.*   → QuantumInspiredLayer
    """
    def __init__(self, backbone: nn.Module, q_layer: QuantumInspiredLayer):
        super().__init__()
        self.backbone = backbone
        self.q_layer  = q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        return self.q_layer(f)


# ════════════════════════════════════════════════════════════════════════════
# Searchable Classification Head
# ════════════════════════════════════════════════════════════════════════════
class SearchableHead(nn.Module):
    """
    Configurable MLP head whose architecture is determined by a chromosome.
    Checkpoint key prefix in GNASModel: head
      head.fcs.*, head.bns.*, head.acts.*, head.dropout, head.head

    Args:
        in_dim : feature dimension (QEFS_FEATURE_DIM)
        arch   : dict with keys n_layers, hidden_dim, dropout,
                 activation, use_bn, use_skip, lr_mult
    """
    def __init__(self, in_dim: int, arch: Dict):
        super().__init__()
        n_layers   = arch["n_layers"]
        hidden_dim = arch["hidden_dim"]
        dropout    = arch["dropout"]
        activation = arch["activation"]
        use_bn     = arch["use_bn"]

        self.fcs     = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.acts    = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.use_bn  = use_bn
        self.use_skip= arch["use_skip"]

        _act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        prev_dim = in_dim
        for _ in range(n_layers):
            self.fcs.append(nn.Linear(prev_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.acts.append(_act_fn[activation]())
            prev_dim = hidden_dim

        self.head = nn.Linear(prev_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, fc in enumerate(self.fcs):
            residual = out
            out = self.dropout(self.acts[i](self.bns[i](fc(out))))
            if self.use_skip and out.shape == residual.shape:
                out = out + residual
        return self.head(out)


# ════════════════════════════════════════════════════════════════════════════
# Full GNAS Model
# ════════════════════════════════════════════════════════════════════════════
class GNASModel(nn.Module):
    """backbone (QEFSFeatureExtractor) + head (SearchableHead)."""
    def __init__(self, backbone: QEFSFeatureExtractor, arch: Dict):
        super().__init__()
        self.backbone = backbone
        self.head     = SearchableHead(QEFS_FEATURE_DIM, arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ════════════════════════════════════════════════════════════════════════════
# Chromosome encoding / decoding
# ════════════════════════════════════════════════════════════════════════════
_N_LAYERS    = [1, 2, 3]
_HIDDEN_DIMS = [128, 256, 512]
_DROPOUTS    = [0.2, 0.3, 0.4, 0.5]
_ACTIVATIONS = ["relu", "gelu", "silu"]
_LR_MULTS    = [0.5, 1.0, 2.0]
CHROM_GENES  = [len(_N_LAYERS), len(_HIDDEN_DIMS), len(_DROPOUTS),
                len(_ACTIVATIONS), 2, 2, len(_LR_MULTS)]  # 2 = boolean


def random_chromosome() -> List[int]:
    return [random.randint(0, g - 1) for g in CHROM_GENES]


def decode_chromosome(chrom: List[int]) -> Dict:
    return {
        "n_layers":   _N_LAYERS   [chrom[0]],
        "hidden_dim": _HIDDEN_DIMS[chrom[1]],
        "dropout":    _DROPOUTS   [chrom[2]],
        "activation": _ACTIVATIONS[chrom[3]],
        "use_bn":     bool(chrom[4]),
        "use_skip":   bool(chrom[5]),
        "lr_mult":    _LR_MULTS   [chrom[6]],
    }


def crossover(p1: List[int], p2: List[int], rate: float = GNAS_CROSSOVER_RATE) -> Tuple[List[int], List[int]]:
    if random.random() > rate:
        return p1[:], p2[:]
    pt = random.randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]


def mutate(chrom: List[int], rate: float = GNAS_MUTATION_RATE) -> List[int]:
    child = chrom[:]
    for i, max_val in enumerate(CHROM_GENES):
        if random.random() < rate:
            child[i] = random.randint(0, max_val - 1)
    return child


def tournament_select(population: List, fitnesses: List[float], k: int = GNAS_TOURNAMENT_K) -> List[int]:
    idxs = random.sample(range(len(population)), k)
    return population[max(idxs, key=lambda i: fitnesses[i])]


# ════════════════════════════════════════════════════════════════════════════
# Build model helper
# ════════════════════════════════════════════════════════════════════════════
def build_model_with_arch(
    model_name: str,
    arch: Dict,
    pretrained_path: Optional[str] = None,
    qefs_path: Optional[str]       = None,
) -> GNASModel:
    """
    Construct a GNASModel for `model_name` with the given `arch`.

    Priority:
      1. Load QEFS backbone from `qefs_path` (backbone.* + q_layer.* keys)
      2. Fall back to baseline CNN from `pretrained_path`
      3. Fall back to ImageNet-pretrained weights
    """
    if qefs_path and os.path.exists(qefs_path):
        ckpt    = torch.load(qefs_path, map_location=DEVICE, weights_only=False)
        raw_dim = _cnn_feat_dims[model_name]
        q_dim   = int(ckpt.get("q_dim", QEFS_FEATURE_DIM))

        # Reconstruct CNN feature extractor
        base_full = get_model(model_name, pretrained=False).to(DEVICE)
        base_ext  = get_feature_extractor(base_full, model_name).to(DEVICE)
        q_layer   = QuantumInspiredLayer(raw_dim, q_dim).to(DEVICE)
        extractor = QEFSFeatureExtractor(base_ext, q_layer).to(DEVICE)

        state    = ckpt["model_state"]
        # Strategy 1: Use QCNNHead as a proxy for structured loading (backbone.* / q_layer.*)
        try:
            _proxy = QCNNHead(get_feature_extractor(
                get_model(model_name, pretrained=False).to(DEVICE), model_name
            ).to(DEVICE), raw_dim, q_dim).to(DEVICE)
            _proxy.load_state_dict(state, strict=False)
            extractor.backbone.load_state_dict(
                _proxy.backbone.state_dict(), strict=True)
            extractor.q_layer.load_state_dict(
                _proxy.q_layer.state_dict(), strict=True)
            del _proxy
            print(f"    [GNAS] QEFS backbone loaded via QCNNHead proxy "
                  f"from {os.path.basename(qefs_path)}")
        except Exception as _e1:
            # Strategy 2: manual key-prefix stripping
            try:
                bb_state = {k[len("backbone."):]: v
                            for k, v in state.items() if k.startswith("backbone.")}
                ql_state = {k[len("q_layer."):]: v
                            for k, v in state.items() if k.startswith("q_layer.")}
                if not bb_state or not ql_state:
                    raise RuntimeError(
                        f"QEFS checkpoint has no backbone.* or q_layer.* keys. "
                        f"Found: {list(state.keys())[:10]}"
                    )
                extractor.backbone.load_state_dict(bb_state, strict=True)
                extractor.q_layer.load_state_dict(ql_state, strict=True)
                print(f"    [GNAS] QEFS backbone loaded via key-strip "
                      f"from {os.path.basename(qefs_path)}")
            except Exception as _e2:
                raise RuntimeError(
                    f"Failed to load QEFS checkpoint '{qefs_path}' "
                    f"for {model_name}.\n"
                    f"  Strategy 1 (proxy): {_e1}\n"
                    f"  Strategy 2 (strip): {_e2}"
                ) from _e2
        uses_qefs = True

    else:
        # No QEFS checkpoint — fall back to baseline CNN
        use_pretrained = (pretrained_path is None)
        base_full = get_model(model_name, pretrained=use_pretrained).to(DEVICE)
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt_b = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
            base_full.load_state_dict(ckpt_b["model_state"])
            print(f"    [GNAS] Loaded baseline backbone from {os.path.basename(pretrained_path)}")
        base_ext  = get_feature_extractor(base_full, model_name).to(DEVICE)
        q_layer   = QuantumInspiredLayer(_cnn_feat_dims[model_name], QEFS_FEATURE_DIM).to(DEVICE)
        extractor = QEFSFeatureExtractor(base_ext, q_layer).to(DEVICE)
        uses_qefs = False
        q_dim     = QEFS_FEATURE_DIM

    model = GNASModel(extractor, arch).to(DEVICE)
    model._uses_qefs = uses_qefs
    model._feat_dim  = q_dim
    return model


# ════════════════════════════════════════════════════════════════════════════
# Multi-directory Dataset (handles class-subfolder + flat aug layouts)
# ════════════════════════════════════════════════════════════════════════════
class MultiDirDataset(PlantDiseaseDataset):
    def __init__(self, df, dirs, transform=None):
        super().__init__(df, dirs[0], transform)
        self.dirs = dirs

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = row.get("label", "")
        for d in self.dirs:
            for p in [os.path.join(d, label, row["image_id"]),
                      os.path.join(d, row["image_id"])]:
                if os.path.exists(p):
                    img = Image.open(p).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, self.class_to_idx[label]
        raise FileNotFoundError(row["image_id"])


def get_image_dirs():
    aug = os.path.join(DATASET_DIR, "aug_images")
    dirs = [TRAIN_IMAGES]
    if os.path.isdir(aug):
        dirs.append(aug)
    return dirs


def make_loaders(train_df, val_df, test_df, dirs, batch_size=BATCH_SIZE):
    tr_ds = MultiDirDataset(train_df, dirs, get_train_transform())
    va_ds = MultiDirDataset(val_df,   dirs, get_val_transform())
    te_ds = MultiDirDataset(test_df,  dirs, get_val_transform())

    targets      = np.array([CLASS_TO_IDX[l] for l in train_df["label"]])
    cls_counts   = np.bincount(targets, minlength=NUM_CLASSES)
    cls_weights  = 1.0 / (cls_counts + 1e-6)
    samp_weights = cls_weights[targets]
    sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return tr_ld, va_ld, te_ld


# ════════════════════════════════════════════════════════════════════════════
# Quick candidate evaluation (few epochs on a subset)
# ════════════════════════════════════════════════════════════════════════════
def _quick_eval(
    model_name: str,
    arch: Dict,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    qefs_path:    Optional[str],
    baseline_path:Optional[str],
    n_epochs:     int = GNAS_EVAL_EPOCHS,
    freeze_backbone: bool = True,
) -> float:
    """Train a candidate for n_epochs; return best val accuracy."""
    model = build_model_with_arch(model_name, arch,
                                  pretrained_path=baseline_path,
                                  qefs_path=qefs_path)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    lr = LEARNING_RATE * arch["lr_mult"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=ETA_MIN)
    crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler= GradScaler(device="cuda", enabled=USE_AMP)
    es    = EarlyStopping(patience=max(3, n_epochs // 3), mode="max")

    best_val = 0.0
    for ep in range(n_epochs):
        train_one_epoch(model, train_loader, opt, crit, scaler)
        _, val_acc, _, _ = evaluate(model, val_loader, crit)
        sched.step()
        if val_acc > best_val:
            best_val = val_acc
        if es(val_acc):
            break

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return best_val


# ════════════════════════════════════════════════════════════════════════════
# Genetic Algorithm
# ════════════════════════════════════════════════════════════════════════════
def run_genetic_search(
    model_name:     str,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    qefs_path:      Optional[str],
    baseline_path:  Optional[str],
    pop_size:       int = GNAS_POP_SIZE,
    n_generations:  int = GNAS_GENERATIONS,
) -> Tuple[Dict, List[float]]:
    """
    Returns the best architecture dict and a list of best fitness per
    generation (for the convergence plot).
    """
    print(f"\n  ── Genetic Search for {model_name} "
          f"(pop={pop_size}, gen={n_generations}) ──")

    population   = [random_chromosome() for _ in range(pop_size)]
    best_arch    = None
    best_fitness = 0.0
    gen_best     = []

    for gen in range(n_generations):
        fitnesses = []
        for i, chrom in enumerate(population):
            arch = decode_chromosome(chrom)
            fit  = _quick_eval(model_name, arch,
                               train_loader, val_loader,
                               qefs_path, baseline_path,
                               n_epochs=GNAS_EVAL_EPOCHS)
            fitnesses.append(fit)
            print(f"    Gen {gen+1:02d}/{n_generations}  "
                  f"ind {i+1:02d}/{pop_size}  fit={fit:.4f}  arch={arch}")

        # Track best
        gen_max      = max(fitnesses)
        gen_best_idx = fitnesses.index(gen_max)
        gen_best.append(gen_max)
        if gen_max > best_fitness:
            best_fitness = gen_max
            best_arch    = decode_chromosome(population[gen_best_idx])
        print(f"  → Gen {gen+1} best: {gen_max:.4f}  overall best: {best_fitness:.4f}")

        # Elitism: keep top-2
        sorted_pop = [c for _, c in sorted(zip(fitnesses, population),
                                           key=lambda x: -x[0])]
        new_pop = sorted_pop[:2]

        # Fill rest via tournament + crossover + mutation
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2))

        population = new_pop

    return best_arch, gen_best


# ════════════════════════════════════════════════════════════════════════════
# Full fine-tune of winner architecture
# ════════════════════════════════════════════════════════════════════════════
def full_finetune(
    model_name:    str,
    arch:          Dict,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    qefs_path:     Optional[str],
    baseline_path: Optional[str],
    class_weights: torch.Tensor,
) -> Tuple[nn.Module, Dict, List, List]:
    """
    Fine-tune the winning architecture for NUM_EPOCHS with full training
    regime (cosine LR, early stopping, AMP).

    Returns (model, best_ckpt_state, train_history, val_history).
    """
    print(f"\n  ── Full fine-tune of {model_name} with arch={arch} ──")

    model = build_model_with_arch(model_name, arch,
                                  pretrained_path=baseline_path,
                                  qefs_path=qefs_path)

    # Stage 1: train only the head (backbone frozen)
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    lr      = LEARNING_RATE * arch["lr_mult"]
    head_params = list(model.head.parameters())
    opt1    = torch.optim.AdamW(head_params, lr=lr, weight_decay=WEIGHT_DECAY)
    sched1  = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10, eta_min=ETA_MIN)
    crit    = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE),
                                   label_smoothing=LABEL_SMOOTHING)
    scaler  = GradScaler(device="cuda", enabled=USE_AMP)

    print("  Stage 1: warm-up head (10 epochs, backbone frozen) …")
    for ep in range(10):
        tl, ta = train_one_epoch(model, train_loader, opt1, crit, scaler)
        vl, va, _, _ = evaluate(model, val_loader, crit)
        sched1.step()
        print(f"    Ep {ep+1:02d}/10  train_loss={tl:.4f} train_acc={ta:.4f}  "
              f"val_loss={vl:.4f} val_acc={va:.4f}")

    # Stage 2: unfreeze and full fine-tune
    for p in model.backbone.parameters():
        p.requires_grad_(True)

    all_params = [
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": model.head.parameters(),     "lr": lr},
    ]
    opt2   = torch.optim.AdamW(all_params, weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=NUM_EPOCHS,
                                                         eta_min=ETA_MIN)
    es     = EarlyStopping(patience=PATIENCE, mode="max")

    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []
    best_val_acc = 0.0
    best_state   = None

    print(f"  Stage 2: full fine-tune ({NUM_EPOCHS} epochs, early stop p={PATIENCE}) …")
    t_train_start = time.time()

    for ep in range(NUM_EPOCHS):
        tl, ta = train_one_epoch(model, train_loader, opt2, crit, scaler)
        vl, va, _, _ = evaluate(model, val_loader, crit)
        sched2.step()

        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);   val_accs.append(va)

        if va > best_val_acc:
            best_val_acc = va
            best_state   = copy.deepcopy(model.state_dict())

        print(f"    Ep {ep+1:03d}/{NUM_EPOCHS}  "
              f"train_loss={tl:.4f} train_acc={ta:.4f}  "
              f"val_loss={vl:.4f} val_acc={va:.4f}  best={best_val_acc:.4f}")

        if es(va):
            print(f"  Early stopping at epoch {ep+1}.")
            break

    training_time = time.time() - t_train_start
    history = {
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc":  train_accs,   "val_acc":  val_accs,
        "training_time_s": training_time,
    }
    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ════════════════════════════════════════════════════════════════════════════
# Comprehensive Metrics
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def full_metrics(
    model:    nn.Module,
    loader:   DataLoader,
    prefix:   str = "test",
) -> Dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    t0 = time.time()
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    inf_time = (time.time() - t0) / len(all_labels) * 1000  # ms/sample

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs  = np.array(all_probs)

    acc        = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted= f1_score(y_true, y_pred, average="weighted", zero_division=0)
    class_accs = {}
    for ci, cname in enumerate(CLASS_NAMES):
        mask = y_true == ci
        class_accs[cname] = float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() > 0 else 0.0

    cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cr  = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                 zero_division=0, output_dict=True)
    return {
        f"{prefix}_accuracy":       acc,
        f"{prefix}_f1_macro":       f1_macro,
        f"{prefix}_f1_weighted":    f1_weighted,
        f"{prefix}_class_accs":     class_accs,
        f"{prefix}_confusion_matrix": cm.tolist(),
        f"{prefix}_classification_report": cr,
        f"{prefix}_inference_ms_per_sample": inf_time,
    }


# ════════════════════════════════════════════════════════════════════════════
# Cross-Validation
# ════════════════════════════════════════════════════════════════════════════
def cross_validate(
    model_name: str,
    arch:       Dict,
    df:         pd.DataFrame,
    dirs:       List[str],
    qefs_path:  Optional[str],
    baseline_path: Optional[str],
    n_folds:    int = CV_FOLDS,
    n_epochs:   int = CV_FOLD_EPOCHS,
) -> Dict:
    print(f"\n  ── 5-Fold CV for {model_name} ──")
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    y_all = df["label"].values
    fold_accs, fold_f1s = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y_all)):
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)
        tr_ld = DataLoader(
            MultiDirDataset(tr_df, dirs, get_train_transform()),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
        va_ld = DataLoader(
            MultiDirDataset(va_df, dirs, get_val_transform()),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
        model = build_model_with_arch(model_name, arch,
                                      pretrained_path=baseline_path,
                                      qefs_path=qefs_path)
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        opt    = torch.optim.AdamW(model.head.parameters(),
                                   lr=LEARNING_RATE * arch["lr_mult"],
                                   weight_decay=WEIGHT_DECAY)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=ETA_MIN)
        crit   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        scaler = GradScaler(device="cuda", enabled=USE_AMP)

        for ep in range(n_epochs):
            train_one_epoch(model, tr_ld, opt, crit, scaler)
            sched.step()

        _, va_acc, va_preds, va_labels = evaluate(model, va_ld, crit)
        va_f1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        fold_accs.append(va_acc)
        fold_f1s.append(va_f1)
        print(f"    Fold {fold+1}/{n_folds}  acc={va_acc:.4f}  f1={va_f1:.4f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 95 % confidence intervals via t-distribution
    def ci95(values):
        n   = len(values)
        m   = np.mean(values)
        sem = stats.sem(values)
        h   = sem * stats.t.ppf(0.975, df=n - 1)
        return float(m), float(h)

    acc_m, acc_h = ci95(fold_accs)
    f1_m,  f1_h  = ci95(fold_f1s)

    return {
        "cv_fold_accs":    fold_accs,
        "cv_fold_f1s":     fold_f1s,
        "cv_mean_acc":     acc_m,
        "cv_acc_ci95":     acc_h,
        "cv_mean_f1":      f1_m,
        "cv_f1_ci95":      f1_h,
    }


# ════════════════════════════════════════════════════════════════════════════
# Confidence intervals (bootstrap)
# ════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(y_true, y_pred, n_boot=500, ci=0.95) -> Dict:
    accs, f1s = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx  = np.random.randint(0, n, size=n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx],
                            average="macro", zero_division=0))
    alpha = 1 - ci
    return {
        "bootstrap_acc_mean": float(np.mean(accs)),
        "bootstrap_acc_lo":   float(np.percentile(accs, 100*alpha/2)),
        "bootstrap_acc_hi":   float(np.percentile(accs, 100*(1-alpha/2))),
        "bootstrap_f1_mean":  float(np.mean(f1s)),
        "bootstrap_f1_lo":    float(np.percentile(f1s, 100*alpha/2)),
        "bootstrap_f1_hi":    float(np.percentile(f1s, 100*(1-alpha/2))),
    }


# ════════════════════════════════════════════════════════════════════════════
# Memory profiling
# ════════════════════════════════════════════════════════════════════════════
def measure_memory(model: nn.Module, loader: DataLoader) -> Dict:
    tracemalloc.start()
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / 1e6

    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(loader))
        _ = model(imgs.to(DEVICE))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_after = proc.memory_info().rss / 1e6

    gpu_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    gpu_peak  = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    return {
        "cpu_ram_before_mb":    ram_before,
        "cpu_ram_after_mb":     ram_after,
        "tracemalloc_peak_mb":  peak / 1e6,
        "gpu_allocated_mb":     gpu_alloc,
        "gpu_peak_mb":          gpu_peak,
    }


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════
def plot_all(
    model_name: str,
    history:    Dict,
    metrics:    Dict,
    arch:       Dict,
    gen_best:   List[float],
    plots_dir:  str,
):
    tag = f"gnas_{model_name}"
    os.makedirs(plots_dir, exist_ok=True)

    # ── 1. Accuracy & Loss curves ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_acc"]) + 1)
    axes[0].plot(ep, history["train_acc"], label="Train")
    axes[0].plot(ep, history["val_acc"],   label="Val")
    axes[0].set_title(f"{model_name} — Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, history["train_loss"], label="Train")
    axes[1].plot(ep, history["val_loss"],   label="Val")
    axes[1].set_title(f"{model_name} — Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(True)
    fig.suptitle(f"GNAS {model_name} — Training Curves", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{tag}_curves.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)

    # ── 2. Confusion Matrix ────────────────────────────────────────────────
    cm = np.array(metrics.get("test_confusion_matrix",
                               metrics.get("val_confusion_matrix", [])))
    if cm.size:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"GNAS {model_name} — Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{tag}_confusion.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    # ── 3. Class-wise Accuracy Bar Chart ──────────────────────────────────
    class_accs = metrics.get("test_class_accs", metrics.get("val_class_accs", {}))
    if class_accs:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(class_accs)), list(class_accs.values()),
               color=sns.color_palette("husl", len(class_accs)))
        ax.set_xticks(range(len(class_accs)))
        ax.set_xticklabels(list(class_accs.keys()), rotation=45, ha="right")
        ax.set_title(f"GNAS {model_name} — Class-wise Accuracy", fontweight="bold")
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
        for i, v in enumerate(class_accs.values()):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{tag}_class_acc.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    # ── 4. GA Convergence Curve ────────────────────────────────────────────
    if gen_best:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, len(gen_best) + 1), gen_best, "o-", color="darkorange")
        ax.set_title(f"GNAS {model_name} — GA Convergence", fontweight="bold")
        ax.set_xlabel("Generation"); ax.set_ylabel("Best Val Accuracy")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{tag}_ga_convergence.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    # ── 5. Architecture Summary ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    rows = [[str(k), str(v)] for k, v in arch.items()]
    tbl  = ax.table(cellText=rows, colLabels=["Hyperparameter", "Value"],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)
    ax.set_title(f"GNAS {model_name} — Best Architecture", fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{tag}_arch_summary.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)

    # ── 6. F1 Heat-map per class ───────────────────────────────────────────
    cr = metrics.get("test_classification_report",
                     metrics.get("val_classification_report", {}))
    if cr:
        f1_per_class = {c: cr.get(c, {}).get("f1-score", 0) for c in CLASS_NAMES}
        fig, ax = plt.subplots(figsize=(12, 3))
        vals     = np.array(list(f1_per_class.values())).reshape(1, -1)
        im = ax.imshow(vals, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks([]); ax.set_title(f"GNAS {model_name} — F1-score Heatmap",
                                         fontweight="bold")
        for j, v in enumerate(f1_per_class.values()):
            ax.text(j, 0, f"{v:.3f}", ha="center", va="center", fontsize=12,
                    color="black" if v > 0.5 else "white")
        plt.colorbar(im, ax=ax, fraction=0.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{tag}_f1_heatmap.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    print(f"  Plots saved → {plots_dir}/{tag}_*.png")


# ════════════════════════════════════════════════════════════════════════════
# SHAP Explainability
# ════════════════════════════════════════════════════════════════════════════
def run_shap(
    model:    nn.Module,
    loader:   DataLoader,
    plots_dir:str,
    model_name: str,
    n_bg:     int = SHAP_BACKGROUND,
    n_test:   int = SHAP_TEST_SAMPLES,
):
    """SHAP DeepExplainer — must NOT use @torch.no_grad() as SHAP needs grads."""
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed — skipping.")
        return

    # Enable gradients for all parameters (DeepExplainer requires grad_fn)
    model.train()  # train mode ensures BN/dropout behave + grads flow
    for p in model.parameters():
        p.requires_grad_(True)

    bg_imgs, test_imgs = [], []
    with torch.no_grad():                      # safe to collect data without grads
        for imgs, _ in loader:
            bg_imgs.append(imgs)
            if sum(x.shape[0] for x in bg_imgs) >= n_bg:
                break
        bg_imgs = torch.cat(bg_imgs)[:n_bg].to(DEVICE)

        for imgs, _ in loader:
            test_imgs.append(imgs)
            if sum(x.shape[0] for x in test_imgs) >= n_test:
                break
        test_imgs = torch.cat(test_imgs)[:n_test].to(DEVICE)

    try:
        # DeepExplainer forward pass needs gradient tracking on inputs
        bg_imgs   = bg_imgs.requires_grad_(True)
        test_imgs = test_imgs.requires_grad_(True)

        explainer   = shap.DeepExplainer(model, bg_imgs)
        shap_values = explainer.shap_values(test_imgs)

        # Mean absolute SHAP across test samples → channel-wise importance
        shap_arr = np.abs(np.array(shap_values))       # (C, N, 3, H, W)
        imp = shap_arr.mean(axis=(1, 2, 3, 4))          # per class

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(imp.reshape(1, -1), annot=True, fmt=".4f",
                    xticklabels=CLASS_NAMES, yticklabels=["SHAP"],
                    cmap="Reds", ax=ax)
        ax.set_title(f"GNAS {model_name} — SHAP Feature Importance",
                     fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"gnas_{model_name}_shap.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        print(f"  SHAP plot saved.")
    except Exception as e:
        print(f"  [SHAP] failed: {e}")
    finally:
        # Always restore eval mode after SHAP
        model.eval()


# ════════════════════════════════════════════════════════════════════════════
# GradCAM feature importance heat-map
# ════════════════════════════════════════════════════════════════════════════
def run_gradcam(model: nn.Module, loader: DataLoader,
                plots_dir: str, model_name: str):
    """Produce a GradCAM overlay for the first test image."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("  [GradCAM] pytorch-grad-cam not installed — skipping.")
        return

    model.eval()
    imgs, labels = next(iter(loader))
    img_tensor = imgs[:1].to(DEVICE)

    # Find the last Conv2d in the CNN backbone
    target_layers = []
    for m in model.backbone.backbone.modules():
        if isinstance(m, nn.Conv2d):
            target_layers = [m]
    if not target_layers:
        return

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=img_tensor)[0]

        img_np = imgs[0].permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_np); ax[0].set_title("Original"); ax[0].axis("off")
        ax[1].imshow(visualization); ax[1].set_title("GradCAM"); ax[1].axis("off")
        fig.suptitle(f"GNAS {model_name} — GradCAM", fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"gnas_{model_name}_gradcam.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        print(f"  GradCAM saved.")
    except Exception as e:
        print(f"  [GradCAM] failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# Per-model training pipeline
# ════════════════════════════════════════════════════════════════════════════
def train_one_model(model_name: str, df: pd.DataFrame, dirs: List[str]):
    print(f"\n{'='*70}")
    print(f"  GNAS  ▸  {model_name}")
    print(f"{'='*70}")

    train_df, val_df, test_df = split_dataframe(df)
    class_weights = compute_class_weights(train_df)

    # Use a smaller subset for GA evaluation to save time
    ga_train_df = train_df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 300), random_state=SEED)
    ).reset_index(drop=True)
    ga_val_df = val_df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 100), random_state=SEED)
    ).reset_index(drop=True)

    ga_tr_ld = DataLoader(
        MultiDirDataset(ga_train_df, dirs, get_train_transform()),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    ga_va_ld = DataLoader(
        MultiDirDataset(ga_val_df, dirs, get_val_transform()),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    tr_ld, va_ld, te_ld = make_loaders(train_df, val_df, test_df, dirs)

    qefs_path     = os.path.join(MODELS_DIR, f"qefs_{model_name}_best.pth")
    baseline_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(qefs_path):
        qefs_path = None
        print(f"  [WARN] QEFS checkpoint not found for {model_name}; "
              f"falling back to baseline CNN backbone.")
        print(f"         Run 04_qefs.py first to generate QEFS checkpoints.")
    else:
        print(f"  [OK] QEFS checkpoint found: {os.path.basename(qefs_path)}")

    # ── 1. Genetic Architecture Search ────────────────────────────────────
    best_arch, gen_best = run_genetic_search(
        model_name, ga_tr_ld, ga_va_ld, qefs_path, baseline_path,
        pop_size=GNAS_POP_SIZE, n_generations=GNAS_GENERATIONS,
    )
    print(f"\n  ▶ Best arch found: {best_arch}")

    # ── 2. Full fine-tune ──────────────────────────────────────────────────
    t_total_start = time.time()
    model, history = full_finetune(
        model_name, best_arch, tr_ld, va_ld, qefs_path, baseline_path,
        class_weights,
    )
    total_time = time.time() - t_total_start

    # ── 3. Save checkpoint ─────────────────────────────────────────────────
    ckpt_path = os.path.join(MODELS_DIR, f"gnas_{model_name}_best.pth")
    torch.save({
        "model_state": model.state_dict(),
        "arch":        best_arch,
        "model_name":  model_name,
        "epoch":       len(history["train_acc"]),
        "uses_qefs":   getattr(model, "_uses_qefs", True),
        "feat_dim":    QEFS_FEATURE_DIM,
    }, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    # ── 4. Evaluation ──────────────────────────────────────────────────────
    train_m = full_metrics(model, tr_ld, "train")
    val_m   = full_metrics(model, va_ld, "val")
    test_m  = full_metrics(model, te_ld, "test")

    # Bootstrap CI on test set
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for imgs, labels in te_ld:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(labels.numpy())
    boot_ci = bootstrap_ci(np.array(y_true_all), np.array(y_pred_all))

    # ── 5. Cross-Validation ────────────────────────────────────────────────
    cv_res = cross_validate(
        model_name, best_arch, df, dirs, qefs_path, baseline_path,
        n_folds=CV_FOLDS, n_epochs=CV_FOLD_EPOCHS,
    )

    # ── 6. Memory profiling ────────────────────────────────────────────────
    mem = measure_memory(model, te_ld)

    # ── 7. Assemble results dict ───────────────────────────────────────────
    results = {
        "model_name":         model_name,
        "best_arch":          best_arch,
        "uses_qefs":          getattr(model, "_uses_qefs", True),
        "training_time_s":    total_time,
        "history":            history,
        "ga_convergence":     gen_best,
        "hyperparams": {
            "epochs":         NUM_EPOCHS,
            "batch_size":     BATCH_SIZE,
            "learning_rate":  LEARNING_RATE,
            "dropout":        best_arch["dropout"],
            "patience":       PATIENCE,
            "label_smoothing":LABEL_SMOOTHING,
            "gnas_pop_size":  GNAS_POP_SIZE,
            "gnas_generations":GNAS_GENERATIONS,
            "gnas_eval_epochs":GNAS_EVAL_EPOCHS,
            "cv_folds":       CV_FOLDS,
        },
        **train_m, **val_m, **test_m,
        **boot_ci, **cv_res, **mem,
    }

    # Save JSON
    results_path = os.path.join(RESULTS_DIR, f"gnas_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved → {results_path}")

    # ── 8. Plots ───────────────────────────────────────────────────────────
    plot_all(model_name, history, {**val_m, **test_m},
             best_arch, gen_best, PLOTS_DIR)
    run_shap(model, te_ld, PLOTS_DIR, model_name)
    run_gradcam(model, te_ld, PLOTS_DIR, model_name)

    # ── 9. Print Summary ───────────────────────────────────────────────────
    print(f"\n  ─ GNAS {model_name} Results ─")
    print(f"    Train Acc  : {train_m['train_accuracy']:.4f}")
    print(f"    Val   Acc  : {val_m['val_accuracy']:.4f}")
    print(f"    Test  Acc  : {test_m['test_accuracy']:.4f}")
    print(f"    Test  F1   : {test_m['test_f1_macro']:.4f} (macro)")
    print(f"    Boot CI 95%: [{boot_ci['bootstrap_acc_lo']:.4f}, "
          f"{boot_ci['bootstrap_acc_hi']:.4f}]")
    print(f"    CV Acc     : {cv_res['cv_mean_acc']:.4f} ± {cv_res['cv_acc_ci95']:.4f}")
    print(f"    Inf. time  : {test_m['test_inference_ms_per_sample']:.3f} ms/sample")
    print(f"    Train time : {total_time/60:.1f} min")
    print(f"    Best arch  : {best_arch}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════
def main():
    # global must be declared before any use of these names in this scope
    global GNAS_POP_SIZE, GNAS_GENERATIONS, GNAS_EVAL_EPOCHS

    # ── CLI overrides ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Krishak Stage 5 — Genetic NAS on QEFS Backbones"
    )
    parser.add_argument("--pop-size",    type=int, default=None,
                        help=f"GA population size (default: {GNAS_POP_SIZE})")
    parser.add_argument("--generations", type=int, default=None,
                        help=f"GA generations (default: {GNAS_GENERATIONS})")
    parser.add_argument("--eval-epochs", type=int, default=None,
                        help=f"Quick-eval epochs per candidate (default: {GNAS_EVAL_EPOCHS})")
    parser.add_argument("--models", nargs="+", default=None,
                        metavar="MODEL",
                        help="Space-separated subset of models to run "
                             f"(e.g. --models mobilenet_v3_large resnet50). "
                             f"Choices: {MODEL_NAMES}")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a completed QEFS result already exists")
    args, _ = parser.parse_known_args()

    # Apply overrides to module-level GA constants
    if args.pop_size    is not None: GNAS_POP_SIZE    = args.pop_size
    if args.generations is not None: GNAS_GENERATIONS  = args.generations
    if args.eval_epochs is not None: GNAS_EVAL_EPOCHS  = args.eval_epochs
    _model_list = list(MODEL_NAMES)
    if args.models is not None:
        invalid = [m for m in args.models if m not in MODEL_NAMES]
        if invalid:
            parser.error(f"Unknown model(s): {invalid}. Choose from {MODEL_NAMES}")
        _model_list = args.models

    print("\n" + "="*70)
    print("  Krishak — Stage 5: Genetic NAS on QEFS Backbones")
    print("="*70)
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Models : {_model_list}")
    print(f"  GA pop : {GNAS_POP_SIZE}  generations: {GNAS_GENERATIONS}  "
          f"eval_epochs: {GNAS_EVAL_EPOCHS}")
    if args.force:
        print(f"  --force: will re-run even completed models")
    print()

    # Load dataset
    csv_path = None
    for p, lbl in [(BALANCED_CSV, "balanced"), (CLEANED_CSV, "cleaned"), (TRAIN_CSV, "raw")]:
        if os.path.exists(p):
            csv_path = p
            print(f"  CSV: {lbl} → {p}")
            break
    if csv_path is None:
        raise FileNotFoundError("No CSV found. Run 01/02 first.")

    df   = pd.read_csv(csv_path)
    dirs = get_image_dirs()
    print(f"  Images dirs: {dirs}")
    print(f"  Dataset rows: {len(df):,}")

    all_results = {}
    for model_name in _model_list:
        results_path = os.path.join(RESULTS_DIR, f"gnas_{model_name}.json")

        # ── Cache check ────────────────────────────────────────────────────
        if os.path.exists(results_path) and not args.force:
            with open(results_path) as _rf:
                _cached = json.load(_rf)

            # New-format result (has uses_qefs=True) → skip
            if _cached.get("uses_qefs") is True:
                _ta = _cached.get("test_accuracy")
                _ta_str = f"{_ta:.4f}" if isinstance(_ta, float) else str(_ta)
                print(f"\n  [Skip] {model_name} — already completed with "
                      f"QEFS backbone (test_acc={_ta_str}).")
                all_results[model_name] = {
                    "test_accuracy": _cached.get("test_accuracy"),
                    "test_f1_macro": _cached.get("test_f1_macro"),
                    "cv_mean_acc":   _cached.get("cv_mean_acc"),
                    "best_arch":     _cached.get("best_arch"),
                }
                continue

            # Old-format or uses_qefs=False/None → invalidate and re-run
            if "gnas_test_acc" in _cached:
                _reason = "old-format result (pre-QEFS run)"
            elif _cached.get("uses_qefs") is False:
                _reason = "was run on baseline CNN (uses_qefs=False)"
            else:
                _reason = f"uses_qefs={_cached.get('uses_qefs')!r} — re-running"
            print(f"\n  [Invalidate] {model_name} — {_reason}. Re-running on QEFS.")
            os.remove(results_path)
        elif os.path.exists(results_path) and args.force:
            print(f"\n  [Force] {model_name} — deleting cached result and re-running.")

        try:
            res = train_one_model(model_name, df, dirs)
            all_results[model_name] = {
                "test_accuracy": res["test_accuracy"],
                "test_f1_macro": res["test_f1_macro"],
                "cv_mean_acc":   res["cv_mean_acc"],
                "best_arch":     res["best_arch"],
            }
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {model_name}: {e}")
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  GNAS SUMMARY")
    print("="*70)
    print(f"  {'Model':<25} {'Test Acc':>10} {'F1 Macro':>10} {'CV Acc':>10}")
    print(f"  {'-'*57}")
    for mn, r in all_results.items():
        if "error" in r:
            print(f"  {mn:<25} ERROR: {r['error']}")
        else:
            print(f"  {mn:<25} {r['test_accuracy']:>10.4f} "
               f"{r.get('f1_macro', 0):>10.4f} {r.get('cv_accuracy', 0)*100:>9.2f}%")
    print("="*70)
    print("  Done. Results saved to outputs/results/")


if __name__ == "__main__":
    main()
