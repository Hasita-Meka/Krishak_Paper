"""
06_continual_learning.py — Continual Learning on Krishak
=============================================================
Implements 5 Continual Learning methods from the Mammoth framework
(https://github.com/aimagelab/mammoth) adapted to PyTorch + our dataset:

  Method       | Description
  ─────────────┼──────────────────────────────────────────────────────
  GDumb        | Store balanced memory, retrain from scratch each task
  DER++        | Dark Experience Replay++ (soft + hard targets distillation)
  X-DER        | Extended DER (bilateral soft-label distillation)
  A-GEM        | Averaged Gradient Episodic Memory
  ER-ACE       | Experience Replay with Asymmetric Cross-Entropy

Setup:
  • 10 classes split into 5 tasks (2 classes per task)
  • Backbone: ResNet50 (pretrained, frozen) + learnable task head
  • Replay buffer: 500 samples (reservoir sampling)
  • Per-method: forward transfer, backward transfer, avg accuracy

Metrics per method:
  • Task accuracy matrix (ACC[i][j] = acc on task j after training task i)
  • Average Accuracy (AA), Backward Transfer (BWT), Forward Transfer (FWT)
  • Confusion matrix on all classes after all tasks
  • Memory usage, training time, inference time

Run:
  python 06_continual_learning.py
"""

import os, sys, json, time, copy, random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import (
    set_seed, get_model, get_feature_extractor,
    PlantDiseaseDataset, split_dataframe,
    compute_full_metrics, compute_class_weights,
    get_train_transform, get_val_transform,
    train_one_epoch, evaluate, EarlyStopping,
    plot_confusion_matrix, plot_class_accuracies,
    save_results, load_results, DEVICE, CLASS_NAMES, NUM_CLASSES,
    DROPOUT, LABEL_SMOOTHING, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_AMP, ETA_MIN
)

# ── CL-specific hyperparameters (from config, with safe fallbacks) ─────────
_CL_LR             = globals().get("CL_LEARNING_RATE",   5e-4)
_CL_FREEZE_BACKBONE= globals().get("CL_FREEZE_BACKBONE", True)

# ── Import QEFS building blocks (04_qefs.py) for QEFS checkpoint loading ──
import importlib.util as _ilu
_qefs_spec = _ilu.spec_from_file_location(
    "qefs_module",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_qefs.py")
)
_qefs_mod = _ilu.module_from_spec(_qefs_spec)
_qefs_spec.loader.exec_module(_qefs_mod)
QCNNHead          = _qefs_mod.QCNNHead
_get_backbone_dim = _qefs_mod.get_backbone_dim   # raw CNN feature dims
# NOTE: QuantumInspiredLayer from 04_qefs.py has theta_x/y/z + H params;
#       the GNAS version (05_genetic_nas.py) is a simpler ent_fc+BN+ReLU.
#       Keep both available under distinct names.
_QEFSQuantumLayer = _qefs_mod.QuantumInspiredLayer   # for QEFS checkpoints

# ── Import GNAS building blocks (05_genetic_nas.py) for GNAS checkpoint loading ──
_gnas_spec = _ilu.spec_from_file_location(
    "gnas_module",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_genetic_nas.py")
)
_gnas_mod = _ilu.module_from_spec(_gnas_spec)
_gnas_spec.loader.exec_module(_gnas_mod)
_GNASQEFSExtractor    = _gnas_mod.QEFSFeatureExtractor   # CNN + GNAS QuantumInspiredLayer
_GNASQuantumLayer     = _gnas_mod.QuantumInspiredLayer   # now the same class as 04_qefs (revision fix)
_gnas_get_backbone_dim = _gnas_mod.get_backbone_dim


class _QEFSFeatureExtractor(nn.Module):
    """backbone (CNN) + QEFS QuantumInspiredLayer → QEFS_FEATURE_DIM-dim features.
    Used only for loading 04_qefs.py checkpoints."""
    def __init__(self, qcnn: QCNNHead):
        super().__init__()
        self.backbone = qcnn.backbone
        self.q_layer  = qcnn.q_layer
    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)): f = f[0]
        return self.q_layer(f)

# ═══════════════════════════════════════════════════════════════════════════
# Task setup
# ═══════════════════════════════════════════════════════════════════════════
N_TASKS  = CL_N_TASKS      # 5
# Task order (R1, R5 #4): 0 = original alphabetical pairing, k>0 = seeded permutation
if CL_TASK_ORDER == 0:
    _ORDERED = list(CLASS_NAMES)
else:
    _ORDERED = list(np.random.RandomState(CL_TASK_ORDER).permutation(CLASS_NAMES))
TASKS    = [_ORDERED[i*2:(i+1)*2] for i in range(N_TASKS)]
# Variant suffix so sweeps (order / seed / buffer / freeze / legacy A-GEM) never
# overwrite the main run; the paper setting keeps the original file names.
_AGEM_LEGACY = os.environ.get("KRISHAK_AGEM_LEGACY") == "1"
_VARIANT_PARTS = []
if CL_TASK_ORDER != 0:       _VARIANT_PARTS.append(f"o{CL_TASK_ORDER}")
if SEED != 42:               _VARIANT_PARTS.append(f"s{SEED}")
if CL_BUFFER_SIZE != 2000:   _VARIANT_PARTS.append(f"b{CL_BUFFER_SIZE}")
if CL_FREEZE_MODE != "1":    _VARIANT_PARTS.append(f"f{CL_FREEZE_MODE}")
if _AGEM_LEGACY:             _VARIANT_PARTS.append("agemlegacy")
CL_VARIANT = ("_" + "_".join(_VARIANT_PARTS)) if _VARIANT_PARTS else ""
# e.g. Task 0: [bacterial_leaf_blight, bacterial_leaf_streak], ...

# CL_MODEL_NAME comes from config (default "resnet50"); override via config patch for tests
BUFFER_SIZE     = CL_BUFFER_SIZE
# Reference-parity switches (defaults = behaviour of the paper runs; used by the Split-CIFAR pilot)
CL_GRAD_CLIP = float(os.environ.get("KRISHAK_CL_CLIP", "1.0"))   # 0 = no clipping (Mammoth reference)
CL_REF_MODE  = os.environ.get("KRISHAK_CL_REF") == "1"           # DER++ exactly as in Buzzega et al.
EPOCHS_PER_TASK = CL_EPOCHS_PER_TASK


def print_tasks():
    print("\n  Task split:")
    for i, t in enumerate(TASKS):
        print(f"    Task {i}: {t}")


# ─── Multi-dir dataset ─────────────────────────────────────────────────────
class MultiDirDataset(PlantDiseaseDataset):
    def __init__(self, df, dirs, transform=None):
        super().__init__(df, dirs[0], transform)
        self.dirs = dirs
    def __getitem__(self, idx):
        from PIL import Image as PILImage
        row   = self.df.iloc[idx]
        label = row.get("label", "")
        for d in self.dirs:
            for p in [os.path.join(d, label, row["image_id"]),
                      os.path.join(d, row["image_id"])]:
                if os.path.exists(p):
                    img = PILImage.open(p).convert("RGB")
                    if self.transform: img = self.transform(img)
                    return img, self.class_to_idx[label]
        raise FileNotFoundError(row["image_id"])


def get_image_dirs():
    aug = os.path.join(DATASET_DIR, "aug_images")
    dirs = [TRAIN_IMAGES]
    if os.path.isdir(aug): dirs.append(aug)
    return dirs


# ═══════════════════════════════════════════════════════════════════════════
# Continual Learning backbone
# ═══════════════════════════════════════════════════════════════════════════
class CLModel(nn.Module):
    """
    Pretrained backbone (frozen) + learnable all-class head.
    Checkpoint priority (best → fallback):
      1. gnas_{name}_best.pth   — GNAS-optimised (may wrap a QEFS backbone)
      2. qefs_{name}_best.pth   — QEFS quantum-enhanced backbone
      3. {name}_best.pth        — baseline CNN checkpoint
    We use a shared head over all NUM_CLASSES to enable backward-transfer measurement.
    """
    def __init__(self, backbone_name: str = CL_MODEL_NAME):
        super().__init__()

        gnas_path = os.path.join(MODELS_DIR, f"gnas_{backbone_name}_best.pth")
        qefs_path = os.path.join(MODELS_DIR, f"qefs_{backbone_name}_best.pth")
        base_path = os.path.join(MODELS_DIR, f"{backbone_name}_best.pth")

        if os.path.exists(gnas_path):
            self._load_from_gnas(backbone_name, gnas_path)
        elif os.path.exists(qefs_path):
            self._load_from_qefs(backbone_name, qefs_path)
        else:
            self._load_from_baseline(backbone_name, base_path)

        self.fc      = nn.Linear(self.feat_dim, NUM_CLASSES)
        self.dropout = nn.Dropout(DROPOUT)
        # Ensure all submodules (backbone + fc + dropout) are on the same device.
        # The backbone loaders each call .to(DEVICE) internally; fc/dropout are
        # created on CPU by default, so we synchronise everything here.
        self.to(DEVICE)

        # ── Backbone freezing (key for zero catastrophic forgetting) ─────────
        if _CL_FREEZE_BACKBONE:
            params = list(self.backbone.parameters())
            for param in params:
                param.requires_grad = False
            if CL_FREEZE_MODE == "partial":
                n_open = max(1, int(round(0.2 * len(params))))
                for param in params[-n_open:]:
                    param.requires_grad = True
                print(f"  [CL] Backbone PARTIAL — last {n_open}/{len(params)} parameter tensors trainable.")
            else:
                print(f"  [CL] Backbone FROZEN ({len(params)} param tensors) — "
                      "only classification head will be trained.")
        else:
            print("  [CL] Backbone UNFROZEN — all parameters will be trained.")
        self.n_trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        self.n_total     = int(sum(p.numel() for p in self.parameters()))

    # ── Backbone loaders ─────────────────────────────────────────────────

    def _load_from_gnas(self, backbone_name: str, gnas_path: str):
        """Reconstruct the GNAS model, extract its backbone, discard the head."""
        ckpt      = torch.load(gnas_path, map_location=DEVICE, weights_only=False)
        uses_qefs = ckpt.get("uses_qefs", False)
        feat_dim  = ckpt.get("feat_dim",  2048)

        if uses_qefs:
            # GNAS backbone = CNN + full 04_qefs QuantumInspiredLayer (+ HFSEA mask)
            raw_dim  = _gnas_get_backbone_dim(backbone_name)
            base     = get_model(backbone_name).to(DEVICE)
            base_ext = get_feature_extractor(base, backbone_name).to(DEVICE)
            q_layer  = _GNASQuantumLayer(raw_dim, int(ckpt.get("q_dim", QEFS_FEATURE_DIM))).to(DEVICE)
            extractor = _GNASQEFSExtractor(base_ext, q_layer, ckpt.get("sel_idx")).to(DEVICE)
            feat_dim  = extractor.out_dim
        else:
            # GNAS backbone = plain CNN feature extractor
            base      = get_model(backbone_name).to(DEVICE)
            extractor = get_feature_extractor(base, backbone_name).to(DEVICE)

        # Pluck only the backbone.* keys from the full GNAS state dict
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in ckpt["model_state"].items()
            if k.startswith("backbone.")
        }
        extractor.load_state_dict(backbone_state)
        self.backbone = extractor
        self.feat_dim = feat_dim
        print(f"  [CL] Loaded GNAS backbone (uses_qefs={uses_qefs}, "
              f"feat_dim={feat_dim}) from {os.path.basename(gnas_path)}")

    def _load_from_qefs(self, backbone_name: str, qefs_path: str):
        """Build a QEFS backbone (CNN + QuantumInspiredLayer → 512-dim)."""
        raw_dim  = _get_backbone_dim(backbone_name)
        base     = get_model(backbone_name).to(DEVICE)
        base_ext = get_feature_extractor(base, backbone_name)
        qcnn     = QCNNHead(base_ext, raw_dim, QEFS_FEATURE_DIM).to(DEVICE)
        ckpt     = torch.load(qefs_path, map_location=DEVICE, weights_only=False)
        state    = ckpt.get("model_state", ckpt)
        qcnn.load_state_dict(state)
        extractor = _QEFSFeatureExtractor(qcnn).to(DEVICE)
        self.backbone = extractor
        self.feat_dim = QEFS_FEATURE_DIM
        print(f"  [CL] Loaded QEFS backbone (feat_dim=512) from {os.path.basename(qefs_path)}")

    def _load_from_baseline(self, backbone_name: str, base_path: str):
        """Standard CNN feature extractor (baseline fallback)."""
        base = get_model(backbone_name).to(DEVICE)
        if os.path.exists(base_path):
            ckpt = torch.load(base_path, map_location=DEVICE, weights_only=False)
            base.load_state_dict(ckpt["model_state"])
            print(f"  [CL] Loaded baseline backbone from {os.path.basename(base_path)}")
        else:
            print(f"  [CL] WARNING: no checkpoint found for {backbone_name}; using random weights")
        self.backbone = get_feature_extractor(base, backbone_name)
        self.feat_dim = _get_backbone_dim(backbone_name)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)): f = f[0]
        return self.fc(self.dropout(f))

    def features(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)): f = f[0]
        return f


# ═══════════════════════════════════════════════════════════════════════════
# Replay Buffer (Reservoir Sampling)
# ═══════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data:   list = []   # list of (img_tensor, label, logits)
        self.n_seen: int  = 0

    def add(self, imgs: torch.Tensor, labels: torch.Tensor,
            logits: torch.Tensor = None):
        """Reservoir sampling."""
        for i in range(imgs.size(0)):
            item = (imgs[i].cpu(), labels[i].item(),
                    logits[i].cpu() if logits is not None else None)
            if len(self.data) < self.capacity:
                self.data.append(item)
            else:
                j = random.randint(0, self.n_seen)
                if j < self.capacity:
                    self.data[j] = item
            self.n_seen += 1

    def sample(self, n: int) -> tuple:
        n = min(n, len(self.data))
        if n == 0:
            return None, None, None
        batch = random.sample(self.data, n)
        imgs   = torch.stack([b[0] for b in batch]).to(DEVICE)
        labels = torch.tensor([b[1] for b in batch], device=DEVICE)
        logits = torch.stack([b[2] for b in batch]).to(DEVICE) \
                 if batch[0][2] is not None else None
        return imgs, labels, logits

    def __len__(self):
        return len(self.data)


def _step(model, opt, scaler):
    """Optimizer step with optional gradient clipping on the TRUE (unscaled) gradients."""
    if CL_GRAD_CLIP > 0:
        scaler.unscale_(opt)   # revision fix: clip the TRUE gradients, not the AMP-scaled ones
        torch.nn.utils.clip_grad_norm_(model.parameters(), CL_GRAD_CLIP)
    scaler.step(opt); scaler.update()


# ═══════════════════════════════════════════════════════════════════════════
# Task data helpers
# ═══════════════════════════════════════════════════════════════════════════
def get_task_loaders(df, dirs, task_idx, split="train"):
    task_classes = TASKS[task_idx]
    task_df = df[df["label"].isin(task_classes)].reset_index(drop=True)
    transform = get_train_transform() if split == "train" else get_val_transform()
    ds = MultiDirDataset(task_df, dirs, transform)
    shuffle = (split == "train")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def get_all_loaders(test_df, dirs):
    """Return per-task test loaders for accuracy matrix."""
    loaders = []
    for t in range(N_TASKS):
        loaders.append(get_task_loaders(test_df, dirs, t, split="test"))
    return loaders


# ═══════════════════════════════════════════════════════════════════════════
# Continual Learning metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_cl_metrics(acc_matrix: np.ndarray) -> dict:
    """
    acc_matrix[i][j] = accuracy on task j after training task i.
    Shape: (N_TASKS, N_TASKS).
    """
    N = N_TASKS
    AA  = float(np.mean([acc_matrix[N-1, j] for j in range(N)]))
    BWT = float(np.mean([acc_matrix[N-1, j] - acc_matrix[j, j] for j in range(N-1)]))
    # FWT: acc on task j before training it, compared to random (0)
    FWT = float(np.mean([acc_matrix[j-1, j] for j in range(1, N)])) if N > 1 else 0.0
    return {"AA": AA, "BWT": BWT, "FWT": FWT}


def task_accuracy(model, loader) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())
    return accuracy_score(all_labels, all_preds)


# ═══════════════════════════════════════════════════════════════════════════
# 1. GDumb
# ═══════════════════════════════════════════════════════════════════════════
class GDumb:
    """
    Store a balanced memory of M samples.
    At each task, retrain the model from scratch on the memory alone.
    """
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = {}  # class → list of (img_tensor, label)
        self.max_per_class = buffer_size // NUM_CLASSES

    def update_memory(self, loader):
        for imgs, labels in loader:
            for img, lbl in zip(imgs, labels):
                c = lbl.item()
                if c not in self.buffer: self.buffer[c] = []
                if len(self.buffer[c]) < self.max_per_class:
                    self.buffer[c].append((img.clone(), c))

    def get_memory_loader(self):
        all_items = []
        for c, items in self.buffer.items():
            all_items.extend(items)
        if not all_items: return None
        random.shuffle(all_items)
        imgs   = torch.stack([x[0] for x in all_items])
        labels = torch.tensor([x[1] for x in all_items])
        ds = torch.utils.data.TensorDataset(imgs, labels)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    def train_from_memory(self, model, n_epochs=EPOCHS_PER_TASK):
        loader = self.get_memory_loader()
        if loader is None: return
        model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        # Only optimise trainable parameters (backbone may be frozen)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=_CL_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-6)
        scaler = GradScaler(device="cuda", enabled=USE_AMP)
        for ep in range(n_epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=USE_AMP):
                    loss = criterion(model(imgs), labels)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            scheduler.step()


# ═══════════════════════════════════════════════════════════════════════════
# 2. DER++ (Dark Experience Replay++)
# ═══════════════════════════════════════════════════════════════════════════
class DERpp:
    """
    Combines:
      • CE loss on current task
      • MSE distillation on buffer logits (dark knowledge)
      • CE on buffer labels
    """
    def __init__(self, alpha=0.8, beta=0.8, buffer_size=BUFFER_SIZE):
        self.alpha  = alpha   # weight for logit distillation
        self.beta   = beta    # weight for buffer CE
        self.buffer = ReplayBuffer(buffer_size)

    def observe(self, model, imgs, labels, opt, scaler):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(imgs)
            loss_ce = F.cross_entropy(logits, labels)

            loss = loss_ce
            if len(self.buffer) > 0 and CL_REF_MODE:
                # Reference DER++ (Buzzega et al.): two INDEPENDENT buffer batches
                d_imgs, _, d_logits = self.buffer.sample(BATCH_SIZE)
                c_imgs, c_labels, _ = self.buffer.sample(BATCH_SIZE)
                loss_dark = F.mse_loss(model(d_imgs).float(), d_logits.float())
                loss_buf  = F.cross_entropy(model(c_imgs).float(), c_labels)
                loss = loss_ce + self.alpha * loss_dark + self.beta * loss_buf
            elif len(self.buffer) > 0:
                b_imgs, b_labels, b_logits = self.buffer.sample(BATCH_SIZE)
                b_out = model(b_imgs)
                # Logit distillation (dark knowledge)
                loss_dark = F.mse_loss(b_out, b_logits) if b_logits is not None else 0.0
                # CE on buffer labels
                loss_buf  = F.cross_entropy(b_out, b_labels)
                loss = loss_ce + self.alpha * loss_dark + self.beta * loss_buf

        scaler.scale(loss).backward()
        _step(model, opt, scaler)

        # Store current batch in buffer WITH logits (no grad)
        if CL_REF_MODE:
            lg = logits.detach().float()        # pre-update logits from the training forward (reference)
        else:
            with torch.no_grad():
                lg = model(imgs).detach()
        self.buffer.add(imgs.cpu(), labels.cpu(), lg.cpu())
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# 3. X-DER (Extended DER)
# ═══════════════════════════════════════════════════════════════════════════
class XDER:
    """
    X-DER adds bilateral distillation:
      • Forward: current model predicts buffer logits
      • Backward: buffer model predicts current logits (stored at add-time)
    Plus a semantic regularisation term.
    """
    def __init__(self, alpha=0.7, beta=0.7, gamma=0.2, buffer_size=BUFFER_SIZE):
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma   # semantic regularisation weight
        self.buffer = ReplayBuffer(buffer_size)

    def observe(self, model, imgs, labels, opt, scaler, task_id: int = 0):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(imgs)
            loss_ce = F.cross_entropy(logits, labels)
            loss = loss_ce

            if len(self.buffer) > 0:
                b_imgs, b_labels, b_logits = self.buffer.sample(BATCH_SIZE)
                b_out = model(b_imgs)

                # Forward distillation (DER++ style)
                loss_fwd = F.mse_loss(b_out, b_logits) if b_logits is not None else 0.0

                # Backward: penalise current logits changing on buffer data
                # (soft targets from stored logits)
                loss_bwd = F.kl_div(
                    F.log_softmax(b_out, dim=-1),
                    F.softmax(b_logits, dim=-1),
                    reduction="batchmean"
                ) if b_logits is not None else 0.0

                # Semantic: current features should be consistent
                loss_sem = F.cross_entropy(b_out, b_labels)

                loss = loss_ce + self.alpha * loss_fwd + self.beta * loss_bwd + \
                       self.gamma * loss_sem

        scaler.scale(loss).backward()
        _step(model, opt, scaler)

        with torch.no_grad():
            lg = model(imgs).detach()
        self.buffer.add(imgs.cpu(), labels.cpu(), lg.cpu())
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# 4. A-GEM (Averaged Gradient Episodic Memory)
# ═══════════════════════════════════════════════════════════════════════════
class AGEM:
    """
    Projects gradients so that the loss on a reference buffer subset
    does not increase. Uses averaged episodic memory constraint.
    """
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = ReplayBuffer(buffer_size)
        # Revision fix (R4): original A-GEM (Chaudhry et al., 2019) constrains
        # updates only against memory from PREVIOUS tasks. The submitted version
        # added current-task samples to the reference buffer immediately. Now the
        # current task fills a staging reservoir that is committed at task end.
        # KRISHAK_AGEM_LEGACY=1 restores the submitted behaviour.
        self.legacy = _AGEM_LEGACY
        self._per_task = max(1, buffer_size // N_TASKS)
        self._cur = ReplayBuffer(self._per_task)

    def end_task(self):
        if self.legacy:
            return
        for item in self._cur.data:
            if len(self.buffer.data) < self.buffer.capacity:
                self.buffer.data.append(item)
        self.buffer.n_seen = len(self.buffer.data)
        self._cur = ReplayBuffer(self._per_task)

    def _project_grad(self, model, ref_grad: torch.Tensor) -> None:
        """Project current gradients so they don't increase buffer loss."""
        params = [p for p in model.parameters() if p.grad is not None]
        cur_grads = torch.cat([p.grad.view(-1) for p in params])

        dot = torch.dot(cur_grads, ref_grad)
        if dot < 0:
            # Project onto the half-space
            proj = cur_grads - dot / (ref_grad.dot(ref_grad) + 1e-8) * ref_grad
            offset = 0
            for p in params:
                n = p.grad.numel()
                p.grad.copy_(proj[offset: offset + n].view_as(p.grad))
                offset += n

    def observe(self, model, imgs, labels, opt, scaler):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # ── Reference gradient from buffer ─────────────────────────────────
        ref_grad = None
        if len(self.buffer) > 0:
            b_imgs, b_labels, _ = self.buffer.sample(BATCH_SIZE)
            opt.zero_grad(set_to_none=True)
            b_loss = F.cross_entropy(model(b_imgs), b_labels)
            b_loss.backward()
            ref_grad = torch.cat([p.grad.view(-1).clone()
                                  for p in model.parameters()
                                  if p.grad is not None])
            opt.zero_grad(set_to_none=True)

        # ── Current task gradient ──────────────────────────────────────────
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=USE_AMP):
            loss = F.cross_entropy(model(imgs), labels)
        scaler.scale(loss).backward()

        if ref_grad is not None:
            self._project_grad(model, ref_grad)

        scaler.step(opt); scaler.update()
        (self.buffer if self.legacy else self._cur).add(imgs.cpu(), labels.cpu())
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# 5. ER-ACE (Experience Replay + Asymmetric Cross-Entropy)
# ═══════════════════════════════════════════════════════════════════════════
class ERACE:
    """
    ER-ACE masks out old-class logits when computing the CE loss on
    current task data, preventing interference. Buffer replayed with
    standard CE.
    """
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = ReplayBuffer(buffer_size)
        self.seen_classes = set()

    def _masked_ce(self, logits: torch.Tensor, labels: torch.Tensor,
                   present_classes: set) -> torch.Tensor:
        """CE only over classes seen in the current task (ACE component)."""
        mask = torch.full((NUM_CLASSES,), float("-inf"), device=DEVICE)
        for c in present_classes:
            mask[c] = 0.0
        return F.cross_entropy(logits + mask, labels)

    def observe(self, model, imgs, labels, opt, scaler, present_classes: set):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        self.seen_classes.update(present_classes)
        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(imgs)
            # Asymmetric CE (mask out unseen old classes)
            loss = self._masked_ce(logits, labels, present_classes)

            if len(self.buffer) > 0:
                b_imgs, b_labels, _ = self.buffer.sample(BATCH_SIZE)
                b_out  = model(b_imgs)
                loss  += F.cross_entropy(b_out, b_labels)

        scaler.scale(loss).backward()
        _step(model, opt, scaler)

        self.buffer.add(imgs.cpu(), labels.cpu())
        return loss.item()


def buffer_report(method_obj, n_train_total: int) -> dict:
    """Replay-memory accounting (R1, R5 #5)."""
    if isinstance(method_obj, GDumb):
        items = [(img, c) for c, lst in method_obj.buffer.items() for img, _ in lst]
        cap = method_obj.max_per_class * NUM_CLASSES
    elif hasattr(method_obj, "buffer") and hasattr(method_obj.buffer, "data"):
        items = [(it[0], it[1]) for it in method_obj.buffer.data]
        cap = method_obj.buffer.capacity
    else:
        return {}
    per_class = {CLASS_NAMES[c]: 0 for c in range(NUM_CLASSES)}
    nbytes = 0
    for img, c in items:
        per_class[CLASS_NAMES[int(c)]] += 1
        nbytes += img.element_size() * img.numel()
    if hasattr(method_obj, "buffer") and hasattr(method_obj.buffer, "data"):
        nbytes += sum(it[2].element_size() * it[2].numel()
                      for it in method_obj.buffer.data if len(it) > 2 and it[2] is not None)
    counts = [v for v in per_class.values() if v > 0]
    return {
        "capacity": int(cap),
        "stored": int(len(items)),
        "per_class": per_class,
        "class_balanced": bool(counts) and (max(counts) - min(counts) <= 1),
        "fraction_of_train": float(len(items) / max(1, n_train_total)),
        "memory_MB": float(nbytes / 1e6),
        "stores": "raw normalised image tensors (3x224x224 float32)"
                  + (" + logits" if isinstance(method_obj, (DERpp, XDER)) else ""),
        "policy": ("class-balanced per-class quota" if isinstance(method_obj, GDumb)
                   else "per-task quota committed at task end" if isinstance(method_obj, AGEM) and not method_obj.legacy
                   else "reservoir sampling over the stream"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Generic training loop for all CL methods
# ═══════════════════════════════════════════════════════════════════════════
def cl_train_loop(method_name: str, method_obj, model: nn.Module,
                  train_df, test_df, dirs, model_name: str = "") -> dict:
    """Run the full CL sequence for a given method."""
    print(f"\n{'─'*60}")
    print(f"  CL Method: {method_name}")
    print(f"{'─'*60}")

    set_seed()
    # Only optimise trainable parameters (backbone may be frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(trainable_params, lr=_CL_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    task_loaders_test = get_all_loaders(test_df, dirs)

    acc_matrix  = np.zeros((N_TASKS, N_TASKS))
    loss_curves = []  # per task
    # FWT reference b_j: accuracy on task j of the model before any CL training
    # (backbone loaded, head freshly initialised). With a shared 10-way head and
    # unseen classes, acc[j-1, j] and b_j are both ~0, so FWT is ~0 by construction
    # in this class-incremental protocol (R1) — reported, but not interpreted.
    model.eval()
    fwt_ref = [float(task_accuracy(model, task_loaders_test[t])) for t in range(N_TASKS)]
    t_total     = time.time()

    for task_id in range(N_TASKS):
        task_classes = TASKS[task_id]
        task_class_ids = {CLASS_TO_IDX[c] for c in task_classes}
        print(f"\n  Task {task_id}: {task_classes}")

        train_ld = get_task_loaders(train_df, dirs, task_id, split="train")
        task_losses = []

        # Cosine LR scheduler — reset for each task.
        # Must reset optimizer LR first; otherwise CosineAnnealingLR inherits
        # eta_min (1e-6) from the previous task and never rises above it.
        for pg in opt.param_groups:
            pg["lr"] = _CL_LR
        task_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS_PER_TASK, eta_min=1e-6)

        # Special case: GDumb — update memory, then retrain from scratch
        if isinstance(method_obj, GDumb):
            method_obj.update_memory(train_ld)
            model_clone = copy.deepcopy(model)
            method_obj.train_from_memory(model_clone, n_epochs=EPOCHS_PER_TASK)
            model.load_state_dict(model_clone.state_dict())
        else:
            # Standard iterative observe loop
            for epoch in range(EPOCHS_PER_TASK):
                ep_loss = 0.0; n_batches = 0
                for imgs, labels in train_ld:
                    if isinstance(method_obj, AGEM):
                        l = method_obj.observe(model, imgs, labels, opt, scaler)
                    elif isinstance(method_obj, ERACE):
                        l = method_obj.observe(model, imgs, labels, opt, scaler,
                                               task_class_ids)
                    elif isinstance(method_obj, (DERpp, XDER)):
                        l = method_obj.observe(model, imgs, labels, opt, scaler)
                    else:
                        l = 0.0
                    ep_loss += l; n_batches += 1
                avg_loss = ep_loss / max(n_batches, 1)
                task_losses.append(avg_loss)
                task_scheduler.step()
                if (epoch + 1) % 10 == 0:
                    cur_lr = task_scheduler.get_last_lr()[0]
                    print(f"    Epoch {epoch+1}/{EPOCHS_PER_TASK} | loss={avg_loss:.4f} | lr={cur_lr:.2e}")

        loss_curves.append(task_losses)
        if hasattr(method_obj, "end_task"):
            method_obj.end_task()

        # Evaluate on ALL tasks seen so far
        model.eval()
        for prev_task in range(N_TASKS):
            acc = task_accuracy(model, task_loaders_test[prev_task])
            acc_matrix[task_id, prev_task] = acc
        print(f"  Acc after Task {task_id}: {acc_matrix[task_id, :task_id+1].tolist()}")

    total_time = time.time() - t_total

    cl_metrics = compute_cl_metrics(acc_matrix)
    print(f"\n  AA={cl_metrics['AA']:.4f}  BWT={cl_metrics['BWT']:.4f}  FWT={cl_metrics['FWT']:.4f}")

    # Final evaluation: all classes
    print(f"\n  Final evaluation on all classes …")
    model.eval()
    all_preds, all_labels = [], []
    full_test_ds = MultiDirDataset(test_df, dirs, get_val_transform())
    full_test_ld = DataLoader(full_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    with torch.no_grad():
        for imgs, lbls in full_test_ld:
            out = model(imgs.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc_all = accuracy_score(labels, preds)
    f1_all  = f1_score(labels, preds, average="macro", zero_division=0)

    # Inference time
    t0 = time.time()
    with torch.no_grad():
        for imgs, _ in full_test_ld:
            _ = model(imgs.to(DEVICE))
    inf_time = (time.time() - t0) / len(full_test_ld.dataset)

    results = {
        "method":             method_name,
        "model_name":         model_name or CL_MODEL_NAME,
        "acc_matrix":         acc_matrix.tolist(),
        "AA":                 cl_metrics["AA"],
        "BWT":                cl_metrics["BWT"],
        "FWT":                cl_metrics["FWT"],
        "final_test_acc":     float(acc_all),
        "final_f1_macro":     float(f1_all),
        "total_training_time_s": total_time,
        "inference_time_per_img_s": float(inf_time),
        "buffer_size":        BUFFER_SIZE,
        "fwt_reference_acc":  fwt_ref,
        "FWT_vs_init":        float(np.mean([acc_matrix[j-1, j] - fwt_ref[j] for j in range(1, N_TASKS)])),
        "task_order":         TASKS,
        "task_order_seed":    CL_TASK_ORDER,
        "seed":               SEED,
        "freeze_mode":        CL_FREEZE_MODE,
        "grad_clip":          CL_GRAD_CLIP,
        "der_ref_mode":       CL_REF_MODE,
        "agem_legacy":        _AGEM_LEGACY,
        "n_trainable_params": getattr(model, "n_trainable", None),
        "n_total_params":     getattr(model, "n_total", None),
        "memory_report":      buffer_report(method_obj, len(train_df)),
        "n_tasks":            N_TASKS,
        "epochs_per_task":    EPOCHS_PER_TASK,
        "preds":              preds.tolist(),
        "labels":             labels.tolist(),
    }
    results.update(compute_full_metrics(preds, labels, method_name))
    _safe_method = method_name.lower().replace("-", "_").replace("+", "p")
    _save_key = (f"cl_{model_name}_{_safe_method}" if model_name else f"cl_{_safe_method}") + CL_VARIANT
    save_results(results, _save_key)

    # Save model checkpoint for this method + backbone (main setting only)
    _ckpt_name = f"cl_{model_name}_{_safe_method}_best.pth" if model_name else f"cl_{_safe_method}_best.pth"
    _ckpt_path = os.path.join(MODELS_DIR, _ckpt_name)
    if CL_VARIANT:
        return results
    torch.save({
        "model_state":    model.state_dict(),
        "method":         method_name,
        "model_name":     model_name,
        "AA":             results.get("AA", 0.0),
        "final_test_acc": results.get("final_test_acc", 0.0),
        "final_f1_macro": results.get("final_f1_macro", 0.0),
        "feat_dim":       model.feat_dim,
    }, _ckpt_path)
    print(f"  [CL] Model saved → {_ckpt_name}")

    # Plots
    plot_confusion_matrix(np.array(results["confusion_matrix"]),
                          method_name, tag="cl")
    plot_task_accuracy_matrix(acc_matrix, method_name)
    plot_task_loss_curves(loss_curves, method_name)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_task_accuracy_matrix(acc_matrix: np.ndarray, method_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(acc_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=[f"Task {i}" for i in range(N_TASKS)],
                yticklabels=[f"After Task {i}" for i in range(N_TASKS)],
                ax=ax, vmin=0, vmax=1)
    ax.set(title=f"Task Accuracy Matrix — {method_name}",
           xlabel="Evaluated on", ylabel="Trained up to")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_acc_matrix_{method_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()


def plot_task_loss_curves(loss_curves: list, method_name: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette("tab10", N_TASKS)
    offset = 0
    for t, lc in enumerate(loss_curves):
        x = range(offset, offset + len(lc))
        ax.plot(x, lc, color=palette[t], label=f"Task {t}", linewidth=2)
        ax.axvline(offset, color=palette[t], linestyle=":", alpha=0.5)
        offset += len(lc)
    ax.set(title=f"Training Loss Curves — {method_name}",
           xlabel="Epoch (cumulative)", ylabel="Loss")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_loss_curves_{method_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()


def plot_cl_comparison(all_results: list, model_name: str = ""):
    if not all_results: return
    tag     = f"_{model_name}" if model_name else ""
    methods = [r["method"] for r in all_results]
    AA      = [r["AA"]  for r in all_results]
    BWT     = [r["BWT"] for r in all_results]
    FWT     = [r["FWT"] for r in all_results]
    final   = [r["final_test_acc"] for r in all_results]
    f1      = [r["final_f1_macro"] for r in all_results]
    times   = [r["total_training_time_s"] / 60 for r in all_results]

    palette = sns.color_palette("Set2", len(methods))
    title_suffix = f" — {model_name}" if model_name else ""

    # ── 1. AA / BWT / FWT ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, vals, title, color in zip(
            axes, [AA, BWT, FWT],
            ["Average Accuracy (AA)", "Backward Transfer (BWT)", "Forward Transfer (FWT)"],
            ["#4e79a7", "#f28e2b", "#59a14f"]):
        bars = ax.bar(methods, vals, color=color, edgecolor="black")
        ax.set(title=title, ylabel="Score")
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.002 if v >= 0 else -0.005),
                    f"{v:.3f}", ha="center", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"Continual Learning Metrics{title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_aa_bwt_fwt{tag}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    # ── 2. Final accuracy & F1 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, vals, title in zip(axes, [final, f1],
                               ["Final Test Accuracy", "Final F1-Macro"]):
        bars = ax.bar(methods, vals, color=palette, edgecolor="black")
        ax.set(title=title, ylabel="Score", ylim=(0, 1.1))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"CL Methods — Final Performance{title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_final_performance{tag}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    # ── 3. Radar chart ─────────────────────────────────────────────────────
    cats = ["AA", "BWT+1", "FWT+1", "Test Acc", "F1 Macro"]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (method, r) in enumerate(zip(methods, all_results)):
        vals_raw = [r["AA"], r["BWT"] + 1, r["FWT"] + 1,
                    r["final_test_acc"], r["final_f1_macro"]]
        vals_raw += [vals_raw[0]]
        ax.plot(angles, vals_raw, "o-", linewidth=2, label=method,
                color=palette[i])
        ax.fill(angles, vals_raw, alpha=0.1, color=palette[i])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=13)
    ax.set_ylim(0, 1.2)
    ax.set_title(f"CL Methods Radar Chart{title_suffix}", fontsize=14, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_radar{tag}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    # ── 4. Training time ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(methods, times, color=palette, edgecolor="black")
    ax.set(title=f"Total Training Time (minutes){title_suffix}", ylabel="Minutes")
    ax.set_xticklabels(methods, rotation=30, ha="right")
    for i, v in enumerate(times): ax.text(i, v + 0.1, f"{v:.1f}m", ha="center")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_training_time{tag}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    # ── 5. Task accuracy progression heatmap grid ─────────────────────────
    n_m = len(all_results)
    cols = min(3, n_m); rows = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    if n_m == 1: axes = [[axes]]
    elif rows == 1: axes = [axes]
    else: axes = axes.tolist()

    flat_axes = [ax for row in axes for ax in (row if isinstance(row, list) else [row])]
    for ax, r in zip(flat_axes, all_results):
        mat = np.array(r["acc_matrix"])
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f"T{i}" for i in range(N_TASKS)],
                    yticklabels=[f"T{i}" for i in range(N_TASKS)],
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(r["method"], fontsize=12)
    for ax in flat_axes[n_m:]: ax.axis("off")
    plt.suptitle(f"Task Accuracy Matrices — All CL Methods{title_suffix}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_all_acc_matrices{tag}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    print(f"\n[CL Plots] saved to {PLOTS_DIR}")


def plot_cl_cross_model_summary(all_model_results: dict):
    """Bar chart comparing best CL method AA across all models."""
    if not all_model_results: return
    model_names = list(all_model_results.keys())
    # Best AA per model (across methods)
    best_AA  = [max(r["AA"]             for r in results) for results in all_model_results.values()]
    best_acc = [max(r["final_test_acc"] for r in results) for results in all_model_results.values()]
    best_f1  = [max(r["final_f1_macro"] for r in results) for results in all_model_results.values()]

    palette = sns.color_palette("tab10", len(model_names))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, vals, title in zip(axes,
                               [best_AA, best_acc, best_f1],
                               ["Best AA across CL Methods",
                                "Best Final Test Acc", "Best Final F1-Macro"]):
        bars = ax.bar(model_names, vals, color=palette, edgecolor="black")
        ax.set(title=title, ylabel="Score", ylim=(0, 1.1))
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("CL Cross-Model Summary (Best per Model)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cl_cross_model_summary.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"[CL] Cross-model summary plot saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_seed()
    print("\n" + "=" * 60)
    print("  Krishak — Continual Learning")
    print("=" * 60)
    print_tasks()

    csv = BALANCED_CSV if os.path.exists(BALANCED_CSV) else CLEANED_CSV
    if not os.path.exists(csv): csv = TRAIN_CSV
    df   = pd.read_csv(csv)
    dirs = get_image_dirs()

    def exists(row):
        label = row.get("label", "")
        return any(
            os.path.exists(os.path.join(d, label, row["image_id"])) or
            os.path.exists(os.path.join(d, row["image_id"]))
            for d in dirs
        )
    df = df[df.apply(exists, axis=1)].reset_index(drop=True)

    train_df, val_df, test_df = split_dataframe(df)
    all_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"  Train: {len(all_df):,}  Test: {len(test_df):,}")

    methods = {
        "GDumb":  (GDumb,  {}),
        "DER++":  (DERpp,  {"alpha": 0.8, "beta": 0.8}),
        "X-DER":  (XDER,   {"alpha": 0.7, "beta": 0.7, "gamma": 0.2}),
        "A-GEM":  (AGEM,   {}),
        "ER-ACE": (ERACE,  {}),
    }
    # KRISHAK_CL_METHODS=A-GEM,ER-ACE runs a subset
    if os.environ.get("KRISHAK_CL_METHODS"):
        _keep = [m.strip() for m in __import__("re").split(r"[,:]", os.environ["KRISHAK_CL_METHODS"]) if m.strip()]   # "," or ":" (":" is safe inside sbatch --export)
        methods = {k: v for k, v in methods.items() if k in _keep}
    print(f"  Variant: '{CL_VARIANT or 'main'}'  buffer={BUFFER_SIZE}  freeze={CL_FREEZE_MODE}  "
          f"order={CL_TASK_ORDER}  seed={SEED}  methods={list(methods)}")

    # ── Model order: best → worst by GNAS test accuracy ───────────────────
    import json as _json
    def _gnas_acc(m):
        p = os.path.join(RESULTS_DIR, f"gnas_{m}.json")
        if not os.path.exists(p): return 0.0
        with open(p) as f: d = _json.load(f)
        return d.get("test_accuracy", d.get("metrics", {}).get("test_accuracy", 0.0))

    _all_models = [
        "mobilenet_v3_large", "resnet50", "densenet161",
        "googlenet", "shufflenet_v2"
    ]
    # Only include models whose GNAS checkpoint exists
    _ranked_models = sorted(
        [m for m in _all_models if m in MODEL_NAMES
         and os.path.exists(os.path.join(MODELS_DIR, f"gnas_{m}_best.pth"))],
        key=_gnas_acc, reverse=True
    )
    print(f"\n  Model order (best → worst by GNAS accuracy):")
    for _m in _ranked_models:
        print(f"    {_m:<28} {_gnas_acc(_m):.4f}")

    # ── Outer loop: one full CL run per model ─────────────────────────────
    all_model_results = {}   # model_name → list of method result dicts

    for model_name in _ranked_models:
        print(f"\n{'='*75}")
        print(f"  Backbone: {model_name}  (GNAS acc={_gnas_acc(model_name):.4f})")
        print(f"{'='*75}")

        model_results = []
        for method_name, (MethodClass, kwargs) in methods.items():
            safe_name = method_name.lower().replace("-", "_").replace("+", "p")
            res_key   = f"cl_{model_name}_{safe_name}" + CL_VARIANT
            res_path  = os.path.join(RESULTS_DIR, f"{res_key}.json")

            if os.path.exists(res_path):
                cached = load_results(res_key)
                if cached and cached.get("model_name", model_name) == model_name:
                    print(f"\n  [Skip] {method_name} already done with {model_name}.")
                    model_results.append(cached)
                    continue
                else:
                    print(f"\n  [Invalidate] Stale result for {method_name}/{model_name} — re-running.")
                    os.remove(res_path)

            # Fresh model + method for each combination
            model = CLModel(model_name).to(DEVICE)
            method_obj = MethodClass(buffer_size=BUFFER_SIZE, **kwargs)
            r = cl_train_loop(method_name, method_obj, model, all_df, test_df, dirs,
                              model_name=model_name)
            if r:
                r["model_name"] = model_name   # ensure model tag is stored
                model_results.append(r)
            del model; torch.cuda.empty_cache()

        all_model_results[model_name] = model_results
        if not CL_VARIANT:
            plot_cl_comparison(model_results, model_name=model_name)

        print("\n" + "=" * 75)
        print(f"  {model_name} — CL Summary")
        print(f"  {'Method':<12} {'AA':>8} {'BWT':>8} {'FWT':>8} {'TestAcc':>10} {'F1':>8}")
        print("  " + "-" * 70)
        for r in model_results:
            print(f"  {r['method']:<12} "
                  f"{r.get('AA', 0):>8.4f} "
                  f"{r.get('BWT', 0):>8.4f} "
                  f"{r.get('FWT', 0):>8.4f} "
                  f"{r.get('final_test_acc', 0):>10.4f} "
                  f"{r.get('final_f1_macro', 0):>8.4f}")
        print("=" * 75)

    # ── Save overall best CL model (best AA across all models × methods) ──
    best_r, best_mn = None, None
    for mn, results in all_model_results.items():
        for r in results:
            if best_r is None or r.get("AA", 0) > best_r.get("AA", 0):
                best_r, best_mn = r, mn
    if best_r is not None and not CL_VARIANT:
        _best_safe = best_r["method"].lower().replace("-", "_").replace("+", "p")
        _src = os.path.join(MODELS_DIR, f"cl_{best_mn}_{_best_safe}_best.pth")
        _dst = os.path.join(MODELS_DIR, "cl_overall_best.pth")
        if os.path.exists(_src):
            import shutil as _shutil
            _shutil.copy2(_src, _dst)
            print(f"\n  [CL] Overall best model: {best_mn} / {best_r['method']} "
                  f"(AA={best_r['AA']:.4f}) → cl_overall_best.pth")

    # ── Cross-model summary plot ──────────────────────────────────────────
    if not CL_VARIANT:
        plot_cl_cross_model_summary(all_model_results)

    # ── Final cross-model table ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  {'Model':<28} {'Method':<12} {'AA':>8} {'BWT':>8} {'TestAcc':>10} {'F1':>8}")
    print("  " + "-" * 85)
    for mn, results in all_model_results.items():
        for r in results:
            print(f"  {mn:<28} {r['method']:<12} {r.get('AA', 0):>8.4f} {r.get('BWT', 0):>8.4f} "
                  f"{r.get('final_test_acc', 0):>10.4f} {r.get('final_f1_macro', 0):>8.4f}")
    print("=" * 90)
