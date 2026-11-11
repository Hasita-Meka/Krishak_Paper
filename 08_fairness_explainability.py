"""
08_fairness_explainability.py — Fairness Analysis & Full Explainability Suite
==============================================================================
Loads all trained models (Baseline CNN, QEFS, GNAS, CL) and generates:

  FAIRNESS ANALYSIS (per-class as "group"):
    • Equal Opportunity (TPR per class)
    • Equalized Odds (TPR + FPR per class)
    • Demographic / Prediction Parity
    • Disparate Impact Ratio (min TPR / max TPR)
    • Statistical Parity Difference
    • Theil Index (inequality across classes)
    • Error Rate Parity (FNR, FPR per class)
    • Calibration per class (reliability diagrams)
    • ROC curves per class (One-vs-Rest)
    • Precision–Recall curves per class
    • Fairness radar charts per model
    • Cross-model fairness comparison heatmaps

  EXPLAINABILITY (XAI):
    • SHAP — GradientExplainer: beeswarm, bar, waterfall, heatmap, force
    • Grad-CAM (standard, Grad-CAM++, EigenCAM, Score-CAM)
    • Integrated Gradients (manual implementation, no captum needed)
    • Vanilla Saliency Maps (gradient × input)
    • Occlusion Sensitivity Maps
    • t-SNE & UMAP feature embedding visualisations
    • LIME image explanations
    • Class Activation Maps (CAM) overlay per class
    • Attribution comparison across XAI methods
    • Prediction confidence histograms & calibration

  ALL-PIPELINE PLOTS:
    • Re-generates any missing plots from scripts 03–07
    • Grand accuracy leaderboard with confidence intervals
    • Method-level ablation waterfall chart
    • Per-class performance across all techniques

Run:
    python 08_fairness_explainability.py

Install extras (if not already present):
    pip install shap lime scikit-learn umap-learn captum
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os, sys, json, time, copy, warnings, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as sp_stats

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
from torch.utils.data import DataLoader
import torchvision.transforms as T
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# Optional heavy deps — graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[warn] shap not installed — SHAP plots skipped. pip install shap")

try:
    import lime
    import lime.lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[warn] lime not installed — LIME plots skipped. pip install lime")

try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[warn] umap-learn not installed — UMAP plots skipped. pip install umap-learn")

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    set_seed, get_model, get_feature_extractor, PlantDiseaseDataset, split_dataframe,
    get_val_transform, load_results, save_results,
    plot_confusion_matrix, plot_class_accuracies, plot_training_curves,
    DEVICE, CLASS_NAMES, NUM_CLASSES, PLOTS_DIR, RESULTS_DIR, MODELS_DIR,
    DATASET_DIR, BALANCED_CSV, TRAIN_IMAGES, USE_AMP, IMAGENET_MEAN, IMAGENET_STD
)

# ── output directories ───────────────────────────────────────────────────────
FAIR_DIR = os.path.join(PLOTS_DIR, "fairness")
XAI_DIR  = os.path.join(PLOTS_DIR, "xai")
for d in [FAIR_DIR, XAI_DIR]:
    os.makedirs(d, exist_ok=True)

set_seed(SEED)
import multiprocessing as _mp_banner
if _mp_banner.current_process().name == "MainProcess":
    print(f"\n{'='*70}")
    print(" Krishak — Fairness & Explainability (Script 08)")
    print(f"{'='*70}")
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ════════════════════════════════════════════════════════════════════════════
# ① DATA — load balanced dataset & test split
# ════════════════════════════════════════════════════════════════════════════
AUG_IMAGES = os.path.join(DATASET_DIR, "aug_images")

class MultiDirDataset(PlantDiseaseDataset):
    """Extends PlantDiseaseDataset to also search in aug_images (flat layout)."""
    def _find_image(self, row):
        # 1. train_images / label / image_id
        p1 = os.path.join(self.images_dir, str(row.get("label", "")), row["image_id"])
        if os.path.exists(p1): return p1
        # 2. train_images / image_id
        p2 = os.path.join(self.images_dir, row["image_id"])
        if os.path.exists(p2): return p2
        # 3. aug_images / image_id  (flat layout)
        p3 = os.path.join(AUG_IMAGES, row["image_id"])
        if os.path.exists(p3): return p3
        raise FileNotFoundError(
            f"Image not found in any layout:\n  {p1}\n  {p2}\n  {p3}")

def get_test_dataset():
    df = pd.read_csv(BALANCED_CSV)
    _, _, test_df = split_dataframe(df)
    ds = MultiDirDataset(test_df, TRAIN_IMAGES, get_val_transform())
    return ds, test_df

def get_test_loader(batch_size=32):
    ds, test_df = get_test_dataset()
    # num_workers=0 avoids Windows spawn re-importing this module in workers
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False), ds

import multiprocessing as _mp
if _mp.current_process().name == "MainProcess":
    TEST_LOADER, TEST_DS = get_test_loader()
    # ── True labels (numpy) ──────────────────────────────────────────────
    ALL_TRUE = np.array([TEST_DS[i][1] for i in range(len(TEST_DS))])
else:
    TEST_LOADER = TEST_DS = ALL_TRUE = None


# ════════════════════════════════════════════════════════════════════════════
# ② MODEL LOADING HELPERS
# ════════════════════════════════════════════════════════════════════════════
def load_baseline_model(model_name: str) -> Optional[nn.Module]:
    path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found"); return None
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = get_model(model_name).to(DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer_model(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (preds, labels, probs)."""
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        with autocast(device_type="cuda", enabled=USE_AMP):
            out = model(imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]
        probs = F.softmax(out, dim=1)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.array(all_preds), np.array(all_labels),
            np.vstack(all_probs))


@torch.no_grad()
def extract_features(model: nn.Module, model_name: str,
                     loader: DataLoader) -> np.ndarray:
    """Extract penultimate-layer features."""
    hooks = {}
    def _hook(m, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        hooks["feat"] = out.detach()

    _layer_map = {
        "densenet161":        "features",
        "googlenet":          "dropout",
        "mobilenet_v3_large": "features",
        "resnet50":           "avgpool",
        "shufflenet_v2":      "conv5",
    }
    target_name = _layer_map.get(model_name)
    if target_name is None:
        return np.array([])
    layer = dict(model.named_modules()).get(target_name)
    if layer is None:
        return np.array([])
    h = layer.register_forward_hook(_hook)

    feats = []
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        with autocast(device_type="cuda", enabled=USE_AMP):
            _ = model(imgs)
        f = hooks["feat"]
        f = f.view(f.size(0), -1).cpu().numpy()
        feats.append(f)
    h.remove()
    return np.vstack(feats)


# ════════════════════════════════════════════════════════════════════════════
# ③ FAIRNESS METRICS
# ════════════════════════════════════════════════════════════════════════════
def compute_fairness_metrics(preds: np.ndarray, labels: np.ndarray,
                              probs: np.ndarray) -> Dict:
    """
    Compute comprehensive fairness metrics treating each class as a group.
    """
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    metrics = {}

    tpr, fpr, tnr, fnr, ppv, npv, pp_rate = [], [], [], [], [], [], []

    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp          # actual c predicted as not-c
        fp = cm[:, c].sum() - tp          # not-c predicted as c
        tn = cm.sum() - tp - fn - fp

        _tpr = tp / max(tp + fn, 1)       # recall / sensitivity
        _fpr = fp / max(fp + tn, 1)       # fall-out
        _tnr = tn / max(tn + fp, 1)       # specificity
        _fnr = fn / max(fn + tp, 1)       # miss rate
        _ppv = tp / max(tp + fp, 1)       # precision
        _npv = tn / max(tn + fn, 1)
        _ppr = (preds == c).sum() / len(preds)  # prediction positive rate

        tpr.append(_tpr); fpr.append(_fpr); tnr.append(_tnr)
        fnr.append(_fnr); ppv.append(_ppv); npv.append(_npv)
        pp_rate.append(_ppr)

    tpr = np.array(tpr); fpr = np.array(fpr); fnr = np.array(fnr)
    pp_rate = np.array(pp_rate)

    # ── Fairness scalars ────────────────────────────────────────────────────
    # Disparate Impact Ratio (min TPR / max TPR)  → ideally 1.0
    dir_score = tpr.min() / max(tpr.max(), 1e-9)

    # Statistical Parity Difference (max PPR − min PPR) → ideally 0
    spd = pp_rate.max() - pp_rate.min()

    # Equal Opportunity Difference (max TPR − min TPR) → ideally 0
    eod = tpr.max() - tpr.min()

    # Equalized Odds Difference (max of EOD_tpr, EOD_fpr) → ideally 0
    eqodds = max(tpr.max() - tpr.min(), fpr.max() - fpr.min())

    # Theil Index (inequality in TPR across classes) → 0 = perfect equality
    mu = tpr.mean()
    theil = np.mean((tpr / max(mu, 1e-9)) * np.log(tpr / max(mu, 1e-9) + 1e-9))

    # Calibration — overall ECE (Expected Calibration Error)
    n_bins   = 10
    bin_lows = np.linspace(0, 1 - 1/n_bins, n_bins)
    ece      = 0.0
    for bl in bin_lows:
        bh     = bl + 1/n_bins
        mask   = (probs.max(axis=1) >= bl) & (probs.max(axis=1) < bh)
        if mask.sum() == 0:
            continue
        avg_conf = probs.max(axis=1)[mask].mean()
        avg_acc  = (preds[mask] == labels[mask]).mean()
        ece     += mask.mean() * abs(avg_acc - avg_conf)

    metrics.update({
        "tpr":   tpr.tolist(),    # Equal Opportunity vector
        "fpr":   fpr.tolist(),
        "fnr":   fnr.tolist(),
        "ppv":   list(ppv),
        "pp_rate": pp_rate.tolist(),
        "dir":   float(dir_score),   # Disparate Impact Ratio
        "spd":   float(spd),         # Statistical Parity Difference
        "eod":   float(eod),         # Equal Opportunity Difference
        "eqodds": float(eqodds),     # Equalized Odds Difference
        "theil": float(theil),       # Theil Index
        "ece":   float(ece),         # Expected Calibration Error
    })
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# ④ FAIRNESS PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_equal_opportunity(fm: Dict, name: str, save_dir=FAIR_DIR):
    tpr = np.array(fm["tpr"])
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ["#2ecc71" if t >= 0.9 else "#e67e22" if t >= 0.7
               else "#e74c3c" for t in tpr]
    bars = ax.bar(CLASS_NAMES, tpr, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(tpr.mean(), color="steelblue", ls="--", lw=1.5, label=f"Mean TPR = {tpr.mean():.3f}")
    ax.axhline(0.8, color="red", ls=":", lw=1.2, label="80% threshold")
    for bar, v in zip(bars, tpr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set(title=f"Equal Opportunity (TPR per Class) — {name}",
           ylabel="True Positive Rate", ylim=(0, 1.1))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=12)
    ax.legend(fontsize=12); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_eqopp_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_equalized_odds(fm: Dict, name: str, save_dir=FAIR_DIR):
    tpr = np.array(fm["tpr"])
    fpr = np.array(fm["fpr"])
    x   = np.arange(NUM_CLASSES)
    w   = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, tpr, w, label="TPR (Sensitivity)", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, fpr, w, label="FPR (Fall-out)",    color="#e74c3c", alpha=0.85)
    ax.set(title=f"Equalized Odds — {name}", ylabel="Rate", ylim=(0, 1.05))
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_eqodds_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_demographic_parity(fm: Dict, name: str, save_dir=FAIR_DIR):
    ppr = np.array(fm["pp_rate"])
    expected = 1.0 / NUM_CLASSES
    fig, ax  = plt.subplots(figsize=(12, 5))
    colors   = ["#27ae60" if abs(p - expected) < 0.02 else "#f39c12"
                if abs(p - expected) < 0.05 else "#e74c3c" for p in ppr]
    bars = ax.bar(CLASS_NAMES, ppr, color=colors)
    ax.axhline(expected, color="navy", ls="--", lw=1.5, label=f"Equal share = {expected:.3f}")
    for bar, v in zip(bars, ppr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12)
    ax.set(title=f"Demographic (Prediction) Parity — {name}",
           ylabel="Prediction Positive Rate", ylim=(0, max(ppr)*1.2))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_demparity_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_error_rate_parity(fm: Dict, name: str, save_dir=FAIR_DIR):
    fnr = np.array(fm["fnr"])
    fpr = np.array(fm["fpr"])
    x   = np.arange(NUM_CLASSES); w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, fnr, w, label="FNR (Miss Rate)",  color="#8e44ad", alpha=0.85)
    ax.bar(x + w/2, fpr, w, label="FPR (Fall-out)",   color="#e74c3c", alpha=0.85)
    ax.set(title=f"Error Rate Parity — {name}", ylabel="Error Rate", ylim=(0, 1.0))
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_errparity_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_fairness_radar(fm: Dict, name: str, save_dir=FAIR_DIR):
    cats   = CLASS_NAMES
    N      = len(cats)
    tpr    = np.array(fm["tpr"])
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]; vals = tpr.tolist() + [tpr[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, "o-", lw=2, color="#3498db")
    ax.fill(angles, vals, alpha=0.25, color="#3498db")
    # Perfect fairness ring
    ax.plot(angles, [1.0]*len(angles), "--", lw=1, color="green", alpha=0.6, label="Perfect (1.0)")
    ax.plot(angles, [0.8]*len(angles), ":",  lw=1, color="orange",alpha=0.6, label="80% threshold")
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=12)
    ax.set(title=f"Equal Opportunity Radar — {name}", ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_radar_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_calibration_per_class(labels: np.ndarray, probs: np.ndarray,
                                name: str, save_dir=FAIR_DIR):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for c, ax in enumerate(axes):
        y_true = (labels == c).astype(int)
        y_prob = probs[:, c]
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
            ax.plot(mean_pred, frac_pos, "s-", color="#e74c3c", lw=2, ms=5)
            ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.2, color="#e74c3c")
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=CLASS_NAMES[c], xlim=(0,1), ylim=(0,1), xlabel="Confidence",
               ylabel="Fraction Positive")
        ax.grid(alpha=0.3)
    plt.suptitle(f"Calibration Curves per Class — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_calibration_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_roc_per_class(labels: np.ndarray, probs: np.ndarray,
                        name: str, save_dir=FAIR_DIR):
    y_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for c, color in enumerate(colors):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, color=color,
                label=f"{CLASS_NAMES[c]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set(title=f"ROC Curves (One-vs-Rest) — {name}",
           xlabel="FPR", ylabel="TPR", xlim=(-0.01, 1.01), ylim=(-0.01, 1.05))
    ax.legend(fontsize=12, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_roc_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_pr_per_class(labels: np.ndarray, probs: np.ndarray,
                       name: str, save_dir=FAIR_DIR):
    y_bin  = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for c, color in enumerate(colors):
        prec, rec, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
        ap           = average_precision_score(y_bin[:, c], probs[:, c])
        ax.plot(rec, prec, lw=1.5, color=color,
                label=f"{CLASS_NAMES[c]} (AP={ap:.3f})")
    ax.set(title=f"Precision–Recall Curves — {name}",
           xlabel="Recall", ylabel="Precision", xlim=(-0.01, 1.01), ylim=(-0.01, 1.05))
    ax.legend(fontsize=12, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_pr_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_confidence_histogram(probs: np.ndarray, preds: np.ndarray,
                               labels: np.ndarray, name: str, save_dir=FAIR_DIR):
    max_conf = probs.max(axis=1)
    correct  = (preds == labels)
    fig, ax  = plt.subplots(figsize=(10, 5))
    ax.hist(max_conf[correct],  bins=40, alpha=0.6, color="#27ae60", label="Correct")
    ax.hist(max_conf[~correct], bins=40, alpha=0.6, color="#e74c3c", label="Wrong")
    ax.set(title=f"Prediction Confidence Distribution — {name}",
           xlabel="Confidence", ylabel="Count")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fair_confhist_{name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_theil_index_comparison(all_metrics: Dict[str, Dict], save_dir=FAIR_DIR):
    """Bar chart of Theil index across all models — lower = fairer."""
    names  = list(all_metrics.keys())
    theils = [all_metrics[n]["theil"] for n in names]
    dirs   = [all_metrics[n]["dir"]   for n in names]
    eods   = [all_metrics[n]["eod"]   for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, vals, title, color in zip(
            axes,
            [theils, dirs, eods],
            ["Theil Index (↓ better)", "Disparate Impact Ratio (↑ better)",
             "Equal Opportunity Diff (↓ better)"],
            ["#8e44ad", "#27ae60", "#e74c3c"]):
        bars = ax.barh(names, vals, color=color, alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=12)
        ax.set(title=title, xlabel="Score")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Fairness Scalar Comparison Across All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fair_scalar_comparison.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_tpr_heatmap_all_models(all_metrics: Dict[str, Dict], save_dir=FAIR_DIR):
    """Heatmap: models × classes — TPR (equal opportunity)."""
    names = list(all_metrics.keys())
    mat   = np.array([all_metrics[n]["tpr"] for n in names])
    fig, ax = plt.subplots(figsize=(14, max(4, len(names)//2 + 1)))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=CLASS_NAMES, yticklabels=names,
                vmin=0.6, vmax=1.0, ax=ax, linewidths=0.3, linecolor="white")
    ax.set(title="Equal Opportunity (TPR) Heatmap — All Models × Classes",
           xlabel="Class", ylabel="Model")
    plt.xticks(rotation=40, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fair_tpr_heatmap_all.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


def plot_ece_comparison(all_metrics: Dict[str, Dict], save_dir=FAIR_DIR):
    names = list(all_metrics.keys())
    eces  = [all_metrics[n]["ece"] for n in names]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ["#27ae60" if e < 0.05 else "#f39c12" if e < 0.1 else "#e74c3c" for e in eces]
    bars = ax.bar(names, eces, color=colors, edgecolor="white")
    ax.axhline(0.05, ls="--", color="orange", lw=1.5, label="0.05 threshold (good)")
    ax.axhline(0.10, ls=":",  color="red",    lw=1.5, label="0.10 threshold (poor)")
    for bar, v in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=12)
    ax.set(title="Expected Calibration Error (ECE) — All Models",
           ylabel="ECE (↓ better)")
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fair_ece_all.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# ⑤ XAI — Integrated Gradients (no captum needed)
# ════════════════════════════════════════════════════════════════════════════
def integrated_gradients(model: nn.Module, inp: torch.Tensor,
                          target_class: int, steps: int = 50) -> np.ndarray:
    baseline  = torch.zeros_like(inp)
    scaled    = [baseline + (float(i)/steps) * (inp - baseline) for i in range(steps + 1)]
    grads     = []
    model.eval()
    for s in scaled:
        s = s.to(DEVICE).requires_grad_(True)
        out = model(s)
        if isinstance(out, (tuple, list)):
            out = out[0]
        score = out[0, target_class]
        model.zero_grad()
        score.backward()
        grads.append(s.grad.detach().cpu().numpy())
    avg_grad = np.mean(grads, axis=0)
    ig = (inp.cpu().numpy() - baseline.cpu().numpy()) * avg_grad
    return ig.squeeze(0)   # (C, H, W)


def plot_integrated_gradients(model: nn.Module, model_name: str,
                               dataset: PlantDiseaseDataset, save_dir=XAI_DIR,
                               n_samples: int = 5):
    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(idxs):
        img_t, label = dataset[idx]
        inp = img_t.unsqueeze(0).to(DEVICE)
        try:
            ig  = integrated_gradients(model, inp, label)
            ig_map = np.abs(ig).sum(axis=0)   # sum over channels
            ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
            orig   = inv_norm(img_t).permute(1,2,0).clamp(0,1).numpy()
            axes[row,0].imshow(orig)
            axes[row,0].set_title(f"Original: {CLASS_NAMES[label]}", fontsize=12)
            axes[row,1].imshow(ig_map, cmap="hot")
            axes[row,1].set_title("IG Attribution", fontsize=12)
            axes[row,2].imshow(orig); axes[row,2].imshow(ig_map, cmap="jet", alpha=0.5)
            axes[row,2].set_title("IG Overlay", fontsize=12)
        except Exception as e:
            for ax in axes[row]: ax.text(0.5,0.5,f"err: {e}", ha="center", transform=ax.transAxes)
        for ax in axes[row]: ax.axis("off")

    plt.suptitle(f"Integrated Gradients — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_ig_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [IG] saved → xai_ig_{model_name}.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑥ XAI — Vanilla Saliency Maps
# ════════════════════════════════════════════════════════════════════════════
def plot_saliency_maps(model: nn.Module, model_name: str,
                        dataset: PlantDiseaseDataset, save_dir=XAI_DIR,
                        n_samples: int = 5):
    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    model.eval()
    for row, idx in enumerate(idxs):
        img_t, label = dataset[idx]
        inp = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
        try:
            out = model(inp)
            if isinstance(out, (tuple, list)): out = out[0]
            out[0, label].backward()
            sal = inp.grad.detach().cpu().numpy().squeeze(0)   # (C,H,W)
            sal = np.abs(sal).max(axis=0)
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            orig = inv_norm(img_t).permute(1,2,0).clamp(0,1).numpy()
            axes[row,0].imshow(orig)
            axes[row,0].set_title(f"Original: {CLASS_NAMES[label]}", fontsize=12)
            axes[row,1].imshow(sal, cmap="hot")
            axes[row,1].set_title("Saliency Map", fontsize=12)
            axes[row,2].imshow(orig); axes[row,2].imshow(sal, cmap="jet", alpha=0.5)
            axes[row,2].set_title("Saliency Overlay", fontsize=12)
        except Exception as e:
            for ax in axes[row]: ax.text(0.5,0.5,str(e), ha="center", transform=ax.transAxes)
        for ax in axes[row]: ax.axis("off")
        model.zero_grad()

    plt.suptitle(f"Vanilla Saliency Maps — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_saliency_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [Saliency] saved → xai_saliency_{model_name}.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑦ XAI — Grad-CAM (standard + Grad-CAM++)
# ════════════════════════════════════════════════════════════════════════════
_LAYER_MAP = {
    "densenet161":        "features.denseblock4.denselayer24.conv2",
    "googlenet":          "inception5b.branch4.1.conv",
    "mobilenet_v3_large": "features.16.block.2",
    "resnet50":           "layer4.2.conv3",
    "shufflenet_v2":      "conv5.0",
}

def get_grad_cam(model: nn.Module, inp: torch.Tensor, target: int,
                  layer_name: str, plusplus: bool = False):
    acts, grads = {}, {}

    def fhook(m, i, o):
        acts["a"] = o.clone().detach()
    def bhook(m, gi, go):
        grads["g"] = go[0].clone().detach() if go[0] is not None else None

    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        return None
    fh = layer.register_forward_hook(fhook)
    bh = layer.register_backward_hook(bhook)

    model.eval(); model.zero_grad()
    inp = inp.to(DEVICE).requires_grad_(True)
    out = model(inp)
    if isinstance(out, (tuple, list)): out = out[0]

    if plusplus:
        # Grad-CAM++: use second-order gradients of class score
        out[0, target].backward(retain_graph=True)
    else:
        out[0, target].backward()

    fh.remove(); bh.remove()
    if grads["g"] is None:
        return None

    A = acts["a"].squeeze(0)  # (C, H, W)
    G = grads["g"].squeeze(0) # (C, H, W)

    if plusplus:
        alpha_num   = G ** 2
        alpha_den   = 2 * G**2 + (A * G**3).sum(dim=(1,2), keepdim=True) + 1e-8
        alpha       = alpha_num / alpha_den
        weights     = (F.relu(out[0, target]) * alpha * G).sum(dim=(1,2))
    else:
        weights = G.mean(dim=(1, 2))  # GAP of gradients

    cam = (weights[:, None, None] * A).sum(0)
    cam = F.relu(torch.from_numpy(cam.cpu().numpy() if not torch.is_tensor(cam) else cam.cpu())).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def plot_grad_cam_variants(model: nn.Module, model_name: str,
                            dataset: PlantDiseaseDataset, save_dir=XAI_DIR,
                            n_samples: int = 4):
    layer = _LAYER_MAP.get(model_name)
    if layer is None:
        print(f"  [Grad-CAM] no layer mapping for {model_name}"); return

    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)

    for variant, plusplus in [("GradCAM", False), ("GradCAM++", True)]:
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1: axes = axes[np.newaxis,:]

        for row, idx in enumerate(idxs):
            img_t, label = dataset[idx]
            inp  = img_t.unsqueeze(0)
            orig = inv_norm(img_t).permute(1,2,0).clamp(0,1).numpy()
            cam = None
            try:
                cam = get_grad_cam(model, inp, label, layer, plusplus=plusplus)
            except Exception:
                pass
            from PIL import Image as PILImage
            if cam is not None:
                try:
                    cam_resized = np.array(
                        PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
                            (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)) / 255.0
                except Exception:
                    cam_resized = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            else:
                cam_resized = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

            axes[row,0].imshow(orig)
            axes[row,0].set_title(f"Original: {CLASS_NAMES[label]}", fontsize=12)
            axes[row,1].imshow(cam_resized, cmap="jet")
            axes[row,1].set_title(f"{variant} Map", fontsize=12)
            axes[row,2].imshow(orig); axes[row,2].imshow(cam_resized, cmap="jet", alpha=0.45)
            axes[row,2].set_title(f"{variant} Overlay", fontsize=12)
            for ax in axes[row]: ax.axis("off")

        plt.suptitle(f"{variant} — {model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        tag = "gradcampp" if plusplus else "gradcam"
        plt.savefig(os.path.join(save_dir, f"xai_{tag}_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15)
        plt.close()
        print(f"  [{variant}] saved → xai_{tag}_{model_name}.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑧ XAI — EigenCAM
# ════════════════════════════════════════════════════════════════════════════
def eigen_cam(model: nn.Module, inp: torch.Tensor, layer_name: str) -> Optional[np.ndarray]:
    acts = {}
    def fhook(m, i, o): acts["a"] = o.clone().detach()
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None: return None
    fh = layer.register_forward_hook(fhook)
    model.eval()
    with torch.no_grad():
        _ = model(inp.to(DEVICE))
    fh.remove()
    A = acts["a"].squeeze(0).cpu().numpy()  # (C, H, W)
    A_flat = A.reshape(A.shape[0], -1).T    # (H*W, C)
    _, _, Vt = np.linalg.svd(A_flat - A_flat.mean(axis=0), full_matrices=False)
    principal = Vt[0]                        # (C,)
    cam = (A * principal[:, None, None]).sum(0)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def plot_eigen_cam(model: nn.Module, model_name: str,
                   dataset: PlantDiseaseDataset, save_dir=XAI_DIR,
                   n_samples: int = 4):
    layer = _LAYER_MAP.get(model_name)
    if layer is None: return
    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1: axes = axes[np.newaxis,:]
    for row, idx in enumerate(idxs):
        img_t, label = dataset[idx]
        inp  = img_t.unsqueeze(0)
        orig = inv_norm(img_t).permute(1,2,0).clamp(0,1).numpy()
        cam  = eigen_cam(model, inp, layer)
        if cam is None:
            for ax in axes[row]: ax.axis("off")
            continue
        from PIL import Image as PILImage
        cam_resized = np.array(
            PILImage.fromarray((cam*255).astype(np.uint8)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)) / 255.0
        axes[row,0].imshow(orig); axes[row,0].set_title(f"Original: {CLASS_NAMES[label]}", fontsize=12)
        axes[row,1].imshow(cam_resized, cmap="jet"); axes[row,1].set_title("EigenCAM", fontsize=12)
        axes[row,2].imshow(orig); axes[row,2].imshow(cam_resized, cmap="jet", alpha=0.45)
        axes[row,2].set_title("EigenCAM Overlay", fontsize=12)
        for ax in axes[row]: ax.axis("off")
    plt.suptitle(f"EigenCAM — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_eigencam_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [EigenCAM] saved → xai_eigencam_{model_name}.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑨ XAI — Occlusion Sensitivity
# ════════════════════════════════════════════════════════════════════════════
def occlusion_sensitivity(model: nn.Module, inp: torch.Tensor,
                           target: int, stride: int = 16,
                           patch: int = 32) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        base_score = F.softmax(
            (lambda o: o[0] if isinstance(o, (tuple,list)) else o)(model(inp.to(DEVICE))),
            dim=1)[0, target].item()
    H, W = inp.shape[-2], inp.shape[-1]
    sens = np.zeros((H, W))
    cnts = np.zeros((H, W))
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            occ = inp.clone()
            occ[0, :, y:y+patch, x:x+patch] = 0.0
            with torch.no_grad():
                score = F.softmax(
                    (lambda o: o[0] if isinstance(o,(tuple,list)) else o)(model(occ.to(DEVICE))),
                    dim=1)[0, target].item()
            drop = base_score - score
            sens[y:y+patch, x:x+patch] += drop
            cnts[y:y+patch, x:x+patch] += 1
    cnts[cnts == 0] = 1
    sens /= cnts
    return sens


def plot_occlusion_sensitivity(model: nn.Module, model_name: str,
                                dataset: PlantDiseaseDataset, save_dir=XAI_DIR,
                                n_samples: int = 3):
    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1: axes = axes[np.newaxis,:]
    for row, idx in enumerate(idxs):
        img_t, label = dataset[idx]
        orig = inv_norm(img_t).permute(1,2,0).clamp(0,1).numpy()
        try:
            # Use smaller image for speed (112×112)
            small_t = T.Compose([T.Resize((112,112))])(img_t)
            sens = occlusion_sensitivity(model, small_t.unsqueeze(0),
                                         label, stride=16, patch=28)
            from PIL import Image as PILImage
            sens_resized = np.array(
                PILImage.fromarray(
                    ((sens - sens.min())/(sens.max()-sens.min()+1e-8)*255).astype(np.uint8)
                ).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)) / 255.0
            axes[row,0].imshow(orig); axes[row,0].set_title(f"Original: {CLASS_NAMES[label]}", fontsize=12)
            axes[row,1].imshow(sens_resized, cmap="RdBu_r"); axes[row,1].set_title("Occlusion Sensitivity", fontsize=12)
            axes[row,2].imshow(orig); axes[row,2].imshow(sens_resized, cmap="jet", alpha=0.5)
            axes[row,2].set_title("Sensitivity Overlay", fontsize=12)
        except Exception as e:
            for ax in axes[row]: ax.text(0.5,0.5,str(e),ha="center",transform=ax.transAxes)
        for ax in axes[row]: ax.axis("off")
    plt.suptitle(f"Occlusion Sensitivity — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_occlusion_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [Occlusion] saved → xai_occlusion_{model_name}.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑩ XAI — SHAP (GradientExplainer)
# ════════════════════════════════════════════════════════════════════════════
def plot_shap_suite(model: nn.Module, model_name: str,
                    dataset: PlantDiseaseDataset, save_dir=XAI_DIR):
    if not SHAP_AVAILABLE or SMOKE:
        return
    print(f"  [SHAP] computing for {model_name} …")
    model.eval()

    class SafeWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            o = self.m(x)
            return o[0] if isinstance(o,(tuple,list)) else o

    wrapped = SafeWrap(model)
    bg_idx  = np.random.choice(len(dataset), SHAP_BACKGROUND, replace=False)
    te_idx  = np.random.choice(len(dataset), SHAP_TEST_SAMPLES, replace=False)
    bg_imgs = torch.stack([dataset[i][0] for i in bg_idx]).to(DEVICE)
    te_imgs = torch.stack([dataset[i][0] for i in te_idx]).to(DEVICE)
    te_lbl  = [dataset[i][1] for i in te_idx]

    try:
        explainer = shap.GradientExplainer(wrapped, bg_imgs)
        shap_vals = explainer.shap_values(te_imgs)   # list[C] of (N,C,H,W)
    except Exception as ex:
        print(f"  [SHAP] GradientExplainer failed ({ex}), trying DeepExplainer …")
        try:
            explainer = shap.DeepExplainer(wrapped, bg_imgs)
            shap_vals = explainer.shap_values(te_imgs)
        except Exception as ex2:
            print(f"  [SHAP] both failed ({ex2}), skipping."); return

    # ── SHAP heatmap: classes × channels ────────────────────────────────────
    # Normalise to (num_classes, C_in) regardless of SHAP version/format.
    # GradientExplainer can return either:
    #   list[K] of (N, C_in, H, W)  — one array per output class (standard)
    #   list[C_in] of (N, K, H, W)  — one array per input channel (older SHAP)
    #   numpy array (N, K, C, H, W) or (N, C, H, W, K)
    if isinstance(shap_vals, list):
        arrays  = [np.abs(sv) for sv in shap_vals]
        n_items = len(arrays)
        ndim0   = arrays[0].ndim if arrays else 4

        if n_items == NUM_CLASSES and ndim0 == 4:
            # Standard: list[K] of (N, C_in, H, W) — one per output class
            mean_abs = np.array([a.mean(axis=(0, 2, 3)) for a in arrays])   # (K, C_in)
        elif ndim0 == 4 and arrays[0].shape[1] == NUM_CLASSES:
            # Older SHAP: list[C_in] of (N, K, H, W) — one per input channel
            # mean(axis=(0,2,3)) on (N,K,H,W) leaves axis-1 = K
            per_channel = np.array([a.mean(axis=(0, 2, 3)) for a in arrays])  # (C_in, K)
            mean_abs = per_channel.T                                            # (K, C_in)
        elif n_items == NUM_CLASSES and ndim0 == 3:
            mean_abs = np.array([a.mean() for a in arrays]).reshape(-1, 1)
        else:
            # Fallback: scalar per item, tiled across classes
            per_item = np.array([a.mean() for a in arrays])
            mean_abs = np.tile(per_item[np.newaxis, :], (NUM_CLASSES, 1))
    else:
        if shap_vals.ndim == 5 and shap_vals.shape[1] == NUM_CLASSES:
            # (N, K, C, H, W) — class dim at axis 1
            mean_abs = np.abs(shap_vals).mean(axis=(0, 3, 4))  # (K, C)
        elif shap_vals.ndim == 5 and shap_vals.shape[-1] == NUM_CLASSES:
            # (N, C, H, W, K) — class dim at last axis (newer SHAP)
            mean_abs = np.abs(shap_vals).mean(axis=(0, 2, 3)).T  # (K, C)
        elif shap_vals.ndim == 5:
            # unknown 5-D: flatten to (1, all)
            mean_abs = np.abs(shap_vals).mean(axis=0).reshape(1, -1)
        elif shap_vals.ndim == 4:        # (N, C, H, W)
            mean_abs = np.abs(shap_vals).mean(axis=(0, 2, 3), keepdims=False).reshape(1, -1)
        else:
            mean_abs = np.abs(shap_vals).reshape(1, -1)

    # Row labels: class names if rows == NUM_CLASSES, else generic
    row_lbl = CLASS_NAMES if mean_abs.shape[0] == NUM_CLASSES \
              else [f"Entry{i}" for i in range(mean_abs.shape[0])]
    col_lbl = ["R", "G", "B"][:mean_abs.shape[1]] if mean_abs.shape[1] <= 3 \
              else [f"Ch{i}" for i in range(mean_abs.shape[1])]

    fig, ax = plt.subplots(figsize=(max(8, mean_abs.shape[1]), max(4, len(row_lbl)//2 + 2)))
    sns.heatmap(mean_abs, annot=mean_abs.shape[1] <= 10, fmt=".4f", cmap="viridis",
                xticklabels=col_lbl, yticklabels=row_lbl, ax=ax)
    ax.set(title=f"SHAP Feature Importance Heatmap — {model_name}",
           xlabel="Input Channel", ylabel="Output Class / Entry")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_shap_heatmap_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()

    # ── SHAP bar (mean |SHAP| per class or per channel) ─────────────────────
    # GradientExplainer may return either:
    #   list of NUM_CLASSES arrays  (N, C, H, W)  — one per output class
    #   list of C arrays            (N, H, W)       — one per input channel (older SHAP)
    if isinstance(shap_vals, list) and len(shap_vals) == NUM_CLASSES:
        # Per-class format — one mean-abs scalar per output class
        class_importance = np.array([np.abs(sv).mean() for sv in shap_vals])
        bar_labels = CLASS_NAMES
    elif isinstance(shap_vals, list):
        # Per-channel or other format — label by index
        class_importance = np.array([np.abs(sv).mean() for sv in shap_vals])
        bar_labels = [f"Ch{i}" for i in range(len(shap_vals))]
    else:
        # numpy array — detect axis layout by shape
        if shap_vals.ndim == 5 and shap_vals.shape[1] == NUM_CLASSES:
            # (N, K, C, H, W)
            class_importance = np.abs(shap_vals).mean(axis=(0, 2, 3, 4))
            bar_labels = CLASS_NAMES
        elif shap_vals.ndim == 5 and shap_vals.shape[-1] == NUM_CLASSES:
            # (N, C, H, W, K) — newer SHAP
            class_importance = np.abs(shap_vals).mean(axis=(0, 1, 2, 3))
            bar_labels = CLASS_NAMES
        else:
            class_importance = np.abs(shap_vals).mean(axis=0).flatten()
            bar_labels = [f"Ch{i}" for i in range(len(class_importance))]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(bar_labels, class_importance,
                  color=sns.color_palette("viridis", len(class_importance)))
    ax.set(title=f"SHAP Mean |Value| per Class — {model_name}", ylabel="Mean |SHAP|")
    ax.set_xticklabels(bar_labels, rotation=40, ha="right")
    for bar, v in zip(bars, class_importance):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1e-6,
                f"{v:.5f}", ha="center", va="bottom", fontsize=12)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_shap_bar_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()

    # ── SHAP image-level overlay (first 4 samples) ──────────────────────────
    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    n_show = min(4, SHAP_TEST_SAMPLES)
    fig, axes = plt.subplots(n_show, 2, figsize=(10, 4*n_show))
    if n_show == 1: axes = axes[np.newaxis,:]
    for i in range(n_show):
        orig = inv_norm(te_imgs[i].cpu()).permute(1,2,0).clamp(0,1).numpy()
        lbl  = int(te_lbl[i])
        if isinstance(shap_vals, list):
            if len(shap_vals) == NUM_CLASSES:
                # Per-class list: shap_vals[class][sample] → (C,H,W)
                sv = shap_vals[lbl][i]
            else:
                # Per-channel list (3 elements): each element is (N,H,W) — stack
                sv = np.stack([shap_vals[c][i] for c in range(len(shap_vals))], axis=0)
        elif shap_vals.ndim == 5 and shap_vals.shape[1] == NUM_CLASSES:
            # (N, NUM_CLASSES, C, H, W)
            sv = shap_vals[i, lbl]                    # (C, H, W)
        elif shap_vals.ndim == 5 and shap_vals.shape[-1] == NUM_CLASSES:
            # (N, C, H, W, NUM_CLASSES) — newer SHAP, class dim at last axis
            sv = shap_vals[i, :, :, :, lbl]          # (C, H, W)
        else:
            # (N, C, H, W) or other — collapse to (C/1, H, W)
            sv = shap_vals[i] if shap_vals.ndim >= 4 else shap_vals[i].reshape(1, -1)
        sv_map = np.abs(sv).sum(axis=0)
        sv_map = (sv_map - sv_map.min()) / (sv_map.max() - sv_map.min() + 1e-8)
        axes[i,0].imshow(orig); axes[i,0].set_title(f"Input: {CLASS_NAMES[lbl]}", fontsize=12)
        axes[i,1].imshow(orig); axes[i,1].imshow(sv_map, cmap="jet", alpha=0.5)
        axes[i,1].set_title("SHAP Overlay", fontsize=12)
        for ax in axes[i]: ax.axis("off")
    plt.suptitle(f"SHAP Attribution Overlays — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_shap_overlay_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [SHAP] all plots saved for {model_name}")


# ════════════════════════════════════════════════════════════════════════════
# ⑪ XAI — LIME image explanations
# ════════════════════════════════════════════════════════════════════════════
def plot_lime(model: nn.Module, model_name: str,
              dataset: PlantDiseaseDataset, save_dir=XAI_DIR,
              n_samples: int = 3):
    if not LIME_AVAILABLE or SMOKE:
        return
    print(f"  [LIME] computing for {model_name} …")
    model.eval()
    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    val_tf = get_val_transform()

    def predict_fn(imgs_np):  # imgs_np: (N, H, W, C) float [0,1]
        tensors = []
        for img_np in imgs_np:
            from PIL import Image as PILImage
            pil = PILImage.fromarray((img_np * 255).astype(np.uint8))
            tensors.append(val_tf(pil))
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, (tuple,list)): out = out[0]
        return F.softmax(out, dim=1).cpu().numpy()

    explainer = lime.lime_image.LimeImageExplainer()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 4*n_samples))
    if n_samples == 1: axes = axes[np.newaxis,:]

    for row, idx in enumerate(idxs):
        img_t, label = dataset[idx]
        orig = inv_norm(img_t).permute(1,2,0).clamp(0,1).numpy()
        try:
            explanation = explainer.explain_instance(
                orig.astype(np.double), predict_fn,
                top_labels=3, hide_color=0, num_samples=300)
            img_exp, mask = explanation.get_image_and_mask(
                label, positive_only=True, num_features=5, hide_rest=False)
            axes[row,0].imshow(orig); axes[row,0].set_title(f"Original: {CLASS_NAMES[label]}", fontsize=12)
            axes[row,1].imshow(mark_boundaries(img_exp, mask))
            axes[row,1].set_title("LIME Positive Regions", fontsize=12)
            img_exp2, mask2 = explanation.get_image_and_mask(
                label, positive_only=False, num_features=10, hide_rest=False)
            axes[row,2].imshow(mark_boundaries(img_exp2, mask2))
            axes[row,2].set_title("LIME All Regions", fontsize=12)
        except Exception as e:
            for ax in axes[row]: ax.text(0.5,0.5,str(e),ha="center",transform=ax.transAxes)
        for ax in axes[row]: ax.axis("off")

    plt.suptitle(f"LIME Explanations — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_lime_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [LIME] saved → xai_lime_{model_name}.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑫ XAI — t-SNE & UMAP Feature Embeddings
# ════════════════════════════════════════════════════════════════════════════
def plot_tsne_umap(model: nn.Module, model_name: str, save_dir=XAI_DIR):
    print(f"  [t-SNE/UMAP] extracting features for {model_name} …")
    feats = extract_features(model, model_name, TEST_LOADER)
    if feats.size == 0:
        print("  [t-SNE/UMAP] feature extraction failed, skipping."); return

    labels = ALL_TRUE
    palette = sns.color_palette("tab10", NUM_CLASSES)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print("  [t-SNE] fitting …")
    n_pts   = min(2000, len(feats))
    idx     = np.random.choice(len(feats), n_pts, replace=False)
    f_sub   = feats[idx]; l_sub = labels[idx]
    import sklearn
    _tsne_iter_kwarg = "max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5) else "n_iter"
    tsne    = TSNE(n_components=2, perplexity=min(30, len(f_sub) - 1), **{_tsne_iter_kwarg: 1000},
                   random_state=SEED, n_jobs=-1)
    emb     = tsne.fit_transform(f_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(NUM_CLASSES):
        mask = l_sub == c
        ax.scatter(emb[mask,0], emb[mask,1], s=15, alpha=0.65,
                   color=palette[c], label=CLASS_NAMES[c])
    ax.set(title=f"t-SNE Feature Embedding — {model_name}", xlabel="Dim 1", ylabel="Dim 2")
    ax.legend(fontsize=12, markerscale=1.5, loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_tsne_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [t-SNE] saved → xai_tsne_{model_name}.png")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    if UMAP_AVAILABLE:
        print("  [UMAP] fitting …")
        try:
            reducer = umap_lib.UMAP(n_components=2, random_state=SEED, n_jobs=-1)
            emb_u   = reducer.fit_transform(f_sub)
            fig, ax = plt.subplots(figsize=(10, 8))
            for c in range(NUM_CLASSES):
                mask = l_sub == c
                ax.scatter(emb_u[mask,0], emb_u[mask,1], s=15, alpha=0.65,
                           color=palette[c], label=CLASS_NAMES[c])
            ax.set(title=f"UMAP Feature Embedding — {model_name}", xlabel="Dim 1", ylabel="Dim 2")
            ax.legend(fontsize=12, markerscale=1.5, loc="best"); ax.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"xai_umap_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
            print(f"  [UMAP] saved → xai_umap_{model_name}.png")
        except Exception as e:
            print(f"  [UMAP] failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ⑬ XAI — Attribution method comparison grid
# ════════════════════════════════════════════════════════════════════════════
def plot_attribution_comparison(model: nn.Module, model_name: str,
                                 dataset: PlantDiseaseDataset, save_dir=XAI_DIR,
                                 n_samples: int = 2):
    """Side-by-side: Original | Saliency | Grad-CAM | IG | EigenCAM."""
    layer = _LAYER_MAP.get(model_name)
    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    cols = ["Original", "Saliency", "Grad-CAM", "Grad-CAM++", "IG", "EigenCAM"]
    fig, axes = plt.subplots(n_samples, len(cols), figsize=(len(cols)*3.5, 4*n_samples))
    if n_samples == 1: axes = axes[np.newaxis,:]
    from PIL import Image as PILImage

    for row, idx in enumerate(idxs):
        img_t, label = dataset[idx]
        orig = inv_norm(img_t).permute(1,2,0).clamp(0,1).numpy()
        inp  = img_t.unsqueeze(0)

        # Saliency
        inp_r = inp.to(DEVICE).requires_grad_(True)
        try:
            model.zero_grad(); out = model(inp_r)
            (out[0] if isinstance(out,(tuple,list)) else out)[0,label].backward()
            sal = np.abs(inp_r.grad.detach().cpu().numpy().squeeze(0)).max(axis=0)
            sal = (sal-sal.min())/(sal.max()-sal.min()+1e-8)
        except Exception: sal = np.zeros((IMAGE_SIZE,IMAGE_SIZE))

        # Grad-CAM
        try:
            cam_gc = get_grad_cam(model, inp, label, layer or "", plusplus=False)
            if cam_gc is not None:
                cam_gc = np.array(PILImage.fromarray((cam_gc*255).astype(np.uint8)).resize(
                    (IMAGE_SIZE,IMAGE_SIZE), PILImage.BILINEAR))/255.0
            else: cam_gc = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
        except Exception: cam_gc = np.zeros((IMAGE_SIZE,IMAGE_SIZE))

        # Grad-CAM++
        try:
            cam_pp = get_grad_cam(model, inp, label, layer or "", plusplus=True)
            if cam_pp is not None:
                cam_pp = np.array(PILImage.fromarray((cam_pp*255).astype(np.uint8)).resize(
                    (IMAGE_SIZE,IMAGE_SIZE), PILImage.BILINEAR))/255.0
            else: cam_pp = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
        except Exception: cam_pp = np.zeros((IMAGE_SIZE,IMAGE_SIZE))

        # Integrated Gradients
        try:
            ig = integrated_gradients(model, inp, label, steps=30)
            ig_map = np.abs(ig).sum(axis=0)
            ig_map = (ig_map-ig_map.min())/(ig_map.max()-ig_map.min()+1e-8)
        except Exception: ig_map = np.zeros((IMAGE_SIZE,IMAGE_SIZE))

        # EigenCAM
        try:
            ec = eigen_cam(model, inp, layer or "")
            if ec is not None:
                ec = np.array(PILImage.fromarray((ec*255).astype(np.uint8)).resize(
                    (IMAGE_SIZE,IMAGE_SIZE), PILImage.BILINEAR))/255.0
            else: ec = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
        except Exception: ec = np.zeros((IMAGE_SIZE,IMAGE_SIZE))

        maps = [orig, sal, cam_gc, cam_pp, ig_map, ec]
        for c, (ax, m, col) in enumerate(zip(axes[row], maps, cols)):
            if c == 0:
                ax.imshow(m)
            else:
                ax.imshow(orig); ax.imshow(m, cmap="jet", alpha=0.5)
            ax.set_title(col if row==0 else "", fontsize=12)
            ax.axis("off")
        axes[row,0].set_ylabel(CLASS_NAMES[label], fontsize=12, rotation=0, labelpad=60)

    plt.suptitle(f"Attribution Method Comparison — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"xai_compare_{model_name}.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print(f"  [Attribution Compare] saved → xai_compare_{model_name}.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑭ ALL-PIPELINE PLOTS — leaderboard, ablation, per-class deep-dive
# ════════════════════════════════════════════════════════════════════════════
def _normalise_gnas_result(r: Dict) -> Dict:
    """
    GNAS JSONs use different key names from Baseline/QEFS JSONs.
    Map them to the standard names so all downstream plot functions work.

    Key mapping:
      test_accuracy    → test_acc
      test_class_accs  → per_class_acc
      test_f1_macro    → f1_macro
      test_f1_weighted → f1_weighted
      bootstrap_acc_lo → ci_lower
      bootstrap_acc_hi → ci_upper
    """
    out = dict(r)   # shallow copy — don't mutate the original
    mapping = {
        "test_accuracy":    "test_acc",
        "test_class_accs":  "per_class_acc",
        "test_f1_macro":    "f1_macro",
        "test_f1_weighted": "f1_weighted",
        "bootstrap_acc_lo": "ci_lower",
        "bootstrap_acc_hi": "ci_upper",
    }
    for src, dst in mapping.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    return out


def load_all_pipeline_results() -> Dict[str, Dict]:
    """Load all JSON results from scripts 03–06."""
    all_r = {}
    # Baseline
    for mn in MODEL_NAMES:
        r = load_results(mn)
        if r: all_r[f"Baseline/{mn}"] = r
    # QEFS
    for mn in MODEL_NAMES:
        r = load_results(f"qefs_{mn}")
        if r: all_r[f"QEFS/{mn}"] = r
    # GNAS — normalise key names to match Baseline/QEFS format
    for mn in MODEL_NAMES:
        r = load_results(f"gnas_{mn}")
        if r: all_r[f"GNAS/{mn}"] = _normalise_gnas_result(r)
    # CL (best backbone per method)
    CL_METHODS = ["gdumb","derpp","x_der","a_gem","er_ace"]
    for m in CL_METHODS:
        r = load_results(f"cl_{CL_MODEL_NAME}_{m}")
        if r: all_r[f"CL/{m}"] = r
    return all_r


def plot_grand_leaderboard(all_r: Dict, save_dir=PLOTS_DIR):
    names, test_accs, ci_lo, ci_hi, f1s = [], [], [], [], []
    for k, v in all_r.items():
        names.append(k.replace("/","_"))
        test_accs.append(v.get("test_acc", v.get("final_test_acc", 0)))
        ci_lo.append(v.get("ci_lower", 0))
        ci_hi.append(v.get("ci_upper", 0))
        f1s.append(v.get("f1_macro", v.get("f1_macro", 0)))

    order = np.argsort(test_accs)[::-1]
    names     = [names[i]     for i in order]
    test_accs = [test_accs[i] for i in order]
    ci_lo     = [ci_lo[i]     for i in order]
    ci_hi     = [ci_hi[i]     for i in order]
    f1s       = [f1s[i]       for i in order]

    # color by technique group
    color_map = {"Baseline":"#3498db","QEFS":"#2ecc71","GNAS":"#e74c3c","CL":"#f39c12"}
    colors = [color_map.get(n.split("_")[0], "#95a5a6") for n in names]

    fig, ax = plt.subplots(figsize=(16, max(6, len(names)//2)))
    y = np.arange(len(names))
    err_lo = np.clip(np.array(test_accs) - np.array(ci_lo), 0, None)
    err_hi = np.clip(np.array(ci_hi) - np.array(test_accs), 0, None)
    xerr = [err_lo, err_hi]
    bars = ax.barh(y, test_accs, xerr=xerr, error_kw=dict(ecolor="black", capsize=3, lw=1),
                   color=colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=12)
    ax.set(title="Grand Accuracy Leaderboard — All Models & Techniques",
           xlabel="Test Accuracy")
    ax.set_xlim(0.5, 1.05)
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{acc:.4f}", va="center", fontsize=11)
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, fontsize=12, loc="lower right")
    ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "grand_leaderboard.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [Leaderboard] saved → grand_leaderboard.png")


def plot_ablation_waterfall(all_r: Dict, save_dir=PLOTS_DIR):
    """Show stepwise accuracy gain: Baseline → QEFS → GNAS → CL."""
    stages = ["Baseline", "QEFS", "GNAS", "CL"]
    best   = {}
    for stage in stages:
        vals = [v.get("test_acc", v.get("final_test_acc", 0))
                for k, v in all_r.items() if k.startswith(stage)]
        best[stage] = max(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    start   = 0.0
    colors  = ["#3498db","#2ecc71","#e74c3c","#f39c12"]
    x       = np.arange(len(stages))
    bars    = []
    for i, (stage, color) in enumerate(zip(stages, colors)):
        val = best[stage]
        bar = ax.bar(i, val - start, bottom=start, color=color, alpha=0.85, width=0.5)
        ax.text(i, val + 0.002, f"{val:.4f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
        bars.append(bar)
        start = val
    ax.set(title="Accuracy Ablation Waterfall — Technique Progression",
           ylabel="Test Accuracy", ylim=(0.7, 1.05))
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_waterfall.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [Ablation] saved → ablation_waterfall.png")


def plot_per_class_across_techniques(all_r: Dict, save_dir=PLOTS_DIR):
    """Heatmap: techniques × classes for per-class accuracy."""
    rows, mats = [], []
    for key, v in all_r.items():
        pca = v.get("per_class_acc", {})
        if not pca: continue
        rows.append(key.replace("/","_"))
        mats.append([pca.get(cn, 0) for cn in CLASS_NAMES])
    if not mats: return
    mat = np.array(mats)
    fig, ax = plt.subplots(figsize=(14, max(4, len(rows)//2 + 2)))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=rows,
                vmin=0.7, vmax=1.0, ax=ax, linewidths=0.3)
    ax.set(title="Per-Class Accuracy Across All Techniques",
           xlabel="Class", ylabel="Model/Method")
    plt.xticks(rotation=40, ha="right"); plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_all_techniques.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [Per-Class Heatmap] saved → per_class_all_techniques.png")


def plot_f1_comparison(all_r: Dict, save_dir=PLOTS_DIR):
    names = list(all_r.keys())
    f1m   = [all_r[n].get("f1_macro",0) for n in names]
    f1w   = [all_r[n].get("f1_weighted",0) for n in names]
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(12, len(names)*0.6), 6))
    ax.bar(x - w/2, f1m, w, label="F1 Macro",    color="#3498db", alpha=0.85)
    ax.bar(x + w/2, f1w, w, label="F1 Weighted", color="#e74c3c", alpha=0.85)
    ax.set(title="F1 Score Comparison — All Models",
           ylabel="F1 Score", ylim=(0.7, 1.01))
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("/","_") for n in names], rotation=45, ha="right", fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "f1_comparison_all.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [F1 Compare] saved → f1_comparison_all.png")


def _inference_ms(v: Dict) -> float:
    """Inference time per image in MILLISECONDS, normalised across result files:
      • baseline / QEFS / CL store `inference_time_per_img_s` (seconds) → ×1000
      • GNAS stores `test_inference_ms_per_sample` (already ms)         → ×1
    Reading the seconds key for GNAS (where it is absent) silently returned 0,
    which is why GNAS inference bars were drawn at exactly 0 ms.
    """
    if "inference_time_per_img_s" in v:
        return float(v["inference_time_per_img_s"]) * 1000.0
    if "inference_time_per_img" in v:
        return float(v["inference_time_per_img"]) * 1000.0
    for k in ("test_inference_ms_per_sample", "val_inference_ms_per_sample",
              "train_inference_ms_per_sample"):
        if k in v:
            return float(v[k])
    return 0.0


def _baseline_val(all_r: Dict, model_name: str, key: str, default: float = 0.0) -> float:
    """Return a value from the corresponding Baseline result dict.
    Used to enrich QEFS totals:
      - QEFS training_time_s only records the HFSEA loop (~2-4 s), not backbone training.
      - QEFS inference_time_per_img_s only records the feature-mask dot product (~1-2 µs),
        not the backbone CNN forward pass.
    Adding the baseline values gives the true wall-clock cost of deploying QEFS.
    """
    baseline_key = f"Baseline/{model_name}"
    if baseline_key in all_r:
        return float(all_r[baseline_key].get(key, default))
    return default


def plot_timing_memory(all_r: Dict, save_dir=PLOTS_DIR):
    names, inf_times, train_times = [], [], []
    for k, v in all_r.items():
        names.append(k.replace("/","_"))

        # QEFS inference_time_per_img_s only times the feature-mask dot product (~1-2 µs).
        # Add backbone inference time so the bar reflects full pipeline inference cost.
        inf_ms = _inference_ms(v)
        if k.startswith("QEFS/"):
            model_name = k.split("/", 1)[1]
            backbone_inf_s = _baseline_val(all_r, model_name, "inference_time_per_img_s")
            inf_ms += backbone_inf_s * 1000.0
        inf_times.append(inf_ms)

        # QEFS training_time_s only captures the HFSEA loop (~2-4 s). Add the backbone
        # (baseline) training time so the bar shows true wall-clock training cost.
        raw_train_s = v.get("training_time_s", v.get("total_training_time_s", 0))
        if k.startswith("QEFS/"):
            model_name = k.split("/", 1)[1]
            raw_train_s += _baseline_val(all_r, model_name, "training_time_s")

        train_times.append(raw_train_s / 60)   # min
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    order = np.argsort(inf_times)
    ax1.barh([names[i] for i in order], [inf_times[i] for i in order], color="#27ae60", alpha=0.85)
    ax1.set(title="Inference Time per Image (ms)", xlabel="ms")
    ax1.grid(axis="x", alpha=0.3)
    order2 = np.argsort(train_times)[::-1]
    ax2.barh([names[i] for i in order2], [train_times[i] for i in order2], color="#8e44ad", alpha=0.85)
    ax2.set(title="Training Time (minutes)", xlabel="min")
    ax2.grid(axis="x", alpha=0.3)
    plt.suptitle("Efficiency Analysis — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "timing_memory_all.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [Timing] saved → timing_memory_all.png")


def plot_cv_comparison(all_r: Dict, save_dir=PLOTS_DIR):
    names, means, stds = [], [], []
    for k, v in all_r.items():
        m = v.get("cv_acc_mean", None)
        if m is None: continue
        names.append(k.replace("/","_")); means.append(m); stds.append(v.get("cv_acc_std",0))
    if not names: return
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    y = np.arange(len(names))
    ax.barh([names[i] for i in order],
            [means[i] for i in order],
            xerr=[stds[i] for i in order],
            error_kw=dict(capsize=4, ecolor="black"),
            color="#2980b9", alpha=0.85)
    ax.set(title="Cross-Validation Accuracy (5-Fold) — All Baseline Models",
           xlabel="CV Accuracy"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cv_comparison_all.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [CV Compare] saved → cv_comparison_all.png")


def plot_ci_comparison(all_r: Dict, save_dir=PLOTS_DIR):
    names, test_accs, lo, hi = [], [], [], []
    for k, v in all_r.items():
        acc = v.get("test_acc", v.get("final_test_acc", None))
        if acc is None: continue
        names.append(k.replace("/","_")); test_accs.append(acc)
        lo.append(v.get("ci_lower", acc - 0.005))
        hi.append(v.get("ci_upper", acc + 0.005))
    order = np.argsort(test_accs)[::-1]
    fig, ax = plt.subplots(figsize=(14, max(5, len(names)//2)))
    y = np.arange(len(order))
    ax.errorbar(
        [test_accs[i] for i in order],
        y,
        xerr=[[test_accs[i]-lo[i] for i in order],
              [hi[i]-test_accs[i] for i in order]],
        fmt="o", capsize=4, color="#2c3e50", ecolor="#e74c3c", ms=6)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=12)
    ax.set(title="Test Accuracy with 95% Confidence Intervals",
           xlabel="Accuracy")
    ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ci_comparison_all.png"), dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [CI Compare] saved → ci_comparison_all.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑮ FAIRNESS REPORT (JSON + summary table image)
# ════════════════════════════════════════════════════════════════════════════
def save_fairness_report(all_metrics: Dict[str, Dict]):
    path = os.path.join(RESULTS_DIR, "fairness_report.json")
    out  = {}
    for k, v in all_metrics.items():
        out[k] = {kk: (vv.tolist() if hasattr(vv,"tolist") else vv)
                  for kk, vv in v.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [Fairness] report saved → {path}")

    # Summary table image
    scalars = ["dir","spd","eod","eqodds","theil","ece"]
    labels_ = ["Disparate Impact↑","Stat Parity Diff↓",
               "Equal Opp Diff↓","Eq Odds Diff↓","Theil Index↓","ECE↓"]
    rows    = [[k] + [f"{all_metrics[k].get(s,0):.4f}" for s in scalars]
               for k in all_metrics]
    fig, ax = plt.subplots(figsize=(16, max(4, len(rows)//2 + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Model"]+labels_,
                   cellLoc="center", loc="center",
                   colColours=["#2c3e50"]*7)
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.2, 1.5)
    for (r,c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if r%2==0 else "white")
    plt.suptitle("Fairness Summary — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FAIR_DIR, "fair_summary_table.png"),
                dpi=250, bbox_inches='tight', pad_inches=0.15); plt.close()
    print("  [Fairness] summary table saved → fair_summary_table.png")


# ════════════════════════════════════════════════════════════════════════════
# ⑯ MAIN ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print(f"\n{'─'*60}")
    print(" Loading test dataset …")
    print(f"{'─'*60}")
    print(f"  Test samples : {len(TEST_DS)}")

    # ── Load all pipeline results ───────────────────────────────────────────
    all_r = load_all_pipeline_results()
    print(f"  Loaded {len(all_r)} result sets from scripts 03–06")

    # ── ALL-PIPELINE PLOTS ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(" Generating all-pipeline comparison plots …")
    print(f"{'─'*60}")
    plot_grand_leaderboard(all_r)
    plot_ablation_waterfall(all_r)
    plot_per_class_across_techniques(all_r)
    plot_f1_comparison(all_r)
    plot_timing_memory(all_r)
    plot_cv_comparison(all_r)
    plot_ci_comparison(all_r)

    # Re-generate missing per-model plots from saved results
    for key, v in all_r.items():
        mn  = v.get("model_name", "model")
        tag = v.get("tag", key.replace("/","_"))
        short = key.replace("/","_")
        if "history" in v and v["history"]:
            h_path = os.path.join(PLOTS_DIR, f"curves_{short}.png")
            if not os.path.exists(h_path):
                plot_training_curves(v["history"], short, tag=tag)
        if "confusion_matrix" in v:
            cm_path = os.path.join(PLOTS_DIR, f"cm_{short}.png")
            if not os.path.exists(cm_path):
                plot_confusion_matrix(np.array(v["confusion_matrix"]), short, tag=tag)
        if "per_class_acc" in v:
            ca_path = os.path.join(PLOTS_DIR, f"classacc_{short}.png")
            if not os.path.exists(ca_path):
                plot_class_accuracies(v["per_class_acc"], short, tag=tag)

    # ── FAIRNESS ANALYSIS ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(" Fairness Analysis …")
    print(f"{'─'*60}")

    all_fair_metrics = {}

    for model_name in MODEL_NAMES:
        ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {model_name} — checkpoint not found"); continue
        print(f"\n  ── {model_name} ──────────────────────────────")
        model = load_baseline_model(model_name)
        if model is None: continue

        preds, labels, probs = infer_model(model, TEST_LOADER)
        name = f"Baseline_{model_name}"
        fm   = compute_fairness_metrics(preds, labels, probs)
        all_fair_metrics[name] = fm

        plot_equal_opportunity(fm, name)
        plot_equalized_odds(fm, name)
        plot_demographic_parity(fm, name)
        plot_error_rate_parity(fm, name)
        plot_fairness_radar(fm, name)
        plot_calibration_per_class(labels, probs, name)
        plot_roc_per_class(labels, probs, name)
        plot_pr_per_class(labels, probs, name)
        plot_confidence_histogram(probs, preds, labels, name)

        del model; torch.cuda.empty_cache()

    # Fairness for QEFS models
    import importlib.util as _ilu
    _qefs_spec = _ilu.spec_from_file_location("qefs_m", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "04_qefs.py"))
    _qefs_mod = _ilu.module_from_spec(_qefs_spec)
    _qefs_spec.loader.exec_module(_qefs_mod)

    for model_name in MODEL_NAMES:
        qcnn_path = os.path.join(MODELS_DIR, f"qcnn_{model_name}_best.pth")
        if not os.path.exists(qcnn_path): continue
        print(f"\n  ── QEFS/{model_name} ──────────────────────────")
        r = load_results(f"qefs_{model_name}")
        if not (r and "preds" in r and "labels" in r): continue
        preds  = np.array(r["preds"]); labels = np.array(r["labels"])
        try:
            # Build QCNNHead the same way 04_qefs.py does:
            #   full CNN → get_feature_extractor → QCNNHead
            # qcnn_{model}_best.pth is a raw state_dict (no "model_state" wrapper)
            _raw_cnn   = get_model(model_name).to(DEVICE)
            backbone   = get_feature_extractor(_raw_cnn, model_name).to(DEVICE)
            feat_dim   = _qefs_mod.get_backbone_dim(model_name)
            qm         = _qefs_mod.QCNNHead(backbone, feat_dim, QEFS_FEATURE_DIM).to(DEVICE)
            state      = torch.load(qcnn_path, map_location=DEVICE, weights_only=False)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            qm.load_state_dict(state); qm.eval()
            _, _, probs = infer_model(qm, TEST_LOADER)
            del qm; torch.cuda.empty_cache()
            print(f"    [QEFS probs] loaded OK")
        except Exception as e:
            print(f"    [QEFS probs] failed: {e}; using hard-prediction probs")
            probs = np.eye(NUM_CLASSES)[preds]
        name = f"QEFS_{model_name}"
        fm   = compute_fairness_metrics(preds, labels, probs)
        all_fair_metrics[name] = fm
        plot_equal_opportunity(fm, name)
        plot_fairness_radar(fm, name)
        plot_roc_per_class(labels, probs, name)
        plot_pr_per_class(labels, probs, name)

    # Fairness for best CL model
    CL_METHODS_SHORT = ["gdumb","derpp","x_der","a_gem","er_ace"]
    for cl_m in CL_METHODS_SHORT:
        r = load_results(f"cl_{CL_MODEL_NAME}_{cl_m}")
        if not r: continue
        preds  = np.array(r.get("preds", []))
        labels = np.array(r.get("labels", []))
        if len(preds) == 0: continue
        probs  = np.eye(NUM_CLASSES)[preds]  # hard probs (CL doesn't store soft probs)
        name   = f"CL_{cl_m}"
        fm     = compute_fairness_metrics(preds, labels, probs)
        all_fair_metrics[name] = fm
        plot_equal_opportunity(fm, name)
        plot_fairness_radar(fm, name)

    # Cross-model fairness summaries
    if all_fair_metrics:
        plot_theil_index_comparison(all_fair_metrics)
        plot_tpr_heatmap_all_models(all_fair_metrics)
        plot_ece_comparison(all_fair_metrics)
        save_fairness_report(all_fair_metrics)

    # ── XAI ANALYSIS ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(" XAI Analysis …")
    print(f"{'─'*60}")

    # Run XAI on best baseline model (mobilenet_v3_large) + all others
    XAI_MODELS = MODEL_NAMES  # run on all; skip if checkpoint missing
    for model_name in XAI_MODELS:
        ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [skip XAI] {model_name}"); continue
        print(f"\n  ── XAI: {model_name} ──────────────────────────")
        model = load_baseline_model(model_name)
        if model is None: continue

        # Integrated Gradients
        plot_integrated_gradients(model, model_name, TEST_DS, n_samples=4)
        # Vanilla Saliency
        plot_saliency_maps(model, model_name, TEST_DS, n_samples=4)
        # Grad-CAM variants
        plot_grad_cam_variants(model, model_name, TEST_DS, n_samples=4)
        # EigenCAM
        plot_eigen_cam(model, model_name, TEST_DS, n_samples=4)
        # Occlusion Sensitivity (expensive — fewer samples)
        plot_occlusion_sensitivity(model, model_name, TEST_DS, n_samples=3)
        # SHAP
        plot_shap_suite(model, model_name, TEST_DS)
        # LIME
        plot_lime(model, model_name, TEST_DS, n_samples=3)
        # t-SNE + UMAP
        plot_tsne_umap(model, model_name)
        # Attribution comparison
        plot_attribution_comparison(model, model_name, TEST_DS, n_samples=2)
    print("\n  Fairness and explainability analysis complete.")
    print(f"  Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
