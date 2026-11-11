"""
16_xai_faithfulness.py — Quantitative explanation faithfulness + corrected SHAP (R1, R5 #8)
==========================================================================================
Part A — faithfulness (Petsiuk et al., 2018, RISE deletion/insertion):
  For N stratified test images and the predicted class, pixels are ranked by the
  attribution map; the class probability is tracked while the top-ranked pixels are
  progressively DELETED (replaced by a blurred copy; lower AUC = more faithful) or
  INSERTED into the blurred image (higher AUC = more faithful). Methods: Grad-CAM,
  vanilla gradient saliency, Integrated Gradients, and a RANDOM map as the reference.
  Two variants are reported: PIXEL-level (the original; random map = scattered pixels) and
  PATCH-level (`deletion_patch` / `insertion_patch`): the map is average-pooled to a
  GRID x GRID grid of cells and whole cells are deleted/inserted. The patch-level variant is the
  fairer one for coarse maps such as Grad-CAM (7x7 resolution), because scattered random pixels
  damage a classifier far more than deleting one contiguous region.
Part B — stability: Spearman correlation between the map of x and the map of
  x + N(0, 0.02²) noise, and of the horizontally flipped image (flipped back).
Part C — SHAP in the space the GNAS head actually uses (replaces Fig. 8, which showed
  "top 60 of 960" channels of a model/layer that is not the stated pipeline): mean |SHAP|
  per class over the HFSEA-selected QIL features that feed the GNAS head.

Env: KRISHAK_MODELS (default mobilenet_v3_large), KRISHAK_XAI_N (default 200),
     KRISHAK_XAI_STAGES (default baseline,gnas), KRISHAK_XAI_GRID (patch grid, default 14).
Out: results/revision/xai/<stage>_<model>.json, plots/revision/shap_gnas_features_<model>.png
"""
import os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa

MODELS = os.environ.get("KRISHAK_MODELS", "mobilenet_v3_large").split(",")
N = 6 if SMOKE else int(os.environ.get("KRISHAK_XAI_N", 200))
STAGES = os.environ.get("KRISHAK_XAI_STAGES", "baseline,gnas").split(",")
STEPS = 16
GRID = int(os.environ.get("KRISHAK_XAI_GRID", 14))
LAYER = {"densenet161": "features.denseblock4", "googlenet": "inception5b",
         "mobilenet_v3_large": "features.16", "resnet50": "layer4", "shufflenet_v2": "conv5"}
PLOT_DIR = os.path.join(PLOTS_DIR, "revision"); os.makedirs(PLOT_DIR, exist_ok=True)


def find_layer(model, name):
    for n, m in model.named_modules():
        if n == name or n.endswith("." + name):
            return m
    raise KeyError(name)


def gradcam(model, layer, x, c):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    x = x.clone().requires_grad_(True); model.zero_grad()
    model(x)[0, c].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    return F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()


def saliency(model, x, c):
    x = x.clone().requires_grad_(True); model.zero_grad()
    model(x)[0, c].backward()
    return x.grad.abs().amax(1)[0].detach()


def integrated_gradients(model, x, c, steps=32):
    base = torch.zeros_like(x); total = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (base + a * (x - base)).requires_grad_(True); model.zero_grad()
        model(xi)[0, c].backward(); total += xi.grad
    return ((x - base) * total / steps).abs().sum(1)[0].detach()


def blur(x):
    k = 31; g = torch.exp(-((torch.arange(k) - k // 2).float() ** 2) / (2 * 8.0 ** 2)); g = (g / g.sum()).to(x.device)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


@torch.no_grad()
def del_ins(model, x, c, amap, grid=None):
    """Deletion / insertion AUC. grid=None: pixel-level ranking; grid=G: the map is average-pooled
    to G x G cells and whole cells are removed/added (nearest-neighbour upsampled mask)."""
    if grid:
        cell = F.adaptive_avg_pool2d(amap[None, None].float(), (grid, grid))[0, 0]
        order = torch.argsort(cell.flatten(), descending=True); HW = cell.numel(); shp = tuple(cell.shape)
    else:
        order = torch.argsort(amap.flatten(), descending=True); HW = amap.numel(); shp = tuple(amap.shape)
    xb = blur(x); dele, ins = [], []
    for s in range(STEPS + 1):
        m = torch.zeros(HW, device=x.device); m[order[: int(HW * s / STEPS)]] = 1
        m = m.view(1, 1, *shp)
        if grid:
            m = F.interpolate(m, size=tuple(x.shape[-2:]), mode="nearest")
        dele.append(x * (1 - m) + xb * m); ins.append(xb * (1 - m) + x * m)
    pd_ = F.softmax(model(torch.cat(dele)), 1)[:, c].cpu().numpy()
    pi_ = F.softmax(model(torch.cat(ins)), 1)[:, c].cpu().numpy()
    auc = lambda v: float((getattr(np, "trapezoid", None) or np.trapz)(v, dx=1 / STEPS))
    return auc(pd_), auc(pi_)


def stability(fn, x, c):
    a = fn(x, c).flatten().cpu().numpy()
    b = fn(x + 0.02 * torch.randn_like(x), c).flatten().cpu().numpy()
    f = fn(torch.flip(x, [3]), c); f = torch.flip(f, [1]).flatten().cpu().numpy()
    return float(spearmanr(a, b)[0]), float(spearmanr(a, f)[0])


def shap_features(mn, te, dirs):
    try:
        import shap
    except ImportError:
        print("  [SHAP] not installed"); return None
    g = load_gnas(mn)
    if g is None: return None
    ld = loader(te, dirs)
    Z, Y = [], []
    with torch.no_grad():
        for x, y in ld:
            Z.append(g.backbone(x.to(DEVICE))); Y.append(y)
    Z = torch.cat(Z); Y = torch.cat(Y).numpy()
    head = g.head.eval()
    bg = Z[torch.randperm(len(Z))[: min(100, len(Z))]]
    sv = shap.GradientExplainer(head, bg).shap_values(Z)
    sv = np.stack(sv, 0) if isinstance(sv, list) else np.moveaxis(np.asarray(sv), -1, 0)  # (K, N, D)
    imp = np.stack([np.abs(sv[k][Y == k]).mean(0) if (Y == k).any() else np.zeros(sv.shape[2])
                    for k in range(NUM_CLASSES)])                                            # (K, D)
    sel = g.backbone.sel_idx.cpu().numpy() if g.backbone.sel_idx is not None else np.arange(imp.shape[1])
    top = np.argsort(-imp.max(0))[:30]
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(imp[:, top] / (imp[:, top].max(1, keepdims=True) + 1e-12), aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels([c.replace("_", " ") for c in CLASS_NAMES])
    ax.set_xticks(range(len(top))); ax.set_xticklabels(sel[top], rotation=90, fontsize=7)
    ax.set_xlabel(f"QIL feature index (top 30 of the {len(sel)} HFSEA-selected features of 512)")
    ax.set_title(f"{mn} + GNAS head — mean |SHAP| per true class (row-normalised)")
    plt.colorbar(im, ax=ax, fraction=0.02); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"shap_gnas_features_{mn}.png"), dpi=250); plt.close()
    return {"n_selected": int(len(sel)), "q_dim": QEFS_FEATURE_DIM, "explained_model": f"{mn} GNAS head",
            "input_space": "HFSEA-selected QIL features", "mean_abs_shap_per_class": imp.tolist(),
            "selected_feature_idx": sel.tolist()}


if __name__ == "__main__":
    tr, va, te, dirs = load_splits()
    sub = te.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), max(1, N // NUM_CLASSES)), random_state=0))
    if SMOKE: sub = sub.head(2)
    ds = MultiDirDataset(sub.reset_index(drop=True), dirs, get_val_transform())
    for mn in MODELS:
        for st in STAGES:
            model = load_baseline(mn) if st == "baseline" else load_gnas(mn)
            if model is None:
                print(f"  [skip] {st}/{mn}"); continue
            model.eval(); layer = find_layer(model, LAYER[mn])
            methods = {"gradcam": lambda x, c: gradcam(model, layer, x, c),
                       "saliency": lambda x, c: saliency(model, x, c),
                       "integrated_gradients": lambda x, c: integrated_gradients(model, x, c),
                       "random": lambda x, c: torch.rand(x.shape[-2:], device=x.device)}
            R = {k: {"deletion": [], "insertion": [], "deletion_patch": [], "insertion_patch": [],
                     "noise_rho": [], "flip_rho": []} for k in methods}
            for i in range(len(ds)):
                x = ds[i][0].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    c = int(model(x).argmax(1))
                for k, fn in methods.items():
                    amap = fn(x, c)
                    d, s = del_ins(model, x, c, amap)
                    dp, sp = del_ins(model, x, c, amap, grid=GRID)
                    R[k]["deletion"].append(d); R[k]["insertion"].append(s)
                    R[k]["deletion_patch"].append(dp); R[k]["insertion_patch"].append(sp)
                    if k != "random":
                        a, b = stability(fn, x, c); R[k]["noise_rho"].append(a); R[k]["flip_rho"].append(b)
            summ = {k: {m: (float(np.mean(v[m])) if v[m] else None) for m in v} for k, v in R.items()}
            out = {"model": mn, "stage": st, "n_images": len(ds), "steps": STEPS, "grid": GRID, "summary": summ,
                   "note": "deletion AUC lower = better; insertion AUC higher = better; random = reference. "
                           "*_patch = cell-level (GRID x GRID) variant, fairer for coarse maps."}
            if st == "gnas":
                out["shap_features"] = shap_features(mn, te, dirs)
            dump(out, f"{st}_{mn}.json", "xai")
            for k, v in summ.items():
                print(f"   {st}/{mn} {k:<22} del={v['deletion']:.3f} ins={v['insertion']:.3f} "
                      f"del_patch={v['deletion_patch']:.3f} ins_patch={v['insertion_patch']:.3f} "
                      f"noise_rho={v['noise_rho'] if v['noise_rho'] is None else round(v['noise_rho'],3)}", flush=True)
