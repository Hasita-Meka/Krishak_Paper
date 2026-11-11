"""
15_calibration.py — Calibration analysis (R1: "ECE = 0.064 alone is insufficient")
=================================================================================
For each backbone and stage (Baseline CNN, QEFS/QCNN, GNAS):
  • ECE (15 equal-width bins), MCE, Brier score, NLL, accuracy
  • class-wise ECE (one-vs-rest, per class and mean)
  • 95% bootstrap CIs (1,000 resamples of the test set) for ECE, Brier, accuracy
  • reliability diagram before / after temperature scaling
  • post-hoc temperature scaling: T fitted on the VALIDATION set (NLL), applied to test

Out: results/revision/calibration/<stage>_<model>.json, plots/revision/calibration_<stage>_<model>.png,
     results/revision/calibration/summary.csv
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa

NB = 15
PLOT_DIR = os.path.join(PLOTS_DIR, "revision"); os.makedirs(PLOT_DIR, exist_ok=True)


def ece_mce(p, y, nb=NB):
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, nb + 1); e = m = 0.0; bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            gap = abs(acc[s].mean() - conf[s].mean()); e += s.mean() * gap; m = max(m, gap)
            bins.append({"lo": lo, "hi": hi, "n": int(s.sum()), "acc": float(acc[s].mean()), "conf": float(conf[s].mean())})
        else:
            bins.append({"lo": lo, "hi": hi, "n": 0, "acc": None, "conf": None})
    return float(e), float(m), bins


def classwise_ece(p, y, nb=NB):
    out = []
    for c in range(p.shape[1]):
        pc, yc = p[:, c], (y == c).astype(float)
        edges = np.linspace(0, 1, nb + 1); e = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (pc > lo) & (pc <= hi) if lo > 0 else (pc >= lo) & (pc <= hi)
            if s.any(): e += s.mean() * abs(yc[s].mean() - pc[s].mean())
        out.append(float(e))
    return out


def metrics(p, y):
    e, m, bins = ece_mce(p, y)
    oh = np.eye(p.shape[1])[y]
    return {"acc": float((p.argmax(1) == y).mean()), "ece": e, "mce": m,
            "brier": float(((p - oh) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            "classwise_ece": dict(zip(CLASS_NAMES, classwise_ece(p, y))), "bins": bins}


def boot_ci(p, y, n=1000, seed=0):
    rng = np.random.RandomState(seed); E, B, A = [], [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y)); e, _, _ = ece_mce(p[i], y[i])
        E.append(e); A.append((p[i].argmax(1) == y[i]).mean())
        B.append(((p[i] - np.eye(p.shape[1])[y[i]]) ** 2).sum(1).mean())
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"ece_ci95": q(E), "brier_ci95": q(B), "acc_ci95": q(A)}


def fit_T(logits, y):
    nll = lambda T: -np.log(np.clip(softmax(logits, T)[np.arange(len(y)), y], 1e-12, 1)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def reliability(ax, bins, title):
    c = [(b["lo"] + b["hi"]) / 2 for b in bins]
    a = [b["acc"] if b["acc"] is not None else 0 for b in bins]
    ax.bar(c, a, width=1 / NB, edgecolor="k", alpha=0.8, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="accuracy", title=title); ax.legend(loc="upper left")


if __name__ == "__main__":
    tr, va, te, dirs = load_splits()
    va_ld, te_ld = loader(va, dirs), loader(te, dirs)
    rows = []
    for mn in MODEL_NAMES:
        for st, fn in (("baseline", load_baseline), ("qefs", load_qcnn), ("gnas", load_gnas)):
            model = fn(mn)
            if model is None:
                print(f"  [skip] {st}/{mn}"); continue
            lv, yv = predict(model, va_ld); lt, yt = predict(model, te_ld)
            T = fit_T(lv, yv)
            before, after = metrics(softmax(lt), yt), metrics(softmax(lt, T), yt)
            r = {"model": mn, "stage": st, "temperature": T, "test_before": before, "test_after_ts": after,
                 "bootstrap_before": boot_ci(softmax(lt), yt), "bootstrap_after": boot_ci(softmax(lt, T), yt)}
            dump(r, f"{st}_{mn}.json", "calibration")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
            reliability(ax[0], before["bins"], f"{st}/{mn} — before (ECE={before['ece']:.3f})")
            reliability(ax[1], after["bins"], f"after temperature scaling T={T:.2f} (ECE={after['ece']:.3f})")
            plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, f"calibration_{st}_{mn}.png"), dpi=200); plt.close()
            rows.append({"model": mn, "stage": st, "acc": before["acc"], "ece": before["ece"],
                         "ece_lo": r["bootstrap_before"]["ece_ci95"][0], "ece_hi": r["bootstrap_before"]["ece_ci95"][1],
                         "mce": before["mce"], "brier": before["brier"], "nll": before["nll"],
                         "classwise_ece_mean": float(np.mean(list(before["classwise_ece"].values()))),
                         "T": T, "ece_after_ts": after["ece"], "nll_after_ts": after["nll"]})
            print(f"   {st:<8} {mn:<20} acc={before['acc']:.4f} ECE={before['ece']:.4f} → {after['ece']:.4f} (T={T:.2f})", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(rev_dir("calibration"), "summary.csv"), index=False)
