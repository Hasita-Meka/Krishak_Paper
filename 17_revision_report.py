"""
17_revision_report.py — Collect every revision result into tables + statistics (CPU)
===================================================================================
Run last (after 03–16). Reads the main run (KRISHAK_RUN_TAG, default v2_leakfree)
plus the multi-seed runs (outputs/seed_43, seed_44) and nested folds
(outputs/nested_f1..f5) when present, and writes

  results/revision/tables/*.csv   one CSV per table
  results/revision/REVISION_RESULTS.md   all tables + statistics in one readable file

Statistics computed here:
  • forgetting–fairness correlation (R1, R4, R5 #7): forgetting = −BWT, fairness
    violation = 1 − DIR (DIR = min class TPR / max class TPR, as in 07); Pearson and
    Spearman r, n, p-value, 95% CI (Fisher z / bootstrap), WITH and WITHOUT A-GEM
  • equivalence (R1): paired differences Baseline vs QEFS / GNAS over nested outer
    folds (and seeds), 90% CI vs a pre-specified ±1.0 pp margin (TOST)
"""
import os, sys, json, glob, re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa
from config import _OUT_ROOT

MARGIN = float(os.environ.get("KRISHAK_EQUIV_MARGIN", 0.01))
TAB = rev_dir("tables")
MD = []


def J(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def md_table(df, title, note=""):
    MD.append(f"\n## {title}\n")
    if note: MD.append(note + "\n")
    if df is None or len(df) == 0:
        MD.append("_no results yet_\n"); return
    cols = list(df.columns)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    MD.append("| " + " | ".join(cols) + " |"); MD.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        MD.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    df.to_csv(os.path.join(TAB, re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".csv"), index=False)


def dir_of(preds, labels):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    tpr = np.array([cm[c, c] / max(cm[c].sum(), 1) for c in range(NUM_CLASSES)])
    return float(tpr.min() / max(tpr.max(), 1e-9))


def corr_block(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"subset": label, "n": n}
    r, p = stats.pearsonr(x, y); rs, ps = stats.spearmanr(x, y)
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(n - 3)
    rng = np.random.RandomState(0); bs = []
    for _ in range(2000):
        i = rng.randint(0, n, n)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0: bs.append(stats.spearmanr(x[i], y[i])[0])
    return {"subset": label, "n": n, "pearson_r": r, "pearson_p": p,
            "pearson_ci95": f"[{np.tanh(z - 1.96 * se):.3f}, {np.tanh(z + 1.96 * se):.3f}]",
            "spearman_rho": rs, "spearman_p": ps,
            "spearman_ci95_boot": f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"}


def tost(diffs, margin=MARGIN):
    d = np.asarray(diffs, float); n = len(d)
    if n < 2: return {"n": n}
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se) if se > 0 else (m, m)
    p1 = 1 - stats.t.cdf((m + margin) / se, n - 1) if se > 0 else 0.0
    p2 = stats.t.cdf((m - margin) / se, n - 1) if se > 0 else 0.0
    return {"n": n, "mean_diff": m, "ci90_lo": lo, "ci90_hi": hi, "margin": margin,
            "tost_p": max(p1, p2), "equivalent": bool(lo > -margin and hi < margin)}


R = RESULTS_DIR
# ── 1. main leak-free results ────────────────────────────────────────────────
rows = []
for m in ALL_MODEL_NAMES:
    b, q, g, gr = J(f"{R}/{m}.json"), J(f"{R}/qefs_{m}.json"), J(f"{R}/gnas_{m}.json"), J(f"{R}/gnasrand_{m}.json")
    if not b: continue
    rows.append({"model": m, "baseline_test_acc": b.get("test_acc"), "baseline_f1": b.get("f1_macro"),
                 "cv15_acc": b.get("cv_acc_mean"), "cv15_sd": b.get("cv_acc_std"),
                 "cv30_acc": (b.get("cv_e30") or {}).get("cv_acc_mean"), "cv60_acc": (b.get("cv_e60") or {}).get("cv_acc_mean"),
                 "qcnn_test_acc": (q or {}).get("qcnn_test_acc"), "qefs_test_acc": (q or {}).get("qefs_test_acc"),
                 "qefs_k": (q or {}).get("n_features_selected"), "hfsea_evals": (q or {}).get("hfsea_fitness_evals"),
                 "hfsea_search_min": ((q or {}).get("hfsea_search_time_s") or np.nan) / 60,
                 "gnas_test_acc": (g or {}).get("test_accuracy"), "gnas_f1": (g or {}).get("test_f1_macro"),
                 "gnas_boot_ci": f"[{(g or {}).get('bootstrap_acc_lo', np.nan):.4f}, {(g or {}).get('bootstrap_acc_hi', np.nan):.4f}]" if g else "",
                 "gnas_search_evals": (g or {}).get("search_candidate_evals"), "gnas_search_min": ((g or {}).get("search_time_s") or np.nan) / 60,
                 "random_search_test_acc": (gr or {}).get("test_accuracy"), "random_search_min": ((gr or {}).get("search_time_s") or np.nan) / 60})
md_table(pd.DataFrame(rows), "Main results on the leakage-safe split",
         "Test = 982 original photos never seen in training (no augmented copies). CV at 15/30/60 epochs per fold (R5 #2).")

# ── 2. nested CV ─────────────────────────────────────────────────────────────
ns = J(os.path.join(rev_dir(), "nested_cv_summary.json"))
md_table(pd.DataFrame(ns["summary"]).drop(columns=["fold_accs"], errors="ignore") if ns else None,
         "Nested cross-validation (outer folds)", "Every stage refitted inside each outer training fold (12_nested_summary.py).")

# ── 3. incremental ablation + imbalance ablation ────────────────────────────
abl = {os.path.basename(p)[:-5]: J(p) for p in glob.glob(os.path.join(rev_dir("data_ablation"), "*.json"))}
inc, imb = [], []
for k, r in sorted(abl.items()):
    row = {"model": r["model_name"], "condition": r["condition"], "seed": r["seed"], "n_train": r["n_train"],
           "imbalance_ratio": r["imbalance_ratio"], "test_acc": r["test_acc"], "macro_f1": r["test_f1_macro"],
           "train_min": r["train_time_s"] / 60, "peak_gpu_MB": r.get("peak_gpu_mem_MB")}
    (imb if r["condition"].startswith("clean") else inc).append(row)
    if r["condition"] in ("clean_none", "clean_aug_all3"): inc.append(row)
cl_main = {os.path.basename(p)[3:-5]: J(p) for p in glob.glob(f"{R}/cl_*.json")
           if not re.search(r"_(o\d+|s\d+|b\d+|f\w+|agemlegacy)$", os.path.basename(p)[:-5])}
for m in sorted({r["model"] for r in inc}):
    q, g = J(f"{R}/qefs_{m}.json"), J(f"{R}/gnas_{m}.json")
    if q: inc.append({"model": m, "condition": "+QEFS (QIL+HFSEA+MLP)", "test_acc": q.get("qefs_test_acc"),
                      "macro_f1": q.get("qefs_f1_macro"), "train_min": (q.get("hfsea_search_time_s") or 0) / 60})
    if g: inc.append({"model": m, "condition": "+GNAS head", "test_acc": g.get("test_accuracy"),
                      "macro_f1": g.get("test_f1_macro"), "train_min": ((g.get("search_time_s") or 0) + (g.get("training_time_s") or 0)) / 60})
    cls = [r for k, r in cl_main.items() if k.startswith(m + "_")]
    for r in cls:
        inc.append({"model": m, "condition": f"+CL {r['method']}", "test_acc": r.get("final_test_acc"),
                    "macro_f1": r.get("final_f1_macro"), "BWT": r.get("BWT"),
                    "train_min": (r.get("total_training_time_s") or 0) / 60,
                    "memory_MB": (r.get("memory_report") or {}).get("memory_MB")})
md_table(pd.DataFrame(inc), "Incremental ablation (R5 #12)", "raw → cleaned → cleaned+balanced (+sampler+weighted loss) → +QEFS → +GNAS → +CL, same leak-free test set.")
md_table(pd.DataFrame(imb), "Imbalance-correction ablation (R1)")

# ── 4. feature selection ─────────────────────────────────────────────────────
fs = []
for p in glob.glob(os.path.join(rev_dir("fs"), "*_baselines.json")):
    d = J(p)
    for meth, r in d["results"].items():
        fs.append({"model": d["model"], "method": meth, "k": r.get("k"), "test_acc": r.get("test_acc"),
                   "test_acc_sd": r.get("test_acc_sd"), "macro_f1": r.get("test_f1_macro"), "select_s": r.get("select_s")})
md_table(pd.DataFrame(fs), "Feature-selection baselines at equal k (R1, R4, R5 #13)")
lam = [dict(model=J(p)["model"], **{k: v for k, v in r.items()}) for p in glob.glob(os.path.join(rev_dir("fs"), "*_lambda.json")) for r in J(p)["rows"]]
md_table(pd.DataFrame(lam), "HFSEA sparsity weight λ sweep (R4)")
it = [dict(model=J(p)["model"], **r) for p in glob.glob(os.path.join(rev_dir("fs"), "*_iters.json")) for r in J(p)["checkpoints"]]
md_table(pd.DataFrame(it), "HFSEA iterations (R3 #1)")
sd = []
for p in glob.glob(os.path.join(rev_dir("fs"), "*_seeds.json")):
    d = J(p)
    for r in d["rows"]:
        sd.append({"model": d["model"], "seed": r["seed"], "n_selected": r["n_selected"], "test_acc": r["test_acc"],
                   "search_min": r["search_s"] / 60, "mask_jaccard_mean": d["mask_jaccard_mean"]})
md_table(pd.DataFrame(sd), "HFSEA run-to-run variability (R1, R5 #15)")

# ── 5. QIL ablation ──────────────────────────────────────────────────────────
qa = [{"model": J(p)["model"], "variant": v, "test_acc_mean": r["test_acc_mean"], "test_acc_sd": r["test_acc_sd"],
       "f1_mean": r["test_f1_mean"], "proj_params": r["runs"][0]["n_params_projection"]}
      for p in glob.glob(os.path.join(rev_dir("qil_ablation"), "*.json")) for v, r in J(p)["variants"].items()]
md_table(pd.DataFrame(qa), "QIL component ablation vs ordinary projections (R1, R4, R3 #2)")

# ── 6. continual learning robustness ────────────────────────────────────────
clv = []
for p in glob.glob(f"{R}/cl_*.json"):
    r = J(p); k = os.path.basename(p)[3:-5]
    if not r or "AA" not in r: continue
    clv.append({"model": r.get("model_name"), "method": r["method"], "order": r.get("task_order_seed", 0),
                "seed": r.get("seed", 42), "buffer": r.get("buffer_size"), "freeze": r.get("freeze_mode", "1"),
                "agem_legacy": r.get("agem_legacy", False), "AA": r["AA"], "BWT": r["BWT"],
                "final_acc": r.get("final_test_acc"), "memory_MB": (r.get("memory_report") or {}).get("memory_MB"),
                "DIR": dir_of(r["preds"], r["labels"]) if r.get("preds") else np.nan})
CLV = pd.DataFrame(clv)
if len(CLV):
    main = CLV[(CLV.buffer == 2000) & (CLV.freeze.astype(str) == "1") & (~CLV.agem_legacy.astype(bool))]
    agg = main.groupby(["model", "method"]).agg(runs=("AA", "size"), AA_mean=("AA", "mean"), AA_sd=("AA", "std"),
                                                BWT_mean=("BWT", "mean"), BWT_sd=("BWT", "std")).reset_index()
    md_table(agg, "CL over task orders and seeds (R1, R5 #4)", "mean ± SD over all (order, seed) runs at buffer 2000, frozen backbone.")
    buf = CLV[(CLV.order == 0) & (CLV.seed == 42) & (CLV.freeze.astype(str) == "1") & (~CLV.agem_legacy.astype(bool))]
    md_table(buf.sort_values(["model", "method", "buffer"])[["model", "method", "buffer", "AA", "BWT", "memory_MB"]],
             "Replay-memory budget (R1, R5 #5)")
    fr = CLV[(CLV.order == 0) & (CLV.seed == 42) & (CLV.buffer == 2000)]
    md_table(fr.sort_values(["model", "method", "freeze"])[["model", "method", "freeze", "agem_legacy", "AA", "BWT", "final_acc"]],
             "Frozen vs trainable backbone; A-GEM fix (R5 #3, R4)")
    # forgetting–fairness correlation on the 25 main runs (5 backbones × 5 methods)
    base = CLV[(CLV.order == 0) & (CLV.seed == 42) & (CLV.buffer == 2000) & (CLV.freeze.astype(str) == "1") & (~CLV.agem_legacy.astype(bool))]
    cor = [corr_block(-base.BWT, 1 - base.DIR, "all main runs"),
           corr_block(-base[base.method != "A-GEM"].BWT, 1 - base[base.method != "A-GEM"].DIR, "excluding A-GEM"),
           corr_block(-main.BWT, 1 - main.DIR, "all orders/seeds"),
           corr_block(-main[main.method != "A-GEM"].BWT, 1 - main[main.method != "A-GEM"].DIR, "all orders/seeds, excl. A-GEM")]
    md_table(pd.DataFrame(cor), "Forgetting vs fairness violation (R1, R4, R5 #7)",
             "x = forgetting = −BWT (mean over first 4 tasks of acc drop); y = 1 − DIR, DIR = min class TPR / max class TPR on the final model.")

# ── 7. equivalence (TOST) ────────────────────────────────────────────────────
eq = []
fold_csv = os.path.join(rev_dir(), "nested_cv_folds.csv")
if os.path.exists(fold_csv):
    F = pd.read_csv(fold_csv)
    for m, g in F.groupby("model"):
        P = g.pivot_table(index="fold", columns="stage", values="acc")
        for st in ("QEFS", "GNAS"):
            if "Baseline" in P and st in P:
                d = (P[st] - P["Baseline"]).dropna()
                eq.append({"model": m, "comparison": f"{st} − Baseline", "unit": "outer fold", **tost(d.values)})
seed_rows = []
for tag in ["v2_leakfree", "seed_43", "seed_44"]:
    rd = os.path.join(_OUT_ROOT, tag, "results")
    for m in ALL_MODEL_NAMES:
        b, q, g = J(f"{rd}/{m}.json"), J(f"{rd}/qefs_{m}.json"), J(f"{rd}/gnas_{m}.json")
        if b: seed_rows.append({"tag": tag, "model": m, "Baseline": b.get("test_acc"),
                                "QEFS": (q or {}).get("qefs_test_acc"), "GNAS": (g or {}).get("test_accuracy")})
S = pd.DataFrame(seed_rows)
if len(S):
    md_table(S.groupby("model")[["Baseline", "QEFS", "GNAS"]].agg(["mean", "std", "count"]).reset_index().pipe(
        lambda d: d.set_axis(["_".join(c).strip("_") for c in d.columns], axis=1)), "Multi-seed test accuracy (R5 #15)")
    for m, g in S.groupby("model"):
        for st in ("QEFS", "GNAS"):
            d = (g[st] - g["Baseline"]).dropna()
            if len(d) >= 2: eq.append({"model": m, "comparison": f"{st} − Baseline", "unit": "seed", **tost(d.values)})
md_table(pd.DataFrame(eq), "Equivalence tests (R1)", f"Pre-specified margin ±{MARGIN*100:.1f} pp; equivalent if the 90% CI lies inside the margin (TOST).")

# ── 8. calibration, latency, XAI, Split-CIFAR ───────────────────────────────
cal = os.path.join(rev_dir("calibration"), "summary.csv")
md_table(pd.read_csv(cal) if os.path.exists(cal) else None, "Calibration (R1)")
lat = []
for p in glob.glob(os.path.join(rev_dir("latency"), "*.json")):
    d = J(p)
    for t in d["timings"]:
        lat.append({"model": d["model"], "device": t["device"], "batch": t["batch_size"],
                    **{k: v for k, v in t["ms_per_image"].items()}, "preprocess_cpu": d["preprocess_ms_per_image_cpu"],
                    "hw": d["hardware"].get("gpu") if t["device"] == "cuda" else d["hardware"].get("cpu")})
md_table(pd.DataFrame(lat), "Latency per image, ms (R5 #9)")
xa = [{"model": J(p)["model"], "stage": J(p)["stage"], "method": k, **v}
      for p in glob.glob(os.path.join(rev_dir("xai"), "*.json")) for k, v in J(p)["summary"].items()]
md_table(pd.DataFrame(xa), "Explanation faithfulness and stability (R1, R5 #8)",
         "deletion AUC ↓ better, insertion AUC ↑ better; 'random' is the reference map.")
sc = [J(p) for p in glob.glob(os.path.join(rev_dir("splitcifar"), "*.json"))]
def _sc_row(r):
    agem = r["method"].startswith("A-GEM")                 # A-GEM never clips / has no reference-style switch
    return {"method": r["method"], "buffer": r["buffer"], "seed": r["seed"],
            "grad_clip": "n/a" if agem else r.get("grad_clip", 1.0),
            "reference_style_DER++": "n/a" if r["method"] != "DER++" else r.get("der_ref_mode", False),
            "class_il_final_acc": r["class_il_final_acc"], "task_il_final_acc": r["task_il_final_acc"], "BWT": r["BWT"]}
_scdf = pd.DataFrame([_sc_row(r) for r in sc])
if len(_scdf): _scdf = _scdf.sort_values(["method", "buffer", "grad_clip", "reference_style_DER++"], key=lambda c: c.astype(str)).reset_index(drop=True)
md_table(_scdf, "Split-CIFAR-10 validation of CL implementations (R4)",
         "grad_clip = 1.0 and reference_style_DER++ = False is the original configuration; the other rows are the pilot variants "
         "(no gradient clipping; DER++ as in Buzzega et al.: two independent buffer batches, stored pre-update logits). "
         "Literature (Mammoth, buffer 200): A-GEM ~20-23 %, DER++ ~65-73 %.")
_pd = os.path.join(rev_dir(), "pilot_decision.json")
if os.path.exists(_pd):
    _d = J(_pd)
    MD.append("\n**Pilot decision** (settings used by the continual-learning runs): " + (", ".join(_d["settings"]) or "none (original settings)") + "\n")

with open(os.path.join(rev_dir(), "REVISION_RESULTS.md"), "w") as f:
    f.write("# Krishak revision — results\n\nGenerated by 17_revision_report.py\n" + "\n".join(MD) + "\n")
print(f"\n  ✓ wrote {os.path.join(rev_dir(), 'REVISION_RESULTS.md')} and {len(os.listdir(TAB))} CSV tables")
