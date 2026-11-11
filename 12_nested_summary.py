"""
12_nested_summary.py — Aggregate the nested cross-validation runs (R1, R5 #2)
============================================================================
Each outer fold k was produced by running 03 → 04 → 05 with
  KRISHAK_RUN_TAG=nested_f<k>  KRISHAK_OUTER_FOLD=<k>  KRISHAK_SKIP_CV=1
so the backbone, QIL, HFSEA mask, GNAS search and final head of fold k never
saw fold k's photographs (and never saw augmented copies of them).

This script reads outputs/nested_f1..f5/results and reports, per backbone and
stage (Baseline, QEFS, GNAS), the outer-fold accuracy / macro-F1: mean ± SD and
95% t-interval over folds.  Out: results/revision/nested_cv_summary.{json,csv}
(in the main RUN_TAG folder).
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa
from config import _OUT_ROOT

STAGES = {  # stage → (file pattern, acc key, f1 key)
    "Baseline": ("{m}.json",      "test_acc",      "f1_macro"),
    "QEFS":     ("qefs_{m}.json", "qefs_test_acc", "qefs_f1_macro"),
    "GNAS":     ("gnas_{m}.json", "test_accuracy", "test_f1_macro"),
}

rows = []
for k in range(1, CV_FOLDS + 1):
    rd = os.path.join(_OUT_ROOT, f"nested_f{k}", "results")
    for m in ALL_MODEL_NAMES:
        for st, (pat, ak, fk) in STAGES.items():
            p = os.path.join(rd, pat.format(m=m))
            if os.path.exists(p):
                d = json.load(open(p))
                rows.append({"fold": k, "model": m, "stage": st, "acc": d.get(ak), "f1": d.get(fk),
                             "n_selected": d.get("n_features_selected")})
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("No nested results found under outputs/nested_f*/results — run the nested jobs first.")

summ = []
for (m, st), g in df.groupby(["model", "stage"]):
    a = g["acc"].astype(float).values
    ci = stats.t.interval(0.95, len(a) - 1, loc=a.mean(), scale=stats.sem(a)) if len(a) > 1 else (np.nan, np.nan)
    summ.append({"model": m, "stage": st, "folds": len(a), "acc_mean": a.mean(), "acc_sd": a.std(ddof=1) if len(a) > 1 else 0,
                 "acc_ci95_lo": ci[0], "acc_ci95_hi": ci[1], "f1_mean": g["f1"].astype(float).mean(),
                 "fold_accs": a.round(4).tolist()})
S = pd.DataFrame(summ).sort_values(["model", "stage"])
out = rev_dir()
S.to_csv(os.path.join(out, "nested_cv_summary.csv"), index=False)
df.to_csv(os.path.join(out, "nested_cv_folds.csv"), index=False)
dump({"summary": S.to_dict("records")}, "nested_cv_summary.json")
pd.set_option("display.width", 160)
print(S.drop(columns=["fold_accs"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
