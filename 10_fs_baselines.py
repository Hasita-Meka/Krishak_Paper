"""
10_fs_baselines.py — Feature-selection baselines and HFSEA sensitivity (CPU only)
=================================================================================
Uses the quantum features cached by 04_qefs.py (results/qefs_features_<model>.npz)
and the HFSEA result (results/qefs_<model>.json). Every method selects on
train+val features (as HFSEA does) and is scored with the SAME final classifier as
04_qefs.py (StandardScaler + MLP(256,128)) trained on train, tested on test.

Parts (KRISHAK_FS_PARTS, comma list; default all):
  baselines  R1/R4/R5 #13: all features, variance, ANOVA-F, mutual information,
             L1-logistic, RFE(logistic), PCA, random subsets (5 draws), plain GA
             (same fitness + evaluation budget as HFSEA) — all at k = HFSEA's |mask|
  lambda     R4: HFSEA with λ ∈ {0, 0.001, 0.01, 0.05, 0.1, 0.25} (Eq. 4)
  iters      R3 #1: one HFSEA run for 100 iterations; test acc of the best mask
             at iterations 0/10/25/50/75/100 (is 50 enough?)
  seeds      R1/R5 #15: HFSEA re-run with seeds 42,43,44 → variability of |mask| and acc

Env: KRISHAK_MODELS, KRISHAK_FS_PARTS, KRISHAK_NJOBS (default SLURM_CPUS_PER_TASK).
Out: results/revision/fs/<model>_<part>.json
"""
import os, sys, time, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa

PARTS = os.environ.get("KRISHAK_FS_PARTS", "baselines,lambda,iters,seeds").split(",")
Q = stage("04_qefs")


def final_score(Xtr, ytr, Xte, yte, seed=SEED):
    """Exactly the 04_qefs.py final classifier."""
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=seed,
                        early_stopping=True, validation_fraction=0.1, learning_rate_init=0.001)
    t = time.time(); clf.fit(sc.transform(Xtr), ytr); tt = time.time() - t
    p = clf.predict(sc.transform(Xte))
    return {"test_acc": float(accuracy_score(yte, p)), "test_f1_macro": float(f1_score(yte, p, average="macro")),
            "clf_train_s": tt}


def topk(scores, k):
    return np.argsort(-np.nan_to_num(scores, nan=-np.inf))[:k]


def plain_ga(X, y, k_budget_evals, D, lam, seed, pop=30):
    """Binary GA with the HFSEA fitness function and the same number of evaluations."""
    rng = np.random.RandomState(seed)
    h = Q.HFSEA(D, pop_size=pop, max_iter=1, min_features=max(10, D // 10), lam=lam)
    popm = (rng.rand(pop, D) > 0.5).astype(float)
    fit = np.array([f for _, f in h._parallel([(h._init_task, i, popm[i], X, y) for i in range(pop)])])
    evals, gen, hist = pop, 0, [float(fit.max())]
    while evals < k_budget_evals:
        gen += 1
        elite = popm[np.argsort(-fit)[:2]].copy()
        kids = []
        while len(kids) < pop - 2:
            a, b = [popm[max(rng.choice(pop, 3, replace=False), key=lambda i: fit[i])] for _ in range(2)]
            c = np.where(rng.rand(D) < 0.5, a, b) if rng.rand() < 0.8 else a.copy()   # uniform crossover
            flip = rng.rand(D) < 1.0 / D
            c[flip] = 1 - c[flip]
            if c.sum() < h.min_features:
                c[rng.choice(D, h.min_features, replace=False)] = 1
            kids.append(c)
        kids = np.array(kids)
        kf = np.array([f for _, f in h._parallel([(h._init_task, i, kids[i], X, y) for i in range(len(kids))])])
        evals += len(kids)
        popm = np.vstack([elite, kids]); fit = np.concatenate([np.sort(fit)[::-1][:2], kf])
        hist.append(float(fit.max()))
    best = popm[int(np.argmax(fit))] > 0.5
    return np.where(best)[0], {"generations": gen, "evals": evals, "history": hist}


def run_hfsea(X, y, lam, iters, seed, D):
    np.random.seed(seed)
    h = Q.HFSEA(D, pop_size=QEFS_POP_SIZE, max_iter=iters, min_features=max(10, D // 10), lam=lam)
    t = time.time(); m, f, hist = h.run(X, y, verbose=True)
    return h, m, f, hist, time.time() - t


if __name__ == "__main__":
    for mn in MODEL_NAMES:
        fp = os.path.join(RESULTS_DIR, f"qefs_features_{mn}.npz")
        qp = os.path.join(RESULTS_DIR, f"qefs_{mn}.json")
        if not (os.path.exists(fp) and os.path.exists(qp)):
            print(f"  [skip] {mn}: run 04_qefs.py first"); continue
        d = np.load(fp); qj = json.load(open(qp))
        Xtr, ytr, Xva, yva, Xte, yte = (d[k] for k in ("X_tr", "y_tr", "X_va", "y_va", "X_te", "y_te"))
        Xall, yall = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
        D = Xall.shape[1]; k = int(qj["n_features_selected"]); budget = int(qj.get("hfsea_fitness_evals") or 1500)
        print(f"\n=== {mn}: D={D}, HFSEA k={k}, HFSEA evals={budget} ===")

        if "baselines" in PARTS:
            out = {}
            hidx = np.array(qj["selected_feature_idx"])
            out["hfsea"] = {**final_score(Xtr[:, hidx], ytr, Xte[:, hidx], yte), "k": len(hidx),
                            "select_s": qj.get("hfsea_search_time_s")}
            out["all_features"] = {**final_score(Xtr, ytr, Xte, yte), "k": D, "select_s": 0.0}
            Z = StandardScaler().fit(Xall).transform(Xall)
            sel = {
                "variance":  lambda: topk(Xall.var(0), k),
                "anova_f":   lambda: topk(f_classif(Xall, yall)[0], k),
                "mutual_info": lambda: topk(mutual_info_classif(Xall, yall, random_state=SEED), k),
                "l1_logistic": lambda: topk(np.abs(LogisticRegression(penalty="l1", solver="saga", C=0.1,
                                        max_iter=2000, random_state=SEED).fit(Z, yall).coef_).sum(0), k),
                "rfe_logistic": lambda: np.where(RFE(LogisticRegression(max_iter=2000), n_features_to_select=k,
                                        step=0.1).fit(Z, yall).support_)[0],
            }
            for name, fn in sel.items():
                t = time.time(); idx = fn(); ts = time.time() - t
                out[name] = {**final_score(Xtr[:, idx], ytr, Xte[:, idx], yte), "k": int(len(idx)), "select_s": ts}
                print(f"   {name:<14} k={len(idx)}  acc={out[name]['test_acc']:.4f}  ({ts:.1f}s)", flush=True)
            t = time.time(); pca = PCA(n_components=k, random_state=SEED).fit(Xall); ts = time.time() - t
            out["pca"] = {**final_score(pca.transform(Xtr), ytr, pca.transform(Xte), yte), "k": k, "select_s": ts}
            rs = [final_score(Xtr[:, idx], ytr, Xte[:, idx], yte)["test_acc"]
                  for idx in (np.random.RandomState(s).choice(D, k, replace=False) for s in range(5))]
            out["random_subset"] = {"test_acc": float(np.mean(rs)), "test_acc_sd": float(np.std(rs)), "k": k, "draws": rs}
            t = time.time(); gidx, ginfo = plain_ga(Xall, yall, budget, D, float(qj.get("hfsea_lambda", 0.001)), SEED)
            out["plain_ga"] = {**final_score(Xtr[:, gidx], ytr, Xte[:, gidx], yte), "k": int(len(gidx)),
                               "select_s": time.time() - t, **ginfo}
            dump({"model": mn, "k_hfsea": k, "results": out}, f"{mn}_baselines.json", "fs")

        if "lambda" in PARTS:
            rows = []
            for lam in ((0.0, 0.1) if SMOKE else (0.0, 0.001, 0.01, 0.05, 0.1, 0.25)):
                h, m, f, hist, ts = run_hfsea(Xall, yall, lam, QEFS_MAX_ITER, SEED, D)
                rows.append({"lambda": lam, "n_selected": int(m.sum()), "retention": float(m.mean()),
                             "fitness": float(f), "search_s": ts, "evals": h.n_evals,
                             **final_score(Xtr[:, m], ytr, Xte[:, m], yte)})
                print(f"   λ={lam:<6} |mask|={int(m.sum())}  acc={rows[-1]['test_acc']:.4f}", flush=True)
            dump({"model": mn, "rows": rows}, f"{mn}_lambda.json", "fs")

        if "iters" in PARTS:
            h, m, f, hist, ts = run_hfsea(Xall, yall, float(qj.get("hfsea_lambda", 0.001)), 2 if SMOKE else 100, SEED, D)
            cps = []
            for it in (0, 10, 25, 50, 75, 100):
                idx = h.mask_history[min(it, len(h.mask_history) - 1)].astype(int)
                cps.append({"iteration": it, "n_selected": int(len(idx)), "fitness": float(hist[min(it, len(hist) - 1)]),
                            **final_score(Xtr[:, idx], ytr, Xte[:, idx], yte)})
            dump({"model": mn, "fitness_history": hist, "selected_history": h.selected_history,
                  "checkpoints": cps, "search_s": ts, "evals": h.n_evals}, f"{mn}_iters.json", "fs")

        if "seeds" in PARTS:
            rows = []
            for s in (42, 43, 44):
                h, m, f, hist, ts = run_hfsea(Xall, yall, float(qj.get("hfsea_lambda", 0.001)), QEFS_MAX_ITER, s, D)
                rows.append({"seed": s, "n_selected": int(m.sum()), "fitness": float(f), "search_s": ts,
                             "evals": h.n_evals, "mask": np.where(m)[0].tolist(),
                             **final_score(Xtr[:, m], ytr, Xte[:, m], yte, seed=s)})
            masks = [set(r["mask"]) for r in rows]
            jac = [len(a & b) / len(a | b) for i, a in enumerate(masks) for b in masks[i + 1:]]
            dump({"model": mn, "rows": rows, "mask_jaccard_mean": float(np.mean(jac))}, f"{mn}_seeds.json", "fs")
