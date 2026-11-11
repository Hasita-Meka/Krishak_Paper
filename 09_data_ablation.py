"""
09_data_ablation.py — Data-stage and imbalance-correction ablation (R1, R4, R5 #12)
===================================================================================
Same backbone, same leak-free val/test sets (originals only), same training
recipe as 03_train_cnn_models.py; only the TRAINING data / imbalance handling change:

  raw_none        raw Paddy Doctor photos (before cleaning), minus the val/test photos
                  and any exact or near-duplicate (pHash Hamming ≤ 8) of a val/test photo
  clean_none      cleaned originals only (duplicates removed), no imbalance correction
  clean_sampler   + class-balanced WeightedRandomSampler only
  clean_wloss     + class-weighted cross-entropy only
  clean_aug       + offline augmentation to ~1,700/class (train photos only), nothing else
  clean_aug_all3  augmentation + sampler + weighted loss (= the submitted recipe)

Rows raw_none → clean_none → clean_aug → (QEFS/GNAS/CL from 04–06) form the
incremental ablation; the clean_* rows answer "is the triple correction needed?".

Env: KRISHAK_ABL_MODEL (default mobilenet_v3_large),
     KRISHAK_ABL_CONDITIONS (comma list, default all), KRISHAK_SEED.
Out: outputs/<RUN_TAG>/results/revision/data_ablation/<model>_<condition>[_s<seed>].json
"""
import os, sys, time, hashlib, copy
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from PIL import Image
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *                      # noqa
from utils import train_one_epoch, evaluate, EarlyStopping, compute_class_weights

MODEL = os.environ.get("KRISHAK_ABL_MODEL", "mobilenet_v3_large")
CONDS = ["raw_none", "clean_none", "clean_sampler", "clean_wloss", "clean_aug", "clean_aug_all3"]
if os.environ.get("KRISHAK_ABL_CONDITIONS"):
    CONDS = [c.strip() for c in os.environ["KRISHAK_ABL_CONDITIONS"].split(",")]


def _bits(path, hash_size=16):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        a = np.asarray(img, dtype=np.float32)
        return hashlib.md5(raw).hexdigest(), (a > a.mean()).flatten()
    except Exception:
        return None, None


def raw_training_set(val_df, test_df):
    """Raw train.csv photos that are safe to train on against this val/test split."""
    from joblib import Parallel, delayed
    raw = pd.read_csv(TRAIN_CSV)
    held = pd.concat([val_df, test_df])
    raw = raw[~raw["image_id"].isin(set(held["image_id"]))].reset_index(drop=True)
    p = lambda r: os.path.join(TRAIN_IMAGES, r["label"], r["image_id"])
    nj = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"  hashing {len(raw):,} raw + {len(held):,} held-out images …")
    rh = Parallel(n_jobs=nj)(delayed(_bits)(p(r)) for _, r in raw.iterrows())
    hh = Parallel(n_jobs=nj)(delayed(_bits)(p(r)) for _, r in held.iterrows())
    ok = np.array([m is not None for m, _ in rh])
    held_md5 = {m for m, _ in hh if m}
    A = np.stack([b for m, b in rh if m is not None]).astype(np.float32)
    B = np.stack([b for m, b in hh if m is not None]).astype(np.float32)
    # Hamming(a,b) = |a| + |b| - 2 a·b, computed in chunks
    near = np.zeros(len(A), bool)
    for i in range(0, len(A), 2048):
        a = A[i:i + 2048]
        d = a.sum(1, keepdims=True) + B.sum(1)[None] - 2 * a @ B.T
        near[i:i + 2048] = (d <= 8).any(1)
    raw_ok = raw[ok].reset_index(drop=True)
    md5dup = np.array([m in held_md5 for m, _ in rh if m is not None])
    keep = ~(near | md5dup)
    print(f"  raw: {len(raw):,} candidates, {int((~ok).sum())} unreadable, "
          f"{int((near | md5dup).sum())} (near-)duplicates of val/test removed → {int(keep.sum()):,}")
    return raw_ok[keep].reset_index(drop=True), {"raw_candidates": len(raw), "unreadable": int((~ok).sum()),
                                                "dup_of_heldout_removed": int((near | md5dup).sum())}


def run(cond, tr_df, va_df, te_df, dirs, extra):
    set_seed()
    use_sampler = cond in ("clean_sampler", "clean_aug_all3")
    use_wloss   = cond in ("clean_wloss", "clean_aug_all3")
    tr_ds = MultiDirDataset(tr_df, dirs, get_train_transform())
    if use_sampler:
        t = np.array(tr_ds.targets); c = np.bincount(t, minlength=NUM_CLASSES)
        sampler = WeightedRandomSampler((1.0 / (c + 1e-6))[t], len(t), replacement=True)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    else:
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    va_ld, te_ld = loader(va_df, dirs), loader(te_df, dirs)

    model = get_model(MODEL).to(DEVICE)
    w = compute_class_weights(tr_df).to(DEVICE) if use_wloss else None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=ETA_MIN)
    scaler, es = GradScaler(device="cuda", enabled=USE_AMP), EarlyStopping(patience=PATIENCE)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    best, best_state, best_ep, t0 = -1, None, 0, time.time()
    for ep in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, tr_ld, opt, crit, scaler)
        _, va, _, _ = evaluate(model, va_ld, nn.CrossEntropyLoss())
        sch.step()
        if va > best:
            best, best_ep, best_state = va, ep, copy.deepcopy(model.state_dict())
        print(f"   [{cond}] ep {ep:3d}  val_acc={va:.4f}  best={best:.4f}", flush=True)
        if es(va):
            break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    _, te_acc, p, y = evaluate(model, te_ld, nn.CrossEntropyLoss())
    counts = tr_df["label"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
    return {
        "condition": cond, "model_name": MODEL, "seed": SEED,
        "n_train": int(len(tr_df)), "train_class_counts": counts.to_dict(),
        "imbalance_ratio": float(counts.max() / max(1, counts.min())),
        "sampler": use_sampler, "weighted_loss": use_wloss,
        "augmented": cond.startswith("clean_aug"),
        "val_acc": float(best), "best_epoch": best_ep,
        "test_acc": float(te_acc), "test_f1_macro": float(f1_score(y, p, average="macro")),
        "train_time_s": train_time,
        "peak_gpu_mem_MB": float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None,
        "preds": p.tolist(), "labels": y.tolist(), **extra,
    }


if __name__ == "__main__":
    print(f"\n=== 09 data ablation | model={MODEL} | conditions={CONDS} | seed={SEED} ===")
    tr, va, te, dirs = load_splits()
    is_aug = tr["is_augmented"].astype(str).str.lower().isin(["true", "1"]) if "is_augmented" in tr else pd.Series(False, index=tr.index)
    sets = {"clean": tr[~is_aug].reset_index(drop=True), "aug": tr.reset_index(drop=True)}
    suffix = f"_s{SEED}" if SEED != 42 else ""
    for cond in CONDS:
        out = os.path.join(rev_dir("data_ablation"), f"{MODEL}_{cond}{suffix}.json")
        if os.path.exists(out):
            print(f"  [skip] {cond} done"); continue
        extra = {}
        if cond == "raw_none":
            tr_df, extra = raw_training_set(va, te)
        elif cond.startswith("clean_aug"):
            tr_df = sets["aug"]
        else:
            tr_df = sets["clean"]
        r = run(cond, tr_df, va, te, dirs, extra)
        dump(r, os.path.basename(out), "data_ablation")
        print(f"  ✓ {cond:<15} n_train={r['n_train']:>6}  test_acc={r['test_acc']:.4f}  F1={r['test_f1_macro']:.4f}")
