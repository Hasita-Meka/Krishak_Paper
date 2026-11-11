"""
11_qil_ablation.py — What does the quantum-inspired layer (QIL) contribute? (R1, R4, R3 #2)
=========================================================================================
Frozen backbone features (baseline checkpoint from 03) are extracted once, then
each projection variant (all → 512-d, then Dropout(0.4) + Linear head, exactly
like QCNNHead) is trained on the SAME cached features with the same optimiser,
epochs and early stopping, over several seeds:

  qil_full        rotation + Hadamard + entanglement + amplitude norm + BN + GELU (04_qefs.py)
  no_rotation     − Rx/Ry/Rz rotations
  no_hadamard     − Hadamard branch (x_r @ H^T)
  no_entangle     − entanglement branch (ent_fc)
  no_amp_norm     − amplitude (L2) normalisation
  no_bn           − BatchNorm (R3: "what is the need of BN in Eq. 3?")
  linear_gelu     plain learned Linear(d,512) + GELU        (R1/R4: "ordinary learned projection")
  mlp_bn_gelu     Linear(d,512) + BN + GELU                 (ordinary MLP layer)
  no_projection   head directly on the d-dim backbone features

Note: features are extracted without train-time augmentation for all variants,
so absolute numbers differ slightly from 04; the comparison between variants is
what this script measures.

Env: KRISHAK_MODELS (default all), KRISHAK_QIL_SEEDS (default 42,43,44), KRISHAK_QIL_EPOCHS (default 50).
Out: results/revision/qil_ablation/<model>.json (+ cached features npz)
"""
import os, sys, copy
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa

Q = stage("04_qefs")
SEEDS = [int(s) for s in os.environ.get("KRISHAK_QIL_SEEDS", "42,43,44").split(",")]
EPOCHS = 1 if SMOKE else int(os.environ.get("KRISHAK_QIL_EPOCHS", 50))
FLAGS = {
    "qil_full":    dict(rot=1, had=1, ent=1, norm=1, bn=1),
    "no_rotation": dict(rot=0, had=1, ent=1, norm=1, bn=1),
    "no_hadamard": dict(rot=1, had=0, ent=1, norm=1, bn=1),
    "no_entangle": dict(rot=1, had=1, ent=0, norm=1, bn=1),
    "no_amp_norm": dict(rot=1, had=1, ent=1, norm=0, bn=1),
    "no_bn":       dict(rot=1, had=1, ent=1, norm=1, bn=0),
}


class QILVariant(Q.QuantumInspiredLayer):
    def __init__(self, d_in, d_out, rot, had, ent, norm, bn):
        super().__init__(d_in, d_out)
        self.f = dict(rot=rot, had=had, ent=ent, norm=norm, bn=bn)

    def forward(self, x):
        f = self.f
        x_r = x
        if f["rot"]:
            x_r = torch.cos(self.theta_x) * x - torch.sin(self.theta_x) * x.roll(1, -1)
            x_r = torch.cos(self.theta_y) * x_r + torch.sin(self.theta_y) * x_r.roll(2, -1)
            x_r = torch.cos(self.theta_z) * x_r - torch.sin(self.theta_z) * x_r.roll(3, -1)
        out = 0
        if f["had"]: out = out + x_r @ self.H.T
        if f["ent"]: out = out + self.ent_fc(x_r)
        if f["norm"]: out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return self.act(self.bn(out)) if f["bn"] else self.act(out)


def make_proj(name, d):
    if name in FLAGS:
        return QILVariant(d, QEFS_FEATURE_DIM, **FLAGS[name]), QEFS_FEATURE_DIM
    if name == "linear_gelu":
        return nn.Sequential(nn.Linear(d, QEFS_FEATURE_DIM), nn.GELU()), QEFS_FEATURE_DIM
    if name == "mlp_bn_gelu":
        return nn.Sequential(nn.Linear(d, QEFS_FEATURE_DIM), nn.BatchNorm1d(QEFS_FEATURE_DIM), nn.GELU()), QEFS_FEATURE_DIM
    return nn.Identity(), d


@torch.no_grad()
def extract(model_name, dfs, dirs):
    cache = os.path.join(rev_dir("qil_ablation"), f"backbone_feats_{model_name}.npz")
    if os.path.exists(cache):
        z = np.load(cache); return [(z[f"X{i}"], z[f"y{i}"]) for i in range(3)]
    base = load_baseline(model_name)
    ext = get_feature_extractor(base.m, model_name).to(DEVICE).eval()
    out = []
    for df in dfs:
        X, Y = [], []
        for x, y in loader(df, dirs, train=False, shuffle=False):
            f = ext(x.to(DEVICE)); f = f[0] if isinstance(f, (tuple, list)) else f
            X.append(f.float().cpu().numpy()); Y.append(y.numpy())
        out.append((np.concatenate(X), np.concatenate(Y)))
    np.savez_compressed(cache, **{f"X{i}": o[0] for i, o in enumerate(out)}, **{f"y{i}": o[1] for i, o in enumerate(out)})
    return out


def train_variant(name, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [(torch.tensor(X, dtype=torch.float32), torch.tensor(y)) for X, y in data]
    proj, dout = make_proj(name, Xtr.shape[1])
    net = nn.Sequential(proj, nn.Dropout(0.4), nn.Linear(dout, NUM_CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            b = perm[i:i + 128]
            if len(b) < 2: continue
            opt.zero_grad(); crit(net(Xtr[b].to(DEVICE)), ytr[b].to(DEVICE)).backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            va = (net(Xva.to(DEVICE)).argmax(1).cpu() == yva).float().mean().item()
        if va > best: best, best_state, bad = va, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        p = net(Xte.to(DEVICE)).argmax(1).cpu().numpy()
    return {"val_acc": best, "test_acc": float((p == yte.numpy()).mean()),
            "test_f1_macro": float(f1_score(yte.numpy(), p, average="macro")),
            "n_params_projection": int(sum(q.numel() for q in proj.parameters()))}


if __name__ == "__main__":
    tr, va, te, dirs = load_splits()
    is_aug = tr["is_augmented"].astype(str).str.lower().isin(["true", "1"])
    for mn in MODEL_NAMES:
        if load_baseline(mn) is None:
            print(f"  [skip] {mn}: no baseline checkpoint"); continue
        print(f"\n=== QIL ablation: {mn} ===")
        data = extract(mn, [tr, va, te], dirs)
        res = {}
        for name in list(FLAGS) + ["linear_gelu", "mlp_bn_gelu", "no_projection"]:
            runs = [train_variant(name, data, s) for s in SEEDS]
            accs = [r["test_acc"] for r in runs]
            res[name] = {"runs": runs, "test_acc_mean": float(np.mean(accs)), "test_acc_sd": float(np.std(accs)),
                         "test_f1_mean": float(np.mean([r["test_f1_macro"] for r in runs]))}
            print(f"   {name:<14} acc={np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)
        dump({"model": mn, "seeds": SEEDS, "epochs": EPOCHS, "variants": res}, f"{mn}.json", "qil_ablation")
