"""
13_agem_splitcifar.py — Validate the CL implementations on Split (Seq-)CIFAR-10 (R4)
====================================================================================
Runs the pipeline's OWN A-GEM / DER++ / ER-ACE classes (imported from
06_continual_learning.py) on the standard class-incremental Seq-CIFAR-10
benchmark (5 tasks × 2 classes, ResNet-18 with CIFAR stem trained from scratch,
SGD lr 0.03, batch 32, 50 epochs/task, buffers 200 and 500), so the numbers can
be compared with published results, e.g. Buzzega et al., NeurIPS 2020 (DER),
Table 2, Seq-CIFAR-10 Class-IL: A-GEM ≈ 20–23 %, DER++ ≈ 65–73 %.
(Verify the reference values against the paper before quoting them.)

A-GEM collapsing to ~20 % class-IL here would show that its poor Krishak result
is the known behaviour of A-GEM in class-incremental settings, not a bug.

Env: KRISHAK_CIFAR_DIR (where CIFAR-10 is / will be downloaded; default Dataset/cifar10),
     KRISHAK_CIFAR_METHODS (default A-GEM,A-GEM-legacy,DER++,ER-ACE),
     KRISHAK_CIFAR_BUFFERS (default 200,500), KRISHAK_CIFAR_EPOCHS (default 50), KRISHAK_SEED.
Out: results/revision/splitcifar/<method>_b<buffer>_s<seed>[<KRISHAK_CIFAR_TAG>].json
"""
import os, sys, time, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa

CL = stage("06_continual_learning")
ROOT = os.environ.get("KRISHAK_CIFAR_DIR", os.path.join(DATASET_DIR, "cifar10"))
# lists may be separated by "," or ":" (":" is safe inside sbatch --export)
METHODS = re.split(r"[,:]", os.environ.get("KRISHAK_CIFAR_METHODS", "A-GEM,A-GEM-legacy,DER++,ER-ACE"))
BUFFERS = [int(b) for b in re.split(r"[,:]", os.environ.get("KRISHAK_CIFAR_BUFFERS", "200,500"))]
EPOCHS = 1 if SMOKE else int(os.environ.get("KRISHAK_CIFAR_EPOCHS", 50))
LR = 0.03
TAG = os.environ.get("KRISHAK_CIFAR_TAG", "")      # suffix for pilot variants (never collides with the main files)
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)


def resnet18_cifar():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def task_loader(ds, classes, train):
    idx = [i for i, t in enumerate(ds.targets) if t in classes]
    if SMOKE: idx = idx[:64]
    return DataLoader(Subset(ds, idx), batch_size=32, shuffle=train, num_workers=NUM_WORKERS, drop_last=train)


@torch.no_grad()
def evaluate(model, loaders):
    model.eval(); cil, til = [], []
    for t, ld in enumerate(loaders):
        c = n = ct = 0
        for x, y in ld:
            o = model(x.to(DEVICE)).float(); y = y.to(DEVICE)
            c += (o.argmax(1) == y).sum().item()
            m = torch.full_like(o, float("-inf")); m[:, TASKS[t]] = 0
            ct += ((o + m).argmax(1) == y).sum().item(); n += len(y)
        cil.append(c / n); til.append(ct / n)
    return cil, til


def run(method, buf):
    torch.manual_seed(SEED); np.random.seed(SEED)
    tf_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_te = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    if SMOKE:   # synthetic stand-in so the code path can be tested without the download
        class _Fake(torch.utils.data.Dataset):
            def __init__(self, n): self.targets = [i % 10 for i in range(n)]
            def __len__(self): return len(self.targets)
            def __getitem__(self, i): return torch.randn(3, 32, 32), self.targets[i]
        tr, te = _Fake(700), _Fake(200)
    else:
        tr = torchvision.datasets.CIFAR10(ROOT, train=True, download=True, transform=tf_tr)
        te = torchvision.datasets.CIFAR10(ROOT, train=False, download=True, transform=tf_te)
    te_lds = [task_loader(te, c, False) for c in TASKS]
    model = resnet18_cifar().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    scaler = GradScaler(device="cuda", enabled=USE_AMP)
    if method.startswith("A-GEM"):
        obj = CL.AGEM(buffer_size=buf); obj.legacy = method.endswith("legacy")
    elif method == "DER++":
        obj = CL.DERpp(alpha=0.1, beta=0.5, buffer_size=buf)
    elif method == "ER-ACE":
        obj = CL.ERACE(buffer_size=buf)
    acc = np.zeros((5, 5)); t0 = time.time()
    for t, cls in enumerate(TASKS):
        ld = task_loader(tr, cls, True)
        for ep in range(EPOCHS):
            for x, y in ld:
                if isinstance(obj, CL.ERACE):
                    obj.observe(model, x, y, opt, scaler, set(cls))
                else:
                    obj.observe(model, x, y, opt, scaler)
        if hasattr(obj, "end_task"):
            obj.end_task()
        cil, til = evaluate(model, te_lds)
        acc[t] = cil
        print(f"   {method} b={buf} task {t}: class-IL={np.mean(cil[:t+1]):.4f}  task-IL={np.mean(til[:t+1]):.4f}", flush=True)
    return {"method": method, "buffer": buf, "seed": SEED, "epochs_per_task": EPOCHS, "lr": LR,
            "grad_clip": CL.CL_GRAD_CLIP, "der_ref_mode": CL.CL_REF_MODE, "tag": TAG,
            "class_il_final_acc": float(np.mean(cil)), "task_il_final_acc": float(np.mean(til)),
            "acc_matrix_class_il": acc.tolist(),
            "BWT": float(np.mean([acc[4, j] - acc[j, j] for j in range(4)])), "time_s": time.time() - t0,
            "hardware": hardware()}


if __name__ == "__main__":
    for b in BUFFERS:
        for m in METHODS:
            name = f"{m.replace('+', 'p')}_b{b}_s{SEED}{TAG}.json"
            if os.path.exists(os.path.join(rev_dir("splitcifar"), name)):
                print(f"  [skip] {name}"); continue
            r = run(m, b)
            dump(r, name, "splitcifar")
            print(f"  ✓ {m:<13} buffer={b}: class-IL {r['class_il_final_acc']:.4f} | task-IL {r['task_il_final_acc']:.4f}")
