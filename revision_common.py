"""
revision_common.py — shared helpers for the revision experiment scripts (09–17).

Every script reuses the pipeline's own code (config/utils and the numbered
stage scripts, imported by path) so that the revision experiments evaluate
exactly the same models, split and preprocessing as the main pipeline.
"""
import os, sys, json, time, importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import *            # noqa: E402,F401
from utils import (             # noqa: E402
    set_seed, get_model, get_feature_extractor, PlantDiseaseDataset, split_dataframe,
    get_train_transform, get_val_transform, save_results, load_results,
)

_MODS = {}


def stage(name: str):
    """Import a numbered stage script (e.g. '04_qefs') as a module, once."""
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(f"stage_{name}", os.path.join(HERE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[name] = mod
    return _MODS[name]


class MultiDirDataset(PlantDiseaseDataset):
    """Looks for each image in train_images/<label>/ and aug_images/ (flat)."""
    def __init__(self, df, dirs, transform=None):
        super().__init__(df, dirs[0], transform)
        self.dirs = dirs

    def path_of(self, idx):
        row = self.df.iloc[idx]
        for d in self.dirs:
            for p in (os.path.join(d, row["label"], row["image_id"]), os.path.join(d, row["image_id"])):
                if os.path.exists(p):
                    return p
        raise FileNotFoundError(row["image_id"])

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.path_of(idx)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[self.df.iloc[idx]["label"]]


def image_dirs():
    dirs = [TRAIN_IMAGES]
    aug = os.path.join(DATASET_DIR, "aug_images")
    if os.path.isdir(aug):
        dirs.append(aug)
    return dirs


def load_splits():
    """Same CSV and leakage-safe split as the pipeline (honours KRISHAK_OUTER_FOLD)."""
    csv = BALANCED_CSV if os.path.exists(BALANCED_CSV) else CLEANED_CSV
    df = pd.read_csv(csv)
    tr, va, te = split_dataframe(df, save_manifest=False)
    return tr, va, te, image_dirs()


def loader(df, dirs, train=False, batch_size=BATCH_SIZE, shuffle=None):
    tf = get_train_transform() if train else get_val_transform()
    return DataLoader(MultiDirDataset(df, dirs, tf), batch_size=batch_size,
                      shuffle=train if shuffle is None else shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def _plain(model):
    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            o = self.m(x)
            return o[0] if isinstance(o, (tuple, list)) else o
    return _W(model)


def load_baseline(model_name):
    p = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(p):
        return None
    m = get_model(model_name, pretrained=False)
    m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=False)["model_state"])
    return _plain(m).to(DEVICE).eval()


def load_gnas(model_name, tag="gnas"):
    return stage("05_genetic_nas").load_gnas_model(model_name, tag=tag)


def load_qcnn(model_name):
    """QEFS model: CNN + QIL + linear head (04_qefs.py checkpoint)."""
    q = stage("04_qefs")
    p = os.path.join(MODELS_DIR, f"qefs_{model_name}_best.pth")
    if not os.path.exists(p):
        return None
    ext = get_feature_extractor(get_model(model_name, pretrained=False), model_name)
    m = q.QCNNHead(ext, q.get_backbone_dim(model_name), QEFS_FEATURE_DIM)
    m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=False)["model_state"])
    return m.to(DEVICE).eval()


@torch.no_grad()
def predict(model, ld):
    """Return (logits, labels) as numpy arrays."""
    model.eval()
    L, Y = [], []
    for x, y in ld:
        L.append(model(x.to(DEVICE)).float().cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(L), np.concatenate(Y)


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def rev_dir(sub=""):
    d = os.path.join(RESULTS_DIR, "revision", sub)
    os.makedirs(d, exist_ok=True)
    return d


def dump(obj, name, sub=""):
    p = os.path.join(rev_dir(sub), name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  saved → {p}")
    return p


def hardware():
    import platform
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cpu": platform.processor() or platform.machine(), "cpu_threads": os.cpu_count(),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK")}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info
