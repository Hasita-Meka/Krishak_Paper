"""
14_latency_profile.py — Stage-wise and end-to-end inference latency (R5 #9, R1 Table 4)
=====================================================================================
Separates classifier-stage speed from raw-image-to-prediction latency:

  preprocess   JPEG decode + resize/normalise (CPU, per image)
  backbone     CNN feature extractor
  qil          quantum-inspired layer (512-d)
  select       HFSEA index selection + StandardScaler
  mlp          sklearn MLP (04_qefs final classifier, CPU)
  gnas_head    GNAS searched head on selected QIL features
  e2e_*        baseline CNN, QEFS path (backbone→QIL→select→MLP) and GNAS model,
               from an already-decoded tensor; e2e_*_with_preprocess adds decoding.

Each is timed on GPU (if present) and CPU, batch sizes 1 and 32, with warm-up and
CUDA synchronisation; median over repeats, reported in ms per image and per batch.

Env: KRISHAK_MODELS, KRISHAK_LAT_REPEATS (default 50), KRISHAK_CPU_THREADS (default 8).
Out: results/revision/latency/<model>.json
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import *     # noqa

REPEATS = 3 if SMOKE else int(os.environ.get("KRISHAK_LAT_REPEATS", 50))
torch.set_num_threads(int(os.environ.get("KRISHAK_CPU_THREADS", 8)))


def timeit(fn, dev, repeats=REPEATS, warm=5):
    for _ in range(min(warm, repeats)): fn()
    ts = []
    for _ in range(repeats):
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if dev == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


if __name__ == "__main__":
    import joblib
    tr, va, te, dirs = load_splits()
    ds = MultiDirDataset(te, dirs, get_val_transform())
    paths = [ds.path_of(i) for i in range(min(64, len(ds)))]
    devs = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    for mn in MODEL_NAMES:
        base, qcnn, gnas = load_baseline(mn), load_qcnn(mn), load_gnas(mn)
        mlp_p = os.path.join(MODELS_DIR, f"qefs_mlp_{mn}.joblib")
        if base is None or qcnn is None or gnas is None or not os.path.exists(mlp_p):
            print(f"  [skip] {mn}: needs 03, 04 and 05 outputs"); continue
        mlp = joblib.load(mlp_p); mask = np.asarray(mlp["mask"], bool)
        from PIL import Image
        tf = get_val_transform()
        pre_ms = timeit(lambda: tf(Image.open(paths[0]).convert("RGB")), "cpu") 
        res = {"model": mn, "hardware": hardware(), "preprocess_ms_per_image_cpu": pre_ms,
               "params": {"baseline": sum(p.numel() for p in base.parameters()),
                          "backbone": sum(p.numel() for p in qcnn.backbone.parameters()),
                          "qil": sum(p.numel() for p in qcnn.q_layer.parameters()),
                          "gnas_head": sum(p.numel() for p in gnas.head.parameters())},
               "n_selected": int(mask.sum()), "timings": []}
        for dev in devs:
            for m in (base, qcnn, gnas): m.to(dev).eval()
            for bs in (1, 32):
                x = torch.stack([ds[i % len(ds)][0] for i in range(bs)]).to(dev)
                with torch.no_grad():
                    f = qcnn.backbone(x); f = f[0] if isinstance(f, (tuple, list)) else f
                    q = qcnn.q_layer(f)
                    qn = q.cpu().numpy()
                    sel = torch.as_tensor(np.where(mask)[0], device=dev)
                    qs = q.index_select(1, sel)
                    def qefs_e2e():
                        ff = qcnn.backbone(x); ff = ff[0] if isinstance(ff, (tuple, list)) else ff
                        z = qcnn.q_layer(ff).cpu().numpy()[:, mask]
                        return mlp["clf"].predict(mlp["scaler"].transform(z))
                    t = {
                        "backbone":  timeit(lambda: qcnn.backbone(x), dev),
                        "qil":       timeit(lambda: qcnn.q_layer(f), dev),
                        "select":    timeit(lambda: mlp["scaler"].transform(qn[:, mask]), "cpu"),
                        "mlp":       timeit(lambda: mlp["clf"].predict(mlp["scaler"].transform(qn[:, mask])), "cpu"),
                        "gnas_head": timeit(lambda: gnas.head(qs), dev),
                        "e2e_baseline_cnn": timeit(lambda: base(x), dev),
                        "e2e_qefs_mlp":     timeit(qefs_e2e, dev),
                        "e2e_gnas":         timeit(lambda: gnas(x), dev),
                    }
                for k in [k for k in t if k.startswith("e2e_")]:
                    t[k + "_with_preprocess"] = t[k] + pre_ms * bs
                res["timings"].append({"device": dev, "batch_size": bs,
                                       "ms_per_batch": t, "ms_per_image": {k: v / bs for k, v in t.items()}})
                print(f"   {mn} {dev} bs={bs}: " + "  ".join(f"{k}={v/bs:.3f}" for k, v in t.items() if not k.endswith("preprocess")), flush=True)
        dump(res, f"{mn}.json", "latency")
