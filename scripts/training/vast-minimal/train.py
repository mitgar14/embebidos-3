#!/usr/bin/env python3
"""Entrenamiento minimo YOLOv8n v1d (4 clases) para Vast.ai.

Version reducida del notebook train_v1d_vastai.ipynb: SIN CommitScheduler, SIN
heartbeat, SIN signal handlers, SIN auto-destroy. Descarga el dataset, entrena,
exporta ONNX (opset 11 para TRT 8.2 del Nano) y sube los artefactos a HF en un
unico upload final. El teardown de la instancia lo hace el usuario a mano con
`vastai destroy instance <id> -y`.

Contrato de clases (orden fijo, lo consume el frontend por indice):
    0 = glass, 1 = paper, 2 = plastic, 3 = cardboard
"""
import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CFG = {
    "hf_dataset": "mitgar14/embebidos3-dataset-v1d",
    "hf_repo": "mitgar14/embebidos-3-models-v1d",
    "model_arch": "yolov8n",
    "imgsz": 416,          # target Jetson Nano (igual que v1c en produccion)
    "epochs": 100,
    "batch": 32,
    "patience": 20,
    "device": 0,
    "seed": 42,
    # Augmentation reforzado (investigacion 2026-05-15: overfit de plastic).
    "aug": {
        "degrees": 20, "translate": 0.1, "scale": 0.5,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.6,
        "mixup": 0.15, "copy_paste": 0.3, "mosaic": 1.0,
        "close_mosaic": 15, "fliplr": 0.5, "flipud": 0.0,
    },
}
EXPECTED_CLASSES = ["glass", "paper", "plastic", "cardboard"]

ROOT = Path("/workspace/embebidos-3")
DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"
MANIFESTS = ROOT / "manifests"
EXPORTS = ROOT / "exports"
for _p in (ROOT, DATASETS, RUNS, MANIFESTS, EXPORTS):
    _p.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN ausente en el entorno (inyectar con vastai create --env).")

# --- 1. Verificacion de GPU / stack ------------------------------------------
import torch
import numpy as np

print(f"[env] python={platform.python_version()} torch={torch.__version__} "
      f"cuda_disp={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    sys.exit("CUDA no disponible: entrenamiento abortado.")
print(f"[env] gpu={torch.cuda.get_device_name(0)} "
      f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)
if int(np.__version__.split(".")[0]) >= 2:
    sys.exit(f"numpy {np.__version__} incompatible (se requiere <2).")

# --- 2. Descarga del dataset (ZIP unico desde HF) ----------------------------
from huggingface_hub import HfApi, hf_hub_download

DATA_DIR = DATASETS / "waste-4class-v1d"
if not (DATA_DIR / "data.yaml").exists():
    print(f"[data] descargando waste-4class-v1d.zip de {CFG['hf_dataset']}", flush=True)
    zip_path = hf_hub_download(
        repo_id=CFG["hf_dataset"], repo_type="dataset",
        filename="waste-4class-v1d.zip", token=HF_TOKEN,
        local_dir=str(DATA_DIR.parent),
    )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR.parent)
    try:
        Path(zip_path).unlink()
    except OSError:
        pass

DATA_YAML = DATA_DIR / "data.yaml"
if not DATA_YAML.exists():
    found = next(DATA_DIR.parent.rglob("data.yaml"), None)
    if found is None:
        sys.exit("data.yaml no encontrado tras descomprimir.")
    DATA_YAML = found
    DATA_DIR = DATA_YAML.parent

import yaml

with open(DATA_YAML) as f:
    dy = yaml.safe_load(f)
raw_names = dy.get("names", [])
CLASSES = ([raw_names[k] for k in sorted(raw_names)]
           if isinstance(raw_names, dict) else list(raw_names))
print(f"[data] data.yaml={DATA_YAML} clases={CLASSES}", flush=True)
if CLASSES != EXPECTED_CLASSES:
    print(f"[WARN] orden de clases {CLASSES} != contrato {EXPECTED_CLASSES}", flush=True)

# --- 3. Entrenamiento --------------------------------------------------------
from ultralytics import YOLO

model = YOLO(f"{CFG['model_arch']}.pt")
model.train(
    data=str(DATA_YAML), epochs=CFG["epochs"], imgsz=CFG["imgsz"],
    batch=CFG["batch"], patience=CFG["patience"], device=CFG["device"],
    seed=CFG["seed"], project=str(RUNS / "detect"), name="train",
    exist_ok=True, plots=True, verbose=True, **CFG["aug"],
)
BEST_PT = RUNS / "detect" / "train" / "weights" / "best.pt"
if not BEST_PT.exists():
    sys.exit(f"best.pt no generado en {BEST_PT}")
print(f"[train] best.pt={BEST_PT} ({BEST_PT.stat().st_size / 1e6:.1f} MB)", flush=True)

# --- 4. Evaluacion sobre el split de validacion ------------------------------
best_model = YOLO(str(BEST_PT))
metrics = best_model.val(
    data=str(DATA_YAML), split="val", imgsz=CFG["imgsz"], batch=CFG["batch"],
    device=CFG["device"], project=str(RUNS / "detect"), name="val_eval",
    exist_ok=True, plots=True, verbose=True,
)
eval_summary = {
    "mAP50": float(metrics.box.map50),
    "mAP50_95": float(metrics.box.map),
    "precision_mean": float(metrics.box.mp),
    "recall_mean": float(metrics.box.mr),
    "per_class": {
        CLASSES[i]: {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "AP50": float(metrics.box.ap50[i]),
            "AP50_95": float(metrics.box.ap[i]),
        }
        for i in range(len(CLASSES))
    },
}
(MANIFESTS / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
print(f"[eval] {json.dumps(eval_summary)}", flush=True)

# --- 5. Export ONNX (opset 11, nms=False: contrato TRT 8.2 del Nano) ---------
import onnx

ONNX_PATH = EXPORTS / "best.onnx"
exported = best_model.export(
    format="onnx", imgsz=CFG["imgsz"], opset=11,
    simplify=True, dynamic=False, nms=False, half=False, device="cpu",
)
shutil.copy(Path(exported), ONNX_PATH)

mdl = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(mdl)
BLACKLIST = {"GridSample", "MultiHeadAttention", "RoiAlign", "NonZero",
             "Reciprocal", "QLinearConv", "QLinearMatMul"}
ops = {n.op_type for n in mdl.graph.node}
opset_v = mdl.opset_import[0].version
gate3 = {
    "opset": int(opset_v),
    "ir_version": int(mdl.ir_version),
    "input_shape": [d.dim_value for d in mdl.graph.input[0].type.tensor_type.shape.dim],
    "n_outputs": len(mdl.graph.output),
    "ops_forbidden": sorted(ops & BLACKLIST),
    "pass": opset_v == 11 and mdl.ir_version <= 8 and not (ops & BLACKLIST),
}
(MANIFESTS / "gate3_onnx.json").write_text(json.dumps(gate3, indent=2))
print(f"[onnx] {json.dumps(gate3)}", flush=True)
if not gate3["pass"]:
    sys.exit(f"Gate 3 fallo (ONNX no apto para TRT 8.2): {gate3}")

# --- 6. Manifest consolidado -------------------------------------------------
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "model_arch": CFG["model_arch"],
    "imgsz": CFG["imgsz"],
    "classes": CLASSES,
    "dataset": {
        "source": "HF", "hf_repo": CFG["hf_dataset"], "version": "v1d",
        "data_yaml": str(DATA_YAML.relative_to(ROOT)),
    },
    "training": {
        "epochs_max": CFG["epochs"], "batch": CFG["batch"],
        "patience": CFG["patience"], "seed": CFG["seed"],
        "augmentation": CFG["aug"],
    },
    "artifacts": {
        "best_pt": {
            "path": "best.pt", "sha256": _sha256(BEST_PT),
            "size_mb": round(BEST_PT.stat().st_size / (1024 ** 2), 2),
        },
        "best_onnx": {
            "path": "best.onnx", "sha256": _sha256(ONNX_PATH),
            "size_mb": round(ONNX_PATH.stat().st_size / (1024 ** 2), 2),
            "opset": 11, "ir_version": gate3["ir_version"],
        },
    },
    "evaluation": eval_summary,
    "gates": {"gate3_onnx": gate3},
    "target_nano": {
        "device": "Jetson Nano B01", "jetpack": "4.6.1", "l4t": "R32.7.1",
        "tensorrt": "8.2.1.8", "cuda": "10.2.300", "compute_capability": "5.3",
        "trtexec": ("trtexec --onnx=best.onnx --saveEngine=yolov8n_waste_fp16.engine "
                    "--fp16 --workspace=1024 --verbose"),
    },
    "training_host": {
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    },
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
(MANIFESTS / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("[manifest] manifest.json escrito", flush=True)

# --- 7. Upload final a HF Hub ------------------------------------------------
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=CFG["hf_repo"], private=True, exist_ok=True, repo_type="model")

uploads = [
    (BEST_PT, "best.pt"),
    (ONNX_PATH, "best.onnx"),
    (EXPORTS / "best.onnx", "exports/best.onnx"),  # compat con layout v1c
    (MANIFESTS / "eval_summary.json", "manifests/eval_summary.json"),
    (MANIFESTS / "manifest.json", "manifests/manifest.json"),
    (MANIFESTS / "gate3_onnx.json", "manifests/gate3_onnx.json"),
    (RUNS / "detect" / "train" / "results.csv", "runs/detect/train/results.csv"),
]
for local, remote in uploads:
    if Path(local).exists():
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=CFG["hf_repo"], repo_type="model")
        print(f"[hf] subido -> {remote}", flush=True)
    else:
        print(f"[hf] OMITIDO (no existe): {remote}", flush=True)

print(f"[DONE] mAP50={eval_summary['mAP50']:.4f} "
      f"mAP50_95={eval_summary['mAP50_95']:.4f} -- entrenamiento y upload completos.",
      flush=True)
