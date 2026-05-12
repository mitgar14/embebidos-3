"""
00_setup_roboflow.py

Sube el dataset arshnoor7389/garbage-classification-dataset al workspace
'embebidos3' de Roboflow como proyecto 'waste-3class' (object-detection).

Pre-requisitos:
- uv instalado (gestor de Python por defecto en este entorno)
- Variable de entorno ROBOFLOW_API_KEY (Settings -> Roboflow API en la UI)
- Dataset arshnoor7389 descargado en ./data/raw/arshnoor (Dataset/images/, Dataset/labels/, Dataset/data.yaml)

Ejecutar (PowerShell, desde la raiz del proyecto):
    $env:ROBOFLOW_API_KEY = "TU_API_KEY"
    uv run python scripts/00_setup_roboflow.py
"""
import os
import sys
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    sys.exit("Falta instalar roboflow: uv add roboflow")

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    sys.exit("Define ROBOFLOW_API_KEY (Settings -> Roboflow API en la UI)")

WORKSPACE = "embebidos3"
PROJECT_ID = "waste-3class"
DATASET_PATH = Path("./data/raw/arshnoor")

if not (DATASET_PATH / "Dataset" / "data.yaml").exists():
    sys.exit(
        f"No se encuentra {DATASET_PATH}/Dataset/data.yaml. "
        "Descarga primero arshnoor7389 con:\n"
        "  kaggle datasets download arshnoor7389/garbage-classification-dataset -p ./data/raw/\n"
        "  Expand-Archive ./data/raw/garbage-classification-dataset.zip ./data/raw/arshnoor"
    )

rf = Roboflow(api_key=API_KEY)
workspace = rf.workspace(WORKSPACE)

print(f"[+] Subiendo dataset (4,3 GB, ~30-60 min). Workspace={WORKSPACE}, project={PROJECT_ID}")
workspace.upload_dataset(
    dataset_path=str(DATASET_PATH / "Dataset"),
    project_name=PROJECT_ID,
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name="arshnoor7389-base",
    num_retries=2,
)
print("[OK] Upload completo. Siguiente paso: generar Version 1 en la UI con Modify Classes + Augmentations 5x.")
