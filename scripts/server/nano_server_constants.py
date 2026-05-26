"""Constantes compartidas entre nano_server.py, builder y helpers."""
import os
from pathlib import Path

# Rutas
ROOT = Path("/home/jetson/embebidos-3")
ENGINES_DIR = ROOT / "engines"
ONNX_DIR = ROOT / "onnx"
LOGS_DIR = ROOT / "logs"
JOBS_LOGS_DIR = LOGS_DIR / "jobs"
STAGING_DIR = ENGINES_DIR / ".staging"
PREVIOUS_DIR = ENGINES_DIR / ".previous"
TEST_IMAGES_DIR = ROOT / "test_images"

ACTIVE_ENGINE = ENGINES_DIR / "best_fp16.engine"
ACTIVE_ENGINE_META = ENGINES_DIR / "best_fp16.engine.meta.json"
ACTIVE_ENGINE_READY = ENGINES_DIR / "best_fp16.engine.ready"
PREVIOUS_ENGINE = PREVIOUS_DIR / "best_fp16.engine.old"
PREVIOUS_ENGINE_META = PREVIOUS_DIR / "best_fp16.engine.old.meta.json"
PREVIOUS_ENGINE_READY = PREVIOUS_DIR / "best_fp16.engine.old.ready"

# Histórico de engines: cada build exitoso archiva el anterior aquí
# (subdir <YYYYMMDDTHHMMSSZ>__<sha8>/ con engine + meta + manifest).
ENGINES_ARCHIVE_DIR = ROOT / "engines-archive"

# NOTA: en Fase A el systemd RuntimeDirectory=embebidos3 materializa /run/embebidos3/.
# En Linux moderno /var/run es symlink a /run, pero usamos /run/embebidos3 directo
# para ser explícitos y evitar sorpresas en otras plataformas.
RUNTIME_DIR = Path("/run/embebidos3")
JOB_STATE_FILE = RUNTIME_DIR / "job.json"
BUILDER_LOCK_FILE = RUNTIME_DIR / "builder.lock"

# HF Hub
# Configurable via env var EMBEBIDOS3_HF_REPO para soportar
# múltiples experimentos (v1, v1c, v1d, ...) sin redeploy de código.
HF_REPO = os.environ.get("EMBEBIDOS3_HF_REPO", "mitgar14/embebidos-3-models")
HF_REVISION_DEFAULT = "main"
ONNX_REMOTE_PATH = "exports/best.onnx"
MANIFEST_REMOTE_PATH = "manifests/manifest.json"
ENGINES_ARCHIVE_PREFIX = "engines-archive"

# Inferencia
IMGSZ = 416
CLASSES = ["glass", "paper", "plastic", "cardboard"]
DEFAULT_CONF = 0.25
DEFAULT_NMS = 0.45

# Builder
TRTEXEC_WORKSPACE_DEFAULT_MB = int(os.environ.get("EMBEBIDOS3_TRTEXEC_WORKSPACE", "512"))
TRTEXEC_TIMEOUT_SEC = 2400  # 40 min
HEARTBEAT_STALE_SEC = 120
