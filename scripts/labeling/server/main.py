"""embebidos3-label — FastAPI API para auto-labeling en RTX 3060 de Nicolas.

Replica el patron de scripts/server/ del Jetson Nano (FastAPI + SSE + job persistence)
adaptado a una maquina con GPU para correr Autodistill + Grounding DINO en WSL2.

Endpoints:
    GET    /health                  -> ping
    GET    /system                  -> nvidia-smi, VRAM, procesos GPU
    POST   /autolabel/job           -> dispara job (subprocess worker)
    GET    /jobs                    -> lista todos los jobs (activos + historicos)
    GET    /jobs/{job_id}/state     -> estado del job especifico
    GET    /jobs/{job_id}/logs      -> SSE stream de logs en vivo
    GET    /jobs/{job_id}/artifact  -> ZIP con labels YOLO (cuando el job termina)
    DELETE /jobs/{job_id}           -> cancela job (SIGTERM al worker)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# --- Configuracion via env vars (puede sobrescribirse en systemd unit) -----
STATE_DIR = Path(os.environ.get("EMBEBIDOS3_LABEL_STATE_DIR", "/run/embebidos3-label"))
JOBS_DIR = Path(os.environ.get("EMBEBIDOS3_LABEL_JOBS_DIR", "/var/lib/embebidos3-label/jobs"))
WORKER_MODULE = os.environ.get("EMBEBIDOS3_LABEL_WORKER", "scripts.labeling.server.worker")
PYTHON_BIN = os.environ.get("EMBEBIDOS3_LABEL_PYTHON", sys.executable)

STATE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_FILE = STATE_DIR / "jobs.json"

# --- Modelos Pydantic -------------------------------------------------------


class JobCreate(BaseModel):
    """Cuerpo de POST /autolabel/job."""

    input_url: str = Field(..., description="URL ZIP de imagenes o repo HF dataset.")
    ontology: dict[str, str] = Field(
        ..., description="Mapeo prompt -> clase YOLO (ej: 'plastic bottle' -> 'plastic')."
    )
    conf: float = Field(0.25, ge=0.05, le=0.9, description="Confidence threshold.")
    model_type: str = Field("tiny", description="GroundingDINO variant: tiny|base.")
    hf_dataset_repo: str | None = Field(
        None, description="Repo HF destino para subir labels (ej: mitgar14/embebidos3-labels)."
    )
    hf_dataset_path: str = Field(
        "batch1", description="Subpath dentro del repo HF (ej: batch1, batch2)."
    )


class JobState(BaseModel):
    job_id: str
    state: str  # queued | running | done | failed | cancelled
    pid: int | None = None
    log_path: str
    artifact_path: str | None = None
    created_at: str
    finished_at: str | None = None
    request: dict | None = None
    error: str | None = None


# --- Persistencia ----------------------------------------------------------


def _load_jobs() -> dict[str, dict]:
    if not JOBS_FILE.exists():
        return {}
    try:
        return json.loads(JOBS_FILE.read_text())
    except Exception:
        return {}


def _save_jobs(jobs: dict[str, dict]) -> None:
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2))
    tmp.replace(JOBS_FILE)


def _update_job(job_id: str, **patch) -> dict:
    jobs = _load_jobs()
    job = jobs.get(job_id, {})
    job.update(patch)
    jobs[job_id] = job
    _save_jobs(jobs)
    return job


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Recovery al startup: detecta jobs cuyo PID ya no existe ---------------


def _recover_jobs() -> None:
    jobs = _load_jobs()
    changed = False
    for job_id, job in jobs.items():
        if job.get("state") == "running" and (pid := job.get("pid")):
            try:
                os.kill(pid, 0)  # PID still alive?
            except (ProcessLookupError, PermissionError):
                # Worker murio sin avisar; marcamos failed con timestamp
                job["state"] = "failed"
                job["error"] = "worker_pid_disappeared"
                job["finished_at"] = _now_iso()
                changed = True
    if changed:
        _save_jobs(jobs)


# --- FastAPI app ------------------------------------------------------------

app = FastAPI(title="embebidos3-label", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    _recover_jobs()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "state_dir": str(STATE_DIR), "jobs_dir": str(JOBS_DIR)}


@app.get("/system")
def system() -> dict:
    """Estado de la GPU para que Nicolas + mitgar14 vean que esta corriendo."""
    nv_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        result = subprocess.run(
            [
                nv_smi,
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_line = result.stdout.strip()
        name, mem_total, mem_used, mem_free, util, temp = [
            x.strip() for x in gpu_line.split(",")
        ]
        return {
            "gpu_name": name,
            "vram_total_mb": int(mem_total),
            "vram_used_mb": int(mem_used),
            "vram_free_mb": int(mem_free),
            "gpu_util_pct": int(util),
            "temp_c": int(temp),
            "active_jobs": sum(1 for j in _load_jobs().values() if j.get("state") == "running"),
        }
    except Exception as e:
        return {"error": f"nvidia-smi failed: {e!r}"}


@app.post("/autolabel/job", response_model=JobState)
def create_job(body: JobCreate) -> JobState:
    # Reglas de concurrencia: solo 1 job activo a la vez (RTX 3060 6 GB no aguanta 2 GroundingDINO).
    active = [j for j in _load_jobs().values() if j.get("state") == "running"]
    if active:
        raise HTTPException(
            status_code=409,
            detail={"error": "job_already_running", "active_job": active[0].get("job_id")},
        )

    job_id = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "worker.log"
    request_path = job_dir / "request.json"
    request_path.write_text(body.model_dump_json(indent=2))

    cmd = [
        PYTHON_BIN,
        "-m",
        WORKER_MODULE,
        "--job-id",
        job_id,
        "--job-dir",
        str(job_dir),
    ]
    env = {**os.environ}
    # Asegura CUDA visible y PATH con nvidia-smi
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PATH"] = env.get("PATH", "") + ":/usr/lib/wsl/lib"

    log_fh = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path(__file__).resolve().parents[3],  # repo root
    )

    job = _update_job(
        job_id,
        job_id=job_id,
        state="running",
        pid=proc.pid,
        log_path=str(log_path),
        artifact_path=None,
        created_at=_now_iso(),
        finished_at=None,
        request=body.model_dump(),
        error=None,
    )
    return JobState(**job)


@app.get("/jobs")
def list_jobs() -> dict[str, list[dict]]:
    jobs = list(_load_jobs().values())
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"jobs": jobs}


@app.get("/jobs/{job_id}/state", response_model=JobState)
def job_state(job_id: str) -> JobState:
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job_not_found")
    job = jobs[job_id]
    # Refrescar estado si el PID ya murio
    if job.get("state") == "running" and (pid := job.get("pid")):
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            # Verificar si hay marker file done/failed
            done_marker = JOBS_DIR / job_id / "DONE"
            failed_marker = JOBS_DIR / job_id / "FAILED"
            if done_marker.exists():
                job = _update_job(
                    job_id,
                    state="done",
                    finished_at=_now_iso(),
                    artifact_path=str(JOBS_DIR / job_id / "labels.zip"),
                )
            elif failed_marker.exists():
                job = _update_job(
                    job_id,
                    state="failed",
                    finished_at=_now_iso(),
                    error=failed_marker.read_text().strip()[:500],
                )
            else:
                job = _update_job(
                    job_id,
                    state="failed",
                    finished_at=_now_iso(),
                    error="worker_pid_disappeared",
                )
    return JobState(**job)


@app.get("/jobs/{job_id}/logs")
def stream_logs(job_id: str) -> StreamingResponse:
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job_not_found")
    log_path = Path(jobs[job_id]["log_path"])

    async def gen():
        # Tail-follow estilo SSE
        with log_path.open("r") as f:
            # Primero, todo el contenido existente
            for line in f:
                yield f"data: {line.rstrip()}\n\n"
            # Luego, tail nuevos lines mientras el job corra
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    job = _load_jobs().get(job_id, {})
                    if job.get("state") not in ("running", "queued"):
                        yield f"event: end\ndata: {job.get('state', 'unknown')}\n\n"
                        break
                    await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/artifact")
def get_artifact(job_id: str) -> FileResponse:
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job_not_found")
    job = jobs[job_id]
    if job.get("state") != "done":
        raise HTTPException(
            status_code=409,
            detail={"error": "job_not_done", "state": job.get("state")},
        )
    artifact = Path(job.get("artifact_path") or (JOBS_DIR / job_id / "labels.zip"))
    if not artifact.exists():
        raise HTTPException(status_code=404, detail="artifact_missing")
    return FileResponse(
        path=artifact,
        media_type="application/zip",
        filename=f"labels-{job_id}.zip",
    )


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> JSONResponse:
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job_not_found")
    job = jobs[job_id]
    if job.get("state") != "running":
        return JSONResponse({"ok": False, "reason": "not_running", "state": job.get("state")})
    pid = job.get("pid")
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    _update_job(
        job_id,
        state="cancelled",
        finished_at=_now_iso(),
        error="cancelled_by_user",
    )
    return JSONResponse({"ok": True, "job_id": job_id})
