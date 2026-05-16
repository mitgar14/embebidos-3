#!/usr/bin/env python3
"""FastAPI + WebSocket server para inferencia YOLOv8 TRT FP16 en Jetson Nano.

Stack (research-code + research-web ronda 1 2026-05-14):
  - FastAPI + uvicorn ASGI (event loop async, WebSocket nativo)
  - 1 GPU worker thread con pycuda Context push/pop (thread-local)
  - asyncio.Queue(maxsize=2) backpressure (drop frames vs acumular lag)
  - cv2.dnn.NMSBoxes V0 (D26 ruta segura sm_53)
  - parche py3.6: asyncio.create_task = asyncio.ensure_future

Patrón blueprint: hasantavision/jetson-security-cam (único repo verificado con
mismo stack: Jetson Nano R32.7.1, Python 3.6, FastAPI, threading GPU worker).

Endpoints:
  GET  /        -> "OK"
  GET  /health  -> {gpu_util, temp_c, ram_mb, fps_avg, conf_th}
  WS   /ws      -> bidirectional:
                   client -> binary JPEG frame  OR  text {"type":"conf","value":0.3}
                   server -> JSON {"bboxes":[...], "t_infer_ms":...,
                                   "t_recv_ms":..., "ts":..., "seq":...}

Run:
  /home/jetson/.local/bin/uvicorn nano_server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import hf_rest
from nano_server_constants import (
    ACTIVE_ENGINE, IMGSZ, CLASSES, DEFAULT_CONF, DEFAULT_NMS,
    JOB_STATE_FILE, HEARTBEAT_STALE_SEC,
    ACTIVE_ENGINE_META, PREVIOUS_ENGINE, PREVIOUS_ENGINE_META, JOBS_LOGS_DIR,
)
from pid_utils import is_pid_alive as _is_pid_alive, check_cmdline as _check_cmdline
from recover_job_state import recover_job_state as _rjs

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,40}$")


# ---------- Parche Python 3.6 (hasantavision) ---------------------------------
if not hasattr(asyncio, "create_task"):
    asyncio.create_task = asyncio.ensure_future


# ---------- Config ------------------------------------------------------------
ENGINE_PATH = os.environ.get("ENGINE_PATH", str(ACTIVE_ENGINE))


# ---------- TRT Engine + GPU Worker -------------------------------------------
class TRTWorker(threading.Thread):
    """Single GPU worker thread. Único que toca el contexto CUDA."""

    def __init__(self, engine_path: str):
        super().__init__(daemon=True, name="trt-worker")
        self.engine_path = engine_path
        self.in_q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=2)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self.conf_th = DEFAULT_CONF
        self._lock = threading.Lock()
        self._fps_window = []  # timestamps últimos N frames procesados
        self._stats = {"total_processed": 0, "last_t_infer_ms": 0.0}
        self._swap_path: Optional[str] = None
        self._swap_event = threading.Event()
        # Engine state as instance attrs (not locals in run()) so _unload can destroy them.
        self._runtime = None
        self._engine = None
        self._trt_ctx = None
        self._stream = None
        self._host_in = self._host_out = None
        self._dev_in = self._dev_out = None
        self._bindings = []
        self._in_shape = self._out_shape = None

    def set_conf(self, v: float) -> None:
        with self._lock:
            self.conf_th = max(0.0, min(1.0, float(v)))

    def get_conf(self) -> float:
        with self._lock:
            return self.conf_th

    def get_stats(self) -> dict:
        with self._lock:
            now = time.perf_counter()
            self._fps_window = [t for t in self._fps_window if now - t < 1.0]
            return {
                "fps_1s": len(self._fps_window),
                "total_processed": self._stats["total_processed"],
                "last_t_infer_ms": self._stats["last_t_infer_ms"],
                "conf_th": self.conf_th,
            }

    def submit(self, item: tuple) -> bool:
        """item = (jpeg_bytes, client_ts_ms, seq, future_obj). True si encoló."""
        try:
            self.in_q.put_nowait(item)
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        self._stop.set()
        try:
            self.in_q.put_nowait(None)
        except queue.Full:
            pass

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    def request_swap(self, new_path: str) -> None:
        """Pide hot-swap a un engine nuevo. Llamado desde el handler HTTP."""
        with self._lock:
            self._swap_path = new_path
        self._swap_event.set()

    def _load_engine(self, path: str) -> None:
        """Carga engine + bindings desde path. Asume cu_ctx pushed.
        En caso de falla parcial, limpia con _unload_engine antes de propagar."""
        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        try:
            with open(path, "rb") as f:
                self._engine = self._runtime.deserialize_cuda_engine(f.read())
            self._trt_ctx = self._engine.create_execution_context()
            self._bindings = []
            for i in range(self._engine.num_bindings):
                shape = tuple(self._engine.get_binding_shape(i))
                dtype = trt.nptype(self._engine.get_binding_dtype(i))
                size = int(np.prod(shape))
                h = cuda.pagelocked_empty(size, dtype=dtype)
                d = cuda.mem_alloc(h.nbytes)
                self._bindings.append(int(d))
                if self._engine.binding_is_input(i):
                    self._host_in = h; self._dev_in = d; self._in_shape = shape
                else:
                    self._host_out = h; self._dev_out = d; self._out_shape = shape
            self._stream = cuda.Stream()
            print(f"[trt-worker] engine cargado desde {path}. in={self._in_shape} out={self._out_shape}", flush=True)
        except Exception:
            self._unload_engine()
            raise

    def _unload_engine(self) -> None:
        """Destruye engine + buffers. Asume cu_ctx pushed.
        Orden de destrucción (mnemon 881e5569): stream -> outputs -> inputs -> ctx -> engine -> runtime."""
        self._stream = None
        self._dev_out = None
        self._host_out = None
        self._dev_in = None
        self._host_in = None
        self._bindings = []
        self._trt_ctx = None
        self._engine = None
        self._runtime = None
        print("[trt-worker] engine descargado", flush=True)

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = IMGSZ / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dx = (IMGSZ - nw) // 2
        dy = (IMGSZ - nh) // 2
        pad = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
        pad[dy:dy + nh, dx:dx + nw] = resized
        return pad, r, dx, dy

    def _postprocess(self, raw: np.ndarray, scale_info, orig_wh) -> list:
        r, dx, dy = scale_info
        ow, oh = orig_wh
        pred = raw[0].T  # (anchors, 4+nc)
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:4 + len(CLASSES)]
        cls_ids = cls_scores.argmax(1)
        confs = cls_scores.max(1)
        conf_th = self.get_conf()
        mask = confs >= conf_th
        if not mask.any():
            return []
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        rects = np.stack([x1, y1, boxes_xywh[:, 2], boxes_xywh[:, 3]], axis=1)
        idx = cv2.dnn.NMSBoxes(rects.tolist(), confs.tolist(), conf_th, DEFAULT_NMS)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()
        dets = []
        for i in idx:
            x, y, w, h = rects[i]
            ox1 = max(0.0, (x - dx) / r)
            oy1 = max(0.0, (y - dy) / r)
            ox2 = min(float(ow), (x + w - dx) / r)
            oy2 = min(float(oh), (y + h - dy) / r)
            dets.append({
                "x1": float(ox1), "y1": float(oy1),
                "x2": float(ox2), "y2": float(oy2),
                "conf": float(confs[i]),
                "cls": int(cls_ids[i]),
                "cls_name": CLASSES[int(cls_ids[i])],
            })
        return dets

    def run(self) -> None:
        cuda.init()
        cu_ctx = cuda.Device(0).make_context()
        try:
            cu_ctx.push()
            self._load_engine(str(self.engine_path))
            cu_ctx.pop()
            self._ready.set()

            while not self._stop.is_set():
                # ¿hay swap pendiente?
                if self._swap_event.is_set():
                    self._swap_event.clear()
                    with self._lock:
                        new_path = self._swap_path
                    if new_path:
                        cu_ctx.push()
                        try:
                            self._unload_engine()
                            try:
                                self._load_engine(new_path)
                                self.engine_path = new_path
                            except Exception as e:
                                print(f"[trt-worker] FATAL: swap a {new_path} falló: {e}. Deteniendo worker.", flush=True)
                                self._stop.set()
                                break
                        finally:
                            cu_ctx.pop()

                # Timeout en in_q.get() para que el loop pueda chequear _swap_event
                # periódicamente aunque no lleguen frames.
                try:
                    item = self.in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break

                jpeg_bytes, client_ts_ms, seq, loop, future = item
                t_start = time.perf_counter()
                try:
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError("imdecode failed")
                    oh, ow = img.shape[:2]
                    lb, r, dx, dy = self._letterbox(img)
                    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
                    inp = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

                    cu_ctx.push()
                    try:
                        np.copyto(self._host_in, inp.ravel())
                        cuda.memcpy_htod_async(self._dev_in, self._host_in, self._stream)
                        self._trt_ctx.execute_async_v2(self._bindings, self._stream.handle)
                        cuda.memcpy_dtoh_async(self._host_out, self._dev_out, self._stream)
                        self._stream.synchronize()
                    finally:
                        cu_ctx.pop()

                    raw = self._host_out.reshape(self._out_shape)
                    dets = self._postprocess(raw, (r, dx, dy), (ow, oh))
                    t_end = time.perf_counter()
                    t_infer_ms = (t_end - t_start) * 1000.0

                    with self._lock:
                        self._fps_window.append(t_end)
                        self._stats["total_processed"] += 1
                        self._stats["last_t_infer_ms"] = t_infer_ms

                    result = {
                        "ok": True,
                        "bboxes": dets,
                        "t_infer_ms": round(t_infer_ms, 2),
                        "client_ts_ms": client_ts_ms,
                        "seq": seq,
                    }
                except Exception as e:
                    result = {"ok": False, "error": str(e), "seq": seq}

                try:
                    loop.call_soon_threadsafe(future.set_result, result)
                except Exception:
                    pass
        finally:
            try:
                cu_ctx.push()
                self._unload_engine()
                cu_ctx.pop()
            except Exception:
                pass
            try:
                cu_ctx.detach()
            except Exception:
                pass
            print("[trt-worker] stopped", flush=True)


# ---------- Helpers GPU/health -----------------------------------------------
def read_gpu_temp_c() -> Optional[float]:
    paths = [
        "/sys/class/thermal/thermal_zone1/temp",  # GPU típico en Nano
        "/sys/class/thermal/thermal_zone0/temp",
    ]
    for p in paths:
        try:
            with open(p) as f:
                v = int(f.read().strip())
            return v / 1000.0
        except Exception:
            continue
    return None


def read_ram_mb() -> dict:
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable"):
                    info[k] = int(v.strip().split()[0]) // 1024  # MB
    except Exception:
        pass
    return info


# ---------- FastAPI app -------------------------------------------------------
app = FastAPI(title="embebidos-3 nano inference server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
worker = TRTWorker(ENGINE_PATH)

_recovered_job_at_startup = None


@app.on_event("startup")
def _startup():
    global _recovered_job_at_startup
    worker.start()
    if not worker.wait_ready(60):
        raise RuntimeError("TRT worker no ready en 60s")
    _recovered_job_at_startup = _rjs()
    if _recovered_job_at_startup:
        print(f"[server] job recovery: status={_recovered_job_at_startup.get('status')} "
              f"job_id={_recovered_job_at_startup.get('job_id')}", flush=True)
    print(f"[server] engine listo. ws://0.0.0.0:8000/ws", flush=True)


@app.on_event("shutdown")
def _shutdown():
    worker.stop()
    worker.join(timeout=5)


@app.get("/", response_class=PlainTextResponse)
def root():
    return "embebidos-3 nano inference server\nWebSocket: ws://<host>:8000/ws\nHealth: GET /health\n"


@app.get("/health")
def health():
    stats = worker.get_stats()
    ram = read_ram_mb()
    return {
        "engine": str(ENGINE_PATH),
        "imgsz": IMGSZ,
        "classes": CLASSES,
        **stats,
        "gpu_temp_c": read_gpu_temp_c(),
        "ram_total_mb": ram.get("MemTotal"),
        "ram_available_mb": ram.get("MemAvailable"),
    }


def _read_engine_meta(meta_path):
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _read_active_job():
    if not JOB_STATE_FILE.exists():
        return None
    try:
        state = json.loads(JOB_STATE_FILE.read_text())
    except Exception:
        return None
    pid = state.get("pid")
    if not pid or not _is_pid_alive(pid) or not _check_cmdline(pid):
        return None  # huérfano, PID reuse, o PID muerto
    return state


@app.get("/model/state")
def model_state():
    """Devuelve el estado del modelo: no_model | ready | building | degraded."""
    active_meta = _read_engine_meta(ACTIVE_ENGINE_META)
    previous_meta = _read_engine_meta(PREVIOUS_ENGINE_META)
    active_job = _read_active_job()

    if active_job:
        return {
            "state": "building",
            "active_engine": active_meta,
            "previous_engine": previous_meta,
            "active_job": active_job,
        }

    if ACTIVE_ENGINE.exists() and active_meta:
        degraded = bool(active_meta.get("from_fallback", False))
        return {
            "state": "degraded" if degraded else "ready",
            "active_engine": active_meta,
            "previous_engine": previous_meta,
            "active_job": None,
        }

    return {
        "state": "no_model",
        "active_engine": None,
        "previous_engine": previous_meta,
        "active_job": None,
    }


# ---------- Job lifecycle endpoints (Fase E) ----------------------------------
class BuildRequest(BaseModel):
    force: bool = False
    workspace_mb: Optional[int] = None  # override env EMBEBIDOS3_TRTEXEC_WORKSPACE (defaults to 512)


def _generate_job_id():
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M")
    suffix = uuid.uuid4().hex[:6]
    return "{}-{}".format(ts, suffix)


@app.post("/model/build", status_code=202)
def model_build(req: BuildRequest = BuildRequest()):
    active = _read_active_job()
    if active:
        raise HTTPException(
            status_code=409,
            detail={"ok": False, "error": "build_in_progress",
                    "active_job_id": active.get("job_id")},
        )
    job_id = _generate_job_id()
    try:
        subprocess.run(
            ["sudo", "/usr/local/bin/embebidos3-builder-launch", job_id],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, {"ok": False, "error": "launch_failed",
                                  "stderr": e.stderr})
    return {
        "ok": True,
        "job_id": job_id,
        "monitor_url": "/jobs/{}".format(job_id),
        "logs_stream_url": "/jobs/{}/logs".format(job_id),
    }


@app.get("/jobs/active")
def jobs_active():
    return _read_active_job()


@app.get("/jobs/{job_id}")
def jobs_get(job_id: str):
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(422, {"ok": False, "error": "invalid_job_id"})
    active = _read_active_job()
    if active and active.get("job_id") == job_id:
        return active
    final = JOBS_LOGS_DIR / "{}.json".format(job_id)
    if final.exists():
        return json.loads(final.read_text())
    raise HTTPException(404, {"ok": False, "error": "job_not_found"})


@app.post("/model/rollback")
def model_rollback():
    if not PREVIOUS_ENGINE.exists():
        raise HTTPException(409, {"ok": False, "error": "no_previous_engine"})
    # swap inverso: usar nombres temp distintos para no colisionar con los meta paths reales
    tmp_active = ACTIVE_ENGINE.parent / (ACTIVE_ENGINE.name + ".swap_tmp")
    tmp_active_meta = ACTIVE_ENGINE_META.parent / (ACTIVE_ENGINE_META.name + ".swap_tmp")
    if ACTIVE_ENGINE.exists():
        ACTIVE_ENGINE.replace(tmp_active)
        if ACTIVE_ENGINE_META.exists():
            ACTIVE_ENGINE_META.replace(tmp_active_meta)
    PREVIOUS_ENGINE.replace(ACTIVE_ENGINE)
    if PREVIOUS_ENGINE_META.exists():
        PREVIOUS_ENGINE_META.replace(ACTIVE_ENGINE_META)
    if tmp_active.exists():
        tmp_active.replace(PREVIOUS_ENGINE)
        if tmp_active_meta.exists():
            tmp_active_meta.replace(PREVIOUS_ENGINE_META)
    # mark as degraded
    meta = _read_engine_meta(ACTIVE_ENGINE_META) or {}
    meta["from_fallback"] = True
    ACTIVE_ENGINE_META.write_text(json.dumps(meta, indent=2))
    # hot-reload worker
    worker.request_swap(str(ACTIVE_ENGINE))
    return {"ok": True, "phase": "rolled_back"}


@app.post("/model/check-updates")
def check_updates():
    current_meta = _read_engine_meta(ACTIVE_ENGINE_META) or {}
    current_rev = current_meta.get("hf_revision")
    current_onnx = current_meta.get("onnx_sha256")
    try:
        latest_rev = hf_rest.get_head_revision()
        latest_onnx = hf_rest.get_file_lfs_sha256("exports/best.onnx", revision=latest_rev)
    except Exception as e:
        raise HTTPException(503, {"ok": False, "error": "hf_unreachable", "detail": str(e)})
    same_revision = current_rev == latest_rev
    same_onnx = (
        bool(current_onnx) and bool(latest_onnx) and current_onnx == latest_onnx
    )
    # up_to_date refleja el CONTENIDO del modelo (ONNX): si el sha256 coincide,
    # el modelo cargado es el mismo aunque haya commits cosméticos nuevos en HF.
    return {
        "up_to_date": same_onnx,
        "same_revision": same_revision,
        "same_onnx": same_onnx,
        "has_engine": current_rev is not None,
        "current_revision": current_rev,
        "latest_revision": latest_rev,
        "current_onnx_sha256": current_onnx,
        "latest_onnx_sha256": latest_onnx,
    }


@app.delete("/jobs/{job_id}")
def jobs_cancel(job_id: str):
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(422, {"ok": False, "error": "invalid_job_id"})
    active = _read_active_job()
    if not active or active.get("job_id") != job_id:
        raise HTTPException(404, {"ok": False, "error": "job_not_active"})
    try:
        subprocess.run(
            ["sudo", "/bin/systemctl", "stop", "embebidos3-builder@{}.service".format(job_id)],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, {"ok": False, "error": "stop_failed", "stderr": e.stderr})
    return {"ok": True, "phase": "cancelling", "job_id": job_id}


@app.get("/jobs/{job_id}/logs")
def jobs_logs(job_id: str, follow: bool = True):
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(422, {"ok": False, "error": "invalid_job_id"})
    log = JOBS_LOGS_DIR / "{}.log".format(job_id)
    if not log.exists():
        raise HTTPException(404, {"ok": False, "error": "log_not_found"})

    def gen():
        with open(log, "r") as f:
            for line in f:
                yield "event: log\ndata: {}\n\n".format(json.dumps({"line": line.rstrip()}))
            if not follow:
                yield "event: done\ndata: {}\n\n".format(json.dumps({"phase": "eof"}))
                return
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    final = JOBS_LOGS_DIR / "{}.json".format(job_id)
                    if final.exists():
                        result = json.loads(final.read_text())
                        yield "event: done\ndata: {}\n\n".format(json.dumps(result))
                        return
                    time.sleep(0.25)
                    f.seek(where)
                else:
                    yield "event: log\ndata: {}\n\n".format(
                        json.dumps({"line": line.rstrip(), "ts": time.time()}))

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    seq = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                # Frame binario (JPEG)
                jpeg = msg["bytes"]
                t_recv_ms = time.time() * 1000.0
                seq += 1
                future = loop.create_future()
                ok = worker.submit((jpeg, t_recv_ms, seq, loop, future))
                if not ok:
                    await ws.send_text(json.dumps({"ok": False, "error": "queue_full", "seq": seq}))
                    continue
                result = await future
                result["t_recv_ms"] = round(t_recv_ms, 2)
                await ws.send_text(json.dumps(result))
            elif "text" in msg and msg["text"] is not None:
                # Mensaje de control JSON
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "conf":
                        worker.set_conf(ctrl["value"])
                        await ws.send_text(json.dumps({"ok": True, "type": "conf_ack", "value": worker.get_conf()}))
                    elif ctrl.get("type") == "ping":
                        await ws.send_text(json.dumps({"ok": True, "type": "pong", "ts": time.time() * 1000.0}))
                    else:
                        await ws.send_text(json.dumps({"ok": False, "error": "unknown_ctrl"}))
                except Exception as e:
                    await ws.send_text(json.dumps({"ok": False, "error": f"ctrl_parse: {e}"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"ok": False, "error": f"server: {e}"}))
        except Exception:
            pass  # cliente puede haber cerrado durante el envío
        finally:
            try:
                await ws.close()
            except Exception:
                pass


def _term(signum, frame):
    worker.stop()


signal.signal(signal.SIGTERM, _term)
signal.signal(signal.SIGINT, _term)
