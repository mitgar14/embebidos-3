#!/usr/bin/env python3
"""Captura de cámara local del Nano (/dev/video0) por GStreamer para el server.

Encapsula la captura de la Logitech C920 en `/dev/video0` y el thread lector que
encola frames `np.ndarray` BGR en la cola del TRTWorker (modo local de Fase 6).

Diseño (investigación 2026-05-25, 06-RESEARCH.md / 06-CONTEXT.md):
  - Pipeline HW `nvv4l2decoder mjpeg=1` (decode en el VIC/NVDEC, unidad distinta
    de los CUDA cores de TRT, coexisten en paralelo real), con fallback automático
    al pipeline CPU `jpegdec` si `cv2.VideoCapture` no abre el HW.
  - El frame de salida es 640x480 BGR NATIVO: el pipeline NO redimensiona a 416
    (no se fuerza width/height en `nvvidconv`), para no aplastar el 4:3 a 1:1.
    El TRTWorker letterboxea internamente (640x480 -> 416 para inferir) igual que
    en el modo remoto, así el video conserva su relación de aspecto real.
  - Captura en un thread daemon `cam-reader` con `cap.read()` bloqueante, NO con
    callbacks GLib (`emit-signals`): bajo asyncio el callback llega cada ~25 s por
    contención del GIL (Pitfall 6 de la investigación).
  - Drop silencioso: `put_nowait` + `except queue.Full: pass`, se descarta el frame
    si la inferencia va más lenta que la cámara (evita backlog y crecimiento de RAM).
  - Cierre: `cap.release()` + pausa ~1 s antes de terminar el thread, para que
    v4l2src libere `/dev/video0`; sin la pausa un re-open inmediato falla con
    "device busy" (Pitfall 7).

Sin dependencias nuevas: usa cv2, threading, queue, time, subprocess (ya en el
server). Compatible Python 3.6.9 (sin walrus, sin f-string debug, subprocess sin
capture_output=/text=).
"""
import queue
import subprocess
import threading
import time

import cv2


# ---------- Constructores de pipeline GStreamer ------------------------------
# Strings verificados por la investigación (NVIDIA con C920/C930, OpenCV 4.1.1):
#   - El frame de salida es 640x480 BGR NATIVO (sin squish; el worker lo lleva a
#     416 con _letterbox). Por eso `nvvidconv` NO lleva width=416,height=416.
#   - `nvvidconv` solo emite BGRx, nunca BGR directo (Pitfall 2): de ahí la cadena
#     nvvidconv ! ...BGRx ! videoconvert ! ...BGR.
#   - `io-mode=2` fuerza MJPG en v4l2src (Pitfall 1): sin él negocia YUYV y se
#     hunde el FPS. Solo aplica al pipeline HW.
#   - `appsink drop=1 max-buffers=1`: se queda con el frame más reciente; sin
#     drop=1 incluso `cap.release()` puede colgar (Pitfall 5).

def _pipeline_hw(device="/dev/video0"):
    """Pipeline HW (nvv4l2decoder mjpeg=1). Frame de salida 640x480 BGR nativo."""
    return (
        "v4l2src device={dev} io-mode=2 ! "
        "image/jpeg,width=640,height=480,framerate=15/1 ! "
        "nvv4l2decoder mjpeg=1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1"
    ).format(dev=device)


def _pipeline_cpu(device="/dev/video0"):
    """Pipeline CPU de fallback (jpegdec, sin io-mode=2). 640x480 BGR nativo."""
    return (
        "v4l2src device={dev} ! "
        "image/jpeg,width=640,height=480,framerate=15/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1"
    ).format(dev=device)


class CameraCapture(object):
    """Captura GStreamer de /dev/video0 con thread lector hacia la cola del worker.

    El thread `cam-reader` lee `cap.read()` y encola items de modo local en
    `target_queue` (la `in_q` del TRTWorker). Cada item tiene la aridad 5 que el
    worker espera: (frame_ndarray, capture_ts_ms, seq_local, None, None). El primer
    elemento es un np.ndarray BGR 640x480 nativo; loop/future son None porque en
    modo local el resultado no se espera con await (sale por on_local_result).
    """

    def __init__(self, target_queue, device="/dev/video0", release_pause_s=1.0):
        self.target_queue = target_queue
        self.device = device
        self.release_pause_s = release_pause_s
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._seq = 0
        self._active_pipeline = None  # "hw" | "cpu" | None

    # ---------- API pública --------------------------------------------------
    def start(self):
        """Arranca la captura. Idempotente: si ya corre, no-op.

        Best-effort jetson_clocks (paso de latencia del VIC), abre la cámara con
        fallback HW->CPU, y lanza el thread daemon cam-reader. Si la apertura
        falla, propaga RuntimeError("camera_open_failed") y deja _running=False.
        """
        if self.is_running():
            return
        self._maybe_jetson_clocks()
        cap = self._open()  # puede lanzar RuntimeError("camera_open_failed")
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, args=(cap,), name="cam-reader", daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Detiene la captura y espera a que el thread libere la cámara. Idempotente."""
        self._running = False
        thread = self._thread
        if thread is not None:
            # join >= release_pause_s + margen: el thread hace release() + sleep
            # antes de terminar, hay que esperar a que complete esa pausa.
            thread.join(timeout=self.release_pause_s + 3.0)
        self._thread = None

    def is_running(self):
        return (
            self._running
            and self._thread is not None
            and self._thread.is_alive()
        )

    def active_pipeline(self):
        """Devuelve "hw" | "cpu" | None (qué pipeline quedó activo)."""
        return self._active_pipeline

    # ---------- Internos -----------------------------------------------------
    def _next_seq(self):
        with self._lock:
            self._seq += 1
            return self._seq

    def _make_item(self, frame):
        """Item de modo local que TRTWorker.submit espera (aridad 5).

        frame es el np.ndarray BGR 640x480 nativo; capture_ts_ms en ms; loop y
        future son None (el resultado sale por on_local_result, no por un future).
        """
        return (frame, time.time() * 1000.0, self._next_seq(), None, None)

    def _open(self):
        """Abre la cámara: intenta HW, cae a CPU. Lanza si ninguno abre."""
        cap = cv2.VideoCapture(_pipeline_hw(self.device), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self._active_pipeline = "hw"
            print("[cam-reader] pipeline activo: hw (nvv4l2decoder mjpeg=1)", flush=True)
            return cap
        # HW no abrió: liberar y reintentar con el pipeline CPU.
        cap.release()
        cap = cv2.VideoCapture(_pipeline_cpu(self.device), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self._active_pipeline = "cpu"
            print("[cam-reader] pipeline activo: cpu(fallback) (jpegdec)", flush=True)
            return cap
        cap.release()
        self._active_pipeline = None
        raise RuntimeError("camera_open_failed")

    def _reader_loop(self, cap):
        """Bucle del thread cam-reader: lee frames y los encola con drop silencioso."""
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                try:
                    self.target_queue.put_nowait(self._make_item(frame))
                except queue.Full:
                    # Drop silencioso: la inferencia va más lenta que la cámara;
                    # nos quedamos con los frames más recientes (decisión de la
                    # investigación: evita backlog y crecimiento de RAM).
                    pass
        finally:
            # Pitfall 7: liberar /dev/video0 y pausar ~1 s antes de terminar el
            # thread, para que un re-open posterior no falle con "device busy".
            cap.release()
            time.sleep(self.release_pause_s)

    def _maybe_jetson_clocks(self):
        """Sube los clocks del Nano best-effort (latencia del VIC en ahorro de energía).

        Decisión LOCKED de la investigación. Es best-effort: si jetson_clocks no
        está en sudoers o falla, el arranque continúa igual (el modo local funciona,
        solo con algo más de latencia). La regla de sudoers para jetson_clocks NO es
        obligatoria para el MVP.

        Python 3.6.9: subprocess.run SIN capture_output=/text= (usar stdout/stderr
        PIPE + universal_newlines=True).
        """
        try:
            subprocess.run(
                ["sudo", "jetson_clocks"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            # No interrumpir el arranque del modo local si jetson_clocks no corre.
            pass
