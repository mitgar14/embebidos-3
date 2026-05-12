"""
bench_jetson.py — Harness comparativo dual-track (corre EN el Jetson Nano B01).

Compara los dos backends del MVP en condiciones idénticas:
    Track A: SSD MobileNet v2 FPNLite 320x320 → TFLite INT8 (CPU + XNNPACK + NEON SIMD ARM Cortex-A57)
    Track B: YOLOv8n         416x416 → TensorRT FP16 (GPU Maxwell, sin tensor cores INT8)

NO es comparación "INT8 vs FP16" sino comparación de BACKEND HARDWARE
(CPU+SIMD vs GPU+CUDA). Maxwell carece de unidades INT8 VNNI (Turing/Ampere las introducen),
por eso TRT INT8 no acelera GPU pero TFLite INT8 sí acelera CPU vía instrucciones NEON.

Reporta: latencia p50/p95/p99, FPS sostenido, RAM peak (vía tegrastats paralelo).

Expectativas según literatura (Jetson Nano B01 JetPack 4.6.x, mismo hardware):
    Track A — TFLite INT8 + XNNPACK 4 hilos (CPU):
        - SSD MobileNet v2 320 INT8: ~70-90 ms/frame (Tobiasz 2023, NobuoTsukamoto)
        - SSD MV2 FPNLite 320 INT8: ~80-105 ms/frame (~10 ms overhead FPN)
        - FPS esperado: 11-14
    Track B — TensorRT FP16 (GPU Maxwell):
        - YOLOv8n 416 FP16 (Nature 2024 Tabla 4): 30 FPS inference-only
        - Pipeline real con preprocess + NMS NumPy: 18-22 FPS efectivos
        - Threshold viabilidad MVP: ≥10 FPS sostenidos
    Reference Track B sin NMS plugin (EfficientNMS_TRT broken Maxwell, NVIDIA/TensorRT#1538):
        - NMS en CPU NumPy añade 1-3 ms overhead. Usado por defecto en este script.

Uso (en el Nano, con tegrastats corriendo en otra terminal):
    sudo nvpmodel -m 0   # MAXN 10 W
    sudo jetson_clocks
    tegrastats --interval 500 --logfile bench.log &
    python3 bench_jetson.py --model_a detect_int8.tflite --model_b yolov8n_waste_fp16.engine

Dependencias en el Nano:
    - tensorflow 2.5.0+nv21.8 (wheel oficial NVIDIA con tf.lite)
    - tensorrt 8.0.1 / 8.2.1 (incluido en JetPack 4.6.x)
    - pycuda (instalar: pip3 install pycuda)
    - opencv-python (incluido), numpy

Dataset bench: dummy random 720x1280 BGR (próximo: dataset real de capturas
con la cámara UVC en su ángulo diagonal final sobre la banda transportadora).
"""
import argparse
import time
from abc import ABC, abstractmethod

import numpy as np


# ============================================================
# Interfaz comun
# ============================================================
class Detector(ABC):
    name = "abstract"

    @abstractmethod
    def detect(self, frame_bgr) -> list:
        """Retorna lista de [x1, y1, x2, y2, score, class_id]."""


# ============================================================
# Track A: SSD MobileNet v2 FPNLite TFLite INT8 (CPU + XNNPACK)
# ============================================================
class TFLiteSSD(Detector):
    name = "TFLite-SSD-INT8"

    def __init__(self, model_path, num_threads=4):
        # Usar tf.lite del wheel oficial NVIDIA en Jetson; tflite_runtime tambien sirve
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self.it = Interpreter(model_path=model_path, num_threads=num_threads)
        self.it.allocate_tensors()
        self.in_det = self.it.get_input_details()[0]
        out = self.it.get_output_details()
        # TFLite_Detection_PostProcess output order: boxes, classes, scores, num_detections
        self.out_boxes = out[0]['index']
        self.out_classes = out[1]['index']
        self.out_scores = out[2]['index']
        self.out_num = out[3]['index']
        self.input_shape = self.in_det['shape'][1:3]   # H, W

    def detect(self, bgr, conf_thr=0.5):
        import cv2
        h, w = self.input_shape
        img = cv2.resize(bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img, axis=0).astype(self.in_det['dtype'])
        self.it.set_tensor(self.in_det['index'], inp)
        self.it.invoke()
        boxes = self.it.get_tensor(self.out_boxes)[0]      # [N, 4] (ymin xmin ymax xmax) normalized
        classes = self.it.get_tensor(self.out_classes)[0]  # [N]
        scores = self.it.get_tensor(self.out_scores)[0]    # [N]
        num = int(self.it.get_tensor(self.out_num)[0])
        H, W = bgr.shape[:2]
        results = []
        for i in range(num):
            if scores[i] < conf_thr:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            results.append([
                int(xmin * W), int(ymin * H), int(xmax * W), int(ymax * H),
                float(scores[i]), int(classes[i]),
            ])
        return results


# ============================================================
# Track B: YOLOv8n TensorRT FP16 (GPU Maxwell)
# ============================================================
class YoloV8TRT(Detector):
    name = "YOLOv8n-TRT-FP16"

    def __init__(self, engine_path, input_shape=(1, 3, 416, 416)):
        import pycuda.autoinit  # noqa: inicializa contexto CUDA
        import pycuda.driver as cuda
        import tensorrt as trt
        self.cuda = cuda
        self.input_shape = input_shape
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            slot = {"host": host_mem, "device": dev_mem, "shape": shape}
            (self.inputs if self.engine.binding_is_input(binding) else self.outputs).append(slot)

    def _decode_yolov8(self, raw, orig_h, orig_w, conf_thr=0.25, iou_thr=0.45):
        """Decode + NMS NumPy. raw shape esperado: [1, 4+nc, N] o [4+nc, N] aplanado."""
        # Reshape segun output (asume single output [1, 7, anchors] para 3 clases)
        nc = 3
        n_ch = 4 + nc
        pred = raw.reshape(n_ch, -1).T   # [N, 7]
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(len(cls_ids)), cls_ids]
        m = confs > conf_thr
        if not m.any():
            return []
        boxes_xywh, confs, cls_ids = boxes_xywh[m], confs[m], cls_ids[m]
        # xywh -> xyxy
        b = np.empty_like(boxes_xywh)
        b[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        b[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        b[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        b[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        # NMS por clase
        sx, sy = orig_w / self.input_shape[3], orig_h / self.input_shape[2]
        out = []
        for cls in np.unique(cls_ids):
            idx = np.where(cls_ids == cls)[0]
            bb, ss = b[idx], confs[idx]
            order = ss.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(idx[i])
                xx1 = np.maximum(bb[i, 0], bb[order[1:], 0])
                yy1 = np.maximum(bb[i, 1], bb[order[1:], 1])
                xx2 = np.minimum(bb[i, 2], bb[order[1:], 2])
                yy2 = np.minimum(bb[i, 3], bb[order[1:], 3])
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                ai = (bb[i, 2] - bb[i, 0]) * (bb[i, 3] - bb[i, 1])
                ar = (bb[order[1:], 2] - bb[order[1:], 0]) * (bb[order[1:], 3] - bb[order[1:], 1])
                iou = inter / (ai + ar - inter + 1e-6)
                order = order[np.where(iou <= iou_thr)[0] + 1]
            for k in keep:
                out.append([
                    int(b[k, 0] * sx), int(b[k, 1] * sy),
                    int(b[k, 2] * sx), int(b[k, 3] * sy),
                    float(confs[k]), int(cls_ids[k]),
                ])
        return out

    def detect(self, bgr, conf_thr=0.25, iou_thr=0.45):
        import cv2
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.resize(bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[None])  # [1, 3, H, W]
        np.copyto(self.inputs[0]["host"], img.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for o in self.outputs:
            self.cuda.memcpy_dtoh_async(o["host"], o["device"], self.stream)
        self.stream.synchronize()
        return self._decode_yolov8(self.outputs[0]["host"], bgr.shape[0], bgr.shape[1], conf_thr, iou_thr)


# ============================================================
# Bench loop
# ============================================================
def bench(detector, n_warm=50, n_iter=200, frame_shape=(720, 1280, 3)):
    dummy = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
    for _ in range(n_warm):
        detector.detect(dummy)
    lat = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        detector.detect(dummy)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.asarray(lat)
    return {
        "name": detector.name,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(np.mean(lat)),
        "std_ms": float(np.std(lat)),
        "fps": float(1000.0 / np.mean(lat)),
        "n_iter": n_iter,
    }


def fmt(r):
    return (
        f"{r['name']:25s}  "
        f"p50={r['p50_ms']:6.1f}ms  "
        f"p95={r['p95_ms']:6.1f}ms  "
        f"p99={r['p99_ms']:6.1f}ms  "
        f"mean={r['mean_ms']:6.1f}±{r['std_ms']:.1f}ms  "
        f"FPS={r['fps']:5.1f}"
    )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", help="Ruta detect_int8.tflite (Track A)")
    ap.add_argument("--model_b", help="Ruta yolov8n_waste_fp16.engine (Track B)")
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--n_warm", type=int, default=50)
    args = ap.parse_args()

    print("=" * 80)
    print(" BENCH JETSON — DUAL-TRACK")
    print(f"  n_warm={args.n_warm}  n_iter={args.n_iter}")
    print("=" * 80)

    if args.model_a:
        print("\n[Track A] Cargando TFLite SSD INT8...")
        det_a = TFLiteSSD(args.model_a, num_threads=4)
        r_a = bench(det_a, args.n_warm, args.n_iter)
        print(fmt(r_a))

    if args.model_b:
        print("\n[Track B] Cargando YOLOv8n TensorRT FP16...")
        det_b = YoloV8TRT(args.model_b)
        r_b = bench(det_b, args.n_warm, args.n_iter)
        print(fmt(r_b))

    print("\n" + "=" * 80)
    print(" Recordatorio: tegrastats --interval 500 corre en otra terminal para RAM/temp")
    print("=" * 80)


if __name__ == "__main__":
    main()
