"""validate_engine.py — carga el engine en TRT context separado y corre 3 imágenes
de test_images/. Pasa si >= 2 imágenes producen >= 1 detección con conf > 0,3.

Exit 0 OK, 1 falla, 2 error de carga.

Usage: validate_engine.py <engine_path>
"""
import sys
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

sys.path.insert(0, str(Path(__file__).parent))
from nano_correctness import letterbox, postprocess, IMGSZ, CLASSES, CONF_TH, NMS_TH

TEST_IMAGES = sorted(glob.glob("/home/jetson/embebidos-3/test_images/*.jpg"))[:3]
MIN_PASS = 2


def run_engine(engine_path):
    cuda.init()
    ctx_cu = cuda.Device(0).make_context()
    try:
        ctx_cu.push()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None
        trt_ctx = engine.create_execution_context()
        bindings = []
        host_in = host_out = None
        dev_in = dev_out = None
        out_shape = None
        for i in range(engine.num_bindings):
            shape = tuple(engine.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            h = cuda.pagelocked_empty(size, dtype=dtype)
            d = cuda.mem_alloc(h.nbytes)
            bindings.append(int(d))
            if engine.binding_is_input(i):
                host_in = h; dev_in = d
            else:
                host_out = h; dev_out = d; out_shape = shape
        stream = cuda.Stream()

        results = []
        for img_path in TEST_IMAGES:
            img = cv2.imread(img_path)
            if img is None:
                results.append({"image": img_path, "detections": 0, "error": "imdecode fail"})
                continue
            oh, ow = img.shape[:2]
            lb, r, dx, dy = letterbox(img, IMGSZ)
            rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
            inp = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            np.copyto(host_in, inp.ravel())
            cuda.memcpy_htod_async(dev_in, host_in, stream)
            trt_ctx.execute_async_v2(bindings, stream.handle)
            cuda.memcpy_dtoh_async(host_out, dev_out, stream)
            stream.synchronize()
            raw = host_out.reshape(out_shape)
            dets = postprocess(raw, (r, dx, dy), (ow, oh))
            good = [d for d in dets if d["conf"] > 0.3]
            results.append({"image": img_path, "detections": len(good)})

        # cleanup
        stream = None
        dev_out = None; host_out = None
        dev_in = None; host_in = None
        trt_ctx = None
        engine = None
        runtime = None
        ctx_cu.pop()
        return results
    finally:
        try: ctx_cu.detach()
        except Exception: pass


def main():
    if len(sys.argv) < 2:
        print("usage: validate_engine.py <engine_path>", file=sys.stderr)
        sys.exit(2)
    engine_path = sys.argv[1]
    if not Path(engine_path).exists():
        print("engine no existe: {}".format(engine_path), file=sys.stderr)
        sys.exit(2)
    results = run_engine(engine_path)
    if results is None:
        print("deserialize_cuda_engine retornó None", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"validation": results}, indent=2))
    passed = sum(1 for r in results if r["detections"] > 0)
    if passed >= MIN_PASS:
        print("PASS ({}/{} imágenes con detecciones)".format(passed, len(results)),
              file=sys.stderr)
        sys.exit(0)
    else:
        print("FAIL ({}/{} imágenes con detecciones, requeridas {})".format(
            passed, len(results), MIN_PASS), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
