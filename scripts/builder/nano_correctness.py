#!/usr/bin/env python3
"""Correctness check end-to-end del engine TRT FP16 sin pycuda.

Flujo:
  1. preprocess imagen local (letterbox 416, RGB, /255, NCHW float32) -> input.bin
  2. scp input.bin -> Nano
  3. ssh nano "trtexec --loadEngine ... --loadInputs=images:input.bin --exportOutput=output.json"
  4. scp output.json <- Nano
  5. parse output -> decode YOLOv8 -> cv2.dnn.NMSBoxes -> dibujar bboxes

Uso:
  uv run --with opencv-python --with numpy python scripts/nano_correctness.py <test_image.jpg>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["glass", "paper", "plastic", "cardboard"]
COLORS = [(0, 255, 0), (255, 200, 0), (0, 100, 255), (19, 69, 139)]
IMGSZ = 416
CONF_TH = 0.25
NMS_TH = 0.45

NANO_HOST = "nano"
NANO_ENGINE = "/home/jetson/embebidos-3/engines/best_fp16.engine"
NANO_TMP = "/tmp/embebidos3"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def letterbox(img: np.ndarray, size: int = IMGSZ):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx = (size - nw) // 2
    dy = (size - nh) // 2
    pad = np.full((size, size, 3), 114, dtype=np.uint8)
    pad[dy:dy + nh, dx:dx + nw] = resized
    return pad, r, dx, dy


def preprocess_to_bin(img_bgr: np.ndarray, out_path: Path):
    lb, r, dx, dy = letterbox(img_bgr, IMGSZ)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    arr = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    arr = np.ascontiguousarray(arr)
    arr.tofile(str(out_path))
    return r, dx, dy, arr.shape


def ssh(cmd):
    return subprocess.run(["ssh", NANO_HOST, cmd], capture_output=True, text=True, check=False)


def scp(src, dst):
    return subprocess.run(["scp", src, dst], capture_output=True, text=True, check=False)


def parse_trtexec_output(json_path: Path) -> np.ndarray:
    """trtexec --exportOutput dumpa un JSON con shape: [{name, dimensions, values: [..]}]"""
    data = json.loads(json_path.read_text())
    out = data[0]
    dims = out["dimensions"]
    if isinstance(dims, str):
        dims = [int(x) for x in dims.split("x")]
    arr = np.array(out["values"], dtype=np.float32).reshape(dims)
    return arr


def postprocess(raw: np.ndarray, scale_info, orig_wh):
    r, dx, dy = scale_info
    ow, oh = orig_wh
    pred = raw[0].T  # (anchors, 4+nc)
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:4 + len(CLASSES)]
    cls_ids = cls_scores.argmax(1)
    confs = cls_scores.max(1)
    mask = confs >= CONF_TH
    if not mask.any():
        return []
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    rects = np.stack([x1, y1, boxes_xywh[:, 2], boxes_xywh[:, 3]], axis=1)
    idx = cv2.dnn.NMSBoxes(rects.tolist(), confs.tolist(), CONF_TH, NMS_TH)
    if len(idx) == 0:
        return []
    idx = np.array(idx).flatten()
    detections = []
    for i in idx:
        x, y, w, h = rects[i]
        ox1 = max(0.0, (x - dx) / r)
        oy1 = max(0.0, (y - dy) / r)
        ox2 = min(float(ow), (x + w - dx) / r)
        oy2 = min(float(oh), (y + h - dy) / r)
        detections.append({
            "x1": float(ox1), "y1": float(oy1),
            "x2": float(ox2), "y2": float(oy2),
            "conf": float(confs[i]),
            "cls": int(cls_ids[i]),
            "cls_name": CLASSES[int(cls_ids[i])],
        })
    return detections


def draw(img: np.ndarray, detections: list) -> np.ndarray:
    out = img.copy()
    for d in detections:
        color = COLORS[d["cls"]]
        p1 = (int(d["x1"]), int(d["y1"]))
        p2 = (int(d["x2"]), int(d["y2"]))
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{d['cls_name']} {d['conf']:.2f}"
        cv2.putText(out, label, (p1[0], max(0, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="ruta local a imagen .jpg")
    parser.add_argument("--out", default="runs/nano_correctness")
    args = parser.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"no se pudo leer {img_path}")
    oh, ow = img.shape[:2]

    tmp = Path("tmp_correctness")
    tmp.mkdir(exist_ok=True)
    bin_path = tmp / "input.bin"
    json_local = tmp / "output.json"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] preprocess local: {img_path.name} ({ow}x{oh})")
    r, dx, dy, shape = preprocess_to_bin(img, bin_path)
    print(f"      letterbox r={r:.3f} dx={dx} dy={dy} shape={shape} bytes={bin_path.stat().st_size}")

    print(f"[2/5] scp input -> {NANO_HOST}:{NANO_TMP}/input.bin")
    ssh(f"mkdir -p {NANO_TMP}")
    cp = scp(str(bin_path), f"{NANO_HOST}:{NANO_TMP}/input.bin")
    if cp.returncode != 0:
        raise SystemExit(f"scp falló: {cp.stderr}")

    print(f"[3/5] ssh trtexec --loadEngine --loadInputs --exportOutput")
    cmd = (
        f"{TRTEXEC} --loadEngine={NANO_ENGINE} "
        f"--loadInputs=images:{NANO_TMP}/input.bin "
        f"--exportOutput={NANO_TMP}/output.json "
        f"--iterations=1 --warmUp=0 --duration=0 --avgRuns=1 2>&1 | tail -5"
    )
    r2 = ssh(cmd)
    print(r2.stdout)
    if "PASSED" not in r2.stdout and "successfully" not in r2.stdout.lower():
        print("(warning: PASSED no detectado en output; revisar)")

    print(f"[4/5] scp output.json <- {NANO_HOST}")
    cp = scp(f"{NANO_HOST}:{NANO_TMP}/output.json", str(json_local))
    if cp.returncode != 0:
        raise SystemExit(f"scp falló: {cp.stderr}")

    print(f"[5/5] parse + decode + NMS")
    raw = parse_trtexec_output(json_local)
    print(f"      output shape: {raw.shape}, min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f}")
    if np.isnan(raw).any():
        raise SystemExit("output contiene NaN — engine roto")
    dets = postprocess(raw, (r, dx, dy), (ow, oh))
    print(f"      detections: {len(dets)}")
    for d in dets:
        print(f"        {d['cls_name']:8s} conf={d['conf']:.3f} bbox=({d['x1']:.0f},{d['y1']:.0f})->({d['x2']:.0f},{d['y2']:.0f})")

    out_img = draw(img, dets)
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), out_img)
    print(f"\nannotated -> {out_path}")


if __name__ == "__main__":
    main()
