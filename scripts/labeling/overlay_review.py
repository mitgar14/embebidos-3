"""Overlay de bboxes auto-generadas sobre las imagenes para revision visual rapida.

Genera:
  - PNGs individuales con bboxes dibujadas en datasets/waste-3class-batch1-auto/overlay/
  - Mosaico 7-columnas con todas las thumbnails en overlay-mosaic.jpg
  - Resumen por imagen: numero de bboxes por clase

Uso:
  uv run --with opencv-python-headless --with numpy python scripts/labeling/overlay_review.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DATASET = Path("datasets/waste-3class-batch1-auto/batch1")
OUT_INDIVIDUAL = Path("datasets/waste-3class-batch1-auto/overlay")
OUT_MOSAIC = Path("datasets/waste-3class-batch1-auto/overlay-mosaic.jpg")

CLASS_NAMES = ["plastic", "paper", "glass"]
# BGR para opencv: rojo=plastic, verde=paper, azul=glass
COLORS = [(0, 64, 255), (64, 200, 64), (255, 128, 0)]


def draw_bboxes(img_path: Path, label_path: Path) -> tuple[np.ndarray, Counter]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"No se pudo leer {img_path}")
    h, w = img.shape[:2]
    counts: Counter = Counter()
    if label_path.exists() and label_path.stat().st_size > 0:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w - 1, int((xc + bw / 2) * w))
            y2 = min(h - 1, int((yc + bh / 2) * h))
            color = COLORS[cls] if cls < len(COLORS) else (200, 200, 200)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"c{cls}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            counts[cls] += 1
    return img, counts


def main() -> None:
    OUT_INDIVIDUAL.mkdir(parents=True, exist_ok=True)

    pairs = []
    for split in ["train", "valid"]:
        imgs_dir = DATASET / split / "images"
        labels_dir = DATASET / split / "labels"
        if not imgs_dir.exists():
            continue
        for img_path in sorted(imgs_dir.glob("*.jpg")):
            label_path = labels_dir / (img_path.stem + ".txt")
            pairs.append((img_path, label_path, split))

    print(f"[INFO] {len(pairs)} pares imagen+label encontrados")

    thumbs: list[np.ndarray] = []
    per_image_summary: list[tuple[str, str, Counter]] = []

    for img_path, label_path, split in pairs:
        annotated, counts = draw_bboxes(img_path, label_path)
        out_path = OUT_INDIVIDUAL / f"{split}_{img_path.name}"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        per_image_summary.append((split, img_path.stem, counts))

        thumb = cv2.resize(annotated, (320, 240), interpolation=cv2.INTER_AREA)
        banner = np.zeros((30, 320, 3), dtype=np.uint8)
        text = f"[{split}] {img_path.stem[-12:]} P:{counts.get(0, 0)}/Pa:{counts.get(1, 0)}/G:{counts.get(2, 0)}"
        cv2.putText(banner, text, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        thumbs.append(np.vstack([thumb, banner]))

    # Mosaico
    cols = 7
    rows = (len(thumbs) + cols - 1) // cols
    cell_h, cell_w = thumbs[0].shape[:2]
    mosaic = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        mosaic[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = t
    cv2.imwrite(str(OUT_MOSAIC), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Resumen
    print(f"\n[OK] {len(thumbs)} overlays individuales en {OUT_INDIVIDUAL}")
    print(f"[OK] Mosaico en {OUT_MOSAIC}")
    print(f"\n=== Resumen por imagen ===")
    print(f"{'split':<6} {'name':<30} {'plastic':>8} {'paper':>6} {'glass':>6} {'total':>6}")
    total_counts: Counter = Counter()
    for split, name, counts in per_image_summary:
        n_total = sum(counts.values())
        total_counts.update(counts)
        print(f"{split:<6} {name:<30} {counts.get(0, 0):>8} {counts.get(1, 0):>6} {counts.get(2, 0):>6} {n_total:>6}")

    print(f"\n=== Totales ===")
    print(f"plastic: {total_counts.get(0, 0)}")
    print(f"paper:   {total_counts.get(1, 0)}")
    print(f"glass:   {total_counts.get(2, 0)}")
    print(f"TOTAL:   {sum(total_counts.values())} bboxes en {len(pairs)} imagenes "
          f"({sum(total_counts.values()) / len(pairs):.1f} bbox/img)")


if __name__ == "__main__":
    main()
