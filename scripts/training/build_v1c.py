"""Construye datasets/waste-3class-v1c combinando v1-B (Roboflow) + batch1-clean (manual).

v1-B   clases: [glass=0, paper=1, plastic=2]
batch1 clases: [plastic=0, paper=1, glass=2]

Remap obligatorio para batch1: {0->2 (plastic), 1->1 (paper), 2->0 (glass)}.
El orden final de clases del v1-c sigue v1-B para compatibilidad con el modelo existente.

Uso:
  uv run python scripts/training/build_v1c.py
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1B = ROOT / "datasets/waste-3class-v1b"
BATCH = ROOT / "datasets/waste-3class-batch1-auto/batch1-clean"
V1C = ROOT / "datasets/waste-3class-v1c"

# batch1 cls_idx -> v1-B cls_idx
REMAP = {0: 2, 1: 1, 2: 0}

DATA_YAML = """names:
- glass
- paper
- plastic
nc: 3
train: train/images
val: valid/images
test: test/images
"""


def main() -> None:
    if V1C.exists():
        print(f"[INFO] Borrando v1-c previo: {V1C}")
        shutil.rmtree(V1C)

    # 1) Copy v1-B base (train + valid + test)
    print(f"[1/3] Copiando v1-B base")
    for split in ["train", "valid", "test"]:
        for sub in ["images", "labels"]:
            src = V1B / split / sub
            dst = V1C / split / sub
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                continue
            count = 0
            for f in src.iterdir():
                shutil.copy2(f, dst / f.name)
                count += 1
        n_imgs = len(list((V1C / split / "images").iterdir()))
        n_lbls = len(list((V1C / split / "labels").iterdir()))
        print(f"  v1-B/{split:6s}: {n_imgs} imgs, {n_lbls} labels")

    # 2) Add batch1 con remap de clases
    print(f"[2/3] Agregando batch1-clean con remap {REMAP}")
    for split in ["train", "valid"]:
        src_imgs = BATCH / split / "images"
        src_lbls = BATCH / split / "labels"
        dst_imgs = V1C / split / "images"
        dst_lbls = V1C / split / "labels"
        if not src_imgs.exists():
            continue
        added = 0
        for f in src_imgs.iterdir():
            shutil.copy2(f, dst_imgs / f.name)
        for f in src_lbls.iterdir():
            new_lines = []
            for line in f.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls not in REMAP:
                    raise ValueError(f"Clase inesperada {cls} en {f.name}")
                new_lines.append(f"{REMAP[cls]} " + " ".join(parts[1:]))
            (dst_lbls / f.name).write_text("\n".join(new_lines))
            added += 1
        print(f"  batch1/{split:6s}: +{added} pares")

    # 3) data.yaml v1-c
    print(f"[3/3] Escribiendo data.yaml")
    (V1C / "data.yaml").write_text(DATA_YAML)

    # Stats finales
    print("\n=== v1-c resumen ===")
    names = ["glass", "paper", "plastic"]
    grand_total = Counter()
    for split in ["train", "valid", "test"]:
        cls_count: Counter = Counter()
        for lf in (V1C / split / "labels").glob("*.txt"):
            for line in lf.read_text().splitlines():
                if line.strip():
                    cls_count[int(line.split()[0])] += 1
        n_imgs = len(list((V1C / split / "images").iterdir()))
        n_lbls = len(list((V1C / split / "labels").iterdir()))
        total = sum(cls_count.values())
        print(f"\n{split:6s}: {n_imgs} imgs / {n_lbls} labels / {total} bboxes")
        for cls_idx in sorted(cls_count):
            cnt = cls_count[cls_idx]
            pct = cnt / total * 100 if total else 0
            print(f"  {names[cls_idx]:8s}: {cnt:6d} ({pct:5.1f}%)")
        grand_total.update(cls_count)

    total = sum(grand_total.values())
    print(f"\n=== Total v1-c ===")
    print(f"Bboxes: {total}")
    for cls_idx in sorted(grand_total):
        cnt = grand_total[cls_idx]
        pct = cnt / total * 100 if total else 0
        print(f"  {names[cls_idx]:8s}: {cnt:6d} ({pct:5.1f}%)")
    print(f"\nEscrito en: {V1C}")


if __name__ == "__main__":
    main()
