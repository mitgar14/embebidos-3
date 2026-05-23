"""Sube las fotos crudas (sin etiquetar) al HF dataset privado.

Default: mitgar14/embebidos3-raw-batches.
Subpath dentro del repo: el argumento --batch (ej: batch1, batch2).
Source default: C:/Users/mitgar14/OneDrive/Imagenes/Album de camara (fotos WIN_*.jpg).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.path.expanduser("~/OneDrive/Imágenes/Álbum de cámara")),
        help="Carpeta local con las fotos JPG.",
    )
    parser.add_argument(
        "--repo",
        default="mitgar14/embebidos3-raw-batches",
        help="Repo HF dataset destino.",
    )
    parser.add_argument(
        "--batch",
        default="batch1",
        help="Subpath dentro del repo (ej: batch1).",
    )
    parser.add_argument(
        "--pattern",
        default="WIN_20260515_*.jpg",
        help="Glob para filtrar archivos (por defecto fotos del 2026-05-15, top-level no recursivo).",
    )
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Solo lista lo que subiria.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN ausente en entorno/.env")

    if not args.source.exists():
        sys.exit(f"Source no existe: {args.source}")

    files = sorted(args.source.glob(args.pattern))
    print(f"[INFO] {len(files)} archivos coinciden con {args.pattern!r} en {args.source}")
    if not files:
        sys.exit("Nada que subir")

    if args.dry_run:
        for f in files[:10]:
            print(f"  - {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
        if len(files) > 10:
            print(f"  ... y {len(files) - 10} mas")
        return

    api = HfApi(token=token)

    # Crear repo si no existe (idempotente)
    print(f"[INFO] Asegurando repo {args.repo} (private={args.private})")
    create_repo(
        repo_id=args.repo,
        token=token,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # Subir como folder
    print(f"[INFO] Subiendo {len(files)} archivos a {args.repo}:{args.batch}/...")
    # Necesitamos un dir temporal con solo los archivos del batch para upload_folder
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / args.batch
        staging.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, staging / f.name)
        api.upload_folder(
            folder_path=str(staging.parent),
            path_in_repo="",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"Upload {args.batch} ({len(files)} imagenes)",
        )

    print(f"[DONE] https://huggingface.co/datasets/{args.repo}/tree/main/{args.batch}")


if __name__ == "__main__":
    main()
