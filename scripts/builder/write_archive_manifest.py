"""write_archive_manifest.py — genera manifest.json para un archive de engine.

Estructura inspirada en patrones DVC/MLflow/TinyMLDelta: metadata liviano que
referencia un binario que vive en otro lado (en este caso, local en el Nano).
Compatible con Python 3.6 (sin f-strings con `=`, sin dataclasses, sin Literal).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone


def build_manifest(archive_dir, engine_path, sha256_hex, size_bytes,
                   archived_at_utc, source_meta_path=None):
    archive_id = os.path.basename(archive_dir.rstrip(os.sep).rstrip("/"))
    manifest = {
        "schema_version": "1",
        "archive_id": archive_id,
        "archived_at_utc": archived_at_utc,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "hardware": {
            "platform": "jetson-nano-b01",
            "jetpack": "4.6.1",
            "tensorrt": "8.2.1.8",
            "cuda": "10.2",
            "arch": "sm_53",
            "precision": "fp16",
        },
        "artifact": {
            "filename": os.path.basename(engine_path),
            "size_bytes": int(size_bytes),
            "sha256": sha256_hex,
            "binary_present_remotely": False,
            "local_path_nano": engine_path,
        },
        "source_meta_inline": None,
        "notes": (
            "Backup local-only del engine TRT previo. El manifest se sube a HF "
            "(repo HF configurado, path engines-archive/<archive_id>/) "
            "como índice remoto buscable. El binario permanece en la SD del Nano; "
            "para recuperarlo sin red, leer engines-archive/<archive_id>/ en el Nano."
        ),
    }
    if source_meta_path and os.path.exists(source_meta_path):
        try:
            with open(source_meta_path, "r") as fh:
                manifest["source_meta_inline"] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            manifest["source_meta_inline_error"] = "{}: {}".format(
                type(exc).__name__, str(exc)
            )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Genera manifest.json para engines-archive/<ts>__<sha>/"
    )
    parser.add_argument("--archive-dir", required=True,
                        help="Directorio del archive (engines-archive/<ts>__<sha>/)")
    parser.add_argument("--engine", required=True,
                        help="Path al engine binario dentro del archive")
    parser.add_argument("--sha256", required=True,
                        help="SHA256 hex del engine (64 chars)")
    parser.add_argument("--size-bytes", required=True, type=int)
    parser.add_argument("--timestamp", required=True,
                        help="UTC YYYYMMDDTHHMMSSZ del archive (mismo que <ts> del dir)")
    parser.add_argument("--source-meta", default=None,
                        help="Path al .meta.json del engine origen (opcional)")
    args = parser.parse_args()

    if len(args.sha256) != 64:
        print("ERROR sha256 debe tener 64 chars hex, recibido {} chars".format(
            len(args.sha256)
        ), file=sys.stderr)
        sys.exit(2)

    manifest = build_manifest(
        archive_dir=args.archive_dir,
        engine_path=args.engine,
        sha256_hex=args.sha256,
        size_bytes=args.size_bytes,
        archived_at_utc=args.timestamp,
        source_meta_path=args.source_meta,
    )
    out_path = os.path.join(args.archive_dir, "manifest.json")
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("OK manifest: {}".format(out_path))


if __name__ == "__main__":
    main()
