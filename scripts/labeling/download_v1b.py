"""Descarga v1-B (o la version mas reciente) del workspace embebidos3 antes de cancelar.

Aplica el workaround para el bug dataset.location del SDK Roboflow (cascada de busqueda
de data.yaml).
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

WORKSPACE = "embebidos3"
PROJECT_ID = "waste-3class-lwld8"
DEFAULT_DEST = PROJECT_ROOT / "datasets" / "waste-3class-v1b"


def find_data_yaml(start: Path, deadline_s: float = 20.0) -> Path | None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        for p in start.rglob("data.yaml"):
            try:
                with p.open() as f:
                    y = yaml.safe_load(f)
                if isinstance(y, dict) and "names" in y:
                    return p
            except Exception:
                continue
        time.sleep(0.5)
    return None


def list_versions(rf: Roboflow) -> list[dict]:
    project = rf.workspace(WORKSPACE).project(PROJECT_ID)
    return [
        {
            "version": v.version,
            "name": getattr(v, "name", "?"),
            "url": getattr(v, "url", getattr(v, "id", "?")),
        }
        for v in project.versions()
    ]


def download_version(rf: Roboflow, version: int, dest: Path, fmt: str = "yolov8") -> Path:
    project = rf.workspace(WORKSPACE).project(PROJECT_ID)
    ver = project.version(version)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Descargando version {version} formato {fmt} a {dest}")
    location_attempts = [
        lambda: ver.download(fmt, location=str(dest), overwrite=True),
        lambda: ver.download(fmt, overwrite=True),
    ]
    for attempt, fn in enumerate(location_attempts, 1):
        try:
            dataset = fn()
            loc = getattr(dataset, "location", None)
            if loc and Path(loc).exists():
                yaml_p = find_data_yaml(Path(loc))
                if yaml_p:
                    if Path(loc) != dest:
                        # mover contenido a dest
                        for item in Path(loc).iterdir():
                            target = dest / item.name
                            if target.exists():
                                shutil.rmtree(target) if target.is_dir() else target.unlink()
                            shutil.move(str(item), str(target))
                        shutil.rmtree(loc, ignore_errors=True)
                    print(f"[OK] data.yaml encontrado: {find_data_yaml(dest)}")
                    return dest
        except Exception as e:
            print(f"[WARN] Intento {attempt} fallo: {e!r}")

    yaml_p = find_data_yaml(dest)
    if yaml_p is None:
        raise RuntimeError(f"No se encontro data.yaml en {dest} tras la descarga")
    print(f"[OK] data.yaml encontrado (fallback): {yaml_p}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--version", type=int, help="Numero de version a descargar")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--format", default="yolov8")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY ausente en entorno o .env")

    rf = Roboflow(api_key=api_key)

    versions = list_versions(rf)
    print("[INFO] Versiones disponibles:")
    for v in versions:
        print(f"  - version {v['version']:>2}  name={v['name']!r}  url={v['url']}")

    if args.list_only:
        return
    if not versions:
        sys.exit("Ninguna version disponible en el proyecto")

    chosen = args.version or max(v["version"] for v in versions)
    print(f"[INFO] Version elegida: {chosen}")
    out = download_version(rf, chosen, args.dest, fmt=args.format)
    print(f"[DONE] Dataset descargado en {out}")


if __name__ == "__main__":
    main()
