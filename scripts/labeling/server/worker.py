"""Worker: descarga imagenes, corre Autodistill + Grounding DINO, sube labels a HF.

Invocado por main.py como subprocess. Lee request.json desde job_dir, escribe
DONE / FAILED como marker files. Logs van a stdout (capturados por main.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# Worker carga lazy: imports pesados solo si llegamos al run.


def log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def download_inputs(input_url: str, dest_dir: Path) -> Path:
    """Descarga imagenes desde URL ZIP o repo HF dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"Descargando inputs desde {input_url}")

    if input_url.startswith("hf://"):
        # Patron: hf://repo_id/subpath
        from huggingface_hub import snapshot_download

        parsed = input_url[5:]  # strip hf://
        parts = parsed.split("/", 2)
        repo_id = "/".join(parts[:2])
        subpath = parts[2] if len(parts) > 2 else ""
        log(f"HF snapshot_download repo_id={repo_id} subpath={subpath}")
        local = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{subpath}/*"] if subpath else None,
            local_dir=str(dest_dir),
            token=os.environ.get("HF_TOKEN"),
        )
        src = Path(local) / subpath if subpath else Path(local)
        return src

    # URL HTTP(S) → asumir ZIP
    import urllib.request

    zip_path = dest_dir / "input.zip"
    urllib.request.urlretrieve(input_url, zip_path)
    log(f"Descargado {zip_path.stat().st_size / 1024:.1f} KB, extrayendo...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir / "extracted")
    zip_path.unlink()
    return dest_dir / "extracted"


def run_autodistill(
    images_dir: Path,
    output_dir: Path,
    ontology: dict,
    conf: float,
    model_type: str,
) -> int:
    """Ejecuta GroundingDINO sobre todas las imagenes. Retorna numero de labels generadas."""
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO

    log(f"Inicializando GroundingDINO model_type={model_type}")
    base = GroundingDINO(
        ontology=CaptionOntology(ontology),
        model_type=model_type,
    )
    log(f"Etiquetando imagenes en {images_dir} → {output_dir}")
    base.label(
        input_folder=str(images_dir),
        output_folder=str(output_dir),
        extension=".jpg",
        conf=conf,
    )
    label_files = list((output_dir / "train" / "labels").glob("*.txt"))
    log(f"Labels generados: {len(label_files)}")
    return len(label_files)


def upload_to_hf(labels_dir: Path, hf_repo: str, hf_path: str) -> str:
    from huggingface_hub import HfApi

    log(f"Subiendo a HF {hf_repo}/{hf_path}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(
        folder_path=str(labels_dir),
        path_in_repo=hf_path,
        repo_id=hf_repo,
        repo_type="dataset",
        commit_message=f"auto-label batch {hf_path}",
    )
    return f"https://huggingface.co/datasets/{hf_repo}/tree/main/{hf_path}"


def package_labels(labels_dir: Path, zip_path: Path) -> None:
    log(f"Empaquetando labels en {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in labels_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(labels_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()

    job_dir: Path = args.job_dir
    req_path = job_dir / "request.json"
    req = json.loads(req_path.read_text())
    log(f"job_id={args.job_id} request={req}")

    try:
        # 1) Descarga inputs
        with tempfile.TemporaryDirectory(prefix="autolabel-") as tmp:
            tmp_dir = Path(tmp)
            inputs_dir = download_inputs(req["input_url"], tmp_dir / "inputs")

            # 2) GroundingDINO
            output_dir = tmp_dir / "autodistill-output"
            n_labels = run_autodistill(
                inputs_dir,
                output_dir,
                req["ontology"],
                conf=req.get("conf", 0.25),
                model_type=req.get("model_type", "tiny"),
            )

            # 3) Gate de calidad: al menos 85% de imagenes con labels
            n_inputs = sum(1 for _ in inputs_dir.rglob("*.jpg"))
            ratio = n_labels / max(n_inputs, 1)
            log(f"Gate: {n_labels}/{n_inputs} = {ratio:.1%} con labels (umbral 85%)")
            if ratio < 0.85:
                raise RuntimeError(
                    f"calidad insuficiente: {ratio:.1%} < 85%. Revisar conf threshold o ontology."
                )

            # 4) Subir a HF (opcional)
            if hf_repo := req.get("hf_dataset_repo"):
                upload_to_hf(output_dir, hf_repo, req.get("hf_dataset_path", "batch"))

            # 5) Empaquetar ZIP local para download via /jobs/{id}/artifact
            zip_path = job_dir / "labels.zip"
            package_labels(output_dir, zip_path)

        (job_dir / "DONE").write_text(f"labels={n_labels} ratio={ratio:.2%}")
        log(f"OK done. labels.zip listo en {zip_path}")
        sys.exit(0)
    except Exception as e:
        tb = traceback.format_exc()
        log(f"FAILED: {e!r}\n{tb}")
        (job_dir / "FAILED").write_text(f"{e!r}\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
