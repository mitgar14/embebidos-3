"""hf_rest.py — cliente minimal HF Hub para Python 3.6 (sin huggingface_hub).
Solo requiere `requests`. Diseñado para Jetson Nano JP 4.6.1.
"""
import os
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict

import requests

REPO = os.environ.get("EMBEBIDOS3_HF_REPO", "mitgar14/embebidos-3-models")
BASE = "https://huggingface.co"


def _headers():
    token = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download(filename: str, local_path: Path, revision: str = "main",
             chunk_size: int = 65536, timeout: int = 120) -> None:
    """Descarga un archivo del repo a local_path. Streaming, atomic write."""
    url = f"{BASE}/{REPO}/resolve/{revision}/{filename}"
    r = requests.get(url, headers=_headers(), stream=True, timeout=timeout)
    if r.status_code in (401, 403):
        raise RuntimeError(
            f"HF auth failed for {filename} (status {r.status_code}). "
            f"Check HF_TOKEN env var."
        )
    r.raise_for_status()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        tmp.rename(local_path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def list_files(revision: str = "main", timeout: int = 30) -> List[Dict]:
    """Lista siblings del repo en la revision dada."""
    url = f"{BASE}/api/models/{REPO}"
    r = requests.get(url, headers=_headers(),
                     params={"revision": revision}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("siblings", [])


def repo_info(revision: str = "main", timeout: int = 30) -> Dict:
    """Info de la revision: sha, lastModified, etc."""
    url = f"{BASE}/api/models/{REPO}"
    r = requests.get(url, headers=_headers(),
                     params={"revision": revision}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_head_revision(timeout: int = 30) -> str:
    """SHA del último commit en main. Raises RuntimeError si HF no devuelve `sha`."""
    sha = repo_info("main", timeout=timeout).get("sha")
    if not sha:
        raise RuntimeError("HF response missing 'sha' field")
    return sha


def get_file_lfs_sha256(filepath: str, revision: str = "main",
                        timeout: int = 30) -> Optional[str]:
    """Devuelve el SHA256 hex del archivo (LFS oid) en revision, o None si no es LFS
    ni existe. Consulta /api/models/{repo}/tree/{rev}[/{dirname}] y filtra por path.
    NO descarga el archivo; solo metadata (~1 KB JSON).
    """
    from posixpath import dirname as _pdirname
    dn = _pdirname(filepath)
    url = f"{BASE}/api/models/{REPO}/tree/{revision}"
    if dn:
        url += f"/{dn}"
    r = requests.get(url, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    for item in r.json():
        if item.get("path") == filepath:
            lfs = item.get("lfs") or {}
            return lfs.get("oid") or None
    return None


INLINE_MAX_BYTES = 10 * 1024 * 1024  # HF fuerza LFS por encima de ~10 MB.


def upload_file_inline(local_path: Path, remote_path: str,
                       commit_msg: str = "embebidos3 backup",
                       branch: str = "main", timeout: int = 300) -> Dict:
    """Upload sin LFS, base64 inline, vía endpoint commit NDJSON.

    Apto solo para archivos pequeños (<10 MB). Por encima, HF responde
    `uploadMode: lfs` en el preupload y rechaza el commit inline; este cliente
    no implementa el flujo LFS (preupload + batch + S3 PUT + verify) — usar
    backup local en disco para binarios grandes.

    Shape verificado contra `_commit_api.py` de huggingface_hub y reimplementación
    independiente HfHub.Commit (Elixir): NDJSON con `header` + `file`, una entrada
    por línea, Content-Type `application/x-ndjson`.
    """
    local_path = Path(local_path)
    size = local_path.stat().st_size
    if size > INLINE_MAX_BYTES:
        raise RuntimeError(
            "Archivo {} pesa {} bytes (>{} MB). HF lo forzaría a LFS y este "
            "cliente solo soporta inline; usar backup local para binarios "
            "grandes y subir solo el manifest a HF.".format(
                remote_path, size, INLINE_MAX_BYTES // (1024 * 1024)
            )
        )
    content_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    header_line = json.dumps({
        "key": "header",
        "value": {"summary": commit_msg, "description": ""},
    })
    file_line = json.dumps({
        "key": "file",
        "value": {
            "path": remote_path,
            "encoding": "base64",
            "content": content_b64,
        },
    })
    body = (header_line + "\n" + file_line + "\n").encode("utf-8")
    url = "{}/api/models/{}/commit/{}".format(BASE, REPO, branch)
    r = requests.post(
        url,
        headers={**_headers(), "Content-Type": "application/x-ndjson"},
        data=body,
        timeout=timeout,
    )
    if r.status_code == 422 and "lfs" in r.text.lower():
        raise RuntimeError(
            "Servidor exige LFS para {}. Este cliente solo soporta inline; "
            "subir solo manifests pequeños.".format(remote_path)
        )
    if r.status_code in (401, 403):
        raise RuntimeError(
            "HF auth failed uploading to {} (status {}). Check HF_TOKEN env var.".format(
                remote_path, r.status_code
            )
        )
    r.raise_for_status()
    return r.json()


def delete_file_inline(remote_path: str,
                       commit_msg: str = "embebidos3 delete",
                       branch: str = "main", timeout: int = 60) -> Dict:
    """Borra un archivo del repo HF vía commit NDJSON con key=deletedFile.

    Borrar el último archivo de un directorio elimina implícitamente el directorio
    en el repo (Git no rastrea directorios vacíos). Para borrar carpetas enteras
    con muchos archivos, usar la clave `deletedFolder` (no implementada aquí).
    """
    header_line = json.dumps({
        "key": "header",
        "value": {"summary": commit_msg, "description": ""},
    })
    delete_line = json.dumps({
        "key": "deletedFile",
        "value": {"path": remote_path},
    })
    body = (header_line + "\n" + delete_line + "\n").encode("utf-8")
    url = "{}/api/models/{}/commit/{}".format(BASE, REPO, branch)
    r = requests.post(
        url,
        headers={**_headers(), "Content-Type": "application/x-ndjson"},
        data=body,
        timeout=timeout,
    )
    if r.status_code in (401, 403):
        raise RuntimeError(
            "HF auth failed deleting {} (status {}). Check HF_TOKEN env var.".format(
                remote_path, r.status_code
            )
        )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="cliente REST HF Hub minimal")
    # Python 3.6 argparse no soporta required=True en add_subparsers (PEP 3.7+),
    # se valida manualmente tras parse_args.
    sub = parser.add_subparsers(dest="cmd")

    p_dl = sub.add_parser("download")
    p_dl.add_argument("filename", help="path en el repo, ej. exports/best.onnx")
    p_dl.add_argument("local_path", help="destino local")
    p_dl.add_argument("--revision", default="main")

    p_up = sub.add_parser("upload")
    p_up.add_argument("local_path", help="archivo local a subir")
    p_up.add_argument("remote_path", help="path en el repo")
    p_up.add_argument("--message", default="embebidos3 backup")

    p_del = sub.add_parser("delete")
    p_del.add_argument("remote_path", help="path en el repo a borrar")
    p_del.add_argument("--message", default="embebidos3 delete")

    p_info = sub.add_parser("head-revision")

    args = parser.parse_args()
    if not args.cmd:
        parser.error("subcommand required: download | upload | delete | head-revision")
    if args.cmd == "download":
        download(args.filename, Path(args.local_path), revision=args.revision)
        print(f"OK: {args.filename} -> {args.local_path}")
    elif args.cmd == "upload":
        result = upload_file_inline(Path(args.local_path), args.remote_path, args.message)
        print(f"OK: {result.get('commitUrl', 'commit OK')}")
    elif args.cmd == "delete":
        result = delete_file_inline(args.remote_path, args.message)
        print(f"OK delete: {result.get('commitUrl', 'commit OK')}")
    elif args.cmd == "head-revision":
        print(get_head_revision())
