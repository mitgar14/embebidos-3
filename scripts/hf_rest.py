"""hf_rest.py — cliente minimal HF Hub para Python 3.6 (sin huggingface_hub).
Solo requiere `requests`. Diseñado para Jetson Nano JP 4.6.1.
"""
import os
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict

import requests

REPO = "mitgar14/embebidos-3-models"
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


def upload_file_inline(local_path: Path, remote_path: str,
                       commit_msg: str = "embebidos3 backup",
                       branch: str = "main", timeout: int = 300) -> Dict:
    """Upload sin LFS, base64 inline. Apto para archivos < ~50 MB.
    Si el server exige LFS (422), levanta RuntimeError con instrucción para fallback."""
    local_path = Path(local_path)
    content_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    payload = {
        "summary": commit_msg,
        "files": [{
            "path": remote_path,
            "encoding": "base64",
            "content": content_b64,
        }]
    }
    url = f"{BASE}/api/models/{REPO}/commit/{branch}"
    r = requests.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    if r.status_code == 422 and "lfs" in r.text.lower():
        raise RuntimeError(f"Servidor exige LFS para {remote_path}. "
                           "Ver fallback con git-lfs en docs.")
    if r.status_code in (401, 403):
        raise RuntimeError(
            f"HF auth failed uploading to {remote_path} (status {r.status_code}). "
            f"Check HF_TOKEN env var."
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

    p_info = sub.add_parser("head-revision")

    args = parser.parse_args()
    if not args.cmd:
        parser.error("subcommand required: download | upload | head-revision")
    if args.cmd == "download":
        download(args.filename, Path(args.local_path), revision=args.revision)
        print(f"OK: {args.filename} -> {args.local_path}")
    elif args.cmd == "upload":
        result = upload_file_inline(Path(args.local_path), args.remote_path, args.message)
        print(f"OK: {result.get('commitUrl', 'commit OK')}")
    elif args.cmd == "head-revision":
        print(get_head_revision())
