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
    r.raise_for_status()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    tmp.rename(local_path)


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
    """SHA del último commit en main."""
    return repo_info("main", timeout=timeout).get("sha", "")


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
    r.raise_for_status()
    return r.json()
