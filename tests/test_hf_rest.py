"""Tests del cliente HF REST. Usa requests-mock para evitar tráfico real."""
import pytest
import requests_mock

from hf_rest import (
    download, REPO, BASE,
    list_files, repo_info, get_head_revision,
    get_file_lfs_sha256,
    upload_file_inline,
)


def test_download_streaming(tmp_path):
    fake_content = b"x" * 1024 * 100  # 100 KB
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, content=fake_content)
        download("exports/best.onnx", out)
    assert out.read_bytes() == fake_content


def test_download_with_revision(tmp_path):
    fake_content = b"y" * 50
    url = f"{BASE}/{REPO}/resolve/abc1234/manifests/manifest.json"
    out = tmp_path / "manifest.json"
    with requests_mock.Mocker() as m:
        m.get(url, content=fake_content)
        download("manifests/manifest.json", out, revision="abc1234")
    assert out.read_bytes() == fake_content


def test_download_failure_raises(tmp_path):
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, status_code=404)
        with pytest.raises(Exception):
            download("exports/best.onnx", out)
    assert not out.exists()


def test_list_files():
    fake_response = {
        "siblings": [
            {"rfilename": "README.md", "size": 1287},
            {"rfilename": "exports/best.onnx", "size": 12169740},
            {"rfilename": "manifests/manifest.json", "size": 3095},
        ]
    }
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json=fake_response)
        files = list_files()
    assert len(files) == 3
    assert any(f["rfilename"] == "exports/best.onnx" for f in files)


def test_repo_info():
    fake_response = {"sha": "65c1634abc", "lastModified": "2026-05-14T18:38:31Z"}
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json=fake_response)
        info = repo_info()
    assert info["sha"] == "65c1634abc"


def test_get_head_revision():
    fake_response = {"sha": "65c1634404ea379e38522885101222a07242f37f9"}
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json=fake_response)
        rev = get_head_revision()
    assert rev == "65c1634404ea379e38522885101222a07242f37f9"


def test_upload_file_inline_success(tmp_path):
    local = tmp_path / "best_fp16.engine"
    local.write_bytes(b"\x00" * 1024)
    fake_response = {"success": True, "commitUrl": "https://huggingface.co/..."}
    url = f"{BASE}/api/models/{REPO}/commit/main"
    with requests_mock.Mocker() as m:
        m.post(url, json=fake_response)
        result = upload_file_inline(local, "engines-archive/test/best_fp16.engine", "test commit")
    assert result["success"] is True


def test_upload_file_inline_lfs_raises(tmp_path):
    local = tmp_path / "best_fp16.engine"
    local.write_bytes(b"\x00" * 1024)
    url = f"{BASE}/api/models/{REPO}/commit/main"
    with requests_mock.Mocker() as m:
        m.post(url, status_code=422, text="LFS upload required for this file type")
        with pytest.raises(RuntimeError, match="LFS"):
            upload_file_inline(local, "engines-archive/test/best_fp16.engine")


def test_download_sends_auth_header(tmp_path, monkeypatch):
    """Auth header must be sent when HF_TOKEN is set."""
    monkeypatch.setenv("HF_TOKEN", "hf_testtoken_abc")
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, content=b"x")
        download("exports/best.onnx", out)
    assert m.last_request.headers["Authorization"] == "Bearer hf_testtoken_abc"


def test_list_files_sends_auth_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_testtoken_abc")
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json={"siblings": []})
        list_files()
    assert m.last_request.headers["Authorization"] == "Bearer hf_testtoken_abc"


def test_no_auth_header_when_token_missing(tmp_path, monkeypatch):
    """When HF_TOKEN is unset, no Authorization header should be sent (public repo case)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, content=b"x")
        download("exports/best.onnx", out)
    assert "Authorization" not in m.last_request.headers


def test_download_401_raises_auth_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_invalid")
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, status_code=401, text="invalid token")
        with pytest.raises(RuntimeError, match="HF auth failed"):
            download("exports/best.onnx", out)
    assert not out.exists()


def test_download_403_raises_auth_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token_no_perms")
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, status_code=403, text="forbidden")
        with pytest.raises(RuntimeError, match="HF auth failed"):
            download("exports/best.onnx", out)


def test_upload_401_raises_auth_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_invalid")
    local = tmp_path / "engine"
    local.write_bytes(b"x")
    url = f"{BASE}/api/models/{REPO}/commit/main"
    with requests_mock.Mocker() as m:
        m.post(url, status_code=401, text="invalid token")
        with pytest.raises(RuntimeError, match="HF auth failed"):
            upload_file_inline(local, "engines-archive/x/engine")


def test_upload_sends_base64_body(tmp_path, monkeypatch):
    """Upload payload must contain base64 of the file bytes."""
    import base64
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    local = tmp_path / "small_engine"
    local.write_bytes(b"hello world")
    url = f"{BASE}/api/models/{REPO}/commit/main"
    with requests_mock.Mocker() as m:
        m.post(url, json={"success": True})
        upload_file_inline(local, "engines-archive/test/small_engine", "test msg")
    payload = m.last_request.json()
    assert payload["summary"] == "test msg"
    assert len(payload["files"]) == 1
    f = payload["files"][0]
    assert f["path"] == "engines-archive/test/small_engine"
    assert f["encoding"] == "base64"
    assert base64.b64decode(f["content"]) == b"hello world"


def test_get_file_lfs_sha256_returns_oid():
    """tree/{rev}/{dirname} debe devolver el oid LFS del archivo."""
    fake_oid = "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad"
    url = f"{BASE}/api/models/{REPO}/tree/main/exports"
    with requests_mock.Mocker() as m:
        m.get(url, json=[
            {"type": "file", "oid": "git-pointer-sha",
             "size": 12169740,
             "lfs": {"oid": fake_oid, "size": 12169740, "pointerSize": 133},
             "path": "exports/best.onnx"},
        ])
        result = get_file_lfs_sha256("exports/best.onnx")
    assert result == fake_oid


def test_get_file_lfs_sha256_with_revision():
    """Debe usar la revision pasada como path-segment, no como query."""
    fake_oid = "deadbeef" * 8
    url = f"{BASE}/api/models/{REPO}/tree/abc1234/exports"
    with requests_mock.Mocker() as m:
        m.get(url, json=[
            {"type": "file", "path": "exports/best.onnx",
             "lfs": {"oid": fake_oid}},
        ])
        result = get_file_lfs_sha256("exports/best.onnx", revision="abc1234")
    assert result == fake_oid


def test_get_file_lfs_sha256_returns_none_when_not_lfs():
    """Si el archivo no es LFS (no tiene campo lfs), devuelve None."""
    url = f"{BASE}/api/models/{REPO}/tree/main"
    with requests_mock.Mocker() as m:
        m.get(url, json=[
            {"type": "file", "path": "README.md", "size": 1287},
        ])
        result = get_file_lfs_sha256("README.md")
    assert result is None


def test_get_file_lfs_sha256_returns_none_when_file_missing():
    """Si el archivo no está en el listing, devuelve None."""
    url = f"{BASE}/api/models/{REPO}/tree/main/exports"
    with requests_mock.Mocker() as m:
        m.get(url, json=[
            {"type": "file", "path": "exports/other.onnx", "lfs": {"oid": "xxx"}},
        ])
        result = get_file_lfs_sha256("exports/best.onnx")
    assert result is None


def test_get_file_lfs_sha256_root_file():
    """Para archivos en la raíz, la URL no debe tener trailing /."""
    fake_oid = "feedface" * 8
    url = f"{BASE}/api/models/{REPO}/tree/main"
    with requests_mock.Mocker() as m:
        m.get(url, json=[
            {"type": "file", "path": "best.onnx", "lfs": {"oid": fake_oid}},
        ])
        result = get_file_lfs_sha256("best.onnx")
    assert result == fake_oid


def test_get_head_revision_raises_on_missing_sha(monkeypatch):
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json={})
        with pytest.raises(RuntimeError, match="missing 'sha'"):
            get_head_revision()


def test_download_cleans_tmp_on_mid_stream_failure(tmp_path, monkeypatch):
    """If iter_content raises mid-stream, the .tmp file must not be left behind."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    # Provide a non-streamable response, then trigger an error inside iter_content
    # by mocking the response to raise on read. requests_mock supports raising via exc=.
    import requests as r_mod
    with requests_mock.Mocker() as m:
        m.get(url, exc=r_mod.exceptions.ConnectionError("simulated drop"))
        with pytest.raises(Exception):
            download("exports/best.onnx", out)
    assert not out.exists()
    # also no .tmp leak
    tmp = out.parent / "best.onnx.tmp"
    assert not tmp.exists()
