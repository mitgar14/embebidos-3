"""Tests del cliente HF REST. Usa requests-mock para evitar tráfico real."""
import pytest
import requests_mock

from hf_rest import download, REPO, BASE, list_files, repo_info, get_head_revision


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
