"""Tests del cliente HF REST. Usa requests-mock para evitar tráfico real."""
import pytest
import requests_mock

from hf_rest import download, REPO, BASE


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
