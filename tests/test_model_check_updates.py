"""Tests POST /model/check-updates."""
import json
from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


_ONNX_SHA_OLD = "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad"
_ONNX_SHA_NEW = "ff" * 32


def _write_meta(tmp_path, monkeypatch, hf_revision, onnx_sha256):
    import nano_server
    meta_path = tmp_path / "best_fp16.engine.meta.json"
    meta_path.write_text(json.dumps({
        "hf_revision": hf_revision,
        "onnx_sha256": onnx_sha256,
    }))
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", meta_path)


def test_check_updates_up_to_date(tmp_path, monkeypatch):
    """Mismo commit + mismo ONNX → up_to_date=True, same_onnx=True."""
    _stub_worker_startup(monkeypatch)
    _write_meta(tmp_path, monkeypatch, "65c1634abc", _ONNX_SHA_OLD)
    monkeypatch.setattr("hf_rest.get_head_revision", lambda: "65c1634abc")
    monkeypatch.setattr("hf_rest.get_file_lfs_sha256",
                        lambda p, revision="main": _ONNX_SHA_OLD)
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["up_to_date"] is True
    assert data["same_revision"] is True
    assert data["same_onnx"] is True
    assert data["has_engine"] is True
    assert data["latest_onnx_sha256"] == _ONNX_SHA_OLD


def test_check_updates_new_iteration(tmp_path, monkeypatch):
    """Commit nuevo + ONNX distinto → up_to_date=False, same_onnx=False."""
    _stub_worker_startup(monkeypatch)
    _write_meta(tmp_path, monkeypatch, "65c1634abc", _ONNX_SHA_OLD)
    monkeypatch.setattr("hf_rest.get_head_revision", lambda: "7a3b8e2new")
    monkeypatch.setattr("hf_rest.get_file_lfs_sha256",
                        lambda p, revision="main": _ONNX_SHA_NEW)
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["up_to_date"] is False
    assert data["same_revision"] is False
    assert data["same_onnx"] is False
    assert data["latest_revision"] == "7a3b8e2new"
    assert data["current_revision"] == "65c1634abc"
    assert data["latest_onnx_sha256"] == _ONNX_SHA_NEW
    assert data["current_onnx_sha256"] == _ONNX_SHA_OLD


def test_check_updates_cosmetic_commit(tmp_path, monkeypatch):
    """Commit nuevo PERO ONNX igual (ej. README) → up_to_date=True, same_revision=False."""
    _stub_worker_startup(monkeypatch)
    _write_meta(tmp_path, monkeypatch, "65c1634abc", _ONNX_SHA_OLD)
    monkeypatch.setattr("hf_rest.get_head_revision", lambda: "newcommit123")
    monkeypatch.setattr("hf_rest.get_file_lfs_sha256",
                        lambda p, revision="main": _ONNX_SHA_OLD)
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["up_to_date"] is True
    assert data["same_revision"] is False
    assert data["same_onnx"] is True


def test_check_updates_no_engine(tmp_path, monkeypatch):
    """Sin engine local (meta inexistente) → has_engine=False, up_to_date=False."""
    _stub_worker_startup(monkeypatch)
    import nano_server
    missing_meta = tmp_path / "missing.meta.json"
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", missing_meta)
    monkeypatch.setattr("hf_rest.get_head_revision", lambda: "newcommit123")
    monkeypatch.setattr("hf_rest.get_file_lfs_sha256",
                        lambda p, revision="main": _ONNX_SHA_NEW)
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["has_engine"] is False
    assert data["up_to_date"] is False
    assert data["current_revision"] is None
    assert data["current_onnx_sha256"] is None
    assert data["latest_revision"] == "newcommit123"
    assert data["latest_onnx_sha256"] == _ONNX_SHA_NEW
