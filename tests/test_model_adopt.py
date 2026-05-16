"""Test POST /model/adopt — registra engine huérfano como si fuera HF HEAD."""
import json

from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


_FAKE_REV = "b93964f9e4f9464cfe55b13ca5a577ba383a4dd5"
_FAKE_ONNX = "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad"


def _setup_engine(tmp_path, monkeypatch, write_engine=True, write_meta=False):
    eng = tmp_path / "best_fp16.engine"
    meta = tmp_path / "best_fp16.engine.meta.json"
    if write_engine:
        eng.write_bytes(b"fake-engine-binary-bytes" * 100)
    if write_meta:
        meta.write_text(json.dumps({"hf_revision": "old"}))
    import nano_server
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE", eng)
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", meta)
    monkeypatch.setattr(nano_server, "PREVIOUS_ENGINE_META", tmp_path / "nope.old.meta.json")
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")
    return eng, meta


def _stub_hf(monkeypatch, rev=_FAKE_REV, onnx=_FAKE_ONNX):
    monkeypatch.setattr("hf_rest.get_head_revision", lambda: rev)
    monkeypatch.setattr("hf_rest.get_file_lfs_sha256",
                        lambda p, revision="main": onnx)
    monkeypatch.setattr("hf_rest.repo_info",
                        lambda revision="main": {"sha": rev,
                                                  "lastModified": "2026-05-15T10:00:00Z"})


def test_adopt_creates_meta_when_orphan_engine(tmp_path, monkeypatch):
    """Caso happy: engine binario presente, sin meta → adopta."""
    _stub_worker_startup(monkeypatch)
    _, meta = _setup_engine(tmp_path, monkeypatch, write_engine=True, write_meta=False)
    _stub_hf(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server.worker, "request_swap", lambda path: None)

    with TestClient(app) as c:
        r = c.post("/model/adopt")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["meta"]["hf_revision"] == _FAKE_REV
    assert data["meta"]["onnx_sha256"] == _FAKE_ONNX
    assert data["meta"]["adopted"] is True
    assert data["meta"]["build_duration_s"] == 0
    # engine_sha256 debe ser hex de 64 chars
    assert len(data["meta"]["engine_sha256"]) == 64
    # archivo en disco
    assert meta.exists()
    on_disk = json.loads(meta.read_text())
    assert on_disk["adopted"] is True
    assert on_disk["hf_revision"] == _FAKE_REV


def test_adopt_404_when_no_engine(tmp_path, monkeypatch):
    """Sin engine binario → 404 no_engine_binary."""
    _stub_worker_startup(monkeypatch)
    _setup_engine(tmp_path, monkeypatch, write_engine=False, write_meta=False)
    _stub_hf(monkeypatch)
    with TestClient(app) as c:
        r = c.post("/model/adopt")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "no_engine_binary"


def test_adopt_409_when_meta_already_exists(tmp_path, monkeypatch):
    """Engine YA tiene meta → 409 meta_already_exists (no machaca)."""
    _stub_worker_startup(monkeypatch)
    _, meta = _setup_engine(tmp_path, monkeypatch, write_engine=True, write_meta=True)
    _stub_hf(monkeypatch)
    with TestClient(app) as c:
        r = c.post("/model/adopt")
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "meta_already_exists"
    # meta original sin tocar
    assert json.loads(meta.read_text())["hf_revision"] == "old"


def test_adopt_503_when_hf_unreachable(tmp_path, monkeypatch):
    """HF caído → 503 hf_unreachable."""
    _stub_worker_startup(monkeypatch)
    _setup_engine(tmp_path, monkeypatch, write_engine=True, write_meta=False)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr("hf_rest.get_head_revision", boom)
    with TestClient(app) as c:
        r = c.post("/model/adopt")
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "hf_unreachable"


def test_state_includes_engine_binary_present(tmp_path, monkeypatch):
    """/model/state debe exponer engine_binary_present para que el frontend
    pueda decidir mostrar el botón adoptar."""
    _stub_worker_startup(monkeypatch)
    _setup_engine(tmp_path, monkeypatch, write_engine=True, write_meta=False)
    with TestClient(app) as c:
        r = c.get("/model/state")
    data = r.json()
    assert data["state"] == "no_model"
    assert data["engine_binary_present"] is True


def test_state_engine_binary_absent_when_no_file(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    _setup_engine(tmp_path, monkeypatch, write_engine=False, write_meta=False)
    with TestClient(app) as c:
        r = c.get("/model/state")
    data = r.json()
    assert data["state"] == "no_model"
    assert data["engine_binary_present"] is False
