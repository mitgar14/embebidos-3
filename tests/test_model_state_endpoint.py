"""Test del endpoint GET /model/state. Mockea filesystem para los estados base."""
import json

from fastapi.testclient import TestClient
from nano_server import app


def _stub_worker_startup(monkeypatch):
    """Evita que el startup intente cargar un engine real (no hay GPU/CUDA en el host)."""
    import nano_server
    monkeypatch.setattr(nano_server.worker, "start", lambda: None)
    monkeypatch.setattr(nano_server.worker, "wait_ready", lambda timeout=60: True)
    monkeypatch.setattr(nano_server.worker, "stop", lambda: None)
    monkeypatch.setattr(nano_server.worker, "join", lambda timeout=5: None)


def test_model_state_no_model(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    monkeypatch.setattr("nano_server_constants.ACTIVE_ENGINE", tmp_path / "nope.engine")
    monkeypatch.setattr("nano_server_constants.ACTIVE_ENGINE_META", tmp_path / "nope.meta.json")
    monkeypatch.setattr("nano_server_constants.PREVIOUS_ENGINE", tmp_path / "nope.old")
    monkeypatch.setattr("nano_server_constants.PREVIOUS_ENGINE_META", tmp_path / "nope.old.meta.json")
    monkeypatch.setattr("nano_server_constants.JOB_STATE_FILE", tmp_path / "nope.json")
    # The server module imports these by NAME at module level — re-import the names too
    import nano_server
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE", tmp_path / "nope.engine")
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", tmp_path / "nope.meta.json")
    monkeypatch.setattr(nano_server, "PREVIOUS_ENGINE_META", tmp_path / "nope.old.meta.json")
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")

    with TestClient(app) as c:
        r = c.get("/model/state")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "no_model"
    assert data["active_engine"] is None


def test_model_state_ready(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    eng = tmp_path / "best_fp16.engine"
    meta = tmp_path / "best_fp16.engine.meta.json"
    eng.write_bytes(b"\x00" * 100)
    meta.write_text(json.dumps({
        "engine_sha256": "abc",
        "onnx_sha256": "def",
        "hf_revision": "65c1634",
        "hf_commit_date": "2026-05-14T18:38:31Z",
        "trtexec_args": ["--fp16"],
        "build_completed_at": "2026-05-16T14:47:18-05:00",
        "build_duration_s": 496,
    }))
    import nano_server
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE", eng)
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", meta)
    monkeypatch.setattr(nano_server, "PREVIOUS_ENGINE_META", tmp_path / "nope.old.meta.json")
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")

    with TestClient(app) as c:
        r = c.get("/model/state")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "ready"
    assert data["active_engine"]["hf_revision"] == "65c1634"


def test_model_state_building(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    eng = tmp_path / "best_fp16.engine"
    meta = tmp_path / "best_fp16.engine.meta.json"
    eng.write_bytes(b"\x00" * 100)
    meta.write_text(json.dumps({"hf_revision": "abc1234"}))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "job_id": "20260516-1422-abc123",
        "pid": 99999,  # no existe, será cleaned
        "phase": "trtexec_building",
        "progress_pct": 47,
        "heartbeat": 1747424793.17,
    }))
    import nano_server
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE", eng)
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", meta)
    monkeypatch.setattr(nano_server, "PREVIOUS_ENGINE_META", tmp_path / "nope.old.meta.json")
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", job)

    with TestClient(app) as c:
        r = c.get("/model/state")
    assert r.status_code == 200
    data = r.json()
    # PID muerto → builder se considera abandoned, state vuelve a ready
    assert data["state"] == "ready"
