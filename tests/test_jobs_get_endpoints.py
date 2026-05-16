"""Tests GET /jobs/active y GET /jobs/<id>."""
import json
from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


def test_jobs_active_none(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "job.json")
    with TestClient(app) as c:
        r = c.get("/jobs/active")
    assert r.status_code == 200
    assert r.json() is None


def test_jobs_get_by_id_terminal(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    logs_dir = tmp_path / "jobs"
    logs_dir.mkdir()
    (logs_dir / "test-jid-001234.json").write_text(json.dumps({
        "job_id": "test-jid-001234",
        "phase": "done",
        "result": {"exit_code": 0},
    }))
    monkeypatch.setattr(nano_server, "JOBS_LOGS_DIR", logs_dir)
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.get("/jobs/test-jid-001234")
    assert r.status_code == 200
    assert r.json()["phase"] == "done"


def test_jobs_get_by_id_404(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    logs_dir = tmp_path / "jobs"
    logs_dir.mkdir()
    monkeypatch.setattr(nano_server, "JOBS_LOGS_DIR", logs_dir)
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.get("/jobs/inexistente1")
    assert r.status_code == 404


def test_jobs_get_invalid_id_short(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")
    monkeypatch.setattr(nano_server, "JOBS_LOGS_DIR", tmp_path / "jobs")
    with TestClient(app) as c:
        r = c.get("/jobs/abc")  # 3 chars, too short
    assert r.status_code == 422
    assert r.json().get("detail", {}).get("error") == "invalid_job_id"


def test_jobs_get_invalid_id_chars(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")
    monkeypatch.setattr(nano_server, "JOBS_LOGS_DIR", tmp_path / "jobs")
    with TestClient(app) as c:
        # 12 chars but contains a forbidden ! character
        r = c.get("/jobs/abc!def!ghi1")
    assert r.status_code == 422
