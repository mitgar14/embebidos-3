"""Tests SSE /jobs/<id>/logs."""
from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


def test_jobs_logs_sse_streams_existing(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    logs_dir = tmp_path / "jobs"
    logs_dir.mkdir()
    log = logs_dir / "test-jid-001234.log"
    log.write_text("[I] line one\n[I] line two\n&&&& PASSED\n")
    final = logs_dir / "test-jid-001234.json"
    final.write_text('{"job_id": "test-jid-001234", "phase": "done", "result": {"exit_code": 0}}')
    monkeypatch.setattr(nano_server, "JOBS_LOGS_DIR", logs_dir)
    with TestClient(app) as c:
        with c.stream("GET", "/jobs/test-jid-001234/logs", params={"follow": False}) as r:
            assert r.status_code == 200
            body = b"".join(r.iter_bytes())
    text = body.decode()
    assert "line one" in text
    assert "PASSED" in text
    assert "event: done" in text


def test_jobs_logs_404(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    logs_dir = tmp_path / "jobs"
    logs_dir.mkdir()
    monkeypatch.setattr(nano_server, "JOBS_LOGS_DIR", logs_dir)
    with TestClient(app) as c:
        r = c.get("/jobs/nonexistent12/logs")
    assert r.status_code == 404


def test_jobs_logs_invalid_id(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server, "JOBS_LOGS_DIR", tmp_path / "jobs")
    with TestClient(app) as c:
        r = c.get("/jobs/abc/logs")
    assert r.status_code == 422
