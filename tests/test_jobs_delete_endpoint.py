"""Tests DELETE /jobs/<id> (cancelar build)."""
from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


def test_cancel_active_job(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    state_file = tmp_path / "job.json"
    state_file.write_text(
        '{"job_id": "test-jid-cancel", "pid": 1, "phase": "trtexec_building", "heartbeat": 9999999999}'
    )
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", state_file)
    monkeypatch.setattr(nano_server, "_is_pid_alive", lambda p: True)
    monkeypatch.setattr(nano_server, "_check_cmdline", lambda p: True)
    called = {}
    def fake_run(cmd, **kw):
        called["cmd"] = cmd
        class R: returncode = 0; stdout = ""; stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)

    with TestClient(app) as c:
        r = c.delete("/jobs/test-jid-cancel")
    assert r.status_code == 200
    assert r.json()["phase"] == "cancelling"
    assert called["cmd"][:3] == ["sudo", "/bin/systemctl", "stop"]


def test_cancel_unknown_job_404(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.delete("/jobs/unknown-jid01")
    assert r.status_code == 404
