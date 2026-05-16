"""Tests POST /model/build."""
from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


def test_build_starts_job(monkeypatch, tmp_path):
    _stub_worker_startup(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", tmp_path / "job.json")
    called = {}
    def fake_run(cmd, **kw):
        called["cmd"] = cmd
        class R: returncode = 0; stdout = ""; stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)

    with TestClient(app) as c:
        r = c.post("/model/build", json={"force": False})
    assert r.status_code == 202
    data = r.json()
    assert data["ok"] is True
    assert "job_id" in data
    assert called["cmd"][0:2] == ["sudo", "/usr/local/bin/embebidos3-builder-launch"]


def test_build_409_when_already_active(monkeypatch, tmp_path):
    _stub_worker_startup(monkeypatch)
    import nano_server
    state_file = tmp_path / "job.json"
    state_file.write_text(
        '{"job_id": "active-jid12345", "pid": 1, "phase": "trtexec", "heartbeat": 9999999999}'
    )
    monkeypatch.setattr(nano_server, "JOB_STATE_FILE", state_file)
    monkeypatch.setattr(nano_server, "_is_pid_alive", lambda p: True)
    monkeypatch.setattr(nano_server, "_check_cmdline", lambda p: True)

    with TestClient(app) as c:
        r = c.post("/model/build")
    assert r.status_code == 409
    detail = r.json().get("detail", {})
    assert detail.get("error") == "build_in_progress"
