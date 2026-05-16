"""Tests POST /model/rollback."""
import json
from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


def test_rollback_success(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    active = tmp_path / "engines" / "best_fp16.engine"
    prev = tmp_path / "engines" / ".previous" / "best_fp16.engine.old"
    active.parent.mkdir(parents=True)
    prev.parent.mkdir(parents=True)
    active.write_bytes(b"NEW")
    prev.write_bytes(b"OLD")
    active_meta = active.parent / "best_fp16.engine.meta.json"
    prev_meta = prev.parent / "best_fp16.engine.old.meta.json"
    active_meta.write_text('{"hf_revision":"new"}')
    prev_meta.write_text('{"hf_revision":"old"}')

    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE", active)
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", active_meta)
    monkeypatch.setattr(nano_server, "PREVIOUS_ENGINE", prev)
    monkeypatch.setattr(nano_server, "PREVIOUS_ENGINE_META", prev_meta)
    monkeypatch.setattr(nano_server.worker, "request_swap", lambda p: None)

    with TestClient(app) as c:
        r = c.post("/model/rollback")
    assert r.status_code == 200
    # active should now have the old engine content
    assert active.read_bytes() == b"OLD"
    # active meta should mark from_fallback = True
    new_meta = json.loads(active_meta.read_text())
    assert new_meta.get("from_fallback") is True


def test_rollback_no_previous(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    monkeypatch.setattr(nano_server, "PREVIOUS_ENGINE", tmp_path / "nope.engine")
    with TestClient(app) as c:
        r = c.post("/model/rollback")
    assert r.status_code == 409
