"""Tests POST /model/check-updates."""
import json
from fastapi.testclient import TestClient
from nano_server import app
from tests.test_model_state_endpoint import _stub_worker_startup


def test_check_updates_up_to_date(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    meta_path = tmp_path / "best_fp16.engine.meta.json"
    meta_path.write_text(json.dumps({"hf_revision": "65c1634abc"}))
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", meta_path)
    monkeypatch.setattr("hf_rest.get_head_revision", lambda: "65c1634abc")
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["up_to_date"] is True


def test_check_updates_new_commit(tmp_path, monkeypatch):
    _stub_worker_startup(monkeypatch)
    import nano_server
    meta_path = tmp_path / "best_fp16.engine.meta.json"
    meta_path.write_text(json.dumps({"hf_revision": "65c1634abc"}))
    monkeypatch.setattr(nano_server, "ACTIVE_ENGINE_META", meta_path)
    monkeypatch.setattr("hf_rest.get_head_revision", lambda: "7a3b8e2new")
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["up_to_date"] is False
    assert data["latest_revision"] == "7a3b8e2new"
    assert data["current_revision"] == "65c1634abc"
