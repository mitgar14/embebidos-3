"""Tests de recovery: server arranca y detecta job activo/huérfano/stalled."""
import json
import os
import time

import recover_job_state as rjs


def test_no_state_file(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    monkeypatch.setattr(rjs, "JOB_STATE_FILE", state_file)
    assert rjs.recover_job_state() is None


def test_dead_pid_returns_none(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    state_file.write_text(json.dumps({
        "job_id": "test-job-001234",
        "pid": 99999999,
        "phase": "trtexec_building",
        "heartbeat": time.time(),
    }))
    monkeypatch.setattr(rjs, "JOB_STATE_FILE", state_file)
    monkeypatch.setattr(rjs, "JOBS_LOGS_DIR", tmp_path / "jobs")
    assert rjs.recover_job_state() is None


def test_alive_pid_with_fresh_heartbeat_returns_running(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    state_file.write_text(json.dumps({
        "job_id": "test-job-001234",
        "pid": os.getpid(),
        "phase": "trtexec_building",
        "heartbeat": time.time(),
        "progress_pct": 50,
    }))
    monkeypatch.setattr(rjs, "JOB_STATE_FILE", state_file)
    monkeypatch.setattr(rjs, "JOBS_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(rjs, "_check_cmdline", lambda p: True)
    state = rjs.recover_job_state()
    assert state is not None
    assert state["status"] == "running"
    assert state["job_id"] == "test-job-001234"


def test_alive_pid_with_stale_heartbeat_returns_stalled(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    state_file.write_text(json.dumps({
        "job_id": "test-job-001234",
        "pid": os.getpid(),
        "phase": "trtexec_building",
        "heartbeat": time.time() - 200,
        "progress_pct": 50,
    }))
    monkeypatch.setattr(rjs, "JOB_STATE_FILE", state_file)
    monkeypatch.setattr(rjs, "JOBS_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(rjs, "_check_cmdline", lambda p: True)
    state = rjs.recover_job_state()
    assert state["status"] == "stalled"
