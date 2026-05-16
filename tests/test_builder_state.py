"""Tests de builder_state.py CLI."""
import json
import os
import subprocess
import sys
from pathlib import Path


def run_helper(*args, env=None):
    """Invoca builder_state.py como subprocess."""
    cmd = [sys.executable, "scripts/builder_state.py"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_phase_writes_state(tmp_path):
    state_file = tmp_path / "job.json"
    env = {**os.environ, "EMBEBIDOS3_JOB_STATE_FILE": str(state_file)}
    r = run_helper("test-job-001234", "phase", "--name", "acquired_lock", "--pct", "5", env=env)
    assert r.returncode == 0, r.stderr
    state = json.loads(state_file.read_text())
    assert state["job_id"] == "test-job-001234"
    assert state["phase"] == "acquired_lock"
    assert state["progress_pct"] == 5
    assert "heartbeat" in state
    assert state["phases_completed"] == ["acquired_lock"]


def test_phase_appends_completed(tmp_path):
    state_file = tmp_path / "job.json"
    env = {**os.environ, "EMBEBIDOS3_JOB_STATE_FILE": str(state_file)}
    run_helper("test-job-abcdef", "phase", "--name", "phase_one", "--pct", "10", env=env)
    run_helper("test-job-abcdef", "phase", "--name", "phase_two", "--pct", "20", env=env)
    state = json.loads(state_file.read_text())
    assert state["phases_completed"] == ["phase_one", "phase_two"]
    assert state["phase"] == "phase_two"


def test_finalize_moves_to_logs(tmp_path):
    state_file = tmp_path / "job.json"
    logs_dir = tmp_path / "logs"
    env = {**os.environ,
           "EMBEBIDOS3_JOB_STATE_FILE": str(state_file),
           "EMBEBIDOS3_JOBS_LOGS_DIR": str(logs_dir)}
    run_helper("test-job-finalize", "phase", "--name", "phase_one", "--pct", "10", env=env)
    run_helper("test-job-finalize", "finalize", "--phase", "done", "--exit-code", "0", env=env)
    assert not state_file.exists()
    final = json.loads((logs_dir / "test-job-finalize.json").read_text())
    assert final["phase"] == "done"
    assert final["result"]["exit_code"] == 0


def test_phase_with_optional_args(tmp_path):
    state_file = tmp_path / "job.json"
    env = {**os.environ, "EMBEBIDOS3_JOB_STATE_FILE": str(state_file)}
    r = run_helper("test-job-opts111", "phase",
                   "--name", "trtexec_optimizing",
                   "--pct", "47",
                   "--message", "Timing Runner: Conv_142",
                   "--eta-seconds", "1200",
                   env=env)
    assert r.returncode == 0
    state = json.loads(state_file.read_text())
    assert state["current_message"] == "Timing Runner: Conv_142"
    assert state["eta_seconds"] == 1200
