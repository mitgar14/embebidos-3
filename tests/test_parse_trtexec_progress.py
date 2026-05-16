"""Tests del parser de progreso de trtexec."""
import json
import os
import subprocess
import sys
from pathlib import Path

SAMPLE_LINES = [
    "[12:34:56] [I] Finished parsing network model. Parse time: 1.234",
    "[12:34:58] [I] [TRT] [MemUsageChange] Init builder: 234 MiB",
    "[12:35:02] [V] [TRT] --------------- Timing Runner: Conv_42 (CaskConvolution)",
    "[12:35:05] [V] [TRT] --------------- Timing Runner: Conv_43 (CaskConvolution)",
    "[12:42:18] [I] Engine built in 442.12 sec.",
    "[12:42:19] [I] [TRT] Loaded engine size: 13 MiB",
    "[12:42:20] [I] Engine deserialized in 0.92 sec.",
    "&&&& PASSED TensorRT.trtexec [TensorRT v8201]",
]


def test_parser_emits_phase_for_known_lines(tmp_path):
    state_file = tmp_path / "job.json"
    logs_dir = tmp_path / "logs"
    env = {**os.environ,
           "EMBEBIDOS3_JOB_STATE_FILE": str(state_file),
           "EMBEBIDOS3_JOBS_LOGS_DIR": str(logs_dir)}
    proc = subprocess.run(
        [sys.executable, "scripts/parse_trtexec_progress.py", "test-jid-aabbccdd"],
        input="\n".join(SAMPLE_LINES),
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    state = json.loads(state_file.read_text())
    completed = state["phases_completed"]
    assert "parsing_done" in completed
    assert "engine_built" in completed


def test_parser_passes_through_stdout(tmp_path):
    state_file = tmp_path / "job.json"
    env = {**os.environ, "EMBEBIDOS3_JOB_STATE_FILE": str(state_file)}
    sample = "line one\nline two\nline three"
    proc = subprocess.run(
        [sys.executable, "scripts/parse_trtexec_progress.py", "test-jid-passthru"],
        input=sample, capture_output=True, text=True, env=env,
    )
    assert "line one" in proc.stdout
    assert "line two" in proc.stdout
    assert "line three" in proc.stdout
