"""Tests de write_engine_meta.py."""
import json
import subprocess
import sys
from pathlib import Path


def test_write_engine_meta(tmp_path):
    engine = tmp_path / "best_fp16.engine"
    engine.write_bytes(b"X" * 1024)
    proc = subprocess.run([
        sys.executable, "scripts/write_engine_meta.py",
        str(engine),
        "65c163404ea3",
        "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad",
        "512",
        "--build-duration-s", "496",
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    meta_path = engine.with_suffix(".engine.meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["onnx_sha256"] == "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad"
    assert meta["hf_revision"] == "65c163404ea3"
    assert meta["trtexec_args"] == ["--fp16", "--workspace=512", "--buildOnly"]
    assert "engine_sha256" in meta
    assert "build_completed_at" in meta
    assert meta["build_duration_s"] == 496


def test_write_engine_meta_with_validation_json(tmp_path):
    engine = tmp_path / "best_fp16.engine"
    engine.write_bytes(b"Y" * 512)
    val = tmp_path / "validation.json"
    val.write_text(json.dumps({"passed": True, "results": [{"image": "a", "detections": 2}]}))
    proc = subprocess.run([
        sys.executable, "scripts/write_engine_meta.py",
        str(engine), "rev123", "onnx_sha_abc", "512",
        "--validation-json", str(val),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    meta = json.loads(engine.with_suffix(".engine.meta.json").read_text())
    assert meta["validation"]["passed"] is True
