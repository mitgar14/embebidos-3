"""write_engine_meta.py — escribe el .meta.json al lado del engine.
Usage: write_engine_meta.py <engine_path> <hf_revision> <onnx_sha256> <workspace_mb>
         [--build-duration-s <n>] [--validation-json <path>] [--hf-commit-date <iso>]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("engine_path")
    p.add_argument("hf_revision")
    p.add_argument("onnx_sha256")
    p.add_argument("workspace_mb", type=int)
    p.add_argument("--build-duration-s", type=int, default=None, dest="build_duration_s")
    p.add_argument("--validation-json", default=None, dest="validation_json")
    p.add_argument("--hf-commit-date", default=None, dest="hf_commit_date")
    args = p.parse_args()

    engine = Path(args.engine_path)
    if not engine.exists():
        print("engine no existe: {}".format(engine), file=sys.stderr)
        sys.exit(2)

    meta = {
        "engine_sha256": sha256_file(engine),
        "onnx_sha256": args.onnx_sha256,
        "hf_revision": args.hf_revision,
        "hf_commit_date": args.hf_commit_date,
        "trtexec_args": ["--fp16", "--workspace={}".format(args.workspace_mb), "--buildOnly"],
        "build_completed_at": datetime.now(timezone.utc).isoformat(),
        "build_duration_s": args.build_duration_s,
    }
    if args.validation_json:
        try:
            meta["validation"] = json.loads(Path(args.validation_json).read_text())
        except Exception as e:
            meta["validation"] = {"passed": None, "error": str(e)}

    meta_path = engine.with_suffix(".engine.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("OK: {}".format(meta_path))


if __name__ == "__main__":
    main()
