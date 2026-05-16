"""builder_state.py — CLI helper para escribir job state desde el builder bash.

Usage:
  builder_state.py <job_id> phase --name <fase> --pct <n> [--message <txt>] [--eta-seconds <n>]
  builder_state.py <job_id> finalize --phase done|failed|cancelled|abandoned --exit-code <n>

Env vars opcionales (para testing):
  EMBEBIDOS3_JOB_STATE_FILE  (default /run/embebidos3/job.json)
  EMBEBIDOS3_JOBS_LOGS_DIR   (default /home/jetson/embebidos-3/logs/jobs)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _state_file():
    return Path(os.environ.get("EMBEBIDOS3_JOB_STATE_FILE",
                               "/run/embebidos3/job.json"))


def _logs_dir():
    return Path(os.environ.get("EMBEBIDOS3_JOBS_LOGS_DIR",
                               "/home/jetson/embebidos-3/logs/jobs"))


def _read_current():
    f = _state_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _atomic_write(data):
    f = _state_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    # Path.replace es atómico y portable (POSIX rename + Windows MoveFileEx).
    # Path.rename falla en Windows si el destino existe.
    tmp.replace(f)


def cmd_phase(job_id, args):
    state = _read_current()
    if not state or state.get("job_id") != job_id:
        state = {
            "job_id": job_id,
            "pid": os.getppid(),
            "started_at_unix": time.time(),
            "phases_completed": [],
        }
    completed = state.get("phases_completed", [])
    if args.name not in completed:
        completed.append(args.name)
    state.update({
        "phase": args.name,
        "phases_completed": completed,
        "progress_pct": args.pct,
        "heartbeat": time.time(),
    })
    if args.message:
        state["current_message"] = args.message
    if args.eta_seconds is not None:
        state["eta_seconds"] = args.eta_seconds
    _atomic_write(state)
    print("[state] {} phase={} pct={}".format(job_id, args.name, args.pct))


def cmd_finalize(job_id, args):
    state = _read_current()
    if state.get("job_id") != job_id:
        print("[state] WARN: job_id en state file ({}) != {}".format(
            state.get("job_id"), job_id))
    state["phase"] = args.phase
    state["ended_at_unix"] = time.time()
    state["result"] = {"exit_code": args.exit_code}
    if state.get("started_at_unix"):
        state["build_duration_s"] = round(state["ended_at_unix"] - state["started_at_unix"], 1)
    logs = _logs_dir()
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "{}.json".format(job_id)).write_text(json.dumps(state, indent=2))
    sf = _state_file()
    if sf.exists():
        sf.unlink()
    print("[state] {} finalized phase={} exit_code={}".format(
        job_id, args.phase, args.exit_code))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job_id")
    sub = p.add_subparsers(dest="cmd")

    p_phase = sub.add_parser("phase")
    p_phase.add_argument("--name", required=True)
    p_phase.add_argument("--pct", type=int, required=True)
    p_phase.add_argument("--message", default=None)
    p_phase.add_argument("--eta-seconds", type=int, default=None, dest="eta_seconds")

    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("--phase", required=True,
                       choices=["done", "failed", "cancelled", "abandoned"])
    p_fin.add_argument("--exit-code", type=int, required=True)

    args = p.parse_args()
    if not args.cmd:
        p.error("subcommand required: phase | finalize")

    if args.cmd == "phase":
        cmd_phase(args.job_id, args)
    elif args.cmd == "finalize":
        cmd_finalize(args.job_id, args)


if __name__ == "__main__":
    main()
