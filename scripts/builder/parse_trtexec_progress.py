"""parse_trtexec_progress.py — lee stdin (output de trtexec), pasa líneas tal cual
a stdout (tee), y detecta fases para llamar builder_state.py con el progreso.

Usage: trtexec ... 2>&1 | parse_trtexec_progress.py <job_id>
"""
import re
import subprocess
import sys
from pathlib import Path

HOOKS = [
    (re.compile(r'Finished parsing network model'), 'parsing_done', 30),
    (re.compile(r'\[MemUsageChange\].*Init builder'), 'mem_init', 35),
    (re.compile(r'Engine built in'), 'engine_built', 70),
    (re.compile(r'Engine deserialized'), 'deserialized', 75),
    (re.compile(r'&&&& PASSED'), 'trtexec_passed', 78),
    (re.compile(r'&&&& FAILED'), 'trtexec_failed', None),
]
TIMING_RX = re.compile(r'Timing Runner')

BUILDER_STATE = str(Path(__file__).parent / "builder_state.py")


def update_phase(job_id, name, pct, message=None):
    cmd = [sys.executable, BUILDER_STATE, job_id, "phase",
           "--name", name, "--pct", str(pct)]
    if message:
        cmd += ["--message", message]
    subprocess.run(cmd, check=False)


def main(job_id):
    timing_count = 0
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
        matched = False
        for rx, phase, pct in HOOKS:
            if rx.search(line):
                if pct is not None:
                    update_phase(job_id, phase, pct, message=line.strip()[:200])
                matched = True
                break
        if not matched and TIMING_RX.search(line):
            timing_count += 1
            if timing_count % 20 == 0:
                pct_est = min(65, 40 + timing_count // 4)
                update_phase(job_id, "trtexec_optimizing", pct_est,
                             message=line.strip()[:200])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parse_trtexec_progress.py <job_id>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
