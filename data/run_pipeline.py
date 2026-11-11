"""Run the M3 modeling pipeline in order and stop at the first error.

Steps (RUN_ANALYSIS.md, section 2):
  train      python train_models.py all [--jobs N]
  explain    python explain.py
  confirm    python confirm_strict.py
  report     python report_models.py
Optional first step (--with-preprocess, or --from preprocess):
  preprocess python preprocess.py   (rebuilds the derived/ model tables; normally not needed)

firth.py and m3_common.py are helper modules used by these scripts, so they are not run on their own.

Each step's output appears live and is also saved to derived/logs/run_pipeline_<time>.txt.
If a step fails, the runner stops. It then shows that step's error (the traceback, or the
last lines of output if there is none), what the error usually means, and how to resume.

Run from the data/ folder:
  python run_pipeline.py                      # everything
  python run_pipeline.py --jobs 4             # training on 4 cores
  python run_pipeline.py --from explain       # resume after training finished
  python run_pipeline.py --only confirm       # one step
  python run_pipeline.py --from train --train-stages phase2 lc sens targetB finalize
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "derived" / "logs"

SCRIPTS = {
    "preprocess": "preprocess.py",
    "train": "train_models.py",
    "explain": "explain.py",
    "confirm": "confirm_strict.py",
    "report": "report_models.py",
}
DEFAULT_STEPS = ["train", "explain", "confirm", "report"]

TB_START = "Traceback (most recent call last):"
EXC_LINE = re.compile(r"^(?:[A-Za-z_][\w.]*\.)?([A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt))\b:?\s?(.*)$")
MAX_SHOWN = 120          # traceback lines repeated in the final error report
KEEP_LINES = 20000       # output lines kept in memory per step (the log file has everything)

TTY = sys.stdout.isatty()


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if TTY else text


def hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# --------------------------------------------------------------------------
# Running one step
# --------------------------------------------------------------------------

def run_step(name: str, cmd: list[str], log) -> tuple[int | None, float, list[str]]:
    """Run one command, stream its output to the terminal and the log.
    Returns (exit code or None if interrupted, seconds, kept output lines)."""
    shown = " ".join(["python", *cmd[2:]])
    banner = f"\n{'=' * 70}\n[{name}] {shown}    (started {datetime.now():%H:%M:%S})\n{'=' * 70}"
    print(color(banner, "1;36"), flush=True)
    log.write(banner + "\n")
    log.flush()

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    lines: collections.deque[str] = collections.deque(maxlen=KEEP_LINES)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace")
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line.rstrip("\n"))
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        rc = None
    log.flush()
    return rc, time.time() - t0, list(lines)


# --------------------------------------------------------------------------
# Making sense of a failure
# --------------------------------------------------------------------------

def extract_error(lines: list[str]) -> tuple[list[str], str | None, str]:
    """(error block to show, exception name, exception message)."""
    starts = [i for i, l in enumerate(lines) if l.lstrip().startswith(TB_START)]
    block = lines[starts[0]:] if starts else lines[-40:]
    for l in reversed(block):
        m = EXC_LINE.match(l.strip())
        if m:
            return block, m.group(1), m.group(2)
    return block, None, ""


def hint(name: str, rc: int | None, exc: str | None, text: str) -> str | None:
    if rc is not None and rc < 0:
        sig = -rc
        if sig == 9:
            return ("The process was killed (SIGKILL). This is usually the computer running out of memory. "
                    "Try fewer cores, e.g. --jobs 4.")
        return f"The process was killed by signal {sig}."
    if exc == "ModuleNotFoundError":
        return "A package is missing. Rerun the install command in RUN_ANALYSIS.md, step 1."
    if exc == "AssertionError":
        if "control gate" in text:
            return ("The control gate (positive/negative controls) failed, so the figures would be misleading. "
                    "Look at derived/models/control_gate.csv.")
        return ("A built-in data check failed. The message names the check. Don't skip it, "
                "because the results would not be valid.")
    if exc == "FileNotFoundError":
        return "An input file is missing. Usually an earlier step did not finish. Rerun from that step."
    if exc in ("MemoryError", "TerminatedWorkerError"):
        return "A process ran out of memory or a worker died. Try fewer cores, e.g. --jobs 4."
    if exc == "KeyError":
        return ("A column, feature or cohort name was not found. Often this means an earlier step's outputs "
                "are stale or incomplete. Rerun the earlier steps.")
    return None


def report_failure(name: str, cmd: list[str], rc: int | None, secs: float, lines: list[str],
                   log_path: Path, train_stages: list[str]) -> None:
    shown = " ".join(["python", *cmd[2:]])
    print("\n" + color("#" * 70, "1;31"))
    if rc is None:
        print(color(f"INTERRUPTED during step '{name}' ({shown}) after {hms(secs)}", "1;31"))
    else:
        print(color(f"FAILED at step '{name}' ({shown}) after {hms(secs)}, exit code {rc}", "1;31"))
    print(color("#" * 70, "1;31"))
    if rc is None:
        return

    block, exc, msg = extract_error(lines)
    if block:
        print(color("\n--- error output ---", "1"))
        if len(block) > MAX_SHOWN:
            print(f"... ({len(block) - MAX_SHOWN} earlier lines in the log file)")
            block = block[-MAX_SHOWN:]
        print("\n".join(block))
        print(color("--- end of error output ---", "1"))
    else:
        print("\n(The step printed no output.)")

    if exc:
        print("\n" + color(f"Error: {exc}" + (f": {msg}" if msg else ""), "1;31"))
    h = hint(name, rc, exc, "\n".join(block))
    if h:
        print(color("What it means: ", "1") + h)
    print(color("Full log: ", "1") + str(log_path))
    resume = f"python run_pipeline.py --from {name}"
    print(color("After fixing it, resume with: ", "1") + resume)
    if name == "train":
        print("  (train_models.py saves each finished stage. To rerun only some stages, add e.g. "
              "--train-stages phase2 lc sens targetB finalize)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Run the M3 pipeline, stopping at the first error.")
    ap.add_argument("--from", dest="start", choices=list(SCRIPTS), help="start at this step")
    ap.add_argument("--only", nargs="+", choices=list(SCRIPTS), help="run only these steps (in pipeline order)")
    ap.add_argument("--with-preprocess", action="store_true", help="run preprocess.py first")
    ap.add_argument("--jobs", type=int, help="CPU cores for train_models.py (default: its own, 8)")
    ap.add_argument("--train-stages", nargs="+", default=["all"], metavar="STAGE",
                    help="stages passed to train_models.py (default: all)")
    a = ap.parse_args()

    order = (["preprocess"] if a.with_preprocess or a.start == "preprocess" or
             (a.only and "preprocess" in a.only) else []) + DEFAULT_STEPS
    if a.only:
        steps = [s for s in order if s in a.only]
    elif a.start:
        steps = order[order.index(a.start):]
    else:
        steps = order

    missing = [SCRIPTS[s] for s in steps if not (HERE / SCRIPTS[s]).exists()]
    if missing:
        print(f"Cannot find {', '.join(missing)} next to run_pipeline.py ({HERE}). "
              "Put run_pipeline.py in the data/ folder with the other scripts.")
        return 2

    def command(step: str) -> list[str]:
        cmd = [sys.executable, "-u", SCRIPTS[step]]
        if step == "train":
            cmd += a.train_stages + (["--jobs", str(a.jobs)] if a.jobs else [])
        return cmd

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_pipeline_{datetime.now():%Y%m%d_%H%M%S}.txt"
    results: list[tuple[str, str, float]] = []
    exit_code = 0
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"run_pipeline.py started {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                  f"steps: {', '.join(steps)}\npython: {sys.executable}\n")
        print(f"Running {len(steps)} step(s): {' -> '.join(steps)}")
        print(f"Log: {log_path}")
        for step in steps:
            cmd = command(step)
            rc, secs, lines = run_step(step, cmd, log)
            if rc == 0:
                results.append((step, "ok", secs))
                print(color(f"[{step}] finished OK in {hms(secs)}", "1;32"))
                continue
            results.append((step, "INTERRUPTED" if rc is None else "FAILED", secs))
            # Mirror the failure report into the log as well.
            report_failure(step, cmd, rc, secs, lines, log_path, a.train_stages)
            log.write(f"\n[{step}] {'interrupted' if rc is None else f'failed, exit code {rc}'} after {hms(secs)}\n")
            exit_code = 130 if rc is None else 1
            break

    done = {s for s, _, _ in results}
    print("\nSummary")
    for s, status, secs in results:
        col = "32" if status == "ok" else "31"
        print(f"  {s:<11} {color(f'{status:<11}', col)} {hms(secs)}")
    for s in steps:
        if s not in done:
            print(f"  {s:<11} {'not run':<11}")
    if exit_code == 0:
        print(color("\nAll steps finished.", "1;32"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
