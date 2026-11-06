"""Poll the E2 Kaggle kernel until it finishes, then download all outputs.

Usage:
    python scripts/monitor_kaggle_e2.py [kernel_slug] [output_dir] [max_minutes]
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "tinacherry/phase-reversal-e2"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "kaggle_phase_e2_output")
MAX_SECONDS = float(sys.argv[3]) * 60 if len(sys.argv) > 3 else 240 * 60
POLL_SECONDS = 30


def _log(message: str, *, quiet: bool = False) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    if not quiet:
        print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "monitor.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    api = KaggleApi()
    api.authenticate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"Monitoring {KERNEL}, outputs -> {OUT_DIR}, budget {MAX_SECONDS / 60:.0f} min")

    started = time.monotonic()
    terminal = None
    last_state = None
    while time.monotonic() - started < MAX_SECONDS:
        try:
            status = api.kernels_status(KERNEL)
        except Exception as exc:  # noqa: BLE001 - transient API errors must not abort polling
            _log(f"status query failed: {exc}", quiet=True)
            time.sleep(POLL_SECONDS)
            continue
        state = str(getattr(status, "status", ""))
        failure = getattr(status, "failureMessage", "") or getattr(status, "failure_message", "") or ""
        _log(f"status={state} failure={failure}", quiet=(state == last_state))
        last_state = state
        if "COMPLETE" in state:
            terminal = "COMPLETE"
            break
        if "ERROR" in state or "CANCELLED" in state or "FAILED" in state:
            terminal = state
            break
        time.sleep(POLL_SECONDS)

    if terminal is None:
        _log("Timed out waiting for the kernel; attempting a partial output download.")

    _log(f"Terminal state: {terminal or 'TIMEOUT'}. Downloading outputs...")
    try:
        api.kernels_output_cli(KERNEL, path=str(OUT_DIR))
        _log("Output download finished.")
    except Exception as exc:  # noqa: BLE001 - report but keep the status record
        _log(f"Output download failed: {exc}")

    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            _log(f"file: {path.relative_to(OUT_DIR)} ({path.stat().st_size} bytes)")
    return 0 if terminal == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
