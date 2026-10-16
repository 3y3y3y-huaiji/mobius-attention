"""Utility script to monitor remote Kaggle kernel execution and fetch outputs."""

import builtins
import os
import sys
import time

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None

# Fix Windows GBK encoding issue for kaggle log download
orig_open = builtins.open


def utf8_open(*args, **kwargs):
    if "encoding" not in kwargs and len(args) > 1 and "b" not in args[1]:
        kwargs["encoding"] = "utf-8"
    return orig_open(*args, **kwargs)


builtins.open = utf8_open


def monitor_kernel(
    kernel: str = "tinacherry/mobius-attention-experiment",
    output_dir: str = "kaggle_adaptive_output",
):
    if KaggleApi is None:
        print("Error: kaggle package is not installed. Install via `pip install kaggle`.")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print(f"Monitoring remote Kaggle kernel: {kernel}...", flush=True)

    start_time = time.time()
    while True:
        res = api.kernels_status(kernel)
        status_str = str(getattr(res, "status", ""))
        fail_msg = getattr(res, "failure_message", "") or getattr(res, "failureMessage", "")
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] Kernel status: {status_str} | failureMessage: {fail_msg}", flush=True)

        if "COMPLETE" in status_str:
            print(f"\nKernel completed successfully after {elapsed}s! Fetching outputs...", flush=True)
            os.makedirs(output_dir, exist_ok=True)
            api.kernels_output_cli(kernel, path=output_dir)
            print(f"Outputs fetched to {output_dir}/ successfully!", flush=True)
            sys.exit(0)
        elif "ERROR" in status_str or "CANCELLED" in status_str:
            print(f"\nKernel ended with status {status_str}! Error details: {fail_msg}", flush=True)
            os.makedirs(output_dir, exist_ok=True)
            api.kernels_output_cli(kernel, path=output_dir)
            sys.exit(1)

        time.sleep(20)


if __name__ == "__main__":
    target_kernel = sys.argv[1] if len(sys.argv) > 1 else "tinacherry/mobius-attention-experiment"
    monitor_kernel(target_kernel)
