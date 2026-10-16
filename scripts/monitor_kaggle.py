import os
import sys
import time
import builtins

# Fix Windows GBK encoding issue for kaggle log download
orig_open = builtins.open
def utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs and len(args) > 1 and 'b' not in args[1]:
        kwargs['encoding'] = 'utf-8'
    return orig_open(*args, **kwargs)
builtins.open = utf8_open

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_453ae5281f11c2531c00b975642eaa6b'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

kernel = 'tinacherry/mobius-attention-experiment'
print(f"Monitoring remote Kaggle kernel: {kernel}...", flush=True)

start_time = time.time()
while True:
    res = api.kernels_status(kernel)
    status_str = str(getattr(res, 'status', ''))
    fail_msg = getattr(res, 'failure_message', '') or getattr(res, 'failureMessage', '')
    elapsed = int(time.time() - start_time)
    print(f"[{elapsed}s] Kernel status: {status_str} | failureMessage: {fail_msg}", flush=True)
    
    if 'COMPLETE' in status_str:
        print(f"\nKernel completed successfully after {elapsed}s! Fetching outputs...", flush=True)
        os.makedirs('kaggle_adaptive_output', exist_ok=True)
        api.kernels_output_cli(kernel, path='kaggle_adaptive_output')
        print("Outputs fetched to kaggle_adaptive_output/ successfully!", flush=True)
        sys.exit(0)
    elif 'ERROR' in status_str or 'CANCELLED' in status_str:
        print(f"\nKernel ended with status {status_str}! Error details: {fail_msg}", flush=True)
        os.makedirs('kaggle_adaptive_output', exist_ok=True)
        api.kernels_output_cli(kernel, path='kaggle_adaptive_output')
        sys.exit(1)
        
    time.sleep(20)
