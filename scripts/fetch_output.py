import os

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_453ae5281f11c2531c00b975642eaa6b'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

kernel = 'tinacherry/mobius-attention-experiment'
print(f"Fetching output for {kernel}...")
api.kernels_output_cli(kernel, path='kaggle_adaptive_output')
print("Done!")
