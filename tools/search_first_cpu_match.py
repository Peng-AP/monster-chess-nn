"""CPU-only deployment control, not a substitute for the GPU strength test.

Same timed driver. Force CUDA invisible BEFORE importing PyTorch, one CPU thread
for its neural evaluator, and write explicit supplemental hardware provenance.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for folder in ('src','tools'):
    sys.path.insert(0,str(ROOT/folder))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from match_evidence import atomic_json,file_hash
from search_first_match import main as match_main


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--out',type=Path,required=True)
    args,_=ap.parse_known_args()
    if torch.cuda.is_available():raise RuntimeError('CPU control unexpectedly has CUDA')
    if args.out.exists():raise FileExistsError(args.out)
    # Keep the match driver's exclusive new-directory contract. A sibling receipt
    # survives a failure before games complete, and is linked by the directory name.
    receipt=args.out.with_name(args.out.name+'_hardware.json')
    if receipt.exists():raise FileExistsError(receipt)
    atomic_json(receipt,dict(driver=file_hash(__file__),torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        environment={k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        incumbent_precision='float32 on CPU (GPU comparison uses float16)',
        claim='deployment control; do not pool with GPU-backed gen47 matches'))
    match_main()


if __name__=='__main__':main()
