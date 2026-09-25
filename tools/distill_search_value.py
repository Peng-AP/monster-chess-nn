"""Cache fixed gen47 raw value predictions for a cheap-evaluator control.

This is compression of existing judgment, NOT new ground truth or deeper search.
Original outcome-trained candidate stays intact; original family splits retained.
"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from train import load_model_for_inference
from match_evidence import atomic_json, file_hash
from worker_lease import worker_lease


def teacher_input(p, channels):
    if channels == 24:
        out = np.array(p, copy=True)
    elif channels == 17:
        out = np.array(p[..., :17], copy=True)
    elif channels == 15:
        out = np.array(p[..., :15], copy=True)
        # Legacy plane14 is white pawn progress, NOT signed rank (B2 plane14).
        out[..., 14] = p[..., 15]
    else:
        raise ValueError(f'Unsupported teacher encoding: {channels}')
    return np.ascontiguousarray(out.transpose(0,3,1,2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT/'iterations/b2_001/processed24')
    ap.add_argument('--teacher', default='models/candidates/bootstrap_main_gen_0047/arena_selected.pt')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=1024)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    p = np.load(args.data/'positions.npy', mmap_mode='r')
    output = np.lib.format.open_memmap(args.out/'teacher_values.npy', mode='w+',
                                     dtype=np.float32, shape=(len(p),))
    with worker_lease():
        net, _ = load_model_for_inference(args.teacher, torch.device('cuda'))
        net.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for offset in range(0, len(p), args.batch):
                batch = p[offset:offset+args.batch]
                x = torch.from_numpy(teacher_input(batch, net.input_channels)).cuda()
                values, _ = net(x)
                output[offset:offset+len(batch)] = values.squeeze(-1).float().cpu().numpy()*batch[:,0,0,12]
                if offset % (args.batch*100) == 0:
                    print(f'distillation {offset+len(batch)}/{len(p)} rows, {time.monotonic()-start:.1f}s', flush=True)
        output.flush()
        atomic_json(args.out/'complete.json', dict(rows=len(p), teacher=file_hash(args.teacher),
            positions=file_hash(args.data/'positions.npy'), split=file_hash(args.data/'splits.npz'),
            labels=file_hash(args.out/'teacher_values.npy'), implementation=file_hash(__file__),
            seconds=time.monotonic()-start, cuda_peak_bytes=torch.cuda.max_memory_allocated(),
            label='raw teacher scalar value converted to White perspective; no terminal clamps'))


if __name__ == '__main__':
    main()
