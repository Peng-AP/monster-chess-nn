"""One frozen-corpus sparse-input value baseline for search-first experiments.

840 inputs: 12x64 absolute piece squares, 3 turn phases, 4 castling rights,
64 raw EP squares, remaining turn budget. Fixed WHITE value perspective.
Uses original family-isolated splits and existing shaped outcome targets.
No policy, mirroring, or checkpoint selection through test games.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from match_evidence import atomic_json, file_hash

INPUTS = 840


def features(positions):
    p = np.asarray(positions)
    if p.shape[1:] != (8, 8, 24):
        raise ValueError('Requires state-complete 24-channel source')
    out = np.zeros((len(p), INPUTS), dtype=np.float32)
    out[:, :768] = p[:, :, :, :12].transpose(0, 3, 1, 2).reshape(-1, 768)
    phase = np.where(p[:, 0, 0, 12] < 0, 2, np.where(p[:, 0, 0, 13] > 0, 1, 0))
    out[np.arange(len(p)), 768 + phase] = 1
    out[:, 771:775] = p[:, 0, 0, 18:22]
    out[:, 775:839] = p[:, :, :, 22].reshape(-1, 64)
    out[:, 839] = p[:, 0, 0, 23]
    return out


def model(width=128, hidden=32, inputs=INPUTS):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(),
                         nn.Linear(width, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1), nn.Tanh())


def export(net, path):
    """Versioned little-endian float32, first-layer feature-major for sparse sums."""
    a, b, c = net[0], net[2], net[4]
    if a.in_features not in (840,6240):
        raise ValueError('Unknown feature schema')
    magic=b'MCSV001\0' if a.in_features==840 else b'MCSV002\0'
    arrays = (a.weight.T, a.bias, b.weight, b.bias, c.weight, c.bias)
    with path.open('xb') as f:
        f.write(struct.pack('<8sIII', magic, a.in_features, a.out_features, b.out_features))
        for array in arrays:
            f.write(array.detach().cpu().contiguous().numpy().astype('<f4').tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT / 'iterations/b2_001/processed24')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=3173)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--targets', type=Path, help='Optional fixed White-perspective teacher targets')
    ap.add_argument('--features', choices=['absolute','king-relative'], default='absolute')
    ap.add_argument('--sparse-inputs', action='store_true',
                    help='Store sparse indices on GPU and construct dense minibatches; no sparse gradient kernels')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    source_names = ('positions.npy', 'game_results.npy', 'value_weights.npy', 'splits.npz', 'split_game_ids.json')
    hashes = {name: file_hash(args.data / name) for name in source_names}
    p = np.load(args.data / 'positions.npy', mmap_mode='r')
    splits = dict(np.load(args.data / 'splits.npz'))
    if set(splits) != {'train', 'val', 'test'}:
        raise ValueError(f'Unexpected splits: {list(splits)}')
    all_ids = np.concatenate(list(splits.values()))
    if len(np.unique(all_ids)) != len(p) or len(all_ids) != len(p):
        raise ValueError('Splits must partition corpus exactly')
    relative=args.features=='king-relative'
    input_count=6240 if relative else INPUTS
    if args.sparse_inputs or relative:
        from search_value_features import sparse_features,MAX_ACTIVE
        feature_ids=torch.empty((len(p),MAX_ACTIVE),dtype=torch.int32,device=args.device)
        feature_weights=torch.empty((len(p),MAX_ACTIVE),device=args.device)
        for offset in range(0,len(p),8192):
            ids,weights=sparse_features(p[offset:offset+8192],relative)
            feature_ids[offset:offset+len(ids)]=torch.from_numpy(ids).to(args.device)
            feature_weights[offset:offset+len(ids)]=torch.from_numpy(weights).to(args.device)
        def batch_features(chunk):
            dense=torch.zeros((len(chunk),input_count+MAX_ACTIVE),device=args.device)
            dense.scatter_(1,feature_ids[chunk].long(),feature_weights[chunk])
            return dense[:,:input_count]
    else:
        x = torch.empty((len(p), INPUTS), device=args.device)
        for offset in range(0, len(p), 8192):
            x[offset:offset+8192] = torch.from_numpy(features(p[offset:offset+8192])).to(args.device)
        def batch_features(chunk):
            return x[chunk]
    target_file = args.targets or args.data / 'game_results.npy'
    target_array = np.load(target_file)
    if target_array.shape != (len(p),) or not np.isfinite(target_array).all() or np.abs(target_array).max() > 1.00001:
        raise ValueError('Targets must be aligned finite White-perspective values in [-1,1]')
    y = torch.from_numpy(target_array).to(args.device)
    w = torch.from_numpy(np.load(args.data / 'value_weights.npy')).to(args.device)
    indices = {k: torch.tensor(v, device=args.device) for k, v in splits.items()}
    net = model(args.width, args.hidden, input_count).to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    source_hashes=hashes, rows=len(p), splits={k:len(v) for k,v in splits.items()},
                    parameters=sum(v.numel() for v in net.parameters()),
                    target=('fixed teacher values' if args.targets else 'existing shaped game_results')+', White perspective',
                    target_hash=file_hash(target_file), implementation=file_hash(__file__),
                    feature_schema='MCSV002' if relative else 'MCSV001',
                    feature_implementation=file_hash(ROOT/'tools/search_value_features.py'),
                    selection='lowest validation weighted MSE; test once at end')
    atomic_json(args.out / 'manifest.json', manifest)

    def evaluate(ids):
        numerator = denominator = 0.
        net.eval()
        with torch.no_grad():
            for chunk in ids.split(args.batch):
                error = (net(batch_features(chunk)).squeeze(-1) - y[chunk]).square()
                numerator += float((error*w[chunk]).sum())
                denominator += float(w[chunk].sum())
        return numerator/max(denominator, 1e-12)

    best = float('inf')
    rows = []
    for epoch in range(1, args.epochs+1):
        net.train()
        ids = indices['train'][torch.randperm(len(indices['train']), device=args.device)]
        total = mass = 0.
        for chunk in ids.split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            error = (net(batch_features(chunk)).squeeze(-1) - y[chunk]).square()
            loss = (error*w[chunk]).sum()/w[chunk].sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            total += float((error.detach()*w[chunk]).sum())
            mass += float(w[chunk].sum())
        val = evaluate(indices['val'])
        row = dict(epoch=epoch, train_mse=total/max(mass,1e-12), val_mse=val,
                   seconds=time.monotonic()-start)
        rows.append(row)
        print(json.dumps(row), flush=True)
        # Save all small snapshots; existing files are never reused by this run.
        torch.save(net.cpu().state_dict(), args.out / f'epoch_{epoch:03}.pt')
        export(net, args.out / f'epoch_{epoch:03}.bin')
        net.to(args.device)
        if val < best:
            best, best_epoch = val, epoch
        atomic_json(args.out/'progress.json', dict(epochs=rows, best_epoch=best_epoch))
    net.load_state_dict(torch.load(args.out/f'epoch_{best_epoch:03}.pt', weights_only=True))
    test = evaluate(indices['test'])
    atomic_json(args.out/'complete.json', dict(best_epoch=best_epoch, val_mse=best,
                test_mse=test, seconds=time.monotonic()-start,
                model_sha256=file_hash(args.out/f'epoch_{best_epoch:03}.bin'),
                cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None))


if __name__ == '__main__':
    main()
