"""Locked 2x2 development screen: evaluator width128/512 x threat extensions0/2.

Same frozen teacher targets and search runtime. This is selection, NOT held-out
confirmation. Every nominee gets actual games; no per-epoch sweep or promotion.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from match_evidence import atomic_json,file_hash


def main():
    out=ROOT/'benchmarks/search_first_20260911/search_sweep'
    out.mkdir(parents=True,exist_ok=False)
    sources=['native/monster_native.pyd','native/src/alphabeta.rs','native/src/tactical.rs',
             'native/src/search_order.rs','native/src/search_cache.rs','native/src/search_bounds.rs',
             'src/native_mcts.py','tools/search_first_match.py',__file__]
    hashes={p:file_hash(p) for p in sources}
    models={}
    for width,folder in [(128,'search_first_distilled_001'),(512,'search_first_distilled_w512_001')]:
        directory=ROOT/'models/candidates'/folder
        receipt=json.loads((directory/'complete.json').read_text())
        path=directory/f"epoch_{receipt['best_epoch']:03}.bin"
        if file_hash(path)!=receipt['model_sha256']:raise ValueError('Changed model')
        models[width]=str(path)
    atomic_json(out/'manifest.json',dict(runtime=hashes,models=models,
        arms=[dict(width=w,extensions=q) for w,q in [(128,0),(128,2),(512,0),(512,2)]],
        note='same first8 diagnostic starts; development selection only; soft clocks with full timings'))

    def stage(name,command):
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed during locked screen')
        atomic_json(out/'status.json',dict(stage=name,complete=False))
        print('STAGE '+name,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)

    try:
        stage('tests',['-m','pytest','tests','-q'])
        rows=[]
        for width,extensions in [(128,0),(128,2),(512,0),(512,2)]:
            name=f'w{width}_q{extensions}'
            stage(name,['tools/search_first_match.py','--value',models[width],
                '--out',str(out/name),'--pairs','8','--seconds','.3','--extensions',str(extensions)])
            row=json.loads((out/name/'complete.json').read_text())
            rows.append(dict(width=width,extensions=extensions,**row))
            atomic_json(out/'summary.json',dict(complete=False,arms=rows))
        atomic_json(out/'summary.json',dict(complete=True,arms=rows))
        atomic_json(out/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':main()
