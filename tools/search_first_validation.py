"""Locked validation of the width512/q0 nominee; no promotion or reselection.

Fresh-for-this-experiment book indices32+ (already used by older campaigns,
not a universally held-out dataset). Compare colors, clocks, a second opponent,
and common-start selfplay. Free play is one deterministic pair, not fake samples.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from match_evidence import atomic_json,file_hash
from analyze_search_first import analyze


def main():
    out=ROOT/'benchmarks/search_first_20260911/validation'
    out.mkdir(parents=True,exist_ok=False)
    value='models/candidates/search_first_distilled_w512_001/epoch_007.bin'
    reference='models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
    second='models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'
    sources=list((ROOT/'native/src').glob('*.rs'))+[
        ROOT/'native/monster_native.pyd',ROOT/'src/native_mcts.py',ROOT/'src/evaluation.py',
        ROOT/'src/monster_chess.py',ROOT/'src/repetition.py',ROOT/'tools/search_first_match.py',
        ROOT/'tools/analyze_search_first.py',Path(__file__),ROOT/value,ROOT/reference,ROOT/second,
        ROOT/'benchmarks/b2_challenger_confirmation_20260910/confirmation_book.json']
    hashes={str(p):file_hash(p) for p in sources}
    profile=json.loads((ROOT/'benchmarks/search_first_20260911/cpu_eval_after.json').read_text())
    if not profile['complete'] or profile['runtime']!=hashes[str(ROOT/'native/monster_native.pyd')]:
        raise ValueError('Need completed profile for current runtime')
    plans=[
        ('gen47_300ms',32,32,.3,[]),
        ('gen47_2s',8,32,2.,[]),
        ('b2_300ms',16,48,.3,['--reference',second]),
        ('self_ab_300ms',16,32,.3,['--selfplay']),
        ('self_gen47_300ms',16,32,.3,['--reference-selfplay']),
        ('free_2s',1,0,2.,['--free']),
    ]
    atomic_json(out/'manifest.json',dict(hashes=hashes,stages=plans,
        nominee='width512, extension0 selected on first8 reused diagnostic starts',
        policy='no runtime edits/reselection during chain; failed tests/proofs stop chain'))

    def stage(name,command):
        if any(file_hash(p)!=h for p,h in hashes.items()):
            raise ValueError('Runtime changed during locked validation')
        atomic_json(out/'status.json',dict(stage=name,complete=False))
        print('STAGE '+name,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)

    try:
        stage('tests',['-m','pytest','tests','-q'])
        reports={}
        for name,pairs,offset,seconds,extra in plans:
            stage(name,['tools/search_first_match.py','--value',value,'--out',str(out/name),
                '--pairs',str(pairs),'--offset',str(offset),'--seconds',str(seconds),*extra])
            report=analyze(out/name)
            atomic_json(out/name/'analysis.json',report)
            if not report['complete'] or report['proof_contradictions']:
                raise ValueError(f'Invalid evidence in {name}')
            reports[name]=report
            atomic_json(out/'summary.json',dict(complete=False,stages=reports))
        atomic_json(out/'summary.json',dict(complete=True,stages=reports))
        atomic_json(out/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':main()
