"""Run the final same-capacity label control after locked validation completes.

Outcome512 and recorded-search512, same original split/seed/optimizer, one offline
nominee each, 16 actual games each on the original development starts. Then probe
known human lines with the already locked raw-distilled512 nominee. No promotion.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from match_evidence import atomic_json,file_hash
from analyze_search_first import analyze


def main():
    out=ROOT/'benchmarks/search_first_20260911/label_control'
    out.mkdir(parents=True,exist_ok=False)
    files=list((ROOT/'native/src').glob('*.rs'))+[
        ROOT/'native/monster_native.pyd',ROOT/'src/native_mcts.py',ROOT/'src/evaluation.py',
        ROOT/'tools/train_search_value.py',ROOT/'tools/search_first_match.py',
        ROOT/'tools/prepare_search_targets.py',ROOT/'tools/probe_search_first_human.py',
        ROOT/'tools/probe_human_line.py',Path(__file__)]
    hashes={str(p):file_hash(p) for p in files}
    atomic_json(out/'manifest.json',dict(hashes=hashes,
        arms=['outcome512','recorded_search512'],width=512,hidden=32,epochs=30,seed=3173,
        selection='minimum validation MSE, no test game checkpoint selection',
        note='last planned label control; same first8 development starts, not confirmation'))
    def stage(name,command):
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
        atomic_json(out/'status.json',dict(stage=name,complete=False))
        print('STAGE '+name,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    try:
        prerequisite=ROOT/'benchmarks/search_first_20260911/validation'
        atomic_json(out/'status.json',dict(stage='waiting_for_validation',complete=False))
        while not (prerequisite/'summary.json').exists() or not json.loads((prerequisite/'summary.json').read_text())['complete']:
            if (prerequisite/'failure.json').exists():raise ValueError('Prerequisite validation failed')
            time.sleep(30)
        stage('prepare_targets',['tools/prepare_search_targets.py','--out',str(out/'targets')])
        reports={}
        for name,folder,extra in [
            ('outcome512','search_first_outcome_w512_001',[]),
            ('recorded_search512','search_first_search_w512_001',['--targets',str(out/'targets/targets.npy')])]:
            model_dir=ROOT/'models/candidates'/folder
            stage('train_'+name,['tools/train_search_value.py','--width','512','--out',str(model_dir),*extra])
            receipt=json.loads((model_dir/'complete.json').read_text())
            value=model_dir/f"epoch_{receipt['best_epoch']:03}.bin"
            if file_hash(value)!=receipt['model_sha256']:raise ValueError('Changed model')
            stage('play_'+name,['tools/search_first_match.py','--value',str(value),
                '--out',str(out/name),'--pairs','8','--seconds','.3'])
            report=analyze(out/name)
            atomic_json(out/name/'analysis.json',report)
            if not report['complete'] or report['proof_contradictions']:raise ValueError('Invalid match evidence')
            reports[name]=report
            atomic_json(out/'summary.json',dict(complete=False,arms=reports))
        stage('human_diagnostic',['tools/probe_search_first_human.py','--out',str(out/'human.json')])
        atomic_json(out/'summary.json',dict(complete=True,arms=reports))
        atomic_json(out/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':main()
