"""Fixed representation control, with conditional fresh-start confirmation.

One heavy job at a time. No checkpoint fishing or production promotion. Check
runtime/data hashes between stages; preserve completed evidence on any failure.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from match_evidence import atomic_json,file_hash
from analyze_search_first import analyze


def main():
    out=ROOT/'benchmarks/search_first_relative_20260911'
    out.mkdir(parents=True,exist_ok=False)
    targets=ROOT/'benchmarks/search_first_20260911/compression_control/labels/teacher_values.npy'
    target_receipt=json.loads(targets.with_name('complete.json').read_text())
    if file_hash(targets)!=target_receipt['labels']:raise ValueError('Teacher cache changed')
    if not (ROOT/'benchmarks/search_first_20260911/cpu_control/complete.json').exists():
        raise ValueError('CPU baseline must finish first')
    files=list((ROOT/'native/src').glob('*.rs'))+[
        ROOT/'native/monster_native.pyd',ROOT/'src/native_mcts.py',ROOT/'src/evaluation.py',
        ROOT/'tools/search_first_match.py',ROOT/'tools/train_search_value.py',
        ROOT/'tools/search_value_features.py',ROOT/'tools/profile_relative_value.py',
        ROOT/'tools/analyze_search_first.py',Path(__file__)]
    hashes={str(p):file_hash(p) for p in files}
    atomic_json(out/'manifest.json',dict(hashes=hashes,teacher=target_receipt,
        width=512,hidden=32,epochs=30,seed=3173,development_indices=[64,80],
        confirmation_trigger='relative minus absolute score >= 0.10 on development starts',
        note='development trigger only; not a promotion threshold; both arms receive games'))
    def stage(name,command):
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Pinned runtime changed')
        atomic_json(out/'status.json',dict(stage=name,complete=False))
        print('STAGE '+name,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    reports={}
    def play(name,value,pairs,offset,extra=()):
        stage(name,['tools/search_first_match.py','--value',str(value),'--pairs',str(pairs),
            '--offset',str(offset),'--seconds','.3','--out',str(out/name),*extra])
        result=analyze(out/name)
        atomic_json(out/name/'analysis.json',result)
        if not result['complete'] or result['proof_contradictions']:raise ValueError('Invalid match evidence')
        reports[name]=result
        atomic_json(out/'summary.json',dict(complete=False,stages=reports))
        return result
    try:
        stage('tests',['-m','pytest','tests','-q'])
        dirs={};values={}
        for name,features in [('absolute','absolute'),('relative','king-relative')]:
            directory=ROOT/'models/candidates'/f'search_relative_{name}_001'
            stage('train_'+name,['tools/train_search_value.py','--width','512','--features',features,
                '--sparse-inputs','--targets',str(targets),'--out',str(directory)])
            receipt=json.loads((directory/'complete.json').read_text())
            if receipt['cuda_peak_bytes']>12*1024**3:raise ValueError('VRAM ceiling exceeded')
            dirs[name]=directory
            values[name]=directory/f"epoch_{receipt['best_epoch']:03}.bin"
        stage('profile',['tools/profile_relative_value.py','--absolute',str(dirs['absolute']),
            '--relative',str(dirs['relative']),'--out',str(out/'profile.json')])
        a=play('development_absolute',values['absolute'],16,64)
        b=play('development_relative',values['relative'],16,64)
        delta=b['scores']['overall']['score']-a['scores']['overall']['score']
        confirm=delta>=.10
        atomic_json(out/'decision.json',dict(development_score_delta=delta,confirmation_triggered=confirm,
            promotion=False,note='small development sample, not an independent strength claim'))
        if confirm:
            value=values['relative']
            play('confirmation_gen47',value,32,128)
            play('confirmation_b2',value,16,128,['--reference',
                'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'])
            play('self_relative',value,16,128,['--selfplay'])
            play('self_gen47',value,16,128,['--reference-selfplay'])
            stage('human',['tools/probe_search_first_human.py','--value',str(value),
                          '--out',str(out/'human.json')])
        atomic_json(out/'summary.json',dict(complete=True,stages=reports))
        atomic_json(out/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':main()
