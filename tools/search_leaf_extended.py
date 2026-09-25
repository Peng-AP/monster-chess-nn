"""Fresh-start clock/scaling confirmation after the leaf campaign succeeds.

No more training, adaptive architecture changes, or promotion. Select one arm
using development games, then compare on a disjoint fresh paired-start set.
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
    earlier=ROOT/'benchmarks/search_leaf_campaign_20260911'
    summary=json.loads((earlier/'summary.json').read_text())
    if not summary['complete'] or not json.loads((earlier/'replay_audit.json').read_text())['complete']:
        raise ValueError('Initial campaign must complete and pass replay audit')
    previous=json.loads((earlier/'manifest.json').read_text())['hashes']
    if any(file_hash(p)!=h for p,h in previous.items()):raise ValueError('Original campaign runtime changed')
    out=ROOT/'benchmarks/search_leaf_extended_20260911'
    out.mkdir(parents=True,exist_ok=False)
    scores={arm:summary['stages']['development_'+arm]['scores']['overall']['score']
            for arm in ('original','replay','leaf')}
    # Stable ties favor the unchanged control, then replay; no leaf favoritism.
    nominee=max(scores,key=scores.get)
    original=ROOT/'models/candidates/search_relative_absolute_001/epoch_007.bin'
    values={'original':original}
    for arm in ('replay','leaf'):
        directory=ROOT/'models/candidates'/f'search_leaf_{arm}_001'
        receipt=json.loads((directory/'complete.json').read_text())
        values[arm]=directory/f"epoch_{receipt['best_epoch']:03}.bin"
        if file_hash(values[arm])!=receipt['model_sha256']:raise ValueError('Model changed')
    hashes={**previous,str(Path(__file__)):file_hash(__file__),
            **{str(p):file_hash(p) for p in values.values()},
            str(ROOT/'tools/search_first_cpu_match.py'):file_hash(ROOT/'tools/search_first_cpu_match.py')}
    atomic_json(out/'manifest.json',dict(hashes=hashes,nominee=nominee,development_scores=scores,
        fresh_pair_indices=[224,256],seconds=2,per_half_move=True,promotion=False,
        note='nomination only uses development games; fresh comparison is independent of those starts'))
    reports={}
    def stage(name,command):
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Pinned runtime changed')
        atomic_json(out/'status.json',dict(stage=name,complete=False))
        print('STAGE '+name,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    def play(name,value,pairs,offset,seconds=2.,extra=(),cpu=False):
        stage(name,['tools/search_first_cpu_match.py' if cpu else 'tools/search_first_match.py',
            '--value',str(value),'--out',str(out/name),'--pairs',str(pairs),
            '--offset',str(offset),'--seconds',str(seconds),*extra])
        report=analyze(out/name)
        if not report['complete'] or report['proof_contradictions']:raise ValueError('Invalid match evidence')
        atomic_json(out/name/'analysis.json',report);reports[name]=report
        atomic_json(out/'summary.json',dict(complete=False,nominee=nominee,stages=reports))
    try:
        play('fresh_original_gen47',original,32,224)
        if nominee!='original':
            play('fresh_nominee_gen47',values[nominee],32,224)
        play('fresh_nominee_b2',values[nominee],16,224,extra=['--reference',
             'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'])
        play('cpu_nominee',values[nominee],32,256,.3,cpu=True)
        play('self_nominee_2s',values[nominee],16,224,extra=['--selfplay'])
        play('self_gen47_2s',values[nominee],16,224,extra=['--reference-selfplay'])
        stage('replay_audit',['tools/audit_search_first_games.py','--root',str(out),
                             '--out',str(out/'replay_audit.json')])
        atomic_json(out/'summary.json',dict(complete=True,nominee=nominee,stages=reports,
            promotion=False,note='Read per-color scores, paired uncertainty and actual clocks; no automatic release'))
        atomic_json(out/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
