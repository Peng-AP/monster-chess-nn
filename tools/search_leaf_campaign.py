"""Receipt-checked, sequential search-leaf training and actual-play experiment."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from match_evidence import atomic_json,file_hash
from analyze_search_first import analyze


def main():
    out=ROOT/'benchmarks/search_leaf_campaign_20260911'
    corpus=ROOT/'benchmarks/search_leaf_corpus_20260911'
    if not json.loads((corpus/'complete.json').read_text())['complete']:
        raise ValueError('Required corpus did not complete successfully')
    out.mkdir(parents=True,exist_ok=False)
    files=list((ROOT/'native/src').glob('*.rs'))+[
        ROOT/'native/monster_native.pyd',ROOT/'src/native_mcts.py',ROOT/'src/evaluation.py',
        ROOT/'tools/search_first_match.py',ROOT/'tools/train_search_leaves.py',
        ROOT/'tools/train_search_value.py',ROOT/'tools/search_value_features.py',
        ROOT/'tools/analyze_search_first.py',Path(__file__)]
    hashes={str(p):file_hash(p) for p in files}
    atomic_json(out/'manifest.json',dict(hashes=hashes,corpus=file_hash(corpus/'samples.json'),
        development=[96,112],confirmation=[160,192],long_clock=[128,136],
        trigger='leaf score >= max(original,replay)+0.10 on development',promotion=False))
    reports={}
    def stage(name,command):
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Pinned implementation changed')
        atomic_json(out/'status.json',dict(stage=name,complete=False))
        print('STAGE '+name,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    def play(name,value,pairs,offset,seconds=.3,extra=()):
        stage(name,['tools/search_first_match.py','--value',str(value),'--out',str(out/name),
            '--pairs',str(pairs),'--offset',str(offset),'--seconds',str(seconds),*extra])
        report=analyze(out/name)
        if not report['complete'] or report['proof_contradictions']:raise ValueError('Invalid play evidence')
        atomic_json(out/name/'analysis.json',report);reports[name]=report
        atomic_json(out/'summary.json',dict(complete=False,stages=reports))
        return report['scores']['overall']['score']
    try:
        stage('tests',['-m','pytest','tests','-q'])
        values={'original':ROOT/'models/candidates/search_relative_absolute_001/epoch_007.bin'}
        for arm in ('replay','leaf'):
            directory=ROOT/'models/candidates'/f'search_leaf_{arm}_001'
            stage('train_'+arm,['tools/train_search_leaves.py','--corpus',str(corpus),
                '--out',str(directory),'--arm',arm])
            receipt=json.loads((directory/'complete.json').read_text())
            values[arm]=directory/f"epoch_{receipt['best_epoch']:03}.bin"
        scores={arm:play('development_'+arm,value,16,96) for arm,value in values.items()}
        confirm=scores['leaf']>=max(scores['original'],scores['replay'])+.10
        atomic_json(out/'decision.json',dict(scores=scores,confirmation_triggered=confirm,
            note='Development selection, not proof of strength; no promotion'))
        if confirm:
            play('confirmation_gen47',values['leaf'],32,160)
            play('confirmation_b2',values['leaf'],16,160,extra=['--reference',
                'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'])
        # Matched higher-clock control regardless of short-clock nomination.
        for arm in ('original','leaf'):
            play('long_'+arm,values[arm],8,128,2.)
        play('self_leaf',values['leaf'],16,160,extra=['--selfplay'])
        play('self_gen47',values['leaf'],16,160,extra=['--reference-selfplay'])
        stage('human',['tools/probe_search_first_human.py','--value',str(values['leaf']),
                      '--out',str(out/'human.json')])
        stage('replay_audit',['tools/audit_search_first_games.py','--root',str(out),
                              '--out',str(out/'replay_audit.json')])
        atomic_json(out/'summary.json',dict(complete=True,stages=reports))
        atomic_json(out/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
