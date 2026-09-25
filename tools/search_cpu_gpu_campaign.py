"""Sequential overnight CPU-search and GPU-cooperation experiment.

Every stage checks receipts and pinned sources. Child failure/timeout stops the
chain, heartbeat receipts distinguish a live stage from a dead parent process.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
from match_evidence import atomic_json,file_hash
from analyze_cpu_gpu import analyze_match,paired_difference

def main():
    out=ROOT/'benchmarks/search_cpu_gpu_20260912/campaign'
    profile=ROOT/'benchmarks/search_cpu_gpu_20260912/workload_final.json'
    receipt=json.loads(profile.read_text())
    if not receipt['complete'] or receipt['runtime']!=file_hash(ROOT/'native/monster_native.pyd'):
        raise ValueError('Required workload profile failed or runtime changed')
    out.mkdir(parents=True,exist_ok=False)
    files=list((ROOT/'native/src').glob('*.rs'))+[
        ROOT/'native/monster_native.pyd',ROOT/'src/cpu_search_engine.py',ROOT/'src/native_mcts.py',
        ROOT/'src/evaluation.py',ROOT/'tools/search_cpu_gpu_match.py',ROOT/'tools/analyze_cpu_gpu.py',
        ROOT/'tools/analyze_search_first.py',ROOT/'tools/audit_search_first_games.py',
        ROOT/'models/candidates/search_leaf_leaf_001/epoch_003.bin',
        ROOT/'models/candidates/bootstrap_main_gen_0047/arena_selected.pt',
        ROOT/'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt',
        ROOT/'benchmarks/b2_challenger_confirmation_20260910/confirmation_book.json',profile,Path(__file__)]
    hashes={str(p):file_hash(p) for p in files}
    started=time.time();reports={}
    atomic_json(out/'manifest.json',dict(hashes=hashes,started=started,profile=receipt['options'],
        development=[288,304],confirmation=[304,336],promotion=False,
        nomination='highest overall development score; stable ties favor baseline',
        clock='soft end-to-end per-halfmove budget; all recurring GPU guidance overhead charged'))
    def stage(name,command):
        if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError('Pinned input or implementation changed')
        stage_start=time.time()
        print('STAGE '+name,flush=True)
        with subprocess.Popen([sys.executable,*command],cwd=ROOT) as child:
            while True:
                atomic_json(out/'status.json',dict(stage=name,complete=False,child_pid=child.pid,
                    stage_started=stage_start,heartbeat=time.time(),elapsed=time.time()-started))
                try:
                    code=child.wait(timeout=60)
                    if code:raise subprocess.CalledProcessError(code,command)
                    break
                except subprocess.TimeoutExpired:
                    if time.time()-stage_start>4*3600:
                        child.terminate();child.wait(timeout=30)
                        raise TimeoutError('Stage exceeded four-hour safety limit: '+name)
    def play(name,mode,pairs,offset,seconds=.3,extra=()):
        stage(name,['tools/search_cpu_gpu_match.py','--options',str(profile),'--mode',mode,
            '--out',str(out/name),'--pairs',str(pairs),'--offset',str(offset),'--seconds',str(seconds),*extra])
        report=analyze_match(out/name)
        if not report['complete'] or report['proof_contradictions']:raise ValueError('Invalid match evidence')
        atomic_json(out/name/'analysis.json',report);reports[name]=report
        atomic_json(out/'summary.json',dict(complete=False,stages=reports))
        return report['scores']['overall']['score']
    try:
        stage('tests',['-m','pytest','tests','-q'])
        for mode in ('cpu','guided','split'):
            play('rehearsal_'+mode,mode,1,288,.015)
        play('rehearsal_cpu_duel','cpu',1,288,.015,['--opponent','baseline'])
        stage('rehearsal_audit',['tools/audit_search_first_games.py','--root',str(out),
                                '--out',str(out/'rehearsal_audit.json')])
        if receipt['options']:
            play('cpu_vs_cpu','cpu',16,288,extra=['--opponent','baseline'])
        modes=['baseline']+(['cpu'] if receipt['options'] else [])+['guided','split']
        scores={mode:play('development_'+mode,mode,16,288) for mode in modes}
        nominee=max(scores,key=scores.get)
        atomic_json(out/'decision.json',dict(nominee=nominee,scores=scores,promotion=False))
        # A fresh common-start control prevents a development-set result from
        # masquerading as confirmation of improved CPU search.
        play('confirmation_baseline','baseline',32,304,2.)
        if nominee!='baseline':
            play('confirmation_nominee',nominee,32,304,2.)
            atomic_json(out/'paired_improvement.json',paired_difference(
                out/'confirmation_baseline',out/'confirmation_nominee'))
        play('confirmation_b2',nominee,16,304,2.,['--reference',
             'models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'])
        play('self_nominee',nominee,16,304,2.,['--selfplay'])
        play('self_gen47','puct',16,304,2.,['--selfplay'])
        stage('replay_audit',['tools/audit_search_first_games.py','--root',str(out),
                             '--out',str(out/'replay_audit.json')])
        atomic_json(out/'summary.json',dict(complete=True,nominee=nominee,stages=reports,
            seconds=time.time()-started,promotion=False))
        atomic_json(out/'status.json',dict(stage='complete',complete=True,heartbeat=time.time()))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),time=time.time()))
        atomic_json(out/'status.json',dict(stage='failed',complete=False,error=repr(exc),heartbeat=time.time()))
        raise

if __name__=='__main__':main()
