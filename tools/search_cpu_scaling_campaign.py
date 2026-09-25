"""Pinned, single-worker CPU cost/scaling experiment with a complete tiny rehearsal."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
from match_evidence import atomic_json,file_hash
from analyze_cpu_gpu import analyze_match,paired_difference

BASE=ROOT/'benchmarks/search_cpu_scaling_20260912'
VALUE=ROOT/'models/candidates/search_leaf_leaf_001/epoch_003.bin'
REFERENCE=ROOT/'models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
BOOK=ROOT/'benchmarks/b2_challenger_confirmation_20260910/confirmation_book.json'

def extension_justified(delta):
    return delta['delta']>=.10 and delta['black_delta']>=0

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation',type=Path,default=BASE/'validation.json')
    ap.add_argument('--out',type=Path,default=BASE/'campaign')
    ap.add_argument('--rehearsal',action='store_true')
    ap.add_argument('--rehearsal-receipt',type=Path,default=BASE/'rehearsal/summary.json')
    args=ap.parse_args();out=args.out
    validation=json.loads(args.validation.read_text())
    if not validation['complete'] or validation['runtime']!=file_hash(ROOT/'native/monster_native.pyd'):
        raise ValueError('Missing successful current-runtime validation')
    if validation['value']!=file_hash(VALUE):raise ValueError('Model changed since validation')
    files=list((ROOT/'native/src').glob('*.rs'))+list((ROOT/'src').glob('*.py'))+[
        ROOT/'native/monster_native.pyd',Path(__file__),ROOT/'tools/search_cpu_gpu_match.py',
        ROOT/'tools/analyze_cpu_gpu.py',ROOT/'tools/analyze_search_first.py',
        ROOT/'tools/audit_search_first_games.py',ROOT/'tools/search_first_match.py',
        ROOT/'tools/profile_search_costs.py',ROOT/'tools/validate_cpu_scaling.py',
        VALUE,REFERENCE,BOOK,args.validation]
    hashes={str(p):file_hash(p) for p in files}
    if not args.rehearsal:
        rehearsal=json.loads(args.rehearsal_receipt.read_text())
        if not rehearsal['complete'] or not rehearsal['rehearsal'] or rehearsal['hashes']!=hashes:
            raise ValueError('Required full-chain rehearsal stale or failed')
    out.mkdir(parents=True,exist_ok=False)
    profiles={}
    for arm,options in validation['arms'].items():
        profile=out/(arm+'_options.json')
        atomic_json(profile,dict(complete=True,runtime=validation['runtime'],value=validation['value'],options=options))
        profiles[arm]=profile
    pinned={**hashes,**{str(p):file_hash(p) for p in profiles.values()}}
    started=time.time();reports={}
    atomic_json(out/'manifest.json',dict(hashes=pinned,started=started,rehearsal=args.rehearsal,
        development=[336,352],scaling=[352,368],extension=[368,384],promotion=False,
        nomination='highest overall development score; ties favor earlier/simpler arm',
        extension_trigger='8s minus 2s overall >= 0.10 and Black delta >= 0; exploratory, not a promotion gate',
        clock='candidate 2 or 8 seconds versus fixed GPU opponent 2 seconds; 100M CPU nodes both clocks'))
    def stage(name,command):
        if any(file_hash(p)!=h for p,h in pinned.items()):raise ValueError('Pinned inputs changed')
        stage_start=time.time();print('STAGE '+name,flush=True)
        with subprocess.Popen([sys.executable,*command],cwd=ROOT) as child:
            try:
                while True:
                    atomic_json(out/'status.json',dict(stage=name,complete=False,child_pid=child.pid,
                        heartbeat=time.time(),stage_started=stage_start,elapsed=time.time()-started))
                    try:
                        code=child.wait(timeout=60)
                        if code:raise subprocess.CalledProcessError(code,command)
                        break
                    except subprocess.TimeoutExpired:
                        if time.time()-stage_start>8*3600:raise TimeoutError('Eight-hour stage safety bound: '+name)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=30)
    def play(name,arm,offset,seconds):
        stage(name,['tools/search_cpu_gpu_match.py','--options',str(profiles[arm]),'--mode','cpu',
            '--out',str(out/name),'--pairs','1' if args.rehearsal else '16','--offset',str(offset),
            '--seconds',str(seconds if not args.rehearsal else (.015 if seconds==2 else .06)),
            '--opponent-seconds','0.015' if args.rehearsal else '2',
            '--node-limit','100000000','--opponent-node-limit','100000000'])
        report=analyze_match(out/name)
        if not report['complete'] or report['proof_contradictions']:raise ValueError('Invalid match evidence')
        expected=2 if args.rehearsal else 32
        if report['games']!=expected:raise ValueError('Incorrect completed game count')
        if any(v['node_limit_interruptions'] for v in report['players']['candidate'].values()):
            raise ValueError('CPU node ceiling confounded assigned time budget')
        atomic_json(out/name/'analysis.json',report);reports[name]=report
        atomic_json(out/'summary.json',dict(complete=False,rehearsal=args.rehearsal,stages=reports))
        return report['scores']['overall']['score']
    try:
        stage('tests',['-m','pytest','tests','-q'])
        scores={arm:play('development_'+arm,arm,336,2) for arm in profiles}
        nominee=max(scores,key=scores.get)
        comparisons={arm:paired_difference(out/'development_baseline',out/('development_'+arm))
                     for arm in profiles if arm!='baseline'}
        atomic_json(out/'decision.json',dict(nominee=nominee,scores=scores,paired=comparisons,promotion=False))
        play('scaling_2s',nominee,352,2);play('scaling_8s',nominee,352,8)
        first=[out/'scaling_2s'];second=[out/'scaling_8s']
        difference=paired_difference(first,second)
        atomic_json(out/'scaling_difference_initial.json',difference)
        extend=args.rehearsal or extension_justified(difference)
        if extend:
            play('extension_2s',nominee,368,2);play('extension_8s',nominee,368,8)
            first.append(out/'extension_2s');second.append(out/'extension_8s')
        combined=paired_difference(first,second)
        atomic_json(out/'scaling_difference.json',combined)
        stage('replay_audit',['tools/audit_search_first_games.py','--root',str(out),'--out',str(out/'replay_audit.json')])
        audit=json.loads((out/'replay_audit.json').read_text())
        if not audit['complete'] or len(audit['arms'])!=len(reports):raise ValueError('Incomplete replay audit')
        atomic_json(out/'summary.json',dict(complete=True,rehearsal=args.rehearsal,hashes=hashes,
            nominee=nominee,stages=reports,paired_development=comparisons,scaling_initial=difference,
            extended=extend,scaling_combined=combined,seconds=time.time()-started,promotion=False,
            continuation='Throughput work supported provisionally' if extension_justified(combined) else
                'Inspect decision errors and better training targets before another throughput-only campaign',
            caution='Unequal clocks; exploratory small-sample scaling evidence, not established promotion strength'))
        atomic_json(out/'status.json',dict(stage='complete',complete=True,heartbeat=time.time()))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),time=time.time()))
        atomic_json(out/'status.json',dict(stage='failed',complete=False,error=repr(exc),heartbeat=time.time()))
        raise

if __name__=='__main__':main()
