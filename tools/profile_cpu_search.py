"""Fixed-input search parity/performance; use snapshot before modifying runtime."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'native'),str(ROOT/'src'),str(ROOT/'tools')]
import monster_native as native
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease

VALUE=ROOT/'models/candidates/search_leaf_leaf_001/epoch_003.bin'
ARMS={'baseline':{},'pvs':{'pvs':True},'tt':{'fresh_tt':True},
      'efficient':{'pvs':True,'fresh_tt':True},
      'large_tt':{'pvs':True,'fresh_tt':True,'tt_capacity':131072}}

def states():
    source=ROOT/'benchmarks/search_first_20260911/cpu_eval_after.json'
    rows=json.loads(source.read_text())['states']
    groups={0:[],1:[],2:[]}
    for row in rows:
        phase=2 if row[0].split()[1]=='b' else int(row[1])
        if len(groups[phase])<8:groups[phase].append(row)
    return [r for group in groups.values() for r in group]

def record(result,elapsed):
    fields=['action','value','completed_depth','nodes','cutoffs','interrupted',
            'eval_cache_hits','tt_hits','tt_cutoffs','max_ply_reached']
    extra=['pvs_scouts','pvs_researches','tt_rejected','phase_nodes','phase_generated']
    return {**{k:getattr(result,k) for k in fields},
            **{k:getattr(result,k) for k in extra if hasattr(result,k)},'seconds':elapsed}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--snapshot',action='store_true')
    ap.add_argument('--compare',type=Path)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    inputs=states();report={'states':inputs,'runtime':file_hash(ROOT/'native/monster_native.pyd'),
        'value':file_hash(VALUE),'fixed_node':[],'fixed_depth':[],'complete':False}
    with worker_lease():
        net=native.CheapValue(str(VALUE))
        for f,h,c in inputs:
            t=time.perf_counter()
            r=native.alphabeta_search(f,pending=h,turn_count=c,evaluator=net,
                seconds=30,node_limit=30000,max_depth=6)
            report['fixed_node'].append(record(r,time.perf_counter()-t))
        if args.compare:
            old=json.loads(args.compare.read_text())
            if old['states']!=inputs or old['value']!=report['value']:raise ValueError('Changed baseline inputs')
            fields=['action','value','completed_depth','nodes','cutoffs','interrupted','eval_cache_hits','tt_hits','tt_cutoffs']
            for a,b in zip(old['fixed_node'],report['fixed_node']):
                if any(a[k]!=b[k] for k in fields):raise ValueError('Default baseline changed')
        if not args.snapshot:
            for i,(f,h,c) in enumerate(inputs):
                results={}
                arms=list(ARMS);arms=arms[i%len(arms):]+arms[:i%len(arms)]
                for arm in arms:
                    t=time.perf_counter()
                    r=native.alphabeta_search(f,pending=h,turn_count=c,evaluator=net,
                        seconds=30,node_limit=10000000,max_depth=4,**ARMS[arm])
                    results[arm]=record(r,time.perf_counter()-t)
                if any(r['interrupted'] for r in results.values()):raise ValueError('Fixed-depth profile interrupted')
                if len({(r['value'],r['completed_depth']) for r in results.values()})!=1:
                    raise ValueError(f'Exact-search parity failed at {i}: {results}')
                report['fixed_depth'].append({'state':i,'arms':results})
                print('fixed depth state',i,flush=True)
            totals={a:sum(r['arms'][a]['seconds'] for r in report['fixed_depth']) for a in ARMS}
            report['seconds_by_arm']=totals
            report['nominee']=min(totals,key=totals.get)
            report['options']=ARMS[report['nominee']]
            report['speedup']=totals['baseline']/totals[report['nominee']]
        report['complete']=True;atomic_json(args.out,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('states','fixed_node','fixed_depth')},indent=2))

if __name__=='__main__':main()
