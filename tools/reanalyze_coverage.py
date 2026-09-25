"""Full-state policy reanalysis with bounded, transitive source-family exposure.

Search, disagreement ranking and policy-only output semantics are unchanged.
This opt-in adapter changes only which source records receive expensive search.
"""
import argparse
from collections import Counter, defaultdict
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import reanalyze
from reanalyze_stateful import reanalyze_one
from match_evidence import atomic_json,digest,file_hash

MAX_PER_FAMILY=16


def phase(record):
    if record['current_player']=='black':return 'black'
    if record['current_player']!='white':raise ValueError('Unknown player')
    return 'white_second' if record.get('half') else 'white_first'


def family_roots(rows):
    parents={}
    for item in rows:
        path=item['path'].replace('\\','/')
        source=item['record'].get('source_record') or {}
        parent=source.get('path')
        if parent:parent=parent.replace('\\','/')
        if path in parents and parents[path]!=parent:raise ValueError('Inconsistent source family')
        parents[path]=parent
    roots={}
    for path in parents:
        seen=set();current=path
        while parents[current]:
            if current in seen:raise ValueError('Source family cycle')
            seen.add(current);current=parents[current]
            if current not in parents:raise ValueError('Missing source family parent')
        roots[path]=current
    return roots


def covered_sample(rows,count,black_fraction,seed,max_per_family=MAX_PER_FAMILY):
    rows=list(rows)
    if count<1 or count>len(rows) or max_per_family<1 or not 0<=black_fraction<=1:
        raise ValueError('Invalid coverage sample limits')
    identities=[(r['path'],r['line']) for r in rows]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate source record')
    roots=family_roots(rows);rng=random.Random(seed)
    buckets={p:defaultdict(list) for p in ('black','white_first','white_second')}
    for row in rows:buckets[phase(row['record'])][roots[row['path'].replace('\\','/')]].append(row)
    families=sorted(set(roots.values()));used=Counter();selected=[]
    for groups in buckets.values():
        for bucket in groups.values():rng.shuffle(bucket)
    black_n=reanalyze._fraction_count(count,black_fraction);white_n=count-black_n
    requested=dict(black=black_n,white_first=white_n//2,white_second=white_n-white_n//2)
    def take(label,target):
        order=families.copy();rng.shuffle(order);taken=0
        while taken<target:
            progress=False
            for family in order:
                bucket=buckets[label].get(family)
                if bucket and used[family]<max_per_family:
                    selected.append(bucket.pop());used[family]+=1;taken+=1;progress=True
                    if taken==target:break
            if not progress:break
        return taken
    for label,target in requested.items():take(label,target)
    # Preserve exact total coverage if one phase has too few eligible records.
    # The census makes any requested-versus-actual phase shortage explicit.
    while len(selected)<count:
        before=len(selected)
        for label in buckets:
            if len(selected)==count:break
            take(label,1)
        if len(selected)==before:raise ValueError('Family cap cannot supply requested independent sample')
    stats=dict(available_rows=len(rows),available_families=len(families),sampled=len(selected),
        sampled_families=len(used),maximum_per_family=max(used.values()),cap=max_per_family,
        requested_phase=requested,actual_phase=dict(Counter(phase(r['record']) for r in selected)),
        family_counts=dict(sorted(used.items())),
        sample_sha256=digest([reanalyze.record_identity(r) for r in selected]))
    return selected,stats,roots


def main():
    ap=argparse.ArgumentParser(description=__doc__,add_help=False)
    ap.add_argument('--coverage-report',type=Path,required=True)
    own,remaining=ap.parse_known_args();capture={}
    def sample(rows,count,black_fraction,seed):
        selected,stats,roots=covered_sample(rows,count,black_fraction,seed)
        capture.update(stats=stats,roots=roots)
        print(json.dumps({k:v for k,v in stats.items() if k!='family_counts'}),flush=True)
        return selected
    reanalyze.stratified_sample=sample
    reanalyze._reanalyze_one=reanalyze_one
    reanalyze.__file__=__file__  # Journal binds the cap and sampling implementation.
    sys.argv=[__file__,*remaining]
    reanalyze.main()
    if '--dry-run' in remaining:return
    output=Path(remaining[remaining.index('--output-dir')+1])
    kept=Counter();kept_phase=Counter();count=0
    for path in sorted(output.glob('teacher_*.jsonl')):
        row=json.loads(path.read_text())
        source=row['source_record']['path'].replace('\\','/')
        kept[capture['roots'][source]]+=1;kept_phase[phase(row)]+=1;count+=1
        if row['value_weight']!=0:raise ValueError('Policy teacher unexpectedly trains value')
    if not count or max(kept.values())>MAX_PER_FAMILY:raise ValueError('Published coverage invariant failed')
    atomic_json(own.coverage_report,dict(complete=True,sample=capture['stats'],
        retained=dict(rows=count,families=len(kept),maximum_per_family=max(kept.values()),phase=dict(kept_phase)),
        implementation=file_hash(__file__),output_summary=file_hash(output/'reanalysis_summary.json'),
        note='Policy-only targets; point values remain actual completed outcomes; family cap before search'))


if __name__=='__main__':
    mp.freeze_support();main()
