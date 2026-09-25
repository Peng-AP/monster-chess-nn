"""Paired-opening uncertainty and Black regression examples for B2 validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from match_evidence import atomic_json,file_hash


def intervals(candidate,reference,seed=9053):
    import numpy as np
    delta=np.asarray(candidate,dtype=float)-np.asarray(reference,dtype=float)
    if delta.ndim!=2 or delta.shape[1]!=2:
        raise ValueError('Expected opening x [White,Black] scores')
    rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return {label:dict(difference=float(values.mean()),
        lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)))
        for label,values,samples in [('white',delta[:,0],draws[:,0]),('black',delta[:,1],draws[:,1]),
                                    ('overall',delta.mean(axis=1),draws.mean(axis=1))]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='benchmarks/b2_validation_20260910')
    args=parser.parse_args();root=Path(args.root)
    summary=json.loads((root/'summary.json').read_text())
    if summary.get('complete') is not True:raise ValueError('Validation incomplete')
    def records(name):
        result={}
        for j in range(4):
            for line in (root/f'{name}_o{j}.jsonl').read_text().splitlines():
                row=json.loads(line)
                key=(row['entry'],row['a_is_white'])
                if key in result:raise ValueError('Duplicate paired record')
                result[key]=row
        return result
    comparisons={};regressions=[]
    names=list(summary['panel'])
    data={name:records(name) for name in names}
    pairs=[(name,'gen47') for name in names if name!='gen47']
    for seed in ('original','replica'):
        pairs += [(f'{seed}_state_cnn_e{e}',f'{seed}_control_e{e}') for e in (8,15)]
    for name,reference in pairs:
        if name not in data or reference not in data:
            comparisons[name+'_minus_'+reference]=dict(unavailable=True,reason='Preregistered epoch unavailable')
            continue
        a,b=data[name],data[reference]
        if set(a)!=set(b):raise ValueError('Unmatched openings')
        entries=sorted({key[0] for key in a})
        values=lambda rows:[[(rows[(entry,side)]['result_for_a']+1)/2 for side in (True,False)] for entry in entries]
        comparisons[name+'_minus_'+reference]=intervals(values(a),values(b))
        for entry in entries:
            row=a[(entry,False)];baseline=b[(entry,False)]
            if row['result_for_a']<baseline['result_for_a']:
                regressions.append(dict(candidate=name,reference=reference,entry=entry,
                    candidate_result=row['result_for_a'],reference_result=baseline['result_for_a'],
                    candidate_game=row.get('game'),reference_game=baseline.get('game'),opening=row.get('opening')))
    atomic_json(root/'paired_analysis.json',dict(comparisons=comparisons,
        note='10000 paired-opening bootstrap samples; exploratory intervals, not multiplicity-adjusted promotion tests.',
        summary_sha256=file_hash(root/'summary.json')))
    atomic_json(root/'black_regression_examples.json',dict(examples=regressions))


if __name__=='__main__':main()
