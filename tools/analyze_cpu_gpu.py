"""Preserve per-player and per-backend timings when both players use CPU search."""
import json
from pathlib import Path
import numpy as np
from analyze_search_first import analyze

def analyze_match(folder):
    folder=Path(folder);report=analyze(folder)
    rows=[json.loads(x) for x in (folder/'games.jsonl').read_text().splitlines()]
    manifest=json.loads((folder/'manifest.json').read_text())
    report['opponent_budget_seconds']=manifest['arguments'].get('opponent_seconds',report['budget_seconds'])
    report['players']={}
    for candidate,name in [(True,'candidate'),(False,'reference')]:
        moves=[d for row in rows for d in row['decisions'] if d.get('candidate')==candidate]
        if not moves:continue
        groups={}
        for side,label in [(True,'white'),(False,'black')]:
            subset=[d for d in moves if d['white']==side]
            if not subset:continue
            cpu=[d for d in subset if d['alphabeta']]
            groups[label]=dict(decisions=len(subset),seconds=sum(d['seconds'] for d in subset),
                median_seconds=float(np.median([d['seconds'] for d in subset])),
                p95_seconds=float(np.quantile([d['seconds'] for d in subset],.95)),
                cpu_decisions=len(cpu),depth_mean=float(np.mean([d['depth'] for d in cpu])) if cpu else None,
                node_limit_interruptions=sum(d.get('node_limit_reached',False) for d in cpu),
                depth_limit_completions=sum(d.get('depth_limit_reached',False) for d in cpu),
                maximum_nodes=max((d['nodes'] for d in cpu),default=0),
                nodes_per_second=sum(d['nodes'] for d in cpu)/sum(d['seconds'] for d in cpu) if cpu else None,
                policy_seconds=sum(d.get('policy_seconds',0) for d in subset),
                backends={b:sum(d['backend']==b for d in subset) for b in {d['backend'] for d in subset}})
        report['players'][name]=groups
    return report

def paired_difference(first,second):
    """Candidate score in second minus first, resampling whole common starts."""
    def scores(folder):
        folders=folder if isinstance(folder,(list,tuple)) else [folder]
        rows=[json.loads(x) for item in folders for x in (Path(item)/'games.jsonl').read_text().splitlines()]
        grouped={}
        for r in rows:
            key=(r['start']['fen'],r['start']['half'],r['start']['turn_count'])
            if r['a_white'] in grouped.setdefault(key,{}):raise ValueError('Duplicate color in paired start')
            grouped[key][r['a_white']]=(r['result_a']+1)/2
        if any(set(v)!={False,True} for v in grouped.values()):raise ValueError('Incomplete color pair')
        return grouped
    a,b=scores(first),scores(second)
    if a.keys()!=b.keys():raise ValueError('Mismatched start sets')
    differences=np.array([np.mean([b[k][c]-a[k][c] for c in (False,True)]) for k in a])
    rng=np.random.default_rng(9274)
    ci=np.quantile(rng.choice(differences,(10000,len(differences)),replace=True).mean(axis=1),[.025,.975])
    return dict(starts=len(differences),delta=float(differences.mean()),ci95=ci.tolist(),
        white_delta=float(np.mean([b[k][True]-a[k][True] for k in a])),
        black_delta=float(np.mean([b[k][False]-a[k][False] for k in a])))
