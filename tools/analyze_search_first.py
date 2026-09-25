"""Summarize timed search diagnostics without rerunning engines or pooling arms."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from match_evidence import atomic_json, file_hash


def analyze(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    rows=[json.loads(x) for x in (folder/'games.jsonl').read_text().splitlines() if x.strip()]
    report=dict(games=len(rows), log_sha256=file_hash(folder/'games.jsonl'),
                complete=(folder/'complete.json').exists(),
                budget_seconds=manifest['arguments']['seconds'],
                selfplay=bool(manifest['arguments'].get('selfplay') or manifest['arguments'].get('reference_selfplay')),
                scores={}, timing={}, proof_contradictions=[])
    for label, color in [('white',True),('black',False),('overall',None)]:
        games=[r for r in rows if color is None or r['a_white']==color]
        if not games: continue
        wins=sum(r['result_a']>0 for r in games)
        draws=sum(r['result_a']==0 for r in games)
        report['scores'][label]=dict(wins=wins,draws=draws,losses=len(games)-wins-draws,
                                     score=(wins+.5*draws)/len(games))
    report['unique_start_states']=len({(r['start']['fen'],r['start']['half'],r['start']['turn_count']) for r in rows})
    report['unique_played_games']=len({(r['start']['fen'],tuple(d['action'] for d in r['decisions'])) for r in rows})
    pairs={}
    for row in rows:
        pairs.setdefault(row['pair'],[]).append((row['result_a']+1)/2)
    if not report['selfplay'] and pairs and all(len(v)==2 for v in pairs.values()):
        scores=np.array([np.mean(v) for v in pairs.values()])
        rng=np.random.default_rng(9151)
        ci=np.quantile(rng.choice(scores,(10000,len(scores)),replace=True).mean(axis=1),[.025,.975])
        report['paired_start_bootstrap_ci95']=ci.tolist()
        report['paired_start_count']=len(scores)
    for label, ab in [('alphabeta',True),('puct',False)]:
        moves=[d for r in rows for d in r['decisions'] if d['alphabeta']==ab]
        if not moves: continue
        times=np.array([d['seconds'] for d in moves])
        report['timing'][label]=dict(decisions=len(moves),seconds_total=float(times.sum()),
            median=float(np.median(times)),p95=float(np.quantile(times,.95)),maximum=float(times.max()))
        if ab:
            for color in (True,False):
                selected=[d for d in moves if d['white']==color]
                if selected:
                    report['timing'][label]['white' if color else 'black']=dict(
                        depth_mean=float(np.mean([d['depth'] for d in selected])),
                        nodes_per_second=sum(d['nodes'] for d in selected)/sum(d['seconds'] for d in selected),
                        extension_nodes=sum(d.get('extension_nodes',0) for d in selected),
                        max_ply_mean=float(np.mean([d.get('max_ply_reached',0) for d in selected])),
                        incomplete_first_iteration=sum(d['depth']==0 for d in selected))
    for row in rows:
        for ply, d in enumerate(row['decisions']):
            if not d['alphabeta'] or d.get('value') is None: continue
            mover_sign=1 if d['white'] else -1
            if d['value']==mover_sign and row['result_white']!=mover_sign:
                report['proof_contradictions'].append(dict(pair=row['pair'],a_white=row['a_white'],
                    ply=ply,claimed_white_value=d['value'],actual_white_result=row['result_white']))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path)
    args=ap.parse_args()
    report=analyze(args.folder)
    atomic_json(args.folder/'analysis.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
