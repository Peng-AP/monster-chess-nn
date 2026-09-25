"""Replay completed search-first match logs and classify actual game endings.

Run after timed campaigns are idle. No engine evaluation, no training, no pooling
of arms or reclassification of cap draws as solved/fortress positions.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import chess
from monster_chess import MonsterChessGame
from repetition import RepetitionTracker
from match_evidence import atomic_json,file_hash
from worker_lease import worker_lease


def audit(folder):
    log=folder/'games.jsonl'
    rows=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    manifest=json.loads((folder/'manifest.json').read_text())
    endings={'white_capture':0,'black_capture':0,'repetition':0,'turn_cap':0}
    for row in rows:
        g=MonsterChessGame(row['start']['fen'])
        g.white_half_pending=row['start']['half']
        g.turn_count=row['start']['turn_count']
        tracker=RepetitionTracker()
        tracker.record(g)
        for ply,d in enumerate(row['decisions']):
            if g.is_terminal() or tracker.fired_at is not None:
                raise ValueError(f'Continuation after termination: {folder}:{row["pair"]}:{ply}')
            if d['white']!=g.is_white_turn or d['half']!=g.white_half_pending:
                raise ValueError('Logged actor/phase mismatch')
            action=chess.Move.from_uci(d['action'])
            if action not in g.get_search_actions():raise ValueError('Illegal logged action')
            g.apply_search_action(action)
            tracker.record(g,ply+1)
        if g.fen()!=row['final_fen']:raise ValueError('Final FEN mismatch')
        repeated=tracker.fired_at is not None
        if repeated!=row['repetition']:raise ValueError('Repetition mismatch')
        if not (g.is_terminal() or repeated):raise ValueError('Unfinished game in completed log')
        value=g.get_result() if g.is_terminal() else 0
        value=int(value) if abs(value)>=1 else 0
        if row['result_white']!=value or row['result_a']!=(value if row['a_white'] else -value):
            raise ValueError('Outcome/perspective mismatch')
        ending=('white_capture' if value>0 else 'black_capture' if value<0 else
                'repetition' if repeated else 'turn_cap')
        endings[ending]+=1
    receipt=json.loads((folder/'complete.json').read_text())
    if len(rows)!=receipt['games']:raise ValueError('Receipt game count differs')
    score=sum((r['result_a']+1)/2 for r in rows)/len(rows)
    if abs(score-receipt['score'])>1e-12:raise ValueError('Receipt score differs')
    return dict(games=len(rows),score=score,endings=endings,
        games_hash=file_hash(log),manifest_hash=file_hash(folder/'manifest.json'),
        arguments=manifest['arguments'],runtime=manifest['hashes'].get('native/monster_native.pyd'),
        note='cap draw is an operational result, not a proof of a game-theoretic draw')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT/'benchmarks/search_first_20260911')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    report=dict(complete=False,arms={})
    with worker_lease():
        for log in sorted(args.root.rglob('games.jsonl')):
            if not (log.parent/'complete.json').exists():
                raise ValueError(f'Incomplete match: {log.parent}')
            row=audit(log.parent)
            report['arms'][str(log.parent.relative_to(args.root))]=row
            print(log.parent.name,row['games'],row['endings'],flush=True)
    report['complete']=True
    atomic_json(args.out,report)


if __name__=='__main__':main()
