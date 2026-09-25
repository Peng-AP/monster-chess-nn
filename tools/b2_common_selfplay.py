"""One self-game per common opening, not two identical color-swapped copies."""
import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from match_evidence import atomic_json, file_hash, runtime_identity
from worker_lease import worker_lease
import match


def initialize(model,sims):
    random.seed(9053)
    match._init_worker(model,model,sims,engine='native')


def play(task):
    return match._result_row(match._play(task),has_book=True)


def outcome_counts(rows):
    """Capture-only W/D/L: the training cap's +/-0.5 lean is a draw."""
    white=sum(match.game_score(r['result_for_a'])==1 for r in rows)
    black=sum(match.game_score(r['result_for_a'])==0 for r in rows)
    return white,black,len(rows)-white-black


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--book',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--games',type=int,default=200)
    parser.add_argument('--sims',type=int,default=3200)
    args=parser.parse_args()
    output=Path(args.output)
    entries,_=match.load_book(args.book)
    if args.games<=0 or args.sims<=0 or len(entries)<args.games:
        raise ValueError('Invalid count/sims or insufficient common starts')
    entries=entries[:args.games]
    if len({(e['fen'],e['half'],e['turn_count']) for e in entries})!=args.games:
        raise ValueError('Duplicate common starts')
    manifest=dict(model=file_hash(args.model),book=file_hash(args.book),runtime=runtime_identity(),
                  implementation=file_hash(__file__),sims=args.sims,games=args.games,seed=2050000000)
    record_path=output.with_suffix('.manifest.json')
    if record_path.exists() and json.loads(record_path.read_text())!=manifest:
        raise ValueError('Self-play provenance changed')
    atomic_json(record_path,manifest)
    log=output.with_suffix('.jsonl')
    rows=[json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    done={r['pair'] for r in rows}
    if len(done)!=len(rows) or not done.issubset(set(range(args.games))):
        raise ValueError('Invalid/duplicate saved self-games')
    tasks=[(True,2050000000+i,0,entry,i) for i,entry in enumerate(entries) if i not in done]
    with worker_lease():
        if tasks:
            pool=futures.ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),
                initializer=initialize,initargs=(args.model,args.sims))
            pending={pool.submit(play,t) for t in tasks}
            try:
                with log.open('a',encoding='utf-8') as stream:
                    while pending:
                        completed,pending=futures.wait(pending,timeout=600,return_when=futures.FIRST_COMPLETED)
                        if not completed:
                            raise TimeoutError('Self-play stalled')
                        for future in completed:
                            row=future.result()
                            rows.append(row)
                            stream.write(json.dumps(row)+'\n')
                            stream.flush()
                        print(f'common self-play {len(rows)}/{args.games}',flush=True)
            except BaseException:
                from data_generation import terminate_pool
                terminate_pool(pool)
                raise
            else:
                pool.shutdown(wait=True)
    white,black,draws=outcome_counts(rows)
    atomic_json(output,dict(complete=True,games=len(rows),white_wins=white,black_wins=black,
        draws=draws,white_score=(white+.5*draws)/len(rows),
        log_sha256=file_hash(log),manifest_sha256=file_hash(record_path)))


if __name__=='__main__':
    mp.freeze_support()
    main()
