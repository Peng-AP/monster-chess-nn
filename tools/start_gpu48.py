"""One bounded GPU data-recipe campaign: rehearse, train, nominate, test, stop.

No architecture search, promotion, optional stopping, or silent train restart.
Receipts bind implementation and inputs; interrupted game stages resume through
their existing journals. Run under tools/runs.py for a durable managed process.
"""
import argparse
from collections import Counter
from contextlib import contextmanager,nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from match_evidence import atomic_json,digest,file_hash,runtime_identity
from worker_lease import worker_lease
from b2_validation_analysis import intervals
from match import game_score
import iterate

BASELINE='models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
HOLDOUT='models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'
TEACHERS=[BASELINE,'models/bootstrap_v27/best_value_net.pt',
          'models/candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt']
PINNED_MODELS={BASELINE:'810297fe98807eb56cfe4184c3210048fed7dc93110c47dfb18c0564fb3865fb',
               HOLDOUT:'fc23076a9f5c7f237785f27cb1a665c10588ea8e8916cd743016d19a96999d15'}
RUN=ROOT/'benchmarks/gpu48_20260913'
SMOKE=ROOT/'iterations/rehearsal_gpu48_20260913'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin_json(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError(f'Changed pinned evidence: {path}')
    else:atomic_json(path,value)


def pilot_sizing():
    summary=ROOT/'iterations/gpu48_pilot_20260913/summary.json'
    record=read(summary);started=read(ROOT/'logs/gpu48_cost_pilot.json')['started']
    if record['saved_games']!=64 or record['failed_games'] or record['num_games_requested']!=64:
        raise ValueError('Full-budget cost pilot did not finish')
    elapsed=summary.stat().st_mtime-datetime.fromisoformat(started).timestamp()
    if elapsed<=0:raise ValueError('Invalid pilot timestamps')
    projected=elapsed*4800/64
    return dict(games=3200 if projected>4*3600 else 4800,elapsed_sec=elapsed,
        nominal_projection_sec=projected,summary_sha256=file_hash(summary),
        pilot_recipe_sha256=file_hash(ROOT/'tools/recipes/gpu48_pilot.json'),
        basis='Managed start to completed summary mtime; startup makes extrapolation conservative')


def identity():
    paths=sorted((ROOT/'tools').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    paths+=sorted((ROOT/'configs').glob('*.json'))
    paths += [ROOT/'tools/recipes'/n for n in ('gen48.json','gen48_rehearsal.json','gpu48_pilot.json')]
    recipe=read(ROOT/'tools/recipes/gen48.json')
    from iterate_stateful import validate_recipe
    validate_recipe(recipe)
    models=sorted({recipe['model'],HOLDOUT,*recipe['prefix_models'],*recipe['opponents']})
    if HOLDOUT in [recipe['model'],*recipe['prefix_models'],*recipe['opponents']]:
        raise ValueError('Holdout opponent leaked into generation recipe')
    model_hashes={p:file_hash(ROOT/p) for p in models}
    for p,h in PINNED_MODELS.items():
        if model_hashes[p]!=h:raise ValueError(f'Frozen model changed: {p}')
    size=pilot_sizing()
    if sum(recipe[k] for k in ('free_games','fresh_games','league_games','fork_games'))!=size['games']:
        raise ValueError('Production recipe does not obey declared pilot sizing rule')
    # Pin the existing seven replay acceptances. Canonical iteration validates
    # their full artifact digests before composition; gen48's later acceptance
    # must not invalidate the launch manifest merely by extending the registry.
    prior=sorted((e for e in read(ROOT/'iterations/accepted_data.json')['entries']
                  if int(e['generation'])<48),key=lambda e:int(e['generation']))[-7:]
    if len(prior)!=7:raise ValueError('Expected seven accepted historical replay sources')
    return dict(runtime=runtime_identity(),implementations={str(p.relative_to(ROOT)):file_hash(p) for p in paths},
        models=model_hashes,replay=prior,pilot=size,
        protocol=dict(games=size['games'],reanalysis_sample=24000,reanalysis_keep=12000,
                      sims=1600,deep_sims=6400,match_sims=3200,post_selection_games=896,
                      book_seed=2130000000,match_seed=2135000000,workers=8,
                      vram_target_mib=12288,training_seed=3173),
        scope='Research only; canonical stop after checkpoint screen; no promotion')


class Campaign:
    def __init__(self,root,provenance,identity_fn=None,label='GPU48'):
        self.root=Path(root);self.provenance=provenance
        self.identity_fn=identity_fn;self.label=label
        pin_json(self.root/'manifest.json',provenance)

    def stage(self,name,command,outputs=(),exclusive=False,allow_existing=True):
        outputs=[str(Path(p).resolve()) for p in outputs]
        receipt=self.root/'receipts'/f'{name}.json'
        if receipt.exists():
            saved=read(receipt)
            if (saved['command']!=command or saved['manifest_sha256']!=file_hash(self.root/'manifest.json')
                or set(saved['outputs'])!=set(outputs)
                or any(not Path(p).exists() or file_hash(p)!=h for p,h in saved['outputs'].items())):
                raise ValueError(f'Changed completed stage: {name}')
            return
        if not allow_existing and any(Path(p).exists() for p in outputs):
            raise FileExistsError(f'Unreceipted non-resumable output for {name}; retained, not overwritten')
        if (self.identity_fn or identity)()!=self.provenance:raise ValueError('Inputs changed during campaign')
        atomic_json(self.root/'status.json',dict(status='running',stage=name,started=time.time()))
        print(f'{self.label} STAGE {name}: {command}',flush=True);started=time.monotonic()
        try:
            with worker_lease() if exclusive else nullcontext():
                subprocess.run([sys.executable,'-u',*command],cwd=ROOT,check=True)
            if (self.identity_fn or identity)()!=self.provenance:raise ValueError('Inputs changed during stage')
            atomic_json(receipt,dict(command=command,elapsed_sec=time.monotonic()-started,
                outputs={p:file_hash(p) for p in outputs},manifest_sha256=file_hash(self.root/'manifest.json')))
        except BaseException as exc:
            atomic_json(self.root/'status.json',dict(status='failed',stage=name,error=str(exc)))
            raise


@contextmanager
def campaign_lock(path):
    """Separate OS lock: do not re-enter the process-global GPU worker lease."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if not path.stat().st_size:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield  # Closing releases the lease, including on exceptions/crashes.


def iteration_command(smoke):
    recipe='tools/recipes/gen48_rehearsal.json' if smoke else 'tools/recipes/gen48.json'
    n=1 if smoke else 48
    command=['tools/iterate_stateful.py','--recipe',recipe,'--expected-generation',str(n),'--',
        '--incumbent',BASELINE,'--games',str(28 if smoke else 3200),'--book-seed-games','0',
        '--anchor-data','none','--workers','8','--engine','native','--seed',str(6173 if smoke else 3173),
        '--sims',str(8 if smoke else 1600),'--reanalysis-sample',str(20 if smoke else 24000),
        '--reanalysis-keep',str(10 if smoke else 12000),'--reanalysis-sims',str(8 if smoke else 6400),
        '--reanalysis-black-fraction','.6','--replay-generations',str(1 if smoke else 8),
        '--epochs',str(1 if smoke else 30),'--patience',str(1 if smoke else 10),
        '--batch-size',str(64 if smoke else 256),'--warmup-epochs',str(1 if smoke else 3),
        '--lr','.002','--ema-decay','.999','--value-floor','.5','--value-horizon','60',
        '--teacher-policy-multiplier','4','--checkpoint-probe-games',str(4 if smoke else 40),
        '--checkpoint-screen-games',str(4 if smoke else 200),
        '--checkpoint-probe-sims',str(8 if smoke else 3200),
        '--checkpoint-screen-sims',str(8 if smoke else 3200),
        '--checkpoint-screen-finalists',str(1 if smoke else 2),'--through-phase','checkpoint_screen']
    if smoke:command+=['--run-root',str(SMOKE),'--offline-positions','128']
    else:command+=['--data-seed-base','1000000000']
    return command


def selected_state(path):
    state=read(path)
    if state['status']!='partial':raise ValueError('Research iteration must stop after selection')
    for name in iterate.PHASES[:iterate.PHASES.index('checkpoint_screen')+1]:
        if state['phases'].get(name,{}).get('status')!='completed':
            raise ValueError(f'Incomplete rehearsal/iteration phase: {name}')
    candidate=iterate._absolute(state['paths']['candidate'])
    if not candidate.is_file():raise FileNotFoundError(candidate)
    return state,candidate


def train(campaign,smoke):
    root=SMOKE if smoke else ROOT/'iterations';n=1 if smoke else 48
    paths=iterate._paths_for_generation(root,n);command=iteration_command(smoke)
    if paths['state'].exists():
        old=read(paths['state'])
        if old['phases'].get('train',{}).get('status') in ('running','failed'):
            raise ValueError('Interrupted training retained; no automatic overwrite or scratch restart')
        command+=['--resume']
    # Resume is an execution flag, not part of a completed stage's identity.
    receipt=campaign.root/'receipts/iteration.json'
    if receipt.exists():command=read(receipt)['command']
    campaign.stage('iteration',command,[paths['state'],paths['candidate'],paths['reports']/'reanalysis_coverage.json'])
    state,candidate=selected_state(paths['state'])
    coverage=read(paths['reports']/'reanalysis_coverage.json')
    expected=20 if smoke else 24000
    if not coverage['complete'] or coverage['sample']['sampled']!=expected:
        raise ValueError('Missing/full-size coverage sample not verified')
    return candidate


def layout(smoke):
    # Six always-run match branches; confirmation uses disjoint start indices.
    count=2 if smoke else 128;transfer=2 if smoke else 64
    return dict(entries=count*2+transfer,h2h_games=count*2,transfer_games=transfer*2,
                self_games=transfer,confirmation_offset=count,holdout_offset=count*2,
                sims=8 if smoke else 3200,book_sims=8 if smoke else 700,
                book_seed=2120000000 if smoke else 2130000000,
                match_seed=2125000000 if smoke else 2135000000)


def audit_rows(path,book_entries,paired=True,offset=0):
    """Replay regular match trajectories; reject illegal/unfinished evidence.

    Book games deliberately start with the harness's fresh repetition history.
    This is not a claim to restore pre-book history (training DOES restore it).
    """
    import chess
    from monster_chess import MonsterChessGame
    from repetition import RepetitionTracker
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    seen=set();endings=Counter();by_side={True:[],False:[]};table={}
    for row in rows:
        key=(row['entry'],row['a_is_white'])
        if key in seen:raise ValueError('Duplicate match entry/color')
        seen.add(key)
        if not offset<=row['entry']<offset+len(book_entries):raise ValueError('Out-of-block entry')
        entry=book_entries[row['entry']-offset]
        trace=row['game']['trajectory']
        if len(trace)!=row['plies']+1 or row['game']['trajectory_sha256']!=digest(trace):
            raise ValueError('Trajectory digest/count mismatch')
        g=MonsterChessGame(entry['fen']);g.white_half_pending=entry['half'];g.turn_count=entry['turn_count']
        tracker=RepetitionTracker();tracker.record(g,0);repeated=False
        for i,snapshot in enumerate(trace):
            if i:
                if g.is_terminal() or repeated:raise ValueError('Continuation after termination')
                moves=[chess.Move.from_uci(m) for m in snapshot['action']]
                action=moves[0] if len(moves)==1 else tuple(moves)
                if action not in g.get_search_actions():raise ValueError('Illegal logged action')
                g.apply_search_action(action);repeated=tracker.record(g,i)
            if (g.fen(),g.white_half_pending,g.turn_count)!=(snapshot['fen'],snapshot['half'],snapshot['turn_count']):
                raise ValueError('Replayed state mismatch')
        raw=0 if repeated else g.get_result()
        reason=('repetition' if repeated else 'king_capture' if abs(raw)==1
                else 'ply_cap' if row['plies']>=600 else 'turn_cap' if g.is_terminal() else 'unfinished')
        if reason=='unfinished' or reason!=row['game']['termination']:raise ValueError('Termination mismatch')
        if row['result_for_a']!=(raw if row['a_is_white'] else -raw):raise ValueError('Outcome mismatch')
        if row['white_score']!=game_score(raw):raise ValueError('Score mismatch')
        ending=('white_capture' if raw==1 else 'black_capture' if raw==-1 else reason)
        endings[ending]+=1
        score=game_score(row['result_for_a']);by_side[row['a_is_white']].append(score);table[key]=score
    expected={(offset+i,side) for i in range(len(book_entries)) for side in ((True,False) if paired else (True,))}
    if seen!=expected:raise ValueError('Missing match entry/color')
    def stats(values):
        return dict(games=len(values),wins=values.count(1),draws=values.count(.5),losses=values.count(0),
                    score=sum(values)/len(values)) if values else None
    summary=dict(overall=stats([game_score(r['result_for_a']) for r in rows]),
        white=stats(by_side[True]),black=stats(by_side[False]),endings=dict(endings),
        games_sha256=file_hash(path),replay_audited=True)
    scores=[[table[(offset+i,s)] for s in (True,False)] for i in range(len(book_entries))] if paired else None
    return summary,scores


def benchmarks(campaign,candidate,smoke):
    spec=layout(smoke);root=campaign.root/'play';book=root/'book.json'
    command=['tools/make_book.py']
    for model in TEACHERS:command+=['--model',model]
    command+=['--entries',str(spec['entries']),'--plies','16','--sims',str(spec['book_sims']),
              '--temperature','.8','--seed',str(spec['book_seed']),'--oversample','8',
              '--workers','8','--out',str(book)]
    campaign.stage('book',command,[book],exclusive=True,allow_existing=False)
    document=read(book);entries=document['entries']
    if len(entries)!=spec['entries'] or len({(e['fen'],e['half'],e['turn_count']) for e in entries})!=len(entries):
        raise ValueError('Book does not contain all unique requested starts')
    hold=spec['holdout_offset'];selfbook=root/'self_book.json'
    pin_json(selfbook,dict(document,entries=entries[hold:],parent_book_sha256=file_hash(book),parent_offset=hold))
    comparisons={};scores={}
    legs=[('h2h',str(candidate),BASELINE,spec['h2h_games'],0),
          ('confirmation',str(candidate),BASELINE,spec['h2h_games'],spec['confirmation_offset']),
          ('holdout_candidate',str(candidate),HOLDOUT,spec['transfer_games'],hold),
          ('holdout_baseline',BASELINE,HOLDOUT,spec['transfer_games'],hold)]
    for name,a,b,games,offset in legs:
        report=root/f'{name}.json';log=report.with_suffix('.jsonl')
        command=['tools/match.py','--model-a',a,'--model-b',b,'--games',str(games),
            '--sims',str(spec['sims']),'--engine','native','--workers','8',
            '--seed',str(spec['match_seed']),'--book',str(book),'--book-offset',str(offset),
            '--game-log',str(log),'--report-path',str(report),'--resume']
        campaign.stage(name,command,[report,log])
        result=read(report)
        if result.get('partial') or result['games']!=games:raise ValueError('Incomplete match')
        comparisons[name],scores[name]=audit_rows(log,entries[offset:offset+games//2],offset=offset)
    selfplay={}
    for name,model in [('candidate',str(candidate)),('baseline',BASELINE)]:
        report=root/f'self_{name}.json';log=report.with_suffix('.jsonl')
        campaign.stage('self_'+name,['tools/b2_common_selfplay.py','--model',model,'--book',str(selfbook),
            '--output',str(report),'--games',str(spec['self_games']),'--sims',str(spec['sims'])],[report,log])
        if read(report).get('complete') is not True:raise ValueError('Incomplete selfplay')
        selfplay[name],_=audit_rows(log,entries[hold:],paired=False)
    uncertainty={'holdout_candidate_minus_baseline':intervals(scores['holdout_candidate'],scores['holdout_baseline'])}
    for name in ('h2h','confirmation'):
        uncertainty[name+'_minus_50percent']=intervals(scores[name],[[.5,.5] for _ in scores[name]])
    summary=dict(complete=True,rehearsal=smoke,candidate=str(candidate),candidate_sha256=file_hash(candidate),
        baseline=BASELINE,holdout=HOLDOUT,comparisons=comparisons,selfplay=selfplay,uncertainty=uncertainty,
        post_selection_games=sum(v['overall']['games'] for v in [*comparisons.values(),*selfplay.values()]),
        manifest_sha256=file_hash(campaign.root/'manifest.json'),book_sha256=file_hash(book),
        timing={p.stem:read(p)['elapsed_sec'] for p in (campaign.root/'receipts').glob('*.json')},
        notes=['Fixed simulations, not strict equal wall time.',
               'Cap draws are operational outcomes, not solved fortresses.',
               'B2 transfer and selfplay share starts; do not pool as independent tests.',
               'Book states start fresh driver history; generation/reanalysis retain full source history.',
               'Exploratory paired-opening intervals, not multiplicity-adjusted promotion tests.',
               'All prescribed branches ran regardless of earlier scores. No promotion.'])
    expected=20 if smoke else 896
    if summary['post_selection_games']!=expected:raise ValueError('Wrong fixed game total')
    pin_json(campaign.root/'summary.json',summary)
    atomic_json(campaign.root/'status.json',dict(status='complete',summary_sha256=file_hash(campaign.root/'summary.json')))
    print(f'GPU48 {"REHEARSAL" if smoke else "PRODUCTION"} COMPLETE: {campaign.root}/summary.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehearsal-only',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT);os.environ['MONSTER_PINNED_INPUT']='1'
    provenance=identity()
    # A campaign-specific OS lease prevents two launchers racing on receipts.
    # Individual child tools retain ownership of the ordinary GPU worker lease.
    with campaign_lock(RUN/'campaign.lock'):
        rehearsal=Campaign(RUN/'rehearsal',provenance)
        rehearsal.stage('tests',['-m','pytest','tests','-q'])
        candidate=train(rehearsal,True);benchmarks(rehearsal,candidate,True)
        if args.rehearsal_only:return
        if not read(rehearsal.root/'summary.json')['complete']:raise ValueError('Rehearsal guard failed')
        campaign=Campaign(RUN/'production',provenance)
        candidate=train(campaign,False);benchmarks(campaign,candidate,False)


if __name__=='__main__':main()
