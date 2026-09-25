"""One locked challenger versus gen47; no training or architecture selection."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from match_evidence import atomic_json,file_hash,runtime_identity
from b2_benchmark import aggregate
from b2_validation_analysis import intervals

CANDIDATE='models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt'
REFERENCE='models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
OPPONENTS=[f'models/bootstrap_v{v}/best_value_net.pt' for v in (24,25,26,27)]


def key(entry):return (entry['fen'],bool(entry['half']),int(entry['turn_count']))


def fresh_entries(documents,excluded,count):
    found=[];seen=set(excluded)
    for doc in documents:
        for entry in doc['entries']:
            if key(entry) not in seen:
                seen.add(key(entry));found.append(entry)
                if len(found)==count:return found
    return found


def main():
    root=ROOT/'benchmarks/b2_challenger_confirmation_20260910'
    teachers=OPPONENTS[:3]+['models/candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt',REFERENCE]
    humans=[ROOT/f'data/raw/human_games/black_2026_07/game_{n:05d}.jsonl' for n in (31,32)]
    prior_books=sorted(p for folder in ROOT.glob('benchmarks/b2*') if folder.is_dir() and folder!=root
                       for p in folder.glob('*book.json'))
    manifest=dict(candidate=file_hash(CANDIDATE),reference=file_hash(REFERENCE),runtime=runtime_identity(),
        implementation=file_hash(__file__),teachers={p:file_hash(p) for p in teachers},
        opponents={p:file_hash(p) for p in OPPONENTS},human_sources={str(p):file_hash(p) for p in humans},
        excluded_books={str(p):file_hash(p) for p in prior_books},
        protocol=dict(direct_paired=800,direct_free=400,direct_deep=400,broad_per_model=400,
                      common_self_per_model=200,sims=3200,deep_sims=6400,workers=8,
                      opening_seed=2060000000,match_seed=2090000000),
        scope='Locked checkpoint. Fixed-simulation tests, NOT equal-time. No promotion or optional stopping.')
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Provenance changed')
    atomic_json(manifest_path,manifest)
    env=dict(os.environ,MONSTER_PINNED_INPUT='1')
    def stage(name,args,outputs):
        receipt=root/'receipts'/f'{name}.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['command']!=args or any(file_hash(p)!=h for p,h in saved['outputs'].items()):raise ValueError('Changed completed stage')
            return
        atomic_json(root/'status.json',dict(status='running',stage=name))
        print(f'STAGE {name}: {args}',flush=True)
        try:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,check=True)
        except BaseException as exc:
            atomic_json(root/'status.json',dict(status='failed',stage=name,error=str(exc)));raise
        atomic_json(receipt,dict(command=args,outputs={str(p):file_hash(p) for p in outputs}))
    # Known human regressions first, with their reconstruction ambiguities retained.
    for i,human in enumerate(humans):
        rows=[json.loads(line) for line in human.read_text().splitlines() if line.strip()]
        output=root/f'human_{i}.json'
        stage(f'human_{i}',['tools/probe_human_line.py','--human-game',str(human),'--models',REFERENCE,CANDIDATE,
            '--opponent-model',REFERENCE,'--cases',*[str(j) for j in range(0,min(20,len(rows)),2)],
            '--sims','3200','6400','--seeds','101','--continue-plies','5','--output',str(output),'--resume'],[output])
    excluded={key(e) for p in prior_books for e in json.loads(p.read_text()).get('entries',[])}
    book=root/'confirmation_book.json';book_receipt=book.with_suffix('.receipt.json')
    if book_receipt.exists():
        if file_hash(book)!=json.loads(book_receipt.read_text())['sha256']:raise ValueError('Book changed')
    else:
        documents=[]
        for i in range(3):
            path=root/f'proposal_{i}.json'
            command=['tools/make_book.py']
            for p in teachers:command+=['--model',p]
            command+=['--entries','400','--seed',str(2060000000+i*1000000),'--plies','16',
                      '--sims','700','--temperature','.5','--oversample','8','--workers','8','--out',str(path)]
            stage(f'proposal_{i}',command,[path]);documents.append(json.loads(path.read_text()))
            entries=fresh_entries(documents,excluded,400)
            if len(entries)==400:break
        if len(entries)!=400:raise ValueError('Not enough unseen openings; no shortened test')
        doc=dict(documents[0],entries=entries,recovery_selection='first400 distinct states absent from prior B2 books',
                 excluded_count=len(excluded),proposal_hashes={str(root/f'proposal_{i}.json'):file_hash(root/f'proposal_{i}.json') for i in range(len(documents))})
        atomic_json(book,doc);atomic_json(book_receipt,dict(sha256=file_hash(book)))
    def match(name,a,b,games,sims,paired,offset=0):
        path=root/f'{name}.json';log=path.with_suffix('.jsonl')
        command=['tools/match.py','--model-a',a,'--model-b',b,'--games',str(games),'--sims',str(sims),
            '--engine','native','--workers','8','--seed','2090000000','--report-path',str(path),'--game-log',str(log),'--resume']
        if paired:command+=['--book',str(book),'--book-offset',str(offset)]
        stage(name,command,[path,log]);r=json.loads(path.read_text())
        if r.get('partial') or r.get('games')!=games:raise ValueError('Incomplete match')
        return r
    direct={}
    for name,games,sims,paired in [('paired',800,3200,True),('free',400,3200,False),('deep',400,6400,True)]:
        direct[name]=match('direct_'+name,CANDIDATE,REFERENCE,games,sims,paired)
    broad={}
    for name,model in [('candidate',CANDIDATE),('reference',REFERENCE)]:
        broad[name]=aggregate([match(f'broad_{name}_o{j}',model,p,100,3200,True,j*50) for j,p in enumerate(OPPONENTS)])
    selfplay={}
    for name,model in [('candidate',CANDIDATE),('reference',REFERENCE)]:
        path=root/f'self_{name}.json'
        stage('self_'+name,['tools/b2_common_selfplay.py','--model',model,'--book',str(book),'--output',str(path)],
              [path,path.with_suffix('.jsonl')]);selfplay[name]=json.loads(path.read_text())
    def scores(paths):
        rows=[json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
        table={(r['entry'],r['a_is_white']):(r['result_for_a']+1)/2 for r in rows}
        return [[table[(entry,side)] for side in (True,False)] for entry in sorted({k[0] for k in table})]
    candidate=scores([root/f'broad_candidate_o{j}.jsonl' for j in range(4)])
    reference=scores([root/f'broad_reference_o{j}.jsonl' for j in range(4)])
    uncertainty=dict(broad_candidate_minus_reference=intervals(candidate,reference))
    for name in ('paired','deep'):
        values=scores([root/f'direct_{name}.jsonl'])
        uncertainty['direct_'+name+'_minus_50percent']=intervals(values,[[.5,.5] for _ in values])
    atomic_json(root/'summary.json',dict(complete=True,direct=direct,broad=broad,selfplay=selfplay,
        uncertainty=uncertainty,manifest_sha256=file_hash(manifest_path),
        notes=['Known human regressions are diagnostic, not pristine held-out positions.',
               'No per-move clock support: equal-time remains untested. No automatic promotion.',
               'Depth and broad tests share subsets of the confirmation book; do not pool as independent samples.']))
    atomic_json(root/'status.json',dict(status='complete'))


if __name__=='__main__':main()
