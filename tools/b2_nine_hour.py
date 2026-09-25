"""Fixed ~9h B2 follow-up: seed replication, common openings, Black/depth checks."""
import argparse
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

OPPONENTS=[f'models/bootstrap_v{v}/best_value_net.pt' for v in (24,25,26,27)]
REFERENCE='models/candidates/bootstrap_main_gen_0047/arena_selected.pt'


def model_panel():
    panel={}
    for seed,label in ((3173,'original'),(9053,'replica')):
        for arm in ('control','state_cnn'):
            directory=f'models/candidates/b2_001_{arm}' if seed==3173 else f'models/candidates/b2_seed9053_{arm}'
            for epoch in (8,15):
                panel[f'{label}_{arm}_e{epoch}']=f'{directory}/selected_epoch_{epoch:03d}.pt'
    panel['gen47']=REFERENCE
    return panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='benchmarks/b2_validation_20260910')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    root=Path(args.root)
    panel=model_panel()
    book_teachers=OPPONENTS[:3]+['models/candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt',REFERENCE]
    original=ROOT/'iterations/b2_001'
    plan=dict(seed=9053,epochs=[8,15],panel=panel,opponents=OPPONENTS,workers=8,
              ordinary_sims=3200,depth_sims=6400,games_per_panel_model=400,
              common_self_models=['original_control_e15','original_state_cnn_e8','original_state_cnn_e15'],
              depth_models=['original_state_cnn_e8','original_state_cnn_e15','gen47'],
              games_total=4800,primary='state-CNN minus control, epoch15, replicated seeds; Black separately',
              secondary='epoch8 tradeoff and6400-sim transfer; no post-hoc checkpoint substitution',
              book_seeds=[2040000000,2041000000,2042000000],
              note='About9hours, fixed stages finish regardless of clock; no automatic promotion.')
    if args.dry_run:
        print(json.dumps(plan,indent=2));return
    prior=json.loads((ROOT/'benchmarks/b2_001_comparison_20260909/summary.json').read_text())
    if prior.get('complete') is not True:
        raise ValueError('Previous experiment incomplete')
    existing=sorted(set(book_teachers+OPPONENTS+[p for k,p in panel.items() if not k.startswith('replica')]))
    manifest=dict(plan=plan,runtime=runtime_identity(),models={p:file_hash(p) for p in existing},
                  implementations={p:file_hash(ROOT/'tools'/p) for p in ('b2_nine_hour.py','b2_common_selfplay.py','b2_benchmark.py','make_book.py','b2_validation_analysis.py')},
                  data_receipts={str(p):file_hash(p) for p in (original/'receipts/prepare15.json',original/'receipts/prepare24.json')})
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Campaign provenance changed')
    atomic_json(manifest_path,manifest)
    env=dict(os.environ,MONSTER_PINNED_INPUT='1')
    def stage(name,command,artifacts):
        receipt=root/'receipts'/f'{name}.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['command']!=command or any(file_hash(p)!=h for p,h in saved['artifacts'].items()):
                raise ValueError(f'Changed stage {name}')
            return
        atomic_json(root/'status.json',dict(status='running',stage=name,command=command))
        print(f'STAGE {name}: {command}',flush=True)
        try:
            subprocess.run([sys.executable,'-u',*command],cwd=ROOT,env=env,check=True)
        except BaseException as exc:
            atomic_json(root/'status.json',dict(status='failed',stage=name,error=str(exc)))
            raise
        files=[]
        for path in artifacts:
            files.extend(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else files.append(path)
        atomic_json(receipt,dict(command=command,artifacts={str(p):file_hash(p) for p in files}))
    for channels in (15,24):
        data=json.loads((original/f'receipts/prepare{channels}.json').read_text())
        for p,h in data['artifacts'].items():
            if file_hash(p)!=h:raise ValueError(f'Changed dataset {p}')
    for arm in ('control','state_cnn'):
        directory=ROOT/f'models/candidates/b2_seed9053_{arm}'
        receipt=root/'receipts'/f'train_{arm}.json'
        if directory.exists() and not receipt.exists():
            raise ValueError(f'Interrupted/unreceipted training directory: {directory}')
        command=json.loads((original/f'receipts/train_{arm}.json').read_text())['command'][1:]
        for flag,value in (('--model-dir',str(directory)),('--seed','9053')):
            command[command.index(flag)+1]=value
        stage(f'train_{arm}',command,[directory])
    # No post-hoc checkpoint substitution if early stopping ends before epoch15.
    # Preserve the useful remaining panels and explicitly report the missing test.
    missing={name:p for name,p in panel.items() if not Path(p).is_file()}
    if any(not name.startswith('replica') for name in missing):
        raise ValueError('An existing original checkpoint disappeared')
    atomic_json(root/'missing_preregistered_epochs.json',missing)
    panel={name:p for name,p in panel.items() if name not in missing}
    panel_hashes={p:file_hash(p) for p in panel.values()}
    panel_receipt=root/'panel_models.json'
    if panel_receipt.exists() and json.loads(panel_receipt.read_text())!=panel_hashes:
        raise ValueError('Panel weights changed')
    atomic_json(panel_receipt,panel_hashes)
    books={}
    for name,count,seed in (('panel',200,2040000000),('self',200,2041000000),('depth',100,2042000000)):
        path=root/f'{name}_book.json'; books[name]=path
        command=['tools/make_book.py']
        for p in book_teachers:command+=['--model',p]
        command+=['--entries',str(count),'--plies','16','--sims','700','--temperature','.5',
                  '--seed',str(seed),'--workers','8','--oversample','8','--out',str(path)]
        stage(f'book_{name}',command,[path])
        entries=json.loads(path.read_text())['entries']
        if len(entries)!=count or len({(e['fen'],e['half'],e['turn_count']) for e in entries})!=count:
            raise ValueError('Opening count/uniqueness failure')
    def panel_match(name,model,j,games,sims,book,offset):
        path=root/f'{name}_o{j}.json'
        command=['tools/match.py','--model-a',model,'--model-b',OPPONENTS[j],
                 '--games',str(games),'--sims',str(sims),'--engine','native','--workers','8',
                 '--seed',str(2045000000+j*10000),'--book',str(book),'--book-offset',str(offset),
                 '--report-path',str(path),'--game-log',str(path.with_suffix('.jsonl')),'--resume']
        stage(name+f'_o{j}',command,[path,path.with_suffix('.jsonl')])
        result=json.loads(path.read_text())
        if result.get('partial') or result.get('games')!=games:raise ValueError('Incomplete match')
        return result
    results={}
    for name,model in panel.items():
        results[name]=aggregate([panel_match(name,model,j,100,3200,books['panel'],j*50) for j in range(4)])
        atomic_json(root/'panel_progress.json',results)
    selfplay={}
    for name in plan['common_self_models']:
        path=root/f'self_{name}.json'
        stage('self_'+name,['tools/b2_common_selfplay.py','--model',panel[name],
              '--book',str(books['self']),'--output',str(path)],[path,path.with_suffix('.jsonl')])
        selfplay[name]=json.loads(path.read_text())
    depth={}
    for name in plan['depth_models']:
        depth[name]=aggregate([panel_match('depth_'+name,panel[name],j,50,6400,books['depth'],j*25) for j in range(4)])
        atomic_json(root/'depth_progress.json',depth)
    atomic_json(root/'summary.json',dict(complete=True,panel=results,selfplay=selfplay,depth=depth,missing_epochs=missing,
                manifest_sha256=file_hash(manifest_path),note=plan['note']))
    stage('paired_analysis',['tools/b2_validation_analysis.py','--root',str(root)],
          [root/'paired_analysis.json',root/'black_regression_examples.json'])
    atomic_json(root/'status.json',dict(status='complete'))


if __name__=='__main__':main()
