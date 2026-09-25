"""Fresh outcome continuations -> frozen-policy value controls -> fixed play tests."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import start_gen49 as prior
import mainline_study as study
import start_mainline_study as base
from start_gpu48 import Campaign,campaign_lock,read,pin_json
from match_evidence import atomic_json,digest,file_hash,runtime_identity

RUN=ROOT/'benchmarks/value_calibration_20260917'
INITIAL='models/candidates/bootstrap_main_gen_0050/arena_selected.pt'
REFERENCE='models/candidates/bootstrap_main_gen_0049/arena_selected.pt'
REPLAY=ROOT/'data/processed/bootstrap_replay_main_gen_0050'
HASHES={INITIAL:'51b5ddb01db51ae9023eaaf8ccbd896b48a805a52b2707633dc1d7e3f8067f25',
        REFERENCE:'4bcc68a0219acf8c3dc53326d738789e6bd767fccba88fd4f471345567b4a647',
        prior.HOLDOUT:prior.PINNED_MODELS[prior.HOLDOUT],prior.RELEASE:prior.PINNED_MODELS[prior.RELEASE]}
_replay_cache={}


def replay_hashes():
    result={}
    for name in ('positions.npy','capture_results.npy','value_weights.npy','splits.npz'):
        path=REPLAY/name;s=path.stat();key=(s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if name not in _replay_cache or _replay_cache[name][0]!=key:
            _replay_cache[name]=(key,file_hash(path))
        result[name]=_replay_cache[name][1]
    return result


def identity():
    paths=sorted((ROOT/'tools').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    paths += [ROOT/n for n in ('run_value_calibration.py','value_calibration.py',
                               'test_value_calibration.py','VALUE_CALIBRATION_PLAN.md')]
    hashes={p:file_hash(ROOT/p) for p in HASHES}
    if hashes!=HASHES:raise ValueError('Frozen model changed')
    return dict(runtime=runtime_identity(),models=hashes,replay=replay_hashes(),
                implementations={str(p.relative_to(ROOT)):file_hash(p) for p in paths})


def parents(smoke):
    state=base.make_state([])
    return [dict(case_id=f'parent_{i:04d}',state=state,case_sha256=digest(state),
        white_model=INITIAL,black_model=INITIAL,white_sims=8 if smoke else 3200,
        black_sims=8 if smoke else 3200,seed=(2610000000 if smoke else 2600000000)+i,
        sample_index=i,kind='game',deadline_seconds=3600) for i in range(12 if smoke else 192)]


def continuations(roots,smoke):
    return [dict(case_id=r['id'],state=r['state'],case_sha256=digest(r),
        white_model=a,black_model=b,white_sims=8 if smoke else 6400,black_sims=8 if smoke else 6400,
        seed=(2630000000 if smoke else 2620000000)+i*2+j,sample_index=j,kind='game',deadline_seconds=3600)
        for i,r in enumerate(roots) for j,(a,b) in enumerate([(INITIAL,REFERENCE),(REFERENCE,INITIAL)])]


def games(campaign,name,tasks):
    config=campaign.root/(name+'_config.json');pin_json(config,dict(tasks=tasks,protocol='outcome_value_v1'))
    out=campaign.root/name
    campaign.stage(name,['tools/mainline_study.py','--config',str(config),'--output',str(out)],
                   [out/'manifest.json',out/'summary.json'])
    s=read(out/'summary.json')
    if not s['complete'] or s['games']!=len(tasks):raise ValueError('Incomplete outcome games')
    for t in tasks:study.load_result(out/'tasks'/(digest(t)+'.json'),t)
    return config,out


def stage(campaign,name,config,out,outputs):
    path=campaign.root/(name+'_config.json');pin_json(path,config)
    action='train' if name.startswith('train_') else name
    campaign.stage(name,['value_calibration.py',action,'--config',str(path),'--out',str(out)],outputs)


def nominate(scores):
    baseline=scores['baseline'];eligible=[]
    for name in ('replay','continuation'):
        s=scores[name]
        if (s['vs_initial']>=.5 and s['black']>=baseline['black']-.05-1e-12
                and s['v27_white']>=baseline['v27_white']-1e-12
                and s['b2_white']>=baseline['b2_white']-1e-12):eligible.append(name)
    rank=lambda n:((scores[n]['v27_white']+scores[n]['b2_white'])/2,scores[n]['vs_initial'],scores[n]['black'],n)
    return max(eligible,key=rank) if eligible else 'baseline',eligible


def work(campaign,smoke):
    root=campaign.root
    pc,po=games(campaign,'parents',parents(smoke))
    common=dict(smoke=smoke,initial=INITIAL,reference=REFERENCE,replay=str(REPLAY),
                replay_hashes=campaign.provenance['replay'],parents_config=str(pc),parents_output=str(po))
    roots=root/'roots.json'
    stage(campaign,'select',common,roots,[roots])
    selected=read(roots)
    if not selected['complete'] or len(selected['roots'])!=len(parents(smoke)):raise ValueError('Missing selected roots')
    cc,co=games(campaign,'continuations',continuations(selected['roots'],smoke))
    data=root/'data'
    config=dict(common,roots=str(roots),continuations_config=str(cc),continuations_output=str(co))
    names=('old_train','old_val','new_train','new_val','new_test')
    outputs=[data/'complete.json']+[data/(n+'.npz') for n in names]+[data/(n+'_families.json') for n in names]
    stage(campaign,'prepare',config,data,outputs)
    models={'baseline':INITIAL}
    for arm in ('replay','continuation'):
        out=root/'fits'/arm
        config=dict(common,data=str(data),arm=arm)
        stage(campaign,'train_'+arm,config,out,[out/'candidate.pt',out/'complete.json'])
        receipt=read(out/'complete.json')
        if not receipt['complete'] or file_hash(out/'candidate.pt')!=receipt['model_sha256']:
            raise ValueError('Incomplete fit')
        models[arm]=str(out/'candidate.pt')
    model_hashes={n:file_hash(p) for n,p in models.items()};pin_json(root/'arms.json',dict(models=models,hashes=model_hashes))
    screen=Campaign(root/'screen',campaign.provenance,identity_fn=identity,label='VALUE SCREEN')
    opponents={'gen49':REFERENCE,'initial':INITIAL,'v27':prior.RELEASE,'b2':prior.HOLDOUT};audits={};scores={}
    for name,model in models.items():
        reports={}
        for i,(opp,other) in enumerate(opponents.items()):
            label=f'{name}_vs_{opp}'
            audits[label]=prior.free_match(screen,label,model,other,4 if smoke else 80,
                    (2650000000 if smoke else 2640000000)+i*1000000,8 if smoke else 3200)
            reports[opp]=read(screen.root/'play'/(label+'.json'))
        def score(r,side):
            d=r['a_as_'+side];return (d['wins']+.5*d['draws'])/d['games']
        scores[name]=dict(vs_initial=(score(reports['initial'],'white')+score(reports['initial'],'black'))/2,
                          black=score(reports['gen49'],'black'),v27_white=score(reports['v27'],'white'),
                          b2_white=score(reports['b2'],'white'))
    chosen,eligible=nominate(scores);nominee=dict(name=chosen,path=models[chosen],sha256=model_hashes[chosen],eligible=eligible,scores=scores)
    pin_json(root/'nominee.json',nominee);print(f'VALUE NOMINEE {chosen}',flush=True)
    confirm=Campaign(root/'confirmation',campaign.provenance,identity_fn=identity,label='VALUE CONFIRM')
    spec=prior.layout(smoke);spec['seed']=2670000000 if smoke else 2660000000
    gate=prior.free_gate(confirm,'vs_initial',models[chosen],INITIAL,spec,True)
    extras={}
    for i,(name,other,sims) in enumerate([('vs_gen49',REFERENCE,3200),('vs_v27',prior.RELEASE,3200),
                    ('vs_b2',prior.HOLDOUT,3200),('self',models[chosen],3200),
                    ('deep_vs_gen49',REFERENCE,12800),('deep_vs_initial',INITIAL,12800)]):
        extras[name]=prior.free_match(confirm,name,models[chosen],other,4 if smoke else 160,
                 (2690000000 if smoke else 2680000000)+i*1000000,8 if smoke else sims)
    _replay_cache.clear() # Rehash full replay once more before final publication.
    if identity()!=campaign.provenance or any(file_hash(p)!=model_hashes[n] for n,p in models.items()):
        raise ValueError('Inputs or candidates changed')
    pin_json(root/'summary.json',dict(complete=True,rehearsal=smoke,nominee=nominee,screen=audits,gate=gate,extras=extras,
        games=120 if smoke else 3696,arms=models,manifest_sha256=file_hash(root/'manifest.json'),
        notes=['All arms play-tested; no offline-only rejection.','No model promoted or overwritten.',
               'Outcome labels depend on frozen continuation players, not perfect-play proofs.']))
    atomic_json(root/'status.json',dict(status='complete'));print('VALUE CAMPAIGN COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rehearsal-only',action='store_true');args=p.parse_args()
    os.chdir(ROOT);os.environ['MONSTER_PINNED_INPUT']='1';provenance=identity()
    with campaign_lock(RUN/'campaign.lock'):
        for smoke in (True,False):
            if not smoke and args.rehearsal_only:break
            campaign=Campaign(RUN/('rehearsal' if smoke else 'production'),provenance,identity_fn=identity,label='VALUE')
            try:
                if smoke:campaign.stage('tests',['-m','pytest','tests','test_value_calibration.py','-q'])
                elif not read(RUN/'rehearsal/summary.json')['complete']:raise ValueError('Rehearsal required')
                work(campaign,smoke)
            except BaseException as exc:
                atomic_json(campaign.root/'status.json',dict(status='failed',error=str(exc)));raise


if __name__=='__main__':main()
