"""Fixed checkpoint recovery: crossover probes, matched roots, screen, confirm."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import start_gen49 as prior
import start_mainline_study as study_driver
import mainline_study as study
from start_gpu48 import Campaign, campaign_lock, read, pin_json
from match_evidence import atomic_json, digest, file_hash, runtime_identity

RUN=ROOT/'benchmarks/gen50_recovery_20260916'
GEN49='models/candidates/bootstrap_main_gen_0049/arena_selected.pt'
MODELS={'gen49':GEN49, **{f'e{e}':f'models/candidates/bootstrap_main_gen_0050/selected_epoch_{e:03d}.pt'
                         for e in (10,13,14,15,23)}}
OPPONENTS={'gen49':GEN49,'v27':prior.RELEASE,'b2':prior.HOLDOUT}


def cases():
    prefixes=[['e2e4','d2d4','d7d5'],['e2e4','d2d4','d7d5','c2c4'],
              ['e2e4','d2d4','d7d5','c2c4','c4c5','g8f6'],
              ['e2e4','d2d4','d7d5','e4e5','c2c4','f7f6'],
              ['e2e4','d2d4','d7d5','e4e5','e1e2','f7f6'],
              ['d2d4','d4d5','c7c5']]
    return [dict(id=f'root_{i}',state=study_driver.make_state(m),
                 source='predeclared inspected regression prefix, not blind') for i,m in enumerate(prefixes)]


def identity():
    paths=sorted((ROOT/'tools').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    paths += [ROOT/n for n in ('run_gen50_recovery.py','recovery_probe.py',
                               'test_gen50_recovery.py','GEN50_RECOVERY_PLAN.md')]
    model_paths=set(MODELS.values())|set(OPPONENTS.values())
    return dict(runtime=runtime_identity(),models={p:file_hash(ROOT/p) for p in sorted(model_paths)},
                implementation={str(p.relative_to(ROOT)):file_hash(p) for p in paths},cases=cases())


def task(case, white, black, sims, seed, sample=0, kind='game'):
    return dict(case_id=case['id'],state=case['state'],case_sha256=digest(case),
                white_model=white,black_model=black,white_sims=sims,black_sims=sims,
                seed=seed,sample_index=sample,kind=kind,deadline_seconds=3600)


def conditional_tasks(smoke):
    result=[]
    for depth_index,sims in enumerate([8] if smoke else [3200,12800]):
        for white in MODELS.values():
            for opp_index,black in enumerate((GEN49,prior.RELEASE)):
                for index,case in enumerate(cases()):
                    for sample in range(1 if smoke else 2):
                        seed=(2410000000 if smoke else 2400000000)+depth_index*10000+opp_index*1000+index*10+sample
                        result.append(task(case,white,black,sims,seed,sample))
    return result


def probe_tasks(smoke):
    result=[]
    for p,v in [('gen49','gen49'),('e14','e14'),('gen49','e14'),('e14','gen49')]:
        for depth_index,sims in enumerate([8] if smoke else [3200,12800,51200]):
            for index,case in enumerate(cases()):
                t=task(case,MODELS[p],MODELS[p],sims,
                       (2420000000 if smoke else 2430000000)+depth_index*100+index,kind='probe')
                t['value_model']=MODELS[v]
                result.append(t)
    return result


def screen_schedule(smoke):
    return [dict(name=f'{name}_vs_{opp}',a=model,b=other,games=4 if smoke else 80,
                 sims=8 if smoke else 3200,
                 seed=(2450000000 if smoke else 2440000000)+i*1000000)
            for name,model in MODELS.items() for i,(opp,other) in enumerate(OPPONENTS.items())]


def nominate(scores):
    baseline=scores['e14']
    eligible=[]
    for name in MODELS:
        if name=='gen49':continue
        s=scores[name]
        if (s['h2h']>=.5 and s['black']>=baseline['black']-.05-1e-12
                and s['v27_white']>=baseline['v27_white']-1e-12
                and s['b2_white']>=baseline['b2_white']-1e-12):
            eligible.append(name)
    rank=lambda n:((scores[n]['v27_white']+scores[n]['b2_white'])/2,
                   scores[n]['black'],scores[n]['h2h'],-int(n[1:]))
    chosen=max(eligible,key=rank) if eligible else 'e14'
    return dict(name=chosen,path=MODELS[chosen],eligible=eligible,scores=scores,
                fallback=not eligible)


def verify_conditional(config,out):
    summary=read(out/'summary.json')
    if not summary['complete'] or summary['games']!=len(config['tasks']):
        raise ValueError('Incomplete conditional stage')
    for t in config['tasks']:
        study.load_result(out/'tasks'/(digest(t)+'.json'),t)
    return {k:v for k,v in summary.items() if k!='evidence'}


def work(campaign,smoke):
    root=campaign.root
    config=root/'probe_config.json'
    pin_json(config,dict(smoke=smoke,models=MODELS,cases=cases(),tasks=probe_tasks(smoke)))
    out=root/'probes'
    campaign.stage('probes',['recovery_probe.py','--config',str(config),'--output',str(out)],
                   [out/'manifest.json',out/'raw_network.json',out/'summary.json'],exclusive=True)
    probes=read(out/'summary.json')
    if not probes['complete'] or len(probes['probes'])!=len(probe_tasks(smoke)):
        raise ValueError('Incomplete crossover probes')
    for p,h in probes['evidence'].items():
        if file_hash(p)!=h:raise ValueError('Changed probe evidence')
    conditional=dict(tasks=conditional_tasks(smoke),protocol='matched_recovery_roots_v1')
    config=root/'conditional_config.json'; pin_json(config,conditional)
    out=root/'conditional'
    campaign.stage('conditional',['tools/mainline_study.py','--config',str(config),'--output',str(out)],
                   [out/'manifest.json',out/'summary.json'])
    cond_summary=verify_conditional(conditional,out)
    screen=Campaign(root/'screen',campaign.provenance,identity_fn=identity,label='RECOVERY SCREEN')
    audits={}
    for item in screen_schedule(smoke):
        audits[item['name']]=prior.free_match(screen,**item)
    scores={}
    for name in MODELS:
        reports={opp:read(screen.root/'play'/f'{name}_vs_{opp}.json') for opp in OPPONENTS}
        # Compute from counts rather than rounded display scores.
        def score(r,side):
            x=r['a_as_'+side]; return (x['wins']+.5*x['draws'])/x['games']
        scores[name]=dict(h2h=(score(reports['gen49'],'white')+score(reports['gen49'],'black'))/2,
             black=score(reports['gen49'],'black'),v27_white=score(reports['v27'],'white'),
             b2_white=score(reports['b2'],'white'))
    nominee=nominate(scores); nominee['sha256']=file_hash(nominee['path'])
    pin_json(root/'nominee.json',nominee)
    print(f'RECOVERY NOMINEE: {nominee["name"]}, eligible={nominee["eligible"]}',flush=True)
    confirm=Campaign(root/'confirmation',campaign.provenance,identity_fn=identity,label='RECOVERY CONFIRM')
    spec=prior.layout(smoke); spec['seed']=2490000000 if smoke else 2480000000
    gate=prior.free_gate(confirm,'vs_gen49',nominee['path'],GEN49,spec,True)
    extras={}
    for i,(name,other,sims) in enumerate([('vs_v27',prior.RELEASE,3200),
                ('vs_b2',prior.HOLDOUT,3200),('self',nominee['path'],3200),('deep_vs_gen49',GEN49,12800)]):
        extras[name]=prior.free_match(confirm,name,nominee['path'],other,4 if smoke else 160,
                     (2510000000 if smoke else 2500000000)+i*1000000,8 if smoke else sims)
    if identity()!=campaign.provenance or file_hash(nominee['path'])!=nominee['sha256']:
        raise ValueError('Inputs changed before publication')
    pin_json(root/'summary.json',dict(complete=True,rehearsal=smoke,nominee=nominee,
        conditional=cond_summary,screen=audits,gate=gate,extras=extras,
        games=172 if smoke else 3568,root_probes=24 if smoke else 72,
        manifest_sha256=file_hash(root/'manifest.json'),
        notes=['Diagnostic roots selected from failures; not blind.',
               'Screen nomination is not independent evidence. No model overwritten or promoted.']))
    atomic_json(root/'status.json',dict(status='complete'))
    print(f'RECOVERY {"REHEARSAL" if smoke else "PRODUCTION"} COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehearsal-only',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT); os.environ['MONSTER_PINNED_INPUT']='1'
    provenance=identity()
    with campaign_lock(RUN/'campaign.lock'):
        for smoke in (True,False):
            if not smoke and args.rehearsal_only:break
            campaign=Campaign(RUN/('rehearsal' if smoke else 'production'),provenance,
                              identity_fn=identity,label='RECOVERY')
            try:
                if smoke:campaign.stage('tests',['-m','pytest','tests','test_gen50_recovery.py','-q'])
                elif not read(RUN/'rehearsal/summary.json')['complete']:raise ValueError('Rehearsal required')
                work(campaign,smoke)
            except BaseException as exc:
                atomic_json(campaign.root/'status.json',dict(status='failed',error=str(exc)))
                raise


if __name__=='__main__':main()
