import numpy as np
import torch
import value_calibration as v
import run_value_calibration as c


def test_value_perspectives_and_white_halves():
    white={'fen':'8/8/8/8/8/8/8/8 w - - 0 1','half':False}
    second=dict(white,half=True);black=dict(white,fen=white['fen'].replace(' w ',' b '))
    assert [v.phase(s) for s in (white,second,black)]==['white_first','white_second','black']
    assert [v.outcome_target(1,s) for s in (white,second,black)]==[1,1,-1]
    assert v.outcome_target(0,black)==0


def test_exact_input_leakage_and_conflicting_targets():
    groups={n:[] for n in ('new_test','new_val','old_val','new_train','old_train')}
    row=lambda k,y,f:dict(key=k,y=y,family=f)
    groups['new_test']=[row('held',1,'test')]
    groups['new_train']=[row('held',1,'train'),row('mixed',1,'a'),row('mixed',-1,'b')]
    groups['old_val']=[row('old-held',0,'v')]
    groups['old_train']=[row('old-held',1,'t'),row('mixed',1,'t2')]
    result,census=v.clean_splits(groups)
    assert len(result['new_train'])==1 and result['new_train'][0]['y']==0
    assert census['new_train']['conflicting_outcomes']==1
    assert census['old_train']['excluded']==1


def test_nonvalue_buffers_are_checked():
    a={'value_head.2.weight':torch.ones(1),'stem.bn.running_mean':torch.ones(1),'policy.weight':torch.ones(1)}
    b={k:x.clone() for k,x in a.items()};b['value_head.2.weight']*=2
    assert v.frozen_equal(a,b)
    b['stem.bn.running_mean']*=2
    assert not v.frozen_equal(a,b)


def test_counts_and_root_history():
    for smoke,count in ((True,12),(False,192)):
        parents=c.parents(smoke);assert len(parents)==count
        game,_=c.study.checked_restore(parents[0]['state']);assert game.turn_count==0
        roots=[dict(id=str(i),state=t['state'],split='train') for i,t in enumerate(parents)]
        tasks=c.continuations(roots,smoke);assert len(tasks)==2*count
        assert len({c.digest(t) for t in tasks})==len(tasks)
        assert count*3+(48 if smoke else 960)+(36 if smoke else 2160)==(120 if smoke else 3696)


def test_nomination_never_rejects_baseline_or_relaxes_black():
    scores={n:dict(vs_initial=.6,black=.9,v27_white=.8,b2_white=.9) for n in ('baseline','replay','continuation')}
    scores['continuation']['v27_white']=.95
    assert c.nominate(scores)[0]=='continuation'
    scores['continuation']['black']=.7
    assert c.nominate(scores)[0]=='replay'
    scores['replay']['b2_white']=.8
    assert c.nominate(scores)==('baseline',[])
