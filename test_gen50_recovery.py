import copy
import run_gen50_recovery as c
from recovery_probe import mix_bridges


def test_bridge_routes_value_and_policy_without_conversion():
    p=lambda *a:(b'policy-model-value',b'policy-model-logits')
    v=lambda *a:(b'value-model-value',b'value-model-logits')
    assert mix_bridges(p,v)(b'input',1,15)==(b'value-model-value',b'policy-model-logits')
    assert mix_bridges(p,p) is p


def test_cases_and_counts():
    for case in c.cases():
        game,_=c.study.checked_restore(case['state'])
        assert game.is_white_turn
    for smoke in (True,False):
        tasks=c.conditional_tasks(smoke)
        assert len(tasks)==(72 if smoke else 288)
        assert len({c.digest(t) for t in tasks})==len(tasks)
        probes=c.probe_tasks(smoke)
        assert len(probes)==(24 if smoke else 72)
        assert len({c.digest(t) for t in probes})==len(probes)
        assert sum(r['games'] for r in c.screen_schedule(smoke))==(72 if smoke else 1440)
        assert len(tasks)+(72 if smoke else 1440)+(28 if smoke else 1840)==(172 if smoke else 3568)


def test_matched_seeds_and_second_white_half():
    tasks=c.conditional_tasks(False)
    assert any(t['state']['half'] for t in tasks)
    for t in tasks:
        peers=[p for p in tasks if p['seed']==t['seed']]
        assert len(peers)==len(c.MODELS)
        assert all(p['state']==t['state'] and p['black_model']==t['black_model'] for p in peers)


def test_selection_protects_black_and_both_white_opponents():
    scores={n:dict(h2h=.6,black=.8,v27_white=.8,b2_white=.9) for n in c.MODELS}
    scores['e10'].update(black=.7,v27_white=1.,b2_white=1.)
    scores['e13'].update(v27_white=1.,b2_white=.89)
    scores['e15'].update(v27_white=.9)
    scores['e23'].update(h2h=.4,v27_white=1.,b2_white=1.)
    r=c.nominate(scores)
    assert r['name']=='e15' and set(r['eligible'])=={'e14','e15'}
    for s in scores.values():s['h2h']=.4
    assert c.nominate(scores)['name']=='e14' and c.nominate(scores)['fallback']


def test_seed_namespaces_do_not_overlap_confirmation():
    from match import build_tasks
    for smoke in (True,False):
        # Shared seeds within matched screens are intentional; confirmation is disjoint.
        seeds={t[1] for r in c.screen_schedule(smoke) for t in build_tasks(r['games'],r['seed'],16)}
        base=2490000000 if smoke else 2480000000
        spec=c.prior.layout(smoke)
        other=set()
        for offset,n in [(4000000,2*spec['final_per_side']),(4100000,spec['final_par']),
                         (4200000,2*spec['final_per_side'])]:
            other.update(t[1] for t in build_tasks(n,base+offset,16))
        for i in range(4):
            other.update(t[1] for t in build_tasks(4 if smoke else 160,
                         (2510000000 if smoke else 2500000000)+i*1000000,16))
        assert not seeds&other
        assert len(other)==(28 if smoke else 1840)
