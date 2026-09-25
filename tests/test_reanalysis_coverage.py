from collections import Counter
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from reanalyze_coverage import covered_sample,family_roots


def row(path,line,side='black',half=0,parent=None):
    record=dict(fen='test',current_player=side,half=half)
    if parent:record['source_record']=dict(path=parent,line=1)
    return dict(path=path,line=line,record=record)


def test_transitive_forks_count_against_same_family():
    rows=[row('root',i) for i in range(30)]
    rows += [row('fork',i,parent='root') for i in range(30)]
    rows += [row('nested',i,parent='fork') for i in range(30)]
    rows += [row('independent',i) for i in range(30)]
    chosen,stats,roots=covered_sample(rows,20,1,3173,max_per_family=10)
    assert roots['nested']=='root'
    assert Counter(roots[r['path']] for r in chosen)=={'root':10,'independent':10}
    assert stats['maximum_per_family']==10


def test_side_phase_quotas_reproducibility_and_coverage():
    rows=[row(str(f),i,'black' if i%3==0 else 'white',int(i%3==2)) for f in range(20) for i in range(90)]
    a,stats,_=covered_sample(rows,100,.6,7)
    b,other,_=covered_sample(reversed(rows),100,.6,7)
    # Repeat the same ordered input, not a promise of row-order-independent RNG.
    assert a==covered_sample(rows,100,.6,7)[0]
    assert stats['actual_phase']=={'black':60,'white_first':20,'white_second':20}
    assert stats['sampled_families']==20 and stats['maximum_per_family']==5
    assert len(a)==len(b)==100
    assert len({(r['path'],r['line']) for r in a})==100


def test_insufficient_family_capacity_fails_instead_of_repeating():
    with pytest.raises(ValueError,match='Family cap'):
        covered_sample([row('one',i) for i in range(100)],17,1,1)


def test_phase_shortage_is_reported_and_total_kept():
    selected,stats,_=covered_sample([row(str(f),i) for f in range(4) for i in range(12)],20,.6,1)
    assert len(selected)==20
    assert stats['requested_phase']['black']==12
    assert stats['actual_phase']=={'black':20}


@pytest.mark.parametrize('rows',[[row('a',1,parent='missing')],
    [row('a',1,parent='b'),row('b',1,parent='a')],
    [row('a',1),row('a',2,parent='b'),row('b',1)]])
def test_bad_family_graph_fails(rows):
    with pytest.raises(ValueError):family_roots(rows)
