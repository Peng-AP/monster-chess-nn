"""Pure diagnostic bookkeeping tests; no active engine mutation."""
import bisect
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from search_leaf_audit import root_map,error_summary


def test_mapping_respects_file_boundaries_and_splits():
    audit={'splits':{'train':['a','b'],'val':['c'],'test':['d']},
           'row_counts':{'a':2,'b':3,'c':1,'d':2}}
    ends,entries=root_map(audit)
    assert ends==[2,5,6,8]
    assert entries[bisect.bisect_right(ends,2)]==('train','b',2)
    assert entries[bisect.bisect_right(ends,5)]==('val','c',5)
    assert entries[bisect.bisect_right(ends,6)]==('test','d',6)


def test_error_summary_is_white_perspective_without_side_flip():
    rows=[{'absolute':.5,'teacher':.25},{'absolute':-.5,'teacher':-.25}]
    result=error_summary(rows,'absolute')
    assert result['mse']==.0625
    assert result['mean_error']==0
    assert result['mae']==.25
    assert error_summary([],'absolute') is None
