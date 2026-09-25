import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from b2_challenger_test import fresh_entries,key


def test_filters_seen_and_duplicate_openings_without_reordering():
    entries=[dict(fen=str(i),half=False,turn_count=10) for i in range(6)]
    docs=[dict(entries=entries[:4]),dict(entries=entries[2:])]
    assert fresh_entries(docs,{key(entries[0])},4)==entries[1:5]


def test_no_fabricated_padding_when_short():
    entry=dict(fen='x',half=True,turn_count=1)
    assert fresh_entries([dict(entries=[entry,entry])],set(),400)==[entry]
