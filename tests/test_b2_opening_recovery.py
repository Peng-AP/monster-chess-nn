import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from b2_resume_benchmark import validate_book


def sample():
    return dict(plies=16,sims=700,temperature=.5,seed=1990000000,
                entries=[dict(fen=str(i),half=False,turn_count=8) for i in range(100)])


def test_full_book_accepted():
    validate_book(sample())


def test_short_or_duplicate_book_rejected():
    doc=sample()
    doc['entries'].pop()
    with pytest.raises(ValueError,match='exactly100'):
        validate_book(doc)
    doc=sample()
    doc['entries'][99]=doc['entries'][0]
    with pytest.raises(ValueError,match='Duplicate'):
        validate_book(doc)


def test_changed_depth_rejected():
    doc=sample()
    doc['plies']=20
    with pytest.raises(ValueError,match='settings'):
        validate_book(doc)
