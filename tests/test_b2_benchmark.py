import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from b2_benchmark import aggregate, rank, shortlist
import pytest


def test_ranking_uses_exact_counts_not_rounded_cell_scores():
    def row(black_wins, black_draws):
        report = dict(games=200,
            a_as_white=dict(wins=52,draws=22,losses=26,games=100,score=.629994),
            a_as_black=dict(wins=black_wins,draws=black_draws,
                            losses=100-black_wins-black_draws,games=100,score=.123))
        return dict(path='test',scores=aggregate([report]))
    epoch11 = row(71,14)
    epoch17 = row(75,13)
    assert epoch11['scores']['white']['score'] == .63
    assert epoch17['scores']['black']['score'] == .815
    assert rank(epoch17) < rank(epoch11)


def test_shortlist_deduplicates_best_epoch(tmp_path):
    for i in range(1,7):
        (tmp_path/f'selected_epoch_{i:03d}.pt').write_bytes(bytes([i]))
    (tmp_path/'train_run_test.json').write_text(json.dumps(dict(best_epoch=4)))
    assert [Path(p).name for p in shortlist(tmp_path)] == [
        'selected_epoch_002.pt','selected_epoch_004.pt','selected_epoch_006.pt']


def test_partial_report_refused():
    with pytest.raises(ValueError,match='Incomplete'):
        aggregate([dict(partial=True,games=200)])
