import copy
import json
from pathlib import Path
import subprocess
import sys

import chess
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import start_gpu48 as campaign
from b2_common_selfplay import outcome_counts
from match_evidence import atomic_json,digest
from monster_chess import MonsterChessGame
from benchmark import play_one


def test_recipe_depths_counts_and_holdout():
    recipe=campaign.read(ROOT/'tools/recipes/gen48.json')
    assert [recipe[k] for k in ('free_games','fresh_games','league_games','fork_games')]==[1600,800,400,400]
    assert recipe['sims']==1600 and recipe['fork_sims']==6400 and recipe['coverage_reanalysis']
    assert campaign.HOLDOUT not in [recipe['model'],*recipe['opponents'],*recipe['prefix_models']]
    assert recipe['model']==campaign.BASELINE


@pytest.mark.parametrize('smoke,total',[(True,20),(False,896)])
def test_fixed_layout_is_disjoint_except_intentional_common_holdout(smoke,total):
    s=campaign.layout(smoke)
    a=set(range(s['h2h_games']//2))
    b=set(range(s['confirmation_offset'],s['confirmation_offset']+s['h2h_games']//2))
    c=set(range(s['holdout_offset'],s['entries']))
    assert not (a&b or a&c or b&c)
    assert len(c)==s['self_games']==s['transfer_games']//2
    assert 2*(s['h2h_games']+s['transfer_games']+s['self_games'])==total


def test_training_is_scratch_same_recipe_and_stops_before_binding():
    cmd=campaign.iteration_command(False)
    parsed=campaign.iterate.build_parser().parse_args(cmd[cmd.index('--')+1:])
    assert (parsed.epochs,parsed.patience,parsed.seed,parsed.lr,parsed.batch_size)==(30,10,3173,.002,256)
    assert parsed.replay_generations==8 and parsed.anchor_data=='none'
    assert parsed.through_phase=='checkpoint_screen' and not parsed.promote_on_pass
    assert not parsed.reject_on_offline_regression and parsed.reanalysis_sample==24000
    assert '--init' not in cmd


def test_selfplay_cap_leans_are_draws():
    rows=[{'result_for_a':r} for r in (1,-1,.5,-.5,0)]
    assert outcome_counts(rows)==(1,1,3)


class CaptureEngine:
    def get_best_action(self,game,temperature):
        move=chess.Move.from_uci('e2e1')
        return move,{move.uci():1},1


def row_for_entry(entry):
    result,plies,_,opening=play_one(CaptureEngine(),CaptureEngine(),start_fen=entry['fen'],
        start_half=entry['half'],start_turn_count=entry['turn_count'],return_opening=True)
    game=opening.pop('game')
    return dict(entry=0,a_is_white=True,result_for_a=result,white_score=campaign.game_score(result),
                plies=plies,opening=opening,game=game)


def test_audit_replays_captures_and_caps(tmp_path):
    for fen,turn,expected in [('7k/8/8/8/8/8/4r3/4K3 b - - 0 1',0,'black_capture'),
                              (MonsterChessGame().fen(),150,'turn_cap')]:
        entry=dict(fen=fen,half=False,turn_count=turn)
        row=row_for_entry(entry);path=tmp_path/'games.jsonl'
        path.write_text(json.dumps(row)+'\n')
        result,_=campaign.audit_rows(path,[entry],paired=False)
        assert result['endings']=={expected:1}
        assert result['overall']['score']==(0 if expected=='black_capture' else .5)


@pytest.mark.parametrize('fault',['duplicate','missing','digest','action','outcome','state'])
def test_audit_rejects_bad_evidence(tmp_path,fault):
    entry=dict(fen='7k/8/8/8/8/8/4r3/4K3 b - - 0 1',half=False,turn_count=0)
    row=row_for_entry(entry);rows=[row]
    if fault=='duplicate':rows.append(copy.deepcopy(row))
    if fault=='missing':rows=[]
    if fault=='digest':row['game']['trajectory_sha256']='changed'
    if fault=='outcome':row['result_for_a']=1
    if fault in ('state','action'):
        last=row['game']['trajectory'][-1]
        if fault=='state':last['turn_count']+=1
        else:last['action']=['a1a8']
        row['game']['trajectory_sha256']=digest(row['game']['trajectory'])
    path=tmp_path/'games.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):campaign.audit_rows(path,[entry],paired=False)


def test_stage_failure_cannot_create_success_receipt(tmp_path,monkeypatch):
    provenance={'fixed':1};job=campaign.Campaign(tmp_path,provenance)
    monkeypatch.setattr(campaign,'identity',lambda:provenance)
    def fail(*a,**kw):raise subprocess.CalledProcessError(2,'failed child')
    monkeypatch.setattr(campaign.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):job.stage('generation',['fake'])
    assert not (tmp_path/'receipts/generation.json').exists()
    assert campaign.read(tmp_path/'status.json')['status']=='failed'


def test_changed_completed_output_is_not_reused(tmp_path,monkeypatch):
    provenance={'fixed':1};job=campaign.Campaign(tmp_path,provenance)
    monkeypatch.setattr(campaign,'identity',lambda:provenance)
    output=tmp_path/'output.json';atomic_json(output,{'x':1})
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**k:None)
    job.stage('test',['fake'],[output]);atomic_json(output,{'x':2})
    with pytest.raises(ValueError,match='Changed completed'):job.stage('test',['fake'],[output])


def test_launcher_lock_does_not_bypass_worker_lease(tmp_path):
    import worker_lease
    assert worker_lease._depth==0
    with campaign.campaign_lock(tmp_path/'campaign.lock'):
        assert worker_lease._depth==0
        with worker_lease.worker_lease(tmp_path/'gpu.lock'):
            assert worker_lease._depth==1
