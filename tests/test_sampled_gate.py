"""v3 estimand, fixed-sample stopping, provenance and recovery contracts."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import gate_sampled
import worker_lease
from free_gate_stats import leg_stats
from sampled_gate_stats import self_par, verdict
from match_evidence import MatchJournal, task_id
from match import build_tasks


def row(i, white=True, result=1):
    return {"a_is_white": white, "result_for_a": result, "plies": 20,
            "opening": {"fen": f"position-{i}", "half": False,
                        "turn_count": 10, "complete": True}}


def test_self_par_uses_all_actual_colors_and_caps_as_draws():
    rows = [row(0, True, 1), row(1, False, -1), row(2, False, .5), row(3, True, -.5)]
    par = self_par(rows)
    assert par["n"] == 4
    assert par["score"] == .5
    assert par["sides"]["white"]["score"] == .75
    assert par["sides"]["black"]["score"] == .25
    assert par["sides"]["white"]["n"] == 4  # same 4, NOT 8 independent games


def test_duplicates_keep_frequency_and_novelty_is_not_a_gate():
    par = self_par([row(i, result=0) for i in range(4)])
    rows = [row(0, aw) for aw in (True, False) for _ in range(3)]
    rows += [row(1, aw, -1) for aw in (True, False)]
    stats = leg_stats(rows, {(True, "position-0", False, 10),
                            (False, "position-0", False, 10)})
    assert stats["sampled"]["score"] == .75
    assert stats["unique"]["score"] == .5
    assert verdict(par, dict(vs_bar=stats, vs_bar_confirm=stats), 4, 4)["verdict"] == "PASS"


def test_endpoint_conflict_is_diagnostic_not_invalid_sample():
    par = self_par([row(i, result=0) for i in range(4)])
    rows = [row(0, aw) for aw in (True, False) for _ in range(3)]
    rows += [row(0, aw, -1) for aw in (True, False)]
    stats = leg_stats(rows)
    assert stats["endpoint_outcome_conflicts"] == 2
    assert verdict(par, dict(vs_bar=stats, vs_bar_confirm=stats), 4, 4)["eligible"]


def test_exact_counts_not_just_at_least_and_black_floor_is_preserved():
    par = self_par([row(i, result=-1) for i in range(4)])  # Black par = 1
    rows = [row(i, aw, 1 if aw or i < 2 else -1) for aw in (True, False) for i in range(4)]
    stats = leg_stats(rows)
    out = verdict(par, dict(vs_bar=stats, vs_bar_confirm=stats), 4, 4)
    assert out["verdict"] == "FAIL"
    assert any("black" in s for s in out["failures"])
    assert verdict(par, dict(vs_bar=stats, vs_bar_confirm=stats), 3, 4)["verdict"] == "INCONCLUSIVE"
    assert verdict(par, dict(vs_bar=stats), 4, 4)["verdict"] == "INCONCLUSIVE"


def setup_gate(tmp_path, monkeypatch):
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    bar = tmp_path / "bar.pt"
    bar.write_bytes(b"bar")
    output = tmp_path / "gate"
    args = gate_sampled.parser().parse_args([
        "--model", str(model), "--bar-model", str(bar), "--sims", "8",
        "--target-per-side", "2", "--par-games", "4", "--run-dir", str(output)])
    monkeypatch.setattr(gate_sampled, "runtime_identity", lambda: {"engine": "test"})
    monkeypatch.setattr(worker_lease, "DEFAULT_PATH", tmp_path / "test_workers.lock")
    return args, output


def fake_match(calls, interrupt=None):
    def run(a, b, **kw):
        tasks = build_tasks(kw["games"], kw["seed"], kw["opening_temp_plies"])
        from argparse import Namespace
        settings = gate_sampled.match_settings(a, b, kw, Namespace(**kw))
        journal = MatchJournal(kw["game_log"], settings, tasks, resume=True)
        for task in journal.pending:
            aw, seed = task[0], task[1]
            calls.append((Path(kw["game_log"]).stem, seed))
            journal.append(dict(row(seed, aw, 0 if a == b else 1),
                                task_id=task_id(task), seed=seed, pair=None))
            if interrupt and interrupt[0]:
                interrupt[0] = False
                raise RuntimeError("simulated interruption")
        return {}
    return run


def test_real_journal_interruption_resume_only_missing_and_completed_no_replay(tmp_path, monkeypatch):
    args, output = setup_gate(tmp_path, monkeypatch)
    calls, interrupt = [], [True]
    monkeypatch.setattr(gate_sampled, "run_match", fake_match(calls, interrupt))
    with pytest.raises(RuntimeError, match="simulated"):
        gate_sampled.run_gate(args)
    assert len(calls) == 1
    args.run_dir, args.resume = None, str(output)
    report = gate_sampled.run_gate(args)
    assert report["verdict"] == "PASS"
    assert len(calls) == 12
    assert len(set(calls)) == 12
    assert gate_sampled.run_gate(args) == report
    assert len(calls) == 12
    with (output / "vs_bar.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="modified"):
        gate_sampled.run_gate(args)


def test_resume_rejects_changed_model_and_settings(tmp_path, monkeypatch):
    args, output = setup_gate(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_sampled, "run_match", fake_match([]))
    gate_sampled.run_gate(args)
    args.run_dir, args.resume = None, str(output)
    args.target_per_side = 3
    with pytest.raises(ValueError, match="provenance"):
        gate_sampled.run_gate(args)
    args.target_per_side = 2
    Path(args.model).write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        gate_sampled.run_gate(args)


def test_seed_blocks_do_not_overlap(tmp_path, monkeypatch):
    args, _ = setup_gate(tmp_path, monkeypatch)
    args.target_per_side, args.par_games = 1000, 2000
    seeds = [t[1] for item in gate_sampled.schedule(args)
             for t in build_tasks(item["games"], item["seed"], 16)]
    assert len(seeds) == len(set(seeds)) == 6000


def test_soft_deadline_finishes_fixed_leg_and_does_not_extend_on_resume(tmp_path, monkeypatch):
    args, output = setup_gate(tmp_path, monkeypatch)
    args.budget_min = 1
    clock, calls = [0.0], []
    monkeypatch.setattr(gate_sampled.time, "monotonic", lambda: clock[0])
    play = fake_match(calls)
    def slow(*a, **kw):
        result = play(*a, **kw)
        clock[0] += 100
        return result
    monkeypatch.setattr(gate_sampled, "run_match", slow)
    report = gate_sampled.run_gate(args)
    assert report["complete"] and report["verdict"] == "INCONCLUSIVE"
    assert len(calls) == 4  # whole par, even though it exceeded the soft budget
    assert not (output / "vs_bar.jsonl").exists()
    args.run_dir, args.resume = None, str(output)
    assert gate_sampled.run_gate(args) == report
    assert len(calls) == 4


def test_report_summary_tampering_does_not_pass_with_unchanged_log_hashes(tmp_path, monkeypatch):
    import copy
    args, _ = setup_gate(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_sampled, "run_match", fake_match([]))
    report = gate_sampled.run_gate(args)
    assert gate_sampled.validate_report(report, args.model, args.bar_model, 8, 2, 4) == "PASS"
    changed = copy.deepcopy(report)
    changed["legs"]["vs_bar"]["sampled"]["score"] = .99
    with pytest.raises(ValueError, match="arithmetic"):
        gate_sampled.validate_report(changed, args.model, args.bar_model, 8, 2, 4)
