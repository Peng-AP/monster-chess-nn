"""Contracts for scoring, evidence independence, and interrupted match recovery."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from free_gate_stats import leg_stats, opening_key, summary, verdict
from gate_free import cache_key
from match import build_tasks
from match_evidence import MatchJournal, task_id, read_rows


def row(i, white=True, result=1, complete=True):
    return {"a_is_white": white, "result_for_a": result, "plies": 20,
            "opening": {"fen": f"position-{i}", "half": False,
                        "turn_count": 10, "complete": complete}}


def test_caps_are_draws_and_color_means_are_equally_weighted():
    rows = [row(0, result=.5), row(1, result=-.5)]
    assert summary(rows)["sides"]["white"]["score"] == .5
    assert summary(rows)["sides"]["white"]["se"] == 0
    rows = [row(i) for i in range(9)] + [row(10, False, -1)]
    assert summary(rows)["score"] == .5  # pooled .9 was wrong


def test_unseen_confirmation_and_incomplete_openings():
    first = [row(0), row(1, False)]
    confirm = first + [row(2, complete=False), row(3, False)]
    stats = leg_stats(confirm, {opening_key(r) for r in first})
    assert stats["overlap"] == 2
    assert stats["novel"]["n"] == 2
    assert stats["incomplete_openings"] == 1
    assert stats["unique"]["n"] == 4


def test_missing_color_or_novel_coverage_cannot_pass():
    par = leg_stats([row(i, aw, 0) for i in range(2) for aw in (True, False)])
    wins = leg_stats([row(i, aw) for i in range(2) for aw in (True, False)])
    assert verdict(par, {"vs_bar": wins, "vs_bar_confirm": wins}, 2, 2)["eligible"]
    overlapping = leg_stats([row(0), row(0, False)], {opening_key(row(0))})
    out = verdict(par, {"vs_bar": wins, "vs_bar_confirm": overlapping}, 1, 2)
    assert out["verdict"] == "INCONCLUSIVE"
    assert not out["eligible"]
    missing = leg_stats([row(i) for i in range(200)])
    assert verdict(par, {"vs_bar": missing}, 2, 2)["verdict"] == "INCONCLUSIVE"


def test_adequately_covered_failure_and_conflicting_endpoint():
    par = leg_stats([row(i, aw, 0) for i in range(2) for aw in (True, False)])
    losses = leg_stats([row(i, aw, -1) for i in range(2) for aw in (True, False)])
    assert verdict(par, {"vs_bar": losses, "vs_bar_confirm": losses}, 2, 2)["verdict"] == "FAIL"
    conflict = leg_stats([row(0), row(0, result=-1), row(0, False)])
    assert conflict["endpoint_outcome_conflicts"] == 1
    assert verdict(par, {"vs_bar": conflict, "vs_bar_confirm": losses}, 1, 2)["verdict"] == "INCONCLUSIVE"


def test_cache_keys_include_model_rules_and_search():
    config = {"runtime": {"repetition": 3}, "sims": 3200, "version": "v2", "seed": 5}
    key = cache_key({"sha256": "a"}, config)
    assert key != cache_key({"sha256": "b"}, config)
    assert key != cache_key({"sha256": "a"}, dict(config, sims=1600))
    assert key != cache_key({"sha256": "a"}, dict(config, runtime={"repetition": 4}))
    assert key == cache_key({"sha256": "a"}, dict(config, seed=6000))


def test_journal_resume_retains_completed_tasks_and_archives_torn_tail(tmp_path):
    tasks = build_tasks(4, 100, 16)
    path = tmp_path / "games.jsonl"
    journal = MatchJournal(path, {"model_hash": "a"}, tasks)
    completed = dict(row(0), task_id=task_id(tasks[0]), seed=100, pair=None)
    journal.append(completed)
    with path.open("ab") as stream:
        stream.write(b'{"task_id":')
    restored = MatchJournal(path, {"model_hash": "a"}, tasks, resume=True)
    assert restored.pending == tasks[1:]
    assert read_rows(path) == [completed]
    assert list(tmp_path.glob("*.torn-*"))
    with pytest.raises(ValueError, match="duplicate"):
        restored.append(completed)
    with pytest.raises(ValueError, match="provenance"):
        MatchJournal(path, {"model_hash": "b"}, tasks, resume=True)
    with pytest.raises(FileExistsError):
        MatchJournal(path, {"model_hash": "a"}, tasks)


def test_malformed_complete_record_fails_loudly(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{oops}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_rows(path, allow_partial_tail=True)


def test_match_interruption_and_resume_schedules_only_missing_tasks(tmp_path, monkeypatch):
    import match
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture checkpoint")
    log = tmp_path / "games.jsonl"
    calls = []
    interrupt = [True]

    class Iterator:
        def __init__(self, tasks):
            self.tasks = iter(tasks)
            self.count = 0

        def next(self, timeout):
            if interrupt[0] and self.count == 1:
                raise match.mp.TimeoutError()
            task = next(self.tasks)
            self.count += 1
            return (1, 20, task[0], task[4], row(task[1])["opening"],
                    {"task_id": task_id(task), "seed": task[1], "game": {}})

    class Pool:
        def __init__(self, *args, **kwargs):
            pass
        def imap_unordered(self, fn, tasks):
            calls.append(list(tasks))
            return Iterator(tasks)
        def terminate(self):
            pass
        close = terminate
        join = terminate

    monkeypatch.setattr(match.mp, "Pool", Pool)
    with pytest.raises(TimeoutError):
        match.run_match(str(model), str(model), 4, 8, 100, game_log=str(log))
    assert len(read_rows(log)) == 1
    interrupt[0] = False
    result = match.run_match(str(model), str(model), 4, 8, 100,
                             game_log=str(log), resume=True)
    assert len(calls[1]) == 3
    assert len(read_rows(log)) == 4
    assert result["games"] == 4


@pytest.mark.local_artifacts
def test_pipeline_free_recipe_and_promotion_provenance(tmp_path, monkeypatch):
    import iterate
    import match_evidence
    from free_gate_stats import SCORING_VERSION
    args = iterate.build_parser().parse_args(["--gate-backend", "free"])
    assert args.anchor_data == "none" and args.replay_generations == 8
    assert args.book_seed_games == 400
    paths = iterate._paths_for_generation(tmp_path, 1)
    arch = iterate._checkpoint_spec(iterate.DEFAULT_CHAMPION)
    plan = iterate._command_plan(args, 1, iterate.DEFAULT_CHAMPION, arch, paths, [])
    assert plan["binding_gate"]["commands"][0][0] == "tools/gate_free.py"
    assert plan["high_fidelity_gate"]["skip_reason"]
    assert not any("anchor=" in part for part in plan["compose"]["commands"][0])

    monkeypatch.setattr(match_evidence, "runtime_identity", lambda: {})
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture")
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_bytes(b"evidence")
    args.free_gate_target_per_side = args.free_gate_par_per_side = 2
    par = leg_stats([row(i, aw, 0) for i in range(2) for aw in (True, False)])
    wins = leg_stats([row(i, aw, 1) for i in range(2) for aw in (True, False)])
    report = {"instrument": SCORING_VERSION, "eligible": True, "confirmed": True, "complete": True,
              "model": match_evidence.model_identity(model), "bar": match_evidence.model_identity(model),
              "protocol": {"sims": args.arena_sims, "target_per_side": 2, "par_per_side": 2, "runtime": {}},
              "bar_free_par": par, "legs": {"vs_bar": wins, "vs_bar_confirm": wins},
              "combined_h2h": wins,
              "evidence_hashes": {str(evidence): match_evidence.file_hash(evidence)}}
    assert iterate._binding_passed(report, args, model, model)
    report["complete"] = False
    assert not iterate._binding_passed(report, args, model, model)
    report["complete"] = True
    model.write_bytes(b"different checkpoint")
    assert not iterate._binding_passed(report, args, model, model)


def test_gate_resumes_partial_confirmation_without_counting_completed_games_twice(tmp_path, monkeypatch):
    import gate_free
    candidate, bar = tmp_path / "candidate.pt", tmp_path / "bar.pt"
    candidate.write_bytes(b"candidate")
    bar.write_bytes(b"bar")
    args = gate_free.parser().parse_args([
        "--model", str(candidate), "--bar-model", str(bar),
        "--run-dir", str(tmp_path / "run"), "--target-per-side", "1",
        "--par-per-side", "1", "--batch-games", "2", "--budget-min", "1"])
    monkeypatch.setattr(gate_free, "ROOT", tmp_path)
    monkeypatch.setattr(gate_free, "runtime_identity", lambda: {})
    interrupted = [False]
    played = []

    def fake_match(model_a, model_b, **kwargs):
        tasks = build_tasks(kwargs["games"], kwargs["seed"], 16)
        journal = MatchJournal(kwargs["game_log"], {"a": model_a, "b": model_b}, tasks, resume=True)
        for task in journal.pending:
            journal.append(dict(row(task[1], task[0], 0 if model_a == model_b else 1),
                                task_id=task_id(task), seed=task[1], pair=None))
            played.append(task_id(task))
            if "confirm" in kwargs["game_log"] and not interrupted[0]:
                interrupted[0] = True
                raise InterruptedError("simulated stop after durable game")

    monkeypatch.setattr(gate_free, "run_match", fake_match)
    with pytest.raises(InterruptedError):
        gate_free.run_gate(args)
    args.resume = args.run_dir
    out = gate_free.run_gate(args)
    assert out["verdict"] == "PASS"
    assert len(played) == len(set(played)) == 6
    assert out["legs"]["vs_bar_confirm"]["novel"]["n"] == 2
    gate_free.run_gate(args)
    assert len(played) == 6


@pytest.mark.local_artifacts
def test_pipeline_pass_path_skips_redundant_gate_and_never_promotes_without_flag(tmp_path, monkeypatch):
    import iterate
    args = iterate.build_parser().parse_args(["--run-root", str(tmp_path),
                                             "--incumbent", str(iterate.DEFAULT_CHAMPION)])
    paths = iterate._paths_for_generation(tmp_path, 1)
    gate_report = tmp_path / "gate.json"
    gate_report.write_text('{"verdict":"PASS"}', encoding="utf-8")
    plan = {phase: {"commands": [], "outputs": []} for phase in iterate.PHASES}
    plan["binding_gate"]["outputs"] = [str(gate_report)]
    plan["offline_gate"]["outputs"] = [str(gate_report)]
    plan["high_fidelity_gate"]["skip_reason"] = "already confirmed by free gate"
    monkeypatch.setattr(iterate, "_command_plan", lambda *a: plan)
    monkeypatch.setattr(iterate, "_archive_incumbent", lambda *a: None)
    monkeypatch.setattr(iterate, "_validate_generation_summaries", lambda *a: {})
    monkeypatch.setattr(iterate, "_accept_generation_data", lambda *a: None)
    monkeypatch.setattr(iterate, "_processed_hashes", lambda *a: {"fixture": "hash"})
    monkeypatch.setattr(iterate, "_binding_passed", lambda *a: True)
    monkeypatch.setattr(iterate, "_promote", lambda *a: pytest.fail("unexpected real promotion"))
    assert iterate.run_generation(args, generation=1) == "passed_not_promoted"
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    assert state["phases"]["high_fidelity_gate"]["status"] == "skipped"
    assert state["phases"]["self_skew"]["status"] == "completed"


def test_resume_rejects_changed_generator_or_composed_replay(tmp_path):
    import iterate
    from match_evidence import file_hash
    model = tmp_path / "model.pt"
    model.write_bytes(b"original generator")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "splits.npz").write_bytes(b"pinned split membership")
    state = {"incumbent": str(model), "incumbent_sha256": file_hash(model),
             "composed_data_sha256": iterate._processed_hashes(corpus)}
    paths = {"replay_processed": corpus}
    iterate._validate_resume_evidence(state, paths, tmp_path)
    (corpus / "splits.npz").write_bytes(b"modified splits")
    with pytest.raises(RuntimeError, match="composed replay"):
        iterate._validate_resume_evidence(state, paths, tmp_path)
    model.write_bytes(b"different generator")
    with pytest.raises(RuntimeError, match="generating checkpoint"):
        iterate._validate_resume_evidence(state, paths, tmp_path)
