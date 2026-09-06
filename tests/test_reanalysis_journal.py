import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import reanalysis_journal as journal_module
from reanalysis_journal import ReanalysisJournal
from reanalyze import record_identity


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    source.mkdir()
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    records = [{"fen": f"position-{i}", "current_player": "black", "half": 0,
                "game_result": -1, "plies_to_end": 10, "policy": {"a1a2": 1.0}}
               for i in range(3)]
    (source / "game.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    rows = [{"path": "game.jsonl", "line": i + 1, "record": r} for i, r in enumerate(records)]
    args = argparse.Namespace(source_dir=str(source), model=str(model), sample=3, black_fraction=.6,
                              simulations=8, engine="native", batch_size=None, seed=10)
    monkeypatch.setattr(journal_module, "runtime_identity", lambda: {"engine": "test"})
    return args, rows, tmp_path / "results.jsonl"


def result(item):
    return {"identity": record_identity(item), "source_path": item["path"], "source_line": item["line"],
            "fen": item["record"]["fen"], "current_player": "black", "half": 0,
            "game_result": -1, "plies_to_end": 10, "deep_policy": {"a1a2": 1.0},
            "deep_value": .5, "metrics": {"priority": .1, "policy_js": .1,
                                           "value_delta": 0, "action_changed": False}}


def test_resume_preserves_every_completed_result_and_archives_torn_tail(tmp_path, monkeypatch):
    args, rows, path = fixture(tmp_path, monkeypatch)
    journal = ReanalysisJournal(path, args, rows, rows, record_identity, __file__)
    journal.append(result(rows[0]))
    with path.open("ab") as stream:
        stream.write(b'{"identity":')
    resumed = ReanalysisJournal(path, args, rows, rows, record_identity, __file__, resume=True)
    assert resumed.pending == rows[1:]
    assert resumed.results == [result(rows[0])]
    assert list(tmp_path.glob("*.torn-*"))
    with pytest.raises(ValueError, match="duplicate"):
        resumed.append(result(rows[0]))


def test_changed_inputs_or_model_fail_and_cache_cannot_enter_raw_tree(tmp_path, monkeypatch):
    args, rows, path = fixture(tmp_path, monkeypatch)
    ReanalysisJournal(path, args, rows, rows, record_identity, __file__)
    with pytest.raises(FileExistsError):
        ReanalysisJournal(path, args, rows, rows, record_identity, __file__)
    Path(args.model).write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        ReanalysisJournal(path, args, rows, rows, record_identity, __file__, resume=True)
    with pytest.raises(ValueError, match="outside"):
        ReanalysisJournal(Path(args.source_dir) / "cache.jsonl", args, rows, rows, record_identity, __file__)


def test_all_cached_results_can_publish_without_research_and_detect_modified_output(tmp_path, monkeypatch):
    from match_evidence import atomic_json
    args, rows, path = fixture(tmp_path, monkeypatch)
    journal = ReanalysisJournal(path, args, rows, rows, record_identity, __file__)
    for item in rows:
        journal.append(result(item))
    output = tmp_path / "teachers"
    output.mkdir()
    for i in range(2):
        (output / f"teacher_{i:05d}.jsonl").write_text('{}\n')
    (output / "reanalysis_summary.json").write_text('{}')
    atomic_json(output / "reanalysis_evidence.json", journal.output_manifest(output, 2))
    resumed = ReanalysisJournal(path, args, rows, rows, record_identity, __file__, resume=True)
    assert resumed.pending == []
    resumed.validate_output(output, 2)
    (output / "teacher_00000.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="changed"):
        resumed.validate_output(output, 2)


def test_invalid_result_cannot_become_durable_training_evidence(tmp_path, monkeypatch):
    args, rows, path = fixture(tmp_path, monkeypatch)
    journal = ReanalysisJournal(path, args, rows, rows, record_identity, __file__)
    bad = result(rows[0])
    bad["deep_value"] = float("nan")
    with pytest.raises(ValueError, match="numeric"):
        journal.append(bad)
    assert not path.exists()


def test_command_resumes_search_then_validates_complete_output_without_workers(tmp_path, monkeypatch):
    import concurrent.futures
    import reanalyze
    import data_generation
    import worker_lease
    args, rows, path = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(worker_lease, "DEFAULT_PATH", tmp_path / "workers.lock")
    output = tmp_path / "teachers"
    monkeypatch.setattr(sys, "argv", ["reanalyze", "--source-dir", args.source_dir,
        "--model", args.model, "--output-dir", str(output), "--sample", "3", "--keep", "2",
        "--simulations", "8", "--journal", str(path), "--resume"])
    calls, failed, pools = [], [False], []
    class Pool:
        def __init__(self, **kw):
            pools.append(self)
        def submit(self, fn, item):
            calls.append(item["line"])
            future = concurrent.futures.Future()
            future.line = item["line"]
            if item["line"] == 2 and not failed[0]:
                failed[0] = True
                future.set_exception(RuntimeError("interrupted search"))
            else:
                future.set_result(result(item))
            return future
        def shutdown(self, wait):
            pass
    def wait(pending, **kw):
        first = min(pending, key=lambda f: f.line)
        return {first}, pending - {first}
    monkeypatch.setattr(reanalyze.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(reanalyze.concurrent.futures, "wait", wait)
    monkeypatch.setattr(data_generation, "terminate_pool", lambda p: None)
    with pytest.raises(RuntimeError, match="interrupted"):
        reanalyze.main()
    reanalyze.main()
    assert calls == [1, 2, 3, 2, 3]
    assert len(pools) == 2
    assert len(list(output.glob("teacher_*.jsonl"))) == 2
    reanalyze.main()
    assert len(pools) == 2
    assert json.loads((output / "reanalysis_summary.json").read_text())["resumed_results"] == 1
