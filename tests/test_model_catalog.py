import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_catalog import discover_model_choices


def checkpoint(root, name):
    path = root / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"network")
    return path


def test_sampled_gate_identity_and_latest_candidates(tmp_path):
    old = checkpoint(tmp_path, "candidates/bootstrap_main_gen_0042/screen_nominee.pt")
    new = checkpoint(tmp_path, "candidates/bootstrap_main_gen_0045/arena_selected.pt")
    report = tmp_path / "iterations/gen_0045/reports/binding_gate.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"model": {"path": str(new), "sha256": hashlib.sha256(b"network").hexdigest()},
                                 "bar": {"path": str(old)}, "verdict": "PASS", "confirmed": True,
                                 "legs": {"vs_bar": {"sampled": {"score": .79}}}}))
    choices = discover_model_choices(tmp_path)
    assert choices[1][1] == str(new)
    assert "PASS+confirmed" in choices[1][0] and "79.00%" in choices[1][0]
    assert str(old) in dict((p, label) for label, p in choices)
    new.write_bytes(b"different network")
    choices = discover_model_choices(tmp_path)
    assert not any("PASS" in label for label, _ in choices)
    assert any(p == str(new) for _, p in choices)


def test_v3_nominee_refresh_and_hidden_retired_models(tmp_path):
    old = checkpoint(tmp_path, "candidates/bootstrap_main_gen_0042/screen_nominee.pt")
    checkpoint(tmp_path, "archive/old/best_value_net.pt")
    checkpoint(tmp_path, "rejected/old/best_value_net.pt")
    checkpoint(tmp_path, "candidates/rehearsal_one/best_value_net.pt")
    assert discover_model_choices(tmp_path)[1][1] == str(old)
    new = checkpoint(tmp_path, "candidates/bootstrap_main_gen_0044/screen_nominee_v3.pt")
    choices = discover_model_choices(tmp_path)
    assert choices[1][1] == str(new)
    assert len(choices) == 3


def test_legacy_gate_and_inconclusive_label(tmp_path):
    path = checkpoint(tmp_path, "candidates/old/selected_epoch_007.pt")
    report = tmp_path / "benchmarks/gate_old.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"model": str(path), "bar": "vs_v24", "verdict": "INCONCLUSIVE",
                                 "legs": {"vs_v24": {"a_score": .6}}}))
    label, value = discover_model_choices(tmp_path)[1]
    assert value == str(path) and "INCONCLUSIVE" in label and "60.00%" in label
    (report.parent / "gate_broken.json").write_text("{")
    assert len(discover_model_choices(tmp_path)) == 2
