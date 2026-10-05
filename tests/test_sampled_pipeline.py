"""New sampled defaults, isolated seed stages, and fail-closed promotion evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import iterate
import gate_sampled
from match import build_tasks
from checkpoint_screen import final_stage_seed
from test_sampled_gate import setup_gate, fake_match


def option(command, name):
    return command[command.index(name) + 1]


@pytest.mark.local_artifacts
def test_sampled_recipe_changes_measurement_not_learning(tmp_path):
    args = iterate.build_parser().parse_args([])
    assert args.gate_backend == "sampled"
    assert not args.promote_on_pass
    paths = iterate._paths_for_generation(tmp_path, 45)
    arch = iterate._checkpoint_spec(iterate.DEFAULT_CHAMPION)
    plan = iterate._command_plan(args, 45, iterate.DEFAULT_CHAMPION, arch, paths, [])
    binding = plan["binding_gate"]["commands"][0]
    screen = plan["checkpoint_screen"]["commands"][0]
    reanalysis = plan["reanalyze"]["commands"][0]
    assert binding[0] == "tools/gate_sampled.py"
    assert option(binding, "--target-per-side") == "200"
    assert option(binding, "--par-games") == "400"
    assert option(screen, "--probe-sims") == "1600"
    assert option(screen, "--sims") == "3200"
    assert option(screen, "--finalists") == "2"
    assert "--resume" in reanalysis
    assert not Path(option(reanalysis, "--journal")).is_relative_to(
        Path(option(reanalysis, "--source-dir")))
    assert plan["high_fidelity_gate"]["skip_reason"]
    assert "--resume-from" not in plan["train"]["commands"][0]
    assert args.anchor_data == "none" and args.replay_generations == 8
    assert args.games == 1000 and args.book_seed_games == 400 and args.sims == 700
    assert args.lr == .002 and args.epochs == 30 and args.patience == 10


@pytest.mark.local_artifacts
def test_sampled_pipeline_seed_namespaces_are_disjoint(tmp_path):
    args = iterate.build_parser().parse_args([])
    arch = iterate._checkpoint_spec(iterate.DEFAULT_CHAMPION)
    all_seeds = set()
    for generation in (44, 45, 46):
        paths = iterate._paths_for_generation(tmp_path, generation)
        plan = iterate._command_plan(args, generation, iterate.DEFAULT_CHAMPION, arch, paths, [])
        screen = plan["checkpoint_screen"]["commands"][0]
        gate = plan["binding_gate"]["commands"][0]
        skew = plan["self_skew"]["commands"][0]
        probe_seed = int(option(screen, "--seed"))
        gate_seed = int(option(gate, "--seed"))
        seeds = (probe_seed, final_stage_seed(probe_seed), gate_seed,
                 gate_seed + 100_000, gate_seed + 200_000, int(option(skew, "--seed")))
        for seed in seeds:
            stage = {task[1] for task in build_tasks(2000, seed, 16)}
            assert not all_seeds & stage
            all_seeds |= stage


def test_pipeline_recomputes_sampled_evidence_and_fails_closed(tmp_path, monkeypatch):
    gate_args, _ = setup_gate(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_sampled, "run_match", fake_match([]))
    report = gate_sampled.run_gate(gate_args)
    args = iterate.build_parser().parse_args([
        "--arena-sims", "8", "--sampled-gate-target-per-side", "2",
        "--sampled-gate-par-games", "4"])
    assert iterate._binding_verdict(report, args, gate_args.model, gate_args.bar_model) == "PASS"
    changed = copy.deepcopy(report)
    changed["legs"]["vs_bar"]["sampled"]["score"] = .99
    assert iterate._binding_verdict(changed, args, gate_args.model, gate_args.bar_model) == "INCONCLUSIVE"
    assert iterate._binding_verdict({"verdict": "PASS"}, args, gate_args.model, gate_args.bar_model) == "INCONCLUSIVE"
    Path(gate_args.model).write_bytes(b"changed model")
    assert not iterate._binding_passed(report, args, gate_args.model, gate_args.bar_model)


def test_native_identity_matches_before_and_after_adapter_import():
    if not list((ROOT / "native").glob("monster_native.*")):
        pytest.skip("repository native extension unavailable")
    code = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from match_evidence import runtime_identity
before = runtime_identity()
import native_mcts
after = runtime_identity()
assert before == after, 'fresh parent omitted the native binary'
assert str(Path(native_mcts.mn.__file__).resolve()) in before['files']
print(json.dumps({'native': native_mcts.mn.__file__, 'stable': True}))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["stable"]


def test_sampled_promotion_refuses_unverified_pass_before_writing_pointer(tmp_path, monkeypatch):
    args = iterate.build_parser().parse_args([])
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "binding_gate.json").write_text('{"verdict":"PASS","eligible":true}')
    pointer = tmp_path / "champion.json"
    monkeypatch.setattr(iterate, "CHAMPION_POINTER", pointer)
    monkeypatch.setattr(iterate, "CHAMPIONS_DIR", tmp_path / "champions")
    state = {"config": vars(args), "incumbent": str(candidate), "generation": 45}
    with pytest.raises(RuntimeError, match="complete compatible"):
        iterate._promote(state, {"candidate": candidate, "reports": reports})
    assert not pointer.exists()
    assert not (tmp_path / "champions").exists()
