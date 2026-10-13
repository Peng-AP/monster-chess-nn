"""Held-out re-measure on the repetition-aware engine (owner, 2026-10-13: "Do c").

gen53 failed promotion only on the held-out mean, 88.3% against v29's 92.7% minus
1.0 pp, and both numbers were measured before repetition awareness (October 7).
This measures gen53 and v29 again against B2 and v27 with the campaign
diagnostics' settings: 160 games, 3,200 sims, 16 sampled opening plies, sampled
score (tools/free_play_audit.py). Resumable; one GPU job.

    py -3 -u tools/heldout_remeasure_20261013.py
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from free_play_audit import audit_match  # noqa: E402

OUT = ROOT / "benchmarks/heldout_remeasure_20261013"
SEED = 3_050_000_000   # one 1M block per match; 3.05e9-3.054e9 is unused
GAMES, SIMS = 160, 3200
CANDIDATES = {"gen53": "models/candidates/bootstrap_main_gen_0053/arena_selected.pt",
              "v29": "models/bootstrap_v29/best_value_net.pt"}
HELD_OUT = {"b2": "models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt",
            "v27": "models/bootstrap_v27/best_value_net.pt"}
OLD = {"gen53": dict(b2=None, v27=None, mean=0.883), "v29": dict(b2=0.9906, v27=0.8625, mean=0.927)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for i, (name, model) in enumerate((n, m) for n in CANDIDATES.items() for m in HELD_OUT.items()):
        (cand, path), (opp, opp_path) = name, model
        report, log = OUT / f"{cand}_vs_{opp}.json", OUT / f"{cand}_vs_{opp}.jsonl"
        seed = SEED + i * 1_000_000
        if not report.exists():
            subprocess.run([sys.executable, "-u", "tools/match.py", "--model-a", path, "--model-b", opp_path,
                            "--games", str(GAMES), "--sims", str(SIMS), "--sims-b", str(SIMS), "--engine", "native",
                            "--workers", "8", "--seed", str(seed), "--opening-temp-plies", "16",
                            "--game-log", str(log), "--report-path", str(report), "--resume"], check=True, cwd=ROOT)
        stats = audit_match(log, path, opp_path, GAMES, seed, SIMS)["diagnostics"]["sampled"]
        result.setdefault(cand, {})[opp] = dict(score=stats["score"], sides=stats["sides"])
        print(f"HELDOUT {cand} vs {opp}: {stats['score'] * 100:.1f}%", flush=True)
    for cand, r in result.items():
        r["mean"] = round((r["b2"]["score"] + r["v27"]["score"]) / 2, 4)
        r["old_engine"] = OLD[cand]
        print(f"HELDOUT {cand} mean {r['mean'] * 100:.1f}% (old engine {OLD[cand]['mean'] * 100:.1f}%)", flush=True)
    bar = result["v29"]["mean"] - 0.01
    result["bar_current_engine"] = round(bar, 4)
    result["gen53_passes_current_engine_bar"] = result["gen53"]["mean"] >= bar - 1e-12
    result["gen53_passes_old_bar"] = result["gen53"]["mean"] >= 0.917 - 1e-12
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(f"HELDOUT bar (v29 mean - 1.0 pp) {bar * 100:.1f}%; gen53 passes: {result['gen53_passes_current_engine_bar']}",
          flush=True)
    print("HELDOUT REMEASURE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
