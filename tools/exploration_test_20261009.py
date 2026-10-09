"""Does training exploration skew the data against White? (owner, 2026-10-09)

Every teacher's training self-play (1,600 simulations, temperature 1.0 for the
first 30 half-moves, then 0.1) ends 69-78% in Black wins, while match self-play
(0.5 for 16, then 0) is far more balanced. Owner: "there's probably only a few
white moves that 'work'. Exploration probably hurts it."

gen55 plays 300 training-style self-play games at each exploration length --
30 (current), 16, 8, 0 half-moves -- through the same generator
(tools/stateful_generation.py), each arm in its own directory. The report gives
colour outcomes and opening diversity (distinct 8-half-move openings), because
less exploration also means less varied training data.

Evidence: benchmarks/exploration_test_20261009/. Produces no model.
"""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/exploration_test_20261009"
MODEL = "models/candidates/bootstrap_main_gen_0055/arena_selected.pt"
ARMS = (30, 16, 8, 0)
GAMES = 300
SEED = 2_700_000_000


def recipe(plies, i):
    return dict(model=MODEL, free_games=GAMES, fresh_games=0, league_games=0, fork_games=0,
                sims=1600, fork_sims=1600, workers=8, seed=SEED + i * 1_000_000,
                coverage_reanalysis=False, temperature_plies=plies, prefix_models=[], opponents=[])


def tally(raw):
    results, openings, plies = collections.Counter(), collections.Counter(), []
    for f in sorted((raw / "selfplay").glob("*.jsonl")):
        last = None
        with open(f, "rb") as fh:
            for line in fh:
                last = line
        r = json.loads(last)
        moves = r["state"]["moves"] + [r["played_action"]]
        results[r["game_result"]] += 1
        openings[" ".join(moves[:8])] += 1
        plies.append(len(moves))
    n = sum(results.values())
    white, black = results.get(1, 0), results.get(-1, 0)
    return dict(games=n, white_wins=white, black_wins=black, draws=n - white - black,
                white_score=round((white + 0.5 * (n - white - black)) / n, 4),
                distinct_openings_8=len(openings), top_opening_share=round(openings.most_common(1)[0][1] / n, 4),
                mean_plies=round(sum(plies) / n, 1))


def main():
    os.chdir(ROOT)
    report = {}
    for i, plies in enumerate(ARMS):
        arm = OUT / f"plies_{plies}"
        arm.mkdir(parents=True, exist_ok=True)
        config, summary = arm / "recipe.json", arm / "summary.json"
        config.write_text(json.dumps(recipe(plies, i), indent=2))
        if not summary.exists():
            print(f"EXPLORATION TEST plies={plies}", flush=True)
            subprocess.run([sys.executable, "-u", "tools/stateful_generation.py", "--config", str(config),
                            "--raw", str(arm / "raw"), "--summary", str(summary)], check=True)
        report[plies] = tally(arm / "raw")
        print(f"EXPLORATION RESULT plies={plies}: {report[plies]}", flush=True)
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("EXPLORATION TEST COMPLETE", flush=True)


if __name__ == "__main__":
    main()
