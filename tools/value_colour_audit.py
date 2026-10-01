"""Is a value head pessimistic about White? Score its evaluations against real outcomes.

Owner request, October 1, 2026: a diagnostic before any retraining. The
hypothesis (GEN52/POOLCAP/LARGE results) is that v29's own self-play outcomes
taught it that White is worse than it is, and that this White pessimism passes
to every student.

Ground truth here is the ~9,000 rating games (round robin, strength ladder,
Arm L depth ladder), played by 16+ different models with whole trajectories
saved. For every position after the 16 sampled opening plies, each audited
model predicts the value (raw network output, White's perspective) and is
compared with the target it was trained to predict: the game's result,
discounted exactly as training discounts it (result x 0.5^(min(plies_to_end,
60)/60), the ramp every current corpus uses).

The headline number is the **signed bias** = mean(target - prediction) from
White's perspective. Positive means White does better than the model expects
(White pessimism). Intervals come from a bootstrap over games, since positions
within a game are not independent. The "without v29" subset drops every game
the v29 network played in, so v29's own play cannot shape the outcomes.

    py -3 tools/value_colour_audit.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

SOURCES = ["benchmarks/elo_rr_20260929", "benchmarks/elo_ladder_20260930",
           "benchmarks/elo_depth_scaling_20261001_armL"]
MODELS = {
    "v29": "models/bootstrap_v29/best_value_net.pt",
    "v28": "models/bootstrap_v28/best_value_net.pt",
    "gen49": "models/candidates/bootstrap_main_gen_0049/arena_selected.pt",
    "gen52B": "models/candidates/bootstrap_main_gen_0052_pool/arena_selected.pt",
    "gen52L": "models/candidates/bootstrap_main_gen_0052_large/arena_selected.pt",
    "B2": "models/candidates/b2_seed9053_state_cnn/selected_epoch_008.pt",
}
OPENING_PLIES = 16
FLOOR, HORIZON = 0.5, 60
PHASES = [(0, 10), (10, 30), (30, 60), (60, 10_000)]
OUT = os.path.join(ROOT, "benchmarks", "value_colour_audit_20261001")


def games():
    """Yield (players, white_name, positions, white_score) for every saved game."""
    for source in SOURCES:
        for log in sorted(glob.glob(os.path.join(ROOT, source, "pairings", "*.jsonl"))):
            report = log[:-1]  # foo.jsonl -> foo.json
            if not os.path.exists(report):
                continue
            with open(report, encoding="utf-8") as fh:
                s = json.load(fh)["summary"]
            with open(log, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    traj = row["game"]["trajectory"]
                    white = s["a"] if row["a_is_white"] else s["b"]
                    yield (s["a"], s["b"]), white, traj, float(row["white_score"]), int(row["plies"])


def positions():
    rows = []
    for gi, (players, white, traj, white_score, plies) in enumerate(games()):
        z = 2.0 * white_score - 1.0
        v29_played = any(p == "v29" or p.startswith("v29@") for p in players)
        for p in traj:
            ply = int(p["plies_reached"])
            if ply < OPENING_PLIES:
                continue
            board = p["fen"].split()[0]
            if "K" not in board or "k" not in board:
                continue
            ptd = plies - ply
            target = z * (FLOOR ** (min(ptd, HORIZON) / HORIZON))
            rows.append(dict(game=gi, fen=p["fen"], half=bool(p.get("half")), turn_count=int(p["turn_count"]),
                             white_to_move=p["fen"].split()[1] == "w", ptd=ptd, z=z, target=target,
                             v29_played=v29_played, white=white))
    return rows


def predict(model_path, rows, batch=4096):
    import torch
    from data_processor import fen_to_tensor
    from train import load_model_for_inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model_for_inference(os.path.join(ROOT, model_path), device)
    model.eval()
    channels = int(model.input_channels)
    out = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        x = np.stack([fen_to_tensor(r["fen"], is_white_turn=r["white_to_move"], half_pending=r["half"],
                                    input_channels=channels,
                                    **({"turn_count": r["turn_count"]} if channels == 24 else {}))
                      for r in chunk]).transpose(0, 3, 1, 2)
        with torch.no_grad():
            value, _policy = model(torch.from_numpy(np.ascontiguousarray(x)).to(device))
        side = value[:, 0].float().cpu().numpy()
        stm = np.array([1.0 if r["white_to_move"] else -1.0 for r in chunk], dtype=np.float32)
        out[start:start + len(chunk)] = side * stm   # side-to-move -> White's perspective
    return out


def bias_with_ci(game_ids, residual, reps=1000, seed=20261001):
    """Mean residual with a bootstrap over games (positions within a game are correlated)."""
    games_u, inverse = np.unique(game_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=residual)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(games_u), len(games_u))
        draws.append(sums[pick].sum() / counts[pick].sum())
    return float(sums.sum() / counts.sum()), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def summarize(rows, pred, mask):
    g = np.array([r["game"] for r in rows])[mask]
    target = np.array([r["target"] for r in rows])[mask]
    p = pred[mask]
    out = dict(positions=int(mask.sum()), games=int(len(np.unique(g))),
               mean_target=float(target.mean()), mean_prediction=float(p.mean()))
    out["bias"], out["bias_ci95"] = bias_with_ci(g, target - p)
    out["mse"] = float(((target - p) ** 2).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    rows = positions()
    print(f"{len(rows):,} positions from {len({r['game'] for r in rows}):,} games", flush=True)
    stm = np.array([r["white_to_move"] for r in rows])
    ptd = np.array([r["ptd"] for r in rows])
    no_v29 = ~np.array([r["v29_played"] for r in rows])
    subsets = {"all": np.ones(len(rows), bool), "without_v29_games": no_v29,
               "white_to_move": stm, "black_to_move": ~stm}
    for lo, hi in PHASES:
        subsets[f"plies_to_end_{lo}_{hi}"] = (ptd >= lo) & (ptd < hi)
    report = dict(sources=SOURCES, opening_plies=OPENING_PLIES, floor=FLOOR, horizon=HORIZON, models={})
    for name in args.models.split(","):
        pred = predict(MODELS[name], rows)
        report["models"][name] = {k: summarize(rows, pred, m) for k, m in subsets.items()}
        # Calibration: predicted-value bins against the realised target.
        bins = np.linspace(-1, 1, 11)
        idx = np.clip(np.digitize(pred, bins) - 1, 0, 9)
        target = np.array([r["target"] for r in rows])
        report["models"][name]["calibration"] = [
            dict(bin=[float(bins[i]), float(bins[i + 1])], n=int((idx == i).sum()),
                 mean_prediction=float(pred[idx == i].mean()) if (idx == i).any() else None,
                 mean_target=float(target[idx == i].mean()) if (idx == i).any() else None)
            for i in range(10)]
        a = report["models"][name]
        print(f"{name:7} bias all {a['all']['bias']:+.4f} {np.round(a['all']['bias_ci95'], 4)} | "
              f"without v29 {a['without_v29_games']['bias']:+.4f} {np.round(a['without_v29_games']['bias_ci95'], 4)} | "
              f"W-to-move {a['white_to_move']['bias']:+.4f} B-to-move {a['black_to_move']['bias']:+.4f} | "
              f"mse {a['all']['mse']:.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"-> {OUT}/report.json")


if __name__ == "__main__":
    main()
