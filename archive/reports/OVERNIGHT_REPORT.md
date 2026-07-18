
# Overnight Phase 4 run — started 2026-07-03 02:53:09

[02:53:09] Waiting for hybrid generation to finish (data\raw\hybrid_v1\generation_summary.json)...
[12:30:20] Hybrid generation done: {"timestamp": "2026-07-03T12:29:53", "num_games_requested": 300, "saved_games": 284, "skipped_empty": 16, "failed_games": 0, "timed_out_games": 0, "total_positions": 19020, "results": {"white_wins": 153, "black_wins": 131, "draws": 0}, "simulations": {"configured_base": 800, "configured_min": 800, "configured_max": 800, "sampled_stats": {"min": 800, "max": 800, "mean": 800.0, "counts": {"800": 300
[12:30:28] Merged corpora: 968 heuristic + 284 hybrid -> data\raw\combined_v2
[12:30:28] START process: src/data_processor.py --raw-dir data\raw\combined_v2 --output-dir data\processed\combined_v2 --seed 42
[12:31:21] END process (exit=0, 0.9 min)
```
  Train position cap after val/test reservation: 479678 (val+test=20322)
  Train source quotas: selfplay=82698, human=0, blackfocus=0, humanseed=0
  Train source quotas (requested): selfplay=479678, human=0, blackfocus=0, humanseed=0
  Train source capacity estimate: selfplay=82698, human=0, blackfocus=0, humanseed=0
  WARNING: Low source diversity in train split (1 active source)

Saved to data\processed\combined_v2:
  positions.npy:    (103020, 8, 8, 15)
  mcts_values.npy:  (103020,)
  game_results.npy: (103020,)
  policies.npy:     (103020, 4096)
  target_lambdas.npy: (103020,)
  source_ids.npy:   (103020,)
  splits.npz:       train=82698, val=9756, test=10566
  split_game_ids.json: game-level split membership saved
```
[12:31:21] START train: src/train.py --data-dir data\processed\combined_v2 --model-dir models\fresh_start_v4 --target game_result --value-head wdl --epochs 30 --seed 42
[14:25:28] END train (exit=0, 114.1 min)
```
Epoch  18  train=2.1260 (v=0.0678 p=2.0152)  val=3.1286 (pow=0.7746 mse=0.6181 p=2.3540 mae=0.4654)  lr=9.8e-04  wdl(train_ce=0.0860 val_ce=0.9266 val_acc=77.8%)
Early stopping at epoch 18

--- Test set evaluation ---
Total loss: 3.0827
Value power loss: 0.7523
Value true MSE:   0.6424
Policy CE:  2.3304
Value MAE:  0.5407
WDL CE:     1.0243
WDL Acc:    69.6%
Winner prediction accuracy (non-draw): 69.6%

Best model saved to models\fresh_start_v4\best_value_net.pt
Run metadata saved to models\fresh_start_v4\train_run_20260703_123125.json
```
[14:25:41] D2 diagnostic (white-perspective avg prediction by actual result):
  White-win: n=6244 avg=+0.679 bar=>+0.15 PASS
  Black-win: n=4322 avg=+0.235 bar=<-0.15 MISS
[14:25:41] START probe: src/data_generation.py --num-games 20 --simulations 800 --use-model models\fresh_start_v4\best_value_net.pt --seed 500 --output-dir data\raw\probe_nn_v4
[15:17:42] END probe (exit=0, 52.0 min)
```
  Game 7: 37 moves, White (1), 349.2s
  Game 8: 157 moves, White (0.5), 1438.9s
  Game 1: 146 moves, Black (-0.5), 2615.5s
  Game 5: 148 moves, Black (-0.5), 2539.0s
  Game 6: 143 moves, Black (-0.5), 2838.5s
  Game 9: 97 moves, White (1), 1047.2s
  Game 12: 10 moves, White (1), 61.4s

  WARNING: 9 game(s) timed out after 600s, killing workers

Done! 800 total positions across 11 saved games (attempted 20).
Timed out games: 9
Results - White: 8, Black: 3, Draw: 0 (saved games only)
Simulation usage stats: min=800 max=800 mean=800.00
Generation summary saved to data\raw\probe_nn_v4\generation_summary.json
```
[15:17:42] PROBE pure-NN self-play: games=11 black_wins=3 mean_recorded_plies=72.7 -> PASS (criteria: >=1 Black win, mean length > 15)
[15:17:42] START benchmark: src/benchmark.py --model models\fresh_start_v4\best_value_net.pt --games 20 --sims 400 --seed 20260703
[16:44:15] END benchmark (exit=0, 86.6 min)
```
  "candidate_sims": 400,
  "anchor_sims": 400,
  "seed": 20260703,
  "start_fen": null,
  "candidate_wins": 5,
  "candidate_draws": 0,
  "candidate_losses": 15,
  "candidate_score": 0.25,
  "candidate_black_win_share": 0.1,
  "mean_game_plies": 179.25,
  "sec_per_decision": 1.4478,
  "elapsed_sec": 5190.5,
  "timestamp": "2026-07-03T16:44:14"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260703_164414.json
```
[16:44:15] ALL STEPS COMPLETE

# Overnight corrected-rules v6 run — started 2026-07-04 04:45:00

[04:45:00] START verify-seed7: src/verify_scripted_mate.py --games 16 --white-sims 200 --max-turns 150 --seed 7
[07:35:23] END verify-seed7 (exit=0, 170.4 min)
```
Game  8: BLACK WIN            turns= 26  start=3k2r1/2q4r/8/8/8/8/8/7K w - - 0 1
Game  9: BLACK WIN            turns= 18  start=2q1rrk1/8/8/8/8/8/2K5/8 w - - 0 1
Game 10: BLACK WIN            turns= 14  start=1k6/2qr3r/8/8/8/8/8/3K4 w - - 0 1
Game 11: BLACK WIN            turns= 24  start=kr4q1/5r2/8/8/8/8/4K3/8 w - - 0 1
Game 12: BLACK WIN            turns= 34  start=3k4/1rq2r2/8/8/8/8/8/6K1 w - - 0 1
Game 13: BLACK WIN            turns= 68  start=4rk2/1q4r1/8/8/8/8/8/2K5 w - - 0 1
Game 14: FAIL (result=1)      turns=  9  start=1k5q/2r1r3/8/8/8/8/5K2/8 w - - 0 1
Game 15: BLACK WIN            turns= 14  start=r6k/q4r2/8/8/8/8/5K2/8 w - - 0 1

15/16 Black wins
  FAILED start: 1k5q/2r1r3/8/8/8/8/5K2/8 w - - 0 1
         final: 3K4/4r3/8/8/8/8/8/8 b - - 0 5 (result=1, turns=9)
```
[07:35:23] START verify-seed99: src/verify_scripted_mate.py --games 16 --white-sims 200 --max-turns 150 --seed 99

# Overnight corrected-rules v6 run — started 2026-07-04 14:24:02

[14:24:02] Wrote 300 fork-safe demo start FENs to data\start_fens\mate_demo_starts_v2.jsonl
[14:24:02] START gen-heuristic: src/data_generation.py --num-games 800 --simulations 400 --curriculum --curriculum-live-results --seed 601 --output-dir data\raw\heuristic_v2
[14:38:16] END gen-heuristic (exit=0, 14.2 min)
```
  Game 633: 157 moves, Black (-1), 158.0s
  Game 798: 28 moves, White (1), 7.6s
  Game 785: 47 moves, White (1), 16.9s
  Game 747: 112 moves, White (1), 49.4s
  Game 783: 153 moves, Black (-1), 63.1s
  Game 570: 204 moves, Black (-0.5), 277.7s

Done! 16851 total positions across 780 saved games (attempted 800).
Skipped empty games: 20
Results - White: 466, Black: 314, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\heuristic_v2\generation_summary.json
```
[14:38:16] START gen-hybrid: src/data_generation.py --num-games 150 --simulations 600 --curriculum --curriculum-live-results --use-model models\fresh_start_v5\best_value_net.pt --hybrid-eval --seed 602 --output-dir data\raw\hybrid_v2
[15:47:11] END gen-hybrid (exit=0, 68.9 min)
```
  Game 120: 144 moves, Black (-1), 986.9s
  Game 148: 11 moves, White (1), 17.9s
  Game 147: 56 moves, White (1), 157.0s
  Game 149: 32 moves, Black (-1), 50.7s
  Game 141: 129 moves, Black (-1), 472.3s
  Game 144: 152 moves, Black (-1), 590.2s

Done! 4051 total positions across 149 saved games (attempted 150).
Skipped empty games: 1
Results - White: 89, Black: 60, Draw: 0 (saved games only)
Simulation usage stats: min=600 max=600 mean=600.00
Generation summary saved to data\raw\hybrid_v2\generation_summary.json
```
[15:47:11] START gen-demos: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\mate_demo_starts_v2.jsonl --record-all-plies --seed 603 --output-dir data\raw\mate_demos_v2
[16:09:02] END gen-demos (exit=0, 21.8 min)
```
  Game 284: 72 moves, Black (-1), 80.6s
  Game 265: 147 moves, Black (-1), 147.5s
  Game 288: 66 moves, Black (-1), 75.6s
  Game 276: 132 moves, Black (-1), 116.1s
  Game 297: 78 moves, Black (-1), 50.4s
  Game 298: 18 moves, Black (-1), 40.2s
  Game 283: 42 moves, Black (-1), 114.4s

Done! 15033 total positions across 300 saved games (attempted 300).
Results - White: 3, Black: 297, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\mate_demos_v2\generation_summary.json
```
[16:09:12] Merged: heuristic=780 hybrid=149 demos=297 (dropped 3 non-wins) human=3 -> data\raw\combined_v4
[16:09:12] START process: src/data_processor.py --raw-dir data\raw\combined_v4 --output-dir data\processed\combined_v4 --seed 42
[16:09:55] END process (exit=0, 0.7 min)
```
  Train source capacity estimate: selfplay=55594, human=0, blackfocus=0, humanseed=0
  WARNING: Low source diversity in train split (1 active source)

Saved to data\processed\combined_v4:
  positions.npy:    (70944, 8, 8, 15)
  mcts_values.npy:  (70944,)
  game_results.npy: (70944,)
  policies.npy:     (70944, 4096)
  target_lambdas.npy: (70944,)
  source_ids.npy:   (70944,)
  splits.npz:       train=55594, val=7254, test=8096
  split_game_ids.json: game-level split membership saved
```
[16:09:55] START train: src/train.py --data-dir data\processed\combined_v4 --model-dir models\fresh_start_v6 --target game_result --value-head wdl --epochs 30 --seed 42
[18:26:20] END train (exit=0, 136.4 min)
```
--- Test set evaluation ---
Total loss: 2.9131
Value power loss: 0.4812
Value true MSE:   0.3812
Policy CE:  2.4319
Value MAE:  0.2868
WDL CE:     0.7083
WDL Acc:    88.6%
Winner prediction accuracy (non-draw): 88.6%

Best model saved to models\fresh_start_v6\best_value_net.pt
Run metadata saved to models\fresh_start_v6\train_run_20260704_160959.json
```
[18:26:37] D2 White-win: n=2998 avg=+0.721 PASS (bar +0.15)
[18:26:37] D2 Black-win: n=5098 avg=-0.797 PASS (bar -0.15)
[18:26:37] START benchmark-v6: src/benchmark.py --model models\fresh_start_v6\best_value_net.pt --games 20 --sims 400 --seed 20260704
[19:38:38] END benchmark-v6 (exit=0, 72.0 min)
```
  "start_fen": null,
  "candidate_wins": 13,
  "candidate_draws": 0,
  "candidate_losses": 7,
  "candidate_score": 0.65,
  "candidate_black_win_share": 0.7,
  "mean_game_plies": 145.2,
  "sec_per_decision": 1.4866,
  "elapsed_sec": 4317.0,
  "timestamp": "2026-07-04T19:38:38"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260704_193838.json
```
[19:38:38] START benchmark-v5-corrected: src/benchmark.py --model models\fresh_start_v5\best_value_net.pt --games 16 --sims 400 --seed 20260705
[20:13:38] END benchmark-v5-corrected (exit=0, 35.0 min)
```
  "start_fen": null,
  "candidate_wins": 10,
  "candidate_draws": 0,
  "candidate_losses": 6,
  "candidate_score": 0.625,
  "candidate_black_win_share": 0.25,
  "mean_game_plies": 98.0,
  "sec_per_decision": 1.3367,
  "elapsed_sec": 2096.0,
  "timestamp": "2026-07-04T20:13:37"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260704_201337.json
```
[20:13:38] START verify-seed7: src/verify_scripted_mate.py --games 16 --white-sims 200 --max-turns 150 --seed 7
[20:19:28] END verify-seed7 (exit=0, 5.8 min)
```
Game  8: BLACK WIN            turns= 26  start=3k2r1/2q4r/8/8/8/8/8/7K w - - 0 1
Game  9: BLACK WIN            turns= 18  start=2q1rrk1/8/8/8/8/8/2K5/8 w - - 0 1
Game 10: BLACK WIN            turns= 14  start=1k6/2qr3r/8/8/8/8/8/3K4 w - - 0 1
Game 11: BLACK WIN            turns= 26  start=kr4q1/5r2/8/8/8/8/4K3/8 w - - 0 1
Game 12: BLACK WIN            turns= 34  start=3k4/1rq2r2/8/8/8/8/8/6K1 w - - 0 1
Game 13: BLACK WIN            turns= 42  start=4rk2/1q4r1/8/8/8/8/8/2K5 w - - 0 1
Game 14: FAIL (result=1)      turns=  9  start=1k5q/2r1r3/8/8/8/8/5K2/8 w - - 0 1
Game 15: BLACK WIN            turns= 14  start=r6k/q4r2/8/8/8/8/5K2/8 w - - 0 1

15/16 Black wins
  FAILED start: 1k5q/2r1r3/8/8/8/8/5K2/8 w - - 0 1
         final: 4r3/2K5/8/8/8/8/8/8 b - - 0 5 (result=1, turns=9)
```
[20:19:28] START verify-seed99: src/verify_scripted_mate.py --games 16 --white-sims 200 --max-turns 150 --seed 99
[20:33:50] END verify-seed99 (exit=0, 14.4 min)
```
Game  6: BLACK WIN            turns= 70  start=k1q5/1r2r3/8/8/8/8/5K2/8 w - - 0 1
Game  7: BLACK WIN            turns= 30  start=1rkr1q2/8/8/8/8/8/2K5/8 w - - 0 1
Game  8: BLACK WIN            turns= 20  start=q2rk1r1/8/8/8/8/8/8/2K5 w - - 0 1
Game  9: BLACK WIN            turns= 26  start=r2r2k1/1q6/8/8/8/8/6K1/8 w - - 0 1
Game 10: BLACK WIN            turns= 14  start=1k4r1/3q3r/8/8/8/8/3K4/8 w - - 0 1
Game 11: BLACK WIN            turns= 76  start=2r1k1r1/7q/8/8/8/8/7K/8 w - - 0 1
Game 12: BLACK WIN            turns= 44  start=4rk2/3r1q2/8/8/8/8/3K4/8 w - - 0 1
Game 13: BLACK WIN            turns= 86  start=5k2/q2rr3/8/8/8/8/3K4/8 w - - 0 1
Game 14: BLACK WIN            turns= 36  start=1r2k2q/3r4/8/8/8/8/8/7K w - - 0 1
Game 15: BLACK WIN            turns= 38  start=kr2q3/4r3/8/8/8/8/8/1K6 w - - 0 1

16/16 Black wins
```
[20:33:50] ALL STEPS COMPLETE

# Eval-fix v7 run (heuristic-only, fix 1aac31b) — started 2026-07-05 01:46:06

[01:46:06] START gen-heuristic: src/data_generation.py --num-games 800 --simulations 400 --curriculum --curriculum-live-results --seed 701 --output-dir data\raw\heuristic_v3
[02:02:13] END gen-heuristic (exit=0, 16.1 min)
```
  Game 794: 41 moves, White (1), 14.7s
  Game 439: 197 moves, Black (-0.5), 420.3s
  Game 782: 71 moves, White (1), 32.1s
  Game 701: 109 moves, Black (-1), 99.7s
  Game 759: 144 moves, Black (-1), 113.5s
  Game 658: 203 moves, Black (-0.5), 219.8s

Done! 16385 total positions across 775 saved games (attempted 800).
Skipped empty games: 25
Results - White: 473, Black: 302, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\heuristic_v3\generation_summary.json
```
[02:02:13] START gen-demos: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\mate_demo_starts_v2.jsonl --record-all-plies --seed 703 --output-dir data\raw\mate_demos_v3
[02:26:56] END gen-demos (exit=0, 24.7 min)
```
  Game 297: 48 moves, Black (-1), 89.9s
  Game 298: 72 moves, Black (-1), 89.9s
  Game 294: 81 moves, Black (-1), 110.0s
  Game 230: 225 moves, Black (-0.5), 450.7s
  Game 293: 72 moves, Black (-1), 140.3s
  Game 271: 165 moves, Black (-1), 292.9s
  Game 291: 126 moves, Black (-1), 181.0s

Done! 15305 total positions across 300 saved games (attempted 300).
Results - White: 7, Black: 293, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\mate_demos_v3\generation_summary.json
```
[02:27:06] Merged: heuristic=775 demos=293 (dropped 7 non-wins) human=5 -> data\raw\combined_v5
[02:27:06] START process: src/data_processor.py --raw-dir data\raw\combined_v5 --output-dir data\processed\combined_v5 --seed 42
[02:27:43] END process (exit=0, 0.6 min)
```
  Train source capacity estimate: selfplay=48066, human=0, blackfocus=0, humanseed=0
  WARNING: Low source diversity in train split (1 active source)

Saved to data\processed\combined_v5:
  positions.npy:    (61510, 8, 8, 15)
  mcts_values.npy:  (61510,)
  game_results.npy: (61510,)
  policies.npy:     (61510, 4096)
  target_lambdas.npy: (61510,)
  source_ids.npy:   (61510,)
  splits.npz:       train=48066, val=6662, test=6782
  split_game_ids.json: game-level split membership saved
```
[02:27:43] START train: src/train.py --data-dir data\processed\combined_v5 --model-dir models\fresh_start_v7 --target game_result --value-head wdl --epochs 30 --seed 42
[03:52:02] END train (exit=0, 84.3 min)
```
--- Test set evaluation ---
Total loss: 2.9457
Value power loss: 0.4260
Value true MSE:   0.3363
Policy CE:  2.5197
Value MAE:  0.2507
WDL CE:     0.3633
WDL Acc:    89.9%
Winner prediction accuracy (non-draw): 89.9%

Best model saved to models\fresh_start_v7\best_value_net.pt
Run metadata saved to models\fresh_start_v7\train_run_20260705_022748.json
```
[03:52:12] D2 White-win: n=2118 avg=+0.895 PASS (bar +0.15)
[03:52:12] D2 Black-win: n=4664 avg=-0.736 PASS (bar -0.15)
[03:52:12] START benchmark-v7: src/benchmark.py --model models\fresh_start_v7\best_value_net.pt --games 20 --sims 400 --seed 20260704
[04:37:42] END benchmark-v7 (exit=0, 45.5 min)
```
  "start_fen": null,
  "candidate_wins": 2,
  "candidate_draws": 0,
  "candidate_losses": 18,
  "candidate_score": 0.1,
  "candidate_black_win_share": 0.0,
  "mean_game_plies": 143.6,
  "sec_per_decision": 0.9493,
  "elapsed_sec": 2726.3,
  "timestamp": "2026-07-05T04:37:41"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260705_043741.json
```
[04:37:42] ALL STEPS COMPLETE

# Black-focus v8 run (dense corpus + backward-chained starts) — started 2026-07-05 05:28:53

[05:28:53] START gen-heuristic: src/data_generation.py --num-games 800 --simulations 400 --curriculum --curriculum-live-results --record-all-plies --seed 801 --output-dir data\raw\heuristic_v4
[05:46:50] END gen-heuristic (exit=0, 18.0 min)
```
  Game 768: 120 moves, Black (-1), 79.0s
  Game 619: 225 moves, Black (-0.5), 284.0s
  Game 774: 137 moves, White (1), 83.5s
  Game 653: 225 moves, Black (-0.5), 280.1s
  Game 708: 225 moves, Black (-0.5), 220.6s
  Game 794: 225 moves, Black (-0.5), 151.0s
  Game 741: 225 moves, Black (-0.5), 241.7s

Done! 19644 total positions across 800 saved games (attempted 800).
Results - White: 460, Black: 340, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\heuristic_v4\generation_summary.json
```
[05:46:50] START make-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\heuristic_v4 --input-dir data\raw\mate_demos_v3 --offsets 6,12,20,30,45,60 --output data\start_fens\blackfocus_starts_v1.jsonl
[05:46:55] END make-bf-starts (exit=0, 0.1 min)
```
Wrote 1337 unique Black-to-move start FENs from 304 Black-won games to data\start_fens\blackfocus_starts_v1.jsonl
```
[05:46:55] START gen-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\blackfocus_starts_v1.jsonl --start-fen-side black --record-all-plies --seed 802 --output-dir data\raw\v8_blackfocus
[06:00:19] END gen-blackfocus (exit=0, 13.4 min)
```
  Game 298: 28 moves, Black (-1), 9.8s
  Game 233: 166 moves, Black (-1), 211.0s
  Game 259: 88 moves, Black (-1), 122.6s
  Game 286: 25 moves, Black (-1), 38.5s
  Game 288: 31 moves, Black (-1), 34.3s
  Game 284: 52 moves, Black (-1), 77.3s
  Game 246: 225 moves, Black (-0.5), 354.3s

Done! 9999 total positions across 300 saved games (attempted 300).
Results - White: 7, Black: 293, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\v8_blackfocus\generation_summary.json
```
[06:00:24] Merged: heuristic=800 demos=293 (dropped 7 non-wins) blackfocus=300 human=5 -> data\raw\combined_v6
[06:00:24] START process: src/data_processor.py --raw-dir data\raw\combined_v6 --output-dir data\processed\combined_v6 --seed 42
[06:01:18] END process (exit=0, 0.9 min)
```
  Train source quotas (requested): selfplay=268037, human=134018, blackfocus=80411, humanseed=0
  Train source capacity estimate: selfplay=53930, human=2192, blackfocus=32512, humanseed=0

Saved to data\processed\combined_v6:
  positions.npy:    (106168, 8, 8, 15)
  mcts_values.npy:  (106168,)
  game_results.npy: (106168,)
  policies.npy:     (106168, 4096)
  target_lambdas.npy: (106168,)
  source_ids.npy:   (106168,)
  splits.npz:       train=88634, val=8618, test=8916
  split_game_ids.json: game-level split membership saved
```
[06:01:18] START train: src/train.py --data-dir data\processed\combined_v6 --model-dir models\fresh_start_v8 --target game_result --value-head wdl --epochs 30 --seed 42
[16:03:34] RESUME v8 chain from train (laptop powered off mid-train)
[16:03:34] START train: src/train.py --data-dir data\processed\combined_v6 --model-dir models\fresh_start_v8 --target game_result --value-head wdl --epochs 30 --seed 42
[18:18:47] END train (exit=0, 135.2 min)
```
--- Test set evaluation ---
Total loss: 2.5213
Value power loss: 0.2423
Value true MSE:   0.1986
Policy CE:  2.2790
Value MAE:  0.1703
WDL CE:     0.1780
WDL Acc:    93.6%
Winner prediction accuracy (non-draw): 93.6%

Best model saved to models\fresh_start_v8\best_value_net.pt
Run metadata saved to models\fresh_start_v8\train_run_20260705_160338.json
```
[18:19:00] D2 White-win: n=2670 avg=+0.753 PASS (bar +0.15)
[18:19:00] D2 Black-win: n=6246 avg=-0.882 PASS (bar -0.15)
[18:19:00] START benchmark-v8: src/benchmark.py --model models\fresh_start_v8\best_value_net.pt --games 20 --sims 400 --seed 20260704
[19:03:49] END benchmark-v8 (exit=0, 44.8 min)
```
  "start_fen": null,
  "candidate_wins": 14,
  "candidate_draws": 0,
  "candidate_losses": 6,
  "candidate_score": 0.7,
  "candidate_black_win_share": 0.5,
  "mean_game_plies": 130.1,
  "sec_per_decision": 1.0321,
  "elapsed_sec": 2685.4,
  "timestamp": "2026-07-05T19:03:47"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260705_190347.json
```
[19:03:49] ALL STEPS COMPLETE (resumed)

# Hybrid v9 run (loop-repeatability gate, v8 as generator) — started 2026-07-07 01:10:08

[01:10:08] START gen-hybrid: src/data_generation.py --num-games 150 --simulations 600 --curriculum --curriculum-live-results --use-model models\fresh_start_v8\best_value_net.pt --hybrid-eval --record-all-plies --seed 901 --output-dir data\raw\hybrid_v3
[02:31:45] END gen-hybrid (exit=0, 81.6 min)
```
  Game 146: 1 moves, Black (-1), 0.0s
  Game 143: 42 moves, White (1), 101.3s
  Game 144: 41 moves, White (1), 128.5s
  Game 149: 21 moves, White (1), 42.9s
  Game 147: 48 moves, White (1), 116.5s
  Game 148: 92 moves, White (1), 266.5s
  Game 107: 225 moves, Black (-0.5), 1402.2s

Done! 4823 total positions across 150 saved games (attempted 150).
Results - White: 93, Black: 57, Draw: 0 (saved games only)
Simulation usage stats: min=600 max=600 mean=600.00
Generation summary saved to data\raw\hybrid_v3\generation_summary.json
```
[02:31:57] Merged: heuristic=800 hybrid=150 demos=293 (dropped 7 non-wins) blackfocus=300 human=7 -> data\raw\combined_v7
[02:31:57] START process: src/data_processor.py --raw-dir data\raw\combined_v7 --output-dir data\processed\combined_v7 --seed 42
[02:32:56] END process (exit=0, 1.0 min)
```
  Train source quotas (requested): selfplay=266361, human=133181, blackfocus=79908, humanseed=0
  Train source capacity estimate: selfplay=60698, human=2336, blackfocus=32224, humanseed=0

Saved to data\processed\combined_v7:
  positions.npy:    (115808, 8, 8, 15)
  mcts_values.npy:  (115808,)
  game_results.npy: (115808,)
  policies.npy:     (115808, 4096)
  target_lambdas.npy: (115808,)
  source_ids.npy:   (115808,)
  splits.npz:       train=95258, val=9602, test=10948
  split_game_ids.json: game-level split membership saved
```
[02:32:56] START train: src/train.py --data-dir data\processed\combined_v7 --model-dir models\fresh_start_v9 --target game_result --value-head wdl --epochs 30 --seed 42
[05:04:27] END train (exit=0, 151.5 min)
```
--- Test set evaluation ---
Total loss: 2.5930
Value power loss: 0.3808
Value true MSE:   0.3085
Policy CE:  2.2122
Value MAE:  0.2532
WDL CE:     0.2785
WDL Acc:    90.6%
Winner prediction accuracy (non-draw): 90.6%

Best model saved to models\fresh_start_v9\best_value_net.pt
Run metadata saved to models\fresh_start_v9\train_run_20260707_023300.json
```
[05:04:42] D2 White-win: n=3638 avg=+0.597 PASS (bar +0.15)
[05:04:42] D2 Black-win: n=7310 avg=-0.881 PASS (bar -0.15)
[05:04:42] START benchmark-v9: src/benchmark.py --model models\fresh_start_v9\best_value_net.pt --games 20 --sims 400 --seed 20260704
[05:37:01] END benchmark-v9 (exit=0, 32.3 min)
```
  "start_fen": null,
  "candidate_wins": 8,
  "candidate_draws": 0,
  "candidate_losses": 12,
  "candidate_score": 0.4,
  "candidate_black_win_share": 0.2,
  "mean_game_plies": 141.3,
  "sec_per_decision": 0.6853,
  "elapsed_sec": 1936.6,
  "timestamp": "2026-07-07T05:37:01"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260707_053701.json
```
[05:37:01] ALL STEPS COMPLETE

# Blackfocus v10 run (oracle-sourced curriculum, aggressive anchor) — started 2026-07-08 00:22:39

[00:22:39] START gen-demos: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\mate_demo_starts_v2.jsonl --record-all-plies --seed 1004 --output-dir data\raw\mate_demos_v4
[00:48:48] END gen-demos (exit=0, 26.2 min)
```
  Game 299: 33 moves, Black (-1), 42.6s
  Game 293: 72 moves, Black (-1), 73.0s
  Game 262: 119 moves, White (1), 326.7s
  Game 292: 135 moves, Black (-1), 138.9s
  Game 290: 83 moves, White (1), 179.0s
  Game 296: 96 moves, Black (-1), 165.9s
  Game 289: 170 moves, White (1), 214.2s

Done! 16035 total positions across 300 saved games (attempted 300).
Results - White: 15, Black: 285, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\mate_demos_v4\generation_summary.json
```
[00:48:48] START gen-heuristic: src/data_generation.py --num-games 800 --simulations 400 --curriculum --curriculum-live-results --record-all-plies --seed 1001 --output-dir data\raw\heuristic_v5
[01:09:13] END gen-heuristic (exit=0, 20.4 min)
```
  Game 641: 222 moves, Black (-1), 278.0s
  Game 696: 168 moves, Black (-1), 163.5s
  Game 791: 29 moves, White (1), 27.5s
  Game 662: 225 moves, Black (-0.5), 221.1s
  Game 704: 225 moves, Black (-0.5), 154.2s
  Game 593: 225 moves, Black (-0.5), 422.2s
  Game 754: 225 moves, Black (-0.5), 137.3s

Done! 22065 total positions across 800 saved games (attempted 800).
Results - White: 462, Black: 338, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\heuristic_v5\generation_summary.json
```
[01:09:13] START make-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\mate_demos_v4 --input-dir data\raw\heuristic_v5 --offsets 6,12,20,30,45,60,80 --output data\start_fens\blackfocus_starts_v2.jsonl
[01:09:19] END make-bf-starts (exit=0, 0.1 min)
```
Wrote 1357 unique Black-to-move start FENs from 297 Black-won games to data\start_fens\blackfocus_starts_v2.jsonl
```
[01:09:19] START gen-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\blackfocus_starts_v2.jsonl --start-fen-side black --record-all-plies --seed 1002 --output-dir data\raw\v10_blackfocus
[01:21:00] END gen-blackfocus (exit=0, 11.7 min)
```
  Game 290: 25 moves, Black (-1), 58.7s
  Game 185: 174 moves, White (1), 291.4s
  Game 249: 225 moves, Black (-0.5), 183.0s
  Game 297: 64 moves, Black (-1), 57.4s
  Game 289: 37 moves, Black (-1), 85.0s
  Game 293: 90 moves, White (1), 89.9s
  Game 281: 202 moves, Black (-1), 167.9s

Done! 9920 total positions across 300 saved games (attempted 300).
Results - White: 8, Black: 292, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\v10_blackfocus\generation_summary.json
```
[01:21:04] Merged: heuristic=800 demos=285 (dropped 15 non-wins) blackfocus=300 human=7 -> data\raw\combined_v8
[01:21:04] START process: src/data_processor.py --raw-dir data\raw\combined_v8 --output-dir data\processed\combined_v8 --seed 42
[01:21:50] END process (exit=0, 0.8 min)
```
Retention summary: kept_games=1104/1392, kept_positions=46533/46833
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=882, val=110, test=112
  Processing positions (augment=True)...

Saved to data\processed\combined_v8:
  positions.npy:    (93066, 8, 8, 15)
  mcts_values.npy:  (93066,)
  game_results.npy: (93066,)
  policies.npy:     (93066, 4096)
  splits.npz:       train=73862, val=8178, test=11026
  split_game_ids.json: game-level split membership saved
```
[01:21:50] START train: src/train.py --data-dir data\processed\combined_v8 --model-dir models\fresh_start_v10 --target game_result --value-head wdl --epochs 30 --seed 42
[03:14:22] END train (exit=0, 112.5 min)
```
--- Test set evaluation ---
Total loss: 2.4760
Value power loss: 0.2316
Value true MSE:   0.2103
Policy CE:  2.2444
Value MAE:  0.2480
WDL CE:     0.2847
WDL Acc:    93.8%
Winner prediction accuracy (non-draw): 93.8%

Best model saved to models\fresh_start_v10\best_value_net.pt
Run metadata saved to models\fresh_start_v10\train_run_20260708_012155.json
```
[03:14:35] D2 White-win: n=2398 avg=+0.798 PASS (bar +0.15)
[03:14:35] D2 Black-win: n=8628 avg=-0.820 PASS (bar -0.15)
[03:14:35] START benchmark-v10: src/benchmark.py --model models\fresh_start_v10\best_value_net.pt --games 20 --sims 400 --seed 20260704
[03:39:24] END benchmark-v10 (exit=0, 24.8 min)
```
    "mean_plies_when_won": null,
    "mean_plies_when_lost": 56.9
  },
  "sec_per_decision": 0.7985,
  "elapsed_sec": 1485.9,
  "timestamp": "2026-07-08T03:39:23"
}

--- Per-side strength (vs heuristic anchor) ---
  As White: score 0.7 (7-3-0 W-L-D over 10), mean plies 129.2 (won 88.14, lost 225.0)
  As Black: score 0.0 (0-10-0 W-L-D over 10), mean plies 56.9 (won None, lost 56.9)
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260708_033923.json
```
[03:39:24] ALL STEPS COMPLETE

# Human pawn-phase v11 run (behind-cliff amplification, x6 human) — started 2026-07-09 17:34:48

[17:34:48] START gen-demos: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\mate_demo_starts_v2.jsonl --record-all-plies --seed 1204 --output-dir data\raw\mate_demos_v5
[18:02:30] END gen-demos (exit=0, 27.7 min)
```
  Game 289: 81 moves, Black (-1), 71.0s
  Game 286: 75 moves, Black (-1), 97.2s
  Game 281: 90 moves, Black (-1), 135.8s
  Game 290: 60 moves, Black (-1), 91.4s
  Game 297: 96 moves, Black (-1), 61.3s
  Game 292: 74 moves, White (1), 96.9s
  Game 295: 72 moves, Black (-1), 98.9s

Done! 17304 total positions across 300 saved games (attempted 300).
Results - White: 18, Black: 282, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\mate_demos_v5\generation_summary.json
```
[18:02:30] START gen-heuristic: src/data_generation.py --num-games 800 --simulations 400 --curriculum --curriculum-live-results --record-all-plies --seed 1201 --output-dir data\raw\heuristic_v6
[18:23:24] END gen-heuristic (exit=0, 20.9 min)
```
  Game 745: 225 moves, Black (-0.5), 173.7s
  Game 792: 70 moves, Black (-1), 73.4s
  Game 757: 225 moves, Black (-0.5), 179.5s
  Game 767: 225 moves, Black (-0.5), 151.6s
  Game 786: 171 moves, Black (-1), 132.3s
  Game 726: 225 moves, Black (-0.5), 283.4s
  Game 794: 225 moves, Black (-0.5), 173.6s

Done! 22852 total positions across 800 saved games (attempted 800).
Results - White: 461, Black: 339, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\heuristic_v6\generation_summary.json
```
[18:23:24] START make-human-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\human_games\black_2026_07 --input-dir data\raw\probe_human_v2 --offsets 4,8,12,16 --output data\start_fens\human_bf_starts_v1.jsonl
[18:23:25] END make-human-bf-starts (exit=0, 0.0 min)
```
Wrote 102 unique Black-to-move start FENs from 27 Black-won games to data\start_fens\human_bf_starts_v1.jsonl
```
[18:23:25] START gen-human-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\human_bf_starts_v1.jsonl --start-fen-side black --record-all-plies --seed 1202 --output-dir data\raw\human_blackfocus
[18:43:11] END gen-human-blackfocus (exit=0, 19.8 min)
```
  Game 268: 124 moves, Black (-1), 91.6s
  Game 253: 94 moves, Black (-1), 159.1s
  Game 291: 151 moves, Black (-1), 58.5s
  Game 282: 199 moves, Black (-1), 111.9s
  Game 296: 225 moves, Black (-0.5), 92.5s
  Game 298: 225 moves, Black (-0.5), 116.0s
  Game 280: 225 moves, Black (-0.5), 182.5s

Done! 15977 total positions across 300 saved games (attempted 300).
Results - White: 44, Black: 256, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\human_blackfocus\generation_summary.json
```
[18:43:21] Merged: heuristic=800 demos=282 (dropped 18 non-wins) human_blackfocus=300 human=17 (x6 on black_2026_07) -> data\raw\combined_v9
[18:43:21] START corpus-audit: tools/corpus_audit.py data\raw\combined_v9
[18:43:27] END corpus-audit (exit=0, 0.1 min)
```
=== data\raw\combined_v9 ===
source             games  B-win  W-win start_bm      black 13-15 (open)      black 8-12 (mid)      black 4-7 (late)      black 0-3 (endg)
                                                    pos (%Bwin-labeled)   pos (%Bwin-labeled)   pos (%Bwin-labeled)   pos (%Bwin-labeled)
(root)              1082    621    461      6.0           6024 (  52%)           9026 (  70%)           2607 (  21%)          20684 (  79%)  
human_blackfocus     300    256     44     12.5           6596 (  91%)           9263 (  89%)            118 (  75%)                       -
human_games           17     14      3     15.0           1926 (  98%)            367 (  97%)                       -                     -
TOTAL               1399    891    508                   14546 (  76%)          18656 (  80%)           2725 (  23%)          20684 (  79%)
```
[18:43:27] START process: src/data_processor.py --raw-dir data\raw\combined_v9 --output-dir data\processed\combined_v9 --seed 42
[18:44:17] END process (exit=0, 0.8 min)
```
Retention summary: kept_games=1114/1399, kept_positions=56306/56611
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=890, val=111, test=113
  Processing positions (augment=True)...

Saved to data\processed\combined_v9:
  positions.npy:    (112612, 8, 8, 15)
  mcts_values.npy:  (112612,)
  game_results.npy: (112612,)
  policies.npy:     (112612, 4096)
  splits.npz:       train=89014, val=11972, test=11626
  split_game_ids.json: game-level split membership saved
```
[18:44:17] START train: src/train.py --data-dir data\processed\combined_v9 --model-dir models\fresh_start_v11 --target game_result --value-head wdl --epochs 30 --seed 42
[21:43:13] END train (exit=0, 178.9 min)
```
--- Test set evaluation ---
Total loss: 2.8197
Value power loss: 0.3857
Value true MSE:   0.3141
Policy CE:  2.4341
Value MAE:  0.2701
WDL CE:     0.4969
WDL Acc:    89.5%
Winner prediction accuracy (non-draw): 89.5%

Best model saved to models\fresh_start_v11\best_value_net.pt
Run metadata saved to models\fresh_start_v11\train_run_20260709_184423.json
```
[21:43:29] D2 White-win: n=2904 avg=+0.783 PASS (bar +0.15)
[21:43:29] D2 Black-win: n=8722 avg=-0.731 PASS (bar -0.15)
[21:43:29] START benchmark-v11: src/benchmark.py --model models\fresh_start_v11\best_value_net.pt --games 60 --sims 400 --seed 20260704
[00:12:18] END benchmark-v11 (exit=0, 148.8 min)
```
    "mean_plies_when_won": 192.64,
    "mean_plies_when_lost": 93.31
  },
  "sec_per_decision": 0.9214,
  "elapsed_sec": 8925.9,
  "timestamp": "2026-07-10T00:12:17"
}

--- Per-side strength (vs heuristic anchor) ---
  As White: score 0.4667 (14-16-0 W-L-D over 30), mean plies 183.23 (won 135.5, lost 225.0)
  As Black: score 0.4667 (14-16-0 W-L-D over 30), mean plies 139.67 (won 192.64, lost 93.31)
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260710_001217.json
```
[00:12:18] ALL STEPS COMPLETE

# Human curriculum v12 run (white+black amplification, x6 human) — started 2026-07-10 03:15:37

[03:15:37] START gen-demos: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\mate_demo_starts_v2.jsonl --record-all-plies --seed 1304 --output-dir data\raw\mate_demos_v6
[03:42:17] END gen-demos (exit=0, 26.7 min)
```
  Game 293: 36 moves, Black (-1), 56.2s
  Game 289: 48 moves, Black (-1), 69.8s
  Game 280: 57 moves, Black (-1), 100.6s
  Game 299: 36 moves, Black (-1), 49.4s
  Game 297: 111 moves, Black (-1), 72.1s
  Game 291: 54 moves, Black (-1), 98.0s
  Game 252: 189 moves, Black (-1), 324.5s

Done! 16821 total positions across 300 saved games (attempted 300).
Results - White: 15, Black: 285, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\mate_demos_v6\generation_summary.json
```
[03:42:17] START gen-heuristic: src/data_generation.py --num-games 800 --simulations 400 --curriculum --curriculum-live-results --record-all-plies --seed 1301 --output-dir data\raw\heuristic_v7
[04:03:38] END gen-heuristic (exit=0, 21.3 min)
```
  Game 667: 225 moves, Black (-0.5), 206.0s
  Game 686: 171 moves, Black (-1), 189.4s
  Game 763: 225 moves, Black (-0.5), 93.7s
  Game 626: 225 moves, Black (-0.5), 357.7s
  Game 765: 225 moves, Black (-0.5), 148.7s
  Game 694: 225 moves, Black (-0.5), 245.9s
  Game 799: 225 moves, Black (-0.5), 126.5s

Done! 22756 total positions across 800 saved games (attempted 800).
Results - White: 469, Black: 331, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\heuristic_v7\generation_summary.json
```
[04:03:38] START make-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\human_games\black_2026_07 --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_human_v2 --offsets 4,8,12,16 --output data\start_fens\human_bf_starts_v2.jsonl
[04:03:39] END make-bf-starts (exit=0, 0.0 min)
```
Wrote 141 unique black-to-move start FENs from 37 black-won games to data\start_fens\human_bf_starts_v2.jsonl
```
[04:03:39] START make-wf-starts: src/make_blackfocus_starts.py --side white --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_whitefocus --offsets 4,8,12,16,22 --output data\start_fens\human_wf_starts_v1.jsonl
[04:03:39] END make-wf-starts (exit=0, 0.0 min)
```
Wrote 152 unique white-to-move start FENs from 50 white-won games to data\start_fens\human_wf_starts_v1.jsonl
```
[04:03:39] START gen-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\human_bf_starts_v2.jsonl --start-fen-side black --record-all-plies --seed 1302 --output-dir data\raw\human_blackfocus_v12
[04:24:40] END gen-blackfocus (exit=0, 21.0 min)
```
  Game 279: 79 moves, Black (-1), 69.6s
  Game 243: 225 moves, Black (-0.5), 254.6s
  Game 293: 37 moves, Black (-1), 53.2s
  Game 268: 190 moves, Black (-1), 116.2s
  Game 286: 85 moves, Black (-1), 70.8s
  Game 275: 225 moves, Black (-0.5), 165.9s
  Game 288: 225 moves, Black (-0.5), 155.5s

Done! 14594 total positions across 300 saved games (attempted 300).
Results - White: 52, Black: 248, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\human_blackfocus_v12\generation_summary.json
```
[04:24:40] START gen-whitefocus: src/data_generation.py --num-games 250 --simulations 400 --start-fen-file data\start_fens\human_wf_starts_v1.jsonl --start-fen-side white --record-all-plies --seed 1303 --output-dir data\raw\v12_whitefocus
[04:30:08] END gen-whitefocus (exit=0, 5.5 min)
```
  Game 208: 153 moves, Black (-1), 129.0s
  Game 167: 225 moves, Black (-0.5), 182.1s
  Game 246: 141 moves, Black (-1), 71.7s
  Game 224: 186 moves, Black (-1), 114.2s
  Game 237: 225 moves, Black (-0.5), 102.6s
  Game 228: 135 moves, Black (-1), 118.3s
  Game 157: 164 moves, White (1), 255.3s

Done! 6328 total positions across 250 saved games (attempted 250).
Results - White: 236, Black: 14, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\v12_whitefocus\generation_summary.json
```
[04:30:19] Merged: heuristic=800 demos=285 (dropped 15 non-wins) blackfocus=300 whitefocus=250 human=37 (x6 on black_2026_07+curriculum) -> data\raw\combined_v10
[04:30:19] START corpus-audit: tools/corpus_audit.py data\raw\combined_v10
[04:30:27] END corpus-audit (exit=0, 0.1 min)
```
=== data\raw\combined_v10 ===
source             games  B-win  W-win start_bm      black 13-15 (open)      black 8-12 (mid)      black 4-7 (late)      black 0-3 (endg)
                                                    pos (%Bwin-labeled)   pos (%Bwin-labeled)   pos (%Bwin-labeled)   pos (%Bwin-labeled)
(root)              1085    616    469      6.1           6270 (  53%)           8801 (  69%)           2547 (  24%)          20549 (  79%)  
human_blackfocus     300    248     52     12.5           6934 (  92%)           7457 (  88%)            203 (  89%)                       -
human_games           37     24     13     14.3           3624 (  87%)           1009 (  85%)             54 (   0%)             60 (   0%)  
whitefocus           250     14    236     10.4           1928 (  45%)           3285 (  44%)            602 (   1%)            513 (   0%)  
TOTAL               1672    902    770                   18756 (  73%)          20552 (  73%)           3406 (  23%)          21122 (  77%)
```
[04:30:27] START process: src/data_processor.py --raw-dir data\raw\combined_v10 --output-dir data\processed\combined_v10 --seed 42
[04:31:24] END process (exit=0, 1.0 min)
```
Retention summary: kept_games=1382/1672, kept_positions=63513/63836
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1105, val=137, test=140
  Processing positions (augment=True)...

Saved to data\processed\combined_v10:
  positions.npy:    (127026, 8, 8, 15)
  mcts_values.npy:  (127026,)
  game_results.npy: (127026,)
  policies.npy:     (127026, 4096)
  splits.npz:       train=102294, val=12482, test=12250
  split_game_ids.json: game-level split membership saved
```
[04:31:24] START train: src/train.py --data-dir data\processed\combined_v10 --model-dir models\fresh_start_v12 --target game_result --value-head wdl --epochs 30 --seed 42
[05:47:44] RESUME v12 chain from train (chain killed mid-train)
[05:47:44] START train: src/train.py --data-dir data\processed\combined_v10 --model-dir models\fresh_start_v12 --target game_result --value-head wdl --epochs 30 --seed 42
[08:00:51] END train (exit=0, 133.1 min)
```
--- Test set evaluation ---
Total loss: 2.6565
Value power loss: 0.1910
Value true MSE:   0.1788
Policy CE:  2.4654
Value MAE:  0.2196
WDL CE:     0.1525
WDL Acc:    95.2%
Winner prediction accuracy (non-draw): 95.2%

Best model saved to models\fresh_start_v12\best_value_net.pt
Run metadata saved to models\fresh_start_v12\train_run_20260710_054758.json
```
[08:01:08] D2 White-win: n=3670 avg=+0.775 PASS (bar +0.15)
[08:01:08] D2 Black-win: n=8580 avg=-0.888 PASS (bar -0.15)
[08:01:08] START benchmark-v12: src/benchmark.py --model models\fresh_start_v12\best_value_net.pt --games 60 --sims 400 --seed 20260704
[10:55:50] END benchmark-v12 (exit=0, 174.7 min)
```
    "mean_plies_when_won": 209.79,
    "mean_plies_when_lost": 61.81
  },
  "sec_per_decision": 1.1361,
  "elapsed_sec": 10479.5,
  "timestamp": "2026-07-10T10:55:50"
}

--- Per-side strength (vs heuristic anchor) ---
  As White: score 0.5667 (17-13-0 W-L-D over 30), mean plies 176.6 (won 139.59, lost 225.0)
  As Black: score 0.4667 (14-16-0 W-L-D over 30), mean plies 130.87 (won 209.79, lost 61.81)
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260710_105550.json
```
[10:55:50] ALL STEPS COMPLETE (resumed)

# Human curriculum v13 run (white+black amplification, x6 human) — started 2026-07-11 00:40:31

[00:40:31] START gen-demos: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\mate_demo_starts_v2.jsonl --record-all-plies --seed 1404 --output-dir data\raw\mate_demos_v7
[01:07:01] END gen-demos (exit=0, 26.5 min)
```
  Game 297: 33 moves, Black (-1), 53.3s
  Game 210: 173 moves, White (1), 481.1s
  Game 298: 102 moves, Black (-1), 66.2s
  Game 299: 42 moves, Black (-1), 59.0s
  Game 261: 225 moves, Black (-0.5), 207.7s
  Game 295: 225 moves, Black (-0.5), 104.0s
  Game 276: 179 moves, White (1), 259.6s

Done! 16441 total positions across 300 saved games (attempted 300).
Results - White: 14, Black: 286, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\mate_demos_v7\generation_summary.json
```
[01:07:01] START gen-heuristic: src/data_generation.py --num-games 800 --simulations 400 --curriculum --curriculum-live-results --record-all-plies --seed 1401 --output-dir data\raw\heuristic_v8
[01:25:32] END gen-heuristic (exit=0, 18.5 min)
```
  Game 760: 102 moves, Black (-1), 82.8s
  Game 798: 156 moves, Black (-1), 61.9s
  Game 740: 225 moves, Black (-0.5), 130.2s
  Game 665: 225 moves, Black (-0.5), 200.5s
  Game 594: 225 moves, Black (-0.5), 294.4s
  Game 699: 225 moves, Black (-0.5), 186.6s
  Game 784: 102 moves, Black (-1), 110.3s

Done! 22382 total positions across 800 saved games (attempted 800).
Results - White: 468, Black: 332, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\heuristic_v8\generation_summary.json
```
[01:25:32] START make-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\human_games\black_2026_07 --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_human_v2 --offsets 4,8,12,16 --output data\start_fens\human_bf_starts_v3.jsonl
[01:25:33] END make-bf-starts (exit=0, 0.0 min)
```
Wrote 197 unique black-to-move start FENs from 51 black-won games to data\start_fens\human_bf_starts_v3.jsonl
```
[01:25:33] START make-wf-starts: src/make_blackfocus_starts.py --side white --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_whitefocus --offsets 4,8,12,16,22 --output data\start_fens\human_wf_starts_v2.jsonl
[01:25:33] END make-wf-starts (exit=0, 0.0 min)
```
Wrote 166 unique white-to-move start FENs from 55 white-won games to data\start_fens\human_wf_starts_v2.jsonl
```
[01:25:33] START gen-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\human_bf_starts_v3.jsonl --start-fen-side black --record-all-plies --seed 1402 --output-dir data\raw\human_blackfocus_v13
[01:41:05] END gen-blackfocus (exit=0, 15.5 min)
```
  Game 252: 225 moves, Black (-0.5), 217.9s
  Game 278: 225 moves, Black (-0.5), 108.3s
  Game 279: 160 moves, Black (-1), 108.9s
  Game 273: 187 moves, Black (-1), 173.4s
  Game 292: 132 moves, White (1), 98.0s
  Game 296: 225 moves, Black (-0.5), 111.3s
  Game 297: 225 moves, Black (-0.5), 143.7s

Done! 13183 total positions across 300 saved games (attempted 300).
Results - White: 82, Black: 218, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\human_blackfocus_v13\generation_summary.json
```
[01:41:05] START gen-whitefocus: src/data_generation.py --num-games 250 --simulations 400 --start-fen-file data\start_fens\human_wf_starts_v2.jsonl --start-fen-side white --record-all-plies --seed 1403 --output-dir data\raw\v13_whitefocus
[01:45:48] END gen-whitefocus (exit=0, 4.7 min)
```
  Game 242: 53 moves, White (1), 15.5s
  Game 65: 225 moves, Black (-0.5), 222.2s
  Game 246: 98 moves, White (1), 25.9s
  Game 157: 168 moves, Black (-1), 167.9s
  Game 204: 132 moves, Black (-1), 86.4s
  Game 133: 225 moves, Black (-0.5), 212.3s
  Game 245: 225 moves, Black (-0.5), 57.6s

Done! 6621 total positions across 250 saved games (attempted 250).
Results - White: 237, Black: 13, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\v13_whitefocus\generation_summary.json
```
[01:46:01] Merged: heuristic=800 demos=286 (dropped 14 non-wins) blackfocus=300 whitefocus=250 human=61 (x6 on black_2026_07+curriculum) -> data\raw\combined_v11
[01:46:01] START corpus-audit: tools/corpus_audit.py data\raw\combined_v11
[01:46:08] END corpus-audit (exit=0, 0.1 min)
```
=== data\raw\combined_v11 ===
source             games  B-win  W-win start_bm      black 13-15 (open)      black 8-12 (mid)      black 4-7 (late)      black 0-3 (endg)
                                                    pos (%Bwin-labeled)   pos (%Bwin-labeled)   pos (%Bwin-labeled)   pos (%Bwin-labeled)
(root)              1086    618    468      6.1           6684 (  55%)           8126 (  66%)           2416 (  21%)          20216 (  79%)  
human_blackfocus     300    218     82     12.4           5058 (  85%)           7514 (  84%)            598 (  61%)             13 (   0%)  
human_games           61     38     23     13.9           5671 (  86%)           2145 (  83%)            192 (  62%)            114 (   0%)  
whitefocus           250     13    237     11.1           2338 (  50%)           3198 (  33%)            502 (   0%)            583 (   0%)  
TOTAL               1697    887    810                   19751 (  71%)          20983 (  69%)           3708 (  27%)          20926 (  76%)
```
[01:46:08] START process: src/data_processor.py --raw-dir data\raw\combined_v11 --output-dir data\processed\combined_v11 --seed 42
[01:47:22] END process (exit=0, 1.2 min)
```
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1123, val=139, test=143
  Processing positions (augment=True)...
  Value targets: game_result discounted per ply (gamma=0.999)

Saved to data\processed\combined_v11:
  positions.npy:    (130092, 8, 8, 15)
  mcts_values.npy:  (130092,)
  game_results.npy: (130092,)
  policies.npy:     (130092, 4096)
  splits.npz:       train=105486, val=11542, test=13064
  split_game_ids.json: game-level split membership saved
```
[01:47:22] START train: src/train.py --data-dir data\processed\combined_v11 --model-dir models\fresh_start_v13 --target game_result --value-head wdl --epochs 30 --seed 42
[03:56:58] END train (exit=0, 129.6 min)
```
--- Test set evaluation ---
Total loss: 2.8485
Value power loss: 0.3581
Value true MSE:   0.3316
Policy CE:  2.4905
Value MAE:  0.3624
WDL CE:     0.3181
WDL Acc:    86.8%
Winner prediction accuracy (non-draw): 86.8%

Best model saved to models\fresh_start_v13\best_value_net.pt
Run metadata saved to models\fresh_start_v13\train_run_20260711_014727.json
```
[03:57:13] D2 White-win: n=3928 avg=+0.552 PASS (bar +0.15)
[03:57:13] D2 Black-win: n=9136 avg=-0.724 PASS (bar -0.15)
[03:57:13] START benchmark-v13: src/benchmark.py --model models\fresh_start_v13\best_value_net.pt --games 60 --sims 400 --seed 20260704
[05:10:28] END benchmark-v13 (exit=0, 73.2 min)
```
    "mean_plies_when_won": 108.38,
    "mean_plies_when_lost": 60.77
  },
  "sec_per_decision": 0.8708,
  "elapsed_sec": 4392.1,
  "timestamp": "2026-07-11T05:10:28"
}

--- Per-side strength (vs heuristic anchor) ---
  As White: score 0.8333 (25-5-0 W-L-D over 30), mean plies 94.67 (won 68.6, lost 225.0)
  As Black: score 0.2667 (8-22-0 W-L-D over 30), mean plies 73.47 (won 108.38, lost 60.77)
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260711_051028.json
```
[05:10:28] ALL STEPS COMPLETE

# v14: reprocess combined_v11 with near-mate ramp (floor 0.97, horizon 10) — started 2026-07-11 10:49:26

[10:49:26] START process: src/data_processor.py --raw-dir data\raw\combined_v11 --output-dir data\processed\combined_v11 --seed 42
[10:50:51] END process (exit=0, 1.4 min)
```
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1123, val=139, test=143
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\combined_v11:
  positions.npy:    (130092, 8, 8, 15)
  mcts_values.npy:  (130092,)
  game_results.npy: (130092,)
  policies.npy:     (130092, 4096)
  splits.npz:       train=105486, val=11542, test=13064
  split_game_ids.json: game-level split membership saved
```
[10:50:51] START train: src/train.py --data-dir data\processed\combined_v11 --model-dir models/fresh_start_v14 --target game_result --value-head wdl --epochs 30 --seed 42
[13:19:11] END train (exit=0, 148.3 min)
```
--- Test set evaluation ---
Total loss: 2.8169
Value power loss: 0.3360
Value true MSE:   0.2986
Policy CE:  2.4809
Value MAE:  0.3130
WDL CE:     0.2843
WDL Acc:    89.5%
Winner prediction accuracy (non-draw): 89.5%

Best model saved to models/fresh_start_v14\best_value_net.pt
Run metadata saved to models/fresh_start_v14\train_run_20260711_105056.json
```
[13:19:31] D2 White-win: n=3928 avg=+0.722 PASS (bar +0.15)
[13:19:31] D2 Black-win: n=9136 avg=-0.757 PASS (bar -0.15)
[13:19:31] START benchmark-v14: src/benchmark.py --model models/fresh_start_v14/best_value_net.pt --games 60 --sims 400 --seed 20260704
[14:50:57] END benchmark-v14 (exit=0, 91.4 min)
```
    "mean_plies_when_won": 87.86,
    "mean_plies_when_lost": 49.74
  },
  "sec_per_decision": 0.8452,
  "elapsed_sec": 5483.8,
  "timestamp": "2026-07-11T14:50:57"
}

--- Per-side strength (vs heuristic anchor) ---
  As White: score 0.6667 (20-10-0 W-L-D over 30), mean plies 157.63 (won 123.95, lost 225.0)
  As Black: score 0.2333 (7-23-0 W-L-D over 30), mean plies 58.63 (won 87.86, lost 49.74)
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260711_145057.json
```
[14:50:57] ALL STEPS COMPLETE (v14 relabel run)

# v15: combined_v11 with win-filtered blackfocus (pretrain gate PASS) — started 2026-07-11 17:52:20

[17:52:20] START pretrain-check: tools/pretrain_check.py data/raw/combined_v11f --reference data/raw/combined_v10
[17:52:23] END pretrain-check (exit=0, 0.1 min)
```
=== pretrain_check: data/raw/combined_v11f ===
total positions: 60259, sources: {'(root)': 37442, 'human_blackfocus': 8074, 'human_games': 8122, 'whitefocus': 6621}
  OK  purity human_blackfocus: 205/205 = 100% (black wins, min 85%)
  OK  purity whitefocus: 237/250 = 95% (white wins, min 85%)
  OK  human data: 8122 positions (13.5% of corpus), 1489 unique, effective duplication x5.5 (max x8)
  OK  phase open: B-win share 72% vs ref 73% (delta -1%)
  OK  phase mid: B-win share 71% vs ref 73% (delta -2%)
  OK  phase late: B-win share 27% vs ref 23% (delta +4%)
  OK  phase endg: B-win share 76% vs ref 77% (delta -1%)
  OK  label bias: mean |target| white-won 0.977 vs black-won 0.973 (rel gap 0.4%, max 5%)
PRETRAIN CHECK: PASS
```
[17:52:23] START process: src/data_processor.py --raw-dir data/raw/combined_v11f --output-dir data/processed/combined_v11f --seed 42
[17:53:35] END process (exit=0, 1.2 min)
```
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1054, val=131, test=134
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data/processed/combined_v11f:
  positions.npy:    (119928, 8, 8, 15)
  mcts_values.npy:  (119928,)
  game_results.npy: (119928,)
  policies.npy:     (119928, 4096)
  splits.npz:       train=96414, val=10316, test=13198
  split_game_ids.json: game-level split membership saved
```
[17:53:35] START train: src/train.py --data-dir data/processed/combined_v11f --model-dir models/fresh_start_v15 --target game_result --value-head wdl --epochs 30 --seed 42
[21:44:31] END train (exit=0, 230.9 min)
```
--- Test set evaluation ---
Total loss: 2.8799
Value power loss: 0.2947
Value true MSE:   0.2523
Policy CE:  2.5852
Value MAE:  0.2526
WDL CE:     0.3416
WDL Acc:    91.8%
Winner prediction accuracy (non-draw): 91.8%

Best model saved to models/fresh_start_v15\best_value_net.pt
Run metadata saved to models/fresh_start_v15\train_run_20260711_175340.json
```
[21:44:56] D2 White-win: n=3558 avg=+0.807 PASS (bar +0.15)
[21:44:56] D2 Black-win: n=9640 avg=-0.805 PASS (bar -0.15)
[21:44:56] START benchmark-v15: src/benchmark.py --model models/fresh_start_v15/best_value_net.pt --games 60 --sims 400 --seed 20260704
[00:17:10] END benchmark-v15 (exit=0, 152.2 min)
```
    "mean_plies_when_won": 154.25,
    "mean_plies_when_lost": 88.0
  },
  "sec_per_decision": 0.9679,
  "elapsed_sec": 9130.0,
  "timestamp": "2026-07-12T00:17:09"
}

--- Per-side strength (vs heuristic anchor) ---
  As White: score 0.6667 (20-10-0 W-L-D over 30), mean plies 199.93 (won 187.4, lost 225.0)
  As Black: score 0.4 (12-18-0 W-L-D over 30), mean plies 114.5 (won 154.25, lost 88.0)
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260712_001709.json
```
[00:17:10] ALL STEPS COMPLETE (v15 win-filtered run)

# v16 run (frozen base + new curriculum + engine king-safety, gated) — started 2026-07-12 02:51:16

[02:51:16] START make-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\human_games\black_2026_07 --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_human_v2 --offsets 4,8,12,16 --output data\start_fens\human_bf_starts_v4.jsonl
[02:51:17] END make-bf-starts (exit=0, 0.0 min)
```
Wrote 223 unique black-to-move start FENs from 58 black-won games to data\start_fens\human_bf_starts_v4.jsonl
```
[02:51:17] START make-wf-starts: src/make_blackfocus_starts.py --side white --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_whitefocus --offsets 4,8,12,16,22 --output data\start_fens\human_wf_starts_v3.jsonl
[02:51:17] END make-wf-starts (exit=0, 0.0 min)
```
Wrote 171 unique white-to-move start FENs from 56 white-won games to data\start_fens\human_wf_starts_v3.jsonl
```
[02:51:17] START gen-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\human_bf_starts_v4.jsonl --start-fen-side black --record-all-plies --seed 1502 --output-dir data\raw\human_blackfocus_v16
[03:13:14] END gen-blackfocus (exit=0, 21.9 min)
```
  Game 271: 225 moves, Black (-0.5), 166.7s
  Game 289: 151 moves, Black (-1), 103.6s
  Game 228: 225 moves, Black (-0.5), 381.0s
  Game 299: 106 moves, Black (-1), 48.6s
  Game 287: 109 moves, Black (-1), 118.7s
  Game 252: 138 moves, White (1), 300.3s
  Game 288: 199 moves, Black (-1), 145.4s
  Game 290: 225 moves, Black (-0.5), 143.1s
  Game 276: 199 moves, Black (-1), 208.6s

Done! 17975 total positions across 300 saved games (attempted 300).
Results - White: 60, Black: 240, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\human_blackfocus_v16\generation_summary.json
```
[03:13:14] START gen-whitefocus: src/data_generation.py --num-games 250 --simulations 400 --start-fen-file data\start_fens\human_wf_starts_v3.jsonl --start-fen-side white --record-all-plies --seed 1503 --output-dir data\raw\v16_whitefocus
[03:24:57] END gen-whitefocus (exit=0, 11.7 min)
```
  Game 249: 68 moves, White (1), 20.4s
  Game 162: 225 moves, Black (-0.5), 247.8s
  Game 219: 225 moves, Black (-0.5), 127.5s
  Game 173: 225 moves, Black (-0.5), 232.1s
  Game 239: 225 moves, Black (-0.5), 87.7s
  Game 158: 225 moves, Black (-0.5), 315.2s
  Game 193: 225 moves, Black (-0.5), 234.8s
  Game 233: 225 moves, Black (-0.5), 152.1s
  Game 197: 198 moves, Black (-1), 290.4s

Done! 11541 total positions across 250 saved games (attempted 250).
Results - White: 216, Black: 34, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\v16_whitefocus\generation_summary.json
```
[03:25:14] Merged: heuristic=800 (frozen) demos=286 (dropped 14) blackfocus=209 (dropped 91 non-wins) whitefocus=216 (dropped 34 non-wins) human=73 (x6 fresh dirs) -> data\raw\combined_v12
[03:25:14] START pretrain-gate: tools/pretrain_check.py data\raw\combined_v12 --reference data\raw\combined_v11f
[03:25:36] END pretrain-gate (exit=0, 0.4 min)
```
=== pretrain_check: data\raw\combined_v12 ===
total positions: 61239, sources: {'(root)': 37442, 'human_blackfocus': 8711, 'human_games': 9641, 'whitefocus': 5445}
  OK  purity human_blackfocus: 209/209 = 100% (black wins, min 85%)
  OK  purity whitefocus: 216/216 = 100% (white wins, min 85%)
  OK  human data: 9641 positions (15.7% of corpus), 1759 unique, effective duplication x5.5 (max x8)
  OK  phase open: B-win share 73% vs ref 72% (delta +1%)
  OK  phase mid: B-win share 66% vs ref 71% (delta -5%)
  OK  phase late: B-win share 23% vs ref 27% (delta -3%)
  OK  phase endg: B-win share 76% vs ref 76% (delta -0%)
  OK  label bias: mean |target| white-won 0.977 vs black-won 0.973 (rel gap 0.4%, max 5%)
PRETRAIN CHECK: PASS
```
[03:25:36] START process: src/data_processor.py --raw-dir data\raw\combined_v12 --output-dir data\processed\combined_v12 --seed 42
[03:26:49] END process (exit=0, 1.2 min)
```
Retention summary: kept_games=1301/1584, kept_positions=60943/61239
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1040, val=129, test=132
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\combined_v12:
  positions.npy:    (121886, 8, 8, 15)
  mcts_values.npy:  (121886,)
  game_results.npy: (121886,)
  policies.npy:     (121886, 4096)
  splits.npz:       train=97884, val=10170, test=13832
  split_game_ids.json: game-level split membership saved
```
[03:26:49] START train: src/train.py --data-dir data\processed\combined_v12 --model-dir models\fresh_start_v16 --target game_result --value-head wdl --epochs 30 --seed 42
[05:27:45] END train (exit=0, 120.9 min)
```
Early stopping at epoch 16

--- Test set evaluation ---
Total loss: 2.8440
Value power loss: 0.2460
Value true MSE:   0.2245
Policy CE:  2.5979
Value MAE:  0.2483
WDL CE:     0.2179
WDL Acc:    91.5%
Winner prediction accuracy (non-draw): 91.5%

Best model saved to models\fresh_start_v16\best_value_net.pt
Run metadata saved to models\fresh_start_v16\train_run_20260712_032654.json
```
[05:28:03] D2 White-win: n=4408 avg=+0.747 PASS (bar +0.15)
[05:28:03] D2 Black-win: n=9424 avg=-0.817 PASS (bar -0.15)
[05:28:03] START benchmark-v16: src/benchmark.py --model models\fresh_start_v16\best_value_net.pt --games 60 --sims 400 --seed 20260704
[07:18:35] END benchmark-v16 (exit=0, 110.5 min)
```
    "score": 0.4667,
    "mean_plies": 127.97,
    "mean_plies_when_won": 189.21,
    "mean_plies_when_lost": 74.38
  },
  "sec_per_decision": 0.8238,
  "elapsed_sec": 6629.5,
  "timestamp": "2026-07-12T07:18:35"
}

--- Per-side strength (vs heuristic anchor) ---
  As White: score 0.8 (24-6-0 W-L-D over 30), mean plies 140.27 (won 119.08, lost 225.0)
  As Black: score 0.4667 (14-16-0 W-L-D over 30), mean plies 127.97 (won 189.21, lost 74.38)
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\benchmark_best_value_net_20260712_071835.json
```
[07:18:35] START match-v16-vs-v15: tools/match.py --model-a models\fresh_start_v16\best_value_net.pt --model-b models\fresh_start_v15\best_value_net.pt --games 20 --workers 6
[08:07:58] END match-v16-vs-v15 (exit=0, 49.4 min)
```
  "a_as_black": {
    "games": 10,
    "wins": 6,
    "losses": 4,
    "draws": 0,
    "score": 0.6,
    "mean_plies": 183.9,
    "mean_plies_when_won": 195.0,
    "mean_plies_when_lost": 167.25
  },
  "elapsed_sec": 2963.0,
  "timestamp": "2026-07-12T08:07:58"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\match_fresh_start_v16_vs_fresh_start_v15_20260712_080758.json
```
[08:07:58] ALL STEPS COMPLETE

# v17 White-runner contrast run — started 2026-07-13 01:14:11

[01:14:11] START make-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\human_games\black_2026_07 --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_human_v2 --offsets 4,8,12,16 --output data\start_fens\human_bf_starts_v5.jsonl
[01:14:12] END make-bf-starts (exit=0, 0.0 min)
```
Wrote 227 unique black-to-move start FENs from 59 black-won games to data\start_fens\human_bf_starts_v5.jsonl
```
[01:14:12] START make-wf-starts: src/make_blackfocus_starts.py --side white --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_whitefocus --offsets 4,8,12,16,22 --output data\start_fens\human_wf_starts_v4.jsonl
[01:14:13] END make-wf-starts (exit=0, 0.0 min)
```
Wrote 171 unique white-to-move start FENs from 56 white-won games to data\start_fens\human_wf_starts_v4.jsonl
```
[01:14:13] START gen-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\human_bf_starts_v5.jsonl --start-fen-side black --record-all-plies --seed 1602 --output-dir data\raw\human_blackfocus_v17
[01:36:16] END gen-blackfocus (exit=0, 22.0 min)
```
  Game 298: 27 moves, White (1), 6.9s
  Game 279: 37 moves, Black (-1), 93.9s
  Game 290: 82 moves, Black (-1), 42.1s
  Game 299: 16 moves, Black (-1), 18.7s
  Game 287: 58 moves, Black (-1), 80.9s
  Game 218: 225 moves, Black (-0.5), 403.8s
  Game 296: 46 moves, Black (-1), 69.2s
  Game 284: 46 moves, Black (-1), 131.9s
  Game 297: 91 moves, Black (-1), 100.8s
  Game 266: 225 moves, Black (-0.5), 242.8s
  Game 277: 225 moves, Black (-0.5), 210.0s
  Game 264: 225 moves, Black (-0.5), 275.4s
  Game 275: 196 moves, Black (-1), 244.7s

Done! 15968 total positions across 300 saved games (attempted 300).
Results - White: 49, Black: 251, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\human_blackfocus_v17\generation_summary.json
```
[01:36:16] START gen-whitefocus: src/data_generation.py --num-games 250 --simulations 400 --start-fen-file data\start_fens\human_wf_starts_v4.jsonl --start-fen-side white --record-all-plies --seed 1603 --output-dir data\raw\v17_whitefocus
[01:45:37] END gen-whitefocus (exit=0, 9.4 min)
```
  Game 247: 20 moves, White (1), 4.1s
  Game 248: 11 moves, White (1), 2.2s
  Game 227: 92 moves, White (1), 33.6s
  Game 133: 225 moves, Black (-0.5), 219.6s
  Game 246: 50 moves, White (1), 15.2s
  Game 249: 41 moves, White (1), 10.8s
  Game 207: 141 moves, Black (-1), 96.9s
  Game 193: 141 moves, Black (-1), 132.2s
  Game 174: 135 moves, Black (-1), 159.6s
  Game 158: 225 moves, Black (-0.5), 201.2s
  Game 216: 168 moves, Black (-1), 102.8s
  Game 97: 198 moves, Black (-1), 375.8s
  Game 189: 225 moves, Black (-0.5), 175.3s

Done! 10358 total positions across 250 saved games (attempted 250).
Results - White: 220, Black: 30, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\v17_whitefocus\generation_summary.json
```
[01:45:37] START gen-promo-white-runner: src/data_generation.py --num-games 160 --simulations 400 --start-fen-file data\start_fens\promo_races_probe.jsonl --start-fen-source promo_white_runner --record-all-plies --seed 1604 --output-dir data\raw\promo_races_v17_raw
[01:50:49] END gen-promo-white-runner (exit=0, 5.2 min)
```
  Game 157: 15 moves, White (1), 4.6s
  Game 151: 44 moves, White (1), 13.8s
  Game 140: 76 moves, Black (-1), 33.1s
  Game 76: 225 moves, Black (-0.5), 145.8s
  Game 159: 17 moves, White (1), 4.2s
  Game 158: 30 moves, White (1), 8.3s
  Game 156: 72 moves, White (1), 22.3s
  Game 149: 10 moves, Black (-1), 36.0s
  Game 138: 10 moves, Black (-1), 55.9s
  Game 31: 225 moves, Black (-0.5), 285.6s
  Game 116: 225 moves, Black (-0.5), 134.1s
  Game 95: 225 moves, Black (-0.5), 159.5s
  Game 37: 225 moves, Black (-0.5), 284.2s

Done! 6103 total positions across 160 saved games (attempted 160).
Results - White: 140, Black: 20, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\promo_races_v17_raw\generation_summary.json
```
[01:50:49] START prepare-promo-policy: src/promotion_data.py data\raw\promo_races_v17_raw data\raw\promo_races_v17 --expected-start-source promo_white_runner
[01:50:51] END prepare-promo-policy (exit=0, 0.0 min)
```
{
  "games": 160,
  "successful_preventions": 15,
  "failed_games": 145,
  "masked_black_positions": 1170,
  "value_records_kept": 6103
}
```
[01:51:12] Merged v17 corpus: {"blackfocus": 227, "blackfocus_dropped": 73, "demos": 286, "demos_dropped": 14, "generated_black_runner_games": 0, "heuristic": 800, "human_duplication": 6, "human_games": 75, "promo_white_runner_games": 160, "recipe": "v16 frozen base + White-runner contrast", "sha256": "5ff06c1690a8da962c7a075ac32ea4bc78b0ac01880c534682bbd9487a60ef68", "whitefocus": 220, "whitefocus_dropped": 30}
[01:51:12] START pretrain-gate: tools/pretrain_check.py data\raw\combined_v13 --reference data\raw\combined_v12
[01:51:15] END pretrain-gate (exit=0, 0.1 min)
```
=== pretrain_check: data\raw\combined_v13 ===
total positions: 67309, sources: {'(root)': 37442, 'human_blackfocus': 8447, 'human_games': 9972, 'promo_races': 6103, 'whitefocus': 5345}
  OK  promo provenance: 6103 White-runner records, 0 generated Black-runner records
  OK  promo policy weights explicit on every Black position
  OK  promo policy weights match prevention outcomes
  OK  purity human_blackfocus: 227/227 = 100% (black wins, min 85%)
  OK  purity whitefocus: 220/220 = 100% (white wins, min 85%)
  OK  human data: 9972 positions (14.8% of corpus), 1824 unique, effective duplication x5.5 (max x8)
  OK  phase open: B-win share 74% vs ref 73% (delta +0%)
  OK  phase mid: B-win share 64% vs ref 66% (delta -2%)
  OK  phase late: B-win share 29% vs ref 23% (delta +6%)
  OK  phase endg: B-win share 73% vs ref 76% (delta -3%)
  OK  label bias: mean |target| white-won 0.977 vs black-won 0.973 (rel gap 0.4%, max 5%)
PRETRAIN CHECK: PASS
```
[01:51:15] START process: src/data_processor.py --raw-dir data\raw\combined_v13 --output-dir data\processed\combined_v13 --seed 42
[01:52:21] END process (exit=0, 1.1 min)
```
Retention summary: kept_games=1468/1768, kept_positions=66979/67309
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1174, val=146, test=148
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\combined_v13:
  positions.npy:    (133958, 8, 8, 15)
  mcts_values.npy:  (133958,)
  game_results.npy: (133958,)
  policies.npy:     (133958, 4096)
  policy_weights.npy: (133958,) (masked=2340)
  splits.npz:       train=107828, val=11870, test=14260
  split_game_ids.json: game-level split membership saved
```
[01:52:21] START train: src/train.py --data-dir data\processed\combined_v13 --model-dir models\fresh_start_v17 --target game_result --value-head wdl --epochs 30 --seed 42
[05:08:27] END train (exit=0, 196.1 min)
```
Epoch  20  train=1.4205 (v=0.0255 p=1.3816)  val=3.0408 (pow=0.4602 mse=0.3569 p=2.5806 mae=0.2629)  lr=8.8e-04  wdl(train_ce=0.0269 val_ce=0.5021 val_acc=89.3%)
Epoch  21  train=1.4144 (v=0.0246 p=1.3767)  val=3.2224 (pow=0.4643 mse=0.3598 p=2.7580 mae=0.2668)  lr=8.4e-04  wdl(train_ce=0.0262 val_ce=0.5731 val_acc=89.3%)
Epoch  22  train=1.4083 (v=0.0232 p=1.3725)  val=3.0804 (pow=0.5033 mse=0.3816 p=2.5771 mae=0.2642)  lr=7.9e-04  wdl(train_ce=0.0252 val_ce=0.8148 val_acc=89.0%)
Epoch  23  train=1.4083 (v=0.0239 p=1.3714)  val=3.0497 (pow=0.4509 mse=0.3472 p=2.5988 mae=0.2547)  lr=7.5e-04  wdl(train_ce=0.0260 val_ce=0.6105 val_acc=89.7%)
Early stopping at epoch 23

--- Test set evaluation ---
Total loss: 2.7755
Value power loss: 0.2607
Value true MSE:   0.2220
Policy CE:  2.5148
Value MAE:  0.2262
WDL CE:     0.3021
WDL Acc:    91.7%
Winner prediction accuracy (non-draw): 91.7%

Best model saved to models\fresh_start_v17\best_value_net.pt
Run metadata saved to models\fresh_start_v17\train_run_20260713_015226.json
```
[05:08:43] D2 White-win: n=4704 avg=+0.903 PASS (bar +0.15)
[05:08:43] D2 Black-win: n=9556 avg=-0.725 PASS (bar -0.15)
[05:08:43] START promotion-gate-black-defends: tools/promotion_probe.py --candidate models\fresh_start_v17\best_value_net.pt --incumbent models\fresh_start_v16\best_value_net.pt --start-fen-file data\start_fens\promo_races_probe.jsonl --source promo_white_runner --defender black --sims 400 --workers 6 --enforce --max-prevention-drop 0 --max-king-survival-drop 0 --max-score-drop 0.05
[05:16:56] END promotion-gate-black-defends (exit=1, 8.2 min)
```
      "king_survival_rate": -0.13333333333333333,
      "defender_score": -0.13333333333333333
    },
    "failures": {
      "king_survival_rate": {
        "delta": -0.13333333333333333,
        "max_drop": 0.0
      },
      "defender_score": {
        "delta": -0.13333333333333333,
        "max_drop": 0.05
      }
    }
  },
  "elapsed_sec": 492.4,
  "timestamp": "2026-07-13T05:16:56"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\promotion_probe_black_defends_20260713_051656.json
```
[05:16:56] STDERR promotion-gate-black-defends:
```

```
[05:16:56] CHAIN ABORTED: RuntimeError('promotion-gate-black-defends failed (exit 1)')
