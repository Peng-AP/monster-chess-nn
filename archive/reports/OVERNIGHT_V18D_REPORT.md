
# DETOX: v18 candidate 3 — started 2026-07-17 02:14:39

[02:14:39] START make-bf-starts: src/make_blackfocus_starts.py --input-dir data\raw\human_games\black_2026_07 --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_human_v2 --offsets 4,8,12,16 --output data\start_fens\human_bf_starts_v6.jsonl
[02:14:40] END make-bf-starts (exit=0, 0.0 min)
```
Wrote 251 unique black-to-move start FENs from 65 black-won games to data\start_fens\human_bf_starts_v6.jsonl
```
[02:14:40] START make-wf-starts: src/make_blackfocus_starts.py --side white --input-dir data\raw\human_games\curriculum_2026_07 --input-dir data\raw\probe_whitefocus --offsets 4,8,12,16,22 --output data\start_fens\human_wf_starts_v5.jsonl
[02:14:41] END make-wf-starts (exit=0, 0.0 min)
```
Wrote 171 unique white-to-move start FENs from 56 white-won games to data\start_fens\human_wf_starts_v5.jsonl
```
[02:14:41] START gen-blackfocus: src/data_generation.py --num-games 300 --simulations 400 --start-fen-file data\start_fens\human_bf_starts_v6.jsonl --start-fen-side black --record-all-plies --seed 1702 --output-dir data\raw\human_blackfocus_v18d
[02:43:00] END gen-blackfocus (exit=0, 28.3 min)
```
  Game 291: 48 moves, White (1), 20.3s
  Game 290: 55 moves, Black (-1), 23.7s
  Game 293: 34 moves, Black (-1), 14.5s
  Game 295: 22 moves, Black (-1), 8.1s
  Game 226: 225 moves, Black (-0.5), 406.4s
  Game 292: 43 moves, Black (-1), 37.8s
  Game 268: 220 moves, Black (-1), 176.1s
  Game 297: 4 moves, Black (-1), 6.7s
  Game 296: 60 moves, White (1), 19.0s
  Game 261: 225 moves, Black (-0.5), 229.8s
  Game 249: 225 moves, Black (-0.5), 305.5s
  Game 299: 114 moves, White (1), 41.9s
  Game 298: 97 moves, Black (-1), 60.9s
  Game 284: 225 moves, Black (-0.5), 137.2s
  Game 274: 225 moves, Black (-0.5), 223.5s
  Game 285: 115 moves, Black (-1), 154.6s
  Game 294: 225 moves, Black (-0.5), 134.2s
  Game 263: 225 moves, Black (-0.5), 336.4s
  Game 289: 201 moves, White (1), 220.0s

Done! 20343 total positions across 300 saved games (attempted 300).
Results - White: 69, Black: 231, Draw: 0 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\human_blackfocus_v18d\generation_summary.json
```
[02:43:00] START gen-whitefocus: src/data_generation.py --num-games 250 --simulations 400 --start-fen-file data\start_fens\human_wf_starts_v5.jsonl --start-fen-side white --record-all-plies --seed 1703 --output-dir data\raw\v18d_whitefocus
[02:53:56] END gen-whitefocus (exit=0, 10.9 min)
```
  Game 211: 114 moves, Black (-1), 110.8s
  Game 242: 11 moves, White (1), 2.4s
  Game 243: 26 moves, White (1), 7.0s
  Game 244: 29 moves, White (1), 9.1s
  Game 245: 26 moves, White (1), 6.1s
  Game 241: 62 moves, White (1), 20.6s
  Game 215: 225 moves, Black (-0.5), 126.3s
  Game 249: 14 moves, White (1), 3.9s
  Game 247: 50 moves, White (1), 14.1s
  Game 246: 53 moves, White (1), 17.4s
  Game 222: 225 moves, Black (-0.5), 134.1s
  Game 163: 225 moves, Black (-0.5), 234.1s
  Game 196: 225 moves, Black (-0.5), 184.5s
  Game 155: 225 moves, Black (-0.5), 300.9s
  Game 223: 225 moves, Black (-0.5), 199.1s
  Game 248: 225 moves, Black (-0.5), 147.1s
  Game 219: 186 moves, Black (-1), 276.3s
  Game 229: 225 moves, Black (-0.5), 233.4s
  Game 110: 225 moves, Black (-0.5), 510.2s

Done! 10263 total positions across 250 saved games (attempted 250).
Results - White: 225, Black: 24, Draw: 1 (saved games only)
Simulation usage stats: min=400 max=400 mean=400.00
Generation summary saved to data\raw\v18d_whitefocus\generation_summary.json
```
[02:54:21] Merged detox corpus: {"blackfocus": 201, "blackfocus_dropped": 99, "demos": 286, "demos_dropped": 14, "generated_black_runner_games": 0, "generator_guards": ["king_safety", "white_first_half", "oscillation_penalty"], "heuristic": 800, "human_ai_policy_masked": true, "human_duplication": 6, "human_games": 83, "position_encoding_channels": 15, "promo_white_runner_games": 160, "recipe": "v17 detox: frozen recipe, masked human-AI policy, guard-regenerated focus, all current human games", "sha256": "0411cda04979d6694b91723827474398fd356b1e0897d9e8b419d0f59e6ec1a4", "value_discount_mode": "near_mate", "value_floor": 0.97, "value_horizon": 10, "whitefocus": 225, "whitefocus_dropped": 25}
[02:54:21] START pretrain-gate: tools/pretrain_check.py data\raw\combined_v16 --reference data\raw\combined_v13
[02:54:35] END pretrain-gate (exit=0, 0.2 min)
```
=== pretrain_check: data\raw\combined_v16 ===
total positions: 70805, sources: {'(root)': 37442, 'human_blackfocus': 10413, 'human_games': 11330, 'promo_races': 6103, 'whitefocus': 5517}
  OK  promo provenance: 6103 White-runner records, 0 generated Black-runner records
  OK  promo policy weights explicit on every Black position
  OK  promo policy weights match prevention outcomes
  OK  purity human_blackfocus: 201/201 = 100% (black wins, min 85%)
  OK  purity whitefocus: 225/225 = 100% (white wins, min 85%)
  OK  human data: 11330 positions (16.0% of corpus), 2077 unique, effective duplication x5.5 (max x8)
  OK  phase open: B-win share 75% vs ref 74% (delta +1%)
  OK  phase mid: B-win share 68% vs ref 64% (delta +4%)
  OK  phase late: B-win share 27% vs ref 29% (delta -1%)
  OK  phase endg: B-win share 72% vs ref 73% (delta -1%)
  OK  label bias: mean |target| white-won 0.977 vs black-won 0.973 (rel gap 0.4%, max 5%)
PRETRAIN CHECK: PASS
```
[02:54:35] START process: src/data_processor.py --raw-dir data\raw\combined_v16 --output-dir data\processed\combined_v16 --seed 42 --channels 15 --value-discount-mode near_mate --value-horizon 10 --value-floor 0.97
[02:55:48] END process (exit=0, 1.2 min)
```
Retention summary: kept_games=1457/1755, kept_positions=70478/70805
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1164, val=145, test=148
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\combined_v16:
  positions.npy:    (140956, 8, 8, 15)
  mcts_values.npy:  (140956,)
  game_results.npy: (140956,)
  policies.npy:     (140956, 4096)
  policy_weights.npy: (140956,) (masked=13372)
  splits.npz:       train=112760, val=11964, test=16232
  split_game_ids.json: game-level split membership saved
```
[02:55:48] START train: src/train.py --data-dir data\processed\combined_v16 --model-dir models\candidates\fresh_start_v18_detox --target game_result --value-head wdl --select-metric decisive --epochs 30 --seed 42
[07:30:27] END train (exit=0, 274.7 min)
```
  -> saved best model (decisive_score=1.2438)
Epoch  25  train=1.3284 (v=0.0219 p=1.2948)  val=3.2066 (pow=0.2970 mse=0.2297 p=2.6979 mae=0.1799)  lr=6.8e-04  wdl(train_ce=0.0234 val_ce=0.4233 val_acc=93.7%)  decisive(top1 W=34.8% B=30.5% sign W=94.4% B=92.2%)
Epoch  26  train=1.3260 (v=0.0217 p=1.2926)  val=3.0737 (pow=0.2852 mse=0.2205 p=2.5849 mae=0.1733)  lr=6.5e-04  wdl(train_ce=0.0234 val_ce=0.4072 val_acc=93.8%)  decisive(top1 W=32.6% B=31.8% sign W=94.2% B=93.0%)
  -> saved best model (decisive_score=1.2476)
Epoch  27  train=1.3219 (v=0.0208 p=1.2898)  val=3.3713 (pow=0.2534 mse=0.1983 p=2.9511 mae=0.1638)  lr=6.1e-04  wdl(train_ce=0.0226 val_ce=0.3336 val_acc=94.4%)  decisive(top1 W=36.2% B=31.9% sign W=95.1% B=93.3%)
  -> saved best model (decisive_score=1.2518)
Epoch  28  train=1.3184 (v=0.0201 p=1.2873)  val=3.0960 (pow=0.2578 mse=0.2011 p=2.6593 mae=0.1649)  lr=5.8e-04  wdl(train_ce=0.0221 val_ce=0.3578 val_acc=94.3%)  decisive(top1 W=34.6% B=31.4% sign W=95.2% B=92.7%)
Epoch  29  train=1.3160 (v=0.0198 p=1.2853)  val=3.0684 (pow=0.2796 mse=0.2154 p=2.5775 mae=0.1712)  lr=5.5e-04  wdl(train_ce=0.0220 val_ce=0.4227 val_acc=94.0%)  decisive(top1 W=32.2% B=31.9% sign W=95.0% B=92.2%)
Epoch  30  train=1.3164 (v=0.0208 p=1.2843)  val=3.2900 (pow=0.2754 mse=0.2115 p=2.7888 mae=0.1669)  lr=5.3e-04  wdl(train_ce=0.0228 val_ce=0.4516 val_acc=94.3%)  decisive(top1 W=34.0% B=31.9% sign W=95.2% B=92.6%)

--- Test set evaluation ---
Total loss: 4.1685
Value power loss: 0.3613
Value true MSE:   0.2952
Policy CE:  3.5438
Value MAE:  0.2626
WDL CE:     0.5268
WDL Acc:    91.8%
Policy top-1 (enabled): W=36.7% B=26.2%
Winner sign (non-draw): W=92.3% B=90.8%
Winner prediction accuracy (non-draw): 91.8%

Best model saved to models\candidates\fresh_start_v18_detox\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_detox\train_run_20260717_025553.json
```
[07:30:33] TRACE 0015 record=14 raw_white_value candidate=-1.000 v17=+0.993 (informational)
[07:30:33] TRACE 0015 record=22 raw_white_value candidate=-1.000 v17=+0.985 (informational)
[07:30:33] TRACE 0015 record=24 raw_white_value candidate=-1.000 v17=+0.942 (informational)
[07:30:34] Clean eval set: 82/82 games held out by both splits
[07:30:34] START process-clean-eval: src/data_processor.py --raw-dir data\raw\eval_clean_v13v16 --output-dir data\processed\eval_clean_v13v16 --seed 42 --channels 15 --value-discount-mode near_mate --value-horizon 10 --value-floor 0.97
[07:30:40] END process-clean-eval (exit=0, 0.1 min)
```
Retention summary: kept_games=82/82, kept_positions=4371/4371
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=65, val=7, test=10
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\eval_clean_v13v16:
  positions.npy:    (8742, 8, 8, 15)
  mcts_values.npy:  (8742,)
  game_results.npy: (8742,)
  policies.npy:     (8742, 4096)
  policy_weights.npy: (8742,) (masked=28)
  splits.npz:       train=6610, val=708, test=1424
  split_game_ids.json: game-level split membership saved
```
[07:30:40] START model-diff-gate-clean: tools/model_diff.py --candidate models\candidates\fresh_start_v18_detox\best_value_net.pt --incumbent models\fresh_start_v17\best_value_net.pt --data-dir data\processed\eval_clean_v13v16 --split all --max-positions 8192 --enforce
[07:31:00] END model-diff-gate-clean (exit=0, 0.3 min)
```
=== model_diff: 8192 all positions from data\processed\eval_clean_v13v16 ===
metric                  candidate  incumbent    delta
policy_ce                  3.0090     2.4171  +0.5919
policy_top1                0.3250     0.2893  +0.0358
policy_top1_white          0.3536     0.3012  +0.0525
policy_top1_black          0.2671     0.2652  +0.0019
sign_acc                   0.9288     0.9006  +0.0282
sign_acc_white             0.9247     0.9113  +0.0133
sign_acc_black             0.9372     0.8792  +0.0580
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\model_diff_20260717_073100.json
MODEL DIFF: PASS
```
[07:31:00] START model-diff-standard-informational: tools/model_diff.py --candidate models\candidates\fresh_start_v18_detox\best_value_net.pt --incumbent models\fresh_start_v17\best_value_net.pt --data-dir data\processed\combined_v16
[07:31:13] END model-diff-standard-informational (exit=0, 0.2 min)
```
=== model_diff: 4096 test positions from data\processed\combined_v16 ===
metric                  candidate  incumbent    delta
policy_ce                  3.3421     2.3287  +1.0135
policy_top1                0.3208     0.3516  -0.0308
policy_top1_white          0.3637     0.3405  +0.0232
policy_top1_black          0.2518     0.3695  -0.1178
sign_acc                   0.9167     0.9192  -0.0024
sign_acc_white             0.9179     0.9255  -0.0077
sign_acc_black             0.9148     0.9081  +0.0067
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\model_diff_20260717_073112.json
  FAIL policy_top1: 0.3208 vs incumbent 0.3516 (drop 0.0308 > margin 0.01)
  FAIL policy_top1_black: 0.2518 vs incumbent 0.3695 (drop 0.1178 > margin 0.01)
MODEL DIFF: FAIL (informational)
```
[07:31:13] START anchor-20: tools/match.py --model-a models\candidates\fresh_start_v18_detox\best_value_net.pt --games 20 --sims 400 --workers 6
[07:47:13] END anchor-20 (exit=0, 16.0 min)
```
  "a_as_white": {
    "games": 10,
    "wins": 4,
    "losses": 6,
    "draws": 0,
    "score": 0.4,
    "mean_plies": 181.7,
    "mean_plies_when_won": 116.75,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 2,
    "losses": 8,
    "draws": 0,
    "score": 0.2,
    "mean_plies": 98.5,
    "mean_plies_when_won": 184.5,
    "mean_plies_when_lost": 77.0
  },
  "elapsed_sec": 960.3,
  "timestamp": "2026-07-17T07:47:13"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\match_fresh_start_v18_detox_vs_heuristic_20260717_074713.json
```
[07:47:13] START match-vs-v17: tools/match.py --model-a models\candidates\fresh_start_v18_detox\best_value_net.pt --model-b models\fresh_start_v17\best_value_net.pt --games 20 --sims 400 --workers 6
