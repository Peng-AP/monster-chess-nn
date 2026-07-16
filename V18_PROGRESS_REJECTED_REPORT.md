
# v18 general learning cleanup - started 2026-07-15 18:20:18

[18:20:42] Merged v18 corpus: {"blackfocus_games": 300, "blackfocus_results": {"black": 251, "draw": 0, "white": 49}, "demo_games": 286, "demo_games_dropped": 14, "heuristic_games": 800, "human_duplication": 1, "human_games": 77, "position_encoding_channels": 17, "promotion_specific_games": 0, "recipe": "general learning cleanup; no promotion injection", "sha256": "43f26c42a2cf68b28803b70c824c3af047f83a5e99816bec465d706586b09bc1", "value_discount_mode": "progress", "value_floor": 0.5, "value_horizon": 225, "whitefocus_games": 250, "whitefocus_results": {"black": 30, "draw": 0, "white": 220}}
[18:20:42] START pretrain-audit: tools/pretrain_check.py data\raw\combined_v14 --reference data\raw\combined_v12 --min-purity 0 --max-dup 2 --value-discount-mode progress --value-horizon 225 --value-floor 0.5 --bias-fail 0.25
[18:20:56] END pretrain-audit (exit=0, 0.2 min)
```
=== pretrain_check: data\raw\combined_v14 ===
total positions: 65745, sources: {'(root)': 37442, 'human_blackfocus': 15968, 'human_games': 1977, 'whitefocus': 10358}
  OK  purity human_blackfocus: 251/300 = 84% (black wins, min 0%)
  OK  purity whitefocus: 220/250 = 88% (white wins, min 0%)
  OK  human data: 1977 positions (3.0% of corpus), 1895 unique, effective duplication x1.0 (max x2)
  OK  phase open: B-win share 71% vs ref 73% (delta -2%)
  OK  phase mid: B-win share 70% vs ref 66% (delta +4%)
  OK  phase late: B-win share 25% vs ref 23% (delta +2%)
  OK  phase endg: B-win share 77% vs ref 76% (delta +1%)
  OK  label bias: mean |target| white-won 0.942 vs black-won 0.826 (rel gap 12.3%, max 25%)
PRETRAIN CHECK: PASS
```
[18:20:56] START process: src/data_processor.py --raw-dir data\raw\combined_v14 --output-dir data\processed\combined_v14 --seed 42 --value-discount-mode progress --value-horizon 225 --value-floor 0.5
[18:21:57] END process (exit=0, 1.0 min)
```
Retention summary: kept_games=1429/1713, kept_positions=65446/65745
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1143, val=142, test=144
  Processing positions (augment=True)...
  Value targets: full-game progress discount (factor 0.5 at 225 plies)

Saved to data\processed\combined_v14:
  positions.npy:    (130892, 8, 8, 17)
  mcts_values.npy:  (130892,)
  game_results.npy: (130892,)
  policies.npy:     (130892, 4096)
  policy_weights.npy: (130892,) (masked=1920)
  splits.npz:       train=104638, val=12728, test=13526
  split_game_ids.json: game-level split membership saved
```
[18:21:57] START train: src/train.py --data-dir data\processed\combined_v14 --model-dir models\fresh_start_v18 --target game_result --value-head hybrid --epochs 30 --seed 42
[20:25:38] END train (exit=0, 123.7 min)
```
Epoch   6  train=1.8119 (v=0.0693 p=1.7039)  val=2.7955 (pow=0.2844 mse=0.2767 p=2.3217 mae=0.3030)  lr=1.8e-03  wdl(train_ce=0.0773 val_ce=0.3787 val_acc=88.8%)
Epoch   7  train=1.7218 (v=0.0601 p=1.6283)  val=2.6798 (pow=0.1573 mse=0.2342 p=2.3721 mae=0.2800)  lr=1.7e-03  wdl(train_ce=0.0666 val_ce=0.3009 val_acc=93.4%)
Epoch   8  train=1.6539 (v=0.0514 p=1.5740)  val=2.7242 (pow=0.1500 mse=0.2141 p=2.4483 mae=0.2724)  lr=1.6e-03  wdl(train_ce=0.0571 val_ce=0.2519 val_acc=93.9%)
Epoch   9  train=1.6097 (v=0.0450 p=1.5399)  val=2.7523 (pow=0.1801 mse=0.2232 p=2.4165 mae=0.2680)  lr=1.5e-03  wdl(train_ce=0.0498 val_ce=0.3114 val_acc=93.2%)
Epoch  10  train=1.5771 (v=0.0395 p=1.5160)  val=2.8274 (pow=0.2029 mse=0.2436 p=2.4313 mae=0.2808)  lr=1.5e-03  wdl(train_ce=0.0433 val_ce=0.3862 val_acc=92.1%)
Epoch  11  train=1.5516 (v=0.0355 p=1.4967)  val=2.7379 (pow=0.1779 mse=0.2565 p=2.4011 mae=0.2964)  lr=1.4e-03  wdl(train_ce=0.0387 val_ce=0.3178 val_acc=92.1%)
Epoch  12  train=1.5303 (v=0.0315 p=1.4815)  val=2.8563 (pow=0.1719 mse=0.2390 p=2.5319 mae=0.2860)  lr=1.3e-03  wdl(train_ce=0.0346 val_ce=0.3049 val_acc=93.4%)
Epoch  13  train=1.5159 (v=0.0292 p=1.4706)  val=2.8055 (pow=0.1637 mse=0.2203 p=2.4813 mae=0.2600)  lr=1.3e-03  wdl(train_ce=0.0320 val_ce=0.3209 val_acc=93.7%)
Epoch  14  train=1.5039 (v=0.0271 p=1.4618)  val=2.7885 (pow=0.2180 mse=0.2462 p=2.3456 mae=0.2748)  lr=1.2e-03  wdl(train_ce=0.0299 val_ce=0.4498 val_acc=92.3%)
Epoch  15  train=1.4900 (v=0.0245 p=1.4522)  val=2.6704 (pow=0.1705 mse=0.2334 p=2.3548 mae=0.2777)  lr=1.1e-03  wdl(train_ce=0.0267 val_ce=0.2901 val_acc=93.2%)
Early stopping at epoch 15

--- Test set evaluation ---
Total loss: 2.7575
Value power loss: 0.2011
Value true MSE:   0.2903
Policy CE:  2.3941
Value MAE:  0.3297
WDL CE:     0.3245
WDL Acc:    88.1%
Winner prediction accuracy (non-draw): 88.2%

Best model saved to models\fresh_start_v18\best_value_net.pt
Run metadata saved to models\fresh_start_v18\train_run_20260715_182204.json
```
[20:25:54] D2 White-result: n=3984 avg=+0.603 (informational)
[20:25:54] D2 Black-result: n=9542 avg=-0.745 (informational)
[20:25:54] TRACE 0015 record=14 raw_white_value v18=-0.853 v17=+0.993 (informational)
[20:25:54] TRACE 0015 record=22 raw_white_value v18=-0.863 v17=+0.985 (informational)
[20:25:54] TRACE 0015 record=24 raw_white_value v18=-0.890 v17=+0.942 (informational)
[20:25:54] START anchor-v18: tools/match.py --model-a models\fresh_start_v18\best_value_net.pt --games 6 --sims 400 --workers 6
[20:32:56] END anchor-v18 (exit=0, 7.0 min)
```
  "a_as_white": {
    "games": 3,
    "wins": 2,
    "losses": 1,
    "draws": 0,
    "score": 0.6667,
    "mean_plies": 119.33,
    "mean_plies_when_won": 66.5,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 3,
    "wins": 0,
    "losses": 3,
    "draws": 0,
    "score": 0.0,
    "mean_plies": 57.0,
    "mean_plies_when_won": null,
    "mean_plies_when_lost": 57.0
  },
  "elapsed_sec": 422.3,
  "timestamp": "2026-07-15T20:32:56"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\match_fresh_start_v18_vs_heuristic_20260715_203256.json
```
[20:32:56] START match-v18-v17: tools/match.py --model-a models\fresh_start_v18\best_value_net.pt --model-b models\fresh_start_v17\best_value_net.pt --games 20 --sims 400 --workers 6
[21:34:47] END match-v18-v17 (exit=0, 61.8 min)
```
  "a_as_white": {
    "games": 10,
    "wins": 5,
    "losses": 5,
    "draws": 0,
    "score": 0.5,
    "mean_plies": 131.2,
    "mean_plies_when_won": 37.4,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 5,
    "losses": 4,
    "draws": 1,
    "score": 0.55,
    "mean_plies": 224.6,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 224.0
  },
  "elapsed_sec": 3710.3,
  "timestamp": "2026-07-15T21:34:47"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\match_fresh_start_v18_vs_fresh_start_v17_20260715_213447.json
```
[21:34:47] ALL STEPS COMPLETE
