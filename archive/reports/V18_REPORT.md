
# v18 general learning cleanup - started 2026-07-15 22:58:35

[22:58:49] Merged v18 corpus: {"blackfocus_games": 300, "blackfocus_results": {"black": 251, "draw": 0, "white": 49}, "demo_games": 286, "demo_games_dropped": 14, "heuristic_games": 800, "human_duplication": 1, "human_games": 77, "position_encoding_channels": 17, "promotion_specific_games": 0, "recipe": "general learning cleanup; no promotion injection", "sha256": "43f26c42a2cf68b28803b70c824c3af047f83a5e99816bec465d706586b09bc1", "value_discount_mode": "near_mate", "value_floor": 0.97, "value_horizon": 10, "whitefocus_games": 250, "whitefocus_results": {"black": 30, "draw": 0, "white": 220}}
[22:58:49] START pretrain-audit: tools/pretrain_check.py data\raw\combined_v14 --reference data\raw\combined_v12 --min-purity 0 --max-dup 2 --value-discount-mode near_mate --value-horizon 10 --value-floor 0.97 --bias-fail 0.25
[22:58:52] END pretrain-audit (exit=0, 0.1 min)
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
  OK  label bias: mean |target| white-won 0.976 vs black-won 0.973 (rel gap 0.4%, max 25%)
PRETRAIN CHECK: PASS
```
[22:58:52] START process: src/data_processor.py --raw-dir data\raw\combined_v14 --output-dir data\processed\combined_v14 --seed 42 --value-discount-mode near_mate --value-horizon 10 --value-floor 0.97
[22:59:53] END process (exit=0, 1.0 min)
```
Retention summary: kept_games=1429/1713, kept_positions=65446/65745
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1143, val=142, test=144
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\combined_v14:
  positions.npy:    (130892, 8, 8, 17)
  mcts_values.npy:  (130892,)
  game_results.npy: (130892,)
  policies.npy:     (130892, 4096)
  policy_weights.npy: (130892,) (masked=1920)
  splits.npz:       train=104638, val=12728, test=13526
  split_game_ids.json: game-level split membership saved
```
[22:59:53] START train: src/train.py --data-dir data\processed\combined_v14 --model-dir models\fresh_start_v18 --target game_result --value-head wdl --epochs 30 --seed 42
[01:30:27] END train (exit=0, 150.6 min)
```
Epoch   9  train=1.6507 (v=0.0667 p=1.5453)  val=2.7479 (pow=0.2423 mse=0.2162 p=2.4129 mae=0.2471)  lr=1.5e-03  wdl(train_ce=0.0775 val_ce=0.1852 val_acc=93.1%)
Epoch  10  train=1.6151 (v=0.0588 p=1.5208)  val=2.9754 (pow=0.3159 mse=0.2717 p=2.4930 mae=0.2775)  lr=1.5e-03  wdl(train_ce=0.0709 val_ce=0.3329 val_acc=91.2%)
Epoch  11  train=1.5882 (v=0.0535 p=1.5010)  val=2.7877 (pow=0.2396 mse=0.2098 p=2.4362 mae=0.2352)  lr=1.4e-03  wdl(train_ce=0.0675 val_ce=0.2238 val_acc=93.1%)
Epoch  12  train=1.5661 (v=0.0484 p=1.4856)  val=2.8658 (pow=0.2429 mse=0.2121 p=2.4992 mae=0.2355)  lr=1.3e-03  wdl(train_ce=0.0643 val_ce=0.2473 val_acc=93.0%)
Epoch  13  train=1.5489 (v=0.0438 p=1.4745)  val=2.8801 (pow=0.2584 mse=0.2214 p=2.4887 mae=0.2356)  lr=1.3e-03  wdl(train_ce=0.0612 val_ce=0.2660 val_acc=93.3%)
Epoch  14  train=1.5335 (v=0.0406 p=1.4634)  val=2.7583 (pow=0.2586 mse=0.2201 p=2.3597 mae=0.2332)  lr=1.2e-03  wdl(train_ce=0.0590 val_ce=0.2800 val_acc=93.5%)
Epoch  15  train=1.5234 (v=0.0388 p=1.4556)  val=2.6819 (pow=0.2361 mse=0.2062 p=2.3342 mae=0.2355)  lr=1.1e-03  wdl(train_ce=0.0579 val_ce=0.2233 val_acc=93.7%)
Epoch  16  train=1.5121 (v=0.0365 p=1.4475)  val=2.7479 (pow=0.2395 mse=0.2042 p=2.3700 mae=0.2220)  lr=1.1e-03  wdl(train_ce=0.0561 val_ce=0.2768 val_acc=94.8%)
Epoch  17  train=1.5014 (v=0.0339 p=1.4401)  val=2.8858 (pow=0.2646 mse=0.2279 p=2.4944 mae=0.2394)  lr=1.0e-03  wdl(train_ce=0.0549 val_ce=0.2535 val_acc=93.6%)
Epoch  18  train=1.4945 (v=0.0326 p=1.4350)  val=2.9707 (pow=0.3039 mse=0.2533 p=2.4716 mae=0.2460)  lr=9.8e-04  wdl(train_ce=0.0538 val_ce=0.3903 val_acc=92.3%)
Early stopping at epoch 18

--- Test set evaluation ---
Total loss: 2.8028
Value power loss: 0.2970
Value true MSE:   0.2667
Policy CE:  2.3808
Value MAE:  0.2897
WDL CE:     0.2499
WDL Acc:    89.6%
Winner prediction accuracy (non-draw): 89.6%

Best model saved to models\fresh_start_v18\best_value_net.pt
Run metadata saved to models\fresh_start_v18\train_run_20260715_225958.json
```
[01:30:43] D2 White-result: n=3984 avg=+0.589 (informational)
[01:30:43] D2 Black-result: n=9542 avg=-0.797 (informational)
[01:30:43] TRACE 0015 record=14 raw_white_value v18=-0.789 v17=+0.993 (informational)
[01:30:43] TRACE 0015 record=22 raw_white_value v18=-0.889 v17=+0.985 (informational)
[01:30:43] TRACE 0015 record=24 raw_white_value v18=-0.918 v17=+0.942 (informational)
[01:30:43] START anchor-v18: tools/match.py --model-a models\fresh_start_v18\best_value_net.pt --games 6 --sims 400 --workers 6
[01:35:34] END anchor-v18 (exit=0, 4.8 min)
```
  "a_as_white": {
    "games": 3,
    "wins": 3,
    "losses": 0,
    "draws": 0,
    "score": 1.0,
    "mean_plies": 79.0,
    "mean_plies_when_won": 79.0,
    "mean_plies_when_lost": null
  },
  "a_as_black": {
    "games": 3,
    "wins": 0,
    "losses": 3,
    "draws": 0,
    "score": 0.0,
    "mean_plies": 104.0,
    "mean_plies_when_won": null,
    "mean_plies_when_lost": 104.0
  },
  "elapsed_sec": 290.4,
  "timestamp": "2026-07-16T01:35:34"
}
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\match_fresh_start_v18_vs_heuristic_20260716_013534.json
```
[01:35:34] START match-v18-v17: tools/match.py --model-a models\fresh_start_v18\best_value_net.pt --model-b models\fresh_start_v17\best_value_net.pt --games 20 --sims 400 --workers 6
