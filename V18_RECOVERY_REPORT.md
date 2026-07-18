
# v18 recovery candidate — started 2026-07-16 03:17:27

[03:17:51] Merged recovery corpus: {"blackfocus": 227, "blackfocus_dropped": 73, "demos": 286, "demos_dropped": 14, "generated_black_runner_games": 0, "heuristic": 800, "human_duplication": 6, "human_games": 79, "position_encoding_channels": 15, "promo_white_runner_games": 160, "recipe": "v17 frozen recipe + new human games + decisive checkpoint selection", "sha256": "0c4b841484e77e3e72d94cd8f2241c9667cecf12df8956829cf2785dcd2669a8", "value_discount_mode": "near_mate", "value_floor": 0.97, "value_horizon": 10, "whitefocus": 220, "whitefocus_dropped": 30}
[03:17:51] START pretrain-gate: tools/pretrain_check.py data\raw\combined_v15 --reference data\raw\combined_v13
[03:18:02] END pretrain-gate (exit=0, 0.2 min)
```
=== pretrain_check: data\raw\combined_v15 ===
total positions: 68016, sources: {'(root)': 37442, 'human_blackfocus': 8447, 'human_games': 10679, 'promo_races': 6103, 'whitefocus': 5345}
  OK  promo provenance: 6103 White-runner records, 0 generated Black-runner records
  OK  promo policy weights explicit on every Black position
  OK  promo policy weights match prevention outcomes
  OK  purity human_blackfocus: 227/227 = 100% (black wins, min 85%)
  OK  purity whitefocus: 220/220 = 100% (white wins, min 85%)
  OK  human data: 10679 positions (15.7% of corpus), 1961 unique, effective duplication x5.4 (max x8)
  OK  phase open: B-win share 74% vs ref 74% (delta +1%)
  OK  phase mid: B-win share 64% vs ref 64% (delta +0%)
  OK  phase late: B-win share 29% vs ref 29% (delta -0%)
  OK  phase endg: B-win share 73% vs ref 73% (delta +0%)
  OK  label bias: mean |target| white-won 0.977 vs black-won 0.973 (rel gap 0.4%, max 5%)
PRETRAIN CHECK: PASS
```
[03:18:02] START process: src/data_processor.py --raw-dir data\raw\combined_v15 --output-dir data\processed\combined_v15 --seed 42 --channels 15 --value-discount-mode near_mate --value-horizon 10 --value-floor 0.97
[03:19:04] END process (exit=0, 1.0 min)
```
Retention summary: kept_games=1472/1772, kept_positions=67686/68016
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1177, val=146, test=149
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\combined_v15:
  positions.npy:    (135372, 8, 8, 15)
  mcts_values.npy:  (135372,)
  game_results.npy: (135372,)
  policies.npy:     (135372, 4096)
  policy_weights.npy: (135372,) (masked=12722)
  splits.npz:       train=109358, val=11516, test=14498
  split_game_ids.json: game-level split membership saved
```
[03:19:04] START train: src/train.py --data-dir data\processed\combined_v15 --model-dir models\candidates\fresh_start_v18_recovery --target game_result --value-head wdl --select-metric decisive --epochs 30 --seed 42
[13:06:45] RESUME: rerunning train after 04:26 session-death; corpus/gate/process reused from the 03:17 launch
[13:06:45] START train: src/train.py --data-dir data\processed\combined_v15 --model-dir models\candidates\fresh_start_v18_recovery --target game_result --value-head wdl --select-metric decisive --epochs 30 --seed 42
[16:33:21] RESUME: rerunning train after 04:26 session-death; corpus/gate/process reused from the 03:17 launch
[16:33:21] START train: src/train.py --data-dir data\processed\combined_v15 --model-dir models\candidates\fresh_start_v18_recovery --target game_result --value-head wdl --select-metric decisive --epochs 30 --seed 42
[19:20:49] END train (exit=0, 167.5 min)
```
Epoch  11  train=1.4530 (v=0.0412 p=1.3933)  val=3.1956 (pow=0.3291 mse=0.2669 p=2.7196 mae=0.2273)  lr=1.4e-03  wdl(train_ce=0.0370 val_ce=0.2936 val_acc=91.5%)  decisive(top1 W=32.7% B=27.2% sign W=92.6% B=89.5%)
Epoch  12  train=1.4385 (v=0.0387 p=1.3821)  val=3.2014 (pow=0.3529 mse=0.2806 p=2.6777 mae=0.2265)  lr=1.3e-03  wdl(train_ce=0.0355 val_ce=0.3415 val_acc=91.4%)  decisive(top1 W=29.7% B=26.6% sign W=91.9% B=90.5%)
Epoch  13  train=1.4247 (v=0.0362 p=1.3712)  val=3.1703 (pow=0.3516 mse=0.2799 p=2.6385 mae=0.2253)  lr=1.3e-03  wdl(train_ce=0.0346 val_ce=0.3604 val_acc=91.2%)  decisive(top1 W=32.5% B=26.7% sign W=92.0% B=89.6%)
Epoch  14  train=1.4125 (v=0.0338 p=1.3621)  val=3.4372 (pow=0.4440 mse=0.3454 p=2.7120 mae=0.2590)  lr=1.2e-03  wdl(train_ce=0.0331 val_ce=0.5623 val_acc=89.3%)  decisive(top1 W=33.5% B=27.3% sign W=89.3% B=89.4%)
Epoch  15  train=1.4011 (v=0.0308 p=1.3546)  val=3.1884 (pow=0.3429 mse=0.2716 p=2.6757 mae=0.2185)  lr=1.1e-03  wdl(train_ce=0.0313 val_ce=0.3397 val_acc=91.2%)  decisive(top1 W=33.4% B=27.0% sign W=92.1% B=89.7%)
Epoch  16  train=1.3938 (v=0.0301 p=1.3484)  val=3.2597 (pow=0.3866 mse=0.3024 p=2.6733 mae=0.2351)  lr=1.1e-03  wdl(train_ce=0.0306 val_ce=0.3996 val_acc=90.9%)  decisive(top1 W=31.6% B=27.3% sign W=92.5% B=88.1%)
Epoch  17  train=1.3834 (v=0.0277 p=1.3416)  val=3.9340 (pow=0.6368 mse=0.4740 p=2.7673 mae=0.3043)  lr=1.0e-03  wdl(train_ce=0.0284 val_ce=1.0598 val_acc=86.7%)  decisive(top1 W=33.3% B=26.3% sign W=87.5% B=85.2%)
Epoch  18  train=1.3767 (v=0.0267 p=1.3359)  val=3.1772 (pow=0.3787 mse=0.2955 p=2.5787 mae=0.2294)  lr=9.8e-04  wdl(train_ce=0.0282 val_ce=0.4397 val_acc=91.0%)  decisive(top1 W=32.2% B=27.3% sign W=92.4% B=88.4%)
Early stopping at epoch 18

--- Test set evaluation ---
Total loss: 3.1125
Value power loss: 0.3201
Value true MSE:   0.2735
Policy CE:  2.6445
Value MAE:  0.2617
WDL CE:     0.2957
WDL Acc:    91.0%
Policy top-1 (enabled): W=29.0% B=24.5%
Winner sign (non-draw): W=91.3% B=90.2%
Winner prediction accuracy (non-draw): 91.0%

Best model saved to models\candidates\fresh_start_v18_recovery\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_recovery\train_run_20260716_163336.json
```
[19:20:52] TRACE 0015 record=14 raw_white_value candidate=-0.986 v17=+0.993 (informational)
[19:20:52] TRACE 0015 record=22 raw_white_value candidate=-0.994 v17=+0.985 (informational)
[19:20:52] TRACE 0015 record=24 raw_white_value candidate=-0.998 v17=+0.942 (informational)
[19:20:52] START model-diff-gate: tools/model_diff.py --candidate models\candidates\fresh_start_v18_recovery\best_value_net.pt --incumbent models\fresh_start_v17\best_value_net.pt --data-dir data\processed\combined_v15 --enforce
[19:21:04] END model-diff-gate (exit=1, 0.2 min)
```
=== model_diff: 4096 test positions from data\processed\combined_v15 ===
metric                  candidate  incumbent    delta
policy_ce                  2.5318     2.4198  +0.1120
policy_top1                0.2732     0.3199  -0.0467
policy_top1_white          0.2845     0.3313  -0.0468
policy_top1_black          0.2524     0.2987  -0.0464
sign_acc                   0.9094     0.9128  -0.0034
sign_acc_white             0.9127     0.9257  -0.0131
sign_acc_black             0.9033     0.8885  +0.0148
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\model_diff_20260716_192104.json
  FAIL policy_top1: 0.2732 vs incumbent 0.3199 (drop 0.0467 > margin 0.01)
  FAIL policy_top1_white: 0.2845 vs incumbent 0.3313 (drop 0.0468 > margin 0.01)
  FAIL policy_top1_black: 0.2524 vs incumbent 0.2987 (drop 0.0464 > margin 0.01)
  FAIL sign_acc_white: 0.9127 vs incumbent 0.9257 (drop 0.0131 > margin 0.01)
MODEL DIFF: FAIL — candidate regresses vs incumbent; do not spend match time on it.
```
[19:21:04] STDERR model-diff-gate:
```

```
[19:21:04] CHAIN ABORTED: RuntimeError('model-diff-gate failed (exit 1)')
[20:02:08] RECOVERY2: v17-faithful policy weights (mask 2340 == v17), gate on leakage-clean eval set
[20:02:08] START train-recovery2: src/train.py --data-dir data\processed\combined_v15_hpol --model-dir models\candidates\fresh_start_v18_recovery2 --target game_result --value-head wdl --select-metric decisive --epochs 30 --seed 42
[23:44:07] END train-recovery2 (exit=0, 222.0 min)
```
Epoch  14  train=1.4617 (v=0.0337 p=1.4117)  val=3.3838 (pow=0.4988 mse=0.3798 p=2.5681 mae=0.2659)  lr=1.2e-03  wdl(train_ce=0.0328 val_ce=0.6337 val_acc=88.7%)  decisive(top1 W=30.6% B=27.2% sign W=89.5% B=87.4%)
Epoch  15  train=1.4516 (v=0.0311 p=1.4048)  val=3.2853 (pow=0.4485 mse=0.3472 p=2.5860 mae=0.2558)  lr=1.1e-03  wdl(train_ce=0.0313 val_ce=0.5016 val_acc=89.6%)  decisive(top1 W=31.5% B=28.4% sign W=92.2% B=85.0%)
Epoch  16  train=1.4441 (v=0.0297 p=1.3991)  val=3.1954 (pow=0.3866 mse=0.3035 p=2.6008 mae=0.2362)  lr=1.1e-03  wdl(train_ce=0.0308 val_ce=0.4159 val_acc=90.6%)  decisive(top1 W=32.4% B=27.9% sign W=93.1% B=86.2%)
Epoch  17  train=1.4356 (v=0.0277 p=1.3933)  val=3.1714 (pow=0.3961 mse=0.3089 p=2.5621 mae=0.2339)  lr=1.0e-03  wdl(train_ce=0.0293 val_ce=0.4264 val_acc=90.3%)  decisive(top1 W=35.0% B=28.6% sign W=92.4% B=86.6%)
Epoch  18  train=1.4294 (v=0.0272 p=1.3879)  val=3.4081 (pow=0.4889 mse=0.3711 p=2.5613 mae=0.2617)  lr=9.8e-04  wdl(train_ce=0.0286 val_ce=0.7159 val_acc=88.9%)  decisive(top1 W=33.5% B=27.0% sign W=90.2% B=86.5%)
Epoch  19  train=1.4255 (v=0.0273 p=1.3836)  val=3.2740 (pow=0.4480 mse=0.3441 p=2.5330 mae=0.2508)  lr=9.3e-04  wdl(train_ce=0.0293 val_ce=0.5860 val_acc=89.7%)  decisive(top1 W=33.0% B=28.5% sign W=90.6% B=88.2%)
Epoch  20  train=1.4212 (v=0.0266 p=1.3803)  val=3.6442 (pow=0.5784 mse=0.4353 p=2.6450 mae=0.2887)  lr=8.8e-04  wdl(train_ce=0.0284 val_ce=0.8415 val_acc=87.1%)  decisive(top1 W=31.2% B=27.5% sign W=87.6% B=86.3%)
Epoch  21  train=1.4126 (v=0.0248 p=1.3746)  val=3.3994 (pow=0.4251 mse=0.3259 p=2.6712 mae=0.2390)  lr=8.4e-04  wdl(train_ce=0.0264 val_ce=0.6061 val_acc=90.4%)  decisive(top1 W=31.5% B=27.7% sign W=91.5% B=88.6%)
Early stopping at epoch 21

--- Test set evaluation ---
Total loss: 2.9934
Value power loss: 0.3021
Value true MSE:   0.2517
Policy CE:  2.5272
Value MAE:  0.2368
WDL CE:     0.3281
WDL Acc:    92.2%
Policy top-1 (enabled): W=31.8% B=24.5%
Winner sign (non-draw): W=92.3% B=92.1%
Winner prediction accuracy (non-draw): 92.2%

Best model saved to models\candidates\fresh_start_v18_recovery2\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_recovery2\train_run_20260716_200215.json
```
[23:44:11] TRACE 0015 record=14 raw_white_value candidate=-1.000 v17=+0.993 (informational)
[23:44:11] TRACE 0015 record=22 raw_white_value candidate=-1.000 v17=+0.985 (informational)
[23:44:11] TRACE 0015 record=24 raw_white_value candidate=-1.000 v17=+0.942 (informational)
[23:44:11] START model-diff-gate-clean: tools/model_diff.py --candidate models\candidates\fresh_start_v18_recovery2\best_value_net.pt --incumbent models\fresh_start_v17\best_value_net.pt --data-dir data\processed\eval_clean_v13v15 --split all --max-positions 8192 --enforce
[23:44:33] END model-diff-gate-clean (exit=1, 0.4 min)
```
=== model_diff: 8192 all positions from data\processed\eval_clean_v13v15 ===
metric                  candidate  incumbent    delta
policy_ce                  2.4489     2.4124  +0.0365
policy_top1                0.2966     0.2938  +0.0028
policy_top1_white          0.3178     0.3014  +0.0163
policy_top1_black          0.2541     0.2785  -0.0244
sign_acc                   0.9358     0.9152  +0.0206
sign_acc_white             0.9331     0.9247  +0.0084
sign_acc_black             0.9410     0.8963  +0.0448
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\model_diff_20260716_234433.json
  FAIL policy_top1_black: 0.2541 vs incumbent 0.2785 (drop 0.0244 > margin 0.01)
MODEL DIFF: FAIL — candidate regresses vs incumbent; do not spend match time on it.
```
[23:44:33] STDERR model-diff-gate-clean:
```

```
[23:44:33] CHAIN ABORTED: RuntimeError('model-diff-gate-clean failed (exit 1)')
[01:17:10] RECOVERY2: informational matches — clean gate missed only policy_top1_black (-2.4); value strongly up; play arbitrates
[01:17:10] START anchor-20-informational: tools/match.py --model-a models\candidates\fresh_start_v18_recovery2\best_value_net.pt --games 20 --sims 400 --workers 6
