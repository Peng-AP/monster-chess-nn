
# FINETUNE — started 2026-07-19 03:15:01

[07-19 03:37:01] sweep h2h finished — machine is free
[07-19 03:37:01] START process-nearmate: src/data_processor.py --raw-dir data\raw\combined_v16 --output-dir data\processed\combined_v16 --seed 42 --channels 15 --value-discount-mode near_mate --value-horizon 10 --value-floor 0.97
[07-19 03:38:29] END process-nearmate (exit=0, 1.5 min)
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
[07-19 03:38:29] START finetune-1ep: src/train.py --data-dir data\processed\combined_v16 --model-dir models\candidates\fresh_start_v18_ft1 --resume-from models\rejected\fresh_start_v18_ramp\best_value_net.pt --target game_result --value-head scalar --select-metric decisive --epochs 1 --lr 0.0002 --warmup-epochs 0 --seed 42
[07-19 03:48:32] END finetune-1ep (exit=0, 10.1 min)
```
Side-specialized heads: False
Value head mode: scalar (wdl_head=False)
Resumed weights from models\rejected\fresh_start_v18_ramp\best_value_net.pt: loaded 122 tensors, skipped 0 incompatible
Optimizer: AdamW (weight_decay=0.0001)
  Param groups: decay=23, no_decay=42
Epoch   1  train=1.3056 (v=0.0173 p=1.2883)  val=2.8553 (pow=0.2247 mse=0.1837 p=2.6306 mae=0.1656)  lr=2.0e-04  decisive(top1 W=34.4% B=32.1% sign W=94.4% B=92.8%)
  -> saved best model (decisive_score=1.2495)

--- Test set evaluation ---
Total loss: 3.3019
Value power loss: 0.2586
Value true MSE:   0.2204
Policy CE:  3.0432
Value MAE:  0.2203
Policy top-1 (enabled): W=35.6% B=27.1%
Winner sign (non-draw): W=93.2% B=89.8%
Winner prediction accuracy (non-draw): 91.9%

Best model saved to models\candidates\fresh_start_v18_ft1\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_ft1\train_run_20260719_033834.json
```
[07-19 03:48:35] TRACE 0015 ft1: -0.970 -0.981 -0.993 (ramp was saturated ~-1.0; graded negative is the goal)
[07-19 03:48:35] START match-ft1_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_ft1\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\finetune\ft1_anchor
[07-19 03:56:44] END match-ft1_anchor (exit=0, 8.2 min)
```
    "draws": 0,
    "score": 1.0,
    "mean_plies": 84.8,
    "mean_plies_when_won": 84.8,
    "mean_plies_when_lost": null
  },
  "a_as_black": {
    "games": 10,
    "wins": 2,
    "losses": 8,
    "draws": 0,
    "score": 0.2,
    "mean_plies": 95.2,
    "mean_plies_when_won": 154.5,
    "mean_plies_when_lost": 80.38
  },
  "elapsed_sec": 489.0,
  "timestamp": "2026-07-19T03:56:44"
}
Saved to benchmarks\finetune\ft1_anchor\match_fresh_start_v18_ft1_vs_heuristic_20260719_035644.json
```
[07-19 03:56:44] SCORE ft1_anchor: {'overall': 0.6, 'white': 1.0, 'black': 0.2}
[07-19 03:56:44] START finetune-3ep: src/train.py --data-dir data\processed\combined_v16 --model-dir models\candidates\fresh_start_v18_ft3 --resume-from models\rejected\fresh_start_v18_ramp\best_value_net.pt --target game_result --value-head scalar --select-metric decisive --epochs 3 --lr 0.0002 --warmup-epochs 0 --seed 42
[07-19 04:24:49] END finetune-3ep (exit=0, 28.1 min)
```
  Param groups: decay=23, no_decay=42
Epoch   1  train=1.3056 (v=0.0173 p=1.2883)  val=2.8553 (pow=0.2247 mse=0.1837 p=2.6306 mae=0.1656)  lr=2.0e-04  decisive(top1 W=34.4% B=32.1% sign W=94.4% B=92.8%)
  -> saved best model (decisive_score=1.2495)
Epoch   2  train=1.2930 (v=0.0143 p=1.2787)  val=2.9089 (pow=0.2219 mse=0.1783 p=2.6870 mae=0.1563)  lr=1.9e-04  decisive(top1 W=34.5% B=31.9% sign W=94.8% B=93.3%)
  -> saved best model (decisive_score=1.2515)
Epoch   3  train=1.2889 (v=0.0131 p=1.2758)  val=2.9226 (pow=0.2291 mse=0.1821 p=2.6935 mae=0.1525)  lr=1.8e-04  decisive(top1 W=35.9% B=32.3% sign W=94.8% B=92.9%)
  -> saved best model (decisive_score=1.2520)

--- Test set evaluation ---
Total loss: 3.4224
Value power loss: 0.3013
Value true MSE:   0.2506
Policy CE:  3.1211
Value MAE:  0.2287
Policy top-1 (enabled): W=36.3% B=26.7%
Winner sign (non-draw): W=92.4% B=89.4%
Winner prediction accuracy (non-draw): 91.3%

Best model saved to models\candidates\fresh_start_v18_ft3\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_ft3\train_run_20260719_035649.json
```
[07-19 04:24:49] TRACE 0015 ft3: -0.982 -0.986 -0.990 (ramp was saturated ~-1.0; graded negative is the goal)
[07-19 04:24:49] START match-ft3_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_ft3\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\finetune\ft3_anchor
[07-19 04:37:21] END match-ft3_anchor (exit=0, 12.5 min)
```
    "draws": 0,
    "score": 0.8,
    "mean_plies": 103.1,
    "mean_plies_when_won": 75.62,
    "mean_plies_when_lost": 213.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 3,
    "losses": 7,
    "draws": 0,
    "score": 0.3,
    "mean_plies": 137.6,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 100.14
  },
  "elapsed_sec": 751.3,
  "timestamp": "2026-07-19T04:37:21"
}
Saved to benchmarks\finetune\ft3_anchor\match_fresh_start_v18_ft3_vs_heuristic_20260719_043721.json
```
[07-19 04:37:21] SCORE ft3_anchor: {'overall': 0.55, 'white': 0.8, 'black': 0.3}
[07-19 04:37:21] START finetune-8ep: src/train.py --data-dir data\processed\combined_v16 --model-dir models\candidates\fresh_start_v18_ft8 --resume-from models\rejected\fresh_start_v18_ramp\best_value_net.pt --target game_result --value-head scalar --select-metric decisive --epochs 8 --lr 0.0002 --warmup-epochs 0 --seed 42
[07-19 05:50:47] END finetune-8ep (exit=0, 73.4 min)
```
Epoch   3  train=1.2889 (v=0.0131 p=1.2758)  val=2.9226 (pow=0.2291 mse=0.1821 p=2.6935 mae=0.1525)  lr=1.8e-04  decisive(top1 W=35.9% B=32.3% sign W=94.8% B=92.9%)
  -> saved best model (decisive_score=1.2520)
Epoch   4  train=1.2864 (v=0.0124 p=1.2740)  val=2.9212 (pow=0.2155 mse=0.1744 p=2.7058 mae=0.1525)  lr=1.7e-04  decisive(top1 W=33.7% B=31.6% sign W=95.0% B=93.0%)
Epoch   5  train=1.2848 (v=0.0122 p=1.2725)  val=2.9474 (pow=0.2335 mse=0.1873 p=2.7140 mae=0.1574)  lr=1.6e-04  decisive(top1 W=34.7% B=31.6% sign W=94.4% B=92.9%)
Epoch   6  train=1.2829 (v=0.0114 p=1.2715)  val=2.9813 (pow=0.2346 mse=0.1867 p=2.7468 mae=0.1541)  lr=1.5e-04  decisive(top1 W=34.7% B=31.9% sign W=94.7% B=92.7%)
Epoch   7  train=1.2817 (v=0.0110 p=1.2707)  val=2.9900 (pow=0.2190 mse=0.1771 p=2.7709 mae=0.1546)  lr=1.5e-04  decisive(top1 W=35.8% B=31.7% sign W=94.8% B=93.0%)
Epoch   8  train=1.2807 (v=0.0107 p=1.2701)  val=3.0130 (pow=0.2253 mse=0.1804 p=2.7877 mae=0.1526)  lr=1.4e-04  decisive(top1 W=34.9% B=32.1% sign W=94.7% B=93.0%)

--- Test set evaluation ---
Total loss: 3.4224
Value power loss: 0.3013
Value true MSE:   0.2506
Policy CE:  3.1211
Value MAE:  0.2287
Policy top-1 (enabled): W=36.3% B=26.7%
Winner sign (non-draw): W=92.4% B=89.4%
Winner prediction accuracy (non-draw): 91.3%

Best model saved to models\candidates\fresh_start_v18_ft8\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_ft8\train_run_20260719_043725.json
```
[07-19 05:50:48] TRACE 0015 ft8: -0.982 -0.986 -0.990 (ramp was saturated ~-1.0; graded negative is the goal)
[07-19 05:50:48] START match-ft8_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_ft8\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\finetune\ft8_anchor
[07-19 06:03:47] END match-ft8_anchor (exit=0, 13.0 min)
```
    "draws": 0,
    "score": 0.8,
    "mean_plies": 103.1,
    "mean_plies_when_won": 75.62,
    "mean_plies_when_lost": 213.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 3,
    "losses": 7,
    "draws": 0,
    "score": 0.3,
    "mean_plies": 137.6,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 100.14
  },
  "elapsed_sec": 778.4,
  "timestamp": "2026-07-19T06:03:47"
}
Saved to benchmarks\finetune\ft8_anchor\match_fresh_start_v18_ft8_vs_heuristic_20260719_060347.json
```
[07-19 06:03:47] SCORE ft8_anchor: {'overall': 0.55, 'white': 0.8, 'black': 0.3}
[07-19 06:03:47] DOSE CURVE: {1: {'overall': 0.6, 'white': 1.0, 'black': 0.2}, 3: {'overall': 0.55, 'white': 0.8, 'black': 0.3}, 8: {'overall': 0.55, 'white': 0.8, 'black': 0.3}} — best ft1
[07-19 06:03:47] START diff-ft1-informational: tools/model_diff.py --candidate models\candidates\fresh_start_v18_ft1\best_value_net.pt --incumbent models\fresh_start_v17\best_value_net.pt --data-dir data\processed\eval_clean_v13v16 --split all --max-positions 8192
[07-19 06:04:07] END diff-ft1-informational (exit=0, 0.3 min)
```
=== model_diff: 8192 all positions from data\processed\eval_clean_v13v16 ===
metric                  candidate  incumbent    delta
policy_ce                  2.6255     2.4171  +0.2083
policy_top1                0.3225     0.2893  +0.0332
policy_top1_white          0.3454     0.3012  +0.0442
policy_top1_black          0.2760     0.2652  +0.0108
sign_acc                   0.9095     0.9006  +0.0089
sign_acc_white             0.9172     0.9113  +0.0059
sign_acc_black             0.8942     0.8792  +0.0151
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\model_diff_20260719_060407.json
MODEL DIFF: PASS
```
[07-19 06:04:07] START match-ft1_h2h: tools/match.py --model-a models\candidates\fresh_start_v18_ft1\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\finetune\ft1_h2h --model-b models\fresh_start_v17\best_value_net.pt
[07-19 06:54:37] END match-ft1_h2h (exit=0, 50.5 min)
```
    "draws": 0,
    "score": 0.6,
    "mean_plies": 121.5,
    "mean_plies_when_won": 52.5,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 5,
    "losses": 5,
    "draws": 0,
    "score": 0.5,
    "mean_plies": 179.1,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 133.2
  },
  "elapsed_sec": 3029.4,
  "timestamp": "2026-07-19T06:54:37"
}
Saved to benchmarks\finetune\ft1_h2h\match_fresh_start_v18_ft1_vs_fresh_start_v17_20260719_065437.json
```
[07-19 06:54:37] SCORE ft1_h2h: {'overall': 0.55, 'white': 0.6, 'black': 0.5}
[07-19 06:54:37] ALL STEPS COMPLETE — best dose ft1 awaits owner play (models/candidates/fresh_start_v18_ft1)
