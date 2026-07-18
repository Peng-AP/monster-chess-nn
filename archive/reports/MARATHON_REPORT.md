
# MARATHON — started 2026-07-17 08:19:04 (budget 29.0h)

[07-17 08:19:04] === STAGE A_ramp (budget 7.0h, 29.0h left) ===
[07-17 08:19:04] START process-ramp: src/data_processor.py --raw-dir data\raw\combined_v16 --output-dir data\processed\combined_v16_r50h60 --seed 42 --channels 15 --value-discount-mode near_mate --value-horizon 60 --value-floor 0.5
[07-17 08:21:13] END process-ramp (exit=0, 2.1 min)
```
Retention summary: kept_games=1457/1755, kept_positions=70478/70805
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1164, val=145, test=148
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.5 -> 1.0 over last 60 plies

Saved to data\processed\combined_v16_r50h60:
  positions.npy:    (140956, 8, 8, 15)
  mcts_values.npy:  (140956,)
  game_results.npy: (140956,)
  policies.npy:     (140956, 4096)
  policy_weights.npy: (140956,) (masked=13372)
  splits.npz:       train=112760, val=11964, test=16232
  split_game_ids.json: game-level split membership saved
```
[07-17 08:21:13] START train-ramp: src/train.py --data-dir data\processed\combined_v16_r50h60 --model-dir models\candidates\fresh_start_v18_ramp --target game_result --value-head scalar --select-metric decisive --epochs 30 --seed 42
[07-17 13:05:08] END train-ramp (exit=0, 283.9 min)
```
Epoch  23  train=1.3098 (v=0.0115 p=1.2983)  val=2.7287 (pow=0.1063 mse=0.1101 p=2.6224 mae=0.1922)  lr=7.5e-04  decisive(top1 W=35.4% B=31.4% sign W=94.9% B=93.2%)
Epoch  24  train=1.3063 (v=0.0107 p=1.2957)  val=2.7461 (pow=0.1114 mse=0.1114 p=2.6347 mae=0.1850)  lr=7.2e-04  decisive(top1 W=34.5% B=32.1% sign W=95.5% B=92.5%)
Epoch  25  train=1.3016 (v=0.0099 p=1.2917)  val=2.8242 (pow=0.1334 mse=0.1270 p=2.6908 mae=0.1909)  lr=6.8e-04  decisive(top1 W=35.6% B=31.0% sign W=94.9% B=93.0%)
Epoch  26  train=1.2987 (v=0.0095 p=1.2892)  val=2.8926 (pow=0.1315 mse=0.1268 p=2.7611 mae=0.1929)  lr=6.5e-04  decisive(top1 W=36.0% B=32.4% sign W=94.7% B=92.6%)
Epoch  27  train=1.2966 (v=0.0096 p=1.2870)  val=2.8775 (pow=0.1295 mse=0.1257 p=2.7480 mae=0.1939)  lr=6.1e-04  decisive(top1 W=35.6% B=31.7% sign W=94.9% B=92.2%)
Epoch  28  train=1.2938 (v=0.0090 p=1.2848)  val=3.1166 (pow=0.1137 mse=0.1148 p=3.0029 mae=0.1891)  lr=5.8e-04  decisive(top1 W=34.7% B=31.6% sign W=94.9% B=92.5%)
Epoch  29  train=1.2917 (v=0.0085 p=1.2832)  val=2.8889 (pow=0.1200 mse=0.1170 p=2.7688 mae=0.1849)  lr=5.5e-04  decisive(top1 W=35.9% B=31.9% sign W=95.7% B=92.8%)
Epoch  30  train=1.2899 (v=0.0086 p=1.2813)  val=2.8193 (pow=0.1251 mse=0.1216 p=2.6942 mae=0.1894)  lr=5.3e-04  decisive(top1 W=33.2% B=32.1% sign W=94.7% B=93.4%)
Early stopping at epoch 30

--- Test set evaluation ---
Total loss: 3.0393
Value power loss: 0.1320
Value true MSE:   0.1375
Policy CE:  2.9073
Value MAE:  0.2242
Policy top-1 (enabled): W=36.3% B=27.9%
Winner sign (non-draw): W=92.1% B=90.0%
Winner prediction accuracy (non-draw): 91.3%

Best model saved to models\candidates\fresh_start_v18_ramp\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_ramp\train_run_20260717_082119.json
```
[07-17 13:05:11] TRACE 0015 ramp: -0.678 -0.718 -0.783 (v17: +0.993 +0.985 +0.942; positions are lost for White)
[07-17 13:05:11] START diff-ramp-informational: tools/model_diff.py --candidate models\candidates\fresh_start_v18_ramp\best_value_net.pt --incumbent models\fresh_start_v17\best_value_net.pt --data-dir data\processed\eval_clean_v13v16 --split all --max-positions 8192
[07-17 13:05:32] END diff-ramp-informational (exit=0, 0.4 min)
```
=== model_diff: 8192 all positions from data\processed\eval_clean_v13v16 ===
metric                  candidate  incumbent    delta
policy_ce                  2.5004     2.4171  +0.0832
policy_top1                0.3298     0.2893  +0.0405
policy_top1_white          0.3569     0.3012  +0.0558
policy_top1_black          0.2749     0.2652  +0.0096
sign_acc                   0.8966     0.9006  -0.0040
sign_acc_white             0.8972     0.9113  -0.0141
sign_acc_black             0.8953     0.8792  +0.0162
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\model_diff_20260717_130532.json
  FAIL sign_acc_white: 0.8972 vs incumbent 0.9113 (drop 0.0141 > margin 0.01)
MODEL DIFF: FAIL (informational)
```
[07-17 13:05:32] START match-ramp_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_ramp\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\marathon\ramp_anchor
[07-17 13:23:01] END match-ramp_anchor (exit=0, 17.5 min)
```
    "wins": 10,
    "losses": 0,
    "draws": 0,
    "score": 1.0,
    "mean_plies": 56.3,
    "mean_plies_when_won": 56.3,
    "mean_plies_when_lost": null
  },
  "a_as_black": {
    "games": 10,
    "wins": 6,
    "losses": 4,
    "draws": 0,
    "score": 0.6,
    "mean_plies": 151.8,
    "mean_plies_when_won": 179.0,
    "mean_plies_when_lost": 111.0
  },
  "elapsed_sec": 1048.1,
  "timestamp": "2026-07-17T13:23:01"
}
Saved to benchmarks\marathon\ramp_anchor\match_fresh_start_v18_ramp_vs_heuristic_20260717_132301.json
```
[07-17 13:23:01] SCORE ramp_anchor: {'overall': 0.8, 'white': 1.0, 'black': 0.6}
[07-17 13:23:01] START match-ramp_h2h: tools/match.py --model-a models\candidates\fresh_start_v18_ramp\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\marathon\ramp_h2h --model-b models\fresh_start_v17\best_value_net.pt
[07-17 14:41:27] END match-ramp_h2h (exit=0, 78.4 min)
```
    "wins": 6,
    "losses": 4,
    "draws": 0,
    "score": 0.6,
    "mean_plies": 140.4,
    "mean_plies_when_won": 84.0,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 8,
    "losses": 2,
    "draws": 0,
    "score": 0.8,
    "mean_plies": 190.0,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 50.0
  },
  "elapsed_sec": 4706.5,
  "timestamp": "2026-07-17T14:41:27"
}
Saved to benchmarks\marathon\ramp_h2h\match_fresh_start_v18_ramp_vs_fresh_start_v17_20260717_144127.json
```
[07-17 14:41:27] SCORE ramp_h2h: {'overall': 0.7, 'white': 0.6, 'black': 0.8}
[07-17 14:41:27] === STAGE B_human_only (budget 1.5h, 22.6h left) ===
[07-17 14:41:28] human-only corpus: 85 games, duplication x1, winner-side policy only
[07-17 14:41:28] START process-human-only: src/data_processor.py --raw-dir data\raw\human_only_v1 --output-dir data\processed\human_only_v1 --seed 42 --channels 15 --value-discount-mode near_mate --value-horizon 10 --value-floor 0.97
[07-17 14:41:31] END process-human-only (exit=0, 0.1 min)
```
Retention summary: kept_games=85/85, kept_positions=2235/2235
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=67, val=8, test=10
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.97 -> 1.0 over last 10 plies

Saved to data\processed\human_only_v1:
  positions.npy:    (4470, 8, 8, 15)
  mcts_values.npy:  (4470,)
  game_results.npy: (4470,)
  policies.npy:     (4470, 4096)
  policy_weights.npy: (4470,) (masked=2176)
  splits.npz:       train=3568, val=410, test=492
  split_game_ids.json: game-level split membership saved
```
[07-17 14:41:31] START train-human-only: src/train.py --data-dir data\processed\human_only_v1 --model-dir models\experiments\human_only_v1 --target game_result --value-head wdl --select-metric decisive --epochs 30 --seed 42
[07-17 14:47:03] END train-human-only (exit=0, 5.5 min)
```
Epoch  13  train=0.2937 (v=0.0840 p=0.1762)  val=12.0037 (pow=3.2414 mse=2.3900 p=7.3336 mae=1.3378)  lr=1.3e-03  wdl(train_ce=0.0669 val_ce=2.8576 val_acc=30.2%)  decisive(top1 W=11.4% B=19.9% sign W=27.2% B=33.3%)
Epoch  14  train=0.2497 (v=0.0922 p=0.1202)  val=8.9855 (pow=0.5113 mse=0.3919 p=8.2596 mae=0.2697)  lr=1.2e-03  wdl(train_ce=0.0745 val_ce=0.4292 val_acc=88.3%)  decisive(top1 W=13.6% B=28.9% sign W=87.4% B=89.2%)
Epoch  15  train=0.2101 (v=0.0677 p=0.1139)  val=8.2163 (pow=0.9148 mse=0.7329 p=6.9585 mae=0.5242)  lr=1.1e-03  wdl(train_ce=0.0571 val_ce=0.6859 val_acc=72.9%)  decisive(top1 W=18.2% B=25.9% sign W=59.7% B=86.3%)
Epoch  16  train=0.2155 (v=0.0739 p=0.1130)  val=7.9249 (pow=0.6759 mse=0.5121 p=6.9397 mae=0.3365)  lr=1.1e-03  wdl(train_ce=0.0572 val_ce=0.6187 val_acc=85.6%)  decisive(top1 W=15.9% B=27.1% sign W=86.9% B=84.3%)
Epoch  17  train=0.1811 (v=0.0624 p=0.0933)  val=8.6929 (pow=0.7610 mse=0.5834 p=7.6297 mae=0.3857)  lr=1.0e-03  wdl(train_ce=0.0507 val_ce=0.6045 val_acc=82.2%)  decisive(top1 W=15.9% B=33.7% sign W=90.3% B=74.0%)
Epoch  18  train=0.1790 (v=0.0606 p=0.0936)  val=8.4474 (pow=0.6340 mse=0.4854 p=7.5450 mae=0.3257)  lr=9.8e-04  wdl(train_ce=0.0497 val_ce=0.5369 val_acc=84.4%)  decisive(top1 W=20.5% B=27.7% sign W=81.1% B=87.7%)
Early stopping at epoch 18

--- Test set evaluation ---
Total loss: 6.2658
Value power loss: 0.4365
Value true MSE:   0.3513
Policy CE:  5.6865
Value MAE:  0.2671
WDL CE:     0.2856
WDL Acc:    87.8%
Policy top-1 (enabled): W=10.0% B=34.2%
Winner sign (non-draw): W=86.3% B=89.3%
Winner prediction accuracy (non-draw): 87.8%

Best model saved to models\experiments\human_only_v1\best_value_net.pt
Run metadata saved to models\experiments\human_only_v1\train_run_20260717_144135.json
```
[07-17 14:47:04] TRACE 0015 human_only: +0.182 +0.512 +0.277 (v17: +0.993 +0.985 +0.942; positions are lost for White)
[07-17 14:47:04] START match-human_only_anchor: tools/match.py --model-a models\experiments\human_only_v1\best_value_net.pt --games 10 --sims 400 --workers 6 --out-dir benchmarks\marathon\human_only_anchor
[07-17 14:55:25] END match-human_only_anchor (exit=0, 8.4 min)
```
    "wins": 4,
    "losses": 1,
    "draws": 0,
    "score": 0.8,
    "mean_plies": 77.2,
    "mean_plies_when_won": 40.25,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 5,
    "wins": 1,
    "losses": 4,
    "draws": 0,
    "score": 0.2,
    "mean_plies": 117.4,
    "mean_plies_when_won": 96.0,
    "mean_plies_when_lost": 122.75
  },
  "elapsed_sec": 501.5,
  "timestamp": "2026-07-17T14:55:25"
}
Saved to benchmarks\marathon\human_only_anchor\match_human_only_v1_vs_heuristic_20260717_145525.json
```
[07-17 14:55:25] SCORE human_only_anchor: {'overall': 0.5, 'white': 0.8, 'black': 0.2}
[07-17 14:55:25] === STAGE C_capacity (budget 11.0h, 22.4h left) ===
[07-17 14:55:25] capacity uses scalar head on data\processed\combined_v16_r50h60 (ramp anchor=0.8, baseline=0.3)
[07-17 14:55:25] START train-cap: src/train.py --data-dir data\processed\combined_v16_r50h60 --model-dir models\candidates\fresh_start_v18_cap --target game_result --value-head scalar --select-metric decisive --epochs 24 --seed 42 --stem-channels 96 --res-channels 96,96,128,128,160,160,160,160
[07-17 20:03:07] END train-cap (exit=0, 307.7 min)
```
Epoch  17  train=1.3353 (v=0.0152 p=1.3201)  val=2.7093 (pow=0.1395 mse=0.1368 p=2.5698 mae=0.2061)  lr=1.0e-03  decisive(top1 W=35.9% B=29.8% sign W=93.2% B=92.3%)
Epoch  18  train=1.3300 (v=0.0140 p=1.3160)  val=2.6483 (pow=0.1718 mse=0.1605 p=2.4764 mae=0.2165)  lr=9.8e-04  decisive(top1 W=36.4% B=30.3% sign W=94.3% B=88.8%)
Epoch  19  train=1.3230 (v=0.0131 p=1.3098)  val=2.6540 (pow=0.1241 mse=0.1220 p=2.5299 mae=0.1916)  lr=9.3e-04  decisive(top1 W=38.0% B=31.3% sign W=95.2% B=92.2%)
Epoch  20  train=1.3177 (v=0.0123 p=1.3054)  val=2.6555 (pow=0.1222 mse=0.1238 p=2.5333 mae=0.2025)  lr=8.8e-04  decisive(top1 W=34.7% B=30.6% sign W=94.2% B=92.5%)
Epoch  21  train=1.3144 (v=0.0119 p=1.3025)  val=2.7240 (pow=0.1310 mse=0.1299 p=2.5931 mae=0.2049)  lr=8.4e-04  decisive(top1 W=34.0% B=31.9% sign W=95.2% B=91.4%)
Epoch  22  train=1.3117 (v=0.0121 p=1.2995)  val=2.8287 (pow=0.1480 mse=0.1412 p=2.6807 mae=0.2033)  lr=7.9e-04  decisive(top1 W=33.1% B=31.8% sign W=93.8% B=91.2%)
Epoch  23  train=1.3062 (v=0.0108 p=1.2953)  val=2.7834 (pow=0.1374 mse=0.1345 p=2.6460 mae=0.2053)  lr=7.5e-04  decisive(top1 W=34.6% B=33.5% sign W=93.9% B=92.3%)
  -> saved best model (decisive_score=1.2585)
Epoch  24  train=1.3032 (v=0.0103 p=1.2929)  val=2.7542 (pow=0.1377 mse=0.1333 p=2.6165 mae=0.1986)  lr=7.2e-04  decisive(top1 W=37.9% B=31.6% sign W=94.0% B=92.3%)

--- Test set evaluation ---
Total loss: 3.1766
Value power loss: 0.1421
Value true MSE:   0.1450
Policy CE:  3.0344
Value MAE:  0.2263
Policy top-1 (enabled): W=38.1% B=27.4%
Winner sign (non-draw): W=91.1% B=90.8%
Winner prediction accuracy (non-draw): 91.0%

Best model saved to models\candidates\fresh_start_v18_cap\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_cap\train_run_20260717_145533.json
```
[07-17 20:03:07] TRACE 0015 cap: -0.673 -0.727 -0.750 (v17: +0.993 +0.985 +0.942; positions are lost for White)
[07-17 20:03:07] START match-cap_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_cap\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\marathon\cap_anchor
[07-17 20:15:10] END match-cap_anchor (exit=0, 12.0 min)
```
    "wins": 10,
    "losses": 0,
    "draws": 0,
    "score": 1.0,
    "mean_plies": 37.4,
    "mean_plies_when_won": 37.4,
    "mean_plies_when_lost": null
  },
  "a_as_black": {
    "games": 10,
    "wins": 3,
    "losses": 7,
    "draws": 0,
    "score": 0.3,
    "mean_plies": 109.1,
    "mean_plies_when_won": 171.0,
    "mean_plies_when_lost": 82.57
  },
  "elapsed_sec": 722.2,
  "timestamp": "2026-07-17T20:15:10"
}
Saved to benchmarks\marathon\cap_anchor\match_fresh_start_v18_cap_vs_heuristic_20260717_201510.json
```
[07-17 20:15:10] SCORE cap_anchor: {'overall': 0.65, 'white': 1.0, 'black': 0.3}
[07-17 20:15:10] START match-cap_h2h: tools/match.py --model-a models\candidates\fresh_start_v18_cap\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\marathon\cap_h2h --model-b models\fresh_start_v17\best_value_net.pt
[07-17 21:10:39] END match-cap_h2h (exit=0, 55.5 min)
```
    "wins": 7,
    "losses": 3,
    "draws": 0,
    "score": 0.7,
    "mean_plies": 99.8,
    "mean_plies_when_won": 46.14,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 5,
    "losses": 5,
    "draws": 0,
    "score": 0.5,
    "mean_plies": 142.0,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 59.0
  },
  "elapsed_sec": 3328.8,
  "timestamp": "2026-07-17T21:10:39"
}
Saved to benchmarks\marathon\cap_h2h\match_fresh_start_v18_cap_vs_fresh_start_v17_20260717_211039.json
```
[07-17 21:10:39] SCORE cap_h2h: {'overall': 0.6, 'white': 0.7, 'black': 0.5}
[07-17 21:10:39] === STAGE D_sims800 (budget 1.5h, 16.1h left) ===
[07-17 21:10:39] sims probe uses A_ramp
[07-17 21:10:39] START match-sims800_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_ramp\best_value_net.pt --games 20 --sims 800 --workers 6 --out-dir benchmarks\marathon\sims800_anchor
[07-17 21:47:12] END match-sims800_anchor (exit=0, 36.6 min)
```
    "wins": 9,
    "losses": 1,
    "draws": 0,
    "score": 0.9,
    "mean_plies": 49.2,
    "mean_plies_when_won": 29.67,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 5,
    "losses": 5,
    "draws": 0,
    "score": 0.5,
    "mean_plies": 158.5,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 92.0
  },
  "elapsed_sec": 2193.3,
  "timestamp": "2026-07-17T21:47:12"
}
Saved to benchmarks\marathon\sims800_anchor\match_fresh_start_v18_ramp_vs_heuristic_20260717_214712.json
```
[07-17 21:47:12] SCORE sims800_anchor: {'overall': 0.7, 'white': 0.9, 'black': 0.5}
[07-17 21:47:12] === STAGE F_ramp2 (budget 7.0h, 15.5h left) ===
[07-17 21:47:12] START process-ramp2: src/data_processor.py --raw-dir data\raw\combined_v16 --output-dir data\processed\combined_v16_r035h90 --seed 42 --channels 15 --value-discount-mode near_mate --value-horizon 90 --value-floor 0.35
[07-17 21:48:50] END process-ramp2 (exit=0, 1.6 min)
```
Retention summary: kept_games=1457/1755, kept_positions=70478/70805
Game split integrity: PASS (no overlap across train/val/test game IDs)
  Games: train=1164, val=145, test=148
  Processing positions (augment=True)...
  Value targets: near-mate ramp 0.35 -> 1.0 over last 90 plies

Saved to data\processed\combined_v16_r035h90:
  positions.npy:    (140956, 8, 8, 15)
  mcts_values.npy:  (140956,)
  game_results.npy: (140956,)
  policies.npy:     (140956, 4096)
  policy_weights.npy: (140956,) (masked=13372)
  splits.npz:       train=112760, val=11964, test=16232
  split_game_ids.json: game-level split membership saved
```
[07-17 21:48:50] START train-ramp2: src/train.py --data-dir data\processed\combined_v16_r035h90 --model-dir models\candidates\fresh_start_v18_ramp2 --target game_result --value-head scalar --select-metric decisive --epochs 30 --seed 42
[07-18 02:33:38] END train-ramp2 (exit=0, 284.8 min)
```
Epoch  23  train=1.3095 (v=0.0113 p=1.2982)  val=2.6454 (pow=0.1149 mse=0.1220 p=2.5305 mae=0.2106)  lr=7.5e-04  decisive(top1 W=31.2% B=31.4% sign W=94.5% B=92.6%)
Epoch  24  train=1.3068 (v=0.0111 p=1.2957)  val=2.7553 (pow=0.1250 mse=0.1247 p=2.6302 mae=0.2017)  lr=7.2e-04  decisive(top1 W=36.1% B=31.5% sign W=94.9% B=93.2%)
  -> saved best model (decisive_score=1.2469)
Epoch  25  train=1.3020 (v=0.0102 p=1.2918)  val=2.7342 (pow=0.1716 mse=0.1616 p=2.5625 mae=0.2207)  lr=6.8e-04  decisive(top1 W=33.0% B=29.5% sign W=95.1% B=88.1%)
Epoch  26  train=1.2995 (v=0.0099 p=1.2896)  val=2.7825 (pow=0.1358 mse=0.1308 p=2.6466 mae=0.1972)  lr=6.5e-04  decisive(top1 W=34.2% B=30.9% sign W=95.1% B=92.3%)
Epoch  27  train=1.2960 (v=0.0091 p=1.2869)  val=2.7284 (pow=0.1322 mse=0.1314 p=2.5962 mae=0.2086)  lr=6.1e-04  decisive(top1 W=34.7% B=31.6% sign W=94.8% B=92.9%)
Epoch  28  train=1.2935 (v=0.0089 p=1.2847)  val=2.8284 (pow=0.1221 mse=0.1242 p=2.7063 mae=0.2035)  lr=5.8e-04  decisive(top1 W=36.5% B=31.4% sign W=94.4% B=92.9%)
Epoch  29  train=1.2921 (v=0.0088 p=1.2832)  val=2.8967 (pow=0.1136 mse=0.1171 p=2.7831 mae=0.1983)  lr=5.5e-04  decisive(top1 W=34.1% B=30.8% sign W=95.1% B=93.5%)
Epoch  30  train=1.2894 (v=0.0084 p=1.2810)  val=3.0117 (pow=0.1418 mse=0.1365 p=2.8700 mae=0.2027)  lr=5.3e-04  decisive(top1 W=38.5% B=31.5% sign W=94.9% B=91.8%)

--- Test set evaluation ---
Total loss: 3.0635
Value power loss: 0.1403
Value true MSE:   0.1473
Policy CE:  2.9233
Value MAE:  0.2390
Policy top-1 (enabled): W=38.2% B=26.6%
Winner sign (non-draw): W=91.4% B=90.0%
Winner prediction accuracy (non-draw): 90.9%

Best model saved to models\candidates\fresh_start_v18_ramp2\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_ramp2\train_run_20260717_214857.json
```
[07-18 02:33:38] TRACE 0015 ramp2: -0.727 -0.806 -0.820 (v17: +0.993 +0.985 +0.942; positions are lost for White)
[07-18 02:33:38] START match-ramp2_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_ramp2\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\marathon\ramp2_anchor
[07-18 02:46:54] END match-ramp2_anchor (exit=0, 13.3 min)
```
    "wins": 10,
    "losses": 0,
    "draws": 0,
    "score": 1.0,
    "mean_plies": 40.4,
    "mean_plies_when_won": 40.4,
    "mean_plies_when_lost": null
  },
  "a_as_black": {
    "games": 10,
    "wins": 3,
    "losses": 7,
    "draws": 0,
    "score": 0.3,
    "mean_plies": 133.7,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 94.57
  },
  "elapsed_sec": 795.8,
  "timestamp": "2026-07-18T02:46:54"
}
Saved to benchmarks\marathon\ramp2_anchor\match_fresh_start_v18_ramp2_vs_heuristic_20260718_024654.json
```
[07-18 02:46:54] SCORE ramp2_anchor: {'overall': 0.65, 'white': 1.0, 'black': 0.3}
[07-18 02:46:54] START match-ramp2_h2h: tools/match.py --model-a models\candidates\fresh_start_v18_ramp2\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\marathon\ramp2_h2h --model-b models\fresh_start_v17\best_value_net.pt
[07-18 04:00:13] END match-ramp2_h2h (exit=0, 73.3 min)
```
    "wins": 6,
    "losses": 4,
    "draws": 0,
    "score": 0.6,
    "mean_plies": 110.7,
    "mean_plies_when_won": 34.5,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 6,
    "losses": 4,
    "draws": 0,
    "score": 0.6,
    "mean_plies": 208.8,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 184.5
  },
  "elapsed_sec": 4398.3,
  "timestamp": "2026-07-18T04:00:13"
}
Saved to benchmarks\marathon\ramp2_h2h\match_fresh_start_v18_ramp2_vs_fresh_start_v17_20260718_040013.json
```
[07-18 04:00:13] SCORE ramp2_h2h: {'overall': 0.6, 'white': 0.6, 'black': 0.6}
[07-18 04:00:13] === STAGE E_evidence (budget 2.5h, 9.3h left) ===
[07-18 04:00:13] EVIDENCE: A_ramp qualifies — gathering promotion evidence
[07-18 04:00:13] START match-A_ramp_anchor60: tools/match.py --model-a models\candidates\fresh_start_v18_ramp\best_value_net.pt --games 60 --sims 400 --workers 6 --out-dir benchmarks\marathon\A_ramp_anchor60
[07-18 04:48:10] END match-A_ramp_anchor60 (exit=0, 48.0 min)
```
    "wins": 30,
    "losses": 0,
    "draws": 0,
    "score": 1.0,
    "mean_plies": 58.5,
    "mean_plies_when_won": 58.5,
    "mean_plies_when_lost": null
  },
  "a_as_black": {
    "games": 30,
    "wins": 15,
    "losses": 15,
    "draws": 0,
    "score": 0.5,
    "mean_plies": 136.03,
    "mean_plies_when_won": 179.2,
    "mean_plies_when_lost": 92.87
  },
  "elapsed_sec": 2877.2,
  "timestamp": "2026-07-18T04:48:10"
}
Saved to benchmarks\marathon\A_ramp_anchor60\match_fresh_start_v18_ramp_vs_heuristic_20260718_044810.json
```
[07-18 04:48:10] SCORE A_ramp_anchor60: {'overall': 0.75, 'white': 1.0, 'black': 0.5}
[07-18 04:48:10] === MARATHON SUMMARY ===
[07-18 04:48:10] {
  "A_ramp": {
    "anchor": {
      "overall": 0.8,
      "white": 1.0,
      "black": 0.6
    },
    "h2h_v17": {
      "overall": 0.7,
      "white": 0.6,
      "black": 0.8
    },
    "anchor60": {
      "overall": 0.75,
      "white": 1.0,
      "black": 0.5
    }
  },
  "B_human_only": {
    "anchor10": {
      "overall": 0.5,
      "white": 0.8,
      "black": 0.2
    }
  },
  "C_capacity": {
    "anchor": {
      "overall": 0.65,
      "white": 1.0,
      "black": 0.3
    },
    "head": "scalar",
    "h2h_v17": {
      "overall": 0.6,
      "white": 0.7,
      "black": 0.5
    }
  },
  "D_sims800": {
    "model": "A_ramp",
    "anchor800": {
      "overall": 0.7,
      "white": 0.9,
      "black": 0.5
    }
  },
  "F_ramp2": {
    "floor": "0.35",
    "horizon": "90",
    "anchor": {
      "overall": 0.65,
      "white": 1.0,
      "black": 0.3
    },
    "h2h_v17": {
      "overall": 0.6,
      "white": 0.6,
      "black": 0.6
    }
  }
}
[07-18 04:48:10] ALL STAGES DONE — nothing is promoted; owner playtest decides. Candidates: models/candidates/fresh_start_v18_{ramp,cap,ramp2}, models/experiments/human_only_v1
