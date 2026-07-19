
# HYBRID — started 2026-07-18 19:27:58

[07-18 19:27:58] hybrid inference blend weight: 0.3 (scalar head trained on end-anchored ramp 0.5/60)
[07-18 19:27:58] START train-hybrid: src/train.py --data-dir data\processed\combined_v16_r50h60 --model-dir models\candidates\fresh_start_v18_hybrid --target game_result --value-head hybrid --select-metric decisive --epochs 30 --seed 42
[07-19 00:37:42] END train-hybrid (exit=0, 309.7 min)
```
Epoch  25  train=1.3081 (v=0.0094 p=1.2928)  val=2.9546 (pow=0.1256 mse=0.1796 p=2.6757 mae=0.2548)  lr=6.8e-04  wdl(train_ce=0.0118 val_ce=0.3066 val_acc=94.5%)  decisive(top1 W=35.2% B=30.0% sign W=95.4% B=92.9%)
Epoch  26  train=1.3031 (v=0.0086 p=1.2894)  val=2.9373 (pow=0.1194 mse=0.1773 p=2.6467 mae=0.2536)  lr=6.5e-04  wdl(train_ce=0.0102 val_ce=0.3424 val_acc=94.7%)  decisive(top1 W=35.3% B=30.9% sign W=95.4% B=93.6%)
  -> saved best model (decisive_score=1.2452)
Epoch  27  train=1.3014 (v=0.0087 p=1.2874)  val=3.0168 (pow=0.1188 mse=0.1785 p=2.7292 mae=0.2581)  lr=6.1e-04  wdl(train_ce=0.0105 val_ce=0.3377 val_acc=94.7%)  decisive(top1 W=37.8% B=30.4% sign W=95.3% B=93.5%)
Epoch  28  train=1.2972 (v=0.0080 p=1.2845)  val=3.0291 (pow=0.1203 mse=0.1889 p=2.7261 mae=0.2604)  lr=5.8e-04  wdl(train_ce=0.0093 val_ce=0.3654 val_acc=94.2%)  decisive(top1 W=37.2% B=30.5% sign W=94.5% B=93.5%)
Epoch  29  train=1.2960 (v=0.0080 p=1.2833)  val=3.0302 (pow=0.1162 mse=0.1727 p=2.7506 mae=0.2493)  lr=5.5e-04  wdl(train_ce=0.0092 val_ce=0.3268 val_acc=94.8%)  decisive(top1 W=34.8% B=31.0% sign W=95.5% B=93.5%)
Epoch  30  train=1.2936 (v=0.0078 p=1.2812)  val=3.3476 (pow=0.1239 mse=0.1812 p=3.0265 mae=0.2553)  lr=5.3e-04  wdl(train_ce=0.0091 val_ce=0.3945 val_acc=94.6%)  decisive(top1 W=34.3% B=30.3% sign W=95.4% B=93.2%)

--- Test set evaluation ---
Total loss: 3.3153
Value power loss: 0.1526
Value true MSE:   0.2471
Policy CE:  2.9022
Value MAE:  0.3309
WDL CE:     0.5210
WDL Acc:    91.7%
Policy top-1 (enabled): W=36.2% B=28.0%
Winner sign (non-draw): W=92.8% B=89.5%
Winner prediction accuracy (non-draw): 91.6%

Best model saved to models\candidates\fresh_start_v18_hybrid\best_value_net.pt
Run metadata saved to models\candidates\fresh_start_v18_hybrid\train_run_20260718_192811.json
```
[07-19 00:37:44] TRACE 0015 hybrid: -0.899 -0.921 -0.941 (v17: +0.993 +0.985 +0.942; positions are lost for White)
[07-19 00:37:44] START diff-clean-informational: tools/model_diff.py --candidate models\candidates\fresh_start_v18_hybrid\best_value_net.pt --incumbent models\fresh_start_v17\best_value_net.pt --data-dir data\processed\eval_clean_v13v16 --split all --max-positions 8192
[07-19 00:38:06] END diff-clean-informational (exit=0, 0.4 min)
```
=== model_diff: 8192 all positions from data\processed\eval_clean_v13v16 ===
metric                  candidate  incumbent    delta
policy_ce                  2.4921     2.4171  +0.0750
policy_top1                0.3302     0.2893  +0.0409
policy_top1_white          0.3531     0.3012  +0.0519
policy_top1_black          0.2838     0.2652  +0.0185
sign_acc                   0.9126     0.9006  +0.0120
sign_acc_white             0.9216     0.9113  +0.0102
sign_acc_black             0.8946     0.8792  +0.0154
Saved to C:\Users\AaronPeng\Desktop\monster-chess-nn\benchmarks\model_diff_20260719_003806.json
MODEL DIFF: PASS
```
[07-19 00:38:06] START match-hybrid_anchor: tools/match.py --model-a models\candidates\fresh_start_v18_hybrid\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\hybrid\hybrid_anchor
[07-19 00:55:49] END match-hybrid_anchor (exit=0, 17.7 min)
```
    "wins": 7,
    "losses": 3,
    "draws": 0,
    "score": 0.7,
    "mean_plies": 149.7,
    "mean_plies_when_won": 117.43,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 3,
    "losses": 7,
    "draws": 0,
    "score": 0.3,
    "mean_plies": 98.9,
    "mean_plies_when_won": 180.0,
    "mean_plies_when_lost": 64.14
  },
  "elapsed_sec": 1062.4,
  "timestamp": "2026-07-19T00:55:49"
}
Saved to benchmarks\hybrid\hybrid_anchor\match_fresh_start_v18_hybrid_vs_heuristic_20260719_005549.json
```
[07-19 00:55:49] SCORE hybrid_anchor: {'overall': 0.5, 'white': 0.7, 'black': 0.3}
[07-19 00:55:49] START match-hybrid_h2h: tools/match.py --model-a models\candidates\fresh_start_v18_hybrid\best_value_net.pt --games 20 --sims 400 --workers 6 --out-dir benchmarks\hybrid\hybrid_h2h --model-b models\fresh_start_v17\best_value_net.pt
[07-19 01:47:43] END match-hybrid_h2h (exit=0, 51.9 min)
```
    "wins": 5,
    "losses": 5,
    "draws": 0,
    "score": 0.5,
    "mean_plies": 154.4,
    "mean_plies_when_won": 83.8,
    "mean_plies_when_lost": 225.0
  },
  "a_as_black": {
    "games": 10,
    "wins": 5,
    "losses": 5,
    "draws": 0,
    "score": 0.5,
    "mean_plies": 173.2,
    "mean_plies_when_won": 225.0,
    "mean_plies_when_lost": 121.4
  },
  "elapsed_sec": 3114.0,
  "timestamp": "2026-07-19T01:47:43"
}
Saved to benchmarks\hybrid\hybrid_h2h\match_fresh_start_v18_hybrid_vs_fresh_start_v17_20260719_014743.json
```
[07-19 01:47:43] SCORE hybrid_h2h: {'overall': 0.5, 'white': 0.5, 'black': 0.5}
[07-19 01:47:43] ALL STEPS COMPLETE — hybrid candidate awaits owner play (models/candidates/fresh_start_v18_hybrid)

## Blend sweep — 2026-07-19 01:55:00

[07-19 01:55:00] START sweep_anchor_w50 (w=0.5)
[07-19 02:14:57] SCORE sweep_anchor_w50 (w=0.5): {'overall': 0.6, 'white': 1.0, 'black': 0.2}
[07-19 02:14:57] START sweep_anchor_w75 (w=0.75)
[07-19 02:34:28] SCORE sweep_anchor_w75 (w=0.75): {'overall': 0.5, 'white': 0.9, 'black': 0.1}
[07-19 02:34:28] START sweep_anchor_w100 (w=1.0)
[07-19 02:47:10] SCORE sweep_anchor_w100 (w=1.0): {'overall': 0.55, 'white': 1.0, 'black': 0.1}
[07-19 02:47:10] SWEEP RESULTS: {0.5: 0.6, 0.75: 0.5, 1.0: 0.55} — best w=0.5
[07-19 02:47:10] START sweep_h2h_w50 (w=0.5)
[07-19 03:36:01] SCORE sweep_h2h_w50 (w=0.5): {'overall': 0.65, 'white': 0.7, 'black': 0.6}
[07-19 03:36:01] SWEEP COMPLETE — set VALUE_HYBRID_PROGRESS_WEIGHT to the chosen w before owner play (or export MONSTER_HYBRID_W)
