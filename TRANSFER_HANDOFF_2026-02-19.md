# Transfer Handoff (2026-02-19)

## Final state on this machine
- All iteration/training processes were stopped.
- Iteration 2 was intentionally aborted.
- Partial iteration-2 data was removed:
  - `data/raw/nn_gen12`
  - `data/raw/nn_gen12_curriculum`
  - `data/raw/nn_gen12_blackfocus` (if created)

## Most recent completed generation work
- Run logs:
  - `models/run_large_20260218_221423.log`
  - `models/run_large_20260218_221423.err.log`
- Generation 11 outputs retained:
  - `data/raw/nn_gen11` (180 games)
  - `data/raw/nn_gen11_blackfocus` (219/260 saved, 41 timed out)
  - `data/raw/nn_gen11_curriculum` (220 games)
- Candidate model from gen11:
  - `models/candidates/gen_0011/best_value_net.pt`
- Arena/gate result (from `models/archive/manifest.jsonl`, `2026-02-19T01:33:22`):
  - `accepted=false`
  - candidate as White: `40/40` wins, score `1.0`
  - candidate as Black: `0/40` wins, score `0.0`
  - overall score `0.5` over 80 games

## Resume command on higher-compute machine
```bash
python src/iterate.py --iterations 2 --games 180 --curriculum-games 220 \
  --black-focus-games 260 --simulations 120 --curriculum-simulations 50 \
  --black-focus-simulations 100 --epochs 12 --warmup-epochs 3 \
  --warmup-start-factor 0.1 --keep-generations 3 --alternating \
  --opponent-sims 140 --pool-size 6 --arena-games 80 --arena-sims 80 \
  --arena-workers 4 --gate-threshold 0.54 --gate-min-other-side 0.42 \
  --seed 20260219 --human-eval
```

## Notes
- The primary failure mode remains unchanged: strong White-side results with weak Black-side gate performance.
- Black-focus generation can hit 600s game timeouts; stronger compute should reduce this.
