# Bootstrap follow-ups after the September 6 controlled iteration

These are hypotheses for separate experiments, not changes bundled into the
gen44-to-gen45 teacher iteration. Start with its actual match results and
[the sampled evaluation protocol](SAMPLED_GATE_PROTOCOL.md). Human games stay
held out; no pawn-saving rule or human-derived training patch is proposed.

1. **Faithful reanalysis state (small implementation, correctness first).**
   Persist the full-turn counter, recent legal move history and repetition
   prefix in new self-play records. Reconstruct and validate them during deep
   reanalysis; make legacy missing-state handling explicit. Compare search
   targets on the same recorded positions before training anything. Current
   reanalysis resets the counter and omits history, but the gen44 late-position
   census does not establish this as the cause of its early Black losses.
   Keep this separate from adding input planes, which changes the model ABI.

2. **Game-level value sampling/weighting (small-to-medium).** One game provides
   one outcome but many correlated row labels; long games and White's two
   half-moves affect their total influence. Audit value weight per source game
   and color, then compare a bounded per-game normalization against the current
   recipe, without throwing away useful policy rows. LC0's documented RL loader
   selects a position from a rescored game chunk and uses a rolling data pool;
   this motivates examining our sampling unit, not copying its loader wholesale.
   [LC0 training documentation](https://github.com/LeelaChessZero/lczero-training/blob/master/docs/README.md).

3. **Mixed search budgets during generation (medium).** KataGo varies search
   caps per turn, uses cheap searches for most play, and trains from selected
   full-search turns. This trades more completed outcomes against policy-target
   quality. In Monster Chess, budget the two White halves coherently and record
   which rows qualify. Existing per-game simulation ranges are not this method.
   Compare equal wall-clock budgets, both colors, and held-out play; do not infer
   a speedup or strength gain from Go's results. Forced playouts with target
   pruning are another coupled experiment, not permission to prune our current
   visit distributions blindly.
   [KataGo paper, sections 3.1–3.2](https://arxiv.org/html/1902.10565v5).

4. **Broader generated positions (medium).** Test a small, declared fraction of
   generic position forks or softer generation exploration. Preserve legal
   state and source-game split ancestry. Score the full sampler separately
   from endpoint coverage: novelty alone is not training quality. KataGo also
   describes prior-temperature and policy-surprise weighting techniques, but
   our existing disagreement miner already overlaps the latter's motivation.
   Measure an ablation before adding another hard-example weight.
   [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md).

5. **Model/input changes (larger, later).** Consider additional state planes or
   a larger trunk only after the data and sampling experiments. Version the
   encoding, processor, checkpoint detection and native bridge together; test
   inference cost at equal wall-clock search budgets. Existing opt-in heads
   and search utilities need a new measured hypothesis, not activation merely
   because another engine has them.

Each learning experiment keeps an unchanged control, reserves fresh evaluation
seeds before play, and reports actual-color self-par, direct H2H, draw reasons,
and uncertainty. Do not select changes by a single human position or by offline
accuracy alone, and do not chain Elo estimates across model-dependent samplers.
