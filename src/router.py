"""Phase router: one engine per turn, chosen by White's pawn count.

The two strongest checkpoints are strong in different phases: v17 (WDL head)
vetoes material loss in the opening but plays fatalist, shuffling chess once
its win probability flattens; the ramp checkpoint (scalar end-anchored ramp)
keeps a value gradient into deep endgames but its flat early values release
the heuristic pawn-chucking prior. The router plays each phase with the model
that is strong there: the opening model while White still has at least
``min_white_pawns`` pawns, the late model after.

Routing happens at the ROOT, once per turn: the chosen engine runs the whole
search, so values from the two models never mix inside one tree. The choice
is sticky across White's m1 -> m2 half-moves, keeping the searched pair and
the half-move tree reuse within a single engine.

Routing is side-aware: Black's game is decided in the pawn phase (the cliff,
2026-07-09), and the ramp checkpoint is the strongest pawn-phase Black on
record — handing Black's opening to v17 measurably destroyed it (validation
2026-07-20: Black 0.25-0.30 vs both parents). ``black_model`` therefore pins
Black roots to one engine ("late" by default in the shipped spec); the phase
rule applies to White roots.

A router is described by a JSON spec (typically models/experiments/*/router.json):

    {"opening_model": "models/fresh_start_v17/best_value_net.pt",
     "late_model": "models/rejected/fresh_start_v18_ramp/best_value_net.pt",
     "min_white_pawns": 3,
     "black_model": "late"}

``black_model`` is "late", "opening", or "phase" (phase = route Black roots
by pawn count like White; the rejected first design).

benchmark._build_engine, tools/match.py, and play.ipynb accept the spec path
anywhere a .pt model path is accepted.
"""
import json
import os

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RouterMCTS:
    """Delegates get_best_action to one of two engines based on the root."""

    def __init__(self, opening_engine, late_engine, min_white_pawns=3,
                 black_engine=None):
        self.opening_engine = opening_engine
        self.late_engine = late_engine
        self.min_white_pawns = int(min_white_pawns)
        # Fixed engine for Black roots; None routes Black by phase like White.
        self.black_engine = black_engine
        self._turn_engine = None

    def _phase_engine(self, state):
        if self.black_engine is not None and not state.is_white_turn:
            return self.black_engine
        wp = len(state.board.pieces(chess.PAWN, chess.WHITE))
        return self.opening_engine if wp >= self.min_white_pawns else self.late_engine

    def _peek(self, state):
        """Engine that would handle this root, without touching stickiness."""
        if getattr(state, "white_half_pending", False) and self._turn_engine is not None:
            return self._turn_engine
        return self._phase_engine(state)

    def _route(self, state):
        engine = self._peek(state)
        self._turn_engine = engine
        return engine

    def get_best_action(self, root_state, temperature=1.0):
        return self._route(root_state).get_best_action(
            root_state, temperature=temperature)

    def eval_for(self, state):
        """Raw evaluator call routed like a search from this root would be."""
        return self._peek(state).eval_fn(state)


def load_router(spec_path, sims, root_noise=False, allow_early_stop=True):
    from evaluation import NNEvaluator
    from mcts import MCTS

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    def build(key):
        path = spec[key]
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        return MCTS(num_simulations=sims, eval_fn=NNEvaluator(path),
                    root_noise=root_noise, allow_early_stop=allow_early_stop)

    opening = build("opening_model")
    late = build("late_model")
    black = {"late": late, "opening": opening,
             "phase": None}[spec.get("black_model", "phase")]
    return RouterMCTS(opening, late,
                      min_white_pawns=spec.get("min_white_pawns", 3),
                      black_engine=black)
