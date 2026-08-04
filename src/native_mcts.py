"""Drop-in native search (DIRECTIVE D5 / E4).

`NativeMCTS` presents the same surface as `mcts.MCTS` — construct with a
simulation count and an evaluator, call `get_best_action(state, temperature)`,
get back `(action, action_probs, search_value)` — so every existing call site
(data_generation, benchmark, play, promotion_probe, generate_bruteforce) can
swap engines without changing how it drives a game.

Three things this adapter exists to get right, each of which produced a real
bug when it was missing:

1. **It passes the whole state, never a bare FEN.** A FEN under-determines a
   Monster Chess position in four separate ways — `white_half_pending` (a
   different action set), `turn_count` (the move-limit cap), move history (the
   oscillation penalty) and an ep square with no legal capture. History is read
   straight off `board.move_stack`, so callers do not have to track it.

2. **It returns the caller's own `Move` object**, not a UCI string, because
   drivers pass the result to `apply_search_action`. The native side speaks in
   UCI; the mapping back happens here.

3. **It matches the evaluator's value source.** `NNEvaluator` takes values from
   the network; `HybridEvaluator` takes values from the heuristic and only the
   policy from the network. Using NN values where generation uses hybrid ones
   gives the same moves early and a different tree once values diverge.

Tree reuse across White's first -> second half-move is handled here too, mirroring
`MCTS._remember_white_continuation`: both nodes are White-to-move so accumulated
Q stays valid, and reuse across a side change is refused rather than rebased.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "native") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "native"))

import monster_native as mn  # noqa: E402

HISTORY_PLIES = 8


def _underlying_nn(evaluator):
    """The NNEvaluator inside `evaluator`, or None for a plain heuristic."""
    if evaluator is None:
        return None
    if hasattr(evaluator, "model") and hasattr(evaluator, "device"):
        return evaluator
    inner = getattr(evaluator, "_nn", None)
    if inner is not None and hasattr(inner, "model"):
        return inner
    return None


def _uses_heuristic_values(evaluator):
    """True for HybridEvaluator: heuristic values, NN policy."""
    return _underlying_nn(evaluator) is not None and not (
        hasattr(evaluator, "model") and hasattr(evaluator, "device"))


def make_bridge(nn_evaluator):
    """(eval_fn, input_channels) for `Tree.run_batched_puct`.

    Values come back in the SIDE-TO-MOVE perspective, exactly as the model
    emits them; the native search converts to White's perspective itself
    because it is what knows each leaf's side. Do not convert here.
    """
    import numpy as np

    torch = nn_evaluator.torch
    channels = nn_evaluator.input_channels

    def eval_fn(buf, n, chans):
        array = np.frombuffer(buf, dtype=np.float32).reshape(n, chans, 8, 8)
        tensor = torch.from_numpy(array.copy()).to(nn_evaluator.device)
        if getattr(nn_evaluator, "_half", False):
            tensor = tensor.half()
        with torch.no_grad():
            value, policy = nn_evaluator.model(tensor)
        return (value.reshape(-1).float().cpu().numpy().astype(np.float32).tobytes(),
                policy.reshape(n, -1).float().cpu().numpy().astype(np.float32).tobytes())

    return eval_fn, channels


class NativeMCTS:
    """Same interface as `mcts.MCTS`, backed by the native core."""

    def __init__(self, num_simulations=800, eval_fn=None, batch_size=16,
                 root_noise=False, allow_early_stop=True, seed=20260803):
        self.num_simulations = num_simulations
        self.eval_fn = eval_fn
        self.batch_size = batch_size
        self.root_noise = root_noise
        self.allow_early_stop = allow_early_stop
        self.seed = seed
        # Advanced once per decision. The native RNG is constructed from the
        # seed it is GIVEN, so passing a constant re-seeds an identical stream
        # every call: temperature sampling then returns the same move every
        # time and Dirichlet noise draws the same vector every time --
        # exploration is frozen while still looking random, because different
        # positions still yield different moves. Caught 2026-08-04 when native
        # self-play produced White 240/240 against python's 14/10 on the same
        # command; gates (a) and (b) run at temperature 0 with noise off and
        # cannot see it.
        self._decisions = 0

        nn = _underlying_nn(eval_fn)
        self._heuristic_values = _uses_heuristic_values(eval_fn)
        if nn is not None:
            self._bridge, self._channels = make_bridge(nn)
        else:
            self._bridge, self._channels = None, None

        # Tree reuse, valid only across White's first -> second half-move.
        self._reuse_tree = None
        self._reuse_key = None

    # ------------------------------------------------------------------
    @staticmethod
    def _state_key(state):
        return (state.fen(), bool(state.is_white_turn),
                bool(getattr(state, "white_half_pending", False)))

    @staticmethod
    def _history(state):
        stack = getattr(state.board, "move_stack", [])
        return [m.uci() for m in stack[-HISTORY_PLIES:]]

    def _tree_for(self, state):
        key = self._state_key(state)
        if self._reuse_tree is not None and self._reuse_key == key:
            tree = self._reuse_tree
            self._reuse_tree = None
            self._reuse_key = None
            return tree
        self._reuse_tree = None
        self._reuse_key = None
        return mn.Tree(state.fen(),
                       bool(getattr(state, "white_half_pending", False)),
                       int(getattr(state, "turn_count", 0)),
                       self._history(state))

    def _remember(self, state, tree, selected_uci):
        """Keep the selected subtree when White still owes its second half."""
        first_white_half = (state.is_white_turn
                            and not getattr(state, "white_half_pending", False))
        if not first_white_half or selected_uci is None:
            self._reuse_tree = None
            self._reuse_key = None
            return
        if tree.reroot(selected_uci):
            self._reuse_tree = tree
            self._reuse_key = (tree.fen(0), tree.is_white_turn(0),
                               tree.white_half_pending(0))

    # ------------------------------------------------------------------
    def get_best_action(self, root_state, temperature=1.0):
        tree = self._tree_for(root_state)
        # A fresh stream per decision, reproducible for a given engine instance.
        decision_seed = self.seed + self._decisions
        self._decisions += 1

        if self._bridge is not None:
            tree.run_batched_puct(
                self.num_simulations, self._bridge,
                batch_size=self.batch_size, channels=self._channels,
                allow_early_stop=self.allow_early_stop,
                root_noise=self.root_noise, seed=decision_seed,
                heuristic_values=self._heuristic_values)
        else:
            tree.run_sequential(self.num_simulations,
                                allow_early_stop=self.allow_early_stop,
                                seed=decision_seed)

        selected_uci, probs, value = tree.best_action(temperature=temperature,
                                                      seed=decision_seed)
        if selected_uci is None:
            self._reuse_tree = None
            self._reuse_key = None
            return None, {}, 0.0

        legal = root_state.get_search_actions()
        move = next((m for m in legal if m.uci() == selected_uci), None)
        if move is None:
            # The native engine offered something this state does not list.
            # Fail loudly: silently substituting would corrupt recorded games.
            raise RuntimeError(
                f"native search returned {selected_uci!r}, not legal at "
                f"{root_state.fen()} (pending="
                f"{getattr(root_state, 'white_half_pending', False)})")

        self._remember(root_state, tree, selected_uci)
        return move, dict(probs), value
