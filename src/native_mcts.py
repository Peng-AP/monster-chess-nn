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
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "native") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "native"))

import monster_native as mn  # noqa: E402
from config import (C_PUCT, FPU_REDUCTION, POLICY_TEMPERATURE,
                    MOVES_LEFT_MAX_EFFECT, MOVES_LEFT_THRESHOLD,
                    MOVES_LEFT_SLOPE)  # noqa: E402

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


# A batch-16 forward costs ~4.7 ms eager and ~0.5 ms replayed from a CUDA
# graph -- 9.4x, measured 2026-08-07 on an idle 5060 Ti. Nearly all of the
# eager cost is kernel-launch and dispatch overhead, not arithmetic: the same
# call takes 4.7 ms at batch 1 and 4.3 ms at batch 256. Capturing it once and
# replaying removes that overhead.
#
# This is why the cross-game inference server is not the answer: moving one
# batch-16 request and its 256 KB reply through an mp.Queue costs 0.62 ms, more
# than a graphed forward takes in total.
GRAPH_ENV = "MONSTER_CUDA_GRAPH"


def _graph_enabled():
    value = os.environ.get(GRAPH_ENV, "1").strip().lower()
    return value not in ("0", "false", "no", "off")


class _GraphedForward:
    """Replay captured forward passes -- one graph per batch size, no padding.

    Graph capture needs a fixed shape, but the search submits variable batches
    (the collection loop stops early when the frontier is exhausted). The first
    implementation padded every call to `width`, which was 2.67x faster and
    *changed move selection*: a partial batch run as padded-16 hits different
    cuDNN kernels than a true batch-n, and in fp16 that shifts the largest
    policy logits by a few ULPs -- enough to flip a move occasionally and
    desynchronise a game.

    Capturing one graph per distinct n keeps every call shaped exactly as the
    eager path would have shaped it, so results stay bit-comparable with every
    pre-graph measurement. Graphs are captured lazily and cached; at batch 16
    that is at most 16 of them, a few MB of static buffers.
    """

    def __init__(self, torch, model, device, width, channels, half,
                 include_moves_left=False):
        self.torch, self.model, self.device = torch, model, device
        self.width, self.channels, self.half = width, channels, half
        self.include_moves_left = bool(include_moves_left)
        self._graphs = {}

    def _capture(self, n):
        torch = self.torch
        static_in = torch.zeros(
            (n, self.channels, 8, 8), device=self.device,
            dtype=torch.half if self.half else torch.float32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):          # warm-up is required before capture
                if self.include_moves_left:
                    self.model.forward_with_aux(static_in)
                else:
                    self.model(static_in)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            if self.include_moves_left:
                out_v, out_p, _out_wdl, out_ml = self.model.forward_with_aux(static_in)
            else:
                out_v, out_p = self.model(static_in)
                out_ml = None
        return static_in, graph, out_v, out_p, out_ml

    def __call__(self, tensor, n):
        entry = self._graphs.get(n)
        if entry is None:
            entry = self._capture(n)
            self._graphs[n] = entry
        static_in, graph, out_v, out_p, out_ml = entry
        static_in.copy_(tensor)
        graph.replay()
        return out_v, out_p, out_ml


def make_bridge(nn_evaluator, policy_temperature=POLICY_TEMPERATURE,
                graph_width=None, include_moves_left=False):
    """(eval_fn, input_channels) for `Tree.run_batched_puct`.

    Values come back in the SIDE-TO-MOVE perspective, exactly as the model
    emits them; the native search converts to White's perspective itself
    because it is what knows each leaf's side. Do not convert here.
    """
    import numpy as np

    torch = nn_evaluator.torch
    channels = nn_evaluator.input_channels
    half = bool(getattr(nn_evaluator, "_half", False))
    device = nn_evaluator.device
    include_moves_left = (bool(include_moves_left)
                          and bool(getattr(nn_evaluator.model,
                                           "use_moves_left_head", False)))

    graphed = None
    if graph_width and device.type == "cuda" and _graph_enabled():
        try:
            graphed = _GraphedForward(torch, nn_evaluator.model, device,
                                      int(graph_width), channels, half,
                                      include_moves_left=include_moves_left)
        except Exception as exc:
            # Capture can fail on driver/allocator quirks. Falling back to the
            # eager path is a slowdown, never a wrong answer.
            print(f"[native_mcts] CUDA graph capture unavailable ({exc}); "
                  f"using eager forwards", flush=True)
            graphed = None

    def eval_fn(buf, n, chans):
        array = np.frombuffer(buf, dtype=np.float32).reshape(n, chans, 8, 8)
        tensor = torch.from_numpy(array.copy()).to(device)
        if half:
            tensor = tensor.half()
        if graphed is not None and n <= graphed.width:
            value, policy, moves_left = graphed(tensor, n)
        else:
            with torch.no_grad():
                if include_moves_left:
                    value, policy, _wdl, moves_left = (
                        nn_evaluator.model.forward_with_aux(tensor))
                else:
                    value, policy = nn_evaluator.model(tensor)
                    moves_left = None
        policy = policy.reshape(n, -1).float() / policy_temperature
        result = (
            value.reshape(-1).float().cpu().numpy().astype(np.float32).tobytes(),
            policy.cpu().numpy().astype(np.float32).tobytes(),
        )
        if moves_left is not None:
            ml_bytes = (moves_left.reshape(-1).float().cpu().numpy()
                        .astype(np.float32).tobytes())
            return result + (ml_bytes,)
        return result

    return eval_fn, channels


class NativeMCTS:
    """Same interface as `mcts.MCTS`, backed by the native core."""

    def __init__(self, num_simulations=800, eval_fn=None, batch_size=16,
                 root_noise=False, allow_early_stop=True, seed=None,
                 reuse_across_moves=False, solver=False, c_puct=C_PUCT,
                 fpu_reduction=FPU_REDUCTION,
                 policy_temperature=POLICY_TEMPERATURE,
                 moves_left_utility=False,
                 moves_left_max_effect=MOVES_LEFT_MAX_EFFECT,
                 moves_left_threshold=MOVES_LEFT_THRESHOLD,
                 moves_left_slope=MOVES_LEFT_SLOPE):
        if not math.isfinite(float(c_puct)) or float(c_puct) < 0:
            raise ValueError("c_puct must be finite and >= 0")
        if not math.isfinite(float(fpu_reduction)) or float(fpu_reduction) < 0:
            raise ValueError("fpu_reduction must be finite and >= 0")
        if (not math.isfinite(float(policy_temperature))
                or float(policy_temperature) <= 0):
            raise ValueError("policy_temperature must be finite and > 0")
        if (not math.isfinite(float(moves_left_max_effect))
                or not 0.0 <= float(moves_left_max_effect) <= 1.0):
            raise ValueError("moves_left_max_effect must be finite and in [0, 1]")
        if (not math.isfinite(float(moves_left_threshold))
                or not 0.0 <= float(moves_left_threshold) < 1.0):
            raise ValueError("moves_left_threshold must be finite and in [0, 1)")
        if (not math.isfinite(float(moves_left_slope))
                or float(moves_left_slope) < 0.0):
            raise ValueError("moves_left_slope must be finite and >= 0")
        # `seed=None` draws from Python's global RNG **at construction**, which
        # is how this engine inherits per-game seeding. `mcts.MCTS` reads that
        # global module directly, and both `data_generation._worker` and
        # `match._play` re-seed it once per game. A constant default instead
        # gives every game the identical noise and temperature draws: 240
        # self-play games that all ended White at fullmove 6-7, against
        # python's 14/10 spread over fullmove 7-75. Caught 2026-08-04.
        # Pass an explicit seed for reproducibility.
        # seed=None is the FAITHFUL mode and the default: every decision draws
        # from Python's global RNG, exactly as `mcts.MCTS` does (it reads that
        # module for temperature sampling and expansion shuffling). This is
        # what makes per-game `random.seed()` reach the engine.
        #
        # It matters because engines are built once per WORKER while harnesses
        # re-seed once per GAME (`promotion_defense_probe._play_out`,
        # `match._play`). An engine that captured a seed at construction
        # ignored that entirely: Python restarted its randomness each game
        # while native's stream ran on across all of them. Observable symptom:
        # noise-on self-play collapsed to near-identical short games (median
        # final fullmove 7 against python's 37 on the same command). Caught
        # 2026-08-04.
        #
        # CORRECTION, same day: the native-vs-python PPC curve gap first
        # blamed on this is UNRELATED. Those playouts run at temperature 0
        # with noise off, and a post-fix re-run reproduced every statistic of
        # the pre-fix curve exactly -- they never touch the RNG. The curve gap
        # is an open question (REPORT.md).
        #
        # An explicit seed keeps the reproducible internal counter instead.
        self.num_simulations = num_simulations
        self.eval_fn = eval_fn
        self.batch_size = batch_size
        self.root_noise = root_noise
        self.allow_early_stop = allow_early_stop
        self.reuse_across_moves = reuse_across_moves
        self.solver = solver
        self.seed = seed
        self.c_puct = float(c_puct)
        self.fpu_reduction = float(fpu_reduction)
        self.policy_temperature = float(policy_temperature)
        self.moves_left_utility = bool(moves_left_utility)
        self.moves_left_max_effect = float(moves_left_max_effect)
        self.moves_left_threshold = float(moves_left_threshold)
        self.moves_left_slope = float(moves_left_slope)
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
        if (self.moves_left_utility
                and (nn is None
                     or not bool(getattr(nn.model,
                                         "use_moves_left_head", False)))):
            raise ValueError("moves-left utility requires a checkpoint with a "
                             "trained moves-left head")
        if nn is not None:
            self._bridge, self._channels = make_bridge(
                nn, policy_temperature=self.policy_temperature,
                graph_width=self.batch_size,
                include_moves_left=self.moves_left_utility)
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
        """Keep the played subtree so the next search continues this tree.

        `reuse_across_moves=False` restores the Python engine's scope -- only
        White's first -> second half -- which is what the two are compared
        against. Beyond that the tree survives the whole game, because
        `Arena::reroot` rebases the new root's frame instead of refusing.
        """
        if selected_uci is None:
            self._reuse_tree = None
            self._reuse_key = None
            return
        if not self.reuse_across_moves:
            first_white_half = (state.is_white_turn
                                and not getattr(state, "white_half_pending", False))
            if not first_white_half:
                self._reuse_tree = None
                self._reuse_key = None
                return
        if tree.reroot(selected_uci):
            self._reuse_tree = tree
            self._reuse_key = (tree.fen(0), tree.is_white_turn(0),
                               tree.white_half_pending(0))
        else:
            self._reuse_tree = None
            self._reuse_key = None

    # ------------------------------------------------------------------
    def get_best_action(self, root_state, temperature=1.0):
        tree = self._tree_for(root_state)
        if self.seed is None:
            # Follow the global RNG, like MCTS. Per-game re-seeding reaches us.
            import random as _random
            decision_seed = _random.randrange(1, 2 ** 62)
        else:
            decision_seed = self.seed + self._decisions
        self._decisions += 1

        if self._bridge is not None:
            tree.run_batched_puct(
                self.num_simulations, self._bridge,
                batch_size=self.batch_size, channels=self._channels,
                allow_early_stop=self.allow_early_stop,
                root_noise=self.root_noise, seed=decision_seed,
                heuristic_values=self._heuristic_values, solver=self.solver,
                c_puct=self.c_puct, fpu_reduction=self.fpu_reduction,
                moves_left_max_effect=(self.moves_left_max_effect
                                       if self.moves_left_utility else 0.0),
                moves_left_threshold=self.moves_left_threshold,
                moves_left_slope=self.moves_left_slope)
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
