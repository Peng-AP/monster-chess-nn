import math
import random
from collections import defaultdict

import chess
import numpy as np

from config import (
    EXPLORATION_CONSTANT, C_PUCT, MCTS_SIMULATIONS, POLICY_SIZE,
    DIRICHLET_ALPHA, DIRICHLET_EPSILON, FPU_REDUCTION, POLICY_TARGET_PSEUDOCOUNT,
    POLICY_TEMPERATURE, MOVES_LEFT_MAX_EFFECT, MOVES_LEFT_THRESHOLD,
    MOVES_LEFT_SLOPE,
)
from evaluation import evaluate


def _turn_completing(state):
    """True when the action to be selected completes a turn: Black's move, or
    White's SECOND half-move. White's first half may legally pass through
    check (only the completed pair must leave the king safe), so the safety
    override below must not fire on it."""
    return (not state.is_white_turn) or getattr(state, "white_half_pending", False)


def _hangs_king(state, action):
    """True if, after this turn-completing action, the opponent can capture
    the mover's king immediately."""
    from evaluation import _white_threat_scan, _black_can_capture_king
    tmp = state.clone()
    apply_fn = getattr(tmp, "apply_search_action", None) or tmp.apply_action
    apply_fn(action)
    if tmp.is_terminal():
        return False  # the action itself ended the game
    if tmp.is_white_turn:
        return _white_threat_scan(tmp)          # mover was Black
    return _black_can_capture_king(tmp.board)   # mover was White


def _king_safety_override(state, selected_action, children_info):
    """Never hand the opponent an immediate king capture when a searched
    alternative survives (owner directive 2026-07-12: engine-wide, so
    training data, arena, and benchmark all inherit it).

    Search normally avoids these via the eval threat clamps, but at value
    saturation (all moves ~ -0.98) selection degenerates to noise and the
    fastest mate gets gifted. The override re-ranks only among the search's
    OWN children, by visit count, and touches nothing unless the chosen
    action actually hangs the king. Cost: one threat scan per move in the
    common case (<1ms).
    """
    if selected_action is None or not _turn_completing(state):
        return selected_action
    if not _hangs_king(state, selected_action):
        return selected_action
    for child, _key, _visits in sorted(children_info, key=lambda x: -x[2]):
        if child.action == selected_action:
            continue
        if not _hangs_king(state, child.action):
            return child.action
    return selected_action  # every searched move loses the king — forced


def _m1_dooms_king(state, m1):
    """True if after this first half-move EVERY legal second half hangs the
    king. A winning m1 (king capture) is never doomed."""
    tmp = state.clone()
    tmp.apply_search_action(m1)
    if tmp.is_terminal():
        return False
    for m2 in tmp.get_search_actions():
        if not _hangs_king(tmp, m2):
            return False
    return True


def _white_first_half_override(state, selected_action, children_info):
    """A first half-move must keep at least one king-safe completion whenever
    a searched alternative does.

    The m2-level king-safety override cannot repair an m1 blunder: once the
    first half walks into a pocket where every second half leaves the king
    capturable, it sees "all moves lose — forced" and stands down. At value
    saturation search picks exactly such m1s (owner game 2026-07-17: a
    promoted queen checked the king and White stood still and let it take,
    despite numerous safe pairs under other first moves). Common-case cost:
    one clone + an early-exiting safe-m2 probe per White move.
    """
    if selected_action is None:
        return selected_action
    if not state.is_white_turn or getattr(state, "white_half_pending", False):
        return selected_action
    if not _m1_dooms_king(state, selected_action):
        return selected_action
    for child, _key, _visits in sorted(children_info, key=lambda x: -x[2]):
        if child.action == selected_action:
            continue
        if not _m1_dooms_king(state, child.action):
            return child.action
    return selected_action  # every first move is doomed — genuinely mated


OSCILLATION_VISIT_PENALTY = 0.10


def _own_previous_moves(state):
    """The side-to-move's recent moves, for oscillation detection.

    Offsets follow the fixed push cycle (Black 1 push, White 2): at a Black
    root the stack ends [..., B_prev, W_m1, W_m2]; at a White m2 root it ends
    [..., W_m1_prev, W_m2_prev, B_prev, W_m1_this]. FEN-constructed positions
    start with an empty stack — no history, no override — and history accrues
    as the driver applies moves. A misaligned offset (null-m2 edge case) can
    only surface an OPPONENT move, whose exact reversal is never a legal own
    move (its from-square is occupied by the opponent), so false positives
    are structurally excluded.
    """
    stack = state.board.move_stack
    if state.is_white_turn:  # selecting White's second half
        offsets = (-1, -3, -4)
    else:
        offsets = (-3,)
    return [stack[i] for i in offsets if len(stack) >= -i]


def _is_reversal(action, prev_moves):
    """True when action exactly reverses one of prev_moves (A->B after B->A)."""
    if not isinstance(action, chess.Move):
        try:
            action = chess.Move.from_uci(str(action))
        except ValueError:
            return False
    return any(
        action.from_square == prev.to_square
        and action.to_square == prev.from_square
        for prev in prev_moves
    )


def _oscillation_adjusted_visits(state, children_info):
    """Visit counts with a mild discount on turn-completing moves that exactly
    reverse the mover's own recent move (owner 2026-07-17: penalize, don't
    forbid — sometimes going back IS best; and the pathology is PIECE
    oscillation, since full-position repetition never recurs during a pawn
    run). The discount only decides ties: at value saturation visits flatten
    and the shuffle would win by noise; a reversal search genuinely prefers
    (clearly more visits) still gets played. Selection-time bias only —
    policy training targets keep the raw visit distribution.
    """
    visits = [info[2] for info in children_info]
    if not _turn_completing(state):
        return visits
    prev = _own_previous_moves(state)
    if not prev:
        return visits
    return [
        v * (1.0 - OSCILLATION_VISIT_PENALTY)
        if _is_reversal(info[0].action, prev) else v
        for info, v in zip(children_info, visits)
    ]


def _selected_child_value(children_info, selected_action, fallback):
    """Search value to report: the SELECTED child's Q, not the root average.

    Root Q is a visit-weighted mean over every simulation, including the ones
    spent refuting losing siblings — a proven mate reads ~+0.7 when a quarter
    of the visits went to exploration (owner 2026-07-17: "put off by non-1
    evals for mate positions when clearly the search can find it"). The
    selected child's Q is the search's actual conclusion and sits at exactly
    +-1.0 for proven king-capture lines. A root child's Q accumulates in the
    root's side-to-move perspective (see _backpropagate), so no sign change.
    """
    for child, _key, _visits in children_info:
        if child.action == selected_action:
            if child.visit_count > 0:
                return child.q_value
            break
    return fallback


def _softmax_masked(logits, indices, temperature=POLICY_TEMPERATURE):
    """Softmax over a subset of logit indices, returning {index: prob}.

    This temperature affects the network prior fed to PUCT. It is separate
    from ``get_best_action(..., temperature=...)``, which samples the final
    move from visit counts.
    """
    if not indices:
        return {}
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("policy temperature must be finite and > 0")
    vals = np.array([logits[i] for i in indices], dtype=np.float64) / temperature
    vals -= vals.max()
    exp_vals = np.exp(vals)
    total = exp_vals.sum()
    if total == 0:
        # Degenerate case: assign uniform
        uniform = 1.0 / len(indices)
        return {idx: uniform for idx in indices}
    return {idx: float(exp_vals[j] / total) for j, idx in enumerate(indices)}


# ----------------------------------------------------------------------
# State access helpers.
#
# The search prefers the half-move API (get_search_actions / apply_search_action)
# when the game exposes it: White's two-move turn is decomposed into two plies so
# each half gets a learned prior and the branching factor collapses from ~900
# (m1, m2) pairs to ~30+30 single moves (REWORK_PLAN.md Phase 3).  When the game
# only offers the atomic pair API these fall back to it unchanged.
# ----------------------------------------------------------------------

def _state_legal_actions(state):
    fn = getattr(state, "get_search_actions", None)
    return fn() if fn is not None else state.get_legal_actions()


def _state_apply(state, action):
    fn = getattr(state, "apply_search_action", None)
    if fn is not None:
        fn(action)
    else:
        state.apply_action(action)


class MCTSNode:
    __slots__ = (
        "state", "parent", "children", "action",
        "visit_count", "total_value", "prior",
        "total_moves_left", "moves_left_count",
        "_untried_actions", "_is_expanded",
    )

    def __init__(self, state, parent=None, action=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.total_moves_left = 0.0
        self.moves_left_count = 0
        self.children = []
        self._untried_actions = None
        self._is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def moves_left(self):
        if self.moves_left_count == 0:
            return None
        return self.total_moves_left / self.moves_left_count

    # --- UCB1 (heuristic mode) ---

    def ucb_score(self, c=EXPLORATION_CONSTANT):
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.q_value
        exploration = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration

    def best_child_ucb(self, c=EXPLORATION_CONSTANT):
        return max(self.children, key=lambda ch: ch.ucb_score(c))

    # --- PUCT (NN policy mode) ---

    def puct_score(self, c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION,
                   moves_left_max_effect=0.0,
                   moves_left_threshold=MOVES_LEFT_THRESHOLD,
                   moves_left_slope=MOVES_LEFT_SLOPE):
        """AlphaZero-style PUCT: Q + c * P * sqrt(N_parent) / (1 + N)."""
        parent_visits = max(1, self.parent.visit_count) if self.parent else 1
        if self.visit_count == 0:
            # FPU: start an unvisited child's Q slightly below the parent's value,
            # expressed in the *current selector's* (parent side-to-move) perspective.
            #
            # A node's stored q_value is in the perspective of the side that moved
            # INTO it — i.e. its own parent's side-to-move (see _backpropagate).  So
            # parent.q_value is in the grandparent's perspective.  It matches the
            # selector's perspective only when grandparent and parent share the same
            # side to move; otherwise it must be flipped.  Under strict alternation
            # that flip happens every non-root ply, but White's two consecutive
            # half-moves do NOT flip side, so the general test below is required.
            if self.parent is None:
                fpu_q = 0.0
            else:
                parent_q = self.parent.q_value
                gp = self.parent.parent
                if gp is not None and gp.state.is_white_turn != self.parent.state.is_white_turn:
                    parent_q = -parent_q
                fpu_q = max(-1.0, min(1.0, parent_q - fpu_reduction))
            return fpu_q + c_puct * self.prior * math.sqrt(parent_visits)
        score = (self.q_value
                 + c_puct * self.prior * math.sqrt(parent_visits)
                 / (1 + self.visit_count))
        if (moves_left_max_effect <= 0 or moves_left_slope <= 0
                or self.parent is None or self.moves_left is None
                or self.parent.moves_left is None):
            return score
        q = self.q_value
        urgency = max(0.0, min(
            1.0,
            (abs(q) - moves_left_threshold) / (1.0 - moves_left_threshold),
        ))
        expected_child = max(0.0, self.parent.moves_left - 1.0)
        length_signal = math.tanh(
            (expected_child - self.moves_left) * moves_left_slope)
        return (score + math.copysign(1.0, q) * moves_left_max_effect
                * urgency * length_signal)

    def best_child_puct(self, c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION,
                        moves_left_max_effect=0.0,
                        moves_left_threshold=MOVES_LEFT_THRESHOLD,
                        moves_left_slope=MOVES_LEFT_SLOPE):
        return max(self.children, key=lambda ch: ch.puct_score(
            c_puct, fpu_reduction=fpu_reduction,
            moves_left_max_effect=moves_left_max_effect,
            moves_left_threshold=moves_left_threshold,
            moves_left_slope=moves_left_slope))

    # --- Expansion ---

    def expand_one(self):
        """Expand one untried action (UCB1 mode). Returns new child or None."""
        if self._untried_actions is None:
            self._untried_actions = list(_state_legal_actions(self.state))
            random.shuffle(self._untried_actions)

        if not self._untried_actions:
            self._is_expanded = True
            return None

        action = self._untried_actions.pop()
        child_state = self.state.clone()
        _state_apply(child_state, action)
        child = MCTSNode(child_state, parent=self, action=action)
        self.children.append(child)

        if not self._untried_actions:
            self._is_expanded = True
        return child

    def expand_all(self, actions_and_priors, max_children=None):
        """Expand children with priors (PUCT mode), optionally pruning low-prior moves."""
        if max_children and len(actions_and_priors) > max_children:
            actions_and_priors.sort(key=lambda x: x[1], reverse=True)
            actions_and_priors = actions_and_priors[:max_children]
            # Renormalize priors
            total = sum(p for _, p in actions_and_priors)
            if total > 0:
                actions_and_priors = [(a, p / total) for a, p in actions_and_priors]
        for action, prior in actions_and_priors:
            child_state = self.state.clone()
            _state_apply(child_state, action)
            child = MCTSNode(child_state, parent=self, action=action, prior=prior)
            self.children.append(child)
        self._is_expanded = True
        self._untried_actions = []

    def is_fully_expanded(self):
        if self._untried_actions is None:
            return False
        return self._is_expanded

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    VIRTUAL_LOSS = 3

    def __init__(self, num_simulations=MCTS_SIMULATIONS, eval_fn=None,
                 batch_size=16, root_noise=True, allow_early_stop=True,
                 c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION,
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
        self.num_simulations = num_simulations
        self.eval_fn = eval_fn or evaluate
        self.c_puct = float(c_puct)
        self.fpu_reduction = float(fpu_reduction)
        self.policy_temperature = float(policy_temperature)
        self.moves_left_utility = bool(moves_left_utility)
        self.moves_left_max_effect = float(moves_left_max_effect)
        self.moves_left_threshold = float(moves_left_threshold)
        self.moves_left_slope = float(moves_left_slope)
        # Leaf-parallel batch width.  Kept small: wide in-tree batching queues many
        # leaves against the same shallow tree and degrades selection quality.  GPU
        # throughput comes from parallelism ACROSS games (workers), not within one
        # tree (REWORK_PLAN.md Phase 1.3).
        self.batch_size = batch_size
        # Dirichlet root noise belongs in self-play generation only.  Arena / eval /
        # benchmark / human play construct with root_noise=False so measurement and
        # play are not perturbed (REWORK_PLAN.md Phase 1.4).
        self.root_noise = root_noise
        # Early stopping truncates the visit distribution that becomes the policy
        # training target, so callers that RECORD data disable it; callers that only
        # need the move (play / arena) keep it (REWORK_PLAN.md Phase 1.5).
        self.allow_early_stop = allow_early_stop
        self._supports_batch = hasattr(self.eval_fn, 'batch_evaluate')
        self._has_policy = hasattr(self.eval_fn, 'evaluate_with_policy')
        # Reuse only across White's first -> second half-move. Both nodes use
        # White perspective, so accumulated Q values remain valid. Reuse across
        # a side change would require rebasing every stored value and is avoided.
        self._white_half_root = None
        self._white_half_key = None

    @staticmethod
    def _state_key(state):
        return (
            state.fen(),
            bool(state.is_white_turn),
            bool(getattr(state, "white_half_pending", False)),
        )

    def _root_for_search(self, root_state):
        key = self._state_key(root_state)
        if self._white_half_root is not None and self._white_half_key == key:
            root = self._white_half_root
            self._white_half_root = None
            self._white_half_key = None
            return root
        self._white_half_root = None
        self._white_half_key = None
        return MCTSNode(root_state.clone())

    def _remember_white_continuation(self, root_state, selected_action,
                                     children_info):
        is_first_white_half = (
            root_state.is_white_turn
            and not getattr(root_state, "white_half_pending", False)
            and not isinstance(selected_action, tuple)
        )
        if not is_first_white_half:
            self._white_half_root = None
            self._white_half_key = None
            return
        selected_child = next(
            (child for child, _key, _visits in children_info
             if child.action == selected_action),
            None,
        )
        if selected_child is None:
            return
        selected_child.parent = None
        selected_child.action = None
        self._white_half_root = selected_child
        self._white_half_key = self._state_key(selected_child.state)

    def _should_stop_early(self, root, sims_done):
        """Stop MCTS early if position is clearly decided or best move dominant."""
        if sims_done < self.num_simulations * 0.3:
            return False  # need minimum exploration
        if not root.children:
            return False
        # Near-terminal value
        if abs(root.q_value) > 0.95:
            return True
        # Best child has insurmountable visit lead
        visits = sorted((c.visit_count for c in root.children), reverse=True)
        if len(visits) >= 2:
            remaining = self.num_simulations - sims_done
            if visits[0] - visits[1] > remaining:
                return True
        return False

    def get_best_action(self, root_state, temperature=1.0):
        """Run MCTS and return (selected_action, action_probs, search_value).

        search_value is the selected child's Q (the search's conclusion about
        the move actually played, side-to-move perspective), falling back to
        the root average only when the child is unvisited."""
        root = self._root_for_search(root_state)

        # Every batching evaluator also exposes a policy head (NNEvaluator,
        # HybridEvaluator); the heuristic evaluator is a plain function with
        # neither. So batching without policy priors has no reachable caller.
        if self._has_policy and self._supports_batch:
            self._run_batched_puct(root)
        else:
            self._run_sequential(root)

        if not root.children:
            self._white_half_root = None
            self._white_half_key = None
            return None, {}, 0.0

        # Collect visit counts
        children_info = []
        for child in root.children:
            key = self._action_key(child.action, root_state.is_white_turn)
            children_info.append((child, key, child.visit_count))

        # Build action_probs from the visit distribution.  POLICY_TARGET_PSEUDOCOUNT
        # is a *fraction of total visits* spread uniformly (0 = raw visit counts, the
        # AlphaZero target).  This keeps smoothing proportional to search effort
        # instead of swamping low-sim targets (REWORK_PLAN.md Phase 1.6).
        total_visits = sum(info[2] for info in children_info)
        pseudo_frac = max(0.0, float(POLICY_TARGET_PSEUDOCOUNT))
        pseudo_total = pseudo_frac * total_visits
        per_child = pseudo_total / len(children_info) if children_info else 0.0
        denom = total_visits + pseudo_total
        if denom <= 0:
            uniform = 1.0 / len(children_info)
            action_probs = {info[1]: uniform for info in children_info}
        else:
            action_probs = {info[1]: (info[2] + per_child) / denom for info in children_info}

        # Temperature-based selection over oscillation-adjusted visit counts
        # (action_probs above keeps the raw distribution for training).
        adj_visits = _oscillation_adjusted_visits(root_state, children_info)
        if temperature < 0.01:
            best_i = max(range(len(children_info)), key=lambda i: adj_visits[i])
            selected_action = children_info[best_i][0].action
        else:
            weights = [max(v, 0) ** (1.0 / temperature) for v in adj_visits]
            total_w = sum(weights)
            if total_w == 0:
                selected_action = random.choice(children_info)[0].action
            else:
                probs = [w / total_w for w in weights]
                idx = random.choices(range(len(children_info)), weights=probs, k=1)[0]
                selected_action = children_info[idx][0].action

        selected_action = _white_first_half_override(root_state, selected_action,
                                                     children_info)
        selected_action = _king_safety_override(root_state, selected_action,
                                                children_info)
        self._remember_white_continuation(
            root_state, selected_action, children_info)
        return selected_action, action_probs, _selected_child_value(
            children_info, selected_action, root.q_value)

    # ------------------------------------------------------------------
    # Sequential MCTS (heuristic eval, UCB1)
    # ------------------------------------------------------------------

    def _run_sequential(self, root):
        for i in range(self.num_simulations):
            node = self._select_ucb(root)
            leaf, value = self._evaluate_and_expand_ucb(node)
            self._backpropagate(leaf, value)
            if (self.allow_early_stop and i % 32 == 31
                    and self._should_stop_early(root, i + 1)):
                break

    def _select_ucb(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            if not node.children:
                return node  # fully expanded with no legal moves
            node = node.best_child_ucb()
        return node

    def _evaluate_and_expand_ucb(self, node):
        if node.state.is_terminal():
            return node, node.state.get_result()
        child = node.expand_one()
        if child is None:
            return node, self.eval_fn(node.state)
        if child.state.is_terminal():
            return child, child.state.get_result()
        return child, self.eval_fn(child.state)

    # ------------------------------------------------------------------
    # Batched MCTS without policy (NN value only, UCB1)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Batched MCTS with PUCT + policy head
    # ------------------------------------------------------------------

    def _add_dirichlet_noise(self, node):
        """Mix Dirichlet noise into root children's priors for exploration."""
        if not node.children:
            return
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(node.children))
        eps = DIRICHLET_EPSILON
        for child, n in zip(node.children, noise):
            child.prior = (1 - eps) * child.prior + eps * n

    def _run_batched_puct(self, root):
        """PUCT-based MCTS with batched NN evaluation and policy priors.

        The root is expanded and backpropagated synchronously first, so the batch
        loop always descends into real children.  During batch collection a
        `pending` set tracks selected nodes; re-selecting a pending node means the
        current frontier is exhausted, so the batch is processed early rather than
        padded with duplicate work (virtual loss cannot diversify an unexpanded
        frontier).  Simulations are counted by completed backpropagations, never by
        selection attempts (REWORK_PLAN.md Phase 1.1-1.2).
        """
        if root.state.is_terminal():
            self._backpropagate(root, root.state.get_result(), moves_left=0.0)
            return

        if (self.moves_left_utility
                and hasattr(self.eval_fn,
                            "evaluate_with_policy_and_moves_left")):
            root_value, root_policy, root_moves_left = (
                self.eval_fn.evaluate_with_policy_and_moves_left(root.state))
        else:
            root_value, root_policy = self.eval_fn.evaluate_with_policy(root.state)
            root_moves_left = None
        if not root.is_fully_expanded():
            self._expand_with_policy(root, root_policy)
        self._backpropagate(root, root_value, moves_left=root_moves_left)
        if self.root_noise and root.children:
            self._add_dirichlet_noise(root)

        sims_done = 1  # the root evaluation is one simulation
        while sims_done < self.num_simulations:
            if self.allow_early_stop and self._should_stop_early(root, sims_done):
                break

            target = min(self.batch_size, self.num_simulations - sims_done)
            leaves = []          # (node, needs_nn, immediate_value_or_None)
            pending = set()
            while len(leaves) < target:
                node = self._select_puct(root)
                if id(node) in pending:
                    break  # frontier exhausted for this batch
                pending.add(id(node))
                self._apply_virtual_loss(node)
                if node.state.is_terminal():
                    leaves.append((node, False, node.state.get_result()))
                elif node.is_fully_expanded():
                    # Selection stopped at an expanded node with no descendable
                    # child (e.g. no legal moves): evaluate in place.
                    leaves.append((node, False, self.eval_fn(node.state)))
                else:
                    leaves.append((node, True, None))

            nn_nodes = [n for (n, needs, _) in leaves if needs]
            nn_results = {}
            if nn_nodes:
                states = [n.state for n in nn_nodes]
                if (self.moves_left_utility
                        and hasattr(self.eval_fn,
                                    "batch_evaluate_with_policy_and_moves_left")):
                    values, policies, moves_left = (
                        self.eval_fn.batch_evaluate_with_policy_and_moves_left(states))
                else:
                    values, policies = self.eval_fn.batch_evaluate_with_policy(states)
                    moves_left = [None] * len(states)
                for n, v, p, ml in zip(nn_nodes, values, policies, moves_left):
                    nn_results[id(n)] = (v, p, ml)

            for node, needs, imm in leaves:
                self._revert_virtual_loss(node)
                if needs:
                    value, policy, moves_left = nn_results[id(node)]
                    if not node.is_fully_expanded():
                        self._expand_with_policy(node, policy)
                    self._backpropagate(node, value, moves_left=moves_left)
                else:
                    terminal_ml = 0.0 if node.state.is_terminal() else None
                    self._backpropagate(node, imm, moves_left=terminal_ml)

            sims_done += len(leaves)

    def _select_puct(self, node):
        """Walk down tree using PUCT until we hit a terminal or unexpanded node."""
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            if not node.children:
                return node  # fully expanded with no legal moves
            node = node.best_child_puct(c_puct=self.c_puct,
                                        fpu_reduction=self.fpu_reduction,
                                        moves_left_max_effect=(
                                            self.moves_left_max_effect
                                            if self.moves_left_utility else 0.0),
                                        moves_left_threshold=self.moves_left_threshold,
                                        moves_left_slope=self.moves_left_slope)
        return node

    def _expand_with_policy(self, node, policy_logits):
        """Expand all children of a node using policy logits for priors.

        Half-move search yields single Move objects for every ply (both White halves
        and Black), so priors come straight from the policy head via move_to_index.
        The legacy atomic path yields (m1, m2) tuples for White; those keep the old
        marginalized-P(m1) priors and the move-count cap.
        """
        actions = _state_legal_actions(node.state)
        if not actions:
            node._is_expanded = True
            node._untried_actions = []
            return

        is_pair = isinstance(actions[0], tuple)

        if policy_logits is None:
            uniform = 1.0 / len(actions)
            max_ch = 80 if is_pair else None
            node.expand_all([(a, uniform) for a in actions], max_children=max_ch)
            return

        if is_pair:
            actions_and_priors = self._white_priors(actions, policy_logits)
            node.expand_all(actions_and_priors, max_children=80)
        else:
            actions_and_priors = self._single_move_priors(actions, policy_logits)
            node.expand_all(actions_and_priors, max_children=None)

    def _single_move_priors(self, legal_actions, policy_logits):
        """Priors for single-move plies (Black, and either White half-move)."""
        from encoding import move_to_policy_index

        promotion_aware = len(policy_logits) > POLICY_SIZE
        indices = [move_to_policy_index(m, promotion_aware)
                   for m in legal_actions]
        probs = _softmax_masked(policy_logits, indices,
                                temperature=self.policy_temperature)
        return [(move, probs.get(idx, 1.0 / len(legal_actions)))
                for move, idx in zip(legal_actions, indices)]

    def _white_priors(self, legal_actions, policy_logits):
        """Priors for White's atomic (m1, m2) pairs (legacy / fallback path).

        Uses P(m1) from the policy head, distributed uniformly across the m2
        continuations for each m1:  P(m1, m2) = P(m1) / |m2s|.
        """
        from encoding import move_to_policy_index

        m1_groups = defaultdict(list)
        promotion_aware = len(policy_logits) > POLICY_SIZE
        for m1, m2 in legal_actions:
            m1_groups[move_to_policy_index(m1, promotion_aware)].append((m1, m2))

        m1_probs = _softmax_masked(policy_logits, list(m1_groups.keys()),
                                   temperature=self.policy_temperature)

        actions_and_priors = []
        for m1_idx, pairs in m1_groups.items():
            p_m1 = m1_probs.get(m1_idx, 1.0 / len(m1_groups))
            p_each = p_m1 / len(pairs)
            for pair in pairs:
                actions_and_priors.append((pair, p_each))

        return actions_and_priors

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _apply_virtual_loss(self, node):
        vl = self.VIRTUAL_LOSS
        n = node
        while n is not None:
            n.visit_count += vl
            n.total_value -= vl
            n = n.parent

    def _revert_virtual_loss(self, node):
        vl = self.VIRTUAL_LOSS
        n = node
        while n is not None:
            n.visit_count -= vl
            n.total_value += vl
            n = n.parent

    def _backpropagate(self, node, value, moves_left=None):
        """Propagate value (from White's perspective) back up to root.

        A node's total_value accumulates in the perspective of the side that moved
        into it (its parent's side-to-move), so Q is read consistently by that
        parent during selection.
        """
        distance = 0.0
        while node is not None:
            node.visit_count += 1
            if node.parent is not None:
                parent_is_white = node.parent.state.is_white_turn
                node.total_value += value if parent_is_white else -value
            else:
                is_white = node.state.is_white_turn
                node.total_value += value if is_white else -value
            if moves_left is not None and math.isfinite(float(moves_left)):
                node.total_moves_left += max(0.0, float(moves_left)) + distance
                node.moves_left_count += 1
            node = node.parent
            distance += 1.0

    @staticmethod
    def _action_key(action, is_white):
        """Stable string key for an action (atomic pair or single half-move)."""
        if isinstance(action, tuple):
            m1, m2 = action
            return f"{m1.uci()},{m2.uci()}"
        return action.uci()
