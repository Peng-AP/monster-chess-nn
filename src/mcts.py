import math
import random
from collections import defaultdict

import numpy as np

from config import (
    EXPLORATION_CONSTANT, C_PUCT, MCTS_SIMULATIONS, POLICY_SIZE,
    DIRICHLET_ALPHA, DIRICHLET_EPSILON, FPU_REDUCTION, POLICY_TARGET_PSEUDOCOUNT,
)
from evaluation import evaluate


def _softmax_masked(logits, indices):
    """Softmax over a subset of logit indices, returning {index: prob}."""
    if not indices:
        return {}
    vals = np.array([logits[i] for i in indices], dtype=np.float64)
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
        "_untried_actions", "_is_expanded",
    )

    def __init__(self, state, parent=None, action=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children = []
        self._untried_actions = None
        self._is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

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

    def puct_score(self, c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION):
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
        return self.q_value + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

    def best_child_puct(self, c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION):
        return max(self.children, key=lambda ch: ch.puct_score(c_puct, fpu_reduction=fpu_reduction))

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
                 batch_size=16, root_noise=True, allow_early_stop=True):
        self.num_simulations = num_simulations
        self.eval_fn = eval_fn or evaluate
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
        """Run MCTS and return (selected_action, action_probs, root_value)."""
        root = MCTSNode(root_state.clone())

        if self._has_policy and self._supports_batch:
            self._run_batched_puct(root)
        elif self._supports_batch:
            self._run_batched(root)
        else:
            self._run_sequential(root)

        if not root.children:
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

        # Temperature-based selection
        if temperature < 0.01:
            best = max(children_info, key=lambda x: x[2])
            selected_action = best[0].action
        else:
            weights = [max(info[2], 0) ** (1.0 / temperature) for info in children_info]
            total_w = sum(weights)
            if total_w == 0:
                selected_action = random.choice(children_info)[0].action
            else:
                probs = [w / total_w for w in weights]
                idx = random.choices(range(len(children_info)), weights=probs, k=1)[0]
                selected_action = children_info[idx][0].action

        return selected_action, action_probs, root.q_value

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

    def _run_batched(self, root):
        sims_done = 0
        while sims_done < self.num_simulations:
            if self.allow_early_stop and self._should_stop_early(root, sims_done):
                break
            batch_needs_eval = []
            batch_terminal = []
            batch_count = min(self.batch_size, self.num_simulations - sims_done)

            for _ in range(batch_count):
                node = self._select_ucb(root)
                if node.state.is_terminal():
                    batch_terminal.append((node, node.state.get_result()))
                    continue
                child = node.expand_one()
                if child is None:
                    batch_terminal.append((node, self.eval_fn(node.state)))
                    continue
                if child.state.is_terminal():
                    batch_terminal.append((child, child.state.get_result()))
                else:
                    self._apply_virtual_loss(child)
                    batch_needs_eval.append(child)

            if batch_needs_eval:
                states = [leaf.state for leaf in batch_needs_eval]
                values = self.eval_fn.batch_evaluate(states)
                for leaf, value in zip(batch_needs_eval, values):
                    self._revert_virtual_loss(leaf)
                    self._backpropagate(leaf, value)

            for leaf, value in batch_terminal:
                self._backpropagate(leaf, value)
            sims_done += batch_count

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
            self._backpropagate(root, root.state.get_result())
            return

        root_value, root_policy = self.eval_fn.evaluate_with_policy(root.state)
        if not root.is_fully_expanded():
            self._expand_with_policy(root, root_policy)
        self._backpropagate(root, root_value)
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
                values, policies = self.eval_fn.batch_evaluate_with_policy(states)
                for n, v, p in zip(nn_nodes, values, policies):
                    nn_results[id(n)] = (v, p)

            for node, needs, imm in leaves:
                self._revert_virtual_loss(node)
                if needs:
                    value, policy = nn_results[id(node)]
                    if not node.is_fully_expanded():
                        self._expand_with_policy(node, policy)
                    self._backpropagate(node, value)
                else:
                    self._backpropagate(node, imm)

            sims_done += len(leaves)

    def _select_puct(self, node):
        """Walk down tree using PUCT until we hit a terminal or unexpanded node."""
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            if not node.children:
                return node  # fully expanded with no legal moves
            node = node.best_child_puct(fpu_reduction=FPU_REDUCTION)
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
        from data_processor import move_to_index

        indices = [move_to_index(m) for m in legal_actions]
        probs = _softmax_masked(policy_logits, indices)
        return [(move, probs.get(idx, 1.0 / len(legal_actions)))
                for move, idx in zip(legal_actions, indices)]

    def _white_priors(self, legal_actions, policy_logits):
        """Priors for White's atomic (m1, m2) pairs (legacy / fallback path).

        Uses P(m1) from the policy head, distributed uniformly across the m2
        continuations for each m1:  P(m1, m2) = P(m1) / |m2s|.
        """
        from data_processor import move_to_index

        m1_groups = defaultdict(list)
        for m1, m2 in legal_actions:
            m1_groups[move_to_index(m1)].append((m1, m2))

        m1_probs = _softmax_masked(policy_logits, list(m1_groups.keys()))

        actions_and_priors = []
        for m1_idx, pairs in m1_groups.items():
            p_m1 = m1_probs.get(m1_idx, 1.0 / len(m1_groups))
            p_each = p_m1 / len(pairs)
            for pair in pairs:
                actions_and_priors.append((pair, p_each))

        return actions_and_priors

    # Backwards-compatible alias (older imports / tests may reference this name).
    def _black_priors(self, legal_actions, policy_logits):
        return self._single_move_priors(legal_actions, policy_logits)

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

    def _backpropagate(self, node, value):
        """Propagate value (from White's perspective) back up to root.

        A node's total_value accumulates in the perspective of the side that moved
        into it (its parent's side-to-move), so Q is read consistently by that
        parent during selection.
        """
        while node is not None:
            node.visit_count += 1
            if node.parent is not None:
                parent_is_white = node.parent.state.is_white_turn
                node.total_value += value if parent_is_white else -value
            else:
                is_white = node.state.is_white_turn
                node.total_value += value if is_white else -value
            node = node.parent

    @staticmethod
    def _action_key(action, is_white):
        """Stable string key for an action (atomic pair or single half-move)."""
        if isinstance(action, tuple):
            m1, m2 = action
            return f"{m1.uci()},{m2.uci()}"
        return action.uci()
