import math
import random
from collections import defaultdict

import numpy as np

from config import (
    EXPLORATION_CONSTANT, C_PUCT, MCTS_SIMULATIONS, POLICY_SIZE,
    DIRICHLET_ALPHA, DIRICHLET_EPSILON, FPU_REDUCTION,
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
            # FPU: start unvisited child Q slightly below parent Q.
            if self.parent is None:
                fpu_q = 0.0
            else:
                fpu_q = max(-1.0, min(1.0, self.parent.q_value - fpu_reduction))
            return fpu_q + c_puct * self.prior * math.sqrt(parent_visits)
        return self.q_value + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

    def best_child_puct(self, c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION):
        return max(self.children, key=lambda ch: ch.puct_score(c_puct, fpu_reduction=fpu_reduction))

    # --- Expansion ---

    def expand_one(self):
        """Expand one untried action (UCB1 mode). Returns new child or None."""
        if self._untried_actions is None:
            self._untried_actions = list(self.state.get_legal_actions())
            random.shuffle(self._untried_actions)

        if not self._untried_actions:
            self._is_expanded = True
            return None

        action = self._untried_actions.pop()
        child_state = self.state.clone()
        child_state.apply_action(action)
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
            child_state.apply_action(action)
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
                 batch_size=128):
        self.num_simulations = num_simulations
        self.eval_fn = eval_fn or evaluate
        self.batch_size = batch_size
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

        # Build action_probs from visit distribution
        total_visits = sum(info[2] for info in children_info)
        if total_visits == 0:
            total_visits = 1
        action_probs = {info[1]: info[2] / total_visits for info in children_info}

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
            if i % 32 == 31 and self._should_stop_early(root, i + 1):
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
        while sims_done < self.num_simulations and not self._should_stop_early(root, sims_done):
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

        Flow per simulation:
          1. Select leaf via PUCT scores
          2. If leaf is unexpanded: add to batch for NN eval
          3. After batch eval: expand leaf with policy priors, backprop value
        """
        noise_added = False
        sims_done = 0
        while sims_done < self.num_simulations and not self._should_stop_early(root, sims_done):
            batch_leaves = []       # unexpanded leaves needing NN eval
            batch_terminal = []     # (leaf, value)
            batch_count = min(self.batch_size, self.num_simulations - sims_done)

            for _ in range(batch_count):
                node = self._select_puct(root)

                if node.state.is_terminal():
                    batch_terminal.append((node, node.state.get_result()))
                    continue

                if node.is_fully_expanded():
                    # Fully expanded but all children terminal or similar
                    batch_terminal.append((node, self.eval_fn(node.state)))
                    continue

                # Node needs expansion — queue for NN eval
                self._apply_virtual_loss(node)
                batch_leaves.append(node)

            # Batch evaluate all leaves
            if batch_leaves:
                states = [leaf.state for leaf in batch_leaves]
                values, policies = self.eval_fn.batch_evaluate_with_policy(states)

                for leaf, value, policy in zip(batch_leaves, values, policies):
                    self._revert_virtual_loss(leaf)

                    # Expand with policy priors (if not already expanded
                    # by a concurrent batch element — race guard)
                    if not leaf.is_fully_expanded():
                        self._expand_with_policy(leaf, policy)

                    self._backpropagate(leaf, value)

                # Add Dirichlet noise to root after its first expansion
                if not noise_added and root.children:
                    self._add_dirichlet_noise(root)
                    noise_added = True

            for leaf, value in batch_terminal:
                self._backpropagate(leaf, value)

            sims_done += batch_count

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
        """Expand all children of a node using policy logits for priors."""
        from data_processor import move_to_index

        actions = node.state.get_legal_actions()
        if not actions:
            node._is_expanded = True
            node._untried_actions = []
            return

        is_white = node.state.is_white_turn

        # Cap White's double-move children to top 80 by prior (move count pruning)
        max_ch = 80 if is_white else None

        if policy_logits is None:
            # No policy available — use uniform priors
            uniform = 1.0 / len(actions)
            node.expand_all([(a, uniform) for a in actions], max_children=max_ch)
            return

        if is_white:
            actions_and_priors = self._white_priors(actions, policy_logits)
        else:
            actions_and_priors = self._black_priors(actions, policy_logits)

        node.expand_all(actions_and_priors, max_children=max_ch)

    def _white_priors(self, legal_actions, policy_logits):
        """Compute priors for White's (m1, m2) pairs.

        Uses P(m1) from the policy head, distributed uniformly across
        the m2 continuations for each m1:  P(m1, m2) = P(m1) / |m2s|.
        """
        from data_processor import move_to_index

        # Group pairs by m1 index
        m1_groups = defaultdict(list)
        for m1, m2 in legal_actions:
            m1_groups[move_to_index(m1)].append((m1, m2))

        # P(m1) via masked softmax over legal first moves
        m1_probs = _softmax_masked(policy_logits, list(m1_groups.keys()))

        # Distribute each P(m1) uniformly among its m2s
        actions_and_priors = []
        for m1_idx, pairs in m1_groups.items():
            p_m1 = m1_probs.get(m1_idx, 1.0 / len(m1_groups))
            p_each = p_m1 / len(pairs)
            for pair in pairs:
                actions_and_priors.append((pair, p_each))

        return actions_and_priors

    def _black_priors(self, legal_actions, policy_logits):
        """Compute priors for Black's single moves from policy logits."""
        from data_processor import move_to_index

        indices = [move_to_index(m) for m in legal_actions]
        probs = _softmax_masked(policy_logits, indices)

        return [(move, probs.get(idx, 1.0 / len(legal_actions)))
                for move, idx in zip(legal_actions, indices)]

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
        """Propagate value (from White's perspective) back up to root."""
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
        if is_white:
            m1, m2 = action
            return f"{m1.uci()},{m2.uci()}"
        else:
            return action.uci()
