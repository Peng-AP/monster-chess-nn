import math
import random
from config import EXPLORATION_CONSTANT, MCTS_SIMULATIONS
from evaluation import evaluate


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

    def ucb_score(self, c=EXPLORATION_CONSTANT):
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.q_value
        exploration = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration

    def best_child(self, c=EXPLORATION_CONSTANT):
        return max(self.children, key=lambda ch: ch.ucb_score(c))

    def expand(self):
        """Expand one untried action and return the new child node."""
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

    def is_fully_expanded(self):
        if self._untried_actions is None:
            return False
        return self._is_expanded

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, num_simulations=MCTS_SIMULATIONS, eval_fn=None):
        self.num_simulations = num_simulations
        self.eval_fn = eval_fn or evaluate

    def search(self, root_state):
        """Run MCTS from root_state and return (action_probs, root_value).

        action_probs: dict mapping action -> visit proportion
        root_value: average value at root from White's perspective
        """
        root = MCTSNode(root_state.clone())

        for _ in range(self.num_simulations):
            node = self._select(root)
            leaf, value = self._evaluate_and_expand(node)
            self._backpropagate(leaf, value)

        # Build action -> visit count mapping
        action_visits = {}
        for child in root.children:
            key = self._action_key(child.action, root_state.is_white_turn)
            action_visits[key] = child.visit_count

        total_visits = sum(action_visits.values())
        if total_visits == 0:
            return {}, 0.0

        action_probs = {a: v / total_visits for a, v in action_visits.items()}
        root_value = root.q_value
        return action_probs, root_value

    def get_best_action(self, root_state, temperature=1.0):
        """Run search and select an action using temperature-based sampling.

        Returns (selected_action, action_probs, root_value).
        """
        root = MCTSNode(root_state.clone())

        for _ in range(self.num_simulations):
            node = self._select(root)
            leaf, value = self._evaluate_and_expand(node)
            self._backpropagate(leaf, value)

        if not root.children:
            return None, {}, 0.0

        # Collect visit counts
        children_info = []
        for child in root.children:
            key = self._action_key(child.action, root_state.is_white_turn)
            children_info.append((child, key, child.visit_count))

        # Build action_probs from visit distribution
        total_visits = sum(info[2] for info in children_info)
        action_probs = {info[1]: info[2] / total_visits for info in children_info}

        # Temperature-based selection
        if temperature < 0.01:
            # Greedy: pick the most visited
            best = max(children_info, key=lambda x: x[2])
            selected_action = best[0].action
        else:
            # Sample proportional to visit_count^(1/temperature)
            weights = [info[2] ** (1.0 / temperature) for info in children_info]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            idx = random.choices(range(len(children_info)), weights=probs, k=1)[0]
            selected_action = children_info[idx][0].action

        root_value = root.q_value
        return selected_action, action_probs, root_value

    # ------------------------------------------------------------------
    # Internal MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node):
        """Walk down the tree using UCB1 until we reach a leaf or unexpanded node."""
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def _evaluate_and_expand(self, node):
        """Expand one child (if possible) and evaluate the resulting state.

        Returns (leaf_node, value) — leaf_node is the node to backprop from.
        """
        if node.state.is_terminal():
            return node, node.state.get_result()

        child = node.expand()
        if child is None:
            return node, self.eval_fn(node.state)

        if child.state.is_terminal():
            return child, child.state.get_result()

        return child, self.eval_fn(child.state)

    def _backpropagate(self, node, value):
        """Propagate the evaluation value back up to the root.

        Value is always from White's perspective, so we store it directly.
        """
        while node is not None:
            node.visit_count += 1
            # Value is from White's perspective.
            # White nodes want to maximise, Black nodes want to minimise.
            # We store value from White's perspective and UCB handles
            # the sign flip via the parent's perspective.
            # For correct UCB: negate value for Black's nodes.
            if node.parent is not None:
                # The node was reached by the *parent's* player making a move.
                # If parent is White, this node represents a state after White moved,
                # so White wants high value → store as-is.
                # If parent is Black, Black wants low value → negate.
                parent_is_white = node.parent.state.is_white_turn
                node.total_value += value if parent_is_white else -value
            else:
                # Root node
                is_white = node.state.is_white_turn
                node.total_value += value if is_white else -value
            node = node.parent

    @staticmethod
    def _action_key(action, is_white):
        """Convert an action to a hashable string key."""
        if is_white:
            m1, m2 = action
            return f"{m1.uci()},{m2.uci()}"
        else:
            return action.uci()
