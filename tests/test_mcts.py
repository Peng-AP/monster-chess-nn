"""Tests for MCTS search components."""
import math
import numpy as np

from config import C_PUCT, FPU_REDUCTION, POLICY_SIZE
from monster_chess import MonsterChessGame
from mcts import MCTSNode, _softmax_masked, MCTS


def test_puct_score_unvisited_is_finite():
    """Unvisited node should return a finite PUCT score."""
    game = MonsterChessGame()
    parent = MCTSNode(game)
    parent.visit_count = 10
    parent.total_value = 5.0
    child_state = game.clone()
    child = MCTSNode(child_state, parent=parent, prior=0.5)
    score = child.puct_score()
    assert math.isfinite(score)


def test_puct_score_visited_is_finite():
    """Visited node should return a finite PUCT score."""
    game = MonsterChessGame()
    parent = MCTSNode(game)
    parent.visit_count = 10
    parent.total_value = 5.0
    child_state = game.clone()
    child = MCTSNode(child_state, parent=parent, prior=0.5)
    child.visit_count = 3
    child.total_value = 1.5
    score = child.puct_score()
    assert math.isfinite(score)


def test_puct_unvisited_higher_than_visited():
    """With same prior, unvisited node should generally score higher
    than a visited node (FPU exploration bonus)."""
    game = MonsterChessGame()
    parent = MCTSNode(game)
    parent.visit_count = 100
    parent.total_value = 50.0  # q_value = 0.5

    child_unvisited = MCTSNode(game.clone(), parent=parent, prior=0.5)
    child_visited = MCTSNode(game.clone(), parent=parent, prior=0.5)
    child_visited.visit_count = 10
    child_visited.total_value = 3.0  # q_value = 0.3

    score_unvisited = child_unvisited.puct_score()
    score_visited = child_visited.puct_score()
    # Unvisited should be higher due to large exploration bonus
    assert score_unvisited > score_visited


def test_softmax_masked_empty():
    """Empty indices should return empty dict."""
    result = _softmax_masked(np.zeros(100), [])
    assert result == {}


def test_softmax_masked_normalization():
    """Probabilities from softmax should sum to ~1."""
    logits = np.random.randn(POLICY_SIZE).astype(np.float32)
    indices = [0, 10, 50, 100, 500]
    probs = _softmax_masked(logits, indices)
    assert len(probs) == len(indices)
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-6


def test_softmax_masked_all_equal():
    """Equal logits should produce uniform probabilities."""
    logits = np.zeros(POLICY_SIZE)
    indices = [0, 1, 2, 3]
    probs = _softmax_masked(logits, indices)
    for idx in indices:
        assert abs(probs[idx] - 0.25) < 1e-6


def test_mcts_heuristic_returns_valid_action():
    """MCTS with heuristic eval should return a valid action."""
    game = MonsterChessGame()
    mcts = MCTS(num_simulations=10, eval_fn=None)
    action, action_probs, root_value = mcts.get_best_action(game, temperature=0.1)
    assert action is not None
    # White action should be a tuple
    assert isinstance(action, tuple)
    assert len(action) == 2


def test_mcts_near_terminal():
    """MCTS on a near-terminal position should still return an action."""
    # White king + pawn vs Black king only — very simple
    fen = "4k3/3P4/8/8/8/8/8/4K3 w - - 0 1"
    game = MonsterChessGame(fen=fen)
    mcts = MCTS(num_simulations=10, eval_fn=None)
    action, action_probs, root_value = mcts.get_best_action(game, temperature=0.1)
    assert action is not None
