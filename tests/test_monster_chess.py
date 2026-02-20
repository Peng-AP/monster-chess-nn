"""Tests for Monster Chess game rules."""
import chess
from monster_chess import MonsterChessGame
from config import STARTING_FEN, MAX_GAME_TURNS


def test_initial_state_not_terminal():
    game = MonsterChessGame()
    assert not game.is_terminal()


def test_initial_state_is_white_turn():
    game = MonsterChessGame()
    assert game.is_white_turn


def test_white_actions_are_tuples():
    game = MonsterChessGame()
    actions = game.get_legal_actions()
    assert len(actions) > 0
    for action in actions:
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert isinstance(action[0], chess.Move)
        assert isinstance(action[1], chess.Move)


def test_black_actions_are_single_moves():
    game = MonsterChessGame()
    # Apply one White action to get to Black's turn
    actions = game.get_legal_actions()
    game.apply_action(actions[0])
    assert not game.is_white_turn
    black_actions = game.get_legal_actions()
    assert len(black_actions) > 0
    for action in black_actions:
        assert isinstance(action, chess.Move)


def test_king_capture_ends_game():
    # Set up a position where White can capture Black's king
    # White king on e1, White pawn on d7 (about to capture king on e8)
    fen = "4k3/3P4/8/8/8/8/8/4K3 w - - 0 1"
    game = MonsterChessGame(fen=fen)
    actions = game.get_legal_actions()
    # Find an action that captures the Black king
    king_capture = None
    for m1, m2 in actions:
        game_clone = game.clone()
        game_clone.apply_action((m1, m2))
        if game_clone.board.king(chess.BLACK) is None:
            king_capture = (m1, m2)
            break
    assert king_capture is not None, "Should find a king capture action"
    game.apply_action(king_capture)
    assert game.is_terminal()
    assert game.get_result() == 1  # White wins


def test_turn_limit_produces_draw():
    game = MonsterChessGame()
    game.turn_count = MAX_GAME_TURNS
    assert game.is_terminal()
    assert game.get_result() == 0


def test_clone_independence():
    game = MonsterChessGame()
    clone = game.clone()
    # Mutate clone
    actions = clone.get_legal_actions()
    clone.apply_action(actions[0])
    # Original unchanged
    assert game.is_white_turn
    assert game.turn_count == 0
    assert not game.is_terminal()


def test_white_self_preservation():
    """White should not leave its own king attacked (unless forced)."""
    game = MonsterChessGame()
    actions = game.get_legal_actions()
    for m1, m2 in actions:
        clone = game.clone()
        clone.apply_action((m1, m2))
        wk = clone.board.king(chess.WHITE)
        if wk is not None:
            # After the action, White's king should not be attacked
            # (unless this is a forced blunder, which we skip here)
            clone.board.turn = chess.BLACK
            if clone.board.is_attacked_by(chess.BLACK, wk):
                # This pair was in the "forced blunder" set —
                # verify no safe pairs exist
                safe_actions = [
                    a for a in actions
                    if _is_safe_action(game, a)
                ]
                assert len(safe_actions) == 0, (
                    "Found unsafe action when safe alternatives exist"
                )


def _is_safe_action(game, action):
    """Check if a White action leaves its king safe."""
    clone = game.clone()
    clone.apply_action(action)
    wk = clone.board.king(chess.WHITE)
    if wk is None:
        return True  # King gone means different issue
    clone.board.turn = chess.BLACK
    return not clone.board.is_attacked_by(chess.BLACK, wk)


def test_apply_random_action_advances_state():
    game = MonsterChessGame()
    result = game.apply_random_action()
    assert result is True
    assert not game.is_white_turn
    assert game.turn_count == 1


def test_action_to_str_roundtrip():
    """action_to_str and str_to_action should roundtrip."""
    game = MonsterChessGame()
    actions = game.get_legal_actions()
    for action in actions[:5]:
        s = MonsterChessGame.action_to_str(action, is_white=True)
        recovered = MonsterChessGame.str_to_action(s, is_white=True)
        assert recovered == action
