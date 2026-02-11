import random


def evaluate(game_state, max_depth=50):
    """Evaluate a position by playing random moves to completion.

    Returns a value in {-1, 0, +1} from White's perspective.
    Designed to be swappable with a neural-network evaluation later.
    """
    state = game_state.clone()
    for _ in range(max_depth):
        if state.is_terminal():
            break
        actions = state.get_legal_actions()
        if not actions:
            break
        state.apply_action(random.choice(actions))

    return float(state.get_result())
