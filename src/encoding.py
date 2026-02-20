"""Shared move encoding utilities used by data_processor and mcts."""


def move_to_index(move):
    """Convert a chess.Move to a flat policy index (from_sq * 64 + to_sq)."""
    return move.from_square * 64 + move.to_square


def mirror_move_index(idx):
    """Mirror a flat policy index across the file axis (a<->h)."""
    from_sq = idx // 64
    to_sq = idx % 64
    from_file, from_rank = from_sq % 8, from_sq // 8
    to_file, to_rank = to_sq % 8, to_sq // 8
    new_from = from_rank * 8 + (7 - from_file)
    new_to = to_rank * 8 + (7 - to_file)
    return new_from * 64 + new_to
