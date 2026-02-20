"""Tests for shared encoding utilities."""
import chess
from encoding import move_to_index, mirror_move_index


def test_move_to_index_formula():
    """Index should be from_square * 64 + to_square."""
    move = chess.Move(12, 28)  # e2 -> e4
    assert move_to_index(move) == 12 * 64 + 28


def test_mirror_move_index_a_file_to_h_file():
    """Mirroring a1->a2 should give h1->h2."""
    # a1 = square 0, a2 = square 8
    idx_a1_a2 = 0 * 64 + 8
    mirrored = mirror_move_index(idx_a1_a2)
    # h1 = square 7, h2 = square 15
    idx_h1_h2 = 7 * 64 + 15
    assert mirrored == idx_h1_h2


def test_mirror_move_index_center_unchanged():
    """Mirroring d-file moves: d1->d2 should map to e1->e2."""
    # d1 = square 3 (file 3, rank 0), d2 = square 11 (file 3, rank 1)
    idx = 3 * 64 + 11
    mirrored = mirror_move_index(idx)
    # e1 = square 4 (file 4, rank 0), e2 = square 12 (file 4, rank 1)
    expected = 4 * 64 + 12
    assert mirrored == expected


def test_mirror_roundtrip_all_valid():
    """Mirror applied twice should be identity for a sampling of indices."""
    import random
    random.seed(42)
    for _ in range(100):
        idx = random.randint(0, 4095)
        assert mirror_move_index(mirror_move_index(idx)) == idx
