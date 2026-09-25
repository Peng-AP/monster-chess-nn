"""Training/export/native parity for the search-first value model."""
import sys
from pathlib import Path
import tempfile
import unittest
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in ('src', 'native', 'tools'):
    sys.path.insert(0, str(ROOT/folder))
import monster_native as native
from encoding import fen_to_tensor
from train_search_value import model, features, export
from distill_search_value import teacher_input
from monster_chess import MonsterChessGame


class CheapValueTests(unittest.TestCase):
    def test_teacher_encoding_legacy_plane_is_not_signed_rank(self):
        fen = 'rnbqkbnr/pppppppp/8/8/8/3P4/2P1PP2/4K3 w kq - 0 1'
        p = fen_to_tensor(fen, True, False, 24, 0)[None]
        for channels in (15,17,24):
            expected = fen_to_tensor(fen, True, False, channels, 0).transpose(2,0,1)[None]
            np.testing.assert_array_equal(teacher_input(p, channels), expected)
    def test_features_and_export_match_pytorch(self):
        cases = [
            ('rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1', False, 0),
            ('r3k2r/8/8/3pP3/8/8/8/4K3 w kq d6 0 1', True, 55),
            ('r3k2r/8/8/8/8/8/2PPPP2/4K3 b kq - 0 1', False, 149),
            ('8/8/8/8/8/8/3P4/4K3 w - - 0 1', False, 150),
        ]
        torch.manual_seed(73)
        net = model(128, 32).eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'weights.bin'
            export(net, path)
            evaluator = native.CheapValue(str(path))
            for fen, pending, count in cases:
                tensor = fen_to_tensor(fen, fen.split()[1]=='w', pending, 24, count)
                expected = features(tensor[None])[0]
                actual = np.zeros(840, dtype=np.float32)
                for i, value in native.CheapValue.features(fen, pending, count):
                    actual[i] = value
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
                with torch.no_grad():
                    value = net(torch.from_numpy(expected)).item()
                self.assertAlmostEqual(evaluator.evaluate(fen, pending, count), value, places=6)
            result = native.alphabeta_search(cases[0][0], seconds=.02, evaluator=evaluator)
            self.assertIsNotNone(result.action)

    def test_invalid_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bad.bin'
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                native.CheapValue(str(path))

    def test_random_play_feature_parity(self):
        rng = random.Random(9183)
        game = MonsterChessGame()
        for _ in range(100):
            if game.is_terminal():
                game = MonsterChessGame()
            fen = game.board.fen(en_passant='fen')
            pending, count = game.white_half_pending, game.turn_count
            expected = features(fen_to_tensor(fen, game.is_white_turn, pending, 24, count)[None])[0]
            actual = np.zeros(840, dtype=np.float32)
            for i, value in native.CheapValue.features(fen, pending, count):
                actual[i] = value
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
            actions = game.get_search_actions()
            if not actions:
                game = MonsterChessGame()
            else:
                game.apply_search_action(rng.choice(actions))
