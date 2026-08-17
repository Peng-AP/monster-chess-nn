"""Side weighting: making "does Black benefit from capacity?" testable.

White moves twice per turn, so every turn emits two White half-move records
against Black's one and the corpus is ~64/36 White/Black **by construction**.
Anything added to the shared trunk -- capacity, epochs, a wider policy head --
is therefore spent mostly on White, because that is where the loss reduction
is. Arm C added a 2.74x tower to unbalanced data, White improved more, and
that outcome was predicted by the imbalance alone: it tested nothing about
whether Black benefits from capacity (owner hypothesis, 2026-08-02).

`--black-weight` scales Black-to-move records so the sides carry comparable
gradient. It is off by default (1.0), so every existing corpus is unaffected.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_processor as dp  # noqa: E402

WHITE_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"
BLACK_FEN = "rnbqkbnr/pppppppp/8/8/2P1P3/8/3P1P2/4K3 b kq - 0 1"


def rec(fen, side, **extra):
    base = {"fen": fen, "current_player": side, "mcts_value": 0.0,
            "game_result": -1,
            "policy": {"e1f1,f2f3": 1.0} if side == "white" else {"e7e5": 1.0}}
    base.update(extra)
    return base


def weights(records, black_weight):
    out = dp._convert_games_to_arrays([{"records": records}], augment=False,
                                      black_weight=black_weight)
    return out[4].tolist(), out[5].tolist()   # policy, value


class TestDefaultIsOff(unittest.TestCase):
    def test_weight_one_leaves_everything_alone(self):
        recs = [rec(WHITE_FEN, "white"), rec(BLACK_FEN, "black")]
        pol, val = weights(recs, 1.0)
        self.assertEqual(pol, [1.0, 1.0])
        self.assertEqual(val, [1.0, 1.0])

    def test_balanced_constant_matches_the_measured_imbalance(self):
        # combined_v19_K is 64.4/35.6, and 64.4/35.6 = 1.81; 1.75 is the
        # rounded figure the corpora actually measure at.
        self.assertAlmostEqual(dp.BLACK_WEIGHT_BALANCED, 1.75, places=2)


class TestScaling(unittest.TestCase):
    def test_only_black_records_scale(self):
        recs = [rec(WHITE_FEN, "white"), rec(BLACK_FEN, "black")]
        pol, val = weights(recs, 1.75)
        self.assertEqual(pol, [1.0, 1.75])
        self.assertEqual(val, [1.0, 1.75])

    def test_a_masked_record_stays_masked(self):
        # Scaling, not assignment: 0 * 1.75 is still 0. If this multiplied a
        # constant in instead, every non-teacher Black record would silently
        # become a teacher -- undoing the echo-chamber fix (law 2).
        recs = [rec(BLACK_FEN, "black", policy_weight=0.0)]
        pol, _val = weights(recs, 1.75)
        self.assertEqual(pol, [0.0])

    def test_explicit_value_weight_is_scaled_not_replaced(self):
        recs = [rec(BLACK_FEN, "black", value_weight=0.5)]
        _pol, val = weights(recs, 2.0)
        self.assertEqual(val, [1.0])

    def test_value_weight_zero_survives_scaling(self):
        # ps_monster records merged as policy-only teachers must not acquire
        # value gradient just because they are Black-to-move.
        recs = [rec(BLACK_FEN, "black", value_weight=0.0)]
        _pol, val = weights(recs, 1.75)
        self.assertEqual(val, [0.0])

    def test_mirror_augmentation_keeps_the_weight(self):
        recs = [rec(BLACK_FEN, "black")]
        out = dp._convert_games_to_arrays([{"records": recs}], augment=True,
                                          black_weight=1.75)
        self.assertEqual(out[4].tolist(), [1.75, 1.75])
        self.assertEqual(out[5].tolist(), [1.75, 1.75])



class TestCliWiring(unittest.TestCase):
    """The flag must survive argparse -> process_raw_data -> conversion.

    It did not, first time: --black-weight was defined and parsed and then
    never passed on, so the corpus came out byte-identical to the unweighted
    one. The unit tests above all passed, because they call the conversion
    function directly. Only an end-to-end run through the CLI catches this.
    """

    def test_flag_reaches_the_weights_on_disk(self):
        import json
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw" / "selfplay"
            raw.mkdir(parents=True)
            for game in range(6):
                recs = [rec(WHITE_FEN, "white"), rec(BLACK_FEN, "black")] * 3
                with open(raw / f"game_{game:04d}.jsonl", "w", encoding="utf-8") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")

            def run(out, *extra):
                cmd = [sys.executable, str(ROOT / "src" / "data_processor.py"),
                       "--raw-dir", str(Path(tmp) / "raw"), "--output-dir", str(out),
                       "--seed", "42", "--channels", "15", *extra]
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

            import numpy as np
            plain, weighted = Path(tmp) / "plain", Path(tmp) / "weighted"
            run(plain)
            run(weighted, "--black-weight", "1.75")
            a = np.load(plain / "policy_weights.npy")
            b = np.load(weighted / "policy_weights.npy")
            self.assertEqual(len(a), len(b))
            self.assertGreater(b.sum(), a.sum(),
                               "--black-weight did not reach the saved weights")
            self.assertAlmostEqual(float(b.max()), 1.75, places=5)

if __name__ == "__main__":
    unittest.main()
