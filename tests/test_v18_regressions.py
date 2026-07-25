import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train


def _load_match_tool():
    spec = importlib.util.spec_from_file_location(
        "monster_chess_match_tool", ROOT / "tools" / "match.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V18RegressionTests(unittest.TestCase):
    def test_hybrid_value_head_mode_is_gone(self):
        """Concluded dead end (2026-07-18): near-mate labels poison Black
        whenever combined with the ramp, at any dose and in any order. The
        mode was removed rather than left as a loaded gun; this pins it."""
        with self.assertRaises(ValueError):
            train.build_model(
                input_channels=17,
                policy_head_channels=2,
                stem_channels=4,
                residual_block_channels=(4,),
                use_wdl_head=True,
                value_head_mode="hybrid",
            )

    def test_anchor_match_defaults_to_no_opening_sampling(self):
        match_tool = _load_match_tool()
        self.assertEqual(
            match_tool.resolve_opening_temp_plies(
                model_b=None, requested=None),
            0,
        )
        self.assertEqual(
            match_tool.resolve_opening_temp_plies(
                model_b="incumbent.pt", requested=None),
            16,
        )
        self.assertEqual(
            match_tool.resolve_opening_temp_plies(
                model_b=None, requested=7),
            7,
        )


if __name__ == "__main__":
    unittest.main()
