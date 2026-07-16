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
    def test_hybrid_checkpoint_uses_progress_as_auxiliary_only_at_inference(self):
        model = train.build_model(
            input_channels=17,
            policy_head_channels=2,
            stem_channels=4,
            residual_block_channels=(4,),
            use_wdl_head=True,
            value_head_mode="hybrid",
            hybrid_progress_weight=0.25,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy_v18.pt"
            torch.save(model.state_dict(), checkpoint)
            loaded, _ = train.load_model_for_inference(
                str(checkpoint), torch.device("cpu"))

        self.assertEqual(loaded.value_head_mode, "hybrid")
        self.assertEqual(float(loaded._hybrid_progress_weight.item()), 0.0)

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
