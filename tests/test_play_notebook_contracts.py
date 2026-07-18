import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'src' / 'play.ipynb'


class PlayNotebookContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with NOTEBOOK.open(encoding='utf-8') as f:
            notebook = json.load(f)
        cls.setup = ''.join(notebook['cells'][1]['source'])
        cls.play = ''.join(notebook['cells'][4]['source'])

    def test_setup_reloads_channel_sensitive_inference_modules(self):
        for alias in (
            '_config_mod',
            '_encoding_mod',
            '_data_processor_mod',
            '_train_mod',
            '_evaluation_mod',
            '_monster_chess_mod',
            '_mcts_mod',
        ):
            self.assertIn(f'importlib.reload({alias})', self.setup)

    def test_setup_discovers_candidate_and_rejected_models(self):
        self.assertIn(
            'os.path.join(MODEL_DIR, \"candidates\", \"*\", '
            '\"best_value_net.pt\")',
            self.setup,
        )
        self.assertIn(
            'os.path.join(MODEL_DIR, \"rejected\", \"*\", '
            '\"best_value_net.pt\")',
            self.setup,
        )
        self.assertIn('candidate/{run_name}', self.setup)
        self.assertIn('rejected/{run_name}', self.setup)

    def test_color_selector_is_beside_model_and_drives_standard_game(self):
        self.assertIn('_color_dropdown = widgets.Dropdown(', self.setup)
        self.assertIn(
            'widgets.HBox([_eval_dropdown, _color_dropdown])',
            self.setup,
        )
        self.assertIn('_selected_color = _color_dropdown.value', self.play)
        self.assertIn('play_loop(play_as=_selected_color', self.play)


if __name__ == '__main__':
    unittest.main()
