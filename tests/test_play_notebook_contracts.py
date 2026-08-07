import json
import re
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
        cls.deck = ''.join(notebook['cells'][6]['source'])
        cls.next_position = ''.join(notebook['cells'][7]['source'])

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
        self.assertIn('os.path.join(MODEL_DIR, \"candidates\", \"*\")',
                      self.setup)
        self.assertIn(
            'os.path.join(MODEL_DIR, \"rejected\", \"*\", '
            '\"best_value_net.pt\")',
            self.setup,
        )
        self.assertIn('rejected/{run_name}', self.setup)

    def test_candidate_picker_offers_the_gate_scored_checkpoint(self):
        """A candidate directory holds two different models.

        `arena_selected.pt` is the epoch the arena chose and what the gates
        scored; `best_value_net.pt` is the lowest-training-loss net and was
        never gated. The picker globbed only the latter until 2026-08-06, so
        selecting a candidate silently loaded the ungated model -- the owner
        would have playtested something no gate ever measured.
        """
        self.assertIn('\"arena_selected.pt\"', self.setup)
        # Both must be reachable, and the labels must say which is which.
        self.assertIn('candidate [GATED] {run_name}', self.setup)
        self.assertIn('candidate [{tag}] {run_name}', self.setup)
        # The marker leads, because a dropdown truncates the tail.
        self.assertLess(self.setup.index('candidate [GATED]'),
                        self.setup.index('{run_name}\", arena'))

    def test_color_selector_is_beside_model_and_drives_standard_game(self):
        self.assertIn('_color_dropdown = widgets.Dropdown(', self.setup)
        self.assertIn(
            'widgets.HBox([_eval_dropdown, _color_dropdown])',
            self.setup,
        )
        self.assertIn('_selected_color = _color_dropdown.value', self.play)
        self.assertIn('play_loop(play_as=_selected_color', self.play)

    def test_standard_game_saves_under_the_selected_color(self):
        """White games must land in white_*, not the cell-1 SAVE_DIR default.

        SAVE_DIR is built from the PLAY_AS constant ("black"), so a play_loop
        call that omitted save_dir would file every White game the owner plays
        into black_2026_07/ — silently misdirecting the scarcest data in the
        project (9 White games vs 23 Black).
        """
        self.assertIn(
            'play_loop(play_as=_selected_color, '
            'save_dir=f"data/raw/human_games/{_selected_color}_2026_07")',
            self.play,
        )

    # --- curriculum deck cells (6-7) ---------------------------------
    # These bypassed a week of work unnoticed because nothing inspected them:
    # a behind-the-cliff deck and a hardcoded v16 opponent that overwrote the
    # dropdown cost the owner an entire session (2026-07-20, game_00048).

    def test_deck_cell_uses_the_pawn_phase_deck(self):
        """Starts must be in front of the cliff, where Black's gap lives."""
        self.assertIn(
            'DECK_FILE = "data/start_fens/cliff_deck_v1.jsonl"', self.deck)

    def test_deck_cells_pin_no_stale_opponent(self):
        for cell in (self.deck, self.next_position):
            self.assertNotIn('fresh_start_v16', cell)

    def test_next_position_never_overrides_a_manual_evaluator(self):
        """The dropdown may only be filled in when it is still unset."""
        assignments = re.findall(
            r'^\s*_eval_dropdown\.value\s*=(?!=)', self.next_position,
            flags=re.MULTILINE)
        self.assertEqual(
            len(assignments), 1,
            'exactly one _eval_dropdown.value assignment expected')
        self.assertRegex(
            self.next_position,
            r'if\s+_eval_dropdown\.value\s+is\s+None:\s*\n\s*'
            r'_eval_dropdown\.value\s*=',
            'the assignment must be guarded by an "is None" check',
        )

    def test_next_position_opponent_is_resolved_not_hardcoded(self):
        """Printed opponent reflects the live selection, not a literal."""
        self.assertIn('_OPP_CHAIN', self.next_position)
        self.assertIn('os.path.exists(p)', self.next_position)
        # every fallback target must also be a dropdown option, or assigning
        # it raises TraitError instead of switching the engine
        self.assertIn('_OPTIONS', self.next_position)
        self.assertIn('p in _OPTIONS', self.next_position)

    # --- no duplicated logic between the notebook and play.py -------

    def test_parse_move_is_imported_not_redefined(self):
        """One definition, one behaviour.

        The notebook used to carry a ~60-line copy that had drifted: with
        legal_set=None it fell through to board.pseudo_legal_moves where
        play.py returns None. Every notebook call passes an explicit
        legal_set (asserted below), so importing is behaviour-preserving.
        """
        for cell in (self.setup, self.play, self.deck, self.next_position):
            self.assertNotIn('def parse_move', cell)
        self.assertIn('parse_move', self.play)
        self.assertRegex(
            self.play, r'from play import \([^)]*parse_move',
            'parse_move must come from play.py',
        )

    def test_every_parse_move_call_supplies_a_legal_set(self):
        calls = [line for line in self.play.split('\n') if 'parse_move(' in line
                 and 'import' not in line]
        self.assertTrue(calls)
        for call in calls:
            self.assertIn('legal_set=', call)

    def test_no_swindle_argument_survives(self):
        """The parameter was accepted and ignored; the king-safety override
        lives in MCTS.get_best_action and applies engine-wide."""
        for cell in (self.setup, self.play):
            self.assertNotIn('swindle', cell)

    def test_auto_finishes_use_their_own_simulation_budget(self):
        """auto_engine is heuristic (sequential UCB1); SIMULATIONS is for NN play."""
        self.assertIn('AUTO_SIMULATIONS = 400', self.setup)
        self.assertIn(
            'auto_engine = MCTS(num_simulations=AUTO_SIMULATIONS, eval_fn=None,',
            self.setup,
        )


if __name__ == '__main__':
    unittest.main()
