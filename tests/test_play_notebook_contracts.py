import pytest
import json
import re
import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
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

    @pytest.mark.local_artifacts
    def test_setup_uses_refreshable_model_catalog(self):
        self.assertIn('importlib.reload(model_catalog)', self.setup)
        self.assertIn('model_catalog.discover_model_choices(PROJECT_ROOT, MODEL_DIR)', self.setup)
        self.assertIn('_refresh_button.on_click(_refresh_models)', self.setup)

    @pytest.mark.local_artifacts
    def test_refresh_preserves_loaded_selection_without_reloading_engine(self):
        import ast
        import ipywidgets as widgets
        callback = next(node for node in ast.parse(self.setup).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_refresh_models')
        dropdown = widgets.Dropdown(options=[('Heuristic', None), ('Old', '/old.pt')], value='/old.pt')
        loads = []
        loader = lambda change=None: loads.append(dropdown.value)
        dropdown.observe(loader, names='value')
        choices = [('Heuristic', None), ('New', '/new.pt'), ('Old', '/old.pt')]
        scope = {'_eval_dropdown': dropdown, '_load_evaluator': loader,
                 '_discover_model_choices': lambda: choices}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), '<refresh>', 'exec'), scope)
        scope['_refresh_models']()
        self.assertEqual(dropdown.value, '/old.pt')
        self.assertEqual(loads, [])
        self.assertIn('/new.pt', scope['_model_paths'])
        choices.pop()
        scope['_refresh_models']()
        self.assertIsNone(dropdown.value)
        self.assertEqual(loads, [None])

    @pytest.mark.local_artifacts
    def test_latest_candidate_discovers_new_generations_at_click_time(self):
        import ast
        import ipywidgets as widgets
        callback = next(node for node in ast.parse(self.setup).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_load_latest_candidate')
        old = '/models/candidates/bootstrap_main_gen_0047/arena_selected.pt'
        new = '/models/candidates/bootstrap_main_gen_0048/arena_selected.pt'
        training = '/models/candidates/bootstrap_main_gen_0049/best_value_net.pt'
        dropdown = widgets.Dropdown(options=[('Heuristic', None), ('old', old)], value=None)
        loads = []
        loader = lambda change=None: loads.append(dropdown.value)
        dropdown.observe(loader, names='value')
        scope = {'_eval_dropdown': dropdown, '_load_evaluator': loader,
                 '_status_label': widgets.HTML()}
        def refresh():
            dropdown.options = [('Heuristic', None), ('old', old), ('new', new), ('training', training)]
            scope['_model_paths'] = [old, new, training]
        scope['_refresh_models'] = refresh
        exec(compile(ast.Module(body=[callback], type_ignores=[]), '<latest>', 'exec'), scope)
        scope['_load_latest_candidate']()
        self.assertEqual(dropdown.value, new)
        self.assertEqual(loads, [new])
        scope['_load_latest_candidate']()
        self.assertEqual(loads, [new, new])
        self.assertIn('_latest_candidate_button.on_click(_load_latest_candidate)', self.setup)

    def test_candidate_picker_offers_the_gate_scored_checkpoint(self):
        from model_catalog import discover_model_choices
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / 'models/candidates/test_run'
            candidate.mkdir(parents=True)
            scored = candidate / 'selected_epoch_007.pt'
            scored.write_bytes(b'network')
            (candidate / 'best_value_net.pt').write_bytes(b'other')
            reports = root / 'benchmarks'
            reports.mkdir()
            (reports / 'gate_test.json').write_text(json.dumps({
                'model': str(scored), 'verdict': 'PASS', 'confirmed': True,
                'bar': 'vs_v24', 'legs': {'vs_v24': {'a_score': .6}}}))
            choices = discover_model_choices(root)
            self.assertEqual(choices[1][1], str(scored))
            self.assertIn('PASS+confirmed', choices[1][0])
            self.assertTrue(any('not necessarily gated' in label for label, _ in choices))

    def test_gated_picker_ranks_by_recency_not_by_score(self):
        from model_catalog import discover_model_choices
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / 'models/candidates/test_run'
            candidate.mkdir(parents=True)
            reports = root / 'benchmarks'
            reports.mkdir()
            for index, score in ((1, .9), (2, .6)):
                checkpoint = candidate / f'epoch_{index}.pt'
                checkpoint.write_bytes(b'network')
                report = reports / f'gate_{index}.json'
                report.write_text(json.dumps({
                    'model': str(checkpoint), 'verdict': 'PASS',
                    'bar': 'vs_v24', 'legs': {'vs_v24': {'a_score': score}}}))
                os.utime(report, (100 + index, 100 + index))
            self.assertTrue(discover_model_choices(root)[1][1].endswith('epoch_2.pt'))

    @pytest.mark.local_artifacts
    def test_color_selector_is_beside_model_and_drives_standard_game(self):
        self.assertIn('_color_dropdown = widgets.Dropdown(', self.setup)
        self.assertIn(
            'widgets.HBox([_eval_dropdown, _color_dropdown, _refresh_button])',
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
        """auto_engine stays heuristic on its own budget; SIMULATIONS is NN play.

        Checks the intent rather than a literal constructor line: the engine is
        now built through the shared factory (2026-08-16) so the notebook picks
        up native search, tree reuse, the finisher and CUDA graphs. Pinning the
        exact `MCTS(...)` text made this assert an implementation detail, and
        it would have blocked the notebook from ever moving to the engine the
        gates actually measure.
        """
        self.assertIn('AUTO_SIMULATIONS = 400', self.setup)
        self.assertIn('auto_engine', self.setup)
        self.assertIn('AUTO_SIMULATIONS', self.setup.split('auto_engine')[1][:200])
        # Heuristic: no model path is handed to the auto engine.
        self.assertIn('_build_engine(None, AUTO_SIMULATIONS', self.setup)

    def test_notebook_plays_the_engine_the_gates_measure(self):
        """The owner's sessions must not silently run a different search.

        Until 2026-08-16 the notebook built `MCTS` directly, so playtests ran
        the PYTHON engine with no tree reuse, no finisher and no repetition --
        while every measurement in the project described the native one.
        """
        self.assertIn('from benchmark import _build_engine', self.setup)
        self.assertIn('engine="native"', self.setup)
        self.assertIn('RepetitionTracker', self.setup)
        self.assertIn('repetition.record(', self.play)


if __name__ == '__main__':
    unittest.main()
