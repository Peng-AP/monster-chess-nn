"""Contract tests for paired opening-book play.

The book's whole value is the *layout* -- each opening played twice with
colours reversed, so the position's bias and the 0.31 colour gap cancel inside
a pair. A layout bug would not crash; it would quietly produce unpaired games
and a standard error that claims a precision the sample does not have. These
tests pin the layout, the scoring rule it shares with the gate, and the
position round-trip a FEN cannot do alone.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import match  # noqa: E402
import make_book  # noqa: E402


def entry(i):
    return {"fen": f"fen-{i}", "half": bool(i % 2), "turn_count": i}


class TestBookDistribution(unittest.TestCase):
    @mock.patch("benchmark._build_engine")
    def test_action_temperature_does_not_also_sharpen_policy(self, build):
        build.return_value = (object(), "test")
        make_book._init_worker("model.pt", 700, "native")
        build.assert_called_once_with("model.pt", 700, None, engine="native")


class TestPairedLayout(unittest.TestCase):
    def test_each_opening_is_played_once_as_white_and_once_as_black(self):
        entries = [entry(i) for i in range(4)]
        tasks = match.build_tasks(8, 500, 16, entries)
        self.assertEqual(len(tasks), 8)
        for pair in range(4):
            first, second = tasks[2 * pair], tasks[2 * pair + 1]
            self.assertTrue(first[0])            # A is White
            self.assertFalse(second[0])          # A is Black
            self.assertIs(first[3], entries[pair])
            self.assertIs(second[3], entries[pair])
            self.assertEqual(first[4], pair)
            self.assertEqual(second[4], pair)

    def test_both_halves_of_a_pair_share_one_seed(self):
        entries = [entry(i) for i in range(3)]
        tasks = match.build_tasks(6, 500, 0, entries)
        for pair in range(3):
            self.assertEqual(tasks[2 * pair][1], tasks[2 * pair + 1][1])
        # ...and distinct pairs do not share one.
        seeds = {t[1] for t in tasks}
        self.assertEqual(len(seeds), 3)

    def test_book_play_never_samples_the_opening(self):
        # Sampling on top of a book would redraw the opening from the
        # candidate's own policy and undo the whole point.
        tasks = match.build_tasks(4, 500, 16, [entry(0), entry(1)])
        self.assertEqual({t[2] for t in tasks}, {0})

    def test_colours_are_exactly_balanced(self):
        tasks = match.build_tasks(20, 1, 0, [entry(i) for i in range(10)])
        self.assertEqual(sum(1 for t in tasks if t[0]), 10)
        self.assertEqual(sum(1 for t in tasks if not t[0]), 10)

    def test_odd_game_counts_are_refused(self):
        with self.assertRaises(SystemExit):
            match.build_tasks(7, 1, 0, [entry(i) for i in range(10)])

    def test_a_book_too_small_to_pair_is_refused_not_recycled(self):
        # Book play is deterministic, so a reused entry replays an identical
        # game instead of adding a sample. Silently recycling would inflate
        # the apparent sample size.
        with self.assertRaises(SystemExit):
            match.build_tasks(10, 1, 0, [entry(i) for i in range(4)])

    def test_bookless_layout_is_unchanged(self):
        # The pre-book seed layout is load-bearing for every historical score.
        tasks = match.build_tasks(20, 777, 16)
        seeds = [t[1] for t in tasks]
        self.assertEqual(seeds, match.match_game_seeds(20, 777))
        self.assertEqual([t[0] for t in tasks], [True] * 10 + [False] * 10)
        self.assertEqual({t[2] for t in tasks}, {16})
        self.assertEqual({t[3] for t in tasks}, {None})
        self.assertEqual({t[4] for t in tasks}, {None})


class TestDisjointBlocks(unittest.TestCase):
    def test_offset_selects_a_later_block(self):
        entries = [entry(i) for i in range(10)]
        tasks = match.build_tasks(4, 1, 0, entries, offset=6)
        self.assertEqual([t[3]["turn_count"] for t in tasks], [6, 6, 7, 7])

    def test_running_past_the_end_is_refused(self):
        with self.assertRaises(SystemExit):
            match.build_tasks(6, 1, 0, [entry(i) for i in range(4)], offset=2)

    def test_gate_legs_and_the_confirmation_never_share_entries(self):
        import gate
        spec = [(name, opp or "m.pt" if name != "anchor" else None, games)
                for name, opp, games in gate.FULL_LEGS]
        offsets, needed = gate.book_leg_offsets(spec)

        # The anchor keeps sampled openings, so it draws no entries.
        self.assertNotIn("anchor", offsets)
        self.assertIn(gate.BAR, offsets)
        self.assertIn(gate.CONFIRM_LEG, offsets)

        games_by_leg = {name: games for name, _o, games in spec}
        games_by_leg[gate.CONFIRM_LEG] = games_by_leg[gate.BAR]
        used = []
        for name, first in offsets.items():
            used.extend(range(first, first + games_by_leg[name] // 2))
        self.assertEqual(len(used), len(set(used)),
                         "two gate legs would replay the same openings and "
                         "agree by construction")
        self.assertEqual(needed, len(used))

    def test_confirmation_is_allocated_after_every_scheduled_leg(self):
        import gate
        spec = [(gate.BAR, "m.pt", 200), ("vs_ramp", "s.pt", 40),
                ("anchor", None, 20)]
        offsets, needed = gate.book_leg_offsets(spec)
        self.assertEqual(offsets[gate.BAR], 0)
        self.assertEqual(offsets["vs_ramp"], 100)
        self.assertEqual(offsets[gate.CONFIRM_LEG], 120)
        self.assertEqual(needed, 220)

    def test_gate_can_follow_a_disjoint_screen_block(self):
        import gate
        spec = [(gate.BAR, "m.pt", 200), ("vs_ramp", "s.pt", 40),
                ("anchor", None, 20)]
        offsets, needed = gate.book_leg_offsets(spec, base_offset=120)
        self.assertEqual(offsets[gate.BAR], 120)
        self.assertEqual(offsets["vs_ramp"], 220)
        self.assertEqual(offsets[gate.CONFIRM_LEG], 240)
        self.assertEqual(needed, 340)


class TestPairedStatistics(unittest.TestCase):
    def test_the_colour_gap_cancels_inside_a_pair(self):
        # A wins every game as White and loses every game as Black: the true
        # difference is nil, and pairing should see that exactly while the
        # per-game sample looks maximally noisy.
        results = []
        for pair in range(10):
            results.append((1, 40, True, pair))
            results.append((-1, 40, False, pair))
        stats = match.paired_stats(results)
        self.assertEqual(stats["pairs"], 10)
        self.assertEqual(stats["pair_mean"], 0.5)
        self.assertEqual(stats["se_paired"], 0.0)
        naive = (0.25 / len(results)) ** 0.5      # per-game binomial SE
        self.assertLess(stats["se_paired"], naive)

    def test_a_real_edge_still_shows_up(self):
        # A sweeps both colours in 6 pairs and splits the other 4:
        # (6 * 1.0 + 4 * 0.5) / 10 = 0.8.
        results = []
        for pair in range(10):
            swept = pair < 6
            results.append((1, 40, True, pair))
            results.append((1 if swept else -1, 40, False, pair))
        stats = match.paired_stats(results)
        self.assertEqual(stats["pair_mean"], 0.8)
        self.assertGreater(stats["se_paired"], 0.0)

    def test_unpaired_results_get_no_paired_error(self):
        self.assertIsNone(match.paired_stats(
            [(1, 40, True, None), (-1, 40, False, None)]))

    def test_incomplete_pairs_are_excluded_not_counted_as_halves(self):
        # A half-finished pair still carries the colour term it was meant to
        # cancel, so it must not enter the variance.
        results = [(1, 40, True, 0), (-1, 40, False, 0),
                   (1, 40, True, 1), (-1, 40, False, 1),
                   (1, 40, True, 2)]
        stats = match.paired_stats(results)
        self.assertEqual(stats["pairs"], 2)
        self.assertEqual(stats["pairs_incomplete"], 1)

    def test_game_score_matches_the_gate_win_rule(self):
        from benchmark import summarize_side
        for result in (1, -1, 0, 0.5, -0.5):
            side = summarize_side([(result, 40)])
            self.assertEqual(match.game_score(result), side["score"],
                             f"scoring diverged from summarize_side at "
                             f"result={result}")

    def test_move_limit_relabels_score_as_draws_in_pairs(self):
        # +-0.5 is the move-limit relabel. Owner rule 2026-08-03: a win
        # requires capturing the king, so these are draws here too.
        self.assertEqual(match.game_score(0.5), 0.5)
        self.assertEqual(match.game_score(-0.5), 0.5)


class TestOpeningDiversity(unittest.TestCase):
    @staticmethod
    def _row(a_is_white, fen, half=False, turn=8, complete=True):
        return (0, 30, a_is_white, None,
                {"fen": fen, "half": half, "turn_count": turn,
                 "complete": complete})

    def test_duplicate_states_are_counted_by_model_colour(self):
        rows = [self._row(True, "same"), self._row(True, "same"),
                self._row(False, "same"), self._row(False, "other")]
        out = match.opening_stats(rows)
        self.assertEqual(out["games_observed"], 4)
        self.assertEqual(out["unique_states"], 2)
        self.assertEqual(out["unique_as_white"], 1)
        self.assertEqual(out["unique_as_black"], 2)
        self.assertEqual(out["effective_unique_games"], 3)
        self.assertEqual(out["max_multiplicity"], 3)
        self.assertEqual(out["multiplicity_histogram"], {"3": 1, "1": 1})

    def test_full_monster_state_is_part_of_the_identity(self):
        rows = [self._row(True, "fen", half=False, turn=8),
                self._row(True, "fen", half=True, turn=8),
                self._row(True, "fen", half=False, turn=9)]
        self.assertEqual(match.opening_stats(rows)["unique_states"], 3)

    def test_legacy_rows_have_no_claimed_diversity(self):
        self.assertIsNone(match.opening_stats([(1, 20, True, None)]))

    def test_incomplete_sampled_prefixes_stay_visible(self):
        out = match.opening_stats([
            self._row(True, "terminal", complete=False)])
        self.assertEqual(out["incomplete_openings"], 1)


class TestBookLoading(unittest.TestCase):
    def _write(self, doc):
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8")
        json.dump(doc, fh)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_metadata_rides_into_the_match_artifact(self):
        path = self._write({"schema_version": 1, "model": "m/best.pt",
                            "model_sha256": "abc", "plies": 16, "sims": 700,
                            "temperature": 0.5, "seed": 9,
                            "entries": [entry(0), entry(1)]})
        entries, meta = match.load_book(path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(meta["entries"], 2)
        self.assertEqual(meta["model_sha256"], "abc")
        self.assertEqual(meta["plies"], 16)

    def test_an_empty_book_is_refused(self):
        with self.assertRaises(SystemExit):
            match.load_book(self._write({"schema_version": 1, "entries": []}))


class TestPositionRoundTrip(unittest.TestCase):
    def test_a_pending_white_half_survives_fen_plus_flag(self):
        from monster_chess import MonsterChessGame
        game = MonsterChessGame()
        game.apply_search_action(game.get_search_actions()[0])
        self.assertTrue(game.white_half_pending)

        restored = MonsterChessGame(game.fen())
        restored.white_half_pending = True
        restored.turn_count = game.turn_count
        self.assertEqual(restored.is_white_turn, game.is_white_turn)
        self.assertEqual([str(a) for a in restored.get_search_actions()],
                         [str(a) for a in game.get_search_actions()])

    def test_turn_count_is_carried_so_the_cap_is_not_extended(self):
        # MonsterChessGame(fen) restarts turn_count at 0. A book position eight
        # turns deep would then get the full 150 turns again, lengthening its
        # games and lowering the draw rate against non-book play.
        from monster_chess import MonsterChessGame
        game = MonsterChessGame()
        for _ in range(6):
            game.apply_search_action(game.get_search_actions()[0])
        self.assertGreater(game.turn_count, 0)
        self.assertEqual(MonsterChessGame(game.fen()).turn_count, 0)

    def test_play_one_restores_both_pieces_of_state(self):
        import inspect
        from benchmark import play_one
        params = inspect.signature(play_one).parameters
        self.assertIn("start_half", params)
        self.assertIn("start_turn_count", params)
        source = inspect.getsource(play_one)
        self.assertIn("game.white_half_pending = bool(start_half)", source)
        self.assertIn("game.turn_count = int(start_turn_count)", source)


if __name__ == "__main__":
    unittest.main()
