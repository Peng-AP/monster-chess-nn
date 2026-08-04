"""Exploration must actually explore (DIRECTIVE E4).

The native RNG is constructed from the seed it is *given*. A caller passing a
constant therefore re-seeds an identical stream on every decision: temperature
sampling returns the same move every time, Dirichlet noise draws the same
vector every time, and the whole thing still looks random because different
positions still produce different moves.

Found 2026-08-04 when native self-play returned White 240/240 on a command
where the Python engine returned 14/10. Gates (a) and (b) both run at
temperature 0 with noise off, so neither could see it -- generation is the only
path that uses both, which is exactly the path whose output is training data.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "native"))

try:
    import monster_native as mn
except ImportError:
    mn = None

from evaluation import evaluate  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def skewed_tree():
    """A root with a descending visit distribution, so sampling has room."""
    game = MonsterChessGame(START_FEN)
    tree = mn.Tree(START_FEN)
    actions = [m.uci() for m in game.get_search_actions()]
    for i, action in enumerate(actions):
        idx = tree.add_child(0, action, 1.0 / len(actions))
        tree.set_stats(idx, max(1, 100 - i * 8), 0.0)
    return tree, actions


@unittest.skipIf(mn is None, "native crate not built")
class TestTheSeedMustAdvance(unittest.TestCase):
    def test_a_constant_seed_freezes_temperature_sampling(self):
        # Documents the trap rather than hiding it: this is the behaviour a
        # caller gets if it does not advance the seed.
        picks = {skewed_tree()[0].best_action(temperature=1.0, seed=7)[0]
                 for _ in range(10)}
        self.assertEqual(len(picks), 1)

    def test_advancing_the_seed_restores_sampling(self):
        picks = {skewed_tree()[0].best_action(temperature=1.0, seed=1000 + i)[0]
                 for i in range(20)}
        self.assertGreater(len(picks), 2, "temperature sampling is not exploring")

    def test_a_constant_seed_freezes_dirichlet_noise(self):
        draws = set()
        for _ in range(5):
            tree, _a = skewed_tree()
            tree.add_root_noise(seed=7)
            draws.add(tuple(round(p, 9) for _x, p in tree.root_priors()))
        self.assertEqual(len(draws), 1)

    def test_advancing_the_seed_restores_noise(self):
        draws = set()
        for i in range(5):
            tree, _a = skewed_tree()
            tree.add_root_noise(seed=2000 + i)
            draws.add(tuple(round(p, 9) for _x, p in tree.root_priors()))
        self.assertEqual(len(draws), 5, "Dirichlet noise is not varying")


@unittest.skipIf(mn is None, "native crate not built")
class TestAdapterAdvancesTheSeed(unittest.TestCase):
    def engine(self, **kw):
        from native_mcts import NativeMCTS
        return NativeMCTS(num_simulations=32, eval_fn=evaluate,
                          allow_early_stop=False, **kw)

    def test_decision_counter_advances(self):
        engine = self.engine()
        state = MonsterChessGame(START_FEN)
        self.assertEqual(engine._decisions, 0)
        engine.get_best_action(state, temperature=0.0)
        engine.get_best_action(state, temperature=0.0)
        self.assertEqual(engine._decisions, 2)

    def test_sampling_varies_across_decisions_from_one_position(self):
        # The generation case: same position, temperature on. A frozen stream
        # returns one move forever; this is what produced White 240/240.
        engine = self.engine()
        state = MonsterChessGame(START_FEN)
        picks = {engine.get_best_action(state, temperature=1.0)[0].uci()
                 for _ in range(25)}
        self.assertGreater(len(picks), 2,
                           "adapter is not advancing the seed per decision")

    def test_two_engines_with_the_same_seed_still_reproduce(self):
        # Advancing must not cost reproducibility: same seed, same sequence.
        a, b = self.engine(seed=99), self.engine(seed=99)
        sa = [a.get_best_action(MonsterChessGame(START_FEN), temperature=1.0)[0].uci()
              for _ in range(8)]
        sb = [b.get_best_action(MonsterChessGame(START_FEN), temperature=1.0)[0].uci()
              for _ in range(8)]
        self.assertEqual(sa, sb)



@unittest.skipIf(mn is None, "native crate not built")
class TestPerGameSeeding(unittest.TestCase):
    """The engine must inherit per-GAME seeding, not just per-decision.

    `mcts.MCTS` reads Python's global `random` module, and both
    `data_generation._worker` and `match._play` re-seed that module once per
    game. An engine with a constant default seed therefore replays the same
    noise and temperature draws in *every game*: 240 native self-play games
    that all ended White at fullmove 6-7, against python's 14/10 spread over
    fullmove 7-75. Advancing the seed per decision (above) does not fix it --
    each game restarts the same sequence.
    """

    def engine(self, **kw):
        from native_mcts import NativeMCTS
        return NativeMCTS(num_simulations=16, eval_fn=evaluate, **kw)

    def test_the_default_follows_the_global_rng_per_decision(self):
        # Harnesses re-seed once per GAME but build engines once per WORKER
        # (promotion_defense_probe._play_out, match._play). An engine that
        # captured a seed at construction ignores that re-seeding entirely --
        # which made the native PPC curve read 0.19 true captures at 200 sims
        # against python's 0.09 on the same deck.
        import random
        from monster_chess import MonsterChessGame
        engine = self.engine()
        self.assertIsNone(engine.seed, "default must follow the global RNG")

        def sequence():
            random.seed(31337)
            eng = self.engine()
            return [eng.get_best_action(MonsterChessGame(START_FEN),
                                        temperature=1.0)[0].uci()
                    for _ in range(6)]

        self.assertEqual(sequence(), sequence(),
                         "re-seeding the global RNG must reproduce the run")

    def test_reseeding_between_games_actually_changes_play(self):
        import random
        from monster_chess import MonsterChessGame
        random.seed(1)
        first = [self.engine().get_best_action(MonsterChessGame(START_FEN),
                                               temperature=1.0)[0].uci()
                 for _ in range(6)]
        random.seed(2)
        second = [self.engine().get_best_action(MonsterChessGame(START_FEN),
                                                temperature=1.0)[0].uci()
                  for _ in range(6)]
        self.assertNotEqual(first, second)

    def test_an_explicit_seed_still_pins_it(self):
        # Explicit seeding keeps the reproducible internal counter, so a run
        # can be pinned without touching global state.
        self.assertEqual(self.engine(seed=1234).seed, 1234)
        from monster_chess import MonsterChessGame
        a = self.engine(seed=7)
        b = self.engine(seed=7)
        sa = [a.get_best_action(MonsterChessGame(START_FEN), temperature=1.0)[0].uci()
              for _ in range(5)]
        sb = [b.get_best_action(MonsterChessGame(START_FEN), temperature=1.0)[0].uci()
              for _ in range(5)]
        self.assertEqual(sa, sb)

if __name__ == "__main__":
    unittest.main()
