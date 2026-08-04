"""Dirichlet root noise and tree reuse (DIRECTIVE E3).

Both are behaviours where a plausible-looking port is silently wrong:

* **Noise** must be a real Dirichlet sample (alpha 0.3 concentrates mass on a
  few children — uniform noise would not explore the same way), applied at the
  root only and in self-play only. D4 puts RNG under statistical parity, so the
  test checks the *distribution*, not a reproduction of numpy's stream.

* **Reuse** is allowed only across White's first -> second half-move. Both
  nodes are White-to-move so accumulated Q stays valid; reusing across a side
  change would need every stored value rebased, which is why the Python engine
  refuses it.
"""
import statistics
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

import config  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1"


def expanded_tree(seed_actions=None):
    tree = mn.Tree(START_FEN)
    game = MonsterChessGame(START_FEN)
    actions = seed_actions or [m.uci() for m in game.get_search_actions()]
    for action in actions:
        tree.add_child(0, action, 1.0 / len(actions))
    return tree, actions


@unittest.skipIf(mn is None, "native crate not built")
class TestDirichletNoise(unittest.TestCase):
    def test_constants_match_config(self):
        self.assertEqual(mn.DIRICHLET_ALPHA, config.DIRICHLET_ALPHA)
        self.assertEqual(mn.DIRICHLET_EPSILON, config.DIRICHLET_EPSILON)

    def test_priors_still_sum_to_one(self):
        tree, actions = expanded_tree()
        tree.add_root_noise(seed=1)
        total = sum(p for _a, p in tree.root_priors())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_noise_actually_moves_the_priors(self):
        tree, actions = expanded_tree()
        before = dict(tree.root_priors())
        tree.add_root_noise(seed=1)
        after = dict(tree.root_priors())
        self.assertNotEqual(before, after)
        # epsilon = 0.25, so no prior can move by more than epsilon.
        for action in before:
            self.assertLessEqual(abs(after[action] - before[action]),
                                 config.DIRICHLET_EPSILON + 1e-9)

    def test_it_is_concentrated_not_uniform(self):
        # alpha = 0.3 < 1 puts most mass on a few children. A uniform sampler
        # would pass "sums to 1" and "priors moved" but not this.
        maxima = []
        for seed in range(40):
            tree, actions = expanded_tree()
            tree.add_root_noise(seed=seed + 1)
            priors = [p for _a, p in tree.root_priors()]
            maxima.append(max(priors))
        uniform_prior = 1.0 / len(actions)
        self.assertGreater(statistics.mean(maxima), uniform_prior * 1.3)

    def test_different_seeds_give_different_noise(self):
        tree_a, _ = expanded_tree()
        tree_b, _ = expanded_tree()
        tree_a.add_root_noise(seed=1)
        tree_b.add_root_noise(seed=2)
        self.assertNotEqual(dict(tree_a.root_priors()), dict(tree_b.root_priors()))

    def test_batched_search_leaves_noise_off_by_default(self):
        # Evaluation and arena games must never get noise, so the default has
        # to be off and it has to be *observably* off. A stub evaluator keeps
        # this a search test rather than a model test.
        def stub(buf, n, channels):
            import numpy as np
            return (np.zeros(n, dtype=np.float32).tobytes(),
                    np.zeros(n * 4096, dtype=np.float32).tobytes())

        default_a = mn.Tree(START_FEN)
        default_b = mn.Tree(START_FEN)
        default_a.run_batched_puct(32, stub, batch_size=8, channels=17, seed=1)
        default_b.run_batched_puct(32, stub, batch_size=8, channels=17, seed=2)
        self.assertEqual(dict(default_a.root_priors()), dict(default_b.root_priors()),
                         "priors differ by seed with noise off")

        noisy_a = mn.Tree(START_FEN)
        noisy_b = mn.Tree(START_FEN)
        noisy_a.run_batched_puct(32, stub, batch_size=8, channels=17,
                                 root_noise=True, seed=1)
        noisy_b.run_batched_puct(32, stub, batch_size=8, channels=17,
                                 root_noise=True, seed=2)
        self.assertNotEqual(dict(noisy_a.root_priors()), dict(noisy_b.root_priors()),
                            "noise requested but priors identical across seeds")


@unittest.skipIf(mn is None, "native crate not built")
class TestTreeReuse(unittest.TestCase):
    def test_reroot_keeps_the_subtree_and_its_statistics(self):
        tree, actions = expanded_tree()
        keep = actions[0]
        idx = 1  # first child added
        grandchild = tree.add_child(idx, "a7a5", 0.5)
        tree.set_stats(idx, 17, 8.5)
        tree.set_stats(grandchild, 5, 2.5)
        before_nodes = tree.node_count()

        self.assertTrue(tree.reroot(keep))
        self.assertLess(tree.node_count(), before_nodes)
        # The kept child is now the root, with its statistics intact.
        self.assertEqual(tree.visit_count(0), 17)
        self.assertAlmostEqual(tree.total_value(0), 8.5)
        self.assertAlmostEqual(tree.q_value(0), 0.5)
        # and its own child survived the re-index
        self.assertEqual(tree.visit_count(1), 5)

    def test_reroot_reports_failure_for_an_unknown_action(self):
        tree, _actions = expanded_tree()
        self.assertFalse(tree.reroot("h1h8"))

    def test_the_new_root_is_the_state_after_that_half_move(self):
        game = MonsterChessGame(START_FEN)
        first = game.get_search_actions()[0].uci()
        tree, _ = expanded_tree()
        tree.reroot(first)
        game.apply_search_action(
            next(m for m in game.get_search_actions() if m.uci() == first))
        self.assertEqual(tree.fen(0), game.fen())
        self.assertTrue(tree.white_half_pending(0))


if __name__ == "__main__":
    unittest.main()
