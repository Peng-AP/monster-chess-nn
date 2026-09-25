"""Ownership-transfer reroot must preserve exact incumbent tree semantics."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native'))
import monster_native as native

START='rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1'


class FastRerootTests(unittest.TestCase):
    @staticmethod
    def bridge(buf,n,channels):
        return np.full(n,.25,np.float32).tobytes(),np.zeros(n*4096,np.float32).tobytes()

    def test_all_node_state_and_statistics_match_through_side_changes(self):
        old,new=native.Tree(START),native.Tree(START)
        for turn in range(6):
            for tree in (old,new):
                tree.run_batched_puct(160,self.bridge,seed=93+turn,allow_early_stop=False)
            a=old.best_action(temperature=0,seed=93)
            self.assertEqual(a,new.best_action(temperature=0,seed=93))
            if a[0] is None:break
            self.assertTrue(old.reroot(a[0]))
            self.assertTrue(new.reroot(a[0],fast=True))
            self.assertEqual(old.node_count(),new.node_count())
            self.assertEqual(old.root_visits(),new.root_visits())
            for i in range(old.node_count()):
                for name in ('fen','is_white_turn','white_half_pending','visit_count',
                             'total_value','q_value','moves_left'):
                    self.assertEqual(getattr(old,name)(i),getattr(new,name)(i),(turn,i,name))
