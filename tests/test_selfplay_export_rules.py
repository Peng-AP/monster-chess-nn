"""Showcase games must obey the same rules as every measurement.

`tools/export_selfplay_replays.py` had no repetition tracking, so the games it
exported ran under pre-2026-08-16 rules. The symptom was visible and was nearly
missed: every drawn game measured exactly 225 plies -- the 150-turn cap, to the
ply -- because shuffling positions ground to the clock instead of being drawn
where the threefold rule ends them.

The real hazard is not long games. A position that repeats at ply 60 can
continue under the old rules and resolve DECISIVELY, which shifts the colour
tally rather than only the lengths, and those tallies were being compared to
gate numbers measured under the rule.
"""
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "export_selfplay_replays.py").read_text(
    encoding="utf-8")


class ExporterAppliesTheRepetitionRule(unittest.TestCase):
    def test_the_recorded_loop_tracks_repetition(self):
        self.assertIn("from repetition import RepetitionTracker", SOURCE)
        self.assertIn("repetition.record(game, 0)", SOURCE)
        self.assertIn("if repetition.record(game, plies):", SOURCE)

    def test_a_repetition_overrides_the_caps_lean(self):
        """`get_result()` leans +-0.5 on a capped game; a repetition is 0.0.

        Taking the cap's lean on a game that ended by rule would mislabel a
        drawn game as Black-dominant, which is exactly the training label the
        2026-08-16 change was about.
        """
        self.assertIn("float(repetition.draw_result) if repeated", SOURCE)

    def test_the_break_happens_after_the_frame_is_recorded(self):
        """The repeating position is the last thing the reader should see.

        Breaking first would drop the move that caused the draw from the
        replay, leaving a game that appears to stop for no reason.
        """
        frame_append = SOURCE.index('"label": f"{plies}. {actor}{detail}')
        rep_break = SOURCE.index("if repetition.record(game, plies):")
        self.assertLess(frame_append, rep_break)

    def test_it_matches_how_benchmark_resolves_the_same_situation(self):
        """One rule, one implementation shape, so they cannot drift apart."""
        benchmark = (ROOT / "src" / "benchmark.py").read_text(encoding="utf-8")
        for fragment in ("repetition.record(game, 0)",
                         "repetition.draw_result"):
            self.assertIn(fragment, benchmark)
            self.assertIn(fragment, SOURCE)


if __name__ == "__main__":
    unittest.main()
