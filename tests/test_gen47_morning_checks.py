import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from gen47_morning_checks import round_command, OPPONENTS
from generation_followup import parser, plan


def test_morning_rounds_are_separate_depth_matched_and_not_training():
    seeds = set()
    for i in range(24):
        command = round_command(i)
        args = parser().parse_args(command[1:])
        assert args.opponent == [OPPONENTS[i % 3]]
        assert args.sims == (3200 if i % 2 == 0 else 6400)
        assert args.seed not in seeds
        seeds.add(args.seed)
        stages = plan(args)
        assert len(stages) == 4
        baseline, candidate = stages[1][1], stages[2][1]
        for flag in ('--seed', '--book', '--sims', '--games'):
            assert baseline[baseline.index(flag)+1] == candidate[candidate.index(flag)+1]
        assert args.workers == 8
        assert not args.raw_generation_dir and not args.replay_data_dir
