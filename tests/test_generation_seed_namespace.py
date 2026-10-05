import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import iterate


def option(command, name):
    return command[command.index(name) + 1]


@pytest.mark.local_artifacts
def test_large_data_namespaces_are_disjoint_and_preserve_training_seed(tmp_path):
    args = iterate.build_parser().parse_args([
        "--seed", "3173", "--data-seed-base", "1000000000",
        "--games", "4000", "--book-seed-games", "1600"])
    iterate._validate_args(args)
    arch = iterate._checkpoint_spec(iterate.DEFAULT_CHAMPION)
    used = set(range(48578, 50578))  # gen45 free generation
    for generation in (46, 47):
        plan = iterate._command_plan(args, generation, iterate.DEFAULT_CHAMPION, arch,
                                     iterate._paths_for_generation(tmp_path, generation), [])
        for command in plan['generate']['commands']:
            seed, count = int(option(command, '--seed')), int(option(command, '--num-games'))
            block = set(range(seed, seed + count))
            assert not block & used
            used |= block
        assert option(plan['train']['commands'][0], '--seed') == '3173'
        assert int(option(plan['binding_gate']['commands'][0], '--seed')) < 1000000000


@pytest.mark.local_artifacts
def test_historical_data_seed_remains_unchanged(tmp_path):
    args = iterate.build_parser().parse_args(['--seed', '3173'])
    arch = iterate._checkpoint_spec(iterate.DEFAULT_CHAMPION)
    plan = iterate._command_plan(args, 45, iterate.DEFAULT_CHAMPION, arch,
                                 iterate._paths_for_generation(tmp_path, 45), [])
    assert option(plan['generate']['commands'][0], '--seed') == '48578'


@pytest.mark.parametrize('base', ['-1', str(2**32)])
def test_invalid_data_namespace_is_rejected(base):
    args = iterate.build_parser().parse_args(['--data-seed-base', base])
    with pytest.raises(ValueError):
        iterate._validate_args(args)
