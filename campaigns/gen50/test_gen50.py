import run_gen50 as c


def test_recipe_and_command_agree():
    for smoke in (True, False):
        command = c.iteration_command(smoke)
        args = c.prior.iterate.build_parser().parse_args(command[command.index('--')+1:])
        recipe = c.read(c.ROOT/command[command.index('--recipe')+1])
        c.prior.validate_recipe(recipe)
        assert recipe['model'] == args.incumbent == c.TEACHER
        assert recipe['sims'] == args.sims
        assert recipe['free_games']+recipe['fork_games'] == args.games
        assert not recipe['prefix_models'] and not recipe['opponents']
        assert not args.promote_on_pass and not args.reject_on_offline_regression
        assert args.through_phase == 'checkpoint_screen'
        assert args.reanalysis_sims == (8 if smoke else 12800)
        assert args.seed == (6173 if smoke else 3173)


def test_evaluation_counts_and_disjoint_seeds():
    for smoke in (True, False):
        schedule = c.evaluation_schedule(c.TEACHER, smoke)
        spec = c.prior.layout(smoke)
        assert sum(r['games'] for r in schedule)+spec['final_par']+4*spec['final_per_side'] == (36 if smoke else 2280)
        seeds = []
        from match import build_tasks
        for r in schedule:
            seeds.extend(t[1] for t in build_tasks(r['games'],r['seed'],16))
        gate_base = (2340000000 if smoke else 2330000000)+4000000
        for offset,n in [(0,2*spec['final_per_side']), (100000,spec['final_par']),
                         (200000,2*spec['final_per_side'])]:
            seeds.extend(t[1] for t in build_tasks(n,gate_base+offset,16))
        assert len(seeds) == len(set(seeds))
        assert max(seeds)<2**32


def test_no_generation_precheck_or_per_epoch_game_gates():
    command = c.iteration_command(False)
    assert command[command.index('--expected-generation')+1] == '50'
    assert command[command.index('--epochs')+1] == '30'
    assert command[command.index('--patience')+1] == '10'
    assert command[command.index('--checkpoint-screen-finalists')+1] == '2'


def test_gate_failure_still_runs_every_diagnostic(tmp_path, monkeypatch):
    from types import SimpleNamespace
    candidate = tmp_path/'candidate.pt'
    candidate.write_bytes(b'fixture')
    (tmp_path/'manifest.json').write_text('{}')
    job = SimpleNamespace(root=tmp_path, provenance={})
    monkeypatch.setattr(c, 'train', lambda *a: candidate)
    monkeypatch.setattr(c, 'identity', lambda: {})
    monkeypatch.setattr(c.prior, 'free_gate', lambda *a: {'verdict':'FAIL'})
    calls = []
    monkeypatch.setattr(c.prior, 'free_match', lambda job, **item: calls.append(item['name']) or {})
    c.work(job, False)
    assert calls == ['vs_b2','vs_v27','self','deep_vs_gen49','deep_vs_b2','deep_self']
    assert c.read(tmp_path/'summary.json')['complete']
