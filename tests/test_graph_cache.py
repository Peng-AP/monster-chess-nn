from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import native_mcts


class Model:
    training = False
    def parameters(self):
        return []
    def buffers(self):
        return []


def test_graph_cache_is_per_evaluator_and_signature(monkeypatch):
    monkeypatch.setenv('MONSTER_CUDA_GRAPH_CACHE', '1')
    monkeypatch.setattr(native_mcts, '_GraphedForward', lambda *a, **kw: object())
    ev = SimpleNamespace(model=Model(), torch=None, device='cuda')
    create = lambda e, n=16: native_mcts._graphed_forward_for(e, n, 15, True, False)
    first = create(ev)
    assert create(ev) is first
    assert create(ev, 8) is not first
    assert create(SimpleNamespace(model=ev.model, torch=None, device='cuda')) is not first
    ev.model = Model()
    assert create(ev) is not first


def test_disabled_or_training_graph_not_cached(monkeypatch):
    monkeypatch.setattr(native_mcts, '_GraphedForward', lambda *a, **kw: object())
    ev = SimpleNamespace(model=Model(), torch=None, device='cuda')
    create = lambda: native_mcts._graphed_forward_for(ev, 16, 15, True, False)
    monkeypatch.setenv('MONSTER_CUDA_GRAPH_CACHE', '0')
    assert create() is not create()
    monkeypatch.setenv('MONSTER_CUDA_GRAPH_CACHE', '1')
    ev.model.training = True
    assert create() is not create()
