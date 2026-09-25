"""Diagnostic-only policy/value buffer crossover; no model mutation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import mainline_study as study
from native_mcts import NativeMCTS, make_bridge
from evaluation import NNEvaluator
from encoding import move_to_index
from match_evidence import atomic_json, digest, file_hash, runtime_identity


def mix_bridges(policy_bridge, value_bridge):
    if policy_bridge is value_bridge:
        return policy_bridge
    def mixed(buf, n, chans):
        p = policy_bridge(buf, n, chans)
        v = value_bridge(buf, n, chans)
        if len(p) != 2 or len(v) != 2:
            raise ValueError('Only scalar-value / ordinary-policy bridges supported')
        return v[0], p[1]
    return mixed


def make_engine(policy, value, sims):
    pe = NNEvaluator(policy)
    ve = pe if policy == value else NNEvaluator(value)
    pb, pc = make_bridge(pe, graph_width=16)
    vb, vc = (pb, pc) if pe is ve else make_bridge(ve, graph_width=16)
    if pc != vc or pc != 15:
        raise ValueError('Diagnostic requires matching 15-plane inputs')
    engine = NativeMCTS(num_simulations=sims, eval_fn=pe, root_noise=False,
                        allow_early_stop=False, reuse_across_moves=True)
    original = engine._bridge
    engine._bridge = mix_bridges(pb, vb)
    # Keep both evaluators and graph buffers alive for the whole experiment.
    engine._diagnostic_owners = (pe, ve)
    return engine, original


def raw_network(path, cases):
    import numpy as np
    evaluator = NNEvaluator(path)
    bridge, channels = make_bridge(evaluator, graph_width=16)
    rows = []
    for case in cases:
        game, _ = study.checked_restore(case['state'])
        if not game.is_white_turn:
            raise ValueError('Raw diagnostic currently declares White-only roots')
        arr = evaluator.fen_to_tensor(game.fen(), is_white_turn=True,
                    half_pending=game.white_half_pending, input_channels=channels)
        arr = np.asarray(arr.transpose(2,0,1)[None], dtype=np.float32)
        value, logits = bridge(arr.tobytes(), 1, channels)
        logits = np.frombuffer(logits, dtype=np.float32)
        moves = game.get_search_actions()
        legal = np.array([logits[move_to_index(m)] for m in moves], dtype=np.float64)
        probs = np.exp(legal-legal.max()); probs /= probs.sum()
        rows.append(dict(case=case['id'], value_white=float(np.frombuffer(value,dtype=np.float32)[0]),
                         legal_policy={m.uci():float(p) for m,p in zip(moves,probs)}))
    return rows


def run(config_path, output):
    import torch
    torch.set_num_threads(1)
    config = study.read(config_path)
    out = Path(output)
    manifest = dict(config=file_hash(config_path), runtime=runtime_identity(),
                    implementation=file_hash(__file__),
                    models={p:file_hash(p) for p in config['models'].values()})
    study.pin(out/'manifest.json', manifest)
    if config['smoke']:
        p=config['models']['e14']; engine, normal = make_engine(p,p,8)
        task=dict(config['tasks'][0],white_model=p,black_model=p,value_model=p)
        study._engines={'probe':engine}
        a=study.run_task(task)
        engine._bridge=normal
        b=study.run_task(task)
        # Timing varies; action/state/value/policy must not.
        def logical(result):
            return [{k:v for k,v in r.items() if k!='decision_seconds'} for r in result['trajectory']]
        if logical(a)!=logical(b):
            raise ValueError('Same-model bridge parity failed')
        del engine, normal, a, b
        study._engines={}
        torch.cuda.empty_cache()
    raw_path=out/'raw_network.json'
    if not raw_path.exists():
        rows={}
        for name,path in config['models'].items():
            rows[name]=raw_network(path,config['cases'])
            torch.cuda.empty_cache()
        atomic_json(raw_path,dict(data=rows,data_sha256=digest(rows)))
    saved_raw=study.read(raw_path)
    if digest(saved_raw['data'])!=saved_raw['data_sha256']:
        raise ValueError('Changed raw network diagnostics')
    results=[]
    active=None
    for task in config['tasks']:
        path=out/'tasks'/(digest(task)+'.json')
        if path.exists():
            result=study.load_result(path,task)
        else:
            key=(task['white_model'],task['value_model'],task['white_sims'])
            if key!=active:
                study._engines={}
                engine=None
                torch.cuda.empty_cache()
                engine,_=make_engine(*key)
                study._engines={'probe':engine}
                active=key
            result=study.run_task(task)
            study.audit_result(task,result)
            atomic_json(path,dict(result_sha256=digest(result),result=result))
        results.append(result)
    rows=study.summarize(config['tasks'],results)
    # Generic probe summary labels policy checkpoint; preserve explicit value source too.
    for row,task in zip(rows['probes'],config['tasks']):
        row['value_model']=task['value_model']
    atomic_json(out/'summary.json',dict(complete=True, probes=rows['probes'],
        raw_network_sha256=file_hash(raw_path),manifest_sha256=file_hash(out/'manifest.json'),
        evidence={str(out/'tasks'/(digest(t)+'.json')):file_hash(out/'tasks'/(digest(t)+'.json'))
                  for t in config['tasks']}))
    print(f'RECOVERY PROBES COMPLETE: {len(results)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.config,args.output)
