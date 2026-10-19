"""Opening atlas: map the opening tree by search and by games (owner, 2026-10-19).

Owner: "Run an extremely comprehensive and deep opening categorization and
search, to map out basically everything." Follows the opening census
(docs/experiments/game_value/OPENING_CENSUS.md) with the current strongest
models and the full width of the tree where it is affordable.

The tree: 58 positions after White's first turn (W1), 1,156 after Black's reply
(B1), about 100,000 after White's second turn (W2). Labels: W1 = its two
half-moves sorted ("d2d4 e2e4"), B1 = "W1 / b", W2 = "W1 / b / m1 m2".

Players: the strongest per colour (COLOUR_BASELINES.md, GEN56_RESULTS.md).
White gen56, gen55; Black gen56, gen53. Four matches (MATCHES) play every book
entry twice with colours reversed, so each entry yields the cells
gen56>gen56 (2 games), gen56>gen53, gen53>gen56, gen55>gen56, gen56>gen55,
gen55>gen53, gen53>gen55 (White>Black). A position's value is the mean White
score over the four TARGET cells (White gen56/gen55 x Black gen56/gen53).
Draws count half. Outcomes decide; search values are a map and are calibrated
against outcomes in the report.

Phases, pre-declared (2026-10-19, before any result):
  S1  Search, all 1,156 B1 positions: gen56, gen55, gen53, v29 at 12,800 sims,
      no noise: White-view value, top-5 first half-moves, and the full best turn.
  G0  Games, all 58 W1 positions: 4 matches, 4 copies, 4 sampled half-moves.
  G1  Games, all 1,156 B1 positions: 4 matches, 1 copy, 4 sampled half-moves.
  G2  Refinement of Black's best replies: per W1, the 3 B1 with the lowest G1
      value, plus any within 0.10 of that minimum (at most 6); 4 matches,
      3 copies, 6 sampled half-moves. Pooled with G1 ("refined value").
  S2a Search, every W2 child of the G3 nodes: gen56 and gen53 at 3,200 sims.
  G3  Critical lines one turn deeper. W1 ranked by refined minimax value (Black
      picks the reply with the lowest refined value): the top 10 plus the four
      most-played W1 not among them. For each, Black's 2 best replies are the
      G3 nodes. Per node, the 12 W2 with the highest S2a mean value plus 4 random
      others (seeded): 4 matches, 1 copy, 4 sampled half-moves.
  D   Depth check at 12,800 sims: every G3 node, and per node the 2 W2 with the
      highest G3 value; 4 matches, 1 copy, 4 sampled half-moves.
  S2b Search, every W2 child of every B1 position not covered by S2a (full width,
      about 100,000 positions): gen56 and gen53 at 3,200. Runs last; resumable.

Seeds 3.060e9-3.079e9 (1M per match). Evidence: benchmarks/opening_atlas_20261019/.
Diagnostic only; produces no training data and no model.

    py -3 -u tools/opening_atlas.py [--rehearsal | --measure]
"""
import argparse
import collections
import concurrent.futures as futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
OUT = ROOT / "benchmarks/opening_atlas_20261019"
MODELS = {"gen56": "models/candidates/bootstrap_main_gen_0056/arena_selected.pt",
          "gen55": "models/candidates/bootstrap_main_gen_0055/arena_selected.pt",
          "gen53": "models/candidates/bootstrap_main_gen_0053/arena_selected.pt",
          "v29": "models/bootstrap_v29/best_value_net.pt"}
MATCHES = [("gen56_self", "gen56", "gen56"), ("gen56_gen53", "gen56", "gen53"),
           ("gen55_gen56", "gen55", "gen56"), ("gen55_gen53", "gen55", "gen53")]
TARGET_CELLS = [f"{w}>{b}" for w in ("gen56", "gen55") for b in ("gen56", "gen53")]
MOST_PLAYED = ["d2d4 e2e4", "e2e4 e4e5", "d2d4 f2f4", "d2d4 d4d5"]
SEED = 3_060_000_000
SIMS, DEEP_SIMS = 3200, 12800
S1_MODELS, S2_MODELS = ["gen56", "gen55", "gen53", "v29"], ["gen56", "gen53"]


# ---------------------------------------------------------------- the tree

def replay(moves):
    import chess
    from monster_chess import MonsterChessGame
    g = MonsterChessGame()
    for uci in moves:
        g.apply_search_action(chess.Move.from_uci(uci))
    return g


def entry(g):
    return dict(fen=g.fen(), half=bool(g.white_half_pending), turn_count=g.turn_count)


def first_turns():
    from opening_census import first_turns as census_first_turns
    return {k: dict(moves=v["moves"], entry=v["entry"]) for k, v in census_first_turns().items()}


def black_replies(w1_label, w1):
    from opening_census import black_replies as census_replies
    return {f"{w1_label} / {b}": dict(moves=v["moves"], entry=v["entry"])
            for b, v in census_replies(w1).items()}


def white_turns(b1_label, b1):
    """Every distinct position after White's next full turn (Black to move), transpositions merged."""
    import chess
    from repetition import position_key
    g = replay(b1["moves"])
    out, seen = {}, set()
    for m1 in g.get_search_actions():
        g1 = g.clone(); g1.apply_search_action(chess.Move.from_uci(m1.uci()))
        seconds = [None] if (g1.is_terminal() or not g1.white_half_pending) else g1.get_search_actions()
        for m2 in seconds:
            g2 = g1 if m2 is None else g1.clone()
            if m2 is not None:
                g2.apply_search_action(chess.Move.from_uci(m2.uci()))
            key = position_key(g2)
            if key in seen:
                continue
            seen.add(key)
            mv = [m1.uci()] + ([] if m2 is None else [m2.uci()])
            out[f"{b1_label} / {' '.join(mv)}"] = dict(moves=b1["moves"] + mv, entry=entry(g2),
                                                       terminal=g2.is_terminal(),
                                                       result=g2.get_result() if g2.is_terminal() else None)
    return out


def tree():
    w1 = first_turns()
    b1 = {}
    for label in sorted(w1):
        b1.update(black_replies(label, w1[label]))
    return w1, b1


# ---------------------------------------------------------------- games

def complete(report, games):
    if not report.exists():
        return False
    r = json.loads(report.read_text())
    return r.get("a_as_white", {}).get("games", 0) + r.get("a_as_black", {}).get("games", 0) == games


def game_phase(name, labels, entries, copies, sims, temp_plies, seed_base, matches=MATCHES):
    """Play every label x copies under every match; per-position cells (opening_census.aggregate)."""
    from opening_census import aggregate, write_book
    phase_dir = OUT / name
    book = phase_dir / "book.json"
    write_book(book, labels, entries, copies, f"opening atlas {name}")
    n = len(labels) * copies
    for i, (match, a, b) in enumerate(matches):
        report, log = phase_dir / f"{match}.json", phase_dir / f"{match}.jsonl"
        if complete(report, 2 * n):
            continue
        print(f"ATLAS {name} {match}: {2 * n} games at {sims}", flush=True)
        subprocess.run([sys.executable, "-u", "tools/match.py", "--model-a", MODELS[a], "--model-b", MODELS[b],
                        "--games", str(2 * n), "--sims", str(sims), "--sims-b", str(sims), "--engine", "native",
                        "--workers", "8", "--seed", str(seed_base + i * 1_000_000), "--book", str(book),
                        "--book-temp-plies", str(temp_plies), "--stall-timeout", "1800",
                        "--game-log", str(log), "--report-path", str(report), "--resume"], check=True, cwd=ROOT)
    cells = aggregate(phase_dir, labels, copies, matches)
    (phase_dir / "report.json").write_text(json.dumps(dict(labels=labels, copies=copies, sims=sims,
                                                           temp_plies=temp_plies, matches=matches,
                                                           positions=cells), indent=1))
    print(f"ATLAS {name} COMPLETE", flush=True)
    return cells


def pool(*reports):
    """Pool target-cell W/D/L across phases: label -> dict(w, d, l, n, value)."""
    out = collections.defaultdict(lambda: dict(w=0, d=0, l=0))
    for report in reports:
        for label, cells in report.items():
            for c in TARGET_CELLS:
                if c in cells:
                    for k in "wdl":
                        out[label][k] += cells[c][k]
    for v in out.values():
        v["n"] = v["w"] + v["d"] + v["l"]
        v["value"] = round((v["w"] + 0.5 * v["d"]) / v["n"], 4) if v["n"] else None
    return dict(out)


def target_value(cells):
    """Mean of the four target cells' White scores (equal weight per cell)."""
    scores = [cells[c]["white_score"] for c in TARGET_CELLS if c in cells and cells[c]["white_score"] is not None]
    return round(sum(scores) / len(scores), 4) if scores else None


# ---------------------------------------------------------------- search

_engines = {}


def _init_search(models):
    import torch
    torch.set_num_threads(1)
    global _MODELS
    _MODELS = models


def _engine(model, sims):
    from benchmark import _build_engine
    key = (model, sims)
    if key not in _engines:
        built = _build_engine(_MODELS[model], sims, engine="native")
        engine = built[0] if isinstance(built, tuple) else built
        if hasattr(engine, "root_noise"):
            engine.root_noise = False
        _engines[key] = engine
    engine = _engines[key]
    for attr in ("_reuse_tree", "_reuse_key"):
        if hasattr(engine, attr):
            setattr(engine, attr, None)
    return engine


def _search(task):
    """task = dict(key, model, sims, entry): White-view value, top-5 moves, the full best turn."""
    from monster_chess import MonsterChessGame
    import random as _random
    import zlib
    _random.seed(zlib.crc32(f"{task['key']}|{task['model']}".encode()))
    e = task["entry"]
    g = MonsterChessGame(fen=e["fen"])
    g.white_half_pending, g.turn_count = bool(e["half"]), int(e["turn_count"])
    engine = _engine(task["model"], task["sims"])
    white = g.is_white_turn
    move, probs, value = engine.get_best_action(g, temperature=0.0)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
    turn = [move.uci()] if move is not None else []
    if move is not None and white:
        g.apply_search_action(move)
        if g.white_half_pending and not g.is_terminal():
            engine2 = _engine(task["model"], task["sims"])
            m2, _p, _v = engine2.get_best_action(g, temperature=0.0)
            if m2 is not None:
                turn.append(m2.uci())
    return dict(key=task["key"], model=task["model"], sims=task["sims"],
                value_white=None if value is None else round(float(value) * (1 if white else -1), 4),
                top=[(u, round(float(p), 4)) for u, p in top], best_turn=turn)


def search_phase(name, positions, models, sims, workers=8):
    """positions: label -> entry. Appends to OUT/name/search.jsonl; resumable by (label, model)."""
    path = OUT / name / "search.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["key"], r["model"]))
    tasks = [dict(key=label, model=m, sims=sims, entry=e) for label, e in positions.items() for m in models
             if (label, m) not in done]
    print(f"ATLAS {name}: {len(tasks)} searches at {sims} ({len(done)} done)", flush=True)
    if tasks:
        start, count = time.time(), 0
        with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                         initializer=_init_search, initargs=(MODELS,)) as ex, \
                open(path, "a", encoding="utf-8") as out:
            for r in ex.map(_search, tasks, chunksize=4):
                out.write(json.dumps(r) + "\n")
                count += 1
                if count % 200 == 0 or count == len(tasks):
                    out.flush()
                    rate = count / (time.time() - start)
                    print(f"  [{count}/{len(tasks)}] {rate * 60:.0f}/min, ~{(len(tasks) - count) / rate / 60:.0f} min left",
                          flush=True)
    results = collections.defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        results[r["key"]][r["model"]] = r
    print(f"ATLAS {name} COMPLETE", flush=True)
    return dict(results)


# ---------------------------------------------------------------- selection rules

def g2_selection(b1_by_w1, g1_values):
    chosen = []
    for w1, replies in sorted(b1_by_w1.items()):
        ranked = sorted((g1_values[r], r) for r in replies if g1_values.get(r) is not None)
        if not ranked:
            continue
        floor = ranked[0][0]
        pick = [r for _, r in ranked[:3]] + [r for v, r in ranked[3:] if v <= floor + 0.10 + 1e-12]
        chosen += pick[:6]
    return chosen


def refined_minimax(b1_by_w1, g1_values, pooled, refined):
    """W1 -> (value, Black's best reply), using pooled G1+G2 values where refined, else G1."""
    out = {}
    for w1, replies in b1_by_w1.items():
        vals = [((pooled[r]["value"] if r in refined else g1_values[r]), r) for r in replies
                if g1_values.get(r) is not None]
        if vals:
            out[w1] = min(vals)
    return out


def g3_nodes(minimax, b1_by_w1, g1_values, pooled, refined):
    ranked = sorted(minimax, key=lambda w: (-minimax[w][0], w))
    w1s = ranked[:10] + [w for w in MOST_PLAYED if w not in ranked[:10]]
    nodes = []
    for w1 in w1s:
        vals = sorted(((pooled[r]["value"] if r in refined else g1_values[r]), r) for r in b1_by_w1[w1]
                      if g1_values.get(r) is not None)
        nodes += [r for _, r in vals[:2]]
    return w1s, nodes


def w2_candidates(node_index, children, s2):
    live = {k: v for k, v in children.items() if not v["terminal"]}
    def mean_value(k):
        vs = [s2[k][m]["value_white"] for m in S2_MODELS if k in s2 and m in s2[k] and s2[k][m]["value_white"] is not None]
        return sum(vs) / len(vs) if vs else -9
    ranked = sorted(live, key=lambda k: (-mean_value(k), k))
    top, rest = ranked[:12], ranked[12:]
    rng = random.Random(SEED + node_index)
    return top, sorted(rng.sample(rest, min(4, len(rest))))


# ---------------------------------------------------------------- driver

def run(rehearsal=False):
    global OUT
    w1, b1 = tree()
    assert len(w1) == 58 and len(b1) == 1156, (len(w1), len(b1))
    b1_by_w1 = collections.defaultdict(list)
    for label in b1:
        b1_by_w1[label.split(" / ")[0]].append(label)
    sims, deep, seed = SIMS, DEEP_SIMS, SEED
    w1_labels, b1_labels = sorted(w1), sorted(b1)
    if rehearsal:
        OUT = OUT / "rehearsal"
        sims, deep, seed = 8, 16, SEED + 18_000_000
        w1_labels = w1_labels[:2]
        b1_by_w1 = {w: sorted(b1_by_w1[w])[:4] for w in w1_labels}
        b1_labels = [r for w in w1_labels for r in b1_by_w1[w]]
    OUT.mkdir(parents=True, exist_ok=True)
    w1_entries = {k: w1[k]["entry"] for k in w1_labels}
    b1_entries = {k: b1[k]["entry"] for k in b1_labels}

    s1 = search_phase("s1", b1_entries, S1_MODELS, deep)
    g0 = game_phase("g0", w1_labels, w1_entries, 1 if rehearsal else 4, sims, 4, seed)
    g1 = game_phase("g1", b1_labels, b1_entries, 1, sims, 4, seed + 4_000_000)
    g1_values = {k: target_value(v) for k, v in g1.items()}

    refined = g2_selection(b1_by_w1, g1_values)
    g2 = game_phase("g2", refined, b1_entries, 1 if rehearsal else 3, sims, 6, seed + 8_000_000)
    pooled = pool(g1, g2)
    minimax = refined_minimax(b1_by_w1, g1_values, pooled, set(refined))
    w1s, nodes = g3_nodes(minimax, b1_by_w1, g1_values, pooled, set(refined))
    if rehearsal:
        nodes = nodes[:2]
    children = {node: white_turns(node, b1[node]) for node in nodes}
    s2a_positions = {k: v["entry"] for c in children.values() for k, v in c.items() if not v["terminal"]}
    s2a = search_phase("s2a", s2a_positions, S2_MODELS, sims)

    g3_labels, g3_entries, g3_sets = [], {}, {}
    for i, node in enumerate(nodes):
        top, controls = w2_candidates(i, children[node], s2a)
        if rehearsal:
            top, controls = top[:2], controls[:1]
        g3_sets[node] = dict(top=top, controls=controls)
        for k in top + controls:
            g3_labels.append(k); g3_entries[k] = children[node][k]["entry"]
    g3 = game_phase("g3", g3_labels, g3_entries, 1, sims, 4, seed + 12_000_000)

    d_labels, d_entries = list(nodes), {n: b1[n]["entry"] for n in nodes}
    for node in nodes:
        ranked = sorted(g3_sets[node]["top"] + g3_sets[node]["controls"],
                        key=lambda k: (-(target_value(g3[k]) or 0), k))
        for k in ranked[:2]:
            d_labels.append(k); d_entries[k] = g3_entries[k]
    d = game_phase("d", d_labels, d_entries, 1, deep, 4, seed + 16_000_000)

    selection = dict(g2=refined, minimax={w: dict(value=v, reply=r) for w, (v, r) in minimax.items()},
                     g3_w1=w1s, g3_nodes=nodes, g3_sets=g3_sets, d=d_labels,
                     terminal_w2={n: {k: v["result"] for k, v in c.items() if v["terminal"]} for n, c in children.items()})
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1))
    print("ATLAS GAMES COMPLETE", flush=True)

    rest = {}
    for label in b1_labels:
        if label in children:
            continue
        for k, v in white_turns(label, b1[label]).items():
            if not v["terminal"]:
                rest[k] = v["entry"]
        if rehearsal and len(rest) > 20:
            break
    search_phase("s2b", rest, S2_MODELS, sims)
    print(f"ATLAS {'REHEARSAL' if rehearsal else 'ALL'} COMPLETE", flush=True)


def measure():
    """Searches per minute at 3,200 and 12,800 on real B1 positions (for the cost estimate)."""
    global OUT
    OUT = OUT / "measure"
    _, b1 = tree()
    labels = sorted(b1)[::29][:40]
    for sims in (SIMS, DEEP_SIMS):
        t = time.time()
        search_phase(f"sims{sims}", {k: b1[k]["entry"] for k in labels}, ["gen56", "gen53"], sims)
        print(f"MEASURE {sims}: {2 * len(labels)} searches in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.measure:
        measure()
    else:
        run(args.rehearsal)


if __name__ == "__main__":
    mp.freeze_support()
    main()
