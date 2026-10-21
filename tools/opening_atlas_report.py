"""Opening atlas report: tables and a summary JSON from the atlas phases (CPU only).

Reads benchmarks/opening_atlas_20261019/{s1,g0,g1,g2,s2a,g3,d,s2b}; writes
atlas_summary.json and prints the markdown tables used in
docs/experiments/game_value/OPENING_ATLAS.md. Values are White's score over the
four target cells (White gen56/gen55 x Black gen56/gen53), draws half.

    py -3 tools/opening_atlas_report.py
"""
import collections
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import opening_atlas as A

R = A.OUT


def load(name):
    return json.loads((R / name / "report.json").read_text())["positions"]


def search(name):
    out = collections.defaultdict(dict)
    path = R / name / "search.jsonl"
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["key"]][r["model"]] = r
    return out


def moved_pieces(moves, prefix):
    """For the half-moves after `prefix`, the moved piece ('P' or 'K') and the move, in order."""
    import chess
    g = A.replay(prefix)
    out = []
    for uci in moves:
        m = chess.Move.from_uci(uci)
        piece = g.board.piece_at(m.from_square)
        out.append(("K" if piece and piece.piece_type == chess.KING else "P", uci))
        g.apply_search_action(m)
    return out


def white_family(moves, prefix):
    seq = moved_pieces(moves, prefix)
    kinds = [k for k, _ in seq]
    if kinds == ["P", "P"] and seq[0][1][2:] == seq[1][1][:2]:
        return "one pawn twice"
    if kinds == ["P", "P"]:
        return "two pawns"
    if kinds == ["K", "K"]:
        return "king walk"
    if len(kinds) == 1:
        return "single half-move"
    return "pawn + king"


def white_detail(moves, prefix):
    seq = moved_pieces(moves, prefix)
    parts = []
    for kind, uci in seq:
        if kind == "K":
            parts.append("K")
        else:
            steps = abs(int(uci[3]) - int(uci[1]))
            parts.append(f"{uci[0]}{'+2' if steps == 2 else '+1'}")
    return " ".join(sorted(parts))


def black_family(b):
    if b[0] in "bg" and b[1] == "8":
        return "knight"
    steps = abs(int(b[3]) - int(b[1]))
    side = "centre pawn" if b[0] in "cdef" else "flank pawn"
    return f"{side} {'two' if steps == 2 else 'one'} step{'s' if steps == 2 else ''}"


def main():
    w1, b1 = A.tree()
    by = collections.defaultdict(list)
    for k in b1:
        by[k.split(" / ")[0]].append(k)
    g0, g1, g2, g3, d = load("g0"), load("g1"), load("g2"), load("g3"), load("d")
    s1, s2a, s2b = search("s1"), search("s2a"), search("s2b")
    g1v = {k: A.target_value(v) for k, v in g1.items()}
    refined = set(A.g2_selection(by, g1v))
    pooled = A.pool(g1, g2)
    rv = {k: (pooled[k]["value"] if k in refined else g1v[k]) for k in g1v}
    mm = A.refined_minimax(by, g1v, pooled, refined)
    w1s, nodes = A.g3_nodes(mm, by, g1v, pooled, refined)

    # deep value per W1 from D: min over its G3 replies of max over the replayed second turns
    deep_w2 = collections.defaultdict(list)
    for k, cells in d.items():
        p = k.split(" / ")
        if len(p) == 3:
            deep_w2[" / ".join(p[:2])].append((A.target_value(cells), A.target_value(g3[k]), p[2]))
    deep_w1 = {}
    for w in w1s:
        reps = [n for n in nodes if n.startswith(w + " / ")]
        if reps and all(n in deep_w2 for n in reps):
            deep_w1[w] = min(max(deep_w2[n])[0] for n in reps)

    # S1 engine minimax per W1 (mean of four models)
    s1mm = {w: min(st.mean(s1[k][m]["value_white"] for m in A.S1_MODELS) for k in ks) for w, ks in by.items()}

    w1_rows = []
    for w in sorted(by, key=lambda w: (-mm[w][0], w)):
        vals = [rv[k] for k in by[w]]
        w1_rows.append(dict(w1=w, family=white_family(w1[w]["moves"], []), detail=white_detail(w1[w]["moves"], []),
                            minimax=mm[w][0], reply=mm[w][1].split(" / ")[1],
                            reply_games=pooled[mm[w][1]]["n"] if mm[w][1] in refined else None,
                            le10=sum(v <= .10 + 1e-12 for v in vals), le20=sum(v <= .20 + 1e-12 for v in vals),
                            replies=len(vals), mean_over_replies=round(st.mean(vals), 4),
                            g0=A.target_value(g0[w]), deep=deep_w1.get(w), engine_minimax=round(s1mm[w], 3)))

    fam = collections.defaultdict(list)
    for r in w1_rows:
        fam[r["family"]].append(r)
    w1_families = {f: dict(count=len(rs), mean_minimax=round(st.mean(r["minimax"] for r in rs), 4),
                           best=max(rs, key=lambda r: r["minimax"])["w1"],
                           best_minimax=max(r["minimax"] for r in rs),
                           zero=sum(r["minimax"] == 0 for r in rs),
                           mean_g0=round(st.mean(r["g0"] for r in rs), 4)) for f, rs in fam.items()}

    bfam = collections.defaultdict(list)
    best_reply_fam = collections.Counter()
    for k, v in rv.items():
        bfam[black_family(k.split(" / ")[1])].append(v)
    for w in by:
        best_reply_fam[black_family(mm[w][1].split(" / ")[1])] += 1
    b1_families = {f: dict(positions=len(vs), mean_white=round(st.mean(vs), 4),
                           share_le10=round(sum(v <= .1 for v in vs) / len(vs), 3),
                           best_reply_count=best_reply_fam[f]) for f, vs in bfam.items()}

    w2fam = collections.defaultdict(list)
    for k, cells in g3.items():
        p = k.split(" / ")
        node = " / ".join(p[:2])
        w2fam[white_family(p[2].split(), b1[node]["moves"])].append(A.target_value(cells))
    w2_families = {f: dict(positions=len(vs), mean_white=round(st.mean(vs), 4), best=max(vs))
                   for f, vs in w2fam.items()}

    # calibration: search value (White view) against game outcome (2*score-1)
    def calib(search_res, games, models):
        out = {}
        keys = [k for k in games if k in search_res]
        y = [2 * A.target_value(games[k]) - 1 for k in keys]
        for m in models:
            x = [search_res[k][m]["value_white"] for k in keys if m in search_res[k]]
            if len(x) != len(y) or len(x) < 3:
                continue
            bins = collections.defaultdict(list)
            for xi, yi in zip(x, y):
                bins[round(max(-1, min(1, xi)) * 4) / 4].append(yi)
            out[m] = dict(n=len(x), mean_search=round(st.mean(x), 3), mean_outcome=round(st.mean(y), 3),
                          bias=round(st.mean(x) - st.mean(y), 3), corr=round(st.correlation(x, y), 3),
                          bins={str(b): dict(n=len(v), outcome=round(st.mean(v), 3)) for b, v in sorted(bins.items())})
        return out
    calibration = dict(s1_vs_g1=calib(s1, g1, A.S1_MODELS), s2a_vs_g3=calib(s2a, g3, A.S2_MODELS))

    # model disagreement at the critical nodes (S1 best full turns), with game values where played
    disagreement = {}
    for n in nodes:
        picks = {m: " ".join(s1[n][m]["best_turn"]) for m in A.S1_MODELS}
        played = {" ".join(k.split(" / ")[2].split()): A.target_value(g3[k]) for k in g3 if k.startswith(n + " / ")}
        def value_of(turn):
            for k, v in played.items():
                if sorted(k.split()) == sorted(turn.split()):
                    return v
            return None
        disagreement[n] = dict(picks=picks, game_values={m: value_of(t) for m, t in picks.items()},
                               s1_values={m: s1[n][m]["value_white"] for m in A.S1_MODELS},
                               best_by_games=max(played.items(), key=lambda kv: kv[1]) if played else None,
                               deep_best=max(deep_w2[n])[2] if n in deep_w2 else None)

    # S2b + S2a: engine-preferred White second turn per B1 position (mean of gen56 and gen53)
    s2 = dict(s2a); s2.update(s2b)
    pref_fam = collections.Counter()
    node_best = {}
    by_node = collections.defaultdict(list)
    for k, res in s2.items():
        if all(m in res for m in A.S2_MODELS):
            by_node[" / ".join(k.split(" / ")[:2])].append((st.mean(res[m]["value_white"] for m in A.S2_MODELS), k))
    for n, lst in by_node.items():
        v, k = max(lst)
        node_best[n] = dict(w2=k.split(" / ")[2], value=round(v, 3), children=len(lst))
        pref_fam[white_family(k.split(" / ")[2].split(), b1[n]["moves"])] += 1
    engine_w2 = dict(nodes=len(node_best), positions=sum(x["children"] for x in node_best.values()),
                     preferred_families=dict(pref_fam))

    summary = dict(w1=w1_rows, w1_families=w1_families, b1_families=b1_families, w2_families_g3=w2_families,
                   g3_w1=w1s, g3_nodes=nodes,
                   deep={n: sorted(([dict(w2=w2, deep=dv, shallow=sv) for dv, sv, w2 in v]), key=lambda r: -r["deep"])
                         for n, v in deep_w2.items()},
                   deep_nodes={k: dict(deep=A.target_value(c), shallow=rv.get(k)) for k, c in d.items()
                               if len(k.split(" / ")) == 2},
                   calibration=calibration, disagreement=disagreement, engine_w2=engine_w2,
                   engine_w2_best=node_best)
    (R / "atlas_summary.json").write_text(json.dumps(summary, indent=1))

    pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"
    print("## W1\n| First turn | Family | Minimax | Best reply | ≤10% | ≤20% | Mean over replies | G0 | Deep | Engine |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in w1_rows:
        print(f"| {r['w1']} | {r['family']} | {pct(r['minimax'])} | …{r['reply']} | {r['le10']} | {r['le20']} | "
              f"{pct(r['mean_over_replies'])} | {pct(r['g0'])} | {pct(r['deep'])} | {r['engine_minimax']:+.2f} |")
    print("\n## W1 families"); [print(f, v) for f, v in w1_families.items()]
    print("\n## B1 families"); [print(f, v) for f, v in sorted(b1_families.items())]
    print("\n## W2 families (G3)"); [print(f, v) for f, v in w2_families.items()]
    print("\n## calibration"); print(json.dumps({k: {m: {x: y for x, y in v.items() if x != 'bins'} for m, v in c.items()}
                                                for k, c in calibration.items()}, indent=1))
    print("\n## disagreement"); [print(n, v["picks"], v["game_values"], v["best_by_games"], v["deep_best"]) for n, v in disagreement.items()]
    print("\n## engine W2", engine_w2)


if __name__ == "__main__":
    main()
