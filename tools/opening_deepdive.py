"""Opening deep dive: White's best tries against Black's best replies, two turns deep (owner, 2026-10-21).

docs/experiments/game_value/OPENING_ATLAS.md, "Next". Owner: "Afterwards, we can
hone in on those 6 first turns, and their only reply(s)" and, approving: "Just make
sure a certain line isn't judged on, say, the 3 best replies after it if 2 are
substantially worse than the best response. Go ahead".

Scoring is strict minimax, never an average over moves:
  - a position where Black moves is worth its BEST reply for Black (the minimum);
  - a position where White moves is worth its BEST move for White (the maximum).
A move "substantially worse" than the best (more than TOL = 0.10 below it for the
mover) is dropped. Moves within TOL are kept only as candidates for the best,
because the games cannot yet tell them apart; they get more games, and the value
is still the single best one. Every reported value is measured on games that
played no part in choosing the move (selection on one sample, value on another).

Pre-declared (2026-10-21, before any result), with the atlas players, cells and
scoring (tools/opening_atlas.py):
  Lines   For each of the six first turns e4-e5, f4-f5, c4+f3, c4+d4, e4+d4, d4+f4:
          Black's candidate replies = the best by refined atlas value (G1 pooled with
          G2) plus any within TOL of it, at most 4. Each (first turn, reply) is a line.
  A       Every non-terminal White second turn on every line: 4 matches x 2 copies
          (16 games), 3,200 sims, 4 sampled half-moves.
  B       Per line, the 4 best second turns by A: fresh 16 games each at 12,800.
          W2* = the best by B (ties: A).
  C       Every Black reply to W2*: 4 matches x 1 copy (8 games), 3,200 sims.
  CR      Per line, Black's candidate replies to W2* = the best by C plus any within
          TOL, at most 4: fresh 24 games each (4 matches x 3 copies), 3,200 sims.
  Values  line = min over CR candidates of their CR value (Black's best answer to
          White's best try); first turn = min over its lines (Black picks the line).
          Also reported: B value of A's choice (depth check of an unbiased pick).

Seeds 3.080e9-3.0959e9 (rehearsal 3.030e9-3.0459e9). Evidence:
benchmarks/opening_deepdive_20261021/. Diagnostic only; no training data, no model.

    py -3 -u tools/opening_deepdive.py [--rehearsal]
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import opening_atlas as A

OUT = ROOT / "benchmarks/opening_deepdive_20261021"
ATLAS = ROOT / "benchmarks/opening_atlas_20261019"
DIVE_W1 = ["e2e4 e4e5", "f2f4 f4f5", "c2c4 f2f3", "c2c4 d2d4", "d2d4 e2e4", "d2d4 f2f4"]
TOL, MAX_CANDIDATES = 0.10, 4
SEED, REHEARSAL_SEED = 3_080_000_000, 3_030_000_000


def candidates(values, best_is_low, tol=TOL, cap=MAX_CANDIDATES):
    """The best move for the mover plus any within `tol` of it (never a substantially worse one).

    values: label -> White-view value. Black wants it low (best_is_low=True), White high.
    Ordered best first; ties broken by label so the rule is deterministic.
    """
    live = {k: v for k, v in values.items() if v is not None}
    if not live:
        return []
    order = sorted(live, key=lambda k: (live[k] if best_is_low else -live[k], k))
    best = live[order[0]]
    near = [k for k in order if (live[k] <= best + tol + 1e-12 if best_is_low else live[k] >= best - tol - 1e-12)]
    return near[:cap]


def black_replies_at(label, moves):
    """Every Black reply (White to move afterwards) from the position after `moves`."""
    from opening_census import black_replies
    return {f"{label} / {b}": dict(moves=v["moves"], entry=v["entry"])
            for b, v in black_replies(dict(moves=moves)).items()}


def run(rehearsal=False):
    global OUT
    w1, b1 = A.tree()
    by = collections.defaultdict(list)
    for k in b1:
        by[k.split(" / ")[0]].append(k)
    g1 = json.loads((ATLAS / "g1/report.json").read_text())["positions"]
    g2 = json.loads((ATLAS / "g2/report.json").read_text())["positions"]
    g1v = {k: A.target_value(v) for k, v in g1.items()}
    pooled = A.pool(g1, g2)
    refined = set(A.g2_selection(by, g1v))
    rv = {k: (pooled[k]["value"] if k in refined else g1v[k]) for k in g1v}

    seed, sims, deep = SEED, A.SIMS, A.DEEP_SIMS
    dive_w1 = list(DIVE_W1)
    if rehearsal:
        OUT = OUT / "rehearsal"
        seed, sims, deep, dive_w1 = REHEARSAL_SEED, 8, 16, dive_w1[:2]
    OUT.mkdir(parents=True, exist_ok=True)
    A.OUT = OUT                                   # game_phase / search_phase write under OUT

    lines = {w: candidates({k: rv[k] for k in by[w]}, best_is_low=True) for w in dive_w1}
    if rehearsal:
        lines = {w: v[:1] for w, v in lines.items()}
    nodes = [n for w in dive_w1 for n in lines[w]]

    # A: every White second turn on every line
    kids, terminal = {}, {}
    a_labels, a_entries = [], {}
    for n in nodes:
        kids[n] = A.white_turns(n, b1[n])
        terminal[n] = {k: v["result"] for k, v in kids[n].items() if v["terminal"]}
        live = sorted(k for k, v in kids[n].items() if not v["terminal"])
        if rehearsal:
            live = live[:3]
        for k in live:
            a_labels.append(k); a_entries[k] = kids[n][k]["entry"]
    a = A.game_phase("a", a_labels, a_entries, 1 if rehearsal else 2, sims, 4, seed)
    a_val = {k: A.target_value(v) for k, v in a.items()}

    # B: the 4 best second turns per line by A, fresh games at depth
    top = {}
    b_labels, b_entries = [], {}
    for n in nodes:
        mine = {k: a_val[k] for k in a_val if k.startswith(n + " / ")}
        top[n] = sorted(mine, key=lambda k: (-mine[k], k))[:4]
        for k in top[n]:
            b_labels.append(k); b_entries[k] = a_entries[k]
    b = A.game_phase("b", b_labels, b_entries, 1 if rehearsal else 2, deep, 4, seed + 4_000_000)
    b_val = {k: A.target_value(v) for k, v in b.items()}
    star = {n: sorted(top[n], key=lambda k: (-b_val[k], -a_val[k], k))[0] for n in nodes}

    # C: every Black reply to W2*; CR: Black's candidates, fresh games
    c_labels, c_entries, c_moves = [], {}, {}
    for n in nodes:
        w2 = star[n]
        replies = black_replies_at(w2, kids[n][w2]["moves"])
        keys = sorted(replies)[:3] if rehearsal else sorted(replies)
        for k in keys:
            c_labels.append(k); c_entries[k] = replies[k]["entry"]; c_moves[k] = replies[k]["moves"]
    c = A.game_phase("c", c_labels, c_entries, 1, sims, 4, seed + 8_000_000)
    c_val = {k: A.target_value(v) for k, v in c.items()}
    cr_sets = {n: candidates({k: c_val[k] for k in c_val if k.startswith(star[n] + " / ")}, best_is_low=True)
               for n in nodes}
    cr_labels = [k for n in nodes for k in cr_sets[n]]
    cr = A.game_phase("cr", cr_labels, {k: c_entries[k] for k in cr_labels}, 1 if rehearsal else 3, sims, 4,
                      seed + 12_000_000)
    cr_val = {k: A.target_value(v) for k, v in cr.items()}

    line_value = {n: min(cr_val[k] for k in cr_sets[n]) for n in nodes}
    line_reply = {n: min(cr_sets[n], key=lambda k: (cr_val[k], k)) for n in nodes}
    w1_value = {w: min(line_value[n] for n in lines[w]) for w in dive_w1}
    a_choice = {n: top[n][0] for n in nodes}
    result = dict(
        rule=dict(tol=TOL, max_candidates=MAX_CANDIDATES, scoring="strict minimax; values on fresh games"),
        lines={w: [dict(line=n, atlas_value=rv[n]) for n in lines[w]] for w in dive_w1},
        first_turn_value=w1_value,
        per_line={n: dict(
            atlas_value=rv[n], white_terminal_wins=terminal[n],
            a_top10=sorted(((a_val[k], k.split(" / ")[2]) for k in a_val if k.startswith(n + " / ")), reverse=True)[:10],
            b=[dict(w2=k.split(" / ")[2], a=a_val[k], b=b_val[k]) for k in top[n]],
            w2_star=star[n].split(" / ")[2], w2_star_a=a_val[star[n]], w2_star_b=b_val[star[n]],
            a_choice=a_choice[n].split(" / ")[2], a_choice_depth_value=b_val[a_choice[n]],
            c_lowest=sorted((c_val[k], k.split(" / ")[3]) for k in c_val if k.startswith(star[n] + " / "))[:6],
            black_candidates=[dict(b2=k.split(" / ")[3], c=c_val[k], cr=cr_val[k]) for k in cr_sets[n]],
            value=line_value[n], black_best=line_reply[n].split(" / ")[3]) for n in nodes})
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    for w in dive_w1:
        print(f"DIVE {w}: {w1_value[w] * 100:.1f}%  " + "; ".join(
            f"…{n.split(' / ')[1]} → {star[n].split(' / ')[2]} → …{line_reply[n].split(' / ')[3]} "
            f"{line_value[n] * 100:.1f}%" for n in lines[w]), flush=True)
    print(f"DIVE {'REHEARSAL' if rehearsal else 'ALL'} COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehearsal", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    run(args.rehearsal)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
