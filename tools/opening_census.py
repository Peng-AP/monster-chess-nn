"""Opening census: what is each White first turn worth against the strongest Black? (owner, 2026-10-12)

docs/experiments/game_value/GAME_VALUE_20261012.md. White has 58 distinct
positions after its first turn; the strong models play 11 of them, 87% e4+d4.
This plays every one of them, then goes deeper where it matters. Pre-declared:

Phase 1 (breadth, 3,200 sims): all 58 first-turn positions (Black to move),
    10 copies each, 4 sampled half-moves after the position so games differ.
    Pairings gen55-v29 (both directions), gen55-gen55, v29-v29: per position,
    White's score for each (White player, Black player) combination.
Phase 2 (depth, 12,800 sims): the 12 first turns with the highest mean White
    score in phase 1, plus any of the 4 most-played turns (e4+d4, e4+e5,
    d4+f4, d4+d5) not among them; 6 copies, same pairings.
Phase 3 (Black's answer, 3,200 sims): for the 6 best first turns by phase-2
    mean White score, every Black reply (White to move); 4 copies;
    gen55-v29 both directions. Which replies refute which setups.

Results: benchmarks/opening_census_20261012/<phase>/report.json and
summary.md. Resumable: finished matches are skipped, unfinished ones resume.
Diagnostic only; produces no training data and no model.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
OUT = ROOT / "benchmarks/opening_census_20261012"
MODELS = {"gen55": "models/candidates/bootstrap_main_gen_0055/arena_selected.pt",
          "v29": "models/bootstrap_v29/best_value_net.pt"}
PAIRINGS = [("gen55_v29", "gen55", "v29"), ("gen55_self", "gen55", "gen55"), ("v29_self", "v29", "v29")]
MOST_PLAYED = ["d2d4 e2e4", "e2e4 e4e5", "d2d4 f2f4", "d2d4 d4d5"]
SEED = 3_640_000_000   # clear of every campaign block; 1M per match
BOOK_TEMP_PLIES = 4


def first_turns():
    """The 58 distinct positions after White's first turn: label -> book entry."""
    import chess
    from monster_chess import MonsterChessGame
    from repetition import position_key
    g0, out = MonsterChessGame(), {}
    for m1 in g0.get_search_actions():
        g1 = g0.clone(); g1.apply_search_action(chess.Move.from_uci(m1.uci()))
        for m2 in g1.get_search_actions():
            g2 = g1.clone(); g2.apply_search_action(chess.Move.from_uci(m2.uci()))
            key = position_key(g2)
            label = " ".join(sorted([m1.uci(), m2.uci()]))
            if key not in {v["key"] for v in out.values()}:
                out[label] = dict(key=key, entry=dict(fen=g2.fen(), half=bool(g2.white_half_pending),
                                                      turn_count=g2.turn_count), moves=[m1.uci(), m2.uci()])
    return out


def black_replies(first):
    """Every Black reply to one first-turn position: label -> book entry (White to move)."""
    import chess
    from monster_chess import MonsterChessGame
    g = MonsterChessGame()
    for uci in first["moves"]:
        g.apply_search_action(chess.Move.from_uci(uci))
    out = {}
    for b in g.get_search_actions():
        h = g.clone(); h.apply_search_action(chess.Move.from_uci(b.uci()))
        if h.is_terminal():
            continue
        out[b.uci()] = dict(entry=dict(fen=h.fen(), half=bool(h.white_half_pending), turn_count=h.turn_count),
                            moves=first["moves"] + [b.uci()])
    return out


def write_book(path, labels, entries, copies, note):
    book = dict(schema_version=1, model=note, plies=0,
                entries=[entries[label] for label in labels for _ in range(copies)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book, indent=1))
    return book


def complete(report, games):
    if not report.exists():
        return False
    r = json.loads(report.read_text())
    return r.get("a_as_white", {}).get("games", 0) + r.get("a_as_black", {}).get("games", 0) == games


def run_pairings(phase_dir, book_path, n_entries, sims, pairings, seed_base):
    for i, (name, a, b) in enumerate(pairings):
        report, log = phase_dir / f"{name}.json", phase_dir / f"{name}.jsonl"
        games = 2 * n_entries
        if complete(report, games):
            print(f"CENSUS {phase_dir.name} {name}: done", flush=True)
            continue
        cmd = [sys.executable, "-u", "tools/match.py", "--model-a", MODELS[a], "--model-b", MODELS[b],
               "--games", str(games), "--sims", str(sims), "--sims-b", str(sims), "--engine", "native",
               "--workers", "8", "--seed", str(seed_base + i * 1_000_000), "--book", str(book_path),
               "--book-temp-plies", str(BOOK_TEMP_PLIES), "--stall-timeout", "1800",
               "--game-log", str(log), "--report-path", str(report), "--resume"]
        print(f"CENSUS {phase_dir.name} {name}: {games} games at {sims}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)


def aggregate(phase_dir, labels, copies, pairings):
    """Per position and (White player, Black player): W/D/L, White score, distinct games."""
    out = {label: {} for label in labels}
    for name, a, b in pairings:
        for line in open(phase_dir / f"{name}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if "game" not in r:
                continue
            label = labels[r["entry"] // copies]
            white, black = (a, b) if r["a_is_white"] else (b, a)
            cell = out[label].setdefault(f"{white}>{black}", dict(w=0, d=0, l=0, games=set()))
            ws = r["white_score"]
            cell["w" if ws == 1 else "l" if ws == 0 else "d"] += 1
            cell["games"].add(tuple(x for t in r["game"]["trajectory"] for x in t.get("action", [])))
    for label, cells in out.items():
        for cell in cells.values():
            n = cell["w"] + cell["d"] + cell["l"]
            cell.update(n=n, white_score=round((cell["w"] + 0.5 * cell["d"]) / n, 4) if n else None,
                        distinct=len(cell.pop("games")))
        scores = [c["white_score"] for c in cells.values() if c["white_score"] is not None]
        cells["mean_white_score"] = round(sum(scores) / len(scores), 4) if scores else None
        cells["min_white_score"] = min(scores) if scores else None
    return out


def summary_md(title, report):
    combos = sorted({k for cells in report.values() for k in cells if ">" in k})
    lines = [f"# {title}", "", "| Position | " + " | ".join(combos) + " | mean | min |",
             "|---|" + "---:|" * (len(combos) + 2)]
    for label, cells in sorted(report.items(), key=lambda kv: -(kv[1]["mean_white_score"] or 0)):
        row = [f"{cells[c]['w']}/{cells[c]['d']}/{cells[c]['l']} ({cells[c]['white_score']*100:.0f}%)"
               if c in cells else "-" for c in combos]
        lines.append(f"| {label} | " + " | ".join(row) +
                     f" | {cells['mean_white_score']*100:.1f}% | {cells['min_white_score']*100:.1f}% |")
    return "\n".join(lines) + "\n"


def phase(name, labels, entries, copies, sims, pairings, seed_base, note):
    phase_dir = OUT / name
    book = phase_dir / "book.json"
    write_book(book, labels, entries, copies, note)
    run_pairings(phase_dir, book, len(labels) * copies, sims, pairings, seed_base)
    report = aggregate(phase_dir, labels, copies, pairings)
    (phase_dir / "report.json").write_text(json.dumps(dict(labels=labels, copies=copies, sims=sims,
                                                           pairings=pairings, positions=report), indent=1))
    (phase_dir / "summary.md").write_text(summary_md(f"Opening census {name} ({sims} sims, {copies} copies)", report))
    print(f"CENSUS {name} COMPLETE", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehearsal", action="store_true", help="2 positions, 1 copy, 16 sims, into OUT/rehearsal")
    args = ap.parse_args()
    global OUT
    os.chdir(ROOT)
    turns = first_turns()
    labels = sorted(turns)
    entries = {k: v["entry"] for k, v in turns.items()}
    if args.rehearsal:
        OUT = OUT / "rehearsal"
        phase("phase1", labels[:2], entries, 1, 16, PAIRINGS, SEED + 90_000_000, "rehearsal")
        print("CENSUS REHEARSAL COMPLETE", flush=True)
        return
    assert len(labels) == 58, len(labels)
    p1 = phase("phase1", labels, entries, 10, 3200, PAIRINGS, SEED, "58 White first turns")
    ranked = sorted(labels, key=lambda k: -p1[k]["mean_white_score"])
    chosen = ranked[:12] + [k for k in MOST_PLAYED if k not in ranked[:12]]
    p2 = phase("phase2", chosen, entries, 6, 12800, PAIRINGS, SEED + 10_000_000, "phase-1 top 12 + most played")
    best = sorted(chosen, key=lambda k: (-p2[k]["mean_white_score"], -p1[k]["mean_white_score"]))[:6]
    for i, label in enumerate(best):
        replies = black_replies(turns[label])
        rlabels = sorted(replies)
        phase(f"phase3_{label.replace(' ', '_')}", rlabels, {k: v["entry"] for k, v in replies.items()}, 4, 3200,
              PAIRINGS[:1], SEED + 20_000_000 + i * 5_000_000, f"Black replies to {label}")
    print("CENSUS ALL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
