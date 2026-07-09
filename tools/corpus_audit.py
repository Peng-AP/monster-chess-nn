"""Corpus composition audit: label mix by game phase, per source.

Answers, BEFORE burning a training night: "what does this corpus teach
about each phase of the game?"  The v10 failure mode (Black opening play
got worse) was visible in composition: opening-phase positions were
overwhelmingly White-win-labeled, so the value head learned "opening =
lost for Black" regardless of Black's moves.

Phase = MATERIAL (Black non-king piece count), not the FEN fullmove
counter and not record index: constructed demo/blackfocus starts reset
the fullmove clock to 1 and start mid-game, so both clock and index
misclassify endgames as "opening".  Material is what the net sees.

Usage:
    python tools/corpus_audit.py data/raw/combined_v8 [more dirs ...]
"""
import json
import os
import sys
from collections import defaultdict

# Phase buckets by Black non-king piece count (full army = 15).
PHASE_BUCKETS = [(13, 15, "black 13-15 (open)"), (8, 12, "black 8-12 (mid)"),
                 (4, 7, "black 4-7 (late)"), (0, 3, "black 0-3 (endg)")]


def _phase(black_men):
    for lo, hi, name in PHASE_BUCKETS:
        if lo <= black_men <= hi:
            return name
    return "?"


def _black_men(fen):
    """Black non-king piece count from the FEN board field."""
    board = fen.split()[0]
    return sum(1 for c in board if c.islower() and c != 'k')


def _source_of(rel):
    rel = rel.replace("\\", "/")
    if "/" not in rel:
        return "(root)"
    return rel.rsplit("/", 1)[0]


def audit_dir(raw_dir):
    # stats[source][phase] = [n_positions, n_blackwin_label, n_whitewin_label]
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    game_stats = defaultdict(lambda: [0, 0, 0])   # per source: games, bwin, wwin
    start_fullmoves = defaultdict(list)

    for dirpath, _dirs, files in os.walk(raw_dir):
        for fname in sorted(files):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, fname)
            rel = os.path.relpath(path, raw_dir)
            source = _source_of(rel)
            with open(path) as f:
                records = [json.loads(ln) for ln in f if ln.strip()]
            if not records:
                continue
            result = records[-1].get("game_result", 0)
            gs = game_stats[source]
            gs[0] += 1
            if result < 0:
                gs[1] += 1
            elif result > 0:
                gs[2] += 1
            try:
                start_fullmoves[source].append(_black_men(records[0]["fen"]))
            except (KeyError, IndexError):
                pass
            for rec in records:
                try:
                    fm = _black_men(rec["fen"])
                except (KeyError, IndexError):
                    continue
                cell = stats[source][_phase(fm)]
                cell[0] += 1
                gr = rec.get("game_result", 0)
                if gr < 0:
                    cell[1] += 1
                elif gr > 0:
                    cell[2] += 1

    phase_names = [b[2] for b in PHASE_BUCKETS]
    print(f"\n=== {raw_dir} ===")
    header = f"{'source':<18}{'games':>6}{'B-win':>7}{'W-win':>7}{'start_bm':>9}  " + \
             "".join(f"{p:>22}" for p in phase_names)
    print(header)
    print(f"{'':<47}  " + "".join(f"{'pos (%Bwin-labeled)':>22}" for _ in phase_names))

    totals = defaultdict(lambda: [0, 0, 0])
    for source in sorted(stats):
        g = game_stats[source]
        sf = start_fullmoves[source]
        mean_sf = sum(sf) / len(sf) if sf else 0
        row = f"{source:<18}{g[0]:>6}{g[1]:>7}{g[2]:>7}{mean_sf:>9.1f}  "
        for p in phase_names:
            n, bw, ww = stats[source][p]
            tot = totals[p]
            tot[0] += n
            tot[1] += bw
            tot[2] += ww
            row += f"{n:>13} ({(100 * bw / n if n else 0):>4.0f}%)  " if n else f"{'-':>22}"
        print(row)

    row = f"{'TOTAL':<18}{sum(g[0] for g in game_stats.values()):>6}" \
          f"{sum(g[1] for g in game_stats.values()):>7}" \
          f"{sum(g[2] for g in game_stats.values()):>7}{'':>9}  "
    for p in phase_names:
        n, bw, _ww = totals[p]
        row += f"{n:>13} ({(100 * bw / n if n else 0):>4.0f}%)  " if n else f"{'-':>22}"
    print(row)


if __name__ == "__main__":
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        sys.exit(1)
    for d in dirs:
        audit_dir(d)
