"""Pre-training corpus gate: fail fast on the failure modes that cost nights.

Run on a merged corpus BEFORE process/train. Exits nonzero on any FAIL so
overnight drivers abort the chain instead of training on a bad batch.

Checks (each one is a failure we actually shipped once):
  1. Amplification purity  — *_blackfocus subdirs must be >= --min-purity
     Black wins; *whitefocus* subdirs the same for White (v14: blackfocus
     pollution 83%->73% silently regressed Black).
  2. Human duplication     — human games are the highest-quality source and
     their UNIQUE share is uncapped; what fails is the in-file duplication
     multiple inflating a small set into memorization (--max-dup).
  3. Composition diff      — per-material-phase Black-win-label share vs a
     reference corpus (the incumbent's); big shifts are flagged before they
     become mystery regressions.
  4. Label-transform bias  — applies data_processor._discounted_results and
     compares mean |target| for White-won vs Black-won positions (v13: a
     global discount taxed Black's long wins 2x White's).

    py -3 tools/pretrain_check.py data/raw/combined_v12 --reference data/raw/combined_v10
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PHASES = [(13, 15, "open"), (8, 12, "mid"), (4, 7, "late"), (0, 3, "endg")]


def _bm(fen):
    board = fen.split()[0]
    return sum(1 for c in board if c.islower() and c != "k")


def _phase(black_men):
    for lo, hi, name in PHASES:
        if lo <= black_men <= hi:
            return name
    return "?"


def _iter_games(raw_dir):
    for dirpath, _dirs, files in os.walk(raw_dir):
        rel = os.path.relpath(dirpath, raw_dir).replace("\\", "/")
        for fname in sorted(files):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, fname)
            with open(path, encoding="utf-8") as f:
                records = [json.loads(ln) for ln in f if ln.strip()]
            if records:
                yield rel if rel != "." else "", records


def _phase_mix(raw_dir):
    """phase -> (positions, black_win_labeled)."""
    mix = defaultdict(lambda: [0, 0])
    for _rel, records in _iter_games(raw_dir):
        for rec in records:
            fen = rec.get("fen")
            if not fen:
                continue
            cell = mix[_phase(_bm(fen))]
            cell[0] += 1
            if rec.get("game_result", 0) < 0:
                cell[1] += 1
    return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_dir")
    ap.add_argument("--reference", default=None,
                    help="incumbent's merged corpus for composition diff")
    ap.add_argument("--min-purity", type=float, default=0.85)
    ap.add_argument("--human-min", type=float, default=0.03)
    ap.add_argument("--max-dup", type=float, default=8.0,
                    help="max effective duplication multiple for human positions")
    ap.add_argument("--diff-warn", type=float, default=0.08)
    ap.add_argument("--diff-fail", type=float, default=0.15)
    ap.add_argument("--bias-fail", type=float, default=0.05,
                    help="max relative gap in mean |target| White-won vs Black-won")
    args = ap.parse_args()

    failures, warnings = [], []

    # Pass over the corpus once, bucketing by top-level source dir.
    src_games = defaultdict(lambda: [0, 0, 0])     # games, black wins, white wins
    src_positions = defaultdict(int)
    total_positions = 0
    from data_processor import _discounted_results
    tgt_sum = {"white": 0.0, "black": 0.0}
    tgt_n = {"white": 0, "black": 0}

    for rel, records in _iter_games(args.merged_dir):
        top = rel.split("/", 1)[0] if rel else "(root)"
        result = records[-1].get("game_result", 0)
        g = src_games[top]
        g[0] += 1
        if result < 0:
            g[1] += 1
        elif result > 0:
            g[2] += 1
        src_positions[top] += len(records)
        total_positions += len(records)
        # label-transform preview (current processor settings). Normalized by
        # the raw label so ONLY the transform is measured — the +-0.5
        # move-limit labels are a deliberate choice, not a transform artifact,
        # and they skew Black-side by design.
        if result != 0:
            side = "black" if result < 0 else "white"
            for t, rec in zip(_discounted_results(records), records):
                raw = rec.get("game_result", 0)
                if raw:
                    tgt_sum[side] += abs(t) / abs(raw)
                    tgt_n[side] += 1

    print(f"=== pretrain_check: {args.merged_dir} ===")
    print(f"total positions: {total_positions}, sources: {dict(src_positions)}")

    # 1. amplification purity
    for top, (n, bw, ww) in sorted(src_games.items()):
        low = top.lower()
        want = None
        if "blackfocus" in low:
            want, wins = "black", bw
        elif "whitefocus" in low:
            want, wins = "white", ww
        if want and n:
            purity = wins / n
            line = f"purity {top}: {wins}/{n} = {purity:.0%} ({want} wins, min {args.min_purity:.0%})"
            if purity < args.min_purity:
                failures.append(line)
            else:
                print("  OK  " + line)

    # 2. human data: quality is not the risk — the duplication multiple is.
    # Unique human positions can be any share (highest-quality source in the
    # corpus, owner 2026-07-11); FAIL only when in-file duplication inflates a
    # small set beyond --max-dup copies.
    human_pos = 0
    human_unique = set()
    for rel, records in _iter_games(args.merged_dir):
        top = rel.split("/", 1)[0] if rel else "(root)"
        if not top.startswith("human_games"):
            continue
        human_pos += len(records)
        for rec in records:
            if rec.get("fen"):
                human_unique.add(" ".join(rec["fen"].split()[:2]))
    share = human_pos / total_positions if total_positions else 0.0
    dup = human_pos / len(human_unique) if human_unique else 0.0
    line = (f"human data: {human_pos} positions ({share:.1%} of corpus), "
            f"{len(human_unique)} unique, effective duplication x{dup:.1f} "
            f"(max x{args.max_dup:.0f})")
    if dup > args.max_dup:
        failures.append(line)
    elif share < args.human_min:
        warnings.append(line + " — thin human signal")
    else:
        print("  OK  " + line)

    # 3. composition diff vs reference
    if args.reference and os.path.isdir(args.reference):
        cur = _phase_mix(args.merged_dir)
        ref = _phase_mix(args.reference)
        for _lo, _hi, name in PHASES:
            cn, cb = cur[name]
            rn, rb = ref[name]
            if not cn or not rn:
                continue
            d = cb / cn - rb / rn
            line = (f"phase {name}: B-win share {cb/cn:.0%} vs ref {rb/rn:.0%} "
                    f"(delta {d:+.0%})")
            if abs(d) >= args.diff_fail:
                failures.append(line)
            elif abs(d) >= args.diff_warn:
                warnings.append(line)
            else:
                print("  OK  " + line)

    # 4. label-transform side bias
    if tgt_n["white"] and tgt_n["black"]:
        mw = tgt_sum["white"] / tgt_n["white"]
        mb = tgt_sum["black"] / tgt_n["black"]
        rel_gap = abs(mw - mb) / max(mw, mb)
        line = (f"label bias: mean |target| white-won {mw:.3f} vs black-won {mb:.3f} "
                f"(rel gap {rel_gap:.1%}, max {args.bias_fail:.0%})")
        if rel_gap > args.bias_fail:
            failures.append(line)
        else:
            print("  OK  " + line)

    for w in warnings:
        print("  WARN " + w)
    for f in failures:
        print("  FAIL " + f)
    if failures:
        print("PRETRAIN CHECK: FAIL — do not train on this corpus.")
        sys.exit(1)
    print("PRETRAIN CHECK: PASS" + (" (with warnings)" if warnings else ""))


if __name__ == "__main__":
    main()
