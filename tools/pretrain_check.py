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
  3. Composition diff      — per-material-phase Black-win-label share among
     value-contributing records vs a reference corpus (the incumbent's); big
     shifts are flagged before they become mystery regressions.
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

from config import (
    VALUE_TARGET_DISCOUNT_MODE,
    VALUE_TARGET_FLOOR,
    VALUE_TARGET_HORIZON,
)

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
    """phase -> (value weight, Black-win value weight).

    A policy-only source cannot alter the value head, so its outcome labels do
    not belong in a value-label composition audit. Fractional value teachers
    contribute in proportion to the same weight the trainer will apply.
    """
    mix = defaultdict(lambda: [0, 0])
    for _rel, records in _iter_games(raw_dir):
        for rec in records:
            fen = rec.get("fen")
            weight = float(rec.get("value_weight", 1.0))
            if not fen or weight <= 0:
                continue
            cell = mix[_phase(_bm(fen))]
            cell[0] += weight
            if rec.get("game_result", 0) < 0:
                cell[1] += weight
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
                    help="max relative gap in mean |target| White-won vs "
                         "Black-won, or max additional gap vs --reference")
    ap.add_argument("--value-discount-mode", choices=["near_mate", "progress"],
                    default=VALUE_TARGET_DISCOUNT_MODE)
    ap.add_argument("--value-horizon", type=int, default=VALUE_TARGET_HORIZON)
    ap.add_argument("--value-floor", type=float, default=VALUE_TARGET_FLOOR)
    args = ap.parse_args()

    failures, warnings = [], []

    # Both rejected v18 candidates shipped through a driver that weakened
    # these thresholds and then failed in exactly the gated ways (v13 label
    # bias, v14 focus pollution). Weakening is allowed but never silent.
    weakened = []
    if args.min_purity < ap.get_default("min_purity"):
        weakened.append(f"min-purity {args.min_purity} < default "
                        f"{ap.get_default('min_purity')}")
    if args.max_dup > ap.get_default("max_dup"):
        weakened.append(f"max-dup {args.max_dup} > default {ap.get_default('max_dup')}")
    if args.diff_fail > ap.get_default("diff_fail"):
        weakened.append(f"diff-fail {args.diff_fail} > default "
                        f"{ap.get_default('diff_fail')}")
    if args.bias_fail > ap.get_default("bias_fail"):
        weakened.append(f"bias-fail {args.bias_fail} > default "
                        f"{ap.get_default('bias_fail')}")
    if weakened:
        print("!" * 72)
        print("!!! WEAKENED GATE: this run does NOT enforce the default corpus")
        print("!!! failure gates. Every past weakening preceded a rejected model.")
        for item in weakened:
            print(f"!!!   {item}")
        print("!" * 72)

    # Pass over the corpus once, bucketing by top-level source dir.
    src_games = defaultdict(lambda: [0, 0, 0])     # games, black wins, white wins
    src_positions = defaultdict(int)
    total_positions = 0
    from data_processor import _discounted_results
    from promotion_data import is_successful_white_runner_prevention
    tgt_sum = {"white": 0.0, "black": 0.0}
    tgt_n = {"white": 0, "black": 0}
    promo_records = 0
    promo_black_runner_records = 0
    promo_black_positions_without_weight = 0
    promo_policy_weight_mismatches = 0

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
        if top == "promo_races":
            successful_prevention = is_successful_white_runner_prevention(records)
            expected_black_weight = 1.0 if successful_prevention else 0.0
            for rec in records:
                promo_records += 1
                if rec.get("start_source") == "promo_black_runner":
                    promo_black_runner_records += 1
                if (rec.get("current_player") == "black"
                        and "policy_weight" not in rec):
                    promo_black_positions_without_weight += 1
                elif rec.get("current_player") == "black":
                    actual = float(rec.get("policy_weight", 1.0))
                    if abs(actual - expected_black_weight) > 1e-9:
                        promo_policy_weight_mismatches += 1
        # label-transform preview (current processor settings). Normalized by
        # the raw label so ONLY the transform is measured — the +-0.5
        # move-limit labels are a deliberate choice, not a transform artifact,
        # and they skew Black-side by design.
        if result != 0:
            side = "black" if result < 0 else "white"
            transformed = _discounted_results(
                records,
                horizon=args.value_horizon,
                floor=args.value_floor,
                mode=args.value_discount_mode,
            )
            for t, rec in zip(transformed, records):
                raw = rec.get("game_result", 0)
                value_weight = float(rec.get("value_weight", 1.0))
                if raw and value_weight > 0:
                    tgt_sum[side] += value_weight * abs(t) / abs(raw)
                    tgt_n[side] += value_weight

    print(f"=== pretrain_check: {args.merged_dir} ===")
    print(f"total positions: {total_positions}, sources: {dict(src_positions)}")

    if promo_records:
        if promo_black_runner_records:
            failures.append(
                f"generated Black-runner contamination: "
                f"{promo_black_runner_records}/{promo_records} promo records"
            )
        else:
            print(f"  OK  promo provenance: {promo_records} White-runner records, "
                  "0 generated Black-runner records")
        if promo_black_positions_without_weight:
            failures.append(
                f"promo policy masking absent on "
                f"{promo_black_positions_without_weight} Black positions"
            )
        else:
            print("  OK  promo policy weights explicit on every Black position")
        if promo_policy_weight_mismatches:
            failures.append(
                f"promo policy-weight mismatch on "
                f"{promo_policy_weight_mismatches} Black positions"
            )
        else:
            print("  OK  promo policy weights match prevention outcomes")

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

    # 3. value-label composition diff vs reference. Policy-only teachers are
    # intentionally absent: their labels cannot reach the value loss.
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

    # 4. label-transform side bias over records that reach the value loss.
    # With a reference, the frozen incumbent recipe is the zero point: an
    # incremental corpus must not add more than --bias-fail. Without one, the
    # absolute guard remains in force. This distinction matters for the
    # approved r50h60 recipe, whose incumbent corpus has a 15.5% raw side gap;
    # an absolute-only check would reject every candidate including the
    # unchanged control and could not detect what the new corpus introduced.
    if tgt_n["white"] and tgt_n["black"]:
        mw = tgt_sum["white"] / tgt_n["white"]
        mb = tgt_sum["black"] / tgt_n["black"]
        rel_gap = abs(mw - mb) / max(mw, mb)
        line = (f"label bias: mean |target| white-won {mw:.3f} vs black-won {mb:.3f} "
                f"(rel gap {rel_gap:.1%}, max {args.bias_fail:.0%})")
        introduced_gap = rel_gap
        if args.reference and os.path.isdir(args.reference):
            ref_sum = {"white": 0.0, "black": 0.0}
            ref_n = {"white": 0.0, "black": 0.0}
            for _rel, records in _iter_games(args.reference):
                result = records[-1].get("game_result", 0)
                if result == 0:
                    continue
                side = "black" if result < 0 else "white"
                transformed = _discounted_results(
                    records,
                    horizon=args.value_horizon,
                    floor=args.value_floor,
                    mode=args.value_discount_mode,
                )
                for target, rec in zip(transformed, records):
                    raw = rec.get("game_result", 0)
                    weight = float(rec.get("value_weight", 1.0))
                    if raw and weight > 0:
                        ref_sum[side] += weight * abs(target) / abs(raw)
                        ref_n[side] += weight
            if ref_n["white"] and ref_n["black"]:
                ref_w = ref_sum["white"] / ref_n["white"]
                ref_b = ref_sum["black"] / ref_n["black"]
                ref_gap = abs(ref_w - ref_b) / max(ref_w, ref_b)
                introduced_gap = rel_gap - ref_gap
                line = (line[:-1] + f"; reference {ref_gap:.1%}, "
                        f"delta {introduced_gap:+.1%})")
        if introduced_gap > args.bias_fail:
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
