import argparse
import glob
import json
import os
import time


SIDE_AWARE_MODES = (
    "side_aware_strict",
    "side_aware_black_focus",
    "side_aware",
)


def _parse_float_list(s):
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def _load_gate_samples(runs_glob, limit):
    run_files = sorted(glob.glob(runs_glob))
    if limit is not None and limit > 0:
        run_files = run_files[-limit:]
    samples = []
    for run_path in run_files:
        try:
            run = json.load(open(run_path, "r", encoding="utf-8"))
        except Exception:
            continue
        for it in run.get("iterations", []):
            gate = it.get("gate", {})
            if not gate or gate.get("enabled") is not True:
                continue
            mode = gate.get("decision_mode")
            if mode not in (
                "side_aware_strict",
                "side_aware_black_focus",
                "side_aware",
                "overall_strict_sides",
                "overall",
            ):
                continue
            black_focus_required = bool(gate.get("black_focus_required", False))
            if mode == "side_aware_black_focus":
                black_focus_required = True
            if gate.get("black_focus_primary_score") is not None:
                black_focus_required = True
            samples.append({
                "run_path": run_path,
                "iteration": it.get("iteration"),
                "mode": mode,
                "overall": gate.get("score"),
                "primary": gate.get("primary_score"),
                "other": gate.get("other_score"),
                "white": gate.get("candidate_white", {}).get("score"),
                "black": gate.get("candidate_black", {}).get("score"),
                "total_games": gate.get("total_games", 0),
                "primary_games": gate.get("primary_games", 0),
                "other_games": gate.get("other_games", 0),
                "black_focus_required": black_focus_required,
                "black_focus_score": gate.get("black_focus_primary_score"),
                "black_focus_games": gate.get("black_focus_primary_games", 0),
                "actual_accepted": bool(it.get("accepted", False)),
            })
    return samples, run_files


def _passes(sample, gate_threshold, min_side_score, min_other_side, black_focus_threshold):
    mode = sample["mode"]
    white = sample["white"]
    black = sample["black"]
    if white is None or black is None:
        return False
    if mode in SIDE_AWARE_MODES:
        if (
            sample["total_games"] <= 0
            or sample["primary_games"] <= 0
            or sample["other_games"] <= 0
            or sample["primary"] is None
            or sample["other"] is None
        ):
            return False
        passed = (
            sample["primary"] >= gate_threshold
            and sample["other"] >= min_other_side
            and white >= min_side_score
            and black >= min_side_score
        )
        if sample["black_focus_required"]:
            bscore = sample["black_focus_score"]
            bgames = sample["black_focus_games"]
            passed = (
                passed
                and bscore is not None
                and bgames > 0
                and bscore >= black_focus_threshold
            )
        return passed
    if mode in ("overall_strict_sides", "overall"):
        if sample["overall"] is None or sample["total_games"] <= 0:
            return False
        return (
            sample["overall"] >= gate_threshold
            and white >= min_side_score
            and black >= min_side_score
        )
    return False


def _safe_div(a, b):
    if b <= 0:
        return 0.0
    return a / b


def _strictness_tuple(row):
    return (
        float(row["gate_threshold"]),
        float(row["min_side_score"]),
        float(row["min_other_side"]),
        float(row["black_focus_threshold"]),
    )


def _evaluate_config(samples, gate_threshold, min_side_score, min_other_side, black_focus_threshold):
    tp = fp = fn = tn = 0
    accepts = 0
    side_total = 0
    side_accepts = 0
    black_focus_required_total = 0
    black_focus_required_accepts = 0

    for s in samples:
        pred = _passes(
            s,
            gate_threshold=gate_threshold,
            min_side_score=min_side_score,
            min_other_side=min_other_side,
            black_focus_threshold=black_focus_threshold,
        )
        actual = bool(s["actual_accepted"])
        if pred:
            accepts += 1
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif (not pred) and actual:
            fn += 1
        else:
            tn += 1
        if s["mode"] in SIDE_AWARE_MODES:
            side_total += 1
            if pred:
                side_accepts += 1
        if s["black_focus_required"]:
            black_focus_required_total += 1
            if pred:
                black_focus_required_accepts += 1

    total = len(samples)
    acceptance_rate = _safe_div(accepts, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    mismatch = fp + fn
    mismatch_rate = _safe_div(mismatch, total)

    return {
        "gate_threshold": gate_threshold,
        "min_side_score": min_side_score,
        "min_other_side": min_other_side,
        "black_focus_threshold": black_focus_threshold,
        "accepted": accepts,
        "total": total,
        "acceptance_rate": acceptance_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mismatch_rate": mismatch_rate,
        "side_aware_acceptance_rate": _safe_div(side_accepts, side_total),
        "side_aware_total": side_total,
        "black_focus_required_acceptance_rate": _safe_div(
            black_focus_required_accepts,
            black_focus_required_total,
        ),
        "black_focus_required_total": black_focus_required_total,
    }


def _build_sensitivity(results, key):
    values = sorted({float(r[key]) for r in results})
    out = []
    for v in values:
        rows = [r for r in results if float(r[key]) == v]
        if not rows:
            continue
        out.append({
            "value": v,
            "configs": len(rows),
            "acceptance_rate_mean": sum(r["acceptance_rate"] for r in rows) / len(rows),
            "acceptance_rate_min": min(r["acceptance_rate"] for r in rows),
            "acceptance_rate_max": max(r["acceptance_rate"] for r in rows),
            "mismatch_rate_mean": sum(r["mismatch_rate"] for r in rows) / len(rows),
            "f1_mean": sum(r["f1"] for r in rows) / len(rows),
        })
    return out


def _recommend(
    results,
    target_accept_rate,
    recommend_k,
    min_precision,
    min_recall,
    max_accept_rate,
    prefer_strict,
):
    candidates = []
    for row in results:
        if row["precision"] < min_precision:
            continue
        if row["recall"] < min_recall:
            continue
        if max_accept_rate is not None and row["acceptance_rate"] > max_accept_rate:
            continue
        objective = (
            abs(row["acceptance_rate"] - target_accept_rate)
            + 0.75 * row["mismatch_rate"]
            + 0.25 * (1.0 - row["f1"])
        )
        cand = dict(row)
        cand["objective"] = objective
        candidates.append(cand)

    def _sort_key(row):
        strict = _strictness_tuple(row)
        if prefer_strict:
            strict_part = tuple(-v for v in strict)
        else:
            strict_part = strict
        return (row["objective"], row["mismatch_rate"], strict_part)

    candidates.sort(key=_sort_key)
    return candidates[:max(1, recommend_k)], candidates


def main():
    parser = argparse.ArgumentParser(
        description="Sweep gate thresholds/floors against iterate run metadata"
    )
    parser.add_argument("--runs-glob", type=str, default=os.path.join("models", "iterate_run_*.json"))
    parser.add_argument("--limit", type=int, default=80,
                        help="Use only the most recent N run files (<=0 means all)")
    parser.add_argument("--thresholds", type=str, default="0.30,0.40,0.50,0.54,0.58")
    parser.add_argument("--min-side-scores", type=str, default="0.00,0.10,0.20,0.30,0.40,0.45")
    parser.add_argument("--min-other-sides", type=str, default="0.00,0.10,0.20,0.30,0.38,0.42,0.45")
    parser.add_argument("--black-focus-thresholds", type=str, default="0.00,0.20,0.35,0.40,0.45")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--recommend-k", type=int, default=5)
    parser.add_argument("--target-accept-rate", type=float, default=None,
                        help="Target predicted acceptance rate (default: match actual acceptance rate)")
    parser.add_argument("--min-precision", type=float, default=0.0,
                        help="Minimum precision constraint for recommendations")
    parser.add_argument("--min-recall", type=float, default=0.0,
                        help="Minimum recall constraint for recommendations")
    parser.add_argument("--max-accept-rate", type=float, default=None,
                        help="Maximum predicted acceptance rate constraint for recommendations")
    parser.add_argument("--prefer-strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Tie-break recommendations toward stricter thresholds/floors")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    thresholds = _parse_float_list(args.thresholds)
    min_side_scores = _parse_float_list(args.min_side_scores)
    min_other_sides = _parse_float_list(args.min_other_sides)
    black_focus_thresholds = _parse_float_list(args.black_focus_thresholds)

    limit = args.limit if args.limit and args.limit > 0 else None
    samples, run_files = _load_gate_samples(args.runs_glob, limit=limit)
    if not samples:
        print("No gate-enabled samples found.")
        return

    actual_accepts = sum(1 for s in samples if s["actual_accepted"])
    total = len(samples)
    print(f"Loaded {total} gate samples from {len(run_files)} run files")
    print(f"Actual acceptance rate: {actual_accepts}/{total} = {actual_accepts/total:.3f}")
    target_accept_rate = (
        args.target_accept_rate
        if args.target_accept_rate is not None
        else (actual_accepts / total)
    )
    print(f"Target acceptance rate: {target_accept_rate:.3f}")

    results = []
    for th in thresholds:
        for ms in min_side_scores:
            for mo in min_other_sides:
                for bft in black_focus_thresholds:
                    row = _evaluate_config(
                        samples=samples,
                        gate_threshold=th,
                        min_side_score=ms,
                        min_other_side=mo,
                        black_focus_threshold=bft,
                    )
                    results.append(row)

    top_acceptance = sorted(
        results,
        key=lambda r: (r["acceptance_rate"], r["accepted"]),
        reverse=True,
    )[: args.top_k]
    recommended, recommendation_pool = _recommend(
        results=results,
        target_accept_rate=target_accept_rate,
        recommend_k=args.recommend_k,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_accept_rate=args.max_accept_rate,
        prefer_strict=bool(args.prefer_strict),
    )
    sensitivity = {
        "gate_threshold": _build_sensitivity(results, "gate_threshold"),
        "min_side_score": _build_sensitivity(results, "min_side_score"),
        "min_other_side": _build_sensitivity(results, "min_other_side"),
        "black_focus_threshold": _build_sensitivity(results, "black_focus_threshold"),
    }

    print("\nRecommended defaults:")
    for row in recommended:
        print(
            "  "
            f"th={row['gate_threshold']:.2f} "
            f"min_side={row['min_side_score']:.2f} "
            f"min_other={row['min_other_side']:.2f} "
            f"bf_th={row['black_focus_threshold']:.2f} "
            f"acc={row['acceptance_rate']:.3f} "
            f"prec={row['precision']:.3f} "
            f"rec={row['recall']:.3f} "
            f"f1={row['f1']:.3f} "
            f"mismatch={row['mismatch_rate']:.3f}"
        )

    print("\nTop by acceptance rate:")
    for row in top_acceptance:
        print(
            "  "
            f"th={row['gate_threshold']:.2f} "
            f"min_side={row['min_side_score']:.2f} "
            f"min_other={row['min_other_side']:.2f} "
            f"bf_th={row['black_focus_threshold']:.2f} "
            f"-> {row['accepted']}/{row['total']} ({row['acceptance_rate']:.3f}) "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f}"
        )

    output = args.output
    if not output:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = os.path.join("models", f"gate_sweep_{ts}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_glob": args.runs_glob,
        "run_files_used": run_files,
        "samples": total,
        "actual_acceptance_rate": actual_accepts / total,
        "actual_acceptance": {"accepted": actual_accepts, "total": total},
        "target_acceptance_rate": target_accept_rate,
        "recommendation_constraints": {
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "max_accept_rate": args.max_accept_rate,
            "prefer_strict": bool(args.prefer_strict),
        },
        "grid": {
            "thresholds": thresholds,
            "min_side_scores": min_side_scores,
            "min_other_sides": min_other_sides,
            "black_focus_thresholds": black_focus_thresholds,
        },
        "recommended": recommended,
        "top_acceptance": top_acceptance,
        "sensitivity": sensitivity,
        "recommendation_pool_size": len(recommendation_pool),
        "top": top_acceptance,
        "all_results": results,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved sweep results: {output}")


if __name__ == "__main__":
    main()
