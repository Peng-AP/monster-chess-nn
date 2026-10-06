"""Read-only, nonbinding sampled-score and continuation audit of a free campaign.

Writes a separate snapshot, never changes the source protocol or its verdict.
Bare-king outcomes are observed conversions, not assertions of forced wins.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from free_gate_stats import leg_stats, opening_key
from gate_free import load_json
from match_evidence import atomic_json, file_hash, read_rows
from sampled_gate_stats import compare, self_par


def conversion_audit(rows):
    counts = Counter()
    remaining = []
    for row in rows:
        frames = row.get("game", {}).get("trajectory", [])
        first = next((f for f in frames if "K" in f["fen"].split()[0]
                      and not any(c in f["fen"].split()[0] for c in "PNBRQ")), None)
        if first is None:
            continue
        value = row["result_for_a"] if row["a_is_white"] else -row["result_for_a"]
        counts["black_king_capture" if value == -1 else "white_king_capture" if value == 1 else "draw"] += 1
        remaining.append(row["plies"] - first["plies_reached"])
    return {"games_reaching_bare_white_king": sum(counts.values()),
            "subsequent_outcomes": dict(counts),
            "mean_remaining_search_plies": sum(remaining) / len(remaining) if remaining else None,
            "interpretation": "observed outcomes; material does not prove a forced win"}


def snapshot(args):
    directory = Path(args.campaign).resolve()
    binding = directory / "binding"
    source_report = load_json(binding / "report.json")
    out = {"source_campaign": str(directory), "binding_verdict_unchanged": source_report["verdict"],
           "binding_instrument": source_report["instrument"],
           "interpretation": "retrospective descriptive audit; NOT a preregistered v3 gate",
           "legs": {}, "source_hashes": {str(binding / "report.json"): file_hash(binding / "report.json")}}
    first, combined, par = [], [], None
    for name in ("par", "vs_bar", "vs_bar_confirm", "gen41", "v24", "selfplay"):
        parent = binding if name in ("par", "vs_bar", "vs_bar_confirm") else directory
        progress = parent / (name + ".json")
        if not progress.exists():
            continue
        state = load_json(progress)
        files = [parent / b["log"] for b in state["batches"] if b["complete"]]
        # Only fully scheduled/completed batches count in a live descriptive
        # snapshot. The final result includes all batches, including slow games.
        rows = [r for p in files for r in read_rows(p)]
        if not rows:
            continue
        out["source_hashes"].update({str(p): file_hash(p) for p in files})
        stats = leg_stats(rows, {opening_key(r) for r in first})
        record = {"stats": stats, "active_minutes": state["active_seconds"] / 60,
                  "stop_reason": state.get("stop_reason"),
                  "endpoint_target_reached": state.get("complete", False),
                  "bare_white_king": conversion_audit(rows),
                  "candidate_as_black_bare_king": conversion_audit([r for r in rows if not r["a_is_white"]])}
        if name == "par":
            par = self_par(rows)
            record["actual_color_par"] = par
        elif name in ("vs_bar", "vs_bar_confirm"):
            record["sampled_actual_color_par_comparisons"] = compare(stats["sampled"], par)
            combined.extend(rows)
            if name == "vs_bar":
                first = rows
        if name == "selfplay":
            record["actual_color_par"] = self_par(rows)
        endpoints = Counter(opening_key(r) for r in rows)
        record["most_frequent_endpoints"] = [{"candidate_white": k[0], "fen": k[1],
            "half": k[2], "turn_count": k[3], "count": n} for k, n in endpoints.most_common(10)]
        out["legs"][name] = record
    if combined:
        out["combined_h2h"] = leg_stats(combined)
        out["combined_sampled_par_comparisons"] = compare(out["combined_h2h"]["sampled"], par)
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError("use a fresh snapshot path; audit does not replace previous evidence")
    atomic_json(destination, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = snapshot(args)
    for name, leg in out["legs"].items():
        stats = leg.get("actual_color_par", leg["stats"]["sampled"])
        print(f"{name}: n={stats['n']} W={stats['sides']['white']['score']:.4f} "
              f"B={stats['sides']['black']['score']:.4f} total={stats['score']:.4f}")


if __name__ == "__main__":
    main()
