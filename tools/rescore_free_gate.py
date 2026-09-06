"""Audit a legacy free gate without replacing its original report or logs."""
import argparse
import json
from pathlib import Path

from gate_free import ROOT
from free_gate_stats import SCORING_VERSION, leg_stats, opening_key, verdict
from match_evidence import atomic_json, file_hash, read_rows


def rescore(report_path, logs, output, target=100, par_target=100):
    report_path, logs, output = Path(report_path), Path(logs), Path(output)
    if output.exists():
        raise FileExistsError(output)
    original = json.loads(report_path.read_text(encoding="utf-8"))
    sources = {str(report_path.resolve()): file_hash(report_path)}

    def read_prefix(prefix):
        paths = sorted(logs.glob(prefix + "_b*.lines.jsonl"),
                       key=lambda p: int(p.name.split("_b")[-1].split(".")[0]))
        if not paths:
            raise ValueError(f"no logs for {prefix}")
        for path in paths:
            sources[str(path.resolve())] = file_hash(path)
        return [row for path in paths for row in read_rows(path)]

    par_rows = read_prefix("par_" + original["bar_name"])
    first = read_prefix("vs_bar")
    confirm = read_prefix("vs_bar_confirm")
    par = leg_stats(par_rows)
    legs = {"vs_bar": leg_stats(first),
            "vs_bar_confirm": leg_stats(confirm, {opening_key(r) for r in first})}
    out = {"instrument": SCORING_VERSION, "historical_rescore_only": True,
           "original_verdict": original.get("verdict"), "source_hashes": sources,
           "target_per_side": target, "par_per_side": par_target,
           "bar_free_par": par, "legs": legs,
           "combined_h2h": leg_stats(first + confirm),
           "limitations": ["Legacy records lack per-game seeds, trajectories and engine hashes.",
                            "Unseen confirmation is a conditional subset, not a natural win rate.",
                            "Rescoring cannot retroactively supply a preregistered protocol."],
           "coverage_reassessment": verdict(par, legs, target, par_target),
           "eligible": False}
    atomic_json(output, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = rescore(args.report, args.logs, args.output)
    print(json.dumps({"reassessment": result["coverage_reassessment"],
                      "combined": result["combined_h2h"]["unique"]}, indent=2))
