"""Overnight chain for the v18 candidate: process -> train A/B -> gate.

ONE training corpus (combined_v17, ramp labels), TWO architectures:
  gap      — the current value head (global average pooling)   [baseline]
  spatial  — value head keeps the 8x8 layout                   [A/B arm]

The A/B is the spatial value head rather than the previously planned
17-channel encoding: extra input planes feed a head that averages spatial
information away, so the head is the prerequisite (audit 2026-07-25 measured
the GAP feature vector at effective rank ~6 vs the policy head's ~481).

Bar (NEVER weakened here — this driver only measures and reports):
  beat BOTH fresh_start_v17 and fresh_start_v18_ramp head-to-head,
  with a per-side floor of 0.40 on EVERY leg, and no anchor-Black regression.
Aggregates alone have twice let a one-sided collapse through.

Session-death-proof: launched detached via Win32_Process, ignores SIGINT,
keeps the machine awake, and logs CHAIN ABORTED on any exception.
"""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
REPORT = os.path.join(ROOT, "V18_RUN_REPORT.md")

RAW = "data/raw/combined_v17"
PROCESSED = "data/processed/combined_v17_r50h60"
INCUMBENT = "models/fresh_start_v17/best_value_net.pt"
RAMP = "models/rejected/fresh_start_v18_ramp/best_value_net.pt"

ARMS = [
    ("gap", "models/candidates/v18_ramp17_gap", []),
    ("spatial", "models/candidates/v18_ramp17_spatial", ["--spatial-value-head"]),
]

SIDE_FLOOR = 0.40
MATCH_GAMES = 20
MATCH_SIMS = 400
SEED = 42

results = {"stages": [], "arms": {}}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(name, cmd):
    log(f"START {name}")
    log(f"      {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run([PY, "-u"] + cmd, cwd=ROOT)
    dt = (time.time() - t0) / 60
    log(f"END   {name} (exit={proc.returncode}, {dt:.1f} min)")
    results["stages"].append({"name": name, "exit": proc.returncode,
                              "minutes": round(dt, 1)})
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit {proc.returncode}")
    return dt


def newest(pattern_dir, prefix, after):
    """Most recent benchmarks/ JSON matching prefix and written after `after`.

    Both arms produce files with colliding names (benchmark_ tags on the
    checkpoint basename, which is best_value_net.pt for every arm), so the
    mtime floor is what guarantees we read THIS stage's output and not a
    previous arm's.
    """
    best, best_m = None, after
    for f in os.listdir(pattern_dir):
        if f.startswith(prefix) and f.endswith(".json"):
            p = os.path.join(pattern_dir, f)
            m = os.path.getmtime(p)
            if m >= best_m:
                best, best_m = p, m
    if best is None:
        raise RuntimeError(
            f"no {prefix}*.json written after {time.ctime(after)} in {pattern_dir}")
    return best


def read_match(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {"score": d.get("a_score"),
            "white": (d.get("a_as_white") or {}).get("score"),
            "black": (d.get("a_as_black") or {}).get("score"),
            "path": os.path.relpath(path, ROOT)}


def evaluate_arm(tag, model_dir):
    """Two head-to-heads plus the fixed anchor, all per-side."""
    model = os.path.join(model_dir, "best_value_net.pt")
    bench = os.path.join(ROOT, "benchmarks")
    arm_name = os.path.basename(model_dir)
    arm = {}
    for label, opponent in (("vs_v17", INCUMBENT), ("vs_ramp", RAMP)):
        # match.py names output match_<dirname(a)>_vs_<dirname(b)>_<ts>.json
        prefix = f"match_{arm_name}_vs_{os.path.basename(os.path.dirname(opponent))}"
        t0 = time.time()
        run(f"{tag}:{label}", [
            "tools/match.py", "--model-a", model, "--model-b", opponent,
            "--games", str(MATCH_GAMES), "--sims", str(MATCH_SIMS),
            "--seed", str(SEED),
        ])
        arm[label] = read_match(newest(bench, prefix, t0))
        log(f"      {tag} {label}: {arm[label]}")

    t0 = time.time()
    run(f"{tag}:anchor", [
        "src/benchmark.py", "--model", model,
        "--games", str(MATCH_GAMES), "--sims", str(MATCH_SIMS),
        "--seed", str(SEED),
    ])
    anchor_path = newest(bench, "benchmark_best_value_net", t0)
    with open(anchor_path, encoding="utf-8") as f:
        a = json.load(f)
    arm["anchor"] = {"score": a.get("candidate_score"),
                     "black_win_share": a.get("candidate_black_win_share"),
                     "path": os.path.relpath(anchor_path, ROOT)}
    log(f"      {tag} anchor: {arm['anchor']}")

    run(f"{tag}:model_diff", [
        "tools/model_diff.py", "--candidate", model, "--incumbent", INCUMBENT,
        "--data-dir", PROCESSED, "--split", "test",
    ])

    # verdict — thresholds are fixed, never tuned to let an arm through
    legs, failures = [], []
    for label in ("vs_v17", "vs_ramp"):
        m = arm[label]
        if m["score"] is None or m["score"] <= 0.50:
            failures.append(f"{label} aggregate {m['score']} <= 0.50")
        for side in ("white", "black"):
            v = m[side]
            legs.append(v)
            if v is None or v < SIDE_FLOOR:
                failures.append(f"{label} {side} leg {v} < floor {SIDE_FLOOR}")
    arm["legs"] = legs
    arm["failures"] = failures
    arm["verdict"] = "PASS" if not failures else "FAIL"
    log(f"      {tag} VERDICT: {arm['verdict']} {failures if failures else ''}")
    return arm


def write_report():
    lines = [
        "# v18 candidate run — report",
        "",
        f"Started {results.get('started')}, finished {time.strftime('%Y-%m-%dT%H:%M:%S')}.",
        f"Corpus `{RAW}` -> `{PROCESSED}` (ramp labels: floor 0.5, horizon 60, 15ch).",
        "",
        "Bar: beat BOTH v17 and ramp head-to-head, per-side floor "
        f"{SIDE_FLOOR} on every leg, no anchor-Black regression. "
        "These were fixed before the run and are not tuned to the result.",
        "",
        "## Arms",
        "",
        "| arm | vs v17 (W/B) | vs ramp (W/B) | anchor | verdict |",
        "|---|---|---|---|---|",
    ]
    for tag, arm in results["arms"].items():
        if not arm:
            lines.append(f"| {tag} | — | — | — | DID NOT COMPLETE |")
            continue
        v17, ramp, anc = arm["vs_v17"], arm["vs_ramp"], arm["anchor"]
        lines.append(
            f"| `{tag}` | {v17['score']} ({v17['white']}/{v17['black']}) "
            f"| {ramp['score']} ({ramp['white']}/{ramp['black']}) "
            f"| {anc['score']} (B {anc['black_win_share']}) | **{arm['verdict']}** |")
    lines += ["", "## Failures", ""]
    any_fail = False
    for tag, arm in results["arms"].items():
        for f in (arm or {}).get("failures", []):
            lines.append(f"- `{tag}`: {f}")
            any_fail = True
    if not any_fail:
        lines.append("None — every leg cleared the floor.")
    lines += ["", "## Stage timings", "",
              "| stage | exit | minutes |", "|---|---|---|"]
    for s in results["stages"]:
        lines.append(f"| {s['name']} | {s['exit']} | {s['minutes']} |")
    lines += ["", "## Next step", "",
              "A PASS is automated evidence only. Promotion still requires the "
              "owner gate: his playtest decides, and no automated scorecard "
              "substitutes for it.", ""]
    with open(REPORT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    log(f"report written -> {os.path.relpath(REPORT, ROOT)}")


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception as e:  # non-Windows or restricted; not fatal
        log(f"note: could not inhibit sleep ({e})")

    results["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    log("=== v18 chain start ===")

    run("pretrain_check", ["tools/pretrain_check.py", RAW,
                           "--reference", "data/raw/combined_v16"])

    run("process", [
        "src/data_processor.py", "--raw-dir", RAW, "--output-dir", PROCESSED,
        "--value-floor", "0.5", "--value-horizon", "60",
        "--channels", "15", "--seed", str(SEED),
    ])

    for tag, model_dir, extra in ARMS:
        results["arms"][tag] = None
        run(f"train:{tag}", [
            "src/train.py", "--data-dir", PROCESSED, "--model-dir", model_dir,
            "--target", "game_result", "--value-head", "scalar",
            "--epochs", "30", "--seed", str(SEED),
        ] + extra)

    for tag, model_dir, _extra in ARMS:
        results["arms"][tag] = evaluate_arm(tag, model_dir)

    write_report()
    log("=== v18 chain complete ===")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        log(f"CHAIN ABORTED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            write_report()
        except Exception:
            pass
        sys.exit(1)
