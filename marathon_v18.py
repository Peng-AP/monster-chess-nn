"""30-hour unattended experiment marathon (owner window 2026-07-17/18).

Owner directive: ~30 hours away, nothing may wait on interaction, use the
time to improve results. Three corpus recipes in a row (recovery-1/-2,
detox) matched or beat v17 offline yet played worse, so this marathon tests
the two untouched levers plus the standing diagnostics:

  A. RAMP  — anti-saturation value targets: floor 0.5 over a 60-ply
     end-anchored horizon, trained with the SCALAR value head. (Key fact:
     under the WDL head the search value is the classification expectation,
     so label-magnitude shaping is invisible at inference; scalar makes it
     real.) Targets both owner observations: "shuffles when out of steam"
     and "allowed a promotion" are flat-value-landscape behavior.
     NOTE: the wide end-anchored ramp lowers mean |target| more for Black
     (longer wins) — a known, deliberate property of this experiment, unlike
     v13's unbounded per-ply tax. The corpus itself is unchanged and already
     passed the pretrain gate at default thresholds (combined_v16, detox run).
  B. HUMAN-ONLY probe — 81 games / ~2.1k positions, winner-side policy only:
     measures the human signal ceiling and is the clean shuffle control.
  C. CAPACITY — wider tower (stem 96, blocks 96,96,128,128,160x4), labels
     chosen by A's anchor result. Owner's own rule: when cycles go flat,
     test learner-capacity levers before requesting more games.
  D. SIMS PROBE — best candidate (else v17) vs anchor at 800 sims: is
     search depth the binding constraint?
  F. RAMP-2 — second point on the label-shape axis, direction chosen by A.
  E. EVIDENCE — if any candidate reaches anchor >= 0.55 without color
     collapse, gather full promotion evidence (h2h + 60-game anchor).

Model-diff gates run informationally this marathon: three cycles proved
offline metrics and play strength are decoupled here, and 20-game anchor
matches cost ~16 minutes — the matches ARE the gate. Promotion still
requires the owner; candidates land in models/candidates|experiments/.

Runs detached (WMI, hidden console, keep-awake, SIGINT-immune).
"""
import ctypes
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

REPORT = os.path.join(ROOT, "MARATHON_REPORT.md")
PY = sys.executable
T0 = time.time()
HOURS_BUDGET = 29.0

MERGED = os.path.join("data", "raw", "combined_v16")          # gate-passed (detox)
NEARMATE_PROCESSED = os.path.join("data", "processed", "combined_v16")
CLEAN_EVAL = os.path.join("data", "processed", "eval_clean_v13v16")
INCUMBENT = os.path.join("models", "fresh_start_v17", "best_value_net.pt")
TRACE_GAME = os.path.join("data", "raw", "human_games", "black_2026_07",
                          "game_00015.jsonl")
MATCH_ROOT = os.path.join("benchmarks", "marathon")
DETOX_ANCHOR_BASELINE = 0.30   # measured 2026-07-17

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

RESULTS = {}   # stage -> dict for the final summary


def hours_left():
    return HOURS_BUDGET - (time.time() - T0) / 3600.0


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args):
    log(f"START {name}: {' '.join(args)}")
    t0 = time.time()
    proc = subprocess.run([PY, "-u"] + args, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-22:])
    log(f"END {name} (exit={proc.returncode}, {dt:.1f} min)\n```\n{tail}\n```")
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-18:])
        log(f"STDERR {name}:\n```\n{err}\n```")
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout or ""


def match(name, model_a, model_b=None, games=20, sims=400):
    """Run a match into its own out-dir and return parsed scores."""
    out_dir = os.path.join(MATCH_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    args = ["tools/match.py", "--model-a", model_a, "--games", str(games),
            "--sims", str(sims), "--workers", "6", "--out-dir", out_dir]
    if model_b:
        args += ["--model-b", model_b]
    run(f"match-{name}", args)
    paths = sorted(glob.glob(os.path.join(out_dir, "*.json")),
                   key=os.path.getmtime)
    with open(paths[-1], encoding="utf-8") as f:
        m = json.load(f)
    scores = {"overall": m["a_score"],
              "white": m["a_as_white"]["score"],
              "black": m["a_as_black"]["score"]}
    log(f"SCORE {name}: {scores}")
    return scores


def trace(name, model_pt):
    if not os.path.exists(TRACE_GAME):
        return
    from evaluation import NNEvaluator
    from monster_chess import MonsterChessGame
    with open(TRACE_GAME, encoding="utf-8") as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
    states = [MonsterChessGame(records[i]["fen"]) for i in (14, 22, 24)]
    vals = NNEvaluator(model_pt).batch_evaluate(states)
    log(f"TRACE 0015 {name}: " + " ".join(f"{v:+.3f}" for v in vals)
        + " (v17: +0.993 +0.985 +0.942; positions are lost for White)")


def stage(name, budget_hours, fn):
    left = hours_left()
    if left < budget_hours:
        log(f"SKIP stage {name}: needs ~{budget_hours}h, only {left:.1f}h left")
        RESULTS[name] = {"skipped": f"time ({left:.1f}h left)"}
        return
    log(f"=== STAGE {name} (budget {budget_hours}h, {left:.1f}h left) ===")
    try:
        fn()
    except Exception as exc:
        log(f"STAGE {name} FAILED: {exc!r} — continuing with next stage")
        RESULTS.setdefault(name, {})["failed"] = repr(exc)


# ----------------------------------------------------------------------

RAMP_PROCESSED = os.path.join("data", "processed", "combined_v16_r50h60")
RAMP_MODEL = os.path.join("models", "candidates", "fresh_start_v18_ramp",
                          "best_value_net.pt")


def stage_a_ramp():
    run("process-ramp", [
        "src/data_processor.py", "--raw-dir", MERGED,
        "--output-dir", RAMP_PROCESSED, "--seed", "42", "--channels", "15",
        "--value-discount-mode", "near_mate",
        "--value-horizon", "60", "--value-floor", "0.5",
    ])
    run("train-ramp", [
        "src/train.py", "--data-dir", RAMP_PROCESSED,
        "--model-dir", os.path.dirname(RAMP_MODEL),
        "--target", "game_result", "--value-head", "scalar",
        "--select-metric", "decisive", "--epochs", "30", "--seed", "42",
    ])
    trace("ramp", RAMP_MODEL)
    run("diff-ramp-informational", [
        "tools/model_diff.py", "--candidate", RAMP_MODEL,
        "--incumbent", INCUMBENT, "--data-dir", CLEAN_EVAL,
        "--split", "all", "--max-positions", "8192",
    ])
    scores = match("ramp_anchor", RAMP_MODEL)
    RESULTS["A_ramp"] = {"anchor": scores}
    if scores["overall"] >= 0.40:
        RESULTS["A_ramp"]["h2h_v17"] = match("ramp_h2h", RAMP_MODEL, INCUMBENT)


HUMAN_RAW = os.path.join("data", "raw", "human_only_v1")
HUMAN_PROCESSED = os.path.join("data", "processed", "human_only_v1")
HUMAN_MODEL = os.path.join("models", "experiments", "human_only_v1",
                           "best_value_net.pt")


def stage_b_human_only():
    dst = os.path.join(HUMAN_RAW, "human_games")
    os.makedirs(dst, exist_ok=False)
    n = 0
    for src in sorted(glob.glob(os.path.join(
            "data", "raw", "human_games", "**", "game_*.jsonl"),
            recursive=True)):
        shutil.copy(src, os.path.join(dst, f"game_{n:04d}.jsonl"))
        n += 1
    log(f"human-only corpus: {n} games, duplication x1, winner-side policy only")
    run("process-human-only", [
        "src/data_processor.py", "--raw-dir", HUMAN_RAW,
        "--output-dir", HUMAN_PROCESSED, "--seed", "42", "--channels", "15",
        "--value-discount-mode", "near_mate",
        "--value-horizon", "10", "--value-floor", "0.97",
    ])
    run("train-human-only", [
        "src/train.py", "--data-dir", HUMAN_PROCESSED,
        "--model-dir", os.path.dirname(HUMAN_MODEL),
        "--target", "game_result", "--value-head", "wdl",
        "--select-metric", "decisive", "--epochs", "30", "--seed", "42",
    ])
    trace("human_only", HUMAN_MODEL)
    RESULTS["B_human_only"] = {
        "anchor10": match("human_only_anchor", HUMAN_MODEL, games=10)}


CAP_MODEL = os.path.join("models", "candidates", "fresh_start_v18_cap",
                         "best_value_net.pt")


def stage_c_capacity():
    a = RESULTS.get("A_ramp", {}).get("anchor", {})
    ramp_helped = a.get("overall", 0.0) >= DETOX_ANCHOR_BASELINE + 0.05
    if ramp_helped:
        data_dir, head = RAMP_PROCESSED, "scalar"
    else:
        data_dir, head = NEARMATE_PROCESSED, "wdl"
    log(f"capacity uses {head} head on {data_dir} "
        f"(ramp anchor={a.get('overall')}, baseline={DETOX_ANCHOR_BASELINE})")
    run("train-cap", [
        "src/train.py", "--data-dir", data_dir,
        "--model-dir", os.path.dirname(CAP_MODEL),
        "--target", "game_result", "--value-head", head,
        "--select-metric", "decisive", "--epochs", "24", "--seed", "42",
        "--stem-channels", "96",
        "--res-channels", "96,96,128,128,160,160,160,160",
    ])
    trace("cap", CAP_MODEL)
    scores = match("cap_anchor", CAP_MODEL)
    RESULTS["C_capacity"] = {"anchor": scores, "head": head}
    if scores["overall"] >= 0.40:
        RESULTS["C_capacity"]["h2h_v17"] = match("cap_h2h", CAP_MODEL, INCUMBENT)


def _best_candidate():
    best_name, best_pt, best_score = None, None, 0.0
    for name, pt in (("A_ramp", RAMP_MODEL), ("C_capacity", CAP_MODEL)):
        s = RESULTS.get(name, {}).get("anchor", {}).get("overall", 0.0)
        if os.path.exists(pt) and s > best_score:
            best_name, best_pt, best_score = name, pt, s
    return best_name, best_pt, best_score


def stage_d_sims():
    name, pt, score = _best_candidate()
    if pt is None or score < 0.30:
        pt, name = INCUMBENT, "v17"
    log(f"sims probe uses {name}")
    RESULTS["D_sims800"] = {"model": name,
                            "anchor800": match("sims800_anchor", pt, sims=800)}


RAMP2_MODEL = os.path.join("models", "candidates", "fresh_start_v18_ramp2",
                           "best_value_net.pt")


def stage_f_ramp2():
    a = RESULTS.get("A_ramp", {}).get("anchor", {}).get("overall")
    if a is None:
        log("SKIP ramp2: stage A produced no anchor score")
        RESULTS["F_ramp2"] = {"skipped": "no stage-A result"}
        return
    if a >= DETOX_ANCHOR_BASELINE + 0.05:
        floor, horizon = "0.35", "90"     # push further in the same direction
    else:
        floor, horizon = "0.7", "30"      # milder shaping
    processed = os.path.join("data", "processed",
                             f"combined_v16_r{floor.replace('.', '')}h{horizon}")
    run("process-ramp2", [
        "src/data_processor.py", "--raw-dir", MERGED,
        "--output-dir", processed, "--seed", "42", "--channels", "15",
        "--value-discount-mode", "near_mate",
        "--value-horizon", horizon, "--value-floor", floor,
    ])
    run("train-ramp2", [
        "src/train.py", "--data-dir", processed,
        "--model-dir", os.path.dirname(RAMP2_MODEL),
        "--target", "game_result", "--value-head", "scalar",
        "--select-metric", "decisive", "--epochs", "30", "--seed", "42",
    ])
    trace("ramp2", RAMP2_MODEL)
    scores = match("ramp2_anchor", RAMP2_MODEL)
    RESULTS["F_ramp2"] = {"floor": floor, "horizon": horizon, "anchor": scores}
    if scores["overall"] >= 0.40:
        RESULTS["F_ramp2"]["h2h_v17"] = match("ramp2_h2h", RAMP2_MODEL, INCUMBENT)


def stage_e_evidence():
    for name, pt in (("A_ramp", RAMP_MODEL), ("C_capacity", CAP_MODEL),
                     ("F_ramp2", RAMP2_MODEL)):
        r = RESULTS.get(name, {})
        s = r.get("anchor", {})
        if (s.get("overall", 0.0) >= 0.55
                and s.get("white", 0.0) >= 0.40 and s.get("black", 0.0) >= 0.40):
            log(f"EVIDENCE: {name} qualifies — gathering promotion evidence")
            if "h2h_v17" not in r:
                r["h2h_v17"] = match(f"{name}_h2h_late", pt, INCUMBENT)
            r["anchor60"] = match(f"{name}_anchor60", pt, games=60)
            return
    log("EVIDENCE: no candidate reached anchor >= 0.55 without color collapse")


def main():
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"\n# MARATHON — started {time.strftime('%Y-%m-%d %H:%M:%S')}"
                f" (budget {HOURS_BUDGET}h)\n\n")
    for req in (MERGED, NEARMATE_PROCESSED, CLEAN_EVAL, INCUMBENT):
        if not os.path.exists(req):
            raise RuntimeError(f"missing input: {req}")
    for new in (RAMP_PROCESSED, os.path.dirname(RAMP_MODEL), HUMAN_RAW,
                HUMAN_PROCESSED, os.path.dirname(HUMAN_MODEL),
                os.path.dirname(CAP_MODEL), os.path.dirname(RAMP2_MODEL)):
        if os.path.exists(new):
            raise RuntimeError(f"output already exists: {new}")

    stage("A_ramp", 7.0, stage_a_ramp)
    stage("B_human_only", 1.5, stage_b_human_only)
    stage("C_capacity", 11.0, stage_c_capacity)
    stage("D_sims800", 1.5, stage_d_sims)
    stage("F_ramp2", 7.0, stage_f_ramp2)
    stage("E_evidence", 2.5, stage_e_evidence)

    log("=== MARATHON SUMMARY ===")
    log(json.dumps(RESULTS, indent=2))
    log("ALL STAGES DONE — nothing is promoted; owner playtest decides. "
        "Candidates: models/candidates/fresh_start_v18_{ramp,cap,ramp2}, "
        "models/experiments/human_only_v1")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        main()
    except BaseException as exc:
        log(f"MARATHON ABORTED: {exc!r}")
        sys.exit(1)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
