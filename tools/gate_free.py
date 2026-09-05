"""Free-play gate. The candidate must clear the bar in the game as played.

Owner, 2026-09-05: "from now on gates should be based on freeplay, dedup until
a certain amount of unique games are played/time elapsed", and "run the gate,
keep it to an hour".

WHY FREE. The 2026-09-04 round robin (45 pairings, 36000 games) measured the
post-gen33 cohort 135-246 free Elo above v24/gen33/gen26 while the book
instrument compressed that same structure into 8-28 Elo, inside its own noise.
Five consecutive generations were recorded as failures by an instrument that
cannot see what they improved. gen36 is the sharpest case: it FAILED its book
gate at 400 sims against gen33, is level with gen33 at 3200 on a book, and
beats it by 164 Elo on free.

DEDUP IS LOAD-BEARING. After the sampled opening prefix play is deterministic,
so two games reaching the same opening state ARE the same game. Duplicate rate
ran 40% between same-era models and 68-76% within the top cohort, so a raw game
count is not a sample size. Legs stop on UNIQUE games or a wall-clock budget,
whichever comes first, and the verdict is scored on distinct games only.

THE PER-SIDE FLOOR CANNOT BE ABSOLUTE. Free-play par is model-specific: v24
scores White 0.8717 against itself, gen33 0.7933, gen38 0.5833. An absolute
floor would pass every old model and fail every new one regardless of strength.
So each colour is judged against the BAR's own free self-match -- measured once
per bar and cached, never per gate, and never the candidate's own. The gate
still clears against the previous best; the self-match is only the ruler.

WHAT THIS GATE CANNOT DO at an hour's budget: ~210 unique games a leg gives
SE ~24 Elo, so it resolves about +48 Elo at 2 SE. It separates tiers, not
neighbours. Raise --target-unique when the budget allows.

Thresholds are constants here, as in tools/gate.py: a threshold is never
weakened to let a recipe through.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from match import run_match  # noqa: E402

AGGREGATE_MIN = 0.50          # must be strictly beaten, both legs
PER_SIDE_BAND = 0.05          # each colour within this of the bar's own par
PAR_CACHE = os.path.join(ROOT, "benchmarks", "free_par_cache.json")
BATCH = 200


def elo(s):
    return -400 * math.log10(1 / s - 1) if 0 < s < 1 else float("nan")


def dedup(log_path):
    """Distinct games keyed on (colour, opening state). Exact, not heuristic."""
    rows = [json.loads(l) for l in open(log_path, encoding="utf-8")
            if l.strip()]
    seen, out = set(), []
    for r in rows:
        op = r.get("opening") or {}
        key = (bool(r["a_is_white"]), op.get("fen"), op.get("half"),
               op.get("turn_count"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out, len(rows)


def leg(name, model_a, model_b, sims, seed, target_unique, budget_min,
        workers, out_dir):
    """Play batches until target unique games or the budget expires."""
    games, played, t0 = [], 0, time.time()
    seen = set()
    while len(games) < target_unique:
        left = (time.time() - t0) / 60
        if left > budget_min:
            print(f"  {name}: budget {budget_min:.0f}m reached at "
                  f"{len(games)} unique", flush=True)
            break
        log_path = os.path.join(out_dir, f"{name}_b{played}.lines.jsonl")
        run_match(model_a, model_b, games=BATCH, sims=sims, sims_b=sims,
                  workers=workers, engine="native", seed=seed + played * 7919,
                  game_log=os.path.relpath(log_path, ROOT))
        rows, total = dedup(log_path)
        played += total
        for r in rows:
            op = r.get("opening") or {}
            key = (bool(r["a_is_white"]), op.get("fen"), op.get("half"),
                   op.get("turn_count"))
            if key not in seen:
                seen.add(key)
                games.append(r)
        print(f"  {name}: {played} played -> {len(games)} unique "
              f"({1 - len(games)/max(played,1):.0%} dupes), "
              f"{(time.time()-t0)/60:.1f}m", flush=True)
    if not games:
        return None
    a = [(g["result_for_a"] + 1) / 2 for g in games]
    w = [(g["result_for_a"] + 1) / 2 for g in games if g["a_is_white"]]
    b = [(g["result_for_a"] + 1) / 2 for g in games if not g["a_is_white"]]
    s = sum(a) / len(a)
    se = math.sqrt(s * (1 - s) / len(a)) if 0 < s < 1 else 0.0
    return {"name": name, "unique": len(a), "played": played,
            "score": round(s, 4), "se": round(se, 4),
            "white": round(sum(w) / len(w), 4) if w else None,
            "black": round(sum(b) / len(b), 4) if b else None,
            "elo": round(elo(s), 1) if 0 < s < 1 else None,
            "minutes": round((time.time() - t0) / 60, 1)}


def bar_par(bar, bar_name, sims, workers, out_dir, target, budget_min):
    """The bar against ITSELF -- the ruler for the per-side check. Cached."""
    cache = {}
    if os.path.exists(PAR_CACHE):
        cache = json.load(open(PAR_CACHE, encoding="utf-8"))
    key = f"{bar_name}@{sims}"
    if key in cache:
        print(f"  par: cached {key} -> W {cache[key]['white']:.4f} "
              f"B {cache[key]['black']:.4f}", flush=True)
        return cache[key]
    r = leg(f"par_{bar_name}", bar, bar, sims, 5150000, target, budget_min,
            workers, out_dir)
    if r is None:
        raise SystemExit("could not measure the bar's free par")
    cache[key] = {"white": r["white"], "black": r["black"],
                  "unique": r["unique"], "sims": sims}
    json.dump(cache, open(PAR_CACHE, "w"), indent=1)
    print(f"  par: {bar_name} W {r['white']:.4f} B {r['black']:.4f} "
          f"({r['unique']} unique)", flush=True)
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bar-model", required=True)
    ap.add_argument("--bar-name", required=True)
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--target-unique", type=int, default=200)
    ap.add_argument("--par-unique", type=int, default=150)
    ap.add_argument("--budget-min", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5200000)
    ap.add_argument("--report-path", default="benchmarks/gate_free.json")
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "benchmarks", "gate_free_legs")
    os.makedirs(out_dir, exist_ok=True)
    model = os.path.join(ROOT, a.model)
    bar = os.path.join(ROOT, a.bar_model)
    t0 = time.time()
    print(f"FREE GATE: {os.path.basename(os.path.dirname(a.model))} "
          f"vs {a.bar_name} @ {a.sims} sims, budget {a.budget_min:.0f}m",
          flush=True)

    # Three legs share the budget: par, bar, confirm.
    par = bar_par(bar, a.bar_name, a.sims, a.workers, out_dir,
                  a.par_unique, a.budget_min * 0.28)
    spent = (time.time() - t0) / 60
    rest = max(a.budget_min - spent, 1.0)
    legs = {}
    for i, nm in enumerate(("vs_bar", "vs_bar_confirm")):
        legs[nm] = leg(nm, model, bar, a.sims, a.seed + i * 848484,
                       a.target_unique, rest / (2 - i), a.workers, out_dir)
        spent = (time.time() - t0) / 60
        rest = max(a.budget_min - spent, 1.0)

    failures = []
    for nm, r in legs.items():
        if r is None:
            failures.append(f"{nm} produced no games")
            continue
        if r["score"] <= AGGREGATE_MIN:
            failures.append(f"{nm} aggregate {r['score']:.4f} "
                            f"<= {AGGREGATE_MIN}")
        for side, parv in (("white", par["white"]), ("black", par["black"])):
            v = r[side]
            if v is not None and parv is not None and v < parv - PER_SIDE_BAND:
                failures.append(f"{nm} {side} {v:.4f} below par {parv:.4f} "
                                f"- {PER_SIDE_BAND}")
    verdict = "PASS" if not failures else "FAIL"
    out = {"model": a.model, "bar": a.bar_model, "bar_name": a.bar_name,
           "sims": a.sims, "instrument": "free_dedup",
           "aggregate_min": AGGREGATE_MIN, "per_side_band": PER_SIDE_BAND,
           "bar_free_par": par, "legs": legs, "failures": failures,
           "verdict": verdict, "confirmed": legs.get("vs_bar_confirm")
           is not None,
           "minutes": round((time.time() - t0) / 60, 1),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = os.path.join(ROOT, a.report_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\npar {a.bar_name}: W {par['white']:.4f} B {par['black']:.4f}",
          flush=True)
    for nm, r in legs.items():
        if r:
            print(f"{nm}: {r['score']:.4f} ({r['elo']:+.1f} Elo) "
                  f"W {r['white']:.4f} B {r['black']:.4f} "
                  f"{r['unique']} unique of {r['played']} "
                  f"SE {r['se']:.4f}", flush=True)
    print(f"VERDICT: {verdict}  {failures}", flush=True)
    print(f"saved {a.report_path}", flush=True)


if __name__ == "__main__":
    main()
