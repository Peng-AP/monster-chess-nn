"""Non-binding Black-first sweep of low-cost PUCT controls.

Each arm plays the current bar checkpoint against itself. Model A receives one
changed search setting; model B retains the established defaults. The primary
read is A-as-Black, while White >= 0.40 and aggregate > 0.50 remain safeguards.
These short screens nominate configurations for the normal two-read gate; they
cannot promote a model or change a binding threshold.

The output is checkpointed after every arm so an interrupted sweep can be
resumed with ``--resume <artifact.json>``.
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

from config import (C_PUCT, DEFAULT_GAME_WORKERS, FPU_REDUCTION,
                    POLICY_TEMPERATURE)  # noqa: E402
from gate import BAR_MODEL, PER_SIDE_FLOOR, AGGREGATE_MIN  # noqa: E402


DEFAULT_ARMS = (
    ("policy_temperature", 0.80),
    ("policy_temperature", 0.90),
    ("policy_temperature", 1.10),
    ("policy_temperature", 1.25),
    ("c_puct", 1.20),
    ("c_puct", 1.80),
    ("fpu_reduction", 0.20),
    ("fpu_reduction", 0.40),
)
BASELINE = {
    "c_puct": C_PUCT,
    "fpu_reduction": FPU_REDUCTION,
    "policy_temperature": POLICY_TEMPERATURE,
}


def parse_arm(spec):
    """Parse ``name=value`` and reject settings outside the engine contract."""
    try:
        name, raw = spec.split("=", 1)
        value = float(raw)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("arm must look like name=value")
    if name not in BASELINE:
        raise argparse.ArgumentTypeError(
            f"unknown arm {name!r}; expected one of {sorted(BASELINE)}")
    if (not math.isfinite(value) or value < 0
            or (name == "policy_temperature" and value == 0)):
        raise argparse.ArgumentTypeError(f"invalid {name}={value}")
    return name, value


def arm_id(name, value):
    return f"{name}={value:g}"


def arm_search_kwargs(candidate):
    """Return explicit A/B kwargs for match.run_match."""
    a = dict(BASELINE)
    a.update(candidate)
    return {
        "c_puct_a": a["c_puct"],
        "fpu_reduction_a": a["fpu_reduction"],
        "policy_temperature_a": a["policy_temperature"],
        "c_puct_b": BASELINE["c_puct"],
        "fpu_reduction_b": BASELINE["fpu_reduction"],
        "policy_temperature_b": BASELINE["policy_temperature"],
    }


def summarize_arm(name, value, match):
    black = match["a_as_black"]["score"]
    white = match["a_as_white"]["score"]
    aggregate = match["a_score"]
    return {
        "id": arm_id(name, value),
        "candidate": {name: value},
        "black_score": black,
        "white_score": white,
        "aggregate_score": aggregate,
        "screen_pass": (
            black >= PER_SIDE_FLOOR
            and white >= PER_SIDE_FLOOR
            and aggregate > AGGREGATE_MIN
        ),
        "match": match,
    }


def rank_arms(arms):
    """Black score is deliberately the first ordering key."""
    return sorted(
        arms,
        key=lambda arm: (
            arm["black_score"], arm["aggregate_score"], arm["white_score"]),
        reverse=True,
    )


def save(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def new_payload(model, games, sims, seed, engine, arms):
    return {
        "experiment": "black_first_search_sweep",
        "binding": False,
        "status": "running",
        "model": os.path.relpath(os.path.abspath(model), ROOT),
        "baseline": dict(BASELINE),
        "priority": "black_score",
        "safeguards": {
            "black_score_min": PER_SIDE_FLOOR,
            "white_score_min": PER_SIDE_FLOOR,
            "aggregate_score_exclusive": AGGREGATE_MIN,
        },
        "games_per_arm": games,
        "sims": sims,
        "seed": seed,
        "seed_spacing": 100000,
        "engine": engine,
        "requested_arms": [arm_id(*arm) for arm in arms],
        "arms": [],
        "ranking": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_sweep(model, arms, games=20, sims=400, seed=20260805,
              workers=DEFAULT_GAME_WORKERS, engine="native", out_path=None,
              payload=None):
    from match import run_match

    if payload is None:
        payload = new_payload(model, games, sims, seed, engine, arms)
    # Normalize artifacts created by earlier harness revisions before resume.
    payload["safeguards"] = {
        "black_score_min": PER_SIDE_FLOOR,
        "white_score_min": PER_SIDE_FLOOR,
        "aggregate_score_exclusive": AGGREGATE_MIN,
    }
    for arm in payload.get("arms", []):
        arm["screen_pass"] = (
            arm["black_score"] >= PER_SIDE_FLOOR
            and arm["white_score"] >= PER_SIDE_FLOOR
            and arm["aggregate_score"] > AGGREGATE_MIN
        )
    finished = {arm["id"] for arm in payload.get("arms", [])}
    for index, (name, value) in enumerate(arms):
        identifier = arm_id(name, value)
        if identifier in finished:
            print(f"[sweep] skip completed {identifier}", flush=True)
            continue
        arm_seed = seed + index * 100000
        print(f"[sweep] {identifier}: candidate search vs baseline; "
              f"{games} games, seed {arm_seed}", flush=True)
        result = run_match(
            model, model, games, sims, arm_seed, workers=workers,
            engine=engine, **arm_search_kwargs({name: value}))
        summary = summarize_arm(name, value, result)
        payload["arms"].append(summary)
        payload["ranking"] = [arm["id"] for arm in rank_arms(payload["arms"])]
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if out_path:
            save(out_path, payload)
        print(f"[sweep]   Black={summary['black_score']:.3f} "
              f"White={summary['white_score']:.3f} "
              f"all={summary['aggregate_score']:.3f} "
              f"screen_pass={summary['screen_pass']}", flush=True)

    payload["status"] = "completed"
    payload["ranking"] = [arm["id"] for arm in rank_arms(payload["arms"])]
    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if out_path:
        save(out_path, payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=BAR_MODEL,
                    help="checkpoint used by both sides (default: current bar)")
    ap.add_argument("--arm", action="append", type=parse_arm,
                    help="name=value; repeat (default: built-in one-factor sweep)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--workers", type=int, default=DEFAULT_GAME_WORKERS)
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--resume", default=None, help="resume this sweep artifact")
    ap.add_argument("--out", default=None, help="artifact path")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(args.model)
    if args.games < 2 or args.sims <= 0 or args.workers <= 0:
        raise ValueError("games must be >= 2; sims and workers must be > 0")
    arms = tuple(args.arm or DEFAULT_ARMS)

    payload = None
    out_path = args.out
    if args.resume:
        with open(args.resume, encoding="utf-8") as f:
            payload = json.load(f)
        out_path = out_path or args.resume
    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(ROOT, "benchmarks", f"search_sweep_{stamp}.json")

    result = run_sweep(
        args.model, arms, games=args.games, sims=args.sims, seed=args.seed,
        workers=args.workers, engine=args.engine, out_path=out_path,
        payload=payload)
    print(json.dumps({
        "status": result["status"],
        "ranking": result["ranking"],
        "artifact": os.path.relpath(out_path, ROOT),
    }, indent=2))


if __name__ == "__main__":
    main()
