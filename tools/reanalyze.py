"""Build policy-only deep-search teachers from ordinary self-play positions.

This is the bootstrap loop's hard-example miner.  It deliberately does not
encode tactical motifs.  Positions are sampled by side, re-searched with the
incumbent at a larger simulation budget, and ranked by general disagreement:

* Jensen-Shannon divergence between recorded and deep visit policies;
* absolute change in the selected-move search value;
* whether the maximum-visit action changed.

The emitted rows teach policy only.  Real completed-game outcomes remain the
only value targets, while trustworthy recorded distance-to-end can still train
the optional moves-left auxiliary head.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_worker_engine = None


def _normalise_policy(policy):
    clean = {str(key): max(0.0, float(value))
             for key, value in (policy or {}).items() if float(value) > 0}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in clean.items()}


def policy_js_divergence(left, right):
    """Bounded, symmetric policy disagreement in natural-log units."""
    p = _normalise_policy(left)
    q = _normalise_policy(right)
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    def kl(source):
        total = 0.0
        for key in keys:
            value = source.get(key, 0.0)
            if value <= 0:
                continue
            midpoint = 0.5 * (p.get(key, 0.0) + q.get(key, 0.0))
            total += value * math.log(value / midpoint)
        return total

    return 0.5 * (kl(p) + kl(q))


def _argmax(policy):
    normalised = _normalise_policy(policy)
    return max(normalised, key=normalised.get) if normalised else None


def _fraction_count(total, fraction):
    """Round halves upward so a one-row Black-priority sample stays Black."""
    return int(math.floor(total * fraction + 0.5))


def disagreement_score(recorded_policy, deep_policy, recorded_value,
                       deep_value):
    js = policy_js_divergence(recorded_policy, deep_policy)
    value_delta = abs(float(deep_value) - float(recorded_value))
    action_changed = _argmax(recorded_policy) != _argmax(deep_policy)
    # JS is bounded by ln(2).  The value and action terms put large belief
    # revisions and discrete search reversals on comparable scales.
    priority = js + 0.25 * min(2.0, value_delta) + (0.25 if action_changed else 0.0)
    return {
        "priority": float(priority),
        "policy_js": float(js),
        "value_delta": float(value_delta),
        "action_changed": bool(action_changed),
    }


def iter_records(source_dir):
    for dirpath, _dirs, files in os.walk(source_dir):
        for name in sorted(files):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    # A follow-up/deeper reanalysis pass may share a tree with
                    # earlier teacher output. Teachers are targets, not fresh
                    # source positions; recursively teaching from them would
                    # duplicate positions and lose the original split parent.
                    if record.get("source") == "deep_search_reanalysis":
                        continue
                    if (record.get("fen") and record.get("current_player")
                            in ("white", "black") and record.get("policy")):
                        yield {
                            "path": os.path.relpath(path, source_dir).replace("\\", "/"),
                            "line": line_number,
                            "record": record,
                        }


def stratified_sample(rows, count, black_fraction, seed):
    """Deterministically sample a requested side mix, filling shortages."""
    rows = list(rows)
    if count <= 0 or count >= len(rows):
        return rows
    rng = random.Random(seed)
    black = [row for row in rows
             if row["record"].get("current_player") == "black"]
    white = [row for row in rows
             if row["record"].get("current_player") == "white"]
    rng.shuffle(black)
    rng.shuffle(white)
    black_n = min(len(black), _fraction_count(count, black_fraction))
    white_n = min(len(white), count - black_n)
    selected = black[:black_n] + white[:white_n]
    if len(selected) < count:
        used = {(row["path"], row["line"]) for row in selected}
        remainder = [row for row in rows
                     if (row["path"], row["line"]) not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[:count - len(selected)])
    return selected


def select_top(rows, count, black_fraction):
    """Keep the hardest rows while retaining the requested side coverage."""
    rows = list(rows)
    if count <= 0 or count >= len(rows):
        return sorted(rows, key=lambda row: row["metrics"]["priority"], reverse=True)
    black = sorted((row for row in rows if row["current_player"] == "black"),
                   key=lambda row: row["metrics"]["priority"], reverse=True)
    white = sorted((row for row in rows if row["current_player"] == "white"),
                   key=lambda row: row["metrics"]["priority"], reverse=True)
    black_n = min(len(black), _fraction_count(count, black_fraction))
    white_n = min(len(white), count - black_n)
    selected = black[:black_n] + white[:white_n]
    if len(selected) < count:
        identities = {row["identity"] for row in selected}
        remainder = sorted((row for row in rows if row["identity"] not in identities),
                           key=lambda row: row["metrics"]["priority"], reverse=True)
        selected.extend(remainder[:count - len(selected)])
    return sorted(selected, key=lambda row: row["metrics"]["priority"], reverse=True)


def _init_worker(model_path, simulations, engine_choice, batch_size):
    global _worker_engine
    from evaluation import NNEvaluator
    evaluator = NNEvaluator(model_path)
    kwargs = dict(num_simulations=simulations, eval_fn=evaluator,
                  root_noise=False, allow_early_stop=False)
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if engine_choice == "native":
        from native_mcts import NativeMCTS
        _worker_engine = NativeMCTS(**kwargs)
    else:
        from mcts import MCTS
        _worker_engine = MCTS(**kwargs)


def _reanalyze_one(item):
    from monster_chess import MonsterChessGame
    record = item["record"]
    is_white = record["current_player"] == "white"
    game = MonsterChessGame(fen=record["fen"])
    game.is_white_turn = is_white
    game.board.turn = is_white
    game.white_half_pending = bool(record.get("half"))
    # The returned visit distribution is the teacher target.  Select its
    # associated child deterministically so the value-disagreement component
    # does not change merely because temperature sampling chose another child.
    _action, deep_policy, deep_value = _worker_engine.get_best_action(
        game, temperature=0.0)
    if not deep_policy:
        raise ValueError(f"deep search returned no policy for {record['fen']}")
    metrics = disagreement_score(
        record["policy"], deep_policy, record.get("mcts_value", 0.0), deep_value)
    identity = hashlib.sha256(
        f"{item['path']}:{item['line']}:{record['fen']}:{record.get('half', 0)}"
        .encode("utf-8")).hexdigest()
    return {
        "identity": identity,
        "source_path": item["path"],
        "source_line": item["line"],
        "fen": record["fen"],
        "current_player": record["current_player"],
        "half": int(bool(record.get("half"))),
        "game_result": float(record.get("game_result", 0.0)),
        "plies_to_end": record.get("plies_to_end"),
        "deep_policy": deep_policy,
        "deep_value": float(deep_value),
        "metrics": metrics,
    }


def _write_teacher(output_dir, rows, model_path, simulations):
    os.makedirs(output_dir)
    for index, row in enumerate(rows):
        decisive = abs(row["game_result"]) >= 0.999 and row["plies_to_end"] is not None
        record = {
            "fen": row["fen"],
            "mcts_value": round(row["deep_value"], 6),
            "policy": row["deep_policy"],
            "current_player": row["current_player"],
            "half": row["half"],
            "game_result": row["game_result"],
            "value_weight": 0.0,
            "policy_weight": 1.0,
            "moves_left_weight": 1.0 if decisive else 0.0,
            "source": "deep_search_reanalysis",
            "teacher_model": os.path.relpath(model_path, ROOT).replace("\\", "/"),
            "teacher_simulations": int(simulations),
            "source_record": {
                "path": row["source_path"], "line": row["source_line"]},
            "disagreement": row["metrics"],
        }
        if row["plies_to_end"] is not None:
            record["plies_to_end"] = int(row["plies_to_end"])
        path = os.path.join(output_dir, f"teacher_{index:05d}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sample", type=int, default=4000,
                    help="positions to deep-search before ranking")
    ap.add_argument("--keep", type=int, default=1000,
                    help="highest-disagreement positions to emit")
    ap.add_argument("--black-fraction", type=float, default=0.50)
    ap.add_argument("--simulations", type=int, default=1600)
    ap.add_argument("--engine", choices=("python", "native"), default="native")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stall-timeout", type=float, default=600.0,
                    help="fail if no deep-search task completes for this many seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the input census and planned sample only")
    args = ap.parse_args()

    if not os.path.isdir(args.source_dir):
        ap.error(f"source directory not found: {args.source_dir}")
    if not os.path.isfile(args.model):
        ap.error(f"model not found: {args.model}")
    if not 0.0 <= args.black_fraction <= 1.0:
        ap.error("--black-fraction must be in [0, 1]")
    if args.sample <= 0 or args.keep <= 0 or args.keep > args.sample:
        ap.error("require 0 < --keep <= --sample")
    if args.simulations <= 0 or args.workers <= 0 or args.stall_timeout <= 0:
        ap.error("--simulations, --workers, and --stall-timeout must be positive")
    if os.path.exists(args.output_dir):
        ap.error(f"output directory already exists: {args.output_dir}")

    all_rows = list(iter_records(args.source_dir))
    sampled = stratified_sample(
        all_rows, min(args.sample, len(all_rows)), args.black_fraction, args.seed)
    census = {
        "available": len(all_rows),
        "sampled": len(sampled),
        "sampled_white": sum(row["record"]["current_player"] == "white"
                             for row in sampled),
        "sampled_black": sum(row["record"]["current_player"] == "black"
                             for row in sampled),
    }
    print(json.dumps(census, indent=2))
    if args.dry_run:
        return

    started = time.time()
    results = []
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(args.model, args.simulations, args.engine,
                  args.batch_size))
    futures = {pool.submit(_reanalyze_one, row) for row in sampled}
    pending = set(futures)
    completed = 0
    try:
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=args.stall_timeout,
                return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                raise TimeoutError(
                    "deep-search reanalysis made no progress for "
                    f"{args.stall_timeout:.0f}s ({len(pending)} tasks remain)")
            for future in done:
                results.append(future.result())
                completed += 1
                if (completed % max(1, len(futures) // 20) == 0
                        or completed == len(futures)):
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed else 0.0
                    left = (len(futures) - completed) / rate if rate else 0.0
                    print(f"[{completed}/{len(futures)}] "
                          f"{elapsed / 60:.1f}m elapsed, "
                          f"~{left / 60:.1f}m left", flush=True)
    except BaseException:
        # A CUDA-blocked worker does not reliably leave a ProcessPoolExecutor
        # context.  Use the generation pipeline's tested bounded teardown so
        # a stalled reanalysis cannot consume the rest of an unattended run.
        from data_generation import terminate_pool
        terminate_pool(pool)
        raise
    else:
        pool.shutdown(wait=True)

    kept = select_top(results, min(args.keep, len(results)), args.black_fraction)
    output_dir = os.path.abspath(args.output_dir)
    # Build outside source_dir so an interrupted staging directory can never
    # be ingested as raw teachers by data_processor.  The completed directory
    # becomes visible in one rename on the same filesystem.
    staging_parent = os.path.dirname(os.path.abspath(args.source_dir))
    staging = os.path.join(
        staging_parent, f".{os.path.basename(output_dir)}.tmp-{os.getpid()}")
    if os.path.exists(staging):
        raise FileExistsError(f"reanalysis staging directory exists: {staging}")
    _write_teacher(staging, kept, args.model, args.simulations)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dir": os.path.relpath(args.source_dir, ROOT).replace("\\", "/"),
        "model": os.path.relpath(args.model, ROOT).replace("\\", "/"),
        "simulations": args.simulations,
        "engine": args.engine,
        "seed": args.seed,
        "census": census,
        "kept": len(kept),
        "kept_white": sum(row["current_player"] == "white" for row in kept),
        "kept_black": sum(row["current_player"] == "black" for row in kept),
        "mean_priority": (sum(row["metrics"]["priority"] for row in kept) / len(kept)
                          if kept else None),
        "mean_policy_js": (sum(row["metrics"]["policy_js"] for row in kept) / len(kept)
                           if kept else None),
        "action_change_rate": (sum(row["metrics"]["action_changed"] for row in kept)
                               / len(kept) if kept else None),
        "elapsed_sec": round(time.time() - started, 1),
    }
    with open(os.path.join(staging, "reanalysis_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(staging, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
