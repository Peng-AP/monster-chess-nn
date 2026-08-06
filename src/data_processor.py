"""Convert raw game JSONL records into training tensors.

Flat conversion + stratified game-level split (REWORK_PLAN Phase 5):
no source quotas, no repetition weights, no per-source value lambdas.
Every kept game contributes all of its positions once (plus the mirror
augmentation). Retention is bounded to two filters: generation age and
a minimum ply count for non-human games.

Source kinds are still *detected* (human_games/ and *_blackfocus/ subdir
names) — human games can be excluded and are exempt from the short-game
filter — but they no longer change weighting or targets.
"""
import json
import os
import re

import numpy as np
from tqdm import tqdm

from config import (
    TENSOR_SHAPE, POLICY_SIZE, PROMOTION_AWARE_POLICY_SIZE,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DATA_RETENTION_MAX_GENERATION_AGE, DATA_RETENTION_MIN_NONHUMAN_PLIES,
    RANDOM_SEED, VALUE_TARGET_HORIZON, VALUE_TARGET_FLOOR,
    VALUE_TARGET_DISCOUNT_MODE,
)
# Re-exported for existing importers (evaluation.py, mcts.py, tests).
from encoding import (  # noqa: F401
    PIECE_TO_LAYER,
    fen_to_tensor, mirror_tensor,
    move_to_index, move_to_policy_index, mirror_move_index,
    policy_dict_to_target, mirror_policy,
)


def _generation_from_game_id(game_id):
    """Extract nn_gen number from a relative game path, else None."""
    top = str(game_id).replace("\\", "/").split("/", 1)[0]
    m = re.match(r'^nn_gen(\d+)(?:_.*)?$', top)
    if not m:
        return None
    return int(m.group(1))


def _source_kind_from_rel(rel, is_human):
    if is_human:
        return "human"
    if "_blackfocus/" in rel or rel.endswith("_blackfocus"):
        return "blackfocus"
    return "selfplay"


def load_all_games(raw_dir, include_human=True,
                   max_generation_age=0, min_nonhuman_plies=0,
                   return_summary=False):
    """Load raw games from raw_dir (recursive), applying bounded retention.

    Returns a list of game dicts {game_id, records, source_kind,
    result_bucket, generation}; with return_summary=True also returns a
    retention summary dict.
    """
    paths = []
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        rel_dir = os.path.relpath(dirpath, raw_dir).replace("\\", "/")
        if (not include_human) and (rel_dir == "human_games" or rel_dir.startswith("human_games/")):
            continue
        for fname in sorted(filenames):
            if fname.endswith(".jsonl"):
                paths.append(os.path.join(dirpath, fname))

    games = []
    input_positions = 0
    for path in tqdm(paths, desc="Loading games"):
        rel = os.path.relpath(path, raw_dir).replace("\\", "/")
        is_human = "human_games/" in rel or rel.startswith("human_games")
        with open(path, "r") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]
        if not records:
            continue
        input_positions += len(records)
        result = records[-1].get("game_result", 0)
        source_record = records[0].get("source_record")
        split_parent = None
        if isinstance(source_record, dict) and source_record.get("path"):
            # Deep-search teachers are alternate labels for an existing
            # position, not independent games.  Keep every teacher in the
            # source game's split so the same FEN cannot leak from training
            # into validation/test under a new teacher filename.
            split_parent = str(source_record["path"]).replace("\\", "/")
        games.append({
            "game_id": rel,
            "records": records,
            "source_kind": _source_kind_from_rel(rel, is_human),
            "result_bucket": _result_bucket(result),
            "generation": _generation_from_game_id(rel),
            "split_parent": split_parent,
        })

    input_games = len(games)

    # Retention: bounded to generation age + minimum non-human plies.
    max_generation_age = int(max_generation_age or 0)
    min_nonhuman_plies = int(min_nonhuman_plies or 0)
    generations = [g["generation"] for g in games if g["generation"] is not None]
    latest_generation = max(generations) if generations else None

    kept = []
    dropped = {"age": {"games": 0, "positions": 0},
               "short_nonhuman": {"games": 0, "positions": 0}}
    for game in games:
        positions = len(game["records"])
        generation = game["generation"]
        if (
            latest_generation is not None
            and generation is not None
            and max_generation_age > 0
            and (latest_generation - int(generation)) > max_generation_age
        ):
            dropped["age"]["games"] += 1
            dropped["age"]["positions"] += positions
            continue
        if (
            game["source_kind"] != "human"
            and min_nonhuman_plies > 0
            and positions < min_nonhuman_plies
        ):
            dropped["short_nonhuman"]["games"] += 1
            dropped["short_nonhuman"]["positions"] += positions
            continue
        kept.append(game)

    summary = {
        "max_generation_age": max_generation_age,
        "min_nonhuman_plies": min_nonhuman_plies,
        "latest_generation": latest_generation,
        "input_games": input_games,
        "input_positions": input_positions,
        "kept_games": len(kept),
        "kept_positions": int(sum(len(g["records"]) for g in kept)),
        "dropped": dropped,
    }
    if not paths:
        print(f"No .jsonl files found in {raw_dir}")
    return (kept, summary) if return_summary else kept


def _result_bucket(result):
    """Bucket scalar game results to {-1, 0, +1} by sign."""
    if result > 0:
        return 1
    if result < 0:
        return -1
    return 0


def _split_games_by_result(games, seed):
    """Stratified group-level split (80/10/10) by result bucket.

    Ordinary games form one-member groups.  Reanalysis teachers carry a
    ``split_parent`` pointing at their source game and therefore move with it.
    This preserves game-level isolation even when one source position has
    several alternate policy labels.
    """
    rng = np.random.default_rng(seed)
    grouped = {}
    for game in games:
        key = game.get("split_parent") or game["game_id"]
        entry = grouped.setdefault(key, {
            "result_bucket": game["result_bucket"], "games": []})
        if entry["result_bucket"] != game["result_bucket"]:
            raise ValueError(
                f"split group {key!r} contains conflicting result buckets")
        entry["games"].append(game)

    by_bucket = {-1: [], 0: [], 1: []}
    for entry in grouped.values():
        by_bucket[entry["result_bucket"]].append(entry["games"])

    split_groups = {"train": [], "val": [], "test": []}
    for bucket in (-1, 0, 1):
        group = by_bucket[bucket]
        if not group:
            continue
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        split_groups["train"].extend(group[:n_train])
        split_groups["val"].extend(group[n_train:n_train + n_val])
        split_groups["test"].extend(group[n_train + n_val:])

    # Small-dataset fallback: keep splits non-empty when possible.
    group_count = len(grouped)
    if group_count >= 3 and len(split_groups["val"]) == 0:
        donor = "test" if len(split_groups["test"]) > 1 else "train"
        if split_groups[donor]:
            split_groups["val"].append(split_groups[donor].pop())
    if group_count >= 2 and len(split_groups["test"]) == 0:
        donor = "val" if len(split_groups["val"]) > 1 else "train"
        if split_groups[donor]:
            split_groups["test"].append(split_groups[donor].pop())
    if len(split_groups["train"]) == 0:
        donor = "test" if split_groups["test"] else "val"
        if split_groups[donor]:
            split_groups["train"].append(split_groups[donor].pop())

    split = {"train": [], "val": [], "test": []}
    for key in split:
        for group in split_groups[key]:
            split[key].extend(group)
        rng.shuffle(split[key])

    return split


def _discounted_results(records, horizon=VALUE_TARGET_HORIZON,
                        floor=VALUE_TARGET_FLOOR,
                        mode=VALUE_TARGET_DISCOUNT_MODE):
    """Build outcome-grounded scalar targets within each game segment.

    ``near_mate`` preserves the v16/v17 target: floor -> 1.0 over the last
    ``horizon`` plies and a flat floor before that.

    ``progress`` uses the same floor-at-horizon parameterization but continues
    the per-ply discount over the whole game. A faster win is worth more and a
    delayed loss is less negative, while the sign remains the observed result.

    Purpose is narrow (owner spec, 2026-07-11): a tiebreak so search at value
    saturation prefers mate-in-1 over mate-in-3 instead of drifting or
    suiciding. The flat plateau beyond the horizon is a uniform scale factor —
    relative ordering of all other positions is untouched, so no other
    behavior is modified (a global per-ply discount taxed Black's long wins;
    v13 rejected). floor >= 1.0 disables.

    Segments: merge drivers duplicate scarce human games by repeating the
    records INSIDE one file (keeps game-level splits leak-free), and distance
    to end must be computed within each copy.

    Preferred: records carry an explicit ``segment`` index (0, 1, 2, ...), and
    a change in that value starts a new copy. Fallback, for the many existing
    corpora written before the field existed: a record whose FEN equals the
    file's first FEN starts a new copy. The FEN rule is a heuristic — a game
    that genuinely revisits its own start position splits in the wrong place,
    and the convention is invisible to anyone writing a new merge driver — so
    new drivers should stamp ``segment``.
    """
    if mode not in ("near_mate", "progress"):
        raise ValueError("value discount mode must be 'near_mate' or 'progress'")
    if floor >= 1.0:
        return [rec.get("game_result", 0) for rec in records]
    gamma = floor ** (1.0 / max(1, horizon))
    if any("segment" in rec for rec in records):
        bounds = [i for i, rec in enumerate(records)
                  if i == 0 or rec.get("segment") != records[i - 1].get("segment")]
    else:
        start_fen = records[0].get("fen")
        bounds = [i for i, rec in enumerate(records)
                  if i == 0 or rec.get("fen") == start_fen]
    bounds.append(len(records))
    out = [0.0] * len(records)
    for a, b in zip(bounds, bounds[1:]):
        for i in range(a, b):
            # Preferred: an explicit distance stamped at generation time.
            # The positional fallback below is correct only for a game recorded
            # whole, so any consumer that drops records (filtering to a phase,
            # truncating a tail) silently relabels the survivors -- the new last
            # record reads as the finish and takes a full-strength target.
            # tests/test_ramp_label_positional.py pins that hazard.
            explicit = records[i].get("plies_to_end")
            plies_to_end = int(explicit) if explicit is not None else b - 1 - i
            if mode == "near_mate":
                plies_to_end = min(plies_to_end, horizon)
            out[i] = records[i].get("game_result", 0) * (gamma ** plies_to_end)
    return out


def policy_weight_for_record(rec, mask_human_ai=True):
    """Return whether a record is a trustworthy policy teacher.

    Explicit weights always win. Ordinary engine/self-play records remain
    teachers. With mask_human_ai=True (v18-era behavior) human games only
    teach policy through moves made by the eventual winning human. v16/v17
    trained with mask_human_ai=False — every human-game record a teacher —
    and both models that masked scored 4-7 points lower policy top-1, so
    the v17-faithful mode stays available via --no-human-ai-mask.
    """
    if "policy_weight" in rec:
        return float(rec["policy_weight"])
    if not mask_human_ai:
        return 1.0
    if rec.get("source") != "human_game":
        return 1.0
    if rec.get("actor") != "human":
        return 0.0
    result = float(rec.get("game_result", 0.0))
    is_white = rec.get("current_player") == "white"
    human_won = (is_white and result > 0) or ((not is_white) and result < 0)
    return 1.0 if human_won else 0.0


def value_weight_for_record(rec):
    """Return whether a record is a trustworthy VALUE teacher.

    Mirrors policy_weight_for_record, and defaults to 1.0 for everything, so
    every corpus built before this existed trains exactly as it did.

    It exists because "teach policy from this source but not value" was not
    expressible (HANDOFF SS7.1 called it bounded but unscoped). ps_monster
    carries mcts_value 0.0 and outcome labels from 1600-blitz games: excellent
    policy teachers, and beliefs about who was winning that nobody wants in
    the value head unless the evidence says so. Setting value_weight on those
    records turns the knowledge-vs-belief question into an A/B (DIRECTIVE D2).
    """
    if "value_weight" in rec:
        return float(rec["value_weight"])
    return 1.0


def moves_left_weight_for_record(rec):
    """Weight for the optional moves-left auxiliary target.

    An explicit weight always wins. Otherwise only decisive, trusted value
    trajectories teach game length. This excludes capped draws and preserves
    the policy-only contract of imported sources such as PlayStrategy.
    """
    if "moves_left_weight" in rec:
        return float(rec["moves_left_weight"])
    # Only exact capture outcomes are trustworthy duration targets. The
    # move-limit proxy is +/-0.5, and curriculum opinions can be fractional;
    # teaching their distance would make the head predict time-to-cap/tier.
    if abs(float(rec.get("game_result", 0.0))) < 0.999:
        return 0.0
    return value_weight_for_record(rec)


def _segment_bounds(records):
    """Return half-open bounds for repeated segments stored in one file."""
    if not records:
        return []
    if any("segment" in rec for rec in records):
        starts = [i for i, rec in enumerate(records)
                  if i == 0 or rec.get("segment") != records[i - 1].get("segment")]
    else:
        start_fen = records[0].get("fen")
        starts = [i for i, rec in enumerate(records)
                  if i == 0 or rec.get("fen") == start_fen]
    return list(zip(starts, starts[1:] + [len(records)]))


def _moves_left_targets(records):
    """Remaining decisions, including the decision represented by each row."""
    out = np.zeros((len(records),), dtype=np.float32)
    for a, b in _segment_bounds(records):
        for i in range(a, b):
            explicit = records[i].get("plies_to_end")
            remaining_after = int(explicit) if explicit is not None else b - 1 - i
            out[i] = max(1, remaining_after + 1)
    return out


def _legal_policy_mask_packed(rec, promotion_aware=False):
    """Packed legal-move mask for one half-move training record."""
    from monster_chess import MonsterChessGame

    is_white = rec["current_player"] == "white"
    game = MonsterChessGame(fen=rec["fen"])
    game.is_white_turn = is_white
    game.board.turn = is_white
    game.white_half_pending = bool(rec.get("half"))
    policy_size = PROMOTION_AWARE_POLICY_SIZE if promotion_aware else POLICY_SIZE
    mask = np.zeros((policy_size,), dtype=np.uint8)
    if not is_white:
        actions = game._get_black_actions(truncate_wins=False)
    elif game.white_half_pending:
        actions = game._white_second_half_moves(truncate_wins=False)
    else:
        actions = game._white_single_moves()
    indices = [move_to_policy_index(move, promotion_aware) for move in actions]
    if indices:
        mask[indices] = 1
    mirrored_indices = [mirror_move_index(index) for index in indices]
    mirrored = np.zeros_like(mask)
    if mirrored_indices:
        mirrored[mirrored_indices] = 1
    return (np.packbits(mask), np.packbits(mirrored),
            frozenset(indices), frozenset(mirrored_indices))


BLACK_WEIGHT_BALANCED = 1.75
"""Multiplier that roughly equalises the two sides' gradient share.

White moves twice per turn in this variant, so every turn emits two White
half-move records against Black's one: the corpus is ~64/36 White/Black by
construction, not by collection bias. Any capacity or training budget added to
the SHARED trunk is therefore spent mostly on White, because that is where the
loss reduction is. Measured on combined_v19_K: White 64.4% / Black 35.6%, and
64/36 = 1.78.

This exists to make "does Black benefit from more capacity?" a testable
question (owner hypothesis, 2026-08-02). Arm C added a 2.74x tower to
unbalanced data and White improved more -- which is what you would predict
either way, so it tested nothing about the hypothesis.
"""


def _convert_games_to_arrays(games, augment, value_horizon=VALUE_TARGET_HORIZON,
                             value_floor=VALUE_TARGET_FLOOR,
                             value_discount_mode=VALUE_TARGET_DISCOUNT_MODE,
                             input_channels=None, mask_human_ai=True,
                             black_weight=1.0, include_moves_left=False,
                             include_legal_masks=False,
                             include_capture_results=False,
                             promotion_aware=False,
                             mask_illegal_policy_targets=True,
                             conversion_stats=None):
    """Flat conversion of game records to tensors for one split."""
    tensors = []
    values = []
    game_results = []
    policy_targets = []
    policy_weights = []
    value_weights = []
    moves_left_targets = []
    moves_left_weights = []
    legal_masks_packed = []
    capture_results = []

    for game in tqdm(games, desc="Converting", leave=False):
        discounted = _discounted_results(
            game["records"], value_horizon, value_floor,
            mode=value_discount_mode,
        )
        game_moves_left = _moves_left_targets(game["records"])
        for rec, gr, moves_left in zip(
                game["records"], discounted, game_moves_left):
            is_white = rec["current_player"] == "white"
            half_pending = bool(rec.get("half"))
            tensor = fen_to_tensor(rec["fen"], is_white_turn=is_white,
                                   half_pending=half_pending,
                                   input_channels=input_channels)
            # mcts_value from data_generation is already from the
            # side-to-move perspective for both White and Black.
            # gr comes pre-discounted from _discounted_results.
            val = rec["mcts_value"]
            pol = policy_dict_to_target(
                rec["policy"], is_white, promotion_aware=promotion_aware)
            pol_weight = policy_weight_for_record(rec, mask_human_ai=mask_human_ai)
            val_weight = value_weight_for_record(rec)
            ml_weight = moves_left_weight_for_record(rec)
            if include_legal_masks:
                (legal_mask, mirrored_legal_mask,
                 legal_indices, _mirrored_legal_indices) = (
                    _legal_policy_mask_packed(
                        rec, promotion_aware=promotion_aware))
                target_indices = frozenset(np.flatnonzero(pol > 0))
                if pol_weight > 0 and not target_indices.issubset(legal_indices):
                    illegal = sorted(target_indices - legal_indices)
                    if not mask_illegal_policy_targets:
                        raise ValueError(
                            "policy target contains illegal move indices "
                            f"{illegal[:8]} at {rec['fen']} "
                            f"(half={rec.get('half', 0)})")
                    pol_weight = 0.0
                    if conversion_stats is not None:
                        key = "illegal_policy_targets_masked"
                        conversion_stats[key] = conversion_stats.get(key, 0) + 1
            if black_weight != 1.0 and not is_white:
                # Scales, so an explicitly masked record (weight 0) stays masked.
                pol_weight *= black_weight
                val_weight *= black_weight
                ml_weight *= black_weight

            tensors.append(tensor)
            values.append(val)
            game_results.append(gr)
            policy_targets.append(pol)
            policy_weights.append(pol_weight)
            value_weights.append(val_weight)
            moves_left_targets.append(moves_left)
            moves_left_weights.append(ml_weight)
            capture_result = float(rec.get("game_result", 0.0))
            capture_result = (1.0 if capture_result >= 1.0 else
                              -1.0 if capture_result <= -1.0 else 0.0)
            capture_results.append(capture_result)
            if include_legal_masks:
                legal_masks_packed.append(legal_mask)
            if augment:
                tensors.append(mirror_tensor(tensor))
                values.append(val)
                game_results.append(gr)
                policy_targets.append(mirror_policy(pol))
                policy_weights.append(pol_weight)
                value_weights.append(val_weight)
                moves_left_targets.append(moves_left)
                moves_left_weights.append(ml_weight)
                capture_results.append(capture_result)
                if include_legal_masks:
                    legal_masks_packed.append(mirrored_legal_mask)

    if tensors:
        X = np.array(tensors, dtype=np.float32)
        y_value = np.array(values, dtype=np.float32)
        y_result = np.array(game_results, dtype=np.float32)
        y_policy = np.array(policy_targets, dtype=np.float32)
        y_policy_weight = np.array(policy_weights, dtype=np.float32)
        y_value_weight = np.array(value_weights, dtype=np.float32)
        y_moves_left = np.array(moves_left_targets, dtype=np.float32)
        y_moves_left_weight = np.array(moves_left_weights, dtype=np.float32)
        y_legal_masks = np.array(legal_masks_packed, dtype=np.uint8)
        y_capture_result = np.array(capture_results, dtype=np.float32)
    else:
        channels = TENSOR_SHAPE[2] if input_channels is None else int(input_channels)
        X = np.zeros((0, 8, 8, channels), dtype=np.float32)
        y_value = np.zeros((0,), dtype=np.float32)
        y_result = np.zeros((0,), dtype=np.float32)
        policy_size = (PROMOTION_AWARE_POLICY_SIZE
                       if promotion_aware else POLICY_SIZE)
        y_policy = np.zeros((0, policy_size), dtype=np.float32)
        y_policy_weight = np.zeros((0,), dtype=np.float32)
        y_value_weight = np.zeros((0,), dtype=np.float32)
        y_moves_left = np.zeros((0,), dtype=np.float32)
        y_moves_left_weight = np.zeros((0,), dtype=np.float32)
        y_legal_masks = np.zeros((0, policy_size // 8), dtype=np.uint8)
        y_capture_result = np.zeros((0,), dtype=np.float32)
    base = (X, y_value, y_result, y_policy, y_policy_weight, y_value_weight)
    if include_moves_left:
        base = base + (y_moves_left, y_moves_left_weight)
    if include_legal_masks:
        base = base + (y_legal_masks,)
    if include_capture_results:
        base = base + (y_capture_result,)
    return base


def process_raw_data(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR,
                     augment=True, seed=RANDOM_SEED, include_human=True,
                     max_generation_age=DATA_RETENTION_MAX_GENERATION_AGE,
                     min_nonhuman_plies=DATA_RETENTION_MIN_NONHUMAN_PLIES,
                     value_horizon=VALUE_TARGET_HORIZON,
                     value_floor=VALUE_TARGET_FLOOR,
                     value_discount_mode=VALUE_TARGET_DISCOUNT_MODE,
                     input_channels=None, mask_human_ai=True, black_weight=1.0,
                     promotion_aware=False):
    """Convert raw game records to training tensors and save.

    When augment=True (default), each position is also horizontally
    mirrored, doubling the dataset size.  The mirror preserves the
    evaluation (same value / game_result) since Monster Chess is
    file-symmetric.
    """
    games, retention_summary = load_all_games(
        raw_dir,
        include_human=include_human,
        max_generation_age=max_generation_age,
        min_nonhuman_plies=min_nonhuman_plies,
        return_summary=True,
    )
    if not games:
        print("No data to process.")
        return
    print(
        "Retention summary: "
        f"kept_games={retention_summary['kept_games']}/{retention_summary['input_games']}, "
        f"kept_positions={retention_summary['kept_positions']}/{retention_summary['input_positions']}"
    )

    split_games = _split_games_by_result(games, seed=seed)
    train_games = split_games["train"]
    val_games = split_games["val"]
    test_games = split_games["test"]

    train_ids = {g["game_id"] for g in train_games}
    val_ids = {g["game_id"] for g in val_games}
    test_ids = {g["game_id"] for g in test_games}
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise RuntimeError("Game-level split overlap detected")
    print("Game split integrity: PASS (no overlap across train/val/test game IDs)")
    print(f"  Games: train={len(train_games)}, val={len(val_games)}, test={len(test_games)}")
    print(f"  Processing positions (augment={augment})...")
    conversion_stats = {}

    if value_floor < 1.0:
        if value_discount_mode == "progress":
            print(f"  Value targets: full-game progress discount "
                  f"(factor {value_floor} at {value_horizon} plies)")
        else:
            print(f"  Value targets: near-mate ramp {value_floor} -> 1.0 "
                  f"over last {value_horizon} plies")
    (X_train, yv_train, yr_train, yp_train, ypw_train, yvw_train,
     yml_train, ymlw_train, ylm_train, ycr_train) = _convert_games_to_arrays(
        train_games, augment, value_horizon, value_floor, value_discount_mode,
        input_channels=input_channels, mask_human_ai=mask_human_ai,
        black_weight=black_weight, include_moves_left=True,
        include_legal_masks=True, include_capture_results=True,
        promotion_aware=promotion_aware,
        conversion_stats=conversion_stats)
    (X_val, yv_val, yr_val, yp_val, ypw_val, yvw_val,
     yml_val, ymlw_val, ylm_val, ycr_val) = _convert_games_to_arrays(
        val_games, augment, value_horizon, value_floor, value_discount_mode,
        input_channels=input_channels, mask_human_ai=mask_human_ai,
        black_weight=black_weight, include_moves_left=True,
        include_legal_masks=True, include_capture_results=True,
        promotion_aware=promotion_aware,
        conversion_stats=conversion_stats)
    (X_test, yv_test, yr_test, yp_test, ypw_test, yvw_test,
     yml_test, ymlw_test, ylm_test, ycr_test) = _convert_games_to_arrays(
        test_games, augment, value_horizon, value_floor, value_discount_mode,
        input_channels=input_channels, mask_human_ai=mask_human_ai,
        black_weight=black_weight, include_moves_left=True,
        include_legal_masks=True, include_capture_results=True,
        promotion_aware=promotion_aware,
        conversion_stats=conversion_stats)

    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y_value = np.concatenate([yv_train, yv_val, yv_test], axis=0)
    y_result = np.concatenate([yr_train, yr_val, yr_test], axis=0)
    y_policy = np.concatenate([yp_train, yp_val, yp_test], axis=0)
    y_policy_weight = np.concatenate([ypw_train, ypw_val, ypw_test], axis=0)
    y_value_weight = np.concatenate([yvw_train, yvw_val, yvw_test], axis=0)
    y_moves_left = np.concatenate([yml_train, yml_val, yml_test], axis=0)
    y_moves_left_weight = np.concatenate(
        [ymlw_train, ymlw_val, ymlw_test], axis=0)
    y_legal_masks = np.concatenate([ylm_train, ylm_val, ylm_test], axis=0)
    y_capture_result = np.concatenate(
        [ycr_train, ycr_val, ycr_test], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "positions.npy"), X)
    np.save(os.path.join(output_dir, "mcts_values.npy"), y_value)
    np.save(os.path.join(output_dir, "game_results.npy"), y_result)
    np.save(os.path.join(output_dir, "policies.npy"), y_policy)
    np.save(os.path.join(output_dir, "policy_weights.npy"), y_policy_weight)
    np.save(os.path.join(output_dir, "value_weights.npy"), y_value_weight)
    np.save(os.path.join(output_dir, "moves_left.npy"), y_moves_left)
    np.save(os.path.join(output_dir, "moves_left_weights.npy"),
            y_moves_left_weight)
    np.save(os.path.join(output_dir, "legal_masks_packed.npy"), y_legal_masks)
    np.save(os.path.join(output_dir, "capture_results.npy"), y_capture_result)

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    splits = {
        "train": np.arange(0, n_train, dtype=np.int64),
        "val": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test": np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64),
    }
    np.savez(os.path.join(output_dir, "splits.npz"), **splits)

    with open(os.path.join(output_dir, "split_game_ids.json"), "w") as f:
        json.dump({
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
            "retention": retention_summary,
            "augment": bool(augment),
            "value_discount_mode": value_discount_mode,
            "value_horizon": int(value_horizon),
            "value_floor": float(value_floor),
            "input_channels": int(X.shape[3]) if len(X) else None,
            "total_positions": int(len(X)),
            "moves_left_target": "remaining recorded decisions including current",
            "moves_left_trust": "decisive records with positive value weight",
            "policy_size": int(y_policy.shape[1]),
            "promotion_aware_policy": bool(promotion_aware),
            "legal_policy_mask": (
                f"{y_policy.shape[1]} bits per row, numpy packbits big-endian"),
            "capture_result": "white-perspective king-capture outcome; move caps are 0",
            "conversion_stats": conversion_stats,
        }, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  positions.npy:    {X.shape}")
    print(f"  mcts_values.npy:  {y_value.shape}")
    print(f"  game_results.npy: {y_result.shape}")
    print(f"  policies.npy:     {y_policy.shape}")
    print(f"  policy_weights.npy: {y_policy_weight.shape} "
          f"(masked={int((y_policy_weight == 0).sum())})")
    print(f"  value_weights.npy:  {y_value_weight.shape} "
          f"(masked={int((y_value_weight == 0).sum())})")
    print(f"  moves_left.npy:     {y_moves_left.shape}")
    print(f"  moves_left_weights.npy: {y_moves_left_weight.shape} "
          f"(masked={int((y_moves_left_weight == 0).sum())})")
    print(f"  legal_masks_packed.npy: {y_legal_masks.shape} uint8")
    unique_capture, capture_counts = np.unique(
        y_capture_result, return_counts=True)
    print("  capture_results.npy: "
          f"{dict(zip(unique_capture.tolist(), capture_counts.tolist()))}")
    print(f"  invalid enabled policy rows masked: "
          f"{conversion_stats.get('illegal_policy_targets_masked', 0)}")
    print(f"  splits.npz:       train={n_train}, val={n_val}, test={n_test}")
    print("  split_game_ids.json: game-level split membership saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process raw game data into training tensors")
    parser.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--no-augment", action="store_true", help="Disable mirror augmentation")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for deterministic game-level splitting (default: {RANDOM_SEED})")
    parser.add_argument("--exclude-human-games", action="store_true",
                        help="Exclude data/raw/human_games from processing")
    parser.add_argument("--max-generation-age", type=int, default=DATA_RETENTION_MAX_GENERATION_AGE,
                        help="Drop nn_gen* games older than this many generations behind latest (<=0 disables)")
    parser.add_argument("--min-nonhuman-plies", type=int, default=DATA_RETENTION_MIN_NONHUMAN_PLIES,
                        help="Drop non-human games shorter than this many plies (<=0 disables)")
    parser.add_argument("--value-horizon", type=int, default=VALUE_TARGET_HORIZON,
                        help="Plies from game end for the near-mate target ramp")
    parser.add_argument("--value-floor", type=float, default=VALUE_TARGET_FLOOR,
                        help="Plateau factor beyond the horizon (>=1.0 disables)")
    parser.add_argument("--value-discount-mode", choices=["near_mate", "progress"],
                        default=VALUE_TARGET_DISCOUNT_MODE,
                        help="Near-mate tiebreak or full-game progress discount")
    parser.add_argument("--channels", type=int, default=None,
                        help="Position encoding width (15 legacy or 17; "
                             "default: config TENSOR_SHAPE)")
    parser.add_argument("--black-weight", type=float, default=1.0,
                        help="multiply policy AND value weights for "
                             f"Black-to-move records ({BLACK_WEIGHT_BALANCED} "
                             "roughly equalises the sides; default 1.0 = off)")
    parser.add_argument("--no-human-ai-mask", action="store_true",
                        help="v16/v17-faithful policy weighting: human-game AI "
                             "moves stay policy teachers (explicit weights "
                             "still honored)")
    parser.add_argument(
        "--promotion-aware-policy", action="store_true",
        help="Keep q/r/b/n promotions distinct in the 4288-logit policy ABI")
    args = parser.parse_args()
    if args.max_generation_age is not None and args.max_generation_age < 0:
        raise ValueError("--max-generation-age must be >= 0")
    if args.min_nonhuman_plies < 0:
        raise ValueError("--min-nonhuman-plies must be >= 0")

    process_raw_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        augment=not args.no_augment,
        seed=args.seed,
        include_human=not args.exclude_human_games,
        max_generation_age=args.max_generation_age,
        min_nonhuman_plies=args.min_nonhuman_plies,
        value_horizon=args.value_horizon,
        value_floor=args.value_floor,
        value_discount_mode=args.value_discount_mode,
        input_channels=args.channels,
        mask_human_ai=not args.no_human_ai_mask,
        black_weight=args.black_weight,
        promotion_aware=args.promotion_aware_policy,
    )
