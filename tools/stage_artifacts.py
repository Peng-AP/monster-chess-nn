"""Validate durable bootstrap stage outputs before an unattended resume.

Directory existence is not a completion marker: generators create their
output directory before doing useful work, and an interruption can leave a
perfectly plausible-looking partial directory behind.  These checks bind the
stage summaries to the requested workload and to the files on disk.
"""
import glob
import json
import os


def _read(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def generation_complete(raw_dir, requested_games):
    """True only for a clean, fully saved self-play generation."""
    summary = _read(os.path.join(raw_dir, "generation_summary.json"))
    if not isinstance(summary, dict):
        return False
    requested_games = int(requested_games)
    if (summary.get("num_games_requested") != requested_games
            or summary.get("saved_games") != requested_games):
        return False
    if any(summary.get(key, 0) != 0 for key in
           ("skipped_empty", "failed_games", "timed_out_games")):
        return False
    files = glob.glob(os.path.join(raw_dir, "game_*.jsonl"))
    return len(files) == requested_games and all(os.path.getsize(p) > 0
                                                 for p in files)


def reanalysis_complete(output_dir, sampled, kept, simulations):
    """True only when the deep-search census and teacher set are complete."""
    summary = _read(os.path.join(output_dir, "reanalysis_summary.json"))
    if not isinstance(summary, dict):
        return False
    census = summary.get("census") or {}
    if (census.get("sampled") != int(sampled)
            or summary.get("kept") != int(kept)
            or summary.get("simulations") != int(simulations)):
        return False
    files = glob.glob(os.path.join(output_dir, "teacher_*.jsonl"))
    return len(files) == int(kept) and all(os.path.getsize(p) > 0
                                          for p in files)
