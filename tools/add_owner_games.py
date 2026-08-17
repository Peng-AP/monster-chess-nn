"""Extend a raw corpus with owner games it does not already contain.

    py -3 tools/add_owner_games.py --base-dir data/raw/combined_v17 \
        --out-dir data/raw/combined_v19_base \
        --add data/raw/human_games/black_2026_03 \
        --add data/raw/human_games/white_2026_03 --copies 6

Why a tool and not a one-off: the owner keeps playing, so this runs again, and
two things about it are easy to get wrong by hand.

**Membership cannot be judged by filename or record count.** Corpus copies are
flattened to game_NNNN.jsonl, and human games are stored at N-fold *in-file*
duplication (combined_v17: 82 games at 6x, 14 at 1x), so a 24-record game sits
in the corpus as a 144-record file. Identity here is the collapsed FEN
sequence, which survives both. Anything already present is skipped, so the tool
is idempotent and cannot silently double-weight a game.

**The base corpus is never modified.** combined_v17 is what v17, ramp and every
v18 arm trained on; mutating it in place would make those runs
irreproducible. This always writes a new corpus directory.

Duplication defaults to 6 to match the multiple the newer human games already
carry -- so the output differs from the base by exactly "these games were
added" and nothing else. (HANDOFF SS4.1 tested 1x vs 6x and found no Black
rescue, with the evidence mildly favouring 6x.)
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def records(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def collapsed_fens(recs):
    """The game's FEN sequence with any N-fold in-file repetition removed."""
    seq = [r["fen"] for r in recs]
    n = len(seq)
    for p in range(1, n + 1):
        if n % p == 0 and all(seq[i] == seq[i % p] for i in range(n)):
            return tuple(seq[:p])
    return tuple(seq)


def existing_fingerprints(human_dir):
    fps = {}
    if not os.path.isdir(human_dir):
        return fps
    for name in sorted(os.listdir(human_dir)):
        if name.endswith(".jsonl"):
            fps[collapsed_fens(records(os.path.join(human_dir, name)))] = name
    return fps


def next_index(human_dir):
    used = []
    for name in os.listdir(human_dir):
        stem = os.path.splitext(name)[0]
        if stem.startswith("game_") and stem[5:].isdigit():
            used.append(int(stem[5:]))
    return max(used) + 1 if used else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="corpus to extend (never modified)")
    ap.add_argument("--out-dir", required=True, help="new corpus directory")
    ap.add_argument("--add", action="append", required=True, metavar="DIR",
                    help="directory of owner games to fold in (repeatable)")
    ap.add_argument("--copies", type=int, default=6,
                    help="in-file duplication for added games (default: 6)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = os.path.abspath(args.base_dir)
    out = os.path.abspath(args.out_dir)
    if os.path.abspath(base) == out:
        ap.error("--out-dir must differ from --base-dir; the base is never modified")
    if os.path.exists(out) and not args.dry_run:
        ap.error(f"{out} already exists -- refusing to overwrite a corpus")

    base_human = os.path.join(base, "human_games")
    fps = existing_fingerprints(base_human)
    print(f"base {os.path.relpath(base, ROOT)}: {len(fps)} human games")

    planned, skipped = [], []
    for d in args.add:
        d = os.path.abspath(d)
        for name in sorted(os.listdir(d)):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(d, name)
            recs = records(path)
            if not recs:
                continue
            fp = collapsed_fens(recs)
            label = f"{os.path.basename(d)}/{name}"
            if fp in fps:
                skipped.append((label, fps[fp]))
            else:
                fps[fp] = label
                planned.append((label, path, recs))

    for label, was in skipped:
        print(f"  skip (already in corpus as {was}): {label}")
    print(f"\n{len(planned)} games to add, {len(skipped)} already present")
    if args.dry_run:
        for label, _p, recs in planned:
            print(f"  + {label}  {len(recs)} records x{args.copies}")
        return 0
    if not planned:
        print("nothing to do")
        return 0

    print(f"copying {os.path.relpath(base, ROOT)} -> {os.path.relpath(out, ROOT)}")
    shutil.copytree(base, out)

    out_human = os.path.join(out, "human_games")
    os.makedirs(out_human, exist_ok=True)
    idx = next_index(out_human)
    added = []
    for label, _path, recs in planned:
        name = f"game_{idx:05d}.jsonl"
        idx += 1
        with open(os.path.join(out_human, name), "w", encoding="utf-8") as f:
            for copy_i in range(args.copies):
                for rec in recs:
                    rec = dict(rec)
                    # data_processor prefers an explicit segment index over its
                    # first-FEN heuristic for finding copy boundaries.
                    rec["segment"] = copy_i
                    f.write(json.dumps(rec) + "\n")
        added.append({"source": label, "as": name, "records": len(recs),
                      "copies": args.copies})
        print(f"  + {label:44s} -> human_games/{name}  {len(recs)} recs x{args.copies}")

    manifest_path = os.path.join(out, "corpus_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path, encoding="utf-8"))
    manifest.setdefault("derived_from", os.path.relpath(base, ROOT).replace("\\", "/"))
    manifest.setdefault("owner_game_intake", []).append({
        "tool": "tools/add_owner_games.py",
        "copies": args.copies,
        "games_added": len(added),
        "games_skipped_already_present": len(skipped),
        "added": added,
    })
    manifest["human_games_total"] = len(
        [n for n in os.listdir(out_human) if n.endswith(".jsonl")])
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"\n{len(added)} games added; human_games now "
          f"{manifest['human_games_total']} files")
    print(f"manifest updated: {os.path.relpath(manifest_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
