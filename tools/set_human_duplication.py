"""Rewrite a raw corpus with a chosen human-game duplication multiple.

Merge drivers duplicate the scarce human games by repeating their records
INSIDE one file (a game-level split then keeps every copy on the same side of
the split, so duplication cannot leak train positions into test).  The multiple
is baked into the corpus at merge time, so testing "did 6x duplication cost
Black?" means rebuilding the corpus at another multiple.

This copies raw-dir to out-dir unchanged except for files under human_games/,
whose records are re-emitted at --copies copies, and stamps an explicit
``segment`` index on every record it writes (data_processor prefers that field
over its first-FEN heuristic for finding copy boundaries).

    py -3 tools/set_human_duplication.py --raw-dir data/raw/combined_v17 \
        --out-dir data/raw/combined_v17_dup1 --copies 1
"""
import argparse
import json
import os
import shutil


def split_segments(records):
    """Records -> list of copies, using data_processor's own boundary rule."""
    if any("segment" in rec for rec in records):
        bounds = [i for i, rec in enumerate(records)
                  if i == 0 or rec.get("segment") != records[i - 1].get("segment")]
    else:
        start_fen = records[0].get("fen")
        bounds = [i for i, rec in enumerate(records)
                  if i == 0 or rec.get("fen") == start_fen]
    bounds.append(len(records))
    return [records[a:b] for a, b in zip(bounds, bounds[1:])]


def rewrite(path, out_path, copies):
    """-> (segments_in, segments_out). Non-uniform copies are left untouched."""
    with open(path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        shutil.copyfile(path, out_path)
        return 0, 0
    segments = split_segments(records)
    # Every copy must be identical for re-duplication to be meaningful; if the
    # file is not a clean N-fold repeat, pass it through rather than guess.
    # The segment index itself is what distinguishes copies — compare without it.
    def bare(seg):
        return [{k: v for k, v in rec.items() if k != "segment"} for rec in seg]

    base = segments[0]
    if any(bare(seg) != bare(base) for seg in segments[1:]):
        shutil.copyfile(path, out_path)
        return len(segments), len(segments)
    with open(out_path, "w") as f:
        for i in range(copies):
            for rec in base:
                rec = dict(rec)
                rec["segment"] = i
                f.write(json.dumps(rec) + "\n")
    return len(segments), copies


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--copies", type=int, required=True,
                    help="human-game copies to emit (1 = no duplication)")
    args = ap.parse_args()
    if args.copies < 1:
        raise SystemExit("--copies must be >= 1")
    if os.path.abspath(args.raw_dir) == os.path.abspath(args.out_dir):
        raise SystemExit("--out-dir must differ from --raw-dir")

    n_files = n_human = 0
    seg_before = seg_after = 0
    skipped = 0
    for dirpath, _dirnames, filenames in os.walk(args.raw_dir):
        rel_dir = os.path.relpath(dirpath, args.raw_dir).replace("\\", "/")
        out_dir = os.path.join(args.out_dir, rel_dir) if rel_dir != "." else args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        is_human = rel_dir == "human_games" or rel_dir.startswith("human_games/")
        for fname in sorted(filenames):
            src = os.path.join(dirpath, fname)
            dst = os.path.join(out_dir, fname)
            n_files += 1
            if not (is_human and fname.endswith(".jsonl")):
                shutil.copyfile(src, dst)
                continue
            n_human += 1
            before, after = rewrite(src, dst, args.copies)
            seg_before += before
            seg_after += after
            if before > 1 and after == before and args.copies != before:
                skipped += 1

    print(f"files copied      : {n_files}")
    print(f"human game files  : {n_human}")
    print(f"human copies      : {seg_before} -> {seg_after}")
    if skipped:
        print(f"passed through    : {skipped} (copies were not identical)")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
