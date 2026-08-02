"""Merge a raw source into a corpus, optionally as a policy-only teacher.

    py -3 tools/merge_source.py --base-dir data/raw/combined_v19_base \
        --source data/raw/ps_monster --as ps_monster --value-weight 0 \
        --out-dir data/raw/combined_v19_K

This is the D2 fork-splitter. HANDOFF SS7.1 framed ps_monster as a bet that
could not be placed either way:

* **Knowledge** -- Black does not know the technique, and 829 games of human
  openings at 41.2% pawn phase are the best teaching data the project has.
* **Belief** -- Black evaluates the phase as lost. ps_monster's outcome labels
  say Black converts 39.7% where the owner converts 100%, so importing them as
  value targets would deepen exactly the belief we want to remove.

`--value-weight 0` stamps every merged record so it teaches policy and
contributes zero value gradient (see data_processor.value_weight_for_record and
train._power_loss). Merging the same source twice, once with the flag and once
without, turns the argument into an A/B.

The base corpus is never modified; the merged source lands as its own
subdirectory so `data_processor` walks it and corpus audits can still see where
every record came from.
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def merge(source, dest, value_weight, policy_weight):
    """Copy source/*.jsonl into dest, stamping weights. -> (games, records)."""
    os.makedirs(dest, exist_ok=True)
    games = records = 0
    for name in sorted(os.listdir(source)):
        if not name.endswith(".jsonl"):
            continue
        src_path = os.path.join(source, name)
        with open(src_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        if not lines:
            continue
        with open(os.path.join(dest, name), "w", encoding="utf-8") as out:
            for line in lines:
                rec = json.loads(line)
                if value_weight is not None:
                    rec["value_weight"] = float(value_weight)
                if policy_weight is not None:
                    rec["policy_weight"] = float(policy_weight)
                out.write(json.dumps(rec) + "\n")
                records += 1
        games += 1
    return games, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="corpus to extend (never modified)")
    ap.add_argument("--source", required=True, help="raw dir of games to merge in")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--as", dest="subdir", default=None,
                    help="subdirectory name inside the corpus (default: source basename)")
    ap.add_argument("--value-weight", type=float, default=None,
                    help="stamp this value_weight on every merged record "
                         "(0 = policy-only teacher). Omit to leave records as-is.")
    ap.add_argument("--policy-weight", type=float, default=None,
                    help="stamp this policy_weight on every merged record")
    args = ap.parse_args()

    base = os.path.abspath(args.base_dir)
    out = os.path.abspath(args.out_dir)
    source = os.path.abspath(args.source)
    subdir = args.subdir or os.path.basename(source.rstrip("/\\"))

    if base == out:
        ap.error("--out-dir must differ from --base-dir; the base is never modified")
    if os.path.exists(out):
        ap.error(f"{out} already exists -- refusing to overwrite a corpus")
    if not os.path.isdir(source):
        ap.error(f"no such source: {source}")

    print(f"copying {os.path.relpath(base, ROOT)} -> {os.path.relpath(out, ROOT)}")
    shutil.copytree(base, out)

    dest = os.path.join(out, subdir)
    if os.path.exists(dest):
        ap.error(f"{subdir}/ already exists in the base corpus")
    games, records = merge(source, dest, args.value_weight, args.policy_weight)
    print(f"merged {games} games / {records} records into {subdir}/")
    print(f"  value_weight  = {args.value_weight if args.value_weight is not None else 'unchanged'}")
    print(f"  policy_weight = {args.policy_weight if args.policy_weight is not None else 'unchanged'}")

    manifest_path = os.path.join(out, "corpus_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    manifest.setdefault("derived_from", os.path.relpath(base, ROOT).replace("\\", "/"))
    manifest.setdefault("merged_sources", []).append({
        "tool": "tools/merge_source.py",
        "source": os.path.relpath(source, ROOT).replace("\\", "/"),
        "as": subdir,
        "games": games,
        "records": records,
        "value_weight": args.value_weight,
        "policy_weight": args.policy_weight,
    })
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest updated: {os.path.relpath(manifest_path, ROOT)}")


if __name__ == "__main__":
    main()
