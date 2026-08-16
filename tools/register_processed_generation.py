"""Import an audited historical generation into the bootstrap replay registry.

This is for generation drivers that predate ``src/iterate.py`` registration.
It never rewrites a processed corpus and refuses to replace an existing
generation entry; normal new runs are registered by iterate.py itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import iterate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", default="iterations")
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--run-state", required=True,
                        help="historical state or generation summary artifact")
    args = parser.parse_args()
    if args.generation <= 0:
        parser.error("--generation must be positive")
    run_root = iterate._absolute(args.run_root)
    processed = iterate._absolute(args.processed_dir)
    incumbent = iterate._absolute(args.incumbent)
    run_state = iterate._absolute(args.run_state)
    if not incumbent.is_file():
        raise FileNotFoundError(incumbent)
    if not run_state.is_file():
        raise FileNotFoundError(run_state)

    registry_path = iterate._accepted_registry_path(run_root)
    registry = iterate._load_json(registry_path, {"entries": []})
    if any(int(row.get("generation", -1)) == args.generation
           for row in registry.get("entries", [])):
        raise RuntimeError(
            f"generation {args.generation} is already registered; refusing "
            "to replace it")
    state = {
        "generation": args.generation,
        "incumbent": iterate._rel(incumbent),
        "incumbent_sha256": iterate._sha256(incumbent),
        "paths": {"state": iterate._rel(run_state)},
    }
    acceptance = iterate._accept_generation_data(
        state, {"new_processed": processed}, run_root)
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
