"""Durable cache of expensive reanalysis results, separate from raw training data."""
import json
import math
from pathlib import Path

from match_evidence import atomic_json, digest, file_hash, model_identity, read_rows, runtime_identity


class ReanalysisJournal:
    def __init__(self, path, args, all_rows, sampled, identity, implementation, resume=False):
        self.path = Path(path).resolve()
        source = Path(args.source_dir).resolve()
        if self.path.is_relative_to(source):
            raise ValueError("reanalysis journal must be outside the raw source tree")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = self.path.with_suffix(self.path.suffix + ".manifest.json")
        self.expected = {identity(item): item for item in sampled}
        if len(self.expected) != len(sampled):
            raise ValueError("duplicate reanalysis task identities")
        self.manifest = {"version": "reanalysis_journal_v1", "source_dir": str(source),
            "model": model_identity(args.model), "runtime": runtime_identity(),
            "implementation": {str(Path(p).resolve()): file_hash(p) for p in (implementation, __file__)},
            "search": {k: getattr(args, k) for k in
                       ("sample", "black_fraction", "simulations", "engine", "batch_size", "seed")},
            "source_hashes": {p: file_hash(source / p) for p in sorted({r["path"] for r in all_rows})},
            "tasks": [{"identity": ident, "record_sha256": digest(item["record"])}
                      for ident, item in self.expected.items()]}
        if self.meta.exists():
            if not resume:
                raise FileExistsError("reanalysis journal exists; use --resume")
            if json.loads(self.meta.read_text(encoding="utf-8")) != self.manifest:
                raise ValueError("reanalysis resume provenance/configuration mismatch")
        else:
            if self.path.exists():
                raise ValueError("reanalysis journal has no manifest")
            atomic_json(self.meta, self.manifest)
        self.results = read_rows(self.path, allow_partial_tail=True)
        self.done = set()
        for result in self.results:
            self._validate(result)
            self.done.add(result["identity"])
        if self.path.exists():
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):
                start = data.rfind(b"\n") + 1
                index = 0
                while self.path.with_suffix(f".torn-{index}").exists():
                    index += 1
                self.path.with_suffix(f".torn-{index}").write_bytes(data[start:])
                with self.path.open("r+b") as stream:
                    stream.truncate(start)
        self.pending = [item for ident, item in self.expected.items() if ident not in self.done]

    def _validate(self, result):
        ident = result["identity"]
        if ident not in self.expected or ident in self.done:
            raise ValueError("unknown or duplicate reanalysis result")
        item = self.expected[ident]
        record = item["record"]
        if (result["source_path"], result["source_line"], result["fen"], result["half"],
                result["current_player"], result["game_result"], result["plies_to_end"]) != (
                item["path"], item["line"], record["fen"], int(bool(record.get("half"))),
                record["current_player"], float(record.get("game_result", 0)), record.get("plies_to_end")):
            raise ValueError("reanalysis result source metadata mismatch")
        if not result.get("deep_policy") or "deep_value" not in result or "metrics" not in result:
            raise ValueError("incomplete reanalysis search result")
        probabilities = list(result["deep_policy"].values())
        if (not math.isfinite(result["deep_value"])
                or not all(math.isfinite(v) and v >= 0 for v in probabilities)
                or sum(probabilities) <= 0
                or not isinstance(result["metrics"].get("action_changed"), bool)
                or not all(math.isfinite(result["metrics"][k]) for k in
                           ("priority", "policy_js", "value_delta"))):
            raise ValueError("invalid numeric reanalysis search result")

    def append(self, result):
        import os
        self._validate(result)
        line = json.dumps(result, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.results.append(result)
        self.done.add(result["identity"])

    def output_manifest(self, directory, keep):
        if len(self.done) != len(self.expected):
            raise ValueError("cannot publish incomplete reanalysis search")
        directory = Path(directory)
        return {"search_manifest_sha256": file_hash(self.meta),
                "journal_sha256": file_hash(self.path), "keep": keep,
                "teacher_hashes": {p.name: file_hash(p) for p in sorted(directory.glob("teacher_*.jsonl"))},
                "summary_sha256": file_hash(directory / "reanalysis_summary.json")}

    def validate_output(self, directory, keep):
        directory = Path(directory)
        meta = directory / "reanalysis_evidence.json"
        if not meta.exists() or json.loads(meta.read_text(encoding="utf-8")) != self.output_manifest(directory, keep):
            raise ValueError("completed reanalysis output missing or changed")
        if len(list(directory.glob("teacher_*.jsonl"))) != min(keep, len(self.expected)):
            raise ValueError("completed reanalysis teacher count mismatch")
