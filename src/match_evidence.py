"""Durable, provenance-checked match journals. No engine or scoring changes."""
import hashlib
from importlib.machinery import PathFinder
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def runtime_identity():
    """Conservative invalidation: source, installed native engine, rule flags."""
    paths = list((ROOT / "src").glob("*.py"))
    paths += [ROOT / "tools" / name for name in
              ("match.py", "gate_free.py", "free_gate_stats.py")]
    # Mirror native_mcts's local-extension search without importing the engine.
    # The old lookup missed native/monster_native.pyd in a fresh parent process,
    # because only worker-side adapter imports had added native/ to sys.path.
    if "monster_native" in sys.modules:
        native = getattr(sys.modules["monster_native"], "__spec__", None)
    else:
        search_path = list(sys.path)
        native_dir = str(ROOT / "native")
        if native_dir not in search_path:
            search_path.insert(0, native_dir)
        native = PathFinder.find_spec("monster_native", search_path)
    if native and native.origin:
        paths.append(Path(native.origin))
    return {"files": {str(p.resolve()): file_hash(p) for p in paths},
            "environment": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith(("MONSTER_", "CUDA_", "CUBLAS_"))}}


def model_identity(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)} if path else None


def task_id(task):
    return digest(task)


def read_rows(path, allow_partial_tail=False):
    """Ignore only an unterminated final record; malformed complete rows fail."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        if not line.endswith(b"\n") and allow_partial_tail and i == len(lines) - 1:
            break
        if line.strip():
            out.append(json.loads(line))
    return out


class MatchJournal:
    """Each completed task is fsynced before it counts as progress.

    Recovery validates the entire manifest and task identities before scheduling
    only missing tasks. A torn tail is archived, then removed from the journal;
    all complete records survive. Existing evidence is never replaced by default.
    """

    def __init__(self, path, manifest, tasks, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = self.path.with_suffix(self.path.suffix + ".manifest.json")
        self.manifest = {"schema_version": 1, "settings": manifest,
                         "tasks": [task_id(t) for t in tasks]}
        if len(set(self.manifest["tasks"])) != len(tasks):
            raise ValueError("duplicate match task IDs")
        if meta.exists():
            if not resume:
                raise FileExistsError(f"journal exists; use explicit resume: {self.path}")
            if json.loads(meta.read_text(encoding="utf-8")) != self.manifest:
                raise ValueError("match resume provenance mismatch")
        else:
            if self.path.exists():
                raise ValueError("journal without manifest cannot be resumed")
            atomic_json(meta, self.manifest)
        self.rows = read_rows(self.path, allow_partial_tail=True)
        expected = {task_id(t): t for t in tasks}
        self.done = set()
        for row in self.rows:
            ident = row.get("task_id")
            if ident not in expected or ident in self.done:
                raise ValueError("unknown or duplicate journal task")
            task = expected[ident]
            if (row["seed"], row["a_is_white"], row["pair"]) != (task[1], task[0], task[4]):
                raise ValueError("journal task metadata mismatch")
            if row["result_for_a"] not in (-1, -.5, 0, .5, 1) or row["plies"] < 0:
                raise ValueError("invalid journal game result")
            self.done.add(ident)
        if self.path.exists():
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):
                tail_start = data.rfind(b"\n") + 1
                # Preserve the torn bytes for diagnosis instead of silently losing them.
                suffix = 0
                while self.path.with_suffix(f".torn-{suffix}").exists():
                    suffix += 1
                self.path.with_suffix(f".torn-{suffix}").write_bytes(data[tail_start:])
                with self.path.open("r+b") as stream:
                    stream.truncate(tail_start)
        self.pending = [t for t in tasks if task_id(t) not in self.done]

    def append(self, row):
        if row["task_id"] in self.done or row["task_id"] not in self.manifest["tasks"]:
            raise ValueError("unexpected/duplicate completed task")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        self.done.add(row["task_id"])
