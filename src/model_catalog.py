"""Notebook model discovery without loading networks or changing model pointers."""
import hashlib
import json
from pathlib import Path
import re


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _path(root, value):
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        return None
    return (root / value).resolve()


def _generation(path):
    match = re.search(r"(?:gen_|_v)(\d+)", path.parent.name)
    return int(match[1]) if match else -1


def discover_model_choices(project_root, model_dir=None):
    """Return widget (label, absolute path) pairs, newest evidence first.

    PASS labels describe saved reports, not a new gate certification. Reports
    with checkpoint hashes must match the current file before labeling it.
    Archived/rejected/rehearsal models are deliberately not normal play choices.
    """
    root = Path(project_root).resolve()
    models = Path(model_dir).resolve() if model_dir else root / "models"
    reports = list((root / "benchmarks").glob("gate_*.json"))
    reports += list((root / "iterations").glob("gen_*/reports/binding_gate.json"))
    reports += list((root / "benchmarks" / "sampled_gate").glob("*/report.json"))
    gated, hashes = {}, {}
    for report_path in reports:
        report = _load(report_path)
        if not isinstance(report, dict):
            continue
        path = _path(root, report.get("model"))
        if path is None or not path.is_file() or not path.is_relative_to(models):
            continue
        if any(p in {"archive", "rejected", "diagnostics"} or "rehearsal" in p
               for p in path.relative_to(models).parts):
            continue
        identity = report.get("model")
        expected = identity.get("sha256") if isinstance(identity, dict) else report.get("model_sha256")
        if expected:
            if path not in hashes:
                hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes[path] != expected:
                continue
        verdict = str(report.get("verdict", "UNKNOWN")).upper()
        legs = report.get("legs") or {}
        first = legs.get("vs_bar", {})
        bar = _path(root, report.get("bar_model"))
        if isinstance(report.get("bar"), dict):
            bar = _path(root, report["bar"])
        elif isinstance(report.get("bar"), str):
            first = legs.get(report["bar"], first)
        score = first.get("sampled", {}).get("score", first.get("a_score"))
        confirmed = bool(report.get("confirmed")) and verdict == "PASS"
        rank = 2 if confirmed else 1 if verdict == "PASS" else 0
        when = report_path.stat().st_mtime
        # Latest report wins for a checkpoint; an old PASS must not hide a later failure.
        if path in gated and gated[path][1] >= when:
            continue
        mark = "PASS+confirmed" if confirmed else verdict
        shown = f" {score:.2%}" if isinstance(score, (int, float)) else ""
        opponent = f" vs {bar.parent.name}" if bar else ""
        label = f"[reported {mark}{shown}{opponent}] {path.parent.name}/{path.stem}"
        gated[path] = (rank, when, label)

    choices = [("Heuristic (fixed anchor)", None)]
    seen = set()

    def add(label, path):
        path = path.resolve()
        if path not in seen:
            choices.append((label, str(path)))
            seen.add(path)

    # The promoted release named by the champion pointer leads the list, but
    # only while the file still matches the pointer's recorded hash.
    pointer = _load(models / "bootstrap" / "champion.json")
    champion = _path(root, pointer.get("checkpoint")) if isinstance(pointer, dict) else None
    if champion is not None and champion.is_file():
        expected = pointer.get("checkpoint_sha256")
        if not expected or hashlib.sha256(champion.read_bytes()).hexdigest() == expected:
            add(f"current release {champion.parent.name} (champion)", champion)

    ordered = sorted(gated.items(), key=lambda item: (item[1][0], _generation(item[0]), item[1][1]), reverse=True)
    for path, (rank, _, label) in ordered:
        if rank:
            add(label, path)
    candidates = []
    for directory in (models / "candidates").glob("*"):
        if not directory.is_dir() or any(s in directory.name for s in ("rehearsal", "smoke", "demo")):
            continue
        for name, tag in (("arena_selected.pt", "arena pick"),
                          ("screen_nominee_v3.pt", "v3 screen pick"),
                          ("screen_nominee.pt", "screen pick"),
                          ("best_value_net.pt", "training pick; not necessarily gated")):
            path = directory / name
            if path.is_file() and path not in gated:
                candidates.append((path, tag))
    candidates.sort(key=lambda item: (_generation(item[0]), item[0].stat().st_mtime), reverse=True)
    for path, tag in candidates:
        add(f"candidate {path.parent.name}/{path.stem} ({tag})", path)
    releases = [p for p in models.glob("*/best_value_net.pt")
                if p.parent.name not in {"archive", "rejected", "diagnostics", "candidates"}]
    # Historical fixed sparring anchor used by the notebook's curriculum deck.
    anchor = models / "rejected" / "fresh_start_v18_ramp" / "best_value_net.pt"
    if anchor.is_file():
        releases.append(anchor)
    for path in sorted(releases, key=lambda p: (_generation(p), p.stat().st_mtime), reverse=True):
        add(f"release/reference {path.parent.name}", path)
    for path, (rank, _, label) in ordered:
        if not rank:
            add(label, path)
    return choices
