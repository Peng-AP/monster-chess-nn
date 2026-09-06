"""OS-released exclusive lease for match/gate worker campaigns."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os
import threading

_guard = threading.RLock()
_depth = 0
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "logs" / "match_workers.lock"


@contextmanager
def worker_lease(path=None):
    global _depth
    with _guard:
        if _depth:
            _depth += 1
            try:
                yield
            finally:
                _depth -= 1
            return
        path = Path(path or DEFAULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+b")
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("another match/gate worker campaign is active") from exc
        _depth = 1
        try:
            yield
        finally:
            _depth = 0
            stream.close()  # also released by OS on process crash


def exclusive_workers(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with worker_lease():
            return fn(*args, **kwargs)
    return wrapped
