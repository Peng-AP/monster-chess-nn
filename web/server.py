"""Play Monster Chess against the engine in a browser: static page + JSON move API.

Standard library only (no new dependencies). Binds 127.0.0.1; public access
goes through a tunnel (see web/README.md). The browser keeps the game as a
list of UCI half-moves and sends it with every request; the server replays it
with the project's rules code, so illegal or tampered games are rejected and
restarts never lose a game in progress.

Endpoints:
  GET  /                      the game page (web/static/*)
  GET  /api/engines           engines a player may choose
  POST /api/state             {moves}                     -> position, legal half-moves, status
  POST /api/engine-move       {moves, engine, game_id}    -> the engine's whole turn, then state
  POST /api/record            {moves, engine, game_id, human_color, reason?} -> store an unfinished game

Finished games are stored once per game_id under data/raw/web_games/<date>/
(moves, result, engine identity, settings; no IP addresses).
"""
import argparse
from collections import defaultdict, deque
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/watch": ("watch.html", "text/html; charset=utf-8"),
                "/watch.html": ("watch.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/watch.js": ("watch.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8")}
GAMES = ROOT / "data/raw/web_games"
MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
MAX_PLIES = 800
MAX_BODY = 64 * 1024
OPENING_TEMP_PLIES = 16     # the gate's sampler: some variety early, then best play
OPENING_TEMPERATURE = 0.5
ENGINE_MOVES_PER_MINUTE = 60  # per client
MAX_WAITING = 8               # searches queued behind the GPU lock before "busy"
# Engine-vs-engine (Watch tab) overrides. Depth is a fixed menu so one visitor
# cannot ask for an arbitrarily long search on the shared GPU.
WATCH_SIMS = (16, 50, 200, 800, 1600, 3200, 6400, 12800)
MAX_TEMPERATURE = 2.0
MAX_TEMP_PLIES = 80

V29, V28 = "models/bootstrap_v29/best_value_net.pt", "models/bootstrap_v28/best_value_net.pt"
GEN53 = "models/candidates/bootstrap_main_gen_0053/arena_selected.pt"
V23, V21 = "models/bootstrap_v23/best_value_net.pt", "models/fresh_start_v21/best_value_net.pt"
V19, V17 = "models/fresh_start_v19/best_value_net.pt", "models/fresh_start_v17/best_value_net.pt"


def _level(elo, text, path, sims, default=False):
    return dict(label=f"{elo} · {text}", elo=elo, path=path, sims=sims, default=default)


# Difficulty levels, strongest first. Elo from the joint fit of the September 29
# round robin and the September 30 ladder (7,680 engine games, v21 = 1600):
# benchmarks/elo_ladder_20260930/ratings.json, docs/experiments/elo_rr/LADDER_RESULTS.md.
# gen53 (unpromoted) is placed relative to v29: +105 Elo in the October 5 top-group
# round robin (docs/experiments/gen53/MORNING_20261005.md), so 2450 + 105.
ENGINES = {
    "gen53": _level(2555, "gen53, strongest yet (experimental, not a release)", GEN53, 3200),
    "v29": _level(2450, "v29, full strength (current release)", V29, 3200, default=True),
    "v28": _level(2326, "v28, previous release", V28, 3200),
    "v29-800": _level(2325, "v29, 800 simulations", V29, 800),
    "v29-200": _level(2201, "v29, 200 simulations", V29, 200),
    "v29-12": _level(2015, "v29, 12 simulations (almost no search)", V29, 12),
    "v23": _level(1748, "v23", V23, 3200),
    "v21": _level(1600, "v21", V21, 3200),
    "v19": _level(1477, "v19", V19, 3200),
    "v17": _level(1255, "v17", V17, 3200),
    "v17-8": _level(1128, "v17, 8 simulations (easiest)", V17, 8),
}
# Saved games may name an engine that has since been renamed; v29 is the former gen51 deep-value.
ALIASES = {"gen51": "v29"}
DEFAULT_ENGINE = next(k for k, v in ENGINES.items() if v["default"])


class BadRequest(Exception):
    pass


# ----------------------------------------------------------------------------
# Rules layer: pure functions, testable without a GPU.

def replay(moves):
    """Rebuild a game from UCI half-moves, rejecting anything illegal."""
    import chess
    from monster_chess import MonsterChessGame
    from repetition import RepetitionTracker
    if not isinstance(moves, list) or len(moves) > MAX_PLIES:
        raise BadRequest("moves must be a list of at most %d half-moves" % MAX_PLIES)
    game, tracker = MonsterChessGame(), RepetitionTracker()
    tracker.record(game, 0)
    repeated = False
    for i, uci in enumerate(moves):
        if not isinstance(uci, str) or not MOVE_RE.match(uci):
            raise BadRequest(f"move {i + 1} is not a UCI move: {uci!r}")
        if repeated or game.is_terminal():
            raise BadRequest(f"move {i + 1} was played after the game ended")
        move = chess.Move.from_uci(uci)
        if move not in game.get_search_actions():
            raise BadRequest(f"move {i + 1} ({uci}) is illegal in this position")
        game.apply_search_action(move)
        repeated = tracker.record(game, i + 1)
    return game, repeated


def status(game, repeated):
    if repeated:
        return dict(over=True, result="draw", reason="repetition")
    if not game.is_terminal():
        return dict(over=False)
    result = game.get_result()
    if result >= 1:
        return dict(over=True, result="white", reason="king captured")
    if result <= -1:
        return dict(over=True, result="black", reason="king captured")
    return dict(over=True, result="draw", reason="turn limit")


def state_payload(moves):
    game, repeated = replay(moves)
    st = status(game, repeated)
    return dict(fen=game.fen(), turn="white" if game.is_white_turn else "black",
                half=2 if game.white_half_pending else 1, plies=len(moves),
                full_turn=game.turn_count // 2 + 1, status=st,
                legal=[] if st["over"] else sorted(m.uci() for m in game.get_search_actions()))


# ----------------------------------------------------------------------------
# Engine layer: one GPU evaluator per model, searches serialized by one lock.

class EnginePool:
    def __init__(self, names):
        from evaluation import NNEvaluator
        from match_evidence import file_hash
        self.models, self.lock, self.waiting = {}, threading.Lock(), 0
        self.count_lock = threading.Lock()
        evaluators = {}  # levels that share a network share one evaluator
        for name in names:
            spec = ENGINES[name]
            if spec["path"] not in evaluators:
                evaluators[spec["path"]] = (NNEvaluator(str(ROOT / spec["path"])), file_hash(ROOT / spec["path"]))
            evaluator, sha = evaluators[spec["path"]]
            self.models[name] = dict(spec, evaluator=evaluator, sha256=sha)
            print(f"loaded {name}: {spec['path']} at {spec['sims']} sims", flush=True)

    def search(self, name, moves, sims=None, temperature=None, temp_plies=None):
        """Play the engine's whole turn (both halves when it is White).

        sims / temperature / temp_plies override the level's depth and the
        opening sampler (the Watch tab); omitted, play is unchanged.
        """
        from benchmark import _FinisherEngine
        from native_mcts import NativeMCTS
        name = ALIASES.get(name, name)
        spec = self.models.get(name)
        if spec is None:
            raise BadRequest(f"unknown engine {name!r}")
        game, repeated = replay(moves)
        if status(game, repeated)["over"]:
            raise BadRequest("the game is already over")
        with self._slot():
            engine = _FinisherEngine(NativeMCTS(num_simulations=sims or spec["sims"], eval_fn=spec["evaluator"],
                                                root_noise=False, allow_early_stop=True,
                                                reuse_across_moves=True))
            played, side = [], game.is_white_turn
            while game.is_white_turn == side and not game.is_terminal():
                ply = len(moves) + len(played)
                limit = OPENING_TEMP_PLIES if temp_plies is None else temp_plies
                warm = OPENING_TEMPERATURE if temperature is None else temperature
                move = engine.get_best_action(game, temperature=warm if ply < limit else 0.0)[0]
                if move is None:
                    raise RuntimeError("engine returned no move in a live position")
                game.apply_search_action(move)
                played.append(move.uci())
                if side is False:
                    break  # Black moves once
        return played

    def _slot(self):
        pool = self

        class Slot:
            def __enter__(self):
                with pool.count_lock:
                    if pool.waiting >= MAX_WAITING:
                        raise Busy()
                    pool.waiting += 1
                pool.lock.acquire()

            def __exit__(self, *exc):
                pool.lock.release()
                with pool.count_lock:
                    pool.waiting -= 1
        return Slot()


class Busy(Exception):
    pass


# ----------------------------------------------------------------------------
# Game records.

_recorded = set()
_record_lock = threading.Lock()


def record_game(game_id, moves, engine, human_color, st, sha256, sims, reason=None):
    if not isinstance(game_id, str) or not re.fullmatch(r"[0-9a-f]{32}", game_id):
        return False
    with _record_lock:
        if game_id in _recorded:
            return False
        day = dt.date.today().isoformat()
        path = GAMES / day / f"{game_id}.json"
        if path.exists():
            _recorded.add(game_id)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            game_id=game_id, finished=dt.datetime.now().isoformat(timespec="seconds"),
            moves=moves, human_color=human_color, engine=engine, engine_sha256=sha256, sims=sims,
            status=st, end=reason or ("finished" if st.get("over") else "abandoned"),
            opening_temperature=[OPENING_TEMPERATURE, OPENING_TEMP_PLIES]), indent=1), encoding="utf-8")
        _recorded.add(game_id)
        return True


# ----------------------------------------------------------------------------
# HTTP.

def asset_version(name):
    import hashlib
    return hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:10]


def search_options(data):
    """Validated Watch-tab overrides from a request body; {} when none are given."""
    options = {}
    if data.get("sims") is not None:
        if data["sims"] not in WATCH_SIMS:
            raise BadRequest(f"sims must be one of {list(WATCH_SIMS)}")
        options["sims"] = int(data["sims"])
    if data.get("temperature") is not None:
        t = data["temperature"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= MAX_TEMPERATURE:
            raise BadRequest(f"temperature must be between 0 and {MAX_TEMPERATURE}")
        options["temperature"] = float(t)
    if data.get("temp_plies") is not None:
        n = data["temp_plies"]
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= MAX_TEMP_PLIES:
            raise BadRequest(f"temp_plies must be an integer between 0 and {MAX_TEMP_PLIES}")
        options["temp_plies"] = n
    return options


def versioned_page(html):
    """Point the page at content-hashed /style.css, /app.js and /watch.js URLs."""
    text = html.decode("utf-8")
    for name in ("style.css", "app.js", "watch.js"):
        text = text.replace(f'"/{name}"', f'"/{name}?v={asset_version(name)}"')
    return text.encode("utf-8")


_rate = defaultdict(deque)
_rate_lock = threading.Lock()


def allowed(client):
    now = time.monotonic()
    with _rate_lock:
        q = _rate[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= ENGINE_MOVES_PER_MINUTE:
            return False
        q.append(now)
        return True


class Handler(BaseHTTPRequestHandler):
    pool = None
    server_version = "MonsterChess/1"

    def log_message(self, fmt, *args):  # no client addresses in logs
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args if "%" in fmt else fmt))

    def client(self):
        # Behind cloudflared the peer is always localhost; use Cloudflare's header then.
        if self.client_address[0] in ("127.0.0.1", "::1"):
            return self.headers.get("CF-Connecting-IP", "local")
        return self.client_address[0]

    def send_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/engines":
            return self.send_json(200, [dict(id=k, label=v["label"], sims=v["sims"], default=v["default"],
                                             elo=v.get("elo"))
                                        for k, v in ENGINES.items() if k in self.pool.models])
        if path not in STATIC_FILES:
            return self.send_json(404, dict(error="not found"))
        name, ctype = STATIC_FILES[path]
        body = (STATIC / name).read_bytes()
        page = name.endswith(".html")
        if page:
            body = versioned_page(body)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page is never cached; assets are, under content-hashed URLs, so a
        # change reaches every visitor at once despite CDN and browser caches.
        self.send_header("Cache-Control", "no-store" if page else "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                raise BadRequest("missing or oversized body")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise BadRequest("body must be a JSON object")
            moves = data.get("moves", [])
            if self.path == "/api/state":
                return self.send_json(200, state_payload(moves))
            if self.path == "/api/engine-move":
                if not allowed(self.client()):
                    return self.send_json(429, dict(error="too many requests; slow down a little"))
                name = ALIASES.get(data.get("engine", DEFAULT_ENGINE), data.get("engine", DEFAULT_ENGINE))
                played = self.pool.search(name, moves, **search_options(data))
                state = state_payload(moves + played)
                # Engine-vs-engine games (Watch tab) are not human games and are not recorded.
                if state["status"]["over"] and not data.get("watch"):
                    spec = self.pool.models[name]
                    record_game(data.get("game_id"), moves + played, name, data.get("human_color"),
                                state["status"], spec["sha256"], spec["sims"])
                return self.send_json(200, dict(engine_moves=played, state=state))
            if self.path == "/api/record":
                name = ALIASES.get(data.get("engine", DEFAULT_ENGINE), data.get("engine", DEFAULT_ENGINE))
                spec = self.pool.models.get(name)
                if spec is None:
                    raise BadRequest(f"unknown engine {name!r}")
                state = state_payload(moves)
                reason = data.get("reason") if data.get("reason") in ("resigned", "new game", "finished") else None
                return self.send_json(200, dict(recorded=record_game(
                    data.get("game_id"), moves, name, data.get("human_color"), state["status"],
                    spec["sha256"], spec["sims"], reason)))
            return self.send_json(404, dict(error="not found"))
        except BadRequest as exc:
            return self.send_json(400, dict(error=str(exc)))
        except Busy:
            return self.send_json(503, dict(error="the engine is busy; try again in a moment"))
        except (ValueError, json.JSONDecodeError):
            return self.send_json(400, dict(error="malformed request"))
        except Exception as exc:  # never leak internals to visitors
            sys.stderr.write(f"internal error: {exc!r}\n")
            return self.send_json(500, dict(error="internal error"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--engines", default=",".join(ENGINES), help="comma-separated engine ids")
    args = ap.parse_args()
    os.chdir(ROOT)
    names = [n for n in args.engines.split(",") if n]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        ap.error(f"unknown engines: {unknown}")
    Handler.pool = EnginePool(names)
    # Warm up once so the first visitor does not pay for CUDA graph capture.
    for name in names:
        Handler.pool.search(name, [])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
