"""Web server contracts without a GPU: rules replay, status, records, HTTP handling."""
import http.client
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "web")]

import server as web


def first_legal(moves):
    return web.state_payload(moves)["legal"][0]


def test_start_position_is_white_first_half_with_legal_moves():
    st = web.state_payload([])
    assert st["turn"] == "white" and st["half"] == 1 and not st["status"]["over"]
    assert "e2e4" in st["legal"] and all(web.MOVE_RE.match(m) for m in st["legal"])


def test_white_moves_twice_then_black_once():
    st = web.state_payload(["e2e4"])
    assert st["turn"] == "white" and st["half"] == 2
    st = web.state_payload(["e2e4", "d2d4"])
    assert st["turn"] == "black"
    st = web.state_payload(["e2e4", "d2d4", "d7d5"])
    assert st["turn"] == "white" and st["half"] == 1 and st["full_turn"] == 2  # one full round played


@pytest.mark.parametrize("moves", [["e2e5"], ["e7e5"], ["zz"], "e2e4", ["e2e4"] * 900, [1]])
def test_illegal_or_malformed_games_are_rejected(moves):
    with pytest.raises(web.BadRequest):
        web.replay(moves)


def test_a_finished_game_reports_its_result_and_accepts_no_further_moves():
    # White shuffles its king, Black its knight: the position repeats.
    cycle = ["e1d1", "d1e1", "g8f6", "e1d1", "d1e1", "f6g8"]
    st = web.state_payload(cycle * 2)
    assert st["status"] == dict(over=True, result="draw", reason="repetition") and st["legal"] == []
    with pytest.raises(web.BadRequest):
        web.replay(cycle * 2 + ["e2e4"])


def test_records_are_written_once_per_game(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "GAMES", tmp_path)
    monkeypatch.setattr(web, "_recorded", set())
    gid = "ab" * 16
    st = dict(over=True, result="draw", reason="repetition")
    assert web.record_game(gid, ["e2e4"], "v28", "black", st, "sha", 3200)
    assert not web.record_game(gid, ["e2e4"], "v28", "black", st, "sha", 3200)
    assert not web.record_game("../../evil", [], "v28", "black", st, "sha", 3200)
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["moves"] == ["e2e4"]
    assert "ip" not in files[0].read_text().lower()


def test_levels_are_rated_ordered_and_have_one_default():
    elos = [spec["elo"] for spec in web.ENGINES.values()]
    assert elos == sorted(elos, reverse=True)
    assert [k for k, v in web.ENGINES.items() if v["default"]] == ["v29"]
    for spec in web.ENGINES.values():
        assert spec["label"].startswith(str(spec["elo"])) and (ROOT / spec["path"]).exists()


class FakePool:
    """Plays the first legal half-move(s) for its side; no GPU."""
    models = {"v29": dict(sha256="x", sims=8)}

    calls = []

    def search(self, name, moves, **options):
        FakePool.calls.append(options)
        name = web.ALIASES.get(name, name)
        if name not in self.models:
            raise web.BadRequest("unknown engine")
        played = []
        side = web.state_payload(moves)["turn"]
        while True:
            st = web.state_payload(moves + played)
            if st["status"]["over"] or st["turn"] != side:
                return played
            played.append(st["legal"][0])
            if side == "black":
                return played


@pytest.fixture()
def http_server(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "GAMES", tmp_path)
    monkeypatch.setattr(web.Handler, "pool", FakePool())
    httpd = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()


def request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(method, path, body=None if body is None else json.dumps(body),
                 headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    return res.status, res.read()


def test_http_page_state_and_engine_turn(http_server):
    status, body = request(http_server, "GET", "/")
    assert status == 200 and b"Monster Chess" in body
    status, body = request(http_server, "GET", "/api/engines")
    assert status == 200 and json.loads(body)[0]["id"] == "v29"
    status, body = request(http_server, "POST", "/api/engine-move", dict(moves=[], engine="v29"))
    data = json.loads(body)
    assert status == 200 and len(data["engine_moves"]) == 2 and data["state"]["turn"] == "black"


def test_http_rejects_traversal_bad_moves_and_unknown_engines(http_server):
    assert request(http_server, "GET", "/../server.py")[0] == 404
    assert request(http_server, "GET", "/static/../../README.md")[0] == 404
    assert request(http_server, "POST", "/api/state", dict(moves=["e2e5"]))[0] == 400
    assert request(http_server, "POST", "/api/engine-move", dict(moves=[], engine="nope"))[0] == 400
    assert request(http_server, "POST", "/api/state", [1, 2])[0] == 400


def test_page_references_content_hashed_assets_and_is_not_cached(http_server):
    conn = http.client.HTTPConnection("127.0.0.1", http_server, timeout=30)
    conn.request("GET", "/")
    res = conn.getresponse()
    body = res.read().decode()
    assert res.getheader("Cache-Control") == "no-store"
    assert f'/style.css?v={web.asset_version("style.css")}' in body
    assert f'/app.js?v={web.asset_version("app.js")}' in body
    conn.request("GET", f'/app.js?v={web.asset_version("app.js")}')
    res = conn.getresponse(); res.read()
    assert res.status == 200 and "immutable" in res.getheader("Cache-Control")


def test_search_options_are_validated():
    assert web.search_options({}) == {}
    assert web.search_options(dict(sims=800, temperature=0, temp_plies=0)) == dict(sims=800, temperature=0.0, temp_plies=0)
    for bad in (dict(sims=999), dict(sims=10 ** 9), dict(temperature=-0.1), dict(temperature=5),
                dict(temperature=True), dict(temp_plies=1.5), dict(temp_plies=81), dict(temp_plies=-1)):
        with pytest.raises(web.BadRequest):
            web.search_options(bad)


def test_watch_page_options_reach_the_engine_and_watch_games_are_not_recorded(http_server, tmp_path):
    status, body = request(http_server, "GET", "/watch")
    assert status == 200 and b"Watch engines" in body and f'/watch.js?v={web.asset_version("watch.js")}'.encode() in body
    FakePool.calls.clear()
    status, _ = request(http_server, "POST", "/api/engine-move",
                        dict(moves=[], engine="v29", watch=True, sims=200, temperature=1.0, temp_plies=4))
    assert status == 200 and FakePool.calls[-1] == dict(sims=200, temperature=1.0, temp_plies=4)
    assert request(http_server, "POST", "/api/engine-move", dict(moves=[], engine="v29", sims=7))[0] == 400
    # The engine's move completes a repetition. A watch game is not written; a played game is.
    cycle = ["e1d1", "d1e1", "g8f6", "e1d1", "d1e1", "f6g8"]
    moves = (cycle * 2)[:-1]

    class Repeats(FakePool):
        def search(self, name, moves, **options):
            return ["f6g8"]

    web.Handler.pool = Repeats()
    status, body = request(http_server, "POST", "/api/engine-move",
                           dict(moves=moves, engine="v29", watch=True, game_id="ab" * 16, human_color="white"))
    assert status == 200 and json.loads(body)["state"]["status"]["over"] and not list(tmp_path.rglob("*.json"))
    status, _ = request(http_server, "POST", "/api/engine-move",
                        dict(moves=moves, engine="v29", game_id="cd" * 16, human_color="white"))
    assert status == 200 and len(list(tmp_path.rglob("*.json"))) == 1
