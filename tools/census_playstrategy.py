"""Census of Monster-chess games on playstrategy.org (public API, no auth).

Snowballs from the Monster leaderboard through opponents, deduping by game id.
Headers only (moves=false) so this stays light; the importer fetches moves later.
Writes ps_monster_census.json incrementally so it can be inspected while running.
"""
import collections
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "monster-chess-nn research (contact: perfpeng@gmail.com)",
      "Accept": "application/x-chess-pgn"}
BASE = "https://playstrategy.org/api/games/user/{}?perfType=monster&moves=false"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "playstrategy", "ps_monster_census.json")

SEEDS = ["woll", "oruro", "chessjun1234", "DrLastGladiator_777", "laitosto",
         "DimeRacaza", "G_MGST", "GMpenguinny2", "RacingKingsNews"]


def fetch(user):
    req = urllib.request.Request(BASE.format(urllib.parse.quote(user)), headers=UA)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:            # be a good citizen
                print("   429 — backing off 60s", flush=True)
                time.sleep(60)
                continue
            return ""
        except Exception as e:
            print(f"   retry after {type(e).__name__}", flush=True)
            time.sleep(5)
    return ""


def parse(pgn):
    out = []
    for block in pgn.split("[Event "):
        if not block.strip():
            continue
        h = dict(re.findall(r'\[(\w+)\s+"([^"]*)"\]', "[Event " + block))
        if h.get("Variant") == "Monster":
            out.append(h)
    return out


def isbot(h, p):
    return h.get(f"{p}Title") == "BOT"


def summarize(games, queried, queue_left):
    hh = [h for h in games.values() if not isbot(h, "P1") and not isbot(h, "P2")]
    vb = [h for h in games.values() if isbot(h, "P1") ^ isbot(h, "P2")]
    bb = len(games) - len(hh) - len(vb)
    print(f"\n=== users queried {queried} (queue left {queue_left}) — "
          f"DISTINCT MONSTER GAMES: {len(games)} ===", flush=True)
    print(f"human vs human : {len(hh)}", flush=True)
    print(f"human vs BOT   : {len(vb)}", flush=True)
    print(f"bot vs bot     : {bb}", flush=True)
    print("h-v-h results  :",
          dict(collections.Counter(h.get("Result") for h in hh)), flush=True)
    print("h-v-h by year  :",
          dict(sorted(collections.Counter(
              h.get("UTCDate", "")[:4] for h in hh).items())), flush=True)
    return hh


def main():
    games, seen, queue, reqs = {}, set(), list(SEEDS), 0
    if os.path.exists(OUT):                      # resume
        games = json.load(open(OUT, encoding="utf-8"))
        seen = {u.lower() for u in SEEDS}
        for h in games.values():
            for p in ("P1", "P2"):
                o = h.get(p, "")
                if o and o.lower() not in seen and o not in queue and not isbot(h, p):
                    queue.append(o)
        print(f"resumed: {len(games)} games, {len(queue)} users queued", flush=True)
    while queue and reqs < 400:
        u = queue.pop(0)
        if u.lower() in seen:
            continue
        seen.add(u.lower())
        reqs += 1
        t0 = time.time()
        hs = parse(fetch(u))
        print(f"[{reqs:3d}] {u:28s} {len(hs):6d} games "
              f"({time.time()-t0:.1f}s, queue {len(queue)})", flush=True)
        for h in hs:
            gid = h.get("Site", "").rsplit("/", 1)[-1]
            if gid:
                games[gid] = h
            for p in ("P1", "P2"):
                o = h.get(p, "")
                # Never queue BOT accounts: PST-Greedy-Tom alone has thousands of
                # games and they are all filtered out downstream anyway.
                if (o and o.lower() not in seen and o not in queue
                        and not isbot(h, p)):
                    queue.append(o)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(games, f, indent=1)
        time.sleep(1.5)

    summarize(games, reqs, len(queue))
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
