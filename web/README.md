# Play Monster Chess in a browser

`web/server.py` serves a game page and a small JSON API from this PC. It uses
the project's own rules code and GPU engine, and only Python's standard
library (no extra packages).

## Run it

```powershell
py -3 tools/runs.py start --name web_server -- py -3 -u web/server.py --port 8765
```

Open <http://127.0.0.1:8765>. Stop with
`py -3 tools/runs.py stop --name web_server`.

- Engines: **v28** (current release, default) and **gen51 deep-value**
  (experimental, unpromoted), both at 3,200 simulations. Change the list in
  `ENGINES` in `server.py`. `--engines v28` serves only v28.
- An engine turn takes about 0.1–0.4 s on the RTX 5060 Ti. Searches run one
  at a time behind a lock; at most 8 can queue before visitors see "busy".
  Each visitor may request 60 engine moves a minute.
- The first 16 half-moves are sampled at temperature 0.5 (the gate's rule) so
  games vary; after that the engine plays its best move.
- **Heavy research jobs and the server share the GPU.** Stop the server
  during a generation run, or expect slower engine replies and slightly slower
  research.

## How it works

- The browser keeps the game as a list of UCI half-moves (White's turn is two
  of them) and sends the whole list with each request.
- The server replays it with `MonsterChessGame`, rejects anything illegal, and
  answers with the legal half-moves or the engine's full turn.
- A server restart never loses a game; the page also saves it in the
  browser's local storage.
- Rules are this project's (king capture wins; draw by repetition or the
  150-turn limit), not playstrategy.org's.
- Finished, resigned and abandoned games are saved once each to
  `data/raw/web_games/<date>/<game id>.json`: moves, result, engine identity
  and settings. No IP addresses are stored, and access logs omit client
  addresses.

## Public access (Cloudflare Tunnel, own domain)

The server listens on 127.0.0.1 only. A Cloudflare Tunnel makes an outbound
connection from this PC, so no router ports are opened and the home IP is not
exposed. One-time setup (the domain must use Cloudflare's DNS):

```powershell
winget install --id Cloudflare.cloudflared      # Windows may ask for approval
cloudflared tunnel login                          # opens a browser: pick the domain
cloudflared tunnel create monster-chess
cloudflared tunnel route dns monster-chess chess.YOURDOMAIN
```

Create `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: monster-chess
credentials-file: C:\Users\perfp\.cloudflared\<TUNNEL-ID>.json
ingress:
  - hostname: chess.YOURDOMAIN
    service: http://localhost:8765
  - service: http_status:404
```

Run the tunnel (alongside the server):

```powershell
py -3 tools/runs.py start --name web_tunnel -- cloudflared tunnel run monster-chess
```

The server then reads each visitor's address from Cloudflare's
`CF-Connecting-IP` header, for rate limiting only.

## Tests

```powershell
py -3 -m pytest tests/test_web_server.py -q
```
