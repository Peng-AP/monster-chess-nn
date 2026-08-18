"""Replay a self-match artifact and export representative games to HTML.

The match summary intentionally stores only aggregates. This tool deterministically
replays its seed range with the recorded search settings, keeps complete FEN/move
histories, and selects an equal number of Black wins, Black losses, and draws.
"""
from __future__ import annotations

import argparse
import html
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import _apply, _build_engine  # noqa: E402
from monster_chess import MonsterChessGame  # noqa: E402


_ENGINES = {}


def classify_black_result(result: float) -> str:
    """Use the match protocol: only king captures win; cap endings draw."""
    if result >= 1:
        return "black_loss"
    if result <= -1:
        return "black_win"
    return "draw"


def game_seeds(match: dict) -> list[int]:
    games = int(match["games"])
    base = int(match["seed"])
    split = games // 2
    return ([base + index for index in range(split)]
            + [base + 1000 + index for index in range(games - split)])


def _init_worker(model: str, sims: int, engine: str,
                 c_puct: float, fpu_reduction: float,
                 policy_temperature: float) -> None:
    kwargs = {
        "engine": engine,
        "c_puct": c_puct,
        "fpu_reduction": fpu_reduction,
        "policy_temperature": policy_temperature,
    }
    _ENGINES["white"], _ = _build_engine(model, sims, **kwargs)
    _ENGINES["black"], _ = _build_engine(model, sims, **kwargs)


def _play_recorded(task: tuple[int, int, float]) -> dict:
    seed, opening_temp_plies, opening_temp = task
    random.seed(seed)
    game = MonsterChessGame()
    frames = [{
        "fen": game.fen(),
        "move": None,
        "actor": None,
        "white_half": None,
        "label": "Start position",
    }]
    plies = 0
    # Threefold repetition is a draw by rule (owner, 2026-08-16, on by default;
    # MONSTER_NO_REPETITION=1 disables). Every gate, match and generation has
    # applied it since; this exporter had not, so its games ran under
    # superseded rules -- shuffling positions ground to the 150-turn cap (225
    # plies, which is what every drawn game here measured to the ply) instead
    # of being drawn where the rule ends them. Worse than long games: a
    # position that repeats at ply 60 could continue and resolve DECISIVELY,
    # which shifts the colour tally, not just the lengths. Showcase games must
    # be played under the rules the engine is actually played under.
    from repetition import RepetitionTracker
    repetition = RepetitionTracker()
    repetition.record(game, 0)
    repeated = False
    while not game.is_terminal() and plies < 600:
        is_white = bool(game.is_white_turn)
        pending = bool(getattr(game, "white_half_pending", False))
        engine = _ENGINES["white" if is_white else "black"]
        temperature = opening_temp if plies < opening_temp_plies else 0.0
        action, _probabilities, _value = engine.get_best_action(
            game, temperature=temperature)
        if action is None:
            break
        uci = action.uci()
        _apply(game, action)
        plies += 1
        half = 2 if is_white and pending else (1 if is_white else None)
        actor = "White" if is_white else "Black"
        detail = f" ({'first' if half == 1 else 'second'} move)" if half else ""
        frames.append({
            "fen": game.fen(),
            "move": uci,
            "actor": actor,
            "white_half": half,
            "label": f"{plies}. {actor}{detail}: {uci}",
        })
        if repetition.record(game, plies):
            repeated = True
            frames[-1]["label"] += "  -- threefold repetition, drawn"
            break
    # A repetition is drawn by rule, so it overrides the cap's +-0.5 lean,
    # exactly as benchmark.play_one resolves it.
    result = (float(repetition.draw_result) if repeated
              else float(game.get_result()))
    category = classify_black_result(result)
    category_label = {
        "black_loss": "Black loss",
        "black_win": "Black win",
        "draw": "Draw",
    }[category]
    return {
        "seed": seed,
        "category": category,
        "category_label": category_label,
        "result_white_perspective": result,
        "plies": plies,
        "frames": frames,
    }


def select_games(games: list[dict], per_category: int) -> list[dict]:
    selected = []
    for category in ("black_loss", "black_win", "draw"):
        matches = [game for game in games if game["category"] == category]
        if len(matches) < per_category:
            raise RuntimeError(
                f"need {per_category} {category} games, found {len(matches)}")
        for ordinal, game in enumerate(matches[:per_category], 1):
            game = dict(game)
            game["title"] = f"{game['category_label']} {ordinal}"
            selected.append(game)
    return selected


def render_html(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(payload["title"])
    return rf"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --light:#e9d7b5; --dark:#8b5e3c; --accent:#63d2ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.4 system-ui,sans-serif; background:#11151b; color:#edf2f7; }}
main {{ max-width:1180px; margin:auto; padding:22px; }}
h1 {{ margin:0 0 4px; font-size:24px; }}
.sub {{ color:#aab6c5; margin-bottom:18px; }}
.layout {{ display:grid; grid-template-columns:minmax(320px,680px) minmax(260px,1fr); gap:20px; }}
.board {{ aspect-ratio:1; display:grid; grid-template-columns:repeat(8,1fr); border:3px solid #2c3440; }}
.sq {{ position:relative; display:grid; place-items:center; font-size:clamp(28px,6vw,64px); user-select:none; }}
.light {{ background:var(--light); }} .dark {{ background:var(--dark); }}
.last {{ box-shadow:inset 0 0 0 5px #f7d154; }}
.coord {{ position:absolute; font-size:10px; font-weight:700; opacity:.65; bottom:2px; right:3px; color:#18202a; }}
.panel {{ background:#1a2029; border:1px solid #303a48; border-radius:10px; padding:14px; min-height:0; }}
select,button {{ background:#263140; color:#fff; border:1px solid #45556a; border-radius:6px; padding:8px 10px; }}
select {{ width:100%; margin-bottom:10px; }} button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent); }}
.controls {{ display:flex; flex-wrap:wrap; gap:7px; margin:10px 0; }}
.status {{ min-height:48px; padding:9px; background:#11161d; border-radius:6px; }}
.meta {{ color:#aab6c5; margin:9px 0; }}
.moves {{ height:min(53vh,510px); overflow:auto; border-top:1px solid #303a48; margin-top:10px; padding-top:8px; }}
.move {{ display:block; width:100%; text-align:left; border:0; border-radius:4px; padding:5px 7px; background:transparent; color:#cbd5e1; }}
.move.active {{ background:#315066; color:#fff; }}
@media(max-width:760px) {{ .layout {{ grid-template-columns:1fr; }} .moves {{ height:260px; }} }}
</style>
</head>
<body><main>
<h1>{title}</h1>
<div class="sub">Nine deterministic replays from the promoted v20 self-match: three Black losses, three Black wins, and three draws.</div>
<div class="layout"><div id="board" class="board"></div><section class="panel">
<select id="game"></select>
<div id="meta" class="meta"></div><div id="status" class="status"></div>
<div class="controls"><button id="start">⏮</button><button id="prev">◀</button><button id="play">▶ Play</button><button id="next">▶</button><button id="end">⏭</button><button id="flip">Flip board</button></div>
<div id="moves" class="moves"></div>
</section></div></main>
<script>
const DATA={data};
const glyph={{K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'}};
let gameIndex=0, frameIndex=0, flipped=false, timer=null;
const board=document.querySelector('#board'), picker=document.querySelector('#game'), moves=document.querySelector('#moves');
function fenMap(fen){{const map={{}}, rows=fen.split(' ')[0].split('/'); rows.forEach((row,r)=>{{let f=0; for(const ch of row){{if(/\d/.test(ch)) f+=+ch; else map['abcdefgh'[f++]+(8-r)]=ch;}}}}); return map;}}
function render(){{const g=DATA.games[gameIndex], fr=g.frames[frameIndex], pieces=fenMap(fr.fen), last=fr.move&&fr.move.length>=4?[fr.move.slice(0,2),fr.move.slice(2,4)]:[]; board.innerHTML=''; const ranks=flipped?[1,2,3,4,5,6,7,8]:[8,7,6,5,4,3,2,1], files=flipped?'hgfedcba'.split(''):'abcdefgh'.split(''); for(const rank of ranks) for(const file of files){{const key=file+rank, s=document.createElement('div'); s.className='sq '+(((file.charCodeAt(0)-97+rank)%2)?'light':'dark')+(last.includes(key)?' last':''); s.innerHTML=(glyph[pieces[key]]||'')+`<span class="coord">${{key}}</span>`; board.appendChild(s);}} document.querySelector('#status').textContent=fr.label; document.querySelector('#meta').textContent=`${{g.category_label}} • seed ${{g.seed}} • ${{g.plies}} plies • frame ${{frameIndex}}/${{g.frames.length-1}}`; [...moves.children].forEach((node,i)=>node.classList.toggle('active',i===frameIndex)); moves.children[frameIndex]?.scrollIntoView({{block:'nearest'}});}}
function loadGame(index){{stop();gameIndex=+index;frameIndex=0;moves.innerHTML='';DATA.games[gameIndex].frames.forEach((fr,i)=>{{const b=document.createElement('button');b.className='move';b.textContent=fr.label;b.onclick=()=>{{frameIndex=i;render();}};moves.appendChild(b);}});render();}}
function stop(){{if(timer)clearInterval(timer);timer=null;document.querySelector('#play').textContent='▶ Play';}}
function togglePlay(){{if(timer){{stop();return;}}document.querySelector('#play').textContent='⏸ Pause';timer=setInterval(()=>{{const g=DATA.games[gameIndex];if(frameIndex>=g.frames.length-1){{stop();return;}}frameIndex++;render();}},650);}}
DATA.games.forEach((g,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{g.title}} — seed ${{g.seed}} (${{g.plies}} plies)`;picker.appendChild(o);}});
picker.onchange=e=>loadGame(e.target.value); document.querySelector('#start').onclick=()=>{{frameIndex=0;render();}};document.querySelector('#end').onclick=()=>{{frameIndex=DATA.games[gameIndex].frames.length-1;render();}};document.querySelector('#prev').onclick=()=>{{frameIndex=Math.max(0,frameIndex-1);render();}};document.querySelector('#next').onclick=()=>{{frameIndex=Math.min(DATA.games[gameIndex].frames.length-1,frameIndex+1);render();}};document.querySelector('#play').onclick=togglePlay;document.querySelector('#flip').onclick=()=>{{flipped=!flipped;render();}};document.onkeydown=e=>{{if(e.key==='ArrowLeft')document.querySelector('#prev').click();if(e.key==='ArrowRight')document.querySelector('#next').click();if(e.key===' '){{e.preventDefault();togglePlay();}}}};loadGame(0);
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    match_path = args.match if args.match.is_absolute() else ROOT / args.match
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    match = json.loads(match_path.read_text(encoding="utf-8"))
    search = match["search_a"]
    seeds = game_seeds(match)
    tasks = [(seed, int(match["opening_temp_plies"]), 0.5) for seed in seeds]
    workers = int(match.get("workers") or 1)
    with mp.Pool(
            workers,
            initializer=_init_worker,
            initargs=(str(model_path), int(match["sims"]), "native",
                      float(search["c_puct"]), float(search["fpu_reduction"]),
                      float(search["policy_temperature"]))) as pool:
        games = list(pool.imap(_play_recorded, tasks))

    counts = {key: sum(game["category"] == key for game in games)
              for key in ("black_loss", "black_win", "draw")}
    expected = {
        "black_loss": (int(match["a_as_white"]["wins"])
                       + int(match["a_as_black"]["losses"])),
        "black_win": (int(match["a_as_white"]["losses"])
                      + int(match["a_as_black"]["wins"])),
        "draw": (int(match["a_as_white"]["draws"])
                 + int(match["a_as_black"]["draws"])),
    }
    if counts != expected:
        raise RuntimeError(f"replay mismatch: got {counts}, expected {expected}")

    payload = {
        "title": "v20 self-play color examples",
        "source_match": str(match_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "selection": counts,
        "games": select_games(games, args.per_category),
    }
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    out_path.with_suffix(".html").write_text(
        render_html(payload), encoding="utf-8")
    print(json.dumps({
        "counts": counts,
        "selected_seeds": [game["seed"] for game in payload["games"]],
        "json": str(out_path.with_suffix('.json')),
        "html": str(out_path.with_suffix('.html')),
    }, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
