// Monster Chess client. The server is the rules authority; the page draws, lets the
// player move (drag or click), annotate (right-drag arrows, right-click squares) and
// step through the game (arrow keys), and asks the server for legality and engine turns.
"use strict";

// U+FE0E asks for the text (not emoji) presentation; phones otherwise draw the pawn as an emoji.
const GLYPH = Object.fromEntries(Object.entries({ k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" })
  .map(([k, v]) => [k, v + "︎"]));
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1";
const STORE = "monster-chess-game-v1";
const THEME = "monster-chess-theme";
const MARK_COLORS = { green: "#15781b", red: "#c0392b", blue: "#1f5fbf", yellow: "#e6a700" };
const $ = (id) => document.getElementById(id);

let game = null;          // { id, engine, human, moves: [] }
let state = null;         // latest /api/state payload (the live position)
let selected = null;      // selected from-square (live position only)
let busy = false;         // engine request in flight
let view = null;          // ply being reviewed; null = live position
let resultHidden = false; // player dismissed the game-over card to review
let boardMap = {};        // square -> piece letter for the drawn position
const fenCache = new Map([[0, START_FEN]]);
const arrows = new Map();  // "e2e4" -> colour
const squares = new Map(); // "e4" -> colour
let drag = null;           // left-button press in progress
let rightStart = null;     // right-button press in progress

// --- helpers --------------------------------------------------------------------

function newId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: "unexpected server response" }));
  if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);
  return data;
}

function save() {
  try { localStorage.setItem(STORE, JSON.stringify(game)); } catch (_) { /* storage may be unavailable */ }
}

function load() {
  try { return JSON.parse(localStorage.getItem(STORE)); } catch (_) { return null; }
}

function parseFen(fen) {
  const board = {};
  fen.split(" ")[0].split("/").forEach((row, r) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += Number(ch); continue; }
      board["abcdefgh"[f] + (8 - r)] = ch;
      f += 1;
    }
  });
  return board;
}

const plies = () => (game ? game.moves.length : 0);
const viewPly = () => (view === null ? plies() : view);
const isLive = () => viewPly() === plies();
const flipped = () => Boolean(game && game.human === "black");
const humanToMove = () => Boolean(state && game && !state.status.over && state.turn === game.human);
const isOwn = (piece) => Boolean(piece) && ((piece === piece.toUpperCase()) === (game.human === "white"));
const legalFrom = (from) => (state ? state.legal.filter((m) => m.slice(0, 2) === from) : []);

// Positions of earlier plies are fetched from the server once, then cached.
async function fenAt(ply) {
  if (ply === plies() && state) return state.fen;
  if (!fenCache.has(ply)) {
    const st = await api("/api/state", { moves: game.moves.slice(0, ply) });
    fenCache.set(ply, st.fen);
  }
  return fenCache.get(ply);
}

// --- drawing ---------------------------------------------------------------------

async function show() {
  let fen = START_FEN;
  if (game) {
    const ply = viewPly();
    try { fen = await fenAt(ply); } catch (err) { renderStatus(err.message, "error"); return; }
    if (ply !== viewPly()) return;  // the player stepped on while this position loaded
  }
  drawBoard(fen);
}

function squareXY(sq) {
  const file = "abcdefgh".indexOf(sq[0]);
  const rank = Number(sq[1]);
  return flipped() ? { x: 7 - file, y: rank - 1 } : { x: file, y: 8 - rank };
}

function drawBoard(fen) {
  boardMap = parseFen(fen);
  const ply = viewPly();
  const last = game && ply > 0 ? game.moves[ply - 1] : null;
  const live = isLive();
  const targets = new Set(selected && live && humanToMove() ? legalFrom(selected).map((m) => m.slice(2, 4)) : []);
  const el = $("board");
  el.innerHTML = "";
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = "abcdefgh"[flipped() ? 7 - col : col];
      const rank = flipped() ? row + 1 : 8 - row;
      const sq = file + rank;
      const cell = document.createElement("div");
      cell.className = `sq ${(col + row) % 2 ? "dark" : "light"}`;
      cell.dataset.sq = sq;
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", sq + (boardMap[sq] ? " " + boardMap[sq] : ""));
      if (last && (last.slice(0, 2) === sq || last.slice(2, 4) === sq)) cell.classList.add("last");
      if (sq === selected && live) cell.classList.add("selected");
      if (targets.has(sq)) cell.classList.add("target", ...(boardMap[sq] ? ["capture"] : []));
      if (drag && drag.ghost && drag.from === sq) cell.classList.add("dragging");
      const piece = boardMap[sq];
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece === piece.toUpperCase() ? "w" : "b"}`;
        span.textContent = GLYPH[piece.toLowerCase()];
        cell.appendChild(span);
      }
      if (row === 7) cell.insertAdjacentHTML("beforeend", `<span class="coord file">${file}</span>`);
      if (col === 0) cell.insertAdjacentHTML("beforeend", `<span class="coord rank">${rank}</span>`);
      el.appendChild(cell);
    }
  }
  $("board").classList.toggle("reviewing", !live);
  drawMarks();
  renderMoves();
  renderStatus();
  renderResult();
  $("resign").disabled = !game || !state || state.status.over;
  $("copy").disabled = !game || !game.moves.length;
  $("first").disabled = $("prev").disabled = !game || ply === 0;
  $("next").disabled = $("last").disabled = !game || live;
}

function drawMarks() {
  const defs = Object.entries(MARK_COLORS).map(([name, c]) =>
    `<marker id="head-${name}" viewBox="0 0 10 10" refX="4" refY="5" markerWidth="2.6" markerHeight="2.6"
       orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`).join("");
  const rects = [...squares].map(([sq, color]) => {
    const { x, y } = squareXY(sq);
    return `<rect x="${x}" y="${y}" width="1" height="1" fill="${MARK_COLORS[color]}" opacity=".5"/>`;
  }).join("");
  const lines = [...arrows].map(([move, color]) => {
    const a = squareXY(move.slice(0, 2));
    const b = squareXY(move.slice(2, 4));
    const x1 = a.x + 0.5, y1 = a.y + 0.5, x2 = b.x + 0.5, y2 = b.y + 0.5;
    const len = Math.hypot(x2 - x1, y2 - y1);
    const cut = 0.38;  // leave room for the arrowhead inside the target square
    const ex = x2 - ((x2 - x1) / len) * cut, ey = y2 - ((y2 - y1) / len) * cut;
    return `<line x1="${x1}" y1="${y1}" x2="${ex}" y2="${ey}" stroke="${MARK_COLORS[color]}" stroke-width=".17"
              stroke-linecap="round" opacity=".82" marker-end="url(#head-${color})"/>`;
  }).join("");
  $("marks").innerHTML = `<defs>${defs}</defs>${rects}${lines}`;
}

function renderMoves() {
  // White's turn is two half-moves; Black replies with one. Each half-move is clickable.
  const list = $("moves");
  list.innerHTML = "";
  if (!game) return;
  const current = viewPly();
  for (let i = 0; i < game.moves.length; i += 3) {
    const li = document.createElement("li");
    for (let j = i; j < Math.min(i + 3, game.moves.length); j++) {
      const span = document.createElement("span");
      span.className = "mv" + (j === i + 2 ? " black" : "") + (j + 1 === current ? " current" : "");
      span.textContent = game.moves[j];
      span.addEventListener("click", () => setView(j + 1));
      li.appendChild(span);
    }
    list.appendChild(li);
  }
  const cur = list.querySelector(".current");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}

function outcome() {
  const st = state && state.status;
  if (!game || !st || !st.over) return null;
  const reasons = { "king captured": "King captured", repetition: "By repetition",
                    "turn limit": "150-turn limit reached", resignation: "By resignation" };
  const reason = reasons[st.reason] || st.reason;
  if (st.result === "draw") return { title: "Draw", kind: "draw", reason };
  const won = st.result === game.human;
  return { title: won ? "You win!" : "You lost", kind: won ? "win" : "loss", reason };
}

function renderResult() {
  const box = $("result");
  const out = outcome();
  if (!out || !isLive() || resultHidden) { box.hidden = true; return; }
  box.className = `result ${out.kind}`;
  box.innerHTML = `<div class="card"><div class="title">${out.title}</div><div class="reason">${out.reason}</div>
    <div class="buttons"><button type="button" data-act="new" class="primary">New game</button>
    <button type="button" data-act="review">Review game</button></div></div>`;
  box.querySelector('[data-act="new"]').addEventListener("click", startGame);
  box.querySelector('[data-act="review"]').addEventListener("click", () => { resultHidden = true; renderResult(); });
  box.hidden = false;
}

function renderStatus(message, kind) {
  const el = $("status");
  el.className = "status" + (kind ? " " + kind : "");
  if (message) { el.textContent = message; return; }
  if (!game || !state) { el.textContent = "Choose an engine and a colour, then start a new game."; return; }
  if (!isLive()) {
    el.classList.add("review");
    el.textContent = `Reviewing half-move ${viewPly()} of ${plies()}. ← → to step, End to return.`;
    return;
  }
  const out = outcome();
  if (out) { el.classList.add("over", out.kind); el.textContent = `${out.title} — ${out.reason.toLowerCase()}.`; return; }
  if (busy) { el.classList.add("thinking"); el.textContent = "Engine is thinking"; return; }
  const who = state.turn === game.human ? "Your move" : "Engine to move";
  const half = state.turn === "white" ? ` — White move ${state.half} of 2` : "";
  el.textContent = `${who}${half}. Turn ${state.full_turn}.`;
}

// --- navigation ------------------------------------------------------------------

function setView(ply) {
  if (!game) return;
  const target = Math.max(0, Math.min(plies(), ply));
  if (target !== viewPly() && outcome()) resultHidden = true;  // stepping through a finished game
  view = target === plies() ? null : target;
  selected = null;
  show();
}

function onKey(e) {
  if (!game || e.altKey || e.ctrlKey || e.metaKey || e.target.closest("select, input, textarea")) return;
  const targets = { ArrowLeft: viewPly() - 1, ArrowRight: viewPly() + 1, Home: 0, ArrowUp: 0,
                    End: plies(), ArrowDown: plies() };
  if (!(e.key in targets)) return;
  e.preventDefault();
  setView(targets[e.key]);
}

// --- pointer input: drag or click to move, right button to annotate ----------------

function squareAt(x, y) {
  const r = $("board").getBoundingClientRect();
  if (x < r.left || y < r.top || x >= r.right || y >= r.bottom) return null;
  const col = Math.floor(((x - r.left) / r.width) * 8);
  const row = Math.floor(((y - r.top) / r.height) * 8);
  const file = "abcdefgh"[flipped() ? 7 - col : col];
  const rank = flipped() ? row + 1 : 8 - row;
  return file + rank;
}

function markColor(e) {
  return e.shiftKey ? "red" : e.altKey ? "blue" : e.ctrlKey || e.metaKey ? "yellow" : "green";
}

function toggle(map, key, color) {
  if (map.get(key) === color) map.delete(key); else map.set(key, color);
}

function onDown(e) {
  const sq = squareAt(e.clientX, e.clientY);
  if (e.button === 2) { rightStart = sq ? { sq, color: markColor(e) } : null; e.preventDefault(); return; }
  if (e.button !== 0) return;
  if (arrows.size || squares.size) { arrows.clear(); squares.clear(); drawMarks(); }
  if (!sq || !game || !state || busy || state.status.over) return;
  if (!isLive()) { setView(plies()); return; }  // a click on a reviewed position returns to the game
  if (!humanToMove()) { engineTurn(); return; }  // retry after a failed engine request
  e.preventDefault();
  if (selected && sq !== selected && legalFrom(selected).some((m) => m.slice(2, 4) === sq)) {
    attempt(selected, sq);
    return;
  }
  if (isOwn(boardMap[sq]) && legalFrom(sq).length) {
    drag = { from: sq, wasSelected: selected === sq, x0: e.clientX, y0: e.clientY, ghost: null };
    selected = sq;
  } else {
    selected = null;
  }
  drawBoard(state.fen);
}

function onMove(e) {
  if (!drag) return;
  if (!drag.ghost) {
    if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < 4) return;
    const size = $("board").getBoundingClientRect().width / 8;
    const piece = boardMap[drag.from];
    const ghost = document.createElement("div");
    ghost.className = `ghost piece ${piece === piece.toUpperCase() ? "w" : "b"}`;
    ghost.textContent = GLYPH[piece.toLowerCase()];
    ghost.style.fontSize = `${size * 0.84}px`;
    document.body.appendChild(ghost);
    drag.ghost = ghost;
    const cell = $("board").querySelector(`[data-sq="${drag.from}"]`);
    if (cell) cell.classList.add("dragging");
  }
  drag.ghost.style.left = `${e.clientX}px`;
  drag.ghost.style.top = `${e.clientY}px`;
  for (const el of $("board").querySelectorAll(".hover")) el.classList.remove("hover");
  const over = squareAt(e.clientX, e.clientY);
  const cell = over && $("board").querySelector(`[data-sq="${over}"].target`);
  if (cell) cell.classList.add("hover");
}

function onUp(e) {
  if (e.button === 2) {
    const sq = squareAt(e.clientX, e.clientY);
    if (rightStart && sq) {
      if (sq === rightStart.sq) toggle(squares, sq, rightStart.color);
      else toggle(arrows, rightStart.sq + sq, rightStart.color);
      drawMarks();
    }
    rightStart = null;
    return;
  }
  if (e.button !== 0 || !drag) return;
  const d = drag;
  drag = null;
  if (d.ghost) {
    d.ghost.remove();
    const sq = squareAt(e.clientX, e.clientY);
    if (sq && sq !== d.from && legalFrom(d.from).some((m) => m.slice(2, 4) === sq)) { attempt(d.from, sq); return; }
    if (sq !== d.from) selected = null;  // dropped off target: put the piece back
  } else if (d.wasSelected) {
    selected = null;  // a second click on the selected piece deselects it
  }
  drawBoard(state.fen);
}

function onCancel() {
  if (drag && drag.ghost) drag.ghost.remove();
  drag = null;
  rightStart = null;
  if (state) drawBoard(state.fen);
}

function attempt(from, to) {
  const options = legalFrom(from).filter((m) => m.slice(2, 4) === to);
  if (options.length === 1) play(options[0]);
  else if (options.length > 1) choosePromotion(options);
}

function choosePromotion(options) {
  const box = $("promotion");
  box.innerHTML = "";
  const white = game.human === "white";
  for (const m of options) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = GLYPH[m[4]];
    btn.style.color = white ? "#fff" : "#111";
    btn.addEventListener("click", () => { box.hidden = true; play(m); });
    box.appendChild(btn);
  }
  box.hidden = false;
}

// --- game flow -------------------------------------------------------------------

async function play(move) {
  selected = null;
  const moves = game.moves.concat([move]);
  try {
    state = await api("/api/state", { moves });
    game.moves = moves;
    view = null;
    fenCache.set(moves.length, state.fen);
    save();
    drawBoard(state.fen);
    if (state.status.over) {  // the player's own move ended the game
      api("/api/record", { moves: game.moves, engine: game.engine, game_id: game.id,
                           human_color: game.human, reason: "finished" }).catch(() => {});
    } else if (state.turn !== game.human) {
      await engineTurn();
    }
  } catch (err) {
    renderStatus(err.message, "error");
  }
}

async function engineTurn() {
  busy = true;
  renderStatus();
  try {
    const res = await api("/api/engine-move", {
      moves: game.moves, engine: game.engine, game_id: game.id, human_color: game.human,
    });
    game.moves = game.moves.concat(res.engine_moves);
    state = res.state;
    fenCache.set(game.moves.length, state.fen);
    view = null;
    save();
  } catch (err) {
    busy = false;
    await show();
    renderStatus(err.message + " — click the board or reload to retry.", "error");
    return;
  }
  busy = false;
  drawBoard(state.fen);
}

function resetView() {
  selected = null; view = null; resultHidden = false;
  fenCache.clear(); fenCache.set(0, START_FEN);
  arrows.clear(); squares.clear();
}

async function startGame() {
  if (game && state && !state.status.over && game.moves.length) {
    api("/api/record", { moves: game.moves, engine: game.engine, game_id: game.id,
                         human_color: game.human, reason: "new game" }).catch(() => {});
  }
  game = { id: newId(), engine: $("engine").value, human: $("color").value, moves: [] };
  resetView();
  save();
  await resume();
}

async function resume() {
  try {
    state = await api("/api/state", { moves: game.moves });
  } catch (err) {
    renderStatus("Could not restore the saved game: " + err.message, "error");
    game = null; state = null; save();
    return;
  }
  fenCache.set(game.moves.length, state.fen);
  drawBoard(state.fen);
  if (!state.status.over && state.turn !== game.human) await engineTurn();
}

async function resign() {
  if (!game || !state || state.status.over) return;
  await api("/api/record", { moves: game.moves, engine: game.engine, game_id: game.id,
                             human_color: game.human, reason: "resigned" }).catch(() => {});
  state = { ...state, status: { over: true, result: game.human === "white" ? "black" : "white",
                                reason: "resignation" }, legal: [] };
  view = null; resultHidden = false;
  drawBoard(state.fen);
}

async function copyMoves() {
  const lines = [];
  for (let i = 0; i < game.moves.length; i += 3) {
    const white = game.moves.slice(i, i + 2).join(", ");
    const black = game.moves[i + 2] ? "   " + game.moves[i + 2] : "";
    lines.push(`${i / 3 + 1}. ${white}${black}`);
  }
  try { await navigator.clipboard.writeText(lines.join("\n")); renderStatus("Moves copied to the clipboard."); }
  catch (_) { renderStatus("Copy failed; select the move list manually.", "error"); }
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme !== "dark";
  if (dark) document.documentElement.dataset.theme = "dark"; else delete document.documentElement.dataset.theme;
  $("theme").innerHTML = dark ? "&#9788;" : "&#9790;";
  try { localStorage.setItem(THEME, dark ? "dark" : "light"); } catch (_) { /* ignore */ }
}

// ?moves=e2e4,d2d4,d7d5 opens a game from that position, playing the side to move.
async function fromLink() {
  const raw = new URLSearchParams(location.search).get("moves");
  if (raw === null) return false;
  const moves = raw.split(",").map((m) => m.trim()).filter(Boolean);
  let st;
  try { st = await api("/api/state", { moves }); }
  catch (err) { renderStatus("That link's moves are not a legal game: " + err.message, "error"); return true; }
  game = { id: newId(), engine: $("engine").value, human: st.turn, moves };
  resetView();
  $("color").value = game.human;
  history.replaceState(null, "", location.pathname);
  save();
  await resume();
  return true;
}

async function init() {
  $("new-game").addEventListener("click", startGame);
  $("theme").addEventListener("click", toggleTheme);
  if (document.documentElement.dataset.theme === "dark") $("theme").innerHTML = "&#9788;";
  $("resign").addEventListener("click", resign);
  $("copy").addEventListener("click", copyMoves);
  $("first").addEventListener("click", () => setView(0));
  $("prev").addEventListener("click", () => setView(viewPly() - 1));
  $("next").addEventListener("click", () => setView(viewPly() + 1));
  $("last").addEventListener("click", () => setView(plies()));
  const board = $("board");
  board.addEventListener("pointerdown", onDown);
  board.addEventListener("contextmenu", (e) => e.preventDefault());
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onCancel);
  document.addEventListener("keydown", onKey);
  try {
    const engines = await api("/api/engines");
    for (const e of engines) {
      const opt = document.createElement("option");
      opt.value = e.id; opt.textContent = e.label;
      if (e.default) opt.selected = true;
      $("engine").appendChild(opt);
    }
  } catch (err) {
    renderStatus("The engine server is not reachable: " + err.message, "error");
    return;
  }
  if (await fromLink()) return;
  const saved = load();
  if (saved && saved.id && Array.isArray(saved.moves)) {
    game = saved;
    if ([...$("engine").options].some((o) => o.value === game.engine)) $("engine").value = game.engine;
    $("color").value = game.human;
    await resume();
  } else {
    drawBoard(START_FEN);
  }
}

init();
