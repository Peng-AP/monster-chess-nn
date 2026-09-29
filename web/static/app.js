// Monster Chess client: the server is the rules authority; the page only draws and asks.
"use strict";

// U+FE0E asks for the text (not emoji) presentation; phones otherwise draw the pawn as an emoji.
const GLYPH = Object.fromEntries(Object.entries({ k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" })
  .map(([k, v]) => [k, v + "︎"]));
const THEME = "monster-chess-theme";
const STORE = "monster-chess-game-v1";
const $ = (id) => document.getElementById(id);

let game = null;      // { id, engine, human, moves: [] }
let state = null;     // last /api/state payload
let selected = null;  // selected from-square, e.g. "e2"
let busy = false;

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

// --- board ------------------------------------------------------------------

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

function humanToMove() {
  return state && !state.status.over && game && state.turn === game.human;
}

function render() {
  const board = state ? parseFen(state.fen) : {};
  const flip = game && game.human === "black";
  const last = lastHumanOrEngineMove();
  const targets = new Map();
  if (selected && humanToMove()) {
    for (const m of state.legal) if (m.slice(0, 2) === selected) targets.set(m.slice(2, 4), true);
  }
  const el = $("board");
  el.innerHTML = "";
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = "abcdefgh"[flip ? 7 - col : col];
      const rank = flip ? row + 1 : 8 - row;
      const sq = file + rank;
      const cell = document.createElement("div");
      cell.className = `sq ${(col + row) % 2 ? "dark" : "light"}`;
      cell.dataset.sq = sq;
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", sq + (board[sq] ? " " + board[sq] : ""));
      if (last && (last.slice(0, 2) === sq || last.slice(2, 4) === sq)) cell.classList.add("last");
      if (sq === selected) cell.classList.add("selected");
      if (targets.has(sq)) cell.classList.add("target", ...(board[sq] ? ["capture"] : []));
      const piece = board[sq];
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece === piece.toUpperCase() ? "w" : "b"}`;
        span.textContent = GLYPH[piece.toLowerCase()];
        cell.appendChild(span);
      }
      if (row === 7) cell.insertAdjacentHTML("beforeend", `<span class="coord file">${file}</span>`);
      if (col === 0) cell.insertAdjacentHTML("beforeend", `<span class="coord rank">${rank}</span>`);
      cell.addEventListener("click", () => onSquare(sq, board));
      el.appendChild(cell);
    }
  }
  renderMoves();
  renderStatus();
  $("resign").disabled = !game || !state || state.status.over;
  $("copy").disabled = !game || !game.moves.length;
}

function lastHumanOrEngineMove() {
  return game && game.moves.length ? game.moves[game.moves.length - 1] : null;
}

function renderMoves() {
  // White's turn is two half-moves ("e2e4, d2d4"); Black replies with one.
  const list = $("moves");
  list.innerHTML = "";
  if (!game) return;
  let i = 0;
  while (i < game.moves.length) {
    const white = game.moves.slice(i, i + 2);
    const black = game.moves[i + 2];
    const li = document.createElement("li");
    li.textContent = white.join(", ") + (black ? "   " + black : "");
    list.appendChild(li);
    i += 3;
  }
  list.scrollTop = list.scrollHeight;
}

function renderStatus(message, kind) {
  const el = $("status");
  el.className = "status" + (kind ? " " + kind : "");
  if (message) { el.textContent = message; return; }
  if (!game || !state) { el.textContent = "Choose an engine and a colour, then start a new game."; return; }
  const st = state.status;
  if (st.over) {
    el.classList.add("over");
    const winner = st.result === "draw" ? null : st.result;
    const outcome = winner ? (winner === game.human ? "You win" : "The engine wins")
                           : "Draw";
    el.textContent = `${outcome} (${st.reason}).`;
    return;
  }
  if (busy) { el.classList.add("thinking"); el.textContent = "Engine is thinking"; return; }
  const who = state.turn === game.human ? "Your move" : "Engine to move";
  const half = state.turn === "white" ? ` — White move ${state.half} of 2` : "";
  el.textContent = `${who}${half}. Turn ${state.full_turn}.`;
}

// --- interaction ------------------------------------------------------------

function onSquare(sq, board) {
  if (busy || !state || state.status.over) return;
  if (!humanToMove()) { engineTurn(); return; }  // retry after a failed engine request
  const own = board[sq] && ((board[sq] === board[sq].toUpperCase()) === (game.human === "white"));
  if (selected && selected !== sq) {
    const options = state.legal.filter((m) => m.slice(0, 2) === selected && m.slice(2, 4) === sq);
    if (options.length === 1) return play(options[0]);
    if (options.length > 1) return choosePromotion(options);
  }
  selected = own && state.legal.some((m) => m.slice(0, 2) === sq) ? (selected === sq ? null : sq) : null;
  render();
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

async function play(move) {
  selected = null;
  const moves = game.moves.concat([move]);
  try {
    state = await api("/api/state", { moves });
    game.moves = moves;
    save();
    render();
    if (!state.status.over && state.turn !== game.human) await engineTurn();
  } catch (err) {
    renderStatus(err.message, "error");
  }
}

async function engineTurn() {
  busy = true;
  render();
  try {
    const res = await api("/api/engine-move", {
      moves: game.moves, engine: game.engine, game_id: game.id, human_color: game.human,
    });
    game.moves = game.moves.concat(res.engine_moves);
    state = res.state;
    save();
  } catch (err) {
    busy = false;
    render();
    renderStatus(err.message + " — click the board or reload to retry.", "error");
    return;
  }
  busy = false;
  render();
}

async function startGame() {
  if (game && state && !state.status.over && game.moves.length) {
    api("/api/record", { moves: game.moves, engine: game.engine, game_id: game.id,
                         human_color: game.human, reason: "new game" }).catch(() => {});
  }
  game = { id: newId(), engine: $("engine").value, human: $("color").value, moves: [] };
  selected = null;
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
  render();
  if (!state.status.over && state.turn !== game.human) await engineTurn();
}

async function resign() {
  if (!game || !state || state.status.over) return;
  await api("/api/record", { moves: game.moves, engine: game.engine, game_id: game.id,
                             human_color: game.human, reason: "resigned" }).catch(() => {});
  state = { ...state, status: { over: true, result: game.human === "white" ? "black" : "white",
                                reason: "resignation" }, legal: [] };
  render();
}

async function copyMoves() {
  const text = $("moves").innerText;
  try { await navigator.clipboard.writeText(text); renderStatus("Moves copied to the clipboard."); }
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
    render();
  }
}

init();
