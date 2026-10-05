// Monster Chess, Watch tab: two engines play each other, one server request per turn.
// The server is the rules authority; this page only draws, paces and asks.
"use strict";

const GLYPH = Object.fromEntries(Object.entries({ k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" })
  .map(([k, v]) => [k, v + "︎"]));
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1";
const THEME = "monster-chess-theme";
const SETTINGS = "monster-chess-watch-v1";
const DEPTHS = [16, 50, 200, 800, 1600, 3200, 6400, 12800];
const $ = (id) => document.getElementById(id);

let engines = [];          // /api/engines
let match = null;          // { white, black, whiteSims, blackSims, temperature, tempPlies, moves: [] }
let state = null;          // latest /api/state payload of the live game
let running = false, paused = false, busy = false;
let view = null;           // reviewed ply; null = follow the live game
let timer = null;
const fenCache = new Map([[0, START_FEN]]);

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: "unexpected server response" }));
  if (!res.ok) { const e = new Error(data.error || `request failed (${res.status})`); e.status = res.status; throw e; }
  return data;
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

const plies = () => (match ? match.moves.length : 0);
const viewPly = () => (view === null ? plies() : view);
const engineLabel = (id) => (engines.find((e) => e.id === id) || { label: id }).label;
const engineSims = (id) => (engines.find((e) => e.id === id) || { sims: 3200 }).sims;

// --- drawing ---------------------------------------------------------------------

async function fenAt(ply) {
  if (ply === plies() && state) return state.fen;
  if (!fenCache.has(ply)) fenCache.set(ply, (await api("/api/state", { moves: match.moves.slice(0, ply) })).fen);
  return fenCache.get(ply);
}

async function show() {
  let fen = START_FEN;
  if (match) {
    const ply = viewPly();
    try { fen = await fenAt(ply); } catch (err) { setStatus(err.message, "error"); return; }
    if (ply !== viewPly()) return;
  }
  drawBoard(fen);
}

function drawBoard(fen) {
  const board = parseFen(fen);
  const ply = viewPly();
  const last = match && ply > 0 ? match.moves[ply - 1] : null;
  const el = $("board");
  el.innerHTML = "";
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = "abcdefgh"[col], rank = 8 - row, sq = file + rank;
      const cell = document.createElement("div");
      cell.className = `sq ${(col + row) % 2 ? "dark" : "light"}`;
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", sq + (board[sq] ? " " + board[sq] : ""));
      if (last && (last.slice(0, 2) === sq || last.slice(2, 4) === sq)) cell.classList.add("last");
      const piece = board[sq];
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
  $("board").classList.toggle("reviewing", view !== null);
  renderMoves();
  renderStatus();
  renderResult();
  const live = view === null;
  $("first").disabled = $("prev").disabled = !match || ply === 0;
  $("next").disabled = $("last").disabled = !match || live;
  $("copy").disabled = !match || !match.moves.length;
  const link = $("take-over");
  // Offer the reviewed (or live) position to a human, unless it is the game's final position.
  link.hidden = !match || ply === 0 || (view === null && state && state.status.over);
  if (match) link.href = "/?moves=" + encodeURIComponent(match.moves.slice(0, ply).join(","));
}

function renderMoves() {
  const list = $("moves");
  list.innerHTML = "";
  if (!match) return;
  const current = viewPly();
  for (let i = 0; i < match.moves.length; i += 3) {
    const li = document.createElement("li");
    for (let j = i; j < Math.min(i + 3, match.moves.length); j++) {
      const span = document.createElement("span");
      span.className = "mv" + (j === i + 2 ? " black" : "") + (j + 1 === current ? " current" : "");
      span.textContent = match.moves[j];
      span.addEventListener("click", () => setView(j + 1));
      li.appendChild(span);
    }
    list.appendChild(li);
  }
  const cur = list.querySelector(".current");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}

function setStatus(message, kind) {
  const el = $("status");
  el.className = "status" + (kind ? " " + kind : "");
  el.textContent = message;
}

function outcome() {
  const st = state && state.status;
  if (!st || !st.over) return null;
  const reasons = { "king captured": "King captured", repetition: "Draw by repetition", "turn limit": "150-turn limit" };
  if (st.result === "draw") return { title: "Draw", reason: reasons[st.reason] || st.reason, kind: "draw" };
  const winner = st.result === "white" ? match.white : match.black;
  return { title: `${st.result === "white" ? "White" : "Black"} wins`, reason: `${engineLabel(winner)} · ${reasons[st.reason] || st.reason}`, kind: "win" };
}

function renderStatus() {
  if (!match || !state) return;
  if (view !== null) {
    setStatus(`Reviewing half-move ${view} of ${plies()}. End returns to the live game.`, "review");
    return;
  }
  const out = outcome();
  if (out) { setStatus(`${out.title}: ${out.reason}.`, "over " + out.kind); return; }
  if (!running) { setStatus("Stopped.", ""); return; }
  if (paused) { setStatus("Paused.", ""); return; }
  const side = state.turn === "white" ? "White" : "Black";
  setStatus(`${side} to move (turn ${state.full_turn})`, busy ? "thinking" : "");
}

function renderResult() {
  const box = $("result");
  const out = outcome();
  if (!out || view !== null) { box.hidden = true; return; }
  box.className = `result ${out.kind}`;
  box.innerHTML = `<div class="card"><div class="title">${out.title}</div><div class="reason">${out.reason}</div>
    <div class="buttons"><button type="button" data-act="again" class="primary">Rematch</button>
    <button type="button" data-act="review">Review game</button></div></div>`;
  box.querySelector('[data-act="again"]').addEventListener("click", startMatch);
  box.querySelector('[data-act="review"]').addEventListener("click", () => setView(0));
  box.hidden = false;
}

function renderMatchup() {
  if (!match) { $("matchup").textContent = ""; return; }
  const depth = (sims) => `${sims.toLocaleString()} sims`;
  $("matchup").innerHTML = `<div><span class="dot w"></span>White: ${engineLabel(match.white)}, ${depth(match.whiteSims)}</div>
    <div><span class="dot b"></span>Black: ${engineLabel(match.black)}, ${depth(match.blackSims)}</div>
    <div class="muted">${match.temperature > 0 && match.tempPlies > 0
      ? `Opening randomness: temperature ${match.temperature} for the first ${match.tempPlies} half-moves`
      : "No opening randomness: both engines play their best move every time"}</div>`;
}

// --- the match loop --------------------------------------------------------------

function schedule(delay) {
  clearTimeout(timer);
  timer = setTimeout(step, delay);
}

async function step() {
  if (!running || paused || busy || !state || state.status.over) return;
  busy = true;
  renderStatus();
  const white = state.turn === "white";
  try {
    const res = await api("/api/engine-move", {
      moves: match.moves, watch: true, engine: white ? match.white : match.black,
      sims: white ? match.whiteSims : match.blackSims,
      temperature: match.temperature, temp_plies: match.tempPlies,
    });
    match.moves = match.moves.concat(res.engine_moves);
    state = res.state;
    fenCache.set(match.moves.length, state.fen);
  } catch (err) {
    busy = false;
    if (err.status === 429 || err.status === 503) {   // rate limit or GPU busy: wait and retry
      setStatus(err.message + " Retrying shortly.", "error");
      schedule(5000);
      return;
    }
    setStatus(err.message, "error");
    stopMatch(false);
    return;
  }
  busy = false;
  if (view === null) drawBoard(state.fen); else renderMoves();
  if (state.status.over) { stopMatch(true); return; }
  schedule(Number($("pace").value));
}

function readSettings() {
  const temperature = Number($("temperature").value);
  const tempPlies = Number($("temp-plies").value);
  if (!(temperature >= 0 && temperature <= 2)) throw new Error("Temperature must be between 0 and 2.");
  if (!(Number.isInteger(tempPlies) && tempPlies >= 0 && tempPlies <= 80)) throw new Error("Randomised half-moves must be 0 to 80.");
  const white = $("white-engine").value, black = $("black-engine").value;
  const sims = (sel, id) => (sel.value === "default" ? engineSims(id) : Number(sel.value));
  const opening = $("opening").value.split(/[\s,]+/).map((m) => m.trim().toLowerCase()).filter(Boolean);
  return { white, black, whiteSims: sims($("white-sims"), white), blackSims: sims($("black-sims"), black),
           temperature, tempPlies, opening };
}

async function startMatch() {
  let s;
  try { s = readSettings(); } catch (err) { setStatus(err.message, "error"); return; }
  clearTimeout(timer);
  saveSettings();
  try { state = await api("/api/state", { moves: s.opening }); }
  catch (err) { setStatus("Those opening moves are not legal: " + err.message, "error"); return; }
  if (state.status.over) { setStatus("That opening already ends the game.", "error"); return; }
  match = { white: s.white, black: s.black, whiteSims: s.whiteSims, blackSims: s.blackSims,
            temperature: s.temperature, tempPlies: s.tempPlies, moves: s.opening };
  fenCache.clear(); fenCache.set(0, START_FEN); fenCache.set(match.moves.length, state.fen);
  running = true; paused = false; busy = false; view = null;
  $("pause").disabled = false; $("pause").textContent = "Pause"; $("stop").disabled = false;
  renderMatchup();
  drawBoard(state.fen);
  schedule(0);
}

function stopMatch(finished) {
  running = false; paused = false;
  clearTimeout(timer);
  $("pause").disabled = true; $("stop").disabled = true;
  if (!finished && state) renderStatus();
}

function togglePause() {
  if (!running) return;
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  renderStatus();
  if (!paused) schedule(0);
}

function setView(ply) {
  if (!match) return;
  const target = Math.max(0, Math.min(plies(), ply));
  view = target === plies() ? null : target;
  show();
}

function onKey(e) {
  if (!match || e.altKey || e.ctrlKey || e.metaKey || e.target.closest("select, input, textarea")) return;
  const targets = { ArrowLeft: viewPly() - 1, ArrowRight: viewPly() + 1, Home: 0, ArrowUp: 0, End: plies(), ArrowDown: plies() };
  if (!(e.key in targets)) return;
  e.preventDefault();
  setView(targets[e.key]);
}

async function copyMoves() {
  const lines = [];
  for (let i = 0; i < match.moves.length; i += 3) {
    lines.push(`${i / 3 + 1}. ${match.moves.slice(i, i + 2).join(", ")}${match.moves[i + 2] ? "   " + match.moves[i + 2] : ""}`);
  }
  try { await navigator.clipboard.writeText(lines.join("\n")); setStatus("Moves copied to the clipboard."); }
  catch (_) { setStatus("Copy failed; select the move list manually.", "error"); }
}

// --- settings and setup ----------------------------------------------------------

function saveSettings() {
  const ids = ["white-engine", "white-sims", "black-engine", "black-sims", "pace", "temperature", "temp-plies", "opening"];
  try { localStorage.setItem(SETTINGS, JSON.stringify(Object.fromEntries(ids.map((id) => [id, $(id).value])))); }
  catch (_) { /* storage may be unavailable */ }
}

function loadSettings() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SETTINGS)); } catch (_) { saved = null; }
  if (!saved) return;
  for (const [id, value] of Object.entries(saved)) {
    const el = $(id);
    if (!el) continue;
    if (el.tagName === "SELECT" && ![...el.options].some((o) => o.value === value)) continue;
    el.value = value;
  }
}

function fillDepths(select) {
  select.innerHTML = `<option value="default">Level default</option>` +
    DEPTHS.map((d) => `<option value="${d}">${d.toLocaleString()} simulations</option>`).join("");
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme !== "dark";
  if (dark) document.documentElement.dataset.theme = "dark"; else delete document.documentElement.dataset.theme;
  $("theme").innerHTML = dark ? "&#9788;" : "&#9790;";
  try { localStorage.setItem(THEME, dark ? "dark" : "light"); } catch (_) { /* ignore */ }
}

async function init() {
  $("theme").addEventListener("click", toggleTheme);
  if (document.documentElement.dataset.theme === "dark") $("theme").innerHTML = "&#9788;";
  $("start").addEventListener("click", startMatch);
  $("pause").addEventListener("click", togglePause);
  $("stop").addEventListener("click", () => stopMatch(false));
  $("copy").addEventListener("click", copyMoves);
  $("first").addEventListener("click", () => setView(0));
  $("prev").addEventListener("click", () => setView(viewPly() - 1));
  $("next").addEventListener("click", () => setView(viewPly() + 1));
  $("last").addEventListener("click", () => setView(plies()));
  document.addEventListener("keydown", onKey);
  fillDepths($("white-sims"));
  fillDepths($("black-sims"));
  try { engines = await api("/api/engines"); }
  catch (err) { setStatus("The engine server is not reachable: " + err.message, "error"); return; }
  for (const id of ["white-engine", "black-engine"]) {
    $(id).innerHTML = engines.map((e) => `<option value="${e.id}">${e.label}</option>`).join("");
  }
  // A sensible first match: the strongest two engines.
  $("white-engine").value = engines[0].id;
  $("black-engine").value = (engines.find((e) => e.default) || engines[1] || engines[0]).id;
  loadSettings();
  drawBoard(START_FEN);
}

init();
