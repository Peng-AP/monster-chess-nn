// Monster Chess, Watch tab: two engines play each other, one server request per turn.
// The server is the rules authority; this page only draws, paces and asks.
"use strict";

const GLYPH = Object.fromEntries(Object.entries({ k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" })
  .map(([k, v]) => [k, v + "︎"]));
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1";
const THEME = "monster-chess-theme";
const SETTINGS = "monster-chess-watch-v2";
const DEPTHS = [16, 50, 200, 800, 1600, 3200, 6400, 12800];
// The engine's own defaults (src/config.py); only values that differ are sent.
const SEARCH_DEFAULTS = { c_puct: 1.5, fpu_reduction: 0.3, policy_temperature: 1, root_noise: false, finisher: true };
const MATCH_DEFAULTS = { temperature: "0.5", "temp-plies": "16", "late-temperature": "0", opening: "", swap: true, "both-eval": true };
const NEXT_GAME_DELAY = 3000;
const $ = (id) => document.getElementById(id);

let engines = [];
let match = null;        // one game: { white, black, whiteSims, blackSims, search: {white, black}, ..., moves, pos }
                         // pos[p] = { white, black }: each engine's value (White's view, -1..1) of the position after p half-moves
let series = null;       // { total, swap, index, a, b, score: {a, b}, draws }
let state = null;        // latest /api/state payload of the live game
let running = false, paused = false, busy = false, flipped = false;
let view = null;         // reviewed ply; null = follow the live game
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
const shortLabel = (id) => engineLabel(id).replace(/^\d+ · /, "").split(",")[0];
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
      const file = "abcdefgh"[flipped ? 7 - col : col], rank = flipped ? row + 1 : 8 - row, sq = file + rank;
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
  renderEval();
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
      const after = match.pos[j + 1] || {};
      const opinions = ["white", "black"].filter((side) => known(after[side]))
        .map((side) => `${engineName(side)}: White ${pct(after[side])}%`);
      if (opinions.length) span.title = "After this move\n" + opinions.join("\n");
      span.addEventListener("click", () => setView(j + 1));
      li.appendChild(span);
    }
    list.appendChild(li);
  }
  const cur = list.querySelector(".current");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}

const known = (v) => v !== null && v !== undefined;
const pct = (v) => Math.round((v + 1) * 50);
const engineName = (side) => (match ? `${shortLabel(match[side])} (${side === "white" ? "White" : "Black"})` : side);

// One engine's latest value at or before position `ply`, and the position it came from.
function evalAt(side, ply) {
  for (let p = Math.min(ply, match.pos.length - 1); p >= 0; p--) {
    const v = match.pos[p] && match.pos[p][side];
    if (known(v)) return { v, p };
  }
  return null;
}

function renderEval() {
  const ply = viewPly();
  for (const side of ["white", "black"]) {
    const e = match ? evalAt(side, ply) : null;
    $(`eval-${side}-name`).textContent = match ? engineName(side) : `${side === "white" ? "White" : "Black"} engine`;
    $(`eval-${side}-fill`).style.width = `${e ? pct(e.v) : 50}%`;
    const text = $(`eval-${side}-text`);
    text.textContent = e ? `White ${pct(e.v)}%` : "—";
    // An estimate carried from an earlier position (this one is not searched yet) is dimmed.
    const stale = !!e && e.p !== ply;
    text.classList.toggle("stale", stale);
    text.title = !e ? "No estimate yet" : stale ? `From the position after half-move ${e.p}` : "This position";
  }
  // Graph over positions 0..n, one line per engine, carried forward over positions it did not search.
  const svg = $("eval-graph");
  if (!match || !match.moves.length) { svg.innerHTML = ""; return; }
  const n = match.moves.length;
  const line = (side) => {
    const pts = [];
    let carry = null;
    for (let p = 0; p <= n; p++) {
      const v = match.pos[p] && match.pos[p][side];
      if (known(v)) carry = v;
      if (carry !== null) pts.push(`${(p / n * 100).toFixed(2)},${(20 - carry * 18).toFixed(2)}`);
    }
    return pts.length ? `<polyline points="${pts.join(" ")}" class="curve ${side}"/>` : "";
  };
  const x = (ply / n * 100).toFixed(2);
  svg.innerHTML = `<line x1="0" y1="20" x2="100" y2="20" class="mid"/>${line("black")}${line("white")}
    <line x1="${x}" y1="0" x2="${x}" y2="40" class="cursor"/>`;
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
  if (st.result === "draw") return { title: "Draw", reason: reasons[st.reason] || st.reason, kind: "draw", winner: null };
  const winner = st.result === "white" ? match.white : match.black;
  return { title: `${st.result === "white" ? "White" : "Black"} wins`, kind: "win", winner: st.result,
           reason: `${shortLabel(winner)} · ${reasons[st.reason] || st.reason}` };
}

function renderStatus() {
  if (!match || !state) return;
  if (view !== null) { setStatus(`Reviewing half-move ${view} of ${plies()}. End returns to the live game.`, "review"); return; }
  const out = outcome();
  if (out) { setStatus(`${out.title}: ${out.reason}.`, "over " + out.kind); return; }
  if (!running) { setStatus("Stopped.", ""); return; }
  if (paused) { setStatus("Paused.", ""); return; }
  setStatus(`${state.turn === "white" ? "White" : "Black"} to move (turn ${state.full_turn})`, busy ? "thinking" : "");
}

function renderResult() {
  const box = $("result");
  const out = outcome();
  if (!out || view !== null) { box.hidden = true; return; }
  const more = series && series.index < series.total;
  box.className = `result ${out.kind}`;
  box.innerHTML = `<div class="card"><div class="title">${out.title}</div><div class="reason">${out.reason}</div>
    ${more ? `<div class="reason">Game ${series.index + 1} of ${series.total} starts shortly.</div>` : ""}
    <div class="buttons">${more ? `<button type="button" data-act="stop" class="primary">Stop the series</button>`
                                : `<button type="button" data-act="again" class="primary">${series && series.total > 1 ? "New series" : "Rematch"}</button>`}
    <button type="button" data-act="review">Review game</button></div></div>`;
  const again = box.querySelector('[data-act="again"]');
  if (again) again.addEventListener("click", () => startMatch(true));
  const stop = box.querySelector('[data-act="stop"]');
  if (stop) stop.addEventListener("click", () => { series.total = series.index; clearTimeout(timer); renderResult(); renderSeries(); });
  box.querySelector('[data-act="review"]').addEventListener("click", () => setView(0));
  box.hidden = false;
}

function renderMatchup() {
  if (!match) { $("matchup").textContent = ""; return; }
  const tweaks = (side) => {
    const s = match.search[side], out = [];
    for (const [k, v] of Object.entries(s)) {
      const names = { c_puct: "exploration", fpu_reduction: "FPU", policy_temperature: "policy temp",
                      root_noise: "root noise", finisher: "solver" };
      out.push(typeof v === "boolean" ? `${names[k]} ${v ? "on" : "off"}` : `${names[k]} ${v}`);
    }
    return out.length ? ` (${out.join(", ")})` : "";
  };
  const depth = (sims) => `${sims.toLocaleString()} sims`;
  const rand = match.temperature > 0 && match.tempPlies > 0
    ? `Opening randomness: temperature ${match.temperature} for the first ${match.tempPlies} half-moves`
    : "No opening randomness";
  const late = match.lateTemperature > 0 ? `; temperature ${match.lateTemperature} after that` : "";
  $("matchup").innerHTML = `<div><span class="dot w"></span>White: ${engineLabel(match.white)}, ${depth(match.whiteSims)}${tweaks("white")}</div>
    <div><span class="dot b"></span>Black: ${engineLabel(match.black)}, ${depth(match.blackSims)}${tweaks("black")}</div>
    <div class="muted">${rand}${late}</div>`;
}

function renderSeries() {
  const el = $("series-score");
  if (!series || series.total <= 1) { el.hidden = true; return; }
  const done = series.index - (running && !(state && state.status.over) ? 1 : 0);
  el.hidden = false;
  el.innerHTML = `<div class="muted">Series · game ${Math.min(series.index, series.total)} of ${series.total}</div>
    <div class="score"><span>${shortLabel(series.a)}${series.swap ? "" : " (White)"}</span><strong>${series.score.a}</strong>
    <span class="muted">–</span><strong>${series.score.b}</strong><span>${shortLabel(series.b)}${series.swap ? "" : " (Black)"}</span></div>
    <div class="muted">${series.draws} draw${series.draws === 1 ? "" : "s"} · ${done} finished</div>`;
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
  const side = state.turn;
  let start;
  try {
    const res = await api("/api/engine-move", {
      moves: match.moves, watch: true, engine: side === "white" ? match.white : match.black,
      sims: side === "white" ? match.whiteSims : match.blackSims,
      temperature: match.temperature, temp_plies: match.tempPlies, late_temperature: match.lateTemperature,
      ...match.search[side],
    });
    start = match.moves.length;
    match.moves = match.moves.concat(res.engine_moves);
    // Each searched move reports the mover's value of the position it was played from.
    (res.evals || []).forEach((e, i) => { setPos(start + i, side, e.value); });
    state = res.state;
    fenCache.set(match.moves.length, state.fen);
  } catch (err) {
    busy = false;
    if (err.status === 429 || err.status === 503) {
      setStatus(err.message + " Retrying shortly.", "error");
      schedule(5000);
      return;
    }
    setStatus(err.message, "error");
    stopMatch();
    return;
  }
  if (view === null) drawBoard(state.fen); else { renderMoves(); renderEval(); }
  if ($("both-eval").checked) {
    const game = match;
    await askOther(side, start);
    if (game !== match) return;  // a new game started meanwhile
    if (view === null) drawBoard(state.fen); else { renderMoves(); renderEval(); }
  }
  busy = false;
  if (!running) return;
  if (state.status.over) { finishGame(); return; }
  schedule(Number($("pace").value));
}

function setPos(ply, side, value, game = match) {
  while (game.pos.length <= ply) game.pos.push({});
  if (known(value)) game.pos[ply][side] = value;
}

// The engine that did not move searches the same positions with its own depth and settings.
// Its opinion is informational: a failure (rate limit, busy engine) skips it and play goes on.
async function askOther(mover, start) {
  const other = mover === "white" ? "black" : "white";
  const game = match;
  const plies = [];
  for (let p = start; p < game.moves.length; p++) plies.push(p);
  if (!plies.length) return;
  const { finisher, ...search } = game.search[other];
  try {
    const res = await api("/api/evaluate", { moves: game.moves, engine: game[other], plies,
      sims: other === "white" ? game.whiteSims : game.blackSims, ...search });
    for (const e of res.evals) setPos(e.ply, other, e.value, game);
  } catch (_) { /* skipped */ }
}

function finishGame() {
  running = false;
  $("pause").disabled = true;
  const out = outcome();
  if (series) {
    if (!out.winner) { series.score.a += 0.5; series.score.b += 0.5; series.draws += 1; }
    // Engine A played White this game exactly when match.aWhite (also right when A and B are the same model).
    else if ((out.winner === "white") === match.aWhite) series.score.a += 1;
    else series.score.b += 1;
  }
  renderSeries();
  renderResult();
  if (series && series.index < series.total) {
    timer = setTimeout(() => startMatch(false), NEXT_GAME_DELAY);
  } else {
    $("stop").disabled = true;
  }
}

function readSearch(side) {
  const out = {};
  for (const input of document.querySelectorAll(`.adv-table [data-side="${side}"][data-key]`)) {
    const key = input.dataset.key;
    const value = input.type === "checkbox" ? input.checked : Number(input.value);
    if (typeof value === "number" && !Number.isFinite(value)) throw new Error(`${side}'s ${key} is not a number.`);
    if (value !== SEARCH_DEFAULTS[key]) out[key] = value;
  }
  const ranges = { c_puct: [0.25, 5], fpu_reduction: [0, 1], policy_temperature: [0.5, 3] };
  for (const [k, [lo, hi]] of Object.entries(ranges)) {
    if (k in out && !(out[k] >= lo && out[k] <= hi)) throw new Error(`${side}'s ${k} must be between ${lo} and ${hi}.`);
  }
  return out;
}

function readSettings() {
  const temperature = Number($("temperature").value);
  const tempPlies = Number($("temp-plies").value);
  const lateTemperature = Number($("late-temperature").value);
  if (!(temperature >= 0 && temperature <= 2)) throw new Error("Opening temperature must be between 0 and 2.");
  if (!(Number.isInteger(tempPlies) && tempPlies >= 0 && tempPlies <= 80)) throw new Error("Randomised half-moves must be 0 to 80.");
  if (!(lateTemperature >= 0 && lateTemperature <= 1)) throw new Error("Late temperature must be between 0 and 1.");
  const white = $("white-engine").value, black = $("black-engine").value;
  const sims = (sel, id) => (sel.value === "default" ? engineSims(id) : Number(sel.value));
  const opening = $("opening").value.split(/[\s,]+/).map((m) => m.trim().toLowerCase()).filter(Boolean);
  return { white, black, whiteSims: sims($("white-sims"), white), blackSims: sims($("black-sims"), black),
           whiteDepth: $("white-sims").value, blackDepth: $("black-sims").value,
           search: { white: readSearch("white"), black: readSearch("black") },
           temperature, tempPlies, lateTemperature, opening,
           total: Number($("series").value), swap: $("swap").checked };
}

async function startMatch(newSeries) {
  let s;
  try { s = readSettings(); } catch (err) { setStatus(err.message, "error"); return; }
  clearTimeout(timer);
  if (newSeries || !series) {
    saveSettings();
    series = { total: s.total, swap: s.swap, index: 0, a: s.white, b: s.black, score: { a: 0, b: 0 }, draws: 0,
               aSettings: { sims: s.whiteSims, search: s.search.white }, bSettings: { sims: s.blackSims, search: s.search.black } };
  }
  let opening;
  try { opening = await api("/api/state", { moves: s.opening }); }
  catch (err) { setStatus("Those opening moves are not legal: " + err.message, "error"); return; }
  if (opening.status.over) { setStatus("That opening already ends the game.", "error"); return; }
  // Engine "a" (initially White) keeps its depth and search settings when colours swap.
  const aWhite = !(series.swap && series.index % 2 === 1);
  const A = { id: series.a, ...series.aSettings }, B = { id: series.b, ...series.bSettings };
  const [W, Bk] = aWhite ? [A, B] : [B, A];
  series.index += 1;
  state = opening;
  match = { white: W.id, black: Bk.id, whiteSims: W.sims, blackSims: Bk.sims, aWhite,
            search: { white: W.search, black: Bk.search },
            temperature: s.temperature, tempPlies: s.tempPlies, lateTemperature: s.lateTemperature,
            moves: s.opening.slice(), pos: [] };
  fenCache.clear(); fenCache.set(0, START_FEN); fenCache.set(match.moves.length, state.fen);
  running = true; paused = false; busy = false; view = null;
  $("pause").disabled = false; $("pause").textContent = "Pause"; $("stop").disabled = false;
  renderMatchup();
  renderSeries();
  drawBoard(state.fen);
  schedule(0);
}

function stopMatch() {
  running = false; paused = false;
  clearTimeout(timer);
  if (series) series.total = series.index;
  $("pause").disabled = true; $("stop").disabled = true;
  if (state) { renderStatus(); renderResult(); renderSeries(); }
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

const SIMPLE_IDS = ["white-engine", "white-sims", "black-engine", "black-sims", "pace", "series", "temperature",
                    "temp-plies", "late-temperature", "opening", "swap", "both-eval"];

function saveSettings() {
  const out = {};
  for (const id of SIMPLE_IDS) out[id] = $(id).type === "checkbox" ? $(id).checked : $(id).value;
  for (const input of document.querySelectorAll(".adv-table [data-key]")) {
    out[`${input.dataset.side}:${input.dataset.key}`] = input.type === "checkbox" ? input.checked : input.value;
  }
  try { localStorage.setItem(SETTINGS, JSON.stringify(out)); } catch (_) { /* storage may be unavailable */ }
}

function loadSettings() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SETTINGS)); } catch (_) { saved = null; }
  if (!saved) return;
  for (const [key, value] of Object.entries(saved)) {
    let el;
    if (key.includes(":")) {
      const [side, k] = key.split(":");
      el = document.querySelector(`.adv-table [data-side="${side}"][data-key="${k}"]`);
    } else {
      el = $(key);
    }
    if (!el) continue;
    if (el.type === "checkbox") { el.checked = Boolean(value); continue; }
    if (el.tagName === "SELECT" && ![...el.options].some((o) => o.value === value)) continue;
    el.value = value;
  }
}

function resetAdvanced() {
  for (const [id, v] of Object.entries(MATCH_DEFAULTS)) {
    if ($(id).type === "checkbox") $(id).checked = v; else $(id).value = v;
  }
  for (const input of document.querySelectorAll(".adv-table [data-key]")) {
    const v = SEARCH_DEFAULTS[input.dataset.key];
    if (input.type === "checkbox") input.checked = v; else input.value = v;
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

function onGraphClick(e) {
  if (!match || !match.moves.length) return;
  const r = $("eval-graph").getBoundingClientRect();
  setView(Math.round(((e.clientX - r.left) / r.width) * match.moves.length));
}

async function init() {
  $("theme").addEventListener("click", toggleTheme);
  if (document.documentElement.dataset.theme === "dark") $("theme").innerHTML = "&#9788;";
  $("start").addEventListener("click", () => startMatch(true));
  $("pause").addEventListener("click", togglePause);
  $("stop").addEventListener("click", stopMatch);
  $("copy").addEventListener("click", copyMoves);
  $("reset-advanced").addEventListener("click", resetAdvanced);
  $("first").addEventListener("click", () => setView(0));
  $("prev").addEventListener("click", () => setView(viewPly() - 1));
  $("next").addEventListener("click", () => setView(viewPly() + 1));
  $("last").addEventListener("click", () => setView(plies()));
  $("flip").addEventListener("click", () => { flipped = !flipped; show(); });
  $("eval-graph").addEventListener("click", onGraphClick);
  document.addEventListener("keydown", onKey);
  fillDepths($("white-sims"));
  fillDepths($("black-sims"));
  try { engines = await api("/api/engines"); }
  catch (err) { setStatus("The engine server is not reachable: " + err.message, "error"); return; }
  for (const id of ["white-engine", "black-engine"]) {
    $(id).innerHTML = engines.map((e) => `<option value="${e.id}">${e.label}</option>`).join("");
  }
  $("white-engine").value = engines[0].id;
  $("black-engine").value = (engines.find((e) => e.default) || engines[1] || engines[0]).id;
  loadSettings();
  drawBoard(START_FEN);
}

init();
