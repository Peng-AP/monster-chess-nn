//! Exact forced-king-capture solver, ported from `src/forced_capture.py`.
//!
//! The Python version is a faithful AND/OR search but spends **92% of its time
//! in `_get_white_actions`** -- generating White's half-move pairs through
//! python-chess. Memoising it bought only 1.8x at depth 4 because the memo
//! cannot touch move generation. This port moves the whole search onto the
//! bitboard engine.
//!
//! Semantics are identical to the Python original and must stay that way:
//!
//! - **Black node (OR)** -- Black needs *one* move that wins.
//! - **White node (AND)** -- *every* White double-move must still lose; White
//!   capturing Black's king refutes immediately.
//! - Depth counts **Black moves**: `d=1` means "Black captures the king now".
//! - The turn cap is neutralised -- the question is whether the position is
//!   *won*, not whether the clock is about to run out.
//! - A king capture is unconditional and pseudo-legal, so "Black can capture
//!   now" is exactly "some Black piece attacks the White king square".
//! - No pseudo-legal White move is **not** a forced capture: the engine does
//!   not score stalemate, so claiming a win there would award something the
//!   rules do not.
//!
//! Budget exhaustion is reported, never swallowed: "no answer" is not "no win".

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::bitboard::{parse_fen, Board, Move, BLACK, WHITE};
use crate::monster::{black_actions, white_second_half_moves, white_single_moves};

/// Exact identity of a searched position.
///
/// En passant is included because White confers it only with the LAST half of
/// its turn, so two orderings of one pair can differ here; castling rights are
/// included because White can capture a Black rook and remove one. Dropping
/// either would merge positions that are not equal. The halfmove/fullmove
/// counters are deliberately excluded -- they never affect this search, and
/// including them would suppress real transpositions.
type PosKey = (u64, u64, u64, u64, u64, u64, u64, u64, u64, i16, bool);

fn key_of(b: &Board) -> PosKey {
    (
        b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
        b.occupied_co[WHITE], b.occupied_co[BLACK],
        b.castling,
        b.ep_square.map(|s| s as i16).unwrap_or(-1),
        b.turn,
    )
}

fn black_can_capture_now(b: &Board) -> bool {
    match b.king_square(WHITE) {
        None => true, // already captured
        Some(k) => b.is_attacked_by(BLACK, k),
    }
}

pub struct Solver {
    budget: u64,
    pub nodes: u64,
    pub memo_hits: u64,
    pub dedup_skipped: u64,
    or_memo: HashMap<(PosKey, u8), bool>,
    and_memo: HashMap<(PosKey, u8), bool>,
}

/// Distinguishes "searched out" from "ran out of budget" without conflating
/// them into a bare boolean, exactly as the Python `NodeBudgetExceeded` does.
pub enum Answer {
    Value(bool),
    Exhausted,
}

impl Solver {
    pub fn new(budget: u64) -> Self {
        Solver {
            budget,
            nodes: 0,
            memo_hits: 0,
            dedup_skipped: 0,
            or_memo: HashMap::new(),
            and_memo: HashMap::new(),
        }
    }

    fn tick(&mut self) -> bool {
        self.nodes += 1;
        self.budget != 0 && self.nodes > self.budget
    }

    /// Every distinct position reachable by one complete White turn.
    ///
    /// Dedup is on the RESULTING POSITION, never on the destination square. A
    /// path that captures a piece and one that does not produce different
    /// positions and are correctly kept apart; only genuinely identical
    /// positions are merged. Returns `None` when White has no pseudo-legal
    /// first half.
    fn white_turn_successors(&mut self, board: &Board) -> Option<Vec<Board>> {
        let firsts = white_single_moves(board);
        if firsts.is_empty() {
            return None;
        }
        let mut out: Vec<Board> = Vec::new();
        let mut seen: Vec<PosKey> = Vec::new();
        for m1 in &firsts {
            let mut after1 = board.clone();
            after1.turn = true;
            after1.push(m1);
            if after1.king_square(BLACK).is_none() {
                // White refutes by capturing the king outright.
                return Some(vec![after1]);
            }
            after1.turn = true; // the turn does not pass between White's halves
            let seconds = white_second_half_moves(&after1);
            if seconds.is_empty() {
                continue;
            }
            for m2 in &seconds {
                let mut after2 = after1.clone();
                after2.push(m2);
                if after2.king_square(BLACK).is_none() {
                    return Some(vec![after2]);
                }
                let k = key_of(&after2);
                if seen.contains(&k) {
                    self.dedup_skipped += 1;
                    continue;
                }
                seen.push(k);
                out.push(after2);
            }
        }
        if out.is_empty() {
            return None;
        }
        Some(out)
    }

    /// Black to move: can Black force a king capture within `d` Black moves?
    pub fn black_wins_within(&mut self, board: &Board, d: u8) -> Answer {
        if d == 0 {
            return Answer::Value(false);
        }
        if black_can_capture_now(board) {
            return Answer::Value(true);
        }
        if d == 1 {
            return Answer::Value(false);
        }
        let key = (key_of(board), d);
        if let Some(&v) = self.or_memo.get(&key) {
            self.memo_hits += 1;
            return Answer::Value(v);
        }
        let mut result = false;
        for mv in black_actions(board, true) {
            if self.tick() {
                return Answer::Exhausted;
            }
            let mut child = board.clone();
            child.push(&mv);
            if child.king_square(WHITE).is_none() {
                result = true;
                break;
            }
            match self.white_all_lose(&child, d - 1) {
                Answer::Exhausted => return Answer::Exhausted,
                Answer::Value(true) => {
                    result = true;
                    break;
                }
                Answer::Value(false) => {}
            }
        }
        // Only a completed search is cached; a budget cut returns above.
        self.or_memo.insert(key, result);
        Answer::Value(result)
    }

    /// White to move: does every White turn still lose within `d` Black moves?
    pub fn white_all_lose(&mut self, board: &Board, d: u8) -> Answer {
        let key = (key_of(board), d);
        if let Some(&v) = self.and_memo.get(&key) {
            self.memo_hits += 1;
            return Answer::Value(v);
        }
        let successors = match self.white_turn_successors(board) {
            None => return Answer::Value(false),
            Some(s) => s,
        };
        let mut result = true;
        for child in successors {
            if self.tick() {
                return Answer::Exhausted;
            }
            if child.king_square(BLACK).is_none() {
                result = false; // White refutes by capturing first
                break;
            }
            match self.black_wins_within(&child, d) {
                Answer::Exhausted => return Answer::Exhausted,
                Answer::Value(false) => {
                    result = false;
                    break;
                }
                Answer::Value(true) => {}
            }
        }
        self.and_memo.insert(key, result);
        Answer::Value(result)
    }

    /// `best_forced_move` across several threads.
    ///
    /// Root Black moves at a given depth are independent, so they shard
    /// cleanly. Each worker keeps its OWN memo -- sharing one would need a
    /// lock on the hottest path -- so this trades memo reuse for parallelism
    /// and is worth it only at depths where a single position costs seconds.
    ///
    /// Two things preserve the single-threaded answer exactly: depths are
    /// still tried in order, and within a depth the winning move with the
    /// LOWEST root index wins, matching the sequential scan. Each worker gets
    /// the full budget, so `threads > 1` permits more total nodes; that can
    /// only turn an "exhausted" into a real answer, never change a proof.
    pub fn best_forced_move_threaded(
        board: &Board,
        max_black_moves: u8,
        budget: u64,
        threads: usize,
    ) -> Result<Option<(String, u8)>, ()> {
        let actions: Vec<Move> = black_actions(board, true);
        let n = actions.len();
        if threads <= 1 || n <= 1 {
            let mut s = Solver::new(budget);
            return s.best_forced_move(board, max_black_moves);
        }
        for d in 1..=max_black_moves {
            let found: std::sync::Mutex<Option<(usize, String, u8)>> =
                std::sync::Mutex::new(None);
            let exhausted = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|scope| {
                for t in 0..threads {
                    let actions = &actions;
                    let found = &found;
                    let exhausted = &exhausted;
                    scope.spawn(move || {
                        let mut s = Solver::new(budget);
                        let mut i = t;
                        while i < n {
                            let mv = &actions[i];
                            let mut child = board.clone();
                            child.push(mv);
                            if child.king_square(WHITE).is_none() {
                                let mut g = found.lock().unwrap();
                                if g.is_none() || g.as_ref().unwrap().0 > i {
                                    *g = Some((i, mv.uci(), 1));
                                }
                                return;
                            }
                            if d > 1 {
                                match s.white_all_lose(&child, d - 1) {
                                    Answer::Exhausted => {
                                        exhausted.store(
                                            true,
                                            std::sync::atomic::Ordering::Relaxed,
                                        );
                                        return;
                                    }
                                    Answer::Value(true) => {
                                        let mut g = found.lock().unwrap();
                                        if g.is_none()
                                            || g.as_ref().unwrap().0 > i
                                        {
                                            *g = Some((i, mv.uci(), d));
                                        }
                                        return;
                                    }
                                    Answer::Value(false) => {}
                                }
                            }
                            i += threads;
                        }
                    });
                }
            });
            if let Some((_i, uci, depth)) = found.into_inner().unwrap() {
                return Ok(Some((uci, depth)));
            }
            if exhausted.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(());
            }
        }
        Ok(None)
    }

    /// The move forcing the fastest capture: (uci, depth) or None.
    ///
    /// Iterative deepening sits OUTSIDE the move loop so the move returned
    /// belongs to a *shortest* forced line -- a finisher that wins slowly
    /// still risks the move limit.
    pub fn best_forced_move(
        &mut self,
        board: &Board,
        max_black_moves: u8,
    ) -> Result<Option<(String, u8)>, ()> {
        let actions: Vec<Move> = black_actions(board, true);
        for d in 1..=max_black_moves {
            for mv in &actions {
                let mut child = board.clone();
                child.push(mv);
                if child.king_square(WHITE).is_none() {
                    return Ok(Some((mv.uci(), 1)));
                }
                if d == 1 {
                    continue;
                }
                match self.white_all_lose(&child, d - 1) {
                    Answer::Exhausted => return Err(()),
                    Answer::Value(true) => return Ok(Some((mv.uci(), d))),
                    Answer::Value(false) => {}
                }
            }
        }
        Ok(None)
    }
}

/// `try_forced_capture_move` for the native engine.
///
/// Returns `(uci_or_None, depth_or_None, exhausted)` -- the same triple as the
/// Python entry point, so callers can swap engines without reinterpreting the
/// result. `exhausted` true means the budget stopped the search; a caller must
/// treat that as "no answer", never as "no win".
#[pyfunction]
#[pyo3(signature = (fen, max_black_moves=3, node_budget=200_000, threads=1))]
fn forced_capture_move(
    fen: &str,
    max_black_moves: u8,
    node_budget: u64,
    threads: usize,
) -> PyResult<(Option<String>, Option<u8>, bool)> {
    let board = parse_fen(fen)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    if board.turn {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expects a Black-to-move position",
        ));
    }
    match Solver::best_forced_move_threaded(
        &board, max_black_moves, node_budget, threads) {
        Err(()) => Ok((None, None, true)),
        Ok(None) => Ok((None, None, false)),
        Ok(Some((uci, d))) => Ok((Some(uci), Some(d), false)),
    }
}

/// Search counters for the last call, for benchmarking rather than play.
#[pyfunction]
#[pyo3(signature = (fen, max_black_moves=3, node_budget=200_000))]
fn forced_capture_stats(
    fen: &str,
    max_black_moves: u8,
    node_budget: u64,
) -> PyResult<(Option<u8>, u64, u64, u64, bool)> {
    let board = parse_fen(fen)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut s = Solver::new(node_budget);
    let r = s.best_forced_move(&board, max_black_moves);
    let (depth, exhausted) = match r {
        Err(()) => (None, true),
        Ok(None) => (None, false),
        Ok(Some((_, d))) => (Some(d), false),
    };
    Ok((depth, s.nodes, s.memo_hits, s.dedup_skipped, exhausted))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forced_capture_move, m)?)?;
    m.add_function(wrap_pyfunction!(forced_capture_stats, m)?)?;
    Ok(())
}
