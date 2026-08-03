//! The Monster Chess action APIs, ported from `src/monster_chess.py`.
//!
//! Two APIs coexist and are separately implemented, deliberately:
//!
//! - **Atomic** — White's turn as `(m1, m2)` pairs. Used for real play and for
//!   strict pair-level legality of recorded games.
//! - **Half-move** — each half as its own ply, so MCTS gives each a learned
//!   prior and branching collapses from ~900 pairs to ~30 + ~30.
//!
//! They differ in one documented way, which is *not* a bug to fix: the
//! half-move path offers an m1 whose every continuation is unsafe (search sees
//! the losing m2 and avoids it), while the atomic path hides it when a
//! globally-safe pair exists.
//!
//! Ordering is contractual. `truncate_wins=false` lists winning actions
//! **first**, then safe ones, then — only if nothing is safe — everything
//! (the forced blunder). `CONTEXT.md` §1.1 records that callers rely on it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bitboard::{
    generate_pseudo_legal, parse_fen, Board, Move, BLACK, KING, WHITE,
};

const NULL_UCI: &str = "0000";

/// White's first half-moves: every pseudo-legal move, no safety filter at all.
/// The defining liberty — White may step into check and step out on m2.
pub fn white_single_moves(board: &Board) -> Vec<Move> {
    let mut b = board.clone();
    b.turn = true;
    generate_pseudo_legal(&b)
}

/// White's legal `(m1, m2)` pairs. `m2 == None` means the null move: m1 already
/// captured the Black king, so the turn ends there.
pub fn white_actions(board: &Board, truncate_wins: bool) -> Vec<(Move, Option<Move>)> {
    let first = white_single_moves(board);
    if first.is_empty() {
        return Vec::new();
    }
    let mut wins = Vec::new();
    let mut safe = Vec::new();
    let mut all = Vec::new();

    for m1 in first {
        let mut after1 = board.clone();
        after1.turn = true;
        after1.push(&m1);

        // A king capture wins unconditionally: the game ends before any check
        // on White's own king can matter.
        if after1.king_square(BLACK).is_none() {
            if truncate_wins {
                return vec![(m1, None)];
            }
            wins.push((m1, None));
            continue;
        }

        after1.turn = true; // the turn does not pass between White's halves
        for m2 in generate_pseudo_legal(&after1) {
            let mut after2 = after1.clone();
            after2.push(&m2);
            if after2.king_square(BLACK).is_none() {
                if truncate_wins {
                    return vec![(m1, Some(m2))];
                }
                wins.push((m1, Some(m2)));
                continue;
            }
            all.push((m1, Some(m2)));
            let attacked = match after2.king_square(WHITE) {
                Some(k) => after2.is_attacked_by(BLACK, k),
                None => false,
            };
            if !attacked {
                safe.push((m1, Some(m2)));
            }
        }
    }

    let mut out = wins;
    out.extend(if safe.is_empty() { all } else { safe });
    out
}

/// White's second half-moves: king-safe ones, else all (forced blunder). A
/// capture of the Black king short-circuits to that single move.
pub fn white_second_half_moves(board: &Board) -> Vec<Move> {
    let mut b = board.clone();
    b.turn = true;
    let candidates = generate_pseudo_legal(&b);
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut safe = Vec::new();
    let mut all = Vec::new();
    for m2 in candidates {
        let mut probe = b.clone();
        probe.push(&m2);
        if probe.king_square(BLACK).is_none() {
            return vec![m2];
        }
        all.push(m2);
        let attacked = match probe.king_square(WHITE) {
            Some(k) => probe.is_attacked_by(BLACK, k),
            None => false,
        };
        if !attacked {
            safe.push(m2);
        }
    }
    if safe.is_empty() {
        all
    } else {
        safe
    }
}

/// Black's moves: pseudo-legal, with the same self-preservation policy and the
/// same forced-blunder fallback White gets.
pub fn black_actions(board: &Board, truncate_wins: bool) -> Vec<Move> {
    let mut b = board.clone();
    b.turn = false;
    let candidates = generate_pseudo_legal(&b);
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut wins = Vec::new();
    let mut safe = Vec::new();
    let mut all = Vec::new();
    for mv in candidates {
        let mut probe = b.clone();
        probe.push(&mv);
        if probe.king_square(WHITE).is_none() {
            if truncate_wins {
                return vec![mv];
            }
            wins.push(mv);
            continue;
        }
        all.push(mv);
        let attacked = match probe.king_square(BLACK) {
            Some(k) => probe.is_attacked_by(WHITE, k),
            None => false,
        };
        if !attacked {
            safe.push(mv);
        }
    }
    let mut out = wins;
    out.extend(if safe.is_empty() { all } else { safe });
    out
}

// ---------------------------------------------------------------------------
// Python bindings — FEN in, action strings out, for the differential harness.
// ---------------------------------------------------------------------------

fn pair_uci(m1: &Move, m2: &Option<Move>) -> String {
    match m2 {
        Some(m) => format!("{},{}", m1.uci(), m.uci()),
        None => format!("{},{}", m1.uci(), NULL_UCI),
    }
}

#[pyfunction]
#[pyo3(signature = (fen, truncate_wins=true))]
fn white_actions_uci(fen: &str, truncate_wins: bool) -> PyResult<Vec<String>> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    Ok(white_actions(&board, truncate_wins)
        .iter()
        .map(|(m1, m2)| pair_uci(m1, m2))
        .collect())
}

#[pyfunction]
#[pyo3(signature = (fen, truncate_wins=true))]
fn black_actions_uci(fen: &str, truncate_wins: bool) -> PyResult<Vec<String>> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    Ok(black_actions(&board, truncate_wins)
        .iter()
        .map(|m| m.uci())
        .collect())
}

#[pyfunction]
fn white_single_moves_uci(fen: &str) -> PyResult<Vec<String>> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    Ok(white_single_moves(&board).iter().map(|m| m.uci()).collect())
}

#[pyfunction]
fn white_second_half_uci(fen: &str) -> PyResult<Vec<String>> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    Ok(white_second_half_moves(&board).iter().map(|m| m.uci()).collect())
}

/// Apply one move to a FEN and return the resulting FEN — lets the harness
/// check state-machine parity without a full game object yet.
#[pyfunction]
fn push_uci(fen: &str, uci: &str) -> PyResult<String> {
    let mut board = parse_fen(fen).map_err(PyValueError::new_err)?;
    let mv = parse_uci(uci).ok_or_else(|| PyValueError::new_err("bad uci"))?;
    board.push(&mv);
    Ok(crate::bitboard::to_fen(&board))
}

fn parse_uci(uci: &str) -> Option<Move> {
    let b = uci.as_bytes();
    if b.len() < 4 {
        return None;
    }
    let sq = |f: u8, r: u8| -> Option<u8> {
        let file = f.wrapping_sub(b'a');
        let rank = r.wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(rank * 8 + file)
        } else {
            None
        }
    };
    let from = sq(b[0], b[1])?;
    let to = sq(b[2], b[3])?;
    let promotion = if b.len() > 4 {
        Some(match b[4] {
            b'q' => 5u8,
            b'r' => 4,
            b'b' => 3,
            b'n' => 2,
            _ => return None,
        })
    } else {
        None
    };
    let _ = KING; // keep the import meaningful if the match above changes
    Some(Move { from, to, promotion })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(white_actions_uci, m)?)?;
    m.add_function(wrap_pyfunction!(black_actions_uci, m)?)?;
    m.add_function(wrap_pyfunction!(white_single_moves_uci, m)?)?;
    m.add_function(wrap_pyfunction!(white_second_half_uci, m)?)?;
    m.add_function(wrap_pyfunction!(push_uci, m)?)?;
    Ok(())
}
