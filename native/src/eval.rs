//! The heuristic evaluator, ported from `src/evaluation.py` (DIRECTIVE E2).
//!
//! This is not a place for cleverness. The heuristic is the cap-relabel oracle,
//! the gate's anchor opponent, and the HybridEvaluator's value source, so its
//! *exact* output is the contract — including the order terms are summed in.
//!
//! **Float parity is achievable here and is the gate (|Δ| ≤ 1e-9).** Every
//! operation is +, -, *, / on f64 with integer `2^n`, so there are no libm
//! differences to chase. What there *is* to get wrong is accumulation order:
//! `score` is a running sum, and re-ordering the terms changes the last ulp.
//! So the sections below appear in the same sequence as the Python, and every
//! loop over squares walks ascending square index, matching python-chess's
//! `SquareSet` iteration.
//!
//! Constants are mirrored from `config.py` and asserted equal by the tests
//! rather than imported, so a config edit fails loudly instead of drifting.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bitboard::{
    generate_pseudo_legal, parse_fen, Board, BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE,
};

pub const WHITE_PAWN_VALUE: f64 = 0.18;
pub const PAWN_ELIMINATION_BONUS: f64 = 0.14;
pub const BLOCKED_PAWN_PENALTY: f64 = 0.12;
pub const KING_DISPLACEMENT_WEIGHT: f64 = 0.06;
pub const KING_MOBILITY_WEIGHT: f64 = 0.01;
pub const BARRIER_RANK_FILE_WEIGHT: f64 = 0.12;
pub const PIECE_SAFETY_BONUS: f64 = 0.08;
pub const BLACK_KING_EXPOSURE_PENALTY: f64 = 0.04;
pub const KING_GEOM_SCALE: f64 = 0.4;
pub const KING_ATTACK_SCALE: f64 = 1.8;

const KING_DELTAS: [(i32, i32); 8] = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
];

#[inline]
fn rank_of(sq: u8) -> i32 {
    (sq >> 3) as i32
}

#[inline]
fn file_of(sq: u8) -> i32 {
    (sq & 7) as i32
}

#[inline]
fn square(file: i32, rank: i32) -> u8 {
    (rank * 8 + file) as u8
}

/// Squares of a bitboard ascending — python-chess `SquareSet` iteration order.
fn squares(mut bb: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(bb.count_ones() as usize);
    while bb != 0 {
        out.push(bb.trailing_zeros() as u8);
        bb &= bb - 1;
    }
    out
}

fn pieces(board: &Board, piece: u8, color: usize) -> u64 {
    let bb = match piece {
        PAWN => board.pawns,
        KNIGHT => board.knights,
        BISHOP => board.bishops,
        ROOK => board.rooks,
        QUEEN => board.queens,
        KING => board.kings,
        _ => 0,
    };
    bb & board.occupied_co[color]
}

/// Can White capture the Black king within its full double move?
fn white_can_capture_king(board: &Board) -> bool {
    let bk = match board.king_square(BLACK) {
        Some(sq) => sq,
        None => return false,
    };
    if board.king_square(WHITE).is_none() {
        return false;
    }
    let mut probe = board.clone();
    probe.turn = true;
    for m1 in generate_pseudo_legal(&probe) {
        if m1.to == bk {
            return true;
        }
        let mut after = probe.clone();
        after.push(&m1);
        after.turn = true;
        for m2 in generate_pseudo_legal(&after) {
            if m2.to == bk {
                return true;
            }
        }
    }
    false
}

/// Can White capture the Black king with the single half-move it has left?
fn white_can_capture_king_single(board: &Board) -> bool {
    let bk = match board.king_square(BLACK) {
        Some(sq) => sq,
        None => return false,
    };
    if board.king_square(WHITE).is_none() {
        return false;
    }
    let mut probe = board.clone();
    probe.turn = true;
    generate_pseudo_legal(&probe).iter().any(|m| m.to == bk)
}

fn black_can_capture_king(board: &Board) -> bool {
    let wk = match board.king_square(WHITE) {
        Some(sq) => sq,
        None => return false,
    };
    if board.king_square(BLACK).is_none() {
        return false;
    }
    let mut probe = board.clone();
    probe.turn = false;
    generate_pseudo_legal(&probe).iter().any(|m| m.to == wk)
}

fn is_passed_pawn(board: &Board, sq: u8, color: usize) -> bool {
    let file = file_of(sq);
    let rank = rank_of(sq);
    let opp = if color == WHITE { BLACK } else { WHITE };
    let ranks: Vec<i32> = if color == WHITE {
        (rank + 1..8).collect()
    } else {
        (0..rank).collect()
    };
    for f in [file - 1, file, file + 1] {
        if !(0..8).contains(&f) {
            continue;
        }
        for &r in &ranks {
            let s = square(f, r);
            if board.pawns & board.occupied_co[opp] & (1u64 << s) != 0 {
                return false;
            }
        }
    }
    true
}

/// The clamps that run **before any NN call**, matching
/// `NNEvaluator._batch_impl` / `evaluate_with_policy`.
///
/// §1.3 of the contract: these are not an optimisation, they are part of the
/// evaluation. A port that skips them sends decided positions to the network
/// and gets a different value for them — and because the policy is `None` in
/// this branch, the node must still be expanded with *uniform* priors.
/// Threat-against-the-mover deliberately does not clamp (the sacrificial-check
/// bug).
pub fn pre_nn_clamp(board: &Board, is_white_turn: bool, white_half_pending: bool) -> Option<f64> {
    if board.king_square(WHITE).is_none() {
        return Some(-1.0);
    }
    if board.king_square(BLACK).is_none() {
        return Some(1.0);
    }
    if is_white_turn {
        let threat = if white_half_pending {
            white_can_capture_king_single(board)
        } else {
            white_can_capture_king(board)
        };
        if threat {
            return Some(0.95);
        }
    } else if black_can_capture_king(board) {
        return Some(-0.95);
    }
    None
}

/// The heuristic. `is_white_turn` / `white_half_pending` come from the game
/// state, not the board, because the threat scan is pending-aware.
pub fn evaluate(board: &Board, is_white_turn: bool, white_half_pending: bool) -> f64 {
    evaluate_scaled(board, is_white_turn, white_half_pending, KING_GEOM_SCALE, KING_ATTACK_SCALE)
}

pub fn evaluate_scaled(
    board: &Board,
    is_white_turn: bool,
    white_half_pending: bool,
    king_geom_scale: f64,
    king_attack_scale: f64,
) -> f64 {
    // Terminal
    let wk = match board.king_square(WHITE) {
        Some(sq) => sq,
        None => return -1.0,
    };
    let bk = match board.king_square(BLACK) {
        Some(sq) => sq,
        None => return 1.0,
    };

    // Capture threat scan — side to move only. A threat *against* the mover
    // deliberately does not clamp (the sacrificial-check bug).
    if is_white_turn {
        let threat = if white_half_pending {
            white_can_capture_king_single(board)
        } else {
            white_can_capture_king(board)
        };
        if threat {
            return 0.95;
        }
    } else if black_can_capture_king(board) {
        return -0.95;
    }

    let mut score: f64 = 0.0;
    let wk_rank = rank_of(wk);
    let wk_file = file_of(wk);
    let bk_rank = rank_of(bk);
    let bk_file = file_of(bk);

    // ---- White's chances ----
    let white_pawns = pieces(board, PAWN, WHITE).count_ones() as f64;
    score += white_pawns * WHITE_PAWN_VALUE;

    let mut blocked_pawns = 0.0f64;
    for sq in squares(pieces(board, PAWN, WHITE)) {
        let rank = rank_of(sq);
        let file = file_of(sq);
        score += (rank - 1) as f64 * 0.05;
        if is_passed_pawn(board, sq, WHITE) {
            let passed_bonus = 0.02 * (1u64 << (rank - 2).max(0)) as f64;
            score += passed_bonus;
            let support_dist = (wk_rank - rank).abs().max((wk_file - file).abs());
            if support_dist <= 2 {
                score += passed_bonus * 0.5;
            }
        }
        if rank + 1 <= 7 {
            let advance = square(file, rank + 1);
            if board.occupied & (1u64 << advance) != 0 {
                blocked_pawns += 1.0;
            } else if board.is_attacked_by(BLACK, advance) && !board.is_attacked_by(WHITE, advance)
            {
                blocked_pawns += 1.0;
            }
        }
    }
    score -= blocked_pawns * BLOCKED_PAWN_PENALTY;

    let white_queens = pieces(board, QUEEN, WHITE).count_ones() as f64;
    score += white_queens * 0.30;

    // ---- King tropism ----
    let king_dist = (wk_rank - bk_rank).abs().max((wk_file - bk_file).abs());
    if king_dist <= 2 {
        score += 0.15 * king_attack_scale;
    } else if king_dist <= 4 {
        score += 0.05 * king_attack_scale;
    }

    for sq in 0u8..64 {
        if board.occupied_co[BLACK] & (1u64 << sq) == 0 {
            continue;
        }
        let piece = match board.piece_type_at(sq) {
            Some(p) if p != KING => p,
            _ => continue,
        };
        let dist = (wk_rank - rank_of(sq)).abs().max((wk_file - file_of(sq)).abs());
        if dist <= 2 && !board.is_attacked_by(BLACK, sq) {
            let value = match piece {
                PAWN => 0.03,
                KNIGHT | BISHOP => 0.08,
                ROOK => 0.12,
                QUEEN => 0.20,
                _ => 0.05,
            };
            score += value * king_attack_scale;
        }
    }

    // ---- Black's chances ----
    let black_queens = pieces(board, QUEEN, BLACK).count_ones() as f64;
    let black_rooks = pieces(board, ROOK, BLACK).count_ones() as f64;
    let black_heavy = black_queens + black_rooks;
    score -= black_heavy * 0.08;

    for sq in squares(pieces(board, PAWN, BLACK)) {
        score -= (6 - rank_of(sq)) as f64 * 0.03;
    }

    if black_queens > 1.0 {
        score -= (black_queens - 1.0) * 0.35;
    }

    let eliminated_pawns = 4.0 - white_pawns;
    score -= eliminated_pawns * PAWN_ELIMINATION_BONUS;

    let heavy_bb = pieces(board, QUEEN, BLACK) | pieces(board, ROOK, BLACK);
    let mut safe_heavy = 0.0f64;
    for sq in squares(heavy_bb) {
        let dist = (wk_rank - rank_of(sq)).abs().max((wk_file - file_of(sq)).abs());
        if dist >= 3 {
            safe_heavy += 1.0;
        }
    }
    score -= safe_heavy * PIECE_SAFETY_BONUS;

    let mut bk_attacked_adj = 0.0f64;
    for (dr, df) in KING_DELTAS {
        let (r, f) = (bk_rank + dr, bk_file + df);
        if (0..8).contains(&r) && (0..8).contains(&f) && board.is_attacked_by(WHITE, square(f, r)) {
            bk_attacked_adj += 1.0;
        }
    }
    score += bk_attacked_adj * BLACK_KING_EXPOSURE_PENALTY;

    // ---- King confinement & sub-goals ----
    let heavy_scale = (black_heavy + 1.0).min(4.0) / 4.0;

    let rank_from_center = (wk_rank as f64 - 3.5).abs();
    let file_from_center = (wk_file as f64 - 3.5).abs();
    let displacement = (rank_from_center + file_from_center) / 7.0;
    score -= displacement * heavy_scale * KING_DISPLACEMENT_WEIGHT * king_geom_scale;

    let rank_edge = wk_rank.min(7 - wk_rank);
    let file_edge = wk_file.min(7 - wk_file);
    let edge_dist = rank_edge.min(file_edge);
    score -= (3 - edge_dist) as f64 * 0.10 * heavy_scale * king_geom_scale;

    let mut adjacent_attacked = 0.0f64;
    for (dr, df) in KING_DELTAS {
        let (r, f) = (wk_rank + dr, wk_file + df);
        if (0..8).contains(&r) && (0..8).contains(&f) && board.is_attacked_by(BLACK, square(f, r)) {
            adjacent_attacked += 1.0;
        }
    }
    score -= adjacent_attacked * 0.06 * heavy_scale;

    for sq in squares(heavy_bb) {
        if rank_of(sq) == wk_rank || file_of(sq) == wk_file {
            score -= 0.08;
        }
    }

    if black_heavy >= 1.0 {
        let mut reachable = [false; 64];
        for (d1r, d1f) in KING_DELTAS {
            let (r1, f1) = (wk_rank + d1r, wk_file + d1f);
            if !(0..8).contains(&r1) || !(0..8).contains(&f1) {
                continue;
            }
            let sq1 = square(f1, r1);
            // Own non-king pieces block (pawns).
            if board.occupied_co[WHITE] & (1u64 << sq1) != 0
                && board.piece_type_at(sq1) != Some(KING)
            {
                continue;
            }
            for (d2r, d2f) in KING_DELTAS {
                let (r2, f2) = (r1 + d2r, f1 + d2f);
                if (0..8).contains(&r2) && (0..8).contains(&f2) {
                    let sq2 = square(f2, r2);
                    if !board.is_attacked_by(BLACK, sq2) {
                        reachable[sq2 as usize] = true;
                    }
                }
            }
        }
        let count = reachable.iter().filter(|x| **x).count() as i32;
        let mobility_loss = (24 - count).max(0) as f64;
        score -= mobility_loss * KING_MOBILITY_WEIGHT;
    }

    // Barrier quality
    let rank_to_edge = wk_rank.min(7 - wk_rank);
    let file_to_edge = wk_file.min(7 - wk_file);
    let heavy_list = squares(heavy_bb);
    let mut barriers = 0.0f64;
    if rank_to_edge <= file_to_edge {
        let step = if wk_rank <= 3 { -1 } else { 1 };
        let mut r = wk_rank + step;
        while (0..8).contains(&r) {
            let mut has_barrier = false;
            for &sq in &heavy_list {
                if rank_of(sq) == r {
                    let dist = (r - wk_rank).abs().max((file_of(sq) - wk_file).abs());
                    if dist >= 3 {
                        has_barrier = true;
                        break;
                    }
                }
            }
            if !has_barrier {
                break;
            }
            barriers += 1.0;
            r += step;
        }
    } else {
        let step = if wk_file <= 3 { -1 } else { 1 };
        let mut f = wk_file + step;
        while (0..8).contains(&f) {
            let mut has_barrier = false;
            for &sq in &heavy_list {
                if file_of(sq) == f {
                    let dist = (rank_of(sq) - wk_rank).abs().max((f - wk_file).abs());
                    if dist >= 3 {
                        has_barrier = true;
                        break;
                    }
                }
            }
            if !has_barrier {
                break;
            }
            barriers += 1.0;
            f += step;
        }
    }
    score -= barriers * BARRIER_RANK_FILE_WEIGHT;

    // ---- Material balance ----
    let black_knights = pieces(board, KNIGHT, BLACK).count_ones() as f64;
    let black_bishops = pieces(board, BISHOP, BLACK).count_ones() as f64;
    score -= (black_knights + black_bishops) * 0.03;

    score.clamp(-0.95, 0.95)
}

#[pyfunction]
#[pyo3(signature = (fen, is_white_turn, white_half_pending=false))]
fn evaluate_fen(fen: &str, is_white_turn: bool, white_half_pending: bool) -> PyResult<f64> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    Ok(evaluate(&board, is_white_turn, white_half_pending))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_fen, m)?)?;
    m.add("WHITE_PAWN_VALUE", WHITE_PAWN_VALUE)?;
    m.add("KING_GEOM_SCALE", KING_GEOM_SCALE)?;
    m.add("KING_ATTACK_SCALE", KING_ATTACK_SCALE)?;
    Ok(())
}
