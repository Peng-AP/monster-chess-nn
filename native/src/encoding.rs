//! Position encoding, ported from `src/encoding.py` (DIRECTIVE E2).
//!
//! Emitted in `(8, 8, C)` rank-major order — numpy's C order for that shape —
//! so the gate can compare byte-for-byte against `fen_to_tensor`. The NN bridge
//! wants `(N, C, 8, 8)`; that transpose belongs at the bridge (D3), not here,
//! because a transpose baked in at this layer could not be checked against the
//! Python reference at all.
//!
//! Two layouts, selected by channel count exactly as the Python does:
//!   * **17ch (current)** — 0-11 pieces, 12 turn, 13 half-move, 14 signed rank
//!     coordinate, 15/16 White/Black pawn progress.
//!   * **15ch (legacy v16/v17)** — same 0-13, then channel 14 carries
//!     White-pawn advancement only.
//!
//! Selecting by count rather than by a flag is what stops an incompatible
//! encoding loading silently against an old checkpoint, so the port keeps it.
//!
//! Arithmetic is done in f64 and narrowed to f32 on store, matching numpy
//! assigning a Python float into a float32 array. Computing in f32 throughout
//! would round differently and break byte-equality.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bitboard::{parse_fen, Board, BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

pub const TURN_LAYER: usize = 12;
pub const MOVE_COUNT_LAYER: usize = 13;
pub const LEGACY_TENSOR_CHANNELS: usize = 15;
pub const LEGACY_PAWN_ADVANCEMENT_LAYER: usize = 14;
pub const RANK_COORD_LAYER: usize = 14;
pub const WHITE_PAWN_PROGRESS_LAYER: usize = 15;
pub const BLACK_PAWN_PROGRESS_LAYER: usize = 16;
pub const CURRENT_TENSOR_CHANNELS: usize = 17;

fn piece_layer(piece: u8, color: usize) -> usize {
    let base = match piece {
        PAWN => 0,
        KNIGHT => 1,
        BISHOP => 2,
        ROOK => 3,
        QUEEN => 4,
        KING => 5,
        _ => 0,
    };
    if color == WHITE {
        base
    } else {
        base + 6
    }
}

pub fn encode(
    board: &Board,
    is_white_turn: bool,
    half_pending: bool,
    channels: usize,
) -> Result<Vec<f32>, String> {
    if channels != LEGACY_TENSOR_CHANNELS && channels != CURRENT_TENSOR_CHANNELS {
        return Err(format!("Unsupported position encoding with {channels} channels"));
    }
    let mut tensor = vec![0.0f32; 8 * 8 * channels];
    let at = |rank: usize, file: usize, layer: usize| rank * 8 * channels + file * channels + layer;

    for sq in 0u8..64 {
        if let Some(piece) = board.piece_type_at(sq) {
            let color = if board.occupied_co[WHITE] & (1u64 << sq) != 0 {
                WHITE
            } else {
                BLACK
            };
            let rank = (sq >> 3) as usize;
            let file = (sq & 7) as usize;
            tensor[at(rank, file, piece_layer(piece, color))] = 1.0;
        }
    }

    let turn_value: f32 = if is_white_turn { 1.0 } else { -1.0 };
    let half_value: f32 = if half_pending { 1.0 } else { 0.0 };
    for rank in 0..8 {
        for file in 0..8 {
            tensor[at(rank, file, TURN_LAYER)] = turn_value;
            tensor[at(rank, file, MOVE_COUNT_LAYER)] = half_value;
        }
    }

    let white_pawns = board.pawns & board.occupied_co[WHITE];
    let black_pawns = board.pawns & board.occupied_co[BLACK];

    if channels == LEGACY_TENSOR_CHANNELS {
        for sq in 0u8..64 {
            if white_pawns & (1u64 << sq) != 0 {
                let rank = (sq >> 3) as usize;
                let file = (sq & 7) as usize;
                let v = (rank as f64 - 1.0) / 6.0;
                tensor[at(rank, file, LEGACY_PAWN_ADVANCEMENT_LAYER)] = v as f32;
            }
        }
        return Ok(tensor);
    }

    for rank in 0..8 {
        let v = ((rank as f64) - 3.5) / 3.5;
        for file in 0..8 {
            tensor[at(rank, file, RANK_COORD_LAYER)] = v as f32;
        }
    }
    for sq in 0u8..64 {
        let rank = (sq >> 3) as usize;
        let file = (sq & 7) as usize;
        if white_pawns & (1u64 << sq) != 0 {
            let v = (((rank as f64) - 1.0) / 6.0).clamp(0.0, 1.0);
            tensor[at(rank, file, WHITE_PAWN_PROGRESS_LAYER)] = v as f32;
        }
        if black_pawns & (1u64 << sq) != 0 {
            let v = ((6.0 - (rank as f64)) / 6.0).clamp(0.0, 1.0);
            tensor[at(rank, file, BLACK_PAWN_PROGRESS_LAYER)] = v as f32;
        }
    }
    Ok(tensor)
}

#[pyfunction]
#[pyo3(signature = (fen, is_white_turn=true, half_pending=false, channels=17))]
fn encode_fen(
    fen: &str,
    is_white_turn: bool,
    half_pending: bool,
    channels: usize,
) -> PyResult<Vec<f32>> {
    let board = parse_fen(fen).map_err(PyValueError::new_err)?;
    encode(&board, is_white_turn, half_pending, channels).map_err(PyValueError::new_err)
}

pub const LEGACY_POLICY_SIZE: usize = 4096;
pub const PROMOTION_AWARE_POLICY_SIZE: usize = 4288;

/// Policy index for the legacy ABI or the distinct-promotion extension.
pub fn policy_index(uci: &str, promotion_aware: bool) -> Result<usize, String> {
    let b = uci.as_bytes();
    if b.len() < 4 {
        return Err("bad uci".to_string());
    }
    let sq = |f: u8, r: u8| -> Result<usize, String> {
        let file = f.wrapping_sub(b'a');
        let rank = r.wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Ok((rank * 8 + file) as usize)
        } else {
            Err("bad square".to_string())
        }
    };
    let from = sq(b[0], b[1])?;
    let to = sq(b[2], b[3])?;
    if !promotion_aware || b.len() < 5 {
        return Ok(from * 64 + to);
    }
    let color_band = match (b[1], b[3]) {
        (b'7', b'8') => 0usize,
        (b'2', b'1') => 1usize,
        _ => return Err("promotion move does not cross a promotion rank".to_string()),
    };
    let direction = b[2] as i16 - b[0] as i16;
    if !(-1..=1).contains(&direction) {
        return Err("bad promotion destination".to_string());
    }
    let piece = match b[4].to_ascii_lowercase() {
        b'q' => 0usize,
        b'r' => 1usize,
        b'b' => 2usize,
        b'n' => 3usize,
        _ => return Err("bad promotion piece".to_string()),
    };
    let from_file = (b[0] - b'a') as usize;
    Ok(LEGACY_POLICY_SIZE + color_band * 96 + from_file * 12
       + (direction + 1) as usize * 4 + piece)
}

/// Flat legacy policy index, `from * 64 + to`.
#[pyfunction]
fn move_to_index(uci: &str) -> PyResult<usize> {
    policy_index(uci, false).map_err(PyValueError::new_err)
}

#[pyfunction]
fn promotion_move_to_index(uci: &str) -> PyResult<usize> {
    policy_index(uci, true).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_fen, m)?)?;
    m.add_function(wrap_pyfunction!(move_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(promotion_move_to_index, m)?)?;
    Ok(())
}
