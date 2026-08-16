//! Native rules/search core for Monster Chess (DIRECTIVE E1-E4).
//!
//! Parity before improvement (DIRECTIVE §0.1): this crate reproduces the
//! measured behaviour of `src/monster_chess.py` and friends. Nothing here may
//! "fix" the Python engine's quirks — the quirks are the contract, and the
//! differential harness treats any divergence as a defect in *this* crate.

use pyo3::prelude::*;

mod bitboard;
mod encoding;
mod eval;
mod game;
mod mcts;
mod monster;
mod solver;

/// Build identity, so the Python side can assert it loaded the crate it built.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn monster_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    bitboard::register(m)?;
    monster::register(m)?;
    game::register(m)?;
    eval::register(m)?;
    encoding::register(m)?;
    mcts::register(m)?;
    solver::register(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::bitboard::{parse_fen, to_fen};
    use crate::encoding::{encode, CURRENT_TENSOR_CHANNELS, TURN_LAYER};
    use crate::game::Game;
    use crate::monster::{black_actions, white_actions};

    const START: &str =
        "rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1";

    #[test]
    fn fen_round_trip_preserves_the_start_state() {
        let board = parse_fen(START).expect("start FEN parses");
        assert_eq!(to_fen(&board), START);
    }

    #[test]
    fn black_king_capture_truncates_to_the_win() {
        let board = parse_fen("k7/8/8/8/8/8/8/r3K3 b - - 0 1").unwrap();
        let actions = black_actions(&board, true);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].uci(), "a1e1");
    }

    #[test]
    fn white_first_half_king_capture_has_a_null_second_move() {
        let board = parse_fen("8/8/8/8/8/4k3/4K3/8 w - - 0 1").unwrap();
        let actions = white_actions(&board, false);
        assert!(actions
            .iter()
            .any(|(m1, m2)| m1.uci() == "e2e3" && m2.is_none()));
    }

    #[test]
    fn encoding_has_the_expected_shape_and_turn_plane() {
        let board = parse_fen(START).unwrap();
        let tensor = encode(&board, false, false, CURRENT_TENSOR_CHANNELS).unwrap();
        assert_eq!(tensor.len(), 8 * 8 * CURRENT_TENSOR_CHANNELS);
        for square in 0..64 {
            assert_eq!(tensor[square * CURRENT_TENSOR_CHANNELS + TURN_LAYER], -1.0);
        }
    }

    #[test]
    fn white_half_pair_advances_one_turn() {
        let mut game = Game::from_fen(START).unwrap();
        game.apply_half("c2c3").unwrap();
        assert!(game.is_white_turn);
        assert!(game.white_half_pending);
        assert_eq!(game.turn_count, 0);

        game.apply_half("d2d3").unwrap();
        assert!(!game.is_white_turn);
        assert!(!game.white_half_pending);
        assert_eq!(game.turn_count, 1);
    }
}
