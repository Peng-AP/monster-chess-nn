//! The Monster Chess game state machine, ported from `src/monster_chess.py`.
//!
//! The delicate part is not the rules but the *turn bookkeeping*, because
//! White's turn is two plies and python-chess flips `board.turn` on every push.
//! The Python engine forces it back to WHITE between halves so `fen()` keeps
//! saying "White to move", and it does so **only when the Black king survives**
//! — a first half-move that captures the king leaves the turn flipped. That
//! asymmetry is reproduced here exactly; it is visible in every recorded FEN.
//!
//! `turn_count` counts completed turns, not plies: it advances on White's
//! second half and on each Black move.
//!
//! **Terminal is king-absence, plus the move-limit cap.** At the cap the result
//! is relabelled by the *sign of the heuristic*, symmetrically: ±0.5 beyond
//! |0.4|, else 0. That is why the native core cannot leave the evaluator behind
//! in Python — search reaches the cap in-tree during late games.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bitboard::{generate_pseudo_legal, parse_fen, to_fen, Board, Move, BLACK, WHITE};
use crate::eval::evaluate;
use crate::monster::{black_actions, white_actions, white_second_half_moves, white_single_moves};

/// Mirrors `config.MAX_GAME_TURNS`. Asserted against Python by the test suite.
pub const MAX_GAME_TURNS: u32 = 150;

#[pyclass]
#[derive(Clone)]
pub struct Game {
    board: Board,
    #[pyo3(get)]
    pub is_white_turn: bool,
    #[pyo3(get)]
    pub turn_count: u32,
    #[pyo3(get)]
    pub white_half_pending: bool,
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
    Some(Move { from, to, promotion })
}

impl Game {
    /// Rust-side constructor. `#[pymethods]` cannot be called across modules,
    /// and the search needs to build states without touching the interpreter.
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let board = parse_fen(fen)?;
        let is_white_turn = board.turn;
        Ok(Game {
            board,
            is_white_turn,
            turn_count: 0,
            white_half_pending: false,
        })
    }

    pub fn fen_string(&self) -> String {
        to_fen(&self.board)
    }

    pub fn board_ref(&self) -> &Board {
        &self.board
    }

    pub fn apply_half(&mut self, uci: &str) -> Result<(), String> {
        let mv = parse_uci(uci).ok_or_else(|| "bad uci".to_string())?;
        if self.is_white_turn && !self.white_half_pending {
            self.board.push(&mv);
            // Only restore White's turn if the Black king survived; a
            // king-capturing first half leaves the flip in place.
            if self.board.king_square(BLACK).is_some() {
                self.board.turn = true;
            }
            self.white_half_pending = true;
        } else if self.is_white_turn {
            self.board.turn = true;
            self.board.push(&mv);
            self.white_half_pending = false;
            self.is_white_turn = false;
            self.turn_count += 1;
        } else {
            self.board.push(&mv);
            self.is_white_turn = true;
            self.turn_count += 1;
        }
        Ok(())
    }
}

#[pymethods]
impl Game {
    #[new]
    fn new(fen: &str) -> PyResult<Self> {
        Game::from_fen(fen).map_err(PyValueError::new_err)
    }

    fn fen(&self) -> String {
        self.fen_string()
    }

    fn clone_game(&self) -> Game {
        self.clone()
    }

    /// True on king absence or at the move-limit cap.
    fn is_terminal(&self) -> bool {
        self.board.king_square(WHITE).is_none()
            || self.board.king_square(BLACK).is_none()
            || self.turn_count >= MAX_GAME_TURNS
    }

    /// +1 White, -1 Black, ±0.5 or 0 at the move-limit cap, None if not terminal.
    fn result(&self) -> Option<f64> {
        if self.board.king_square(WHITE).is_none() {
            Some(-1.0)
        } else if self.board.king_square(BLACK).is_none() {
            Some(1.0)
        } else if self.turn_count >= MAX_GAME_TURNS {
            // Position-dependent proxy, applied equally in both directions --
            // a game reaching the cap with one side decisively ahead is not a
            // true draw. Only a king capture is a *win* (owner, 2026-08-03);
            // this value is a training label, not a scoreline.
            let h = evaluate(&self.board, self.is_white_turn, self.white_half_pending);
            Some(if h < -0.4 {
                -0.5
            } else if h > 0.4 {
                0.5
            } else {
                0.0
            })
        } else {
            None
        }
    }

    /// The heuristic at this state, for callers that want it directly.
    fn evaluate(&self) -> f64 {
        evaluate(&self.board, self.is_white_turn, self.white_half_pending)
    }

    fn at_turn_cap(&self) -> bool {
        self.turn_count >= MAX_GAME_TURNS
            && self.board.king_square(WHITE).is_some()
            && self.board.king_square(BLACK).is_some()
    }

    /// Complete legality oracle. White: "m1,m2" pairs. Black: single moves.
    #[pyo3(signature = (truncate_wins=false))]
    fn legal_actions(&self, truncate_wins: bool) -> Vec<String> {
        if self.is_terminal() {
            return Vec::new();
        }
        if self.is_white_turn {
            white_actions(&self.board, truncate_wins)
                .iter()
                .map(|(m1, m2)| match m2 {
                    Some(m) => format!("{},{}", m1.uci(), m.uci()),
                    None => format!("{},0000", m1.uci()),
                })
                .collect()
        } else {
            black_actions(&self.board, truncate_wins)
                .iter()
                .map(|m| m.uci())
                .collect()
        }
    }

    /// The half-move (search) API.
    fn search_actions(&self) -> Vec<String> {
        if self.is_terminal() {
            return Vec::new();
        }
        let moves = if !self.is_white_turn {
            black_actions(&self.board, true)
        } else if self.white_half_pending {
            white_second_half_moves(&self.board)
        } else {
            white_single_moves(&self.board)
        };
        moves.iter().map(|m| m.uci()).collect()
    }

    fn apply_search_action(&mut self, uci: &str) -> PyResult<()> {
        self.apply_half(uci).map_err(PyValueError::new_err)
    }

    /// Atomic action: "m1,m2" for White (m2 may be "0000"), single uci for Black.
    fn apply_action(&mut self, action: &str) -> PyResult<()> {
        if self.is_white_turn {
            let (a, b) = action
                .split_once(',')
                .ok_or_else(|| PyValueError::new_err("white action needs 'm1,m2'"))?;
            let m1 = parse_uci(a).ok_or_else(|| PyValueError::new_err("bad m1"))?;
            self.board.push(&m1);
            if self.board.king_square(BLACK).is_some() && b != "0000" {
                self.board.turn = true;
                let m2 = parse_uci(b).ok_or_else(|| PyValueError::new_err("bad m2"))?;
                // m1 may have made m2 illegal; the Python engine re-checks.
                if generate_pseudo_legal(&self.board).iter().any(|m| *m == m2) {
                    self.board.push(&m2);
                }
            }
        } else {
            let mv = parse_uci(action).ok_or_else(|| PyValueError::new_err("bad uci"))?;
            self.board.push(&mv);
        }
        self.is_white_turn = !self.is_white_turn;
        self.turn_count += 1;
        self.white_half_pending = false;
        Ok(())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add("MAX_GAME_TURNS", MAX_GAME_TURNS)?;
    Ok(())
}
