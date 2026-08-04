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
use crate::bitboard::square_name_pub;
use crate::eval::evaluate;
use crate::monster::{black_actions, white_actions, white_second_half_moves, white_single_moves};

/// Mirrors `config.MAX_GAME_TURNS`. Asserted against Python by the test suite.
pub const MAX_GAME_TURNS: u32 = 150;

/// How many plies of move history to carry. Mirrors `CLONE_HISTORY_PLIES`:
/// only the oscillation detector reads it, at offsets -1/-3/-4.
pub const HISTORY_PLIES: usize = 8;

#[pyclass]
#[derive(Clone)]
pub struct Game {
    board: Board,
    /// Recent (from, to) squares, oldest first, capped at HISTORY_PLIES.
    /// A FEN-constructed position starts empty — no history, no oscillation
    /// override — and history accrues as the driver applies actions, exactly
    /// as python-chess's `move_stack` does.
    pub history: Vec<(u8, u8)>,
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
        Game::from_state(fen, false, 0)
    }

    /// A FEN alone does **not** determine a Monster Chess state: it cannot say
    /// whether White has already spent the first half of its turn, and that
    /// flag selects a completely different action set and a different
    /// (pending-aware) threat scan. Constructing from a FEN alone silently
    /// assumed `pending = false` and made two engines search different
    /// positions -- found while measuring search agreement, 2026-08-03.
    pub fn from_state(fen: &str, white_half_pending: bool, turn_count: u32) -> Result<Self, String> {
        Game::from_state_with_history(fen, white_half_pending, turn_count, &[])
    }

    /// The third thing a FEN cannot carry, after `white_half_pending` and
    /// `turn_count`: **recent move history**. The oscillation penalty reads it
    /// at offsets -1/-3/-4, so a state rebuilt without it silently loses the
    /// penalty — the override quietly stops firing rather than failing.
    pub fn from_state_with_history(
        fen: &str,
        white_half_pending: bool,
        turn_count: u32,
        history: &[String],
    ) -> Result<Self, String> {
        let board = parse_fen(fen)?;
        let is_white_turn = board.turn;
        let mut recent: Vec<(u8, u8)> = Vec::new();
        for uci in history.iter().rev().take(HISTORY_PLIES).rev() {
            if let Some(mv) = parse_uci(uci) {
                recent.push((mv.from, mv.to));
            }
        }
        Ok(Game {
            board,
            history: recent,
            is_white_turn,
            turn_count,
            white_half_pending: white_half_pending && is_white_turn,
        })
    }

    pub fn fen_string(&self) -> String {
        to_fen(&self.board)
    }

    pub fn board_ref(&self) -> &Board {
        &self.board
    }

    /// Terminal test, callable from the search without the interpreter.
    pub fn is_terminal_rust(&self) -> bool {
        self.board.king_square(WHITE).is_none()
            || self.board.king_square(BLACK).is_none()
            || self.turn_count >= MAX_GAME_TURNS
    }

    /// Terminal result, including the cap's heuristic relabel. None if live.
    pub fn result_rust(&self) -> Option<f64> {
        if self.board.king_square(WHITE).is_none() {
            Some(-1.0)
        } else if self.board.king_square(BLACK).is_none() {
            Some(1.0)
        } else if self.turn_count >= MAX_GAME_TURNS {
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

    /// The half-move action list, as UCI strings.
    pub fn search_actions_rust(&self) -> Vec<String> {
        if self.is_terminal_rust() {
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

    fn record_history(&mut self, mv: &Move) {
        if self.history.len() == HISTORY_PLIES {
            self.history.remove(0);
        }
        self.history.push((mv.from, mv.to));
    }

    /// The side-to-move's own recent moves, for oscillation detection.
    ///
    /// Offsets follow the fixed push cycle (Black 1 push, White 2). A
    /// misaligned offset can only surface an OPPONENT move, whose exact
    /// reversal is never a legal own move, so false positives are excluded
    /// structurally rather than by checking.
    pub fn own_previous_moves(&self) -> Vec<(u8, u8)> {
        let offsets: &[usize] = if self.is_white_turn { &[1, 3, 4] } else { &[3] };
        let len = self.history.len();
        offsets
            .iter()
            .filter(|&&o| len >= o)
            .map(|&o| self.history[len - o])
            .collect()
    }

    /// True when the action to be selected completes a turn: Black's move, or
    /// White's SECOND half. White's first half may pass through check.
    pub fn turn_completing(&self) -> bool {
        !self.is_white_turn || self.white_half_pending
    }

    pub fn apply_half(&mut self, uci: &str) -> Result<(), String> {
        let mv = parse_uci(uci).ok_or_else(|| "bad uci".to_string())?;
        self.record_history(&mv);
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
    #[pyo3(signature = (fen, white_half_pending=false, turn_count=0, history=None))]
    fn new(
        fen: &str,
        white_half_pending: bool,
        turn_count: u32,
        history: Option<Vec<String>>,
    ) -> PyResult<Self> {
        Game::from_state_with_history(
            fen,
            white_half_pending,
            turn_count,
            &history.unwrap_or_default(),
        )
        .map_err(PyValueError::new_err)
    }

    /// Recent moves as "fromto" square pairs, oldest first.
    fn history_uci(&self) -> Vec<String> {
        self.history
            .iter()
            .map(|(f, t)| format!("{}{}", square_name_pub(*f), square_name_pub(*t)))
            .collect()
    }

    /// The side-to-move's own recent moves, per the oscillation offsets.
    fn own_previous_uci(&self) -> Vec<String> {
        self.own_previous_moves()
            .iter()
            .map(|(f, t)| format!("{}{}", square_name_pub(*f), square_name_pub(*t)))
            .collect()
    }

    fn turn_completing_py(&self) -> bool {
        self.turn_completing()
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
