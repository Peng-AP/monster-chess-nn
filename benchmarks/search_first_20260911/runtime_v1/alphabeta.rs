//! Opt-in search-first prototype. White max/max, Black min; depth is completed
//! turns. No selective pruning, no captures-only qsearch, no TT score reuse.
//! TT stores ordering moves only: repetition is history-dependent. Unlike the
//! incumbent's training-label terminal, turn caps here score an actual draw.
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use crate::cheap_value::{CheapValue, Weights};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::game::{Game, MAX_GAME_TURNS};
use crate::bitboard::{WHITE, BLACK};

fn repetition_key(g: &Game) -> String {
    // Settled nodes only. Callers provide keys in this same four-field format.
    g.fen_string().split_whitespace().take(4).collect::<Vec<_>>().join(" ")
}

struct Search {
    deadline: Instant,
    node_limit: u64,
    nodes: u64,
    cutoffs: u64,
    ordering: HashMap<String, String>,
    path: Vec<String>,
    prune: bool,
    evaluator: Option<Arc<Weights>>,
}

impl Search {
    fn visit(&mut self, g: &Game, depth: u32, mut alpha: f64, mut beta: f64)
        -> Result<(f64, Option<String>), ()>
    {
        if self.nodes >= self.node_limit || Instant::now() >= self.deadline {
            return Err(());
        }
        self.nodes += 1;
        let b = g.board_ref();
        // Captures take precedence over clocks/repetition.
        if b.king_square(WHITE).is_none() { return Ok((-1.0, None)); }
        if b.king_square(BLACK).is_none() { return Ok((1.0, None)); }
        if g.turn_count >= MAX_GAME_TURNS { return Ok((0.0, None)); }
        let settled = !g.white_half_pending;
        let rep = if settled { Some(repetition_key(g)) } else { None };
        if let Some(ref k) = rep {
            if self.path.iter().filter(|x| *x == k).count() >= 2 {
                return Ok((0.0, None));
            }
        }
        // Never evaluate a White turn halfway through. Initial plumbing uses
        // the existing heuristic, not an assumed cheap or adequate evaluator.
        if depth == 0 && settled {
            // This scan proves a capture by the mover within the current turn;
            // it is not a neural estimate. A finite network output must never
            // outrank an exact win or make an exact loss look preferable.
            if let Some(proven) = crate::eval::pre_nn_clamp(b, g.is_white_turn, false) {
                return Ok((proven.signum(), None));
            }
            let value = match &self.evaluator {
                Some(net) => net.evaluate(g),
                None => crate::eval::evaluate(b, g.is_white_turn, false),
            };
            return Ok((value.clamp(-0.99, 0.99), None));
        }
        if let Some(k) = rep { self.path.push(k); }
        let answer = self.children(g, depth, &mut alpha, &mut beta);
        if settled { self.path.pop(); }
        answer
    }

    fn children(&mut self, g: &Game, depth: u32, alpha: &mut f64, beta: &mut f64)
        -> Result<(f64, Option<String>), ()>
    {
        let key = format!("{} {} {}", g.fen_string(), g.white_half_pending, g.turn_count);
        let mut moves = g.search_actions_rust();
        let preferred = self.ordering.get(&key);
        // Stable ordering, no omission. Captures/promotions are ordering only.
        moves.sort_by_key(|m| {
            let sq = (m.as_bytes()[3] - b'1') * 8 + m.as_bytes()[2] - b'a';
            let capture = g.board_ref().piece_type_at(sq).unwrap_or(0) as i32;
            -(if preferred == Some(m) { 1000 } else { 0 }
              + capture * 10 + if m.len() == 5 { 9 } else { 0 })
        });
        if moves.is_empty() { return Ok((0.0, None)); }
        let mut best = if g.is_white_turn { -2.0 } else { 2.0 };
        let mut best_move = None;
        for mv in moves {
            let mut child = g.clone();
            child.apply_half(&mv).expect("generated move parses");
            let next_depth = if g.turn_completing() { depth.saturating_sub(1) } else { depth };
            let (value, _) = self.visit(&child, next_depth, *alpha, *beta)?;
            if best_move.is_none() || (g.is_white_turn && value > best)
                || (!g.is_white_turn && value < best) {
                best = value;
                best_move = Some(mv);
            }
            if g.is_white_turn { *alpha = alpha.max(best); }
            else { *beta = beta.min(best); }
            if self.prune && *alpha >= *beta { self.cutoffs += 1; break; }
        }
        if let Some(ref mv) = best_move {
            // Bounded prototype table; eviction only affects ordering.
            if self.ordering.len() < 200_000 { self.ordering.insert(key, mv.clone()); }
        }
        Ok((best, best_move))
    }
}

#[pyclass(get_all)]
pub struct AlphaBetaResult {
    pub action: Option<String>,
    /// Fixed White perspective. None if no iteration completed.
    pub value: Option<f64>,
    pub completed_depth: u32,
    pub nodes: u64,
    pub cutoffs: u64,
    pub elapsed_seconds: f64,
    pub interrupted: bool,
}

#[pyfunction]
#[pyo3(signature = (fen, pending=false, turn_count=0, seconds=1.0,
                   max_depth=12, node_limit=10000000, prior_positions=None, evaluator=None))]
fn alphabeta_search(py: Python<'_>, fen: &str, pending: bool, turn_count: u32,
                    seconds: f64, max_depth: u32, node_limit: u64,
                    prior_positions: Option<Vec<String>>, evaluator: Option<PyRef<'_, CheapValue>>) -> PyResult<AlphaBetaResult> {
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 86400.0
        || max_depth == 0 || max_depth > 150 || node_limit == 0 {
        return Err(PyValueError::new_err("positive finite limits required; depth <= 150, seconds <= 86400"));
    }
    let game = Game::from_state(fen, pending, turn_count).map_err(PyValueError::new_err)?;
    let prior = prior_positions.unwrap_or_default();
    let evaluator = evaluator.map(|e| Arc::clone(&e.weights));
    // Historical settled keys EXCLUDING current position. Strict format avoids
    // silently accepting six-field FENs that never compare equal.
    if prior.iter().any(|x| x.split_whitespace().count() != 4) {
        return Err(PyValueError::new_err("prior_positions needs settled four-field FEN keys, excluding root"));
    }
    Ok(py.detach(move || {
        let start = Instant::now();
        let mut search = Search { deadline: start + Duration::from_secs_f64(seconds),
            node_limit, nodes: 0, cutoffs: 0, ordering: HashMap::new(), path: prior, prune: true, evaluator };
        let mut result = AlphaBetaResult { action: game.search_actions_rust().first().cloned(),
            value: None, completed_depth: 0, nodes: 0, cutoffs: 0,
            elapsed_seconds: 0.0, interrupted: false };
        for depth in 1..=max_depth {
            match search.visit(&game, depth, -2.0, 2.0) {
                Ok((value, action)) => {
                    result.value = Some(value); result.action = action;
                    result.completed_depth = depth;
                    if value.abs() == 1.0 || result.action.is_none() { break; }
                },
                Err(()) => { result.interrupted = true; break; }
            }
        }
        result.nodes = search.nodes; result.cutoffs = search.cutoffs;
        result.elapsed_seconds = start.elapsed().as_secs_f64();
        result
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlphaBetaResult>()?;
    m.add_function(wrap_pyfunction!(alphabeta_search, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(g: &Game, depth: u32, prune: bool, path: Vec<String>) -> (f64, Option<String>) {
        Search { deadline: Instant::now() + Duration::from_secs(30), node_limit: 1_000_000,
            nodes: 0, cutoffs: 0, ordering: HashMap::new(), path, prune, evaluator: None }
            .visit(g, depth, -2.0, 2.0).unwrap()
    }
    #[test]
    fn two_white_moves_keep_the_same_objective() {
        let g = Game::from_fen("8/8/8/8/4k3/8/4K3/8 w - - 0 1").unwrap();
        assert_eq!(run(&g, 1, true, vec![]).0, 1.0);
        assert_eq!(run(&g, 1, false, vec![]).0, 1.0);
    }
    #[test]
    fn black_minimizes_white_value() {
        let g = Game::from_fen("k7/8/8/8/8/8/8/r3K3 b - - 0 1").unwrap();
        assert_eq!(run(&g, 1, true, vec![]), (-1.0, Some("a1e1".into())));
    }
    #[test]
    fn cap_is_draw_not_training_relabel() {
        let g = Game::from_state("k7/8/8/8/8/8/8/r3K3 b - - 0 1", false, 150).unwrap();
        assert_eq!(run(&g, 1, true, vec![]).0, 0.0);
    }
    #[test]
    fn frontier_capture_is_proven_not_a_point_ninety_five_estimate() {
        let black = Game::from_fen("k7/8/8/8/8/8/8/r3K3 b - - 0 1").unwrap();
        assert_eq!(run(&black, 0, true, vec![]).0, -1.0);
        let white = Game::from_fen("8/8/8/8/4k3/8/4K3/8 w - - 0 1").unwrap();
        assert_eq!(run(&white, 0, true, vec![]).0, 1.0);
    }
    #[test]
    fn third_settled_occurrence_is_draw() {
        let g = Game::from_fen("k7/8/8/8/8/8/8/r3K3 b - - 0 1").unwrap();
        let k = repetition_key(&g);
        assert_eq!(run(&g, 1, true, vec![k.clone(), k]).0, 0.0);
    }
    #[test]
    fn shallow_pruning_matches_exhaustive_in_all_phases() {
        for (fen, pending) in [
            ("k7/8/8/8/8/8/3P4/4K3 w - - 0 1", false),
            ("k7/8/8/8/8/8/3P4/4K3 w - - 0 1", true),
            ("k7/8/8/8/8/8/3P4/4K3 b - - 0 1", false)] {
            let g = Game::from_state(fen, pending, 0).unwrap();
            for depth in 1..=2 {
                assert_eq!(run(&g, depth, true, vec![]).0, run(&g, depth, false, vec![]).0);
            }
        }
    }
}
