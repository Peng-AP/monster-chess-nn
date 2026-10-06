//! Opt-in search-first prototype. White max/max, Black min; depth is completed
//! turns. No selective pruning and no captures-only qsearch. Searched bounds
//! use exact depth AND path identity; static evaluation has its own cache.
//! Repetition is history-dependent. Unlike the
//! incumbent's training-label terminal, turn caps here score an actual draw.
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use crate::cheap_value::{CheapValue, Weights, Scratch};
use crate::search_order::{Ordering, PositionKey};
use crate::search_cache::EvalCache;
use crate::search_bounds::{ScoreCache, BoundKey};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::game::{Game, MAX_GAME_TURNS};
use crate::bitboard::{Move, WHITE, BLACK};
use crate::monster::{black_actions, white_single_moves, white_second_half_moves};

fn repetition_key(g: &Game) -> PositionKey { PositionKey::repetition(g) }

fn moves(g: &Game) -> Vec<Move> {
    if !g.is_white_turn { black_actions(g.board_ref(), true) }
    else if g.white_half_pending { white_second_half_moves(g.board_ref()) }
    else { white_single_moves(g.board_ref()) }
}

struct Search {
    deadline: Instant,
    node_limit: u64,
    nodes: u64,
    cutoffs: u64,
    ordering: Ordering,
    path: Vec<PositionKey>,
    counts: HashMap<PositionKey, usize>,
    eval_cache: EvalCache,
    score_cache: ScoreCache,
    optimizations: bool,
    prune: bool,
    evaluator: Option<Arc<Weights>>,
    eval_scratch: Scratch,
    extension_nodes: u64,
    max_ply: usize,
    recorder: crate::leaf_recorder::Recorder,
}

impl Search {
    fn static_value(&mut self, g: &Game) -> f64 {
        if self.optimizations {
            if let Some(value)=self.eval_cache.get(g) {return value;}
        }
        let proven=if self.optimizations {crate::tactical::capture_in_turn(g)}
            else {crate::eval::pre_nn_clamp(g.board_ref(),g.is_white_turn,g.white_half_pending).is_some()};
        let value=if proven {if g.is_white_turn {1.0} else {-1.0}}
            else {match &self.evaluator {
                Some(net)=>{
                    let raw=net.evaluate_with_scratch(g, &mut self.eval_scratch);
                    self.recorder.observe(g,raw);
                    raw
                },
                None=>crate::eval::evaluate(g.board_ref(),g.is_white_turn,g.white_half_pending),
            }.clamp(-0.99,0.99)};
        if self.optimizations {self.eval_cache.put(g,value);}
        value
    }

    fn visit(&mut self, g: &Game, mut depth: u32, mut alpha: f64, mut beta: f64, ply: usize,
             mut extensions: u32)
        -> Result<(f64, Option<Move>), ()>
    {
        if self.nodes >= self.node_limit || Instant::now() >= self.deadline {
            return Err(());
        }
        self.nodes += 1;
        self.max_ply=self.max_ply.max(ply);
        let b = g.board_ref();
        // Captures take precedence over clocks/repetition.
        if b.king_square(WHITE).is_none() { return Ok((-1.0, None)); }
        if b.king_square(BLACK).is_none() { return Ok((1.0, None)); }
        if g.turn_count >= MAX_GAME_TURNS { return Ok((0.0, None)); }
        let settled = !g.white_half_pending;
        let rep = if settled { Some(repetition_key(g)) } else { None };
        if let Some(ref k) = rep {
            if self.counts.get(k).copied().unwrap_or(0) >= 2 {
                return Ok((0.0, None));
            }
        }
        // Never evaluate a White turn halfway through. Initial plumbing uses
        // the existing heuristic, not an assumed cheap or adequate evaluator.
        if depth == 0 && settled {
            let value=self.static_value(g);
            if value.abs()==1.0 || extensions==0 || !crate::tactical::mover_under_threat(g) {
                return Ok((value,None));
            }
            // All responses, through the WHOLE turn. No stand-pat, captures-only
            // filtering, or proof inferred when the extra-turn budget runs out.
            depth=1; extensions-=1; self.extension_nodes+=1;
        }
        if let Some(k) = rep {
            self.path.push(k);
            *self.counts.entry(k).or_insert(0) += 1;
        }
        let answer = self.children(g, depth, &mut alpha, &mut beta, ply, extensions);
        if let Some(k) = rep {
            self.path.pop();
            let count=self.counts.get_mut(&k).expect("pushed repetition count");
            *count-=1;
            if *count==0 { self.counts.remove(&k); }
        }
        answer
    }

    fn children(&mut self, g: &Game, depth: u32, alpha: &mut f64, beta: &mut f64, ply: usize,
                extensions: u32)
        -> Result<(f64, Option<Move>), ()>
    {
        let key=if self.optimizations && self.prune { Some(BoundKey::new(g,depth,extensions,&self.path)) } else { None };
        if let Some(ref k)=key {
            if let Some(answer)=self.score_cache.probe(k,alpha,beta) { return Ok(answer); }
        }
        // Save the probed window BEFORE child search mutates it.
        let (alpha_start,beta_start)=(*alpha,*beta);
        let mut actions = moves(g);
        if self.optimizations { self.ordering.sort(g,&mut actions,ply); }
        else { actions.sort_by_key(|m| -(g.board_ref().piece_type_at(m.to).unwrap_or(0) as i32*10
                                         + if m.promotion.is_some() {9} else {0})); }
        if actions.is_empty() { return Ok((0.0, None)); }
        let mut best = if g.is_white_turn { -2.0 } else { 2.0 };
        let mut best_move = None;
        for mv in actions {
            let mut child = g.clone();
            child.apply_half_move(&mv, false);
            let next_depth = if g.turn_completing() { depth.saturating_sub(1) } else { depth };
            let (value, _) = self.visit(&child, next_depth, *alpha, *beta, ply+1,extensions)?;
            if best_move.is_none() || (g.is_white_turn && value > best)
                || (!g.is_white_turn && value < best) {
                best = value;
                best_move = Some(mv);
            }
            if g.is_white_turn { *alpha = alpha.max(best); }
            else { *beta = beta.min(best); }
            if self.prune && *alpha >= *beta {
                self.cutoffs += 1;
                if self.optimizations { self.ordering.cutoff(g,&mv,ply,depth); }
                break;
            }
        }
        if self.optimizations {
            if let Some(ref mv)=best_move { self.ordering.best(g,mv); }
        }
        if let Some(k)=key { self.score_cache.store(k,best,best_move,alpha_start,beta_start); }
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
    pub eval_cache_hits: u64,
    pub tt_hits: u64,
    pub tt_cutoffs: u64,
    pub extension_nodes: u64,
    pub max_ply_reached: usize,
    pub leaf_samples: Vec<crate::leaf_recorder::Sample>,
    pub leaf_evaluations: u64,
}

#[pyfunction]
#[pyo3(signature = (fen, pending=false, turn_count=0, seconds=1.0,
                   max_depth=12, node_limit=10000000, prior_positions=None, evaluator=None,
                   optimizations=true, extension_turns=0, collect_leaves=0, leaf_seed=9175))]
fn alphabeta_search(py: Python<'_>, fen: &str, pending: bool, turn_count: u32,
                    seconds: f64, max_depth: u32, node_limit: u64,
                    prior_positions: Option<Vec<String>>, evaluator: Option<PyRef<'_, CheapValue>>,
                    optimizations: bool, extension_turns: u32, collect_leaves: usize,
                    leaf_seed: u64) -> PyResult<AlphaBetaResult> {
    if collect_leaves>4096 {return Err(PyValueError::new_err("collect_leaves <= 4096 required"));}
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 86400.0
        || max_depth == 0 || max_depth > 150 || node_limit == 0 || extension_turns>8 {
        return Err(PyValueError::new_err("positive finite limits required; depth <= 150, seconds <= 86400, extensions <= 8"));
    }
    let game = Game::from_state(fen, pending, turn_count).map_err(PyValueError::new_err)?;
    let prior = prior_positions.unwrap_or_default();
    let evaluator = evaluator.map(|e| Arc::clone(&e.weights));
    // Historical settled keys EXCLUDING current position. Strict format avoids
    // silently accepting six-field FENs that never compare equal.
    if prior.iter().any(|x| x.split_whitespace().count() != 4) {
        return Err(PyValueError::new_err("prior_positions needs settled four-field FEN keys, excluding root"));
    }
    let mut counts=HashMap::new();
    for fen in prior {
        let historical=Game::from_fen(&format!("{fen} 0 1")).map_err(PyValueError::new_err)?;
        *counts.entry(repetition_key(&historical)).or_insert(0)+=1;
    }
    Ok(py.detach(move || {
        let start = Instant::now();
        let mut search = Search { deadline: start + Duration::from_secs_f64(seconds),
            node_limit, nodes: 0, cutoffs: 0, ordering: Ordering::new(), path: vec![], counts,
            eval_cache: EvalCache::new(), score_cache: ScoreCache::new(), optimizations, prune: true,
            evaluator, eval_scratch:Scratch::default(), extension_nodes:0, max_ply:0,
            recorder:crate::leaf_recorder::Recorder::new(collect_leaves,leaf_seed) };
        let mut result = AlphaBetaResult { action: game.search_actions_rust().first().cloned(),
            value: None, completed_depth: 0, nodes: 0, cutoffs: 0,
            elapsed_seconds: 0.0, interrupted: false, eval_cache_hits: 0, tt_hits: 0, tt_cutoffs: 0,
            extension_nodes:0, max_ply_reached:0, leaf_samples:vec![], leaf_evaluations:0 };
        for depth in 1..=max_depth {
            match search.visit(&game, depth, -2.0, 2.0, 0,extension_turns) {
                Ok((value, action)) => {
                    result.value = Some(value); result.action = action.map(|m| m.uci());
                    result.completed_depth = depth;
                    if value.abs() == 1.0 || result.action.is_none() { break; }
                },
                Err(()) => { result.interrupted = true; break; }
            }
        }
        result.nodes = search.nodes; result.cutoffs = search.cutoffs;
        result.elapsed_seconds = start.elapsed().as_secs_f64();
        result.eval_cache_hits=search.eval_cache.hits;
        result.tt_hits=search.score_cache.hits;
        result.tt_cutoffs=search.score_cache.cutoffs;
        result.extension_nodes=search.extension_nodes;
        result.max_ply_reached=search.max_ply;
        result.leaf_evaluations=search.recorder.seen;
        result.leaf_samples=search.recorder.finish();
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
    fn run(g: &Game, depth: u32, prune: bool, path: Vec<PositionKey>) -> (f64, Option<String>) {
        run_extended(g,depth,prune,path,0)
    }
    fn run_extended(g: &Game, depth: u32, prune: bool, path: Vec<PositionKey>, extensions: u32)
        -> (f64, Option<String>) {
        let mut counts=HashMap::new();
        for k in path { *counts.entry(k).or_insert(0)+=1; }
        Search { deadline: Instant::now() + Duration::from_secs(30), node_limit: 1_000_000,
            nodes: 0, cutoffs: 0, ordering: Ordering::new(), path:vec![], counts, prune,
            eval_cache:EvalCache::new(),score_cache:ScoreCache::new(),optimizations:prune,evaluator: None,
            eval_scratch:Scratch::default(), extension_nodes:0, max_ply:0,
            recorder:crate::leaf_recorder::Recorder::new(0,0) }
            .visit(g, depth, -2.0, 2.0,0,extensions).map(|(v,m)| (v,m.map(|x|x.uci()))).unwrap()
    }
    #[test]
    fn threat_extension_is_a_complete_turn_not_a_capture_subset() {
        for fen in ["k3r3/8/8/8/8/8/8/4K3 w - - 0 1",
                    "8/8/8/8/4k3/8/4K3/8 b - - 0 1"] {
            let g=Game::from_fen(fen).unwrap();
            assert!(crate::tactical::mover_under_threat(&g));
            assert!(!crate::tactical::capture_in_turn(&g));
            assert_eq!(run_extended(&g,0,true,vec![],1).0,run(&g,1,false,vec![]).0);
            assert_eq!(run_extended(&g,2,true,vec![],2).0,run_extended(&g,2,false,vec![],2).0);
        }
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
