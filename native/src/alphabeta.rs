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
    pvs: bool,
    pvs_scouts: u64,
    pvs_researches: u64,
    root_order: Vec<Move>,
    phase_nodes: [u64;3],
    phase_generated: [u64;3],
    profile: crate::search_profile::Profile,
    incremental_eval: bool,
    trace_leaves: bool,
    action_path: Vec<Move>,
}

impl Search {
    fn static_value(&mut self, g: &Game) -> f64 {
        let timer=self.profile.start();
        if self.optimizations {
            if let Some(value)=self.eval_cache.get(g) {self.profile.end(0,timer);return value;}
        }
        self.profile.end(0,timer);
        let timer=self.profile.start();
        let proven=if self.optimizations {crate::tactical::capture_in_turn(g)}
            else {crate::eval::pre_nn_clamp(g.board_ref(),g.is_white_turn,g.white_half_pending).is_some()};
        self.profile.end(1,timer);
        let timer=self.profile.start();
        let value=if proven {if g.is_white_turn {1.0} else {-1.0}}
            else {match &self.evaluator {
                Some(net)=>{
                    let raw=if self.incremental_eval {net.evaluate_incremental(g,&mut self.eval_scratch)}
                        else {net.evaluate_with_scratch(g, &mut self.eval_scratch)};
                    self.recorder.observe_with_path(g,raw,&self.action_path);
                    raw
                },
                None=>crate::eval::evaluate(g.board_ref(),g.is_white_turn,g.white_half_pending),
            }.clamp(-0.99,0.99)};
        self.profile.end(2,timer);
        let timer=self.profile.start();
        if self.optimizations {self.eval_cache.put(g,value);}
        self.profile.end(0,timer);
        value
    }

    fn descend(&mut self,g: &Game,depth: u32,alpha: f64,beta: f64,ply: usize,
               extensions: u32,mv: &Move)->Result<(f64,Option<Move>),()> {
        if self.trace_leaves {self.action_path.push(*mv);}
        let result=self.visit(g,depth,alpha,beta,ply,extensions);
        if self.trace_leaves {self.action_path.pop();}
        result
    }

    fn visit(&mut self, g: &Game, mut depth: u32, mut alpha: f64, mut beta: f64, ply: usize,
             mut extensions: u32)
        -> Result<(f64, Option<Move>), ()>
    {
        if self.nodes >= self.node_limit || Instant::now() >= self.deadline {
            return Err(());
        }
        self.nodes += 1;
        let phase=if !g.is_white_turn {2} else {g.white_half_pending as usize};
        self.phase_nodes[phase]+=1;
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
        let timer=self.profile.start();
        let key=if self.optimizations && self.prune { Some(BoundKey::new(g,depth,extensions,&self.path)) } else { None };
        if let Some(ref k)=key {
            if let Some(answer)=self.score_cache.probe(k,alpha,beta) { self.profile.end(6,timer);return Ok(answer); }
        }
        self.profile.end(6,timer);
        // Save the probed window BEFORE child search mutates it.
        let (alpha_start,beta_start)=(*alpha,*beta);
        let timer=self.profile.start();
        let mut actions = moves(g);
        self.profile.end(3,timer);
        let timer=self.profile.start();
        if self.optimizations { self.ordering.sort(g,&mut actions,ply); }
        else { actions.sort_by_key(|m| -(g.board_ref().piece_type_at(m.to).unwrap_or(0) as i32*10
                                         + if m.promotion.is_some() {9} else {0})); }
        if ply==0 && !self.root_order.is_empty() {
            self.ordering.root_hint(g,&mut actions,&self.root_order);
        }
        self.profile.end(4,timer);
        let phase=if !g.is_white_turn {2} else {g.white_half_pending as usize};
        self.phase_generated[phase]+=actions.len() as u64;
        if actions.is_empty() { return Ok((0.0, None)); }
        let mut best = if g.is_white_turn { -2.0 } else { 2.0 };
        let mut best_move = None;
        for mv in actions {
            let timer=self.profile.start();
            let mut child = g.clone();
            child.apply_half_move(&mv, false);
            self.profile.end(5,timer);
            let next_depth = if g.turn_completing() { depth.saturating_sub(1) } else { depth };
            let value=if self.pvs && self.prune && best_move.is_some() {
                let (a,b)=crate::search_window::scout(g.is_white_turn,*alpha,*beta);
                self.pvs_scouts+=1;
                let (scout_value,_)=self.descend(&child,next_depth,a,b,ply+1,extensions,&mv)?;
                if crate::search_window::needs_research(scout_value,*alpha,*beta) {
                    self.pvs_researches+=1;
                    self.descend(&child,next_depth,*alpha,*beta,ply+1,extensions,&mv)?.0
                } else {scout_value}
            } else {self.descend(&child,next_depth,*alpha,*beta,ply+1,extensions,&mv)?.0};
            if best_move.is_none() || (g.is_white_turn && value > best)
                || (!g.is_white_turn && value < best) {
                best = value;
                best_move = Some(mv);
            }
            if g.is_white_turn { *alpha = alpha.max(best); }
            else { *beta = beta.min(best); }
            if self.prune && *alpha >= *beta {
                self.cutoffs += 1;
                let timer=self.profile.start();
                if self.optimizations { self.ordering.cutoff(g,&mv,ply,depth); }
                self.profile.end(4,timer);
                break;
            }
        }
        let timer=self.profile.start();
        if self.optimizations {
            if let Some(ref mv)=best_move { self.ordering.best(g,mv); }
        }
        self.profile.end(4,timer);
        let timer=self.profile.start();
        if let Some(k)=key { self.score_cache.store(k,best,best_move,alpha_start,beta_start); }
        self.profile.end(7,timer);
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
    pub leaf_paths: Vec<Vec<String>>,
    pub leaf_evaluations: u64,
    pub pvs_scouts: u64,
    pub pvs_researches: u64,
    pub tt_rejected: u64,
    /// White first, White second, Black, respectively.
    pub phase_nodes: Vec<u64>,
    pub phase_generated: Vec<u64>,
    pub profile_labels: Vec<String>,
    pub profile_seconds: Vec<f64>,
    pub profile_calls: Vec<u64>,
    pub incremental_updates: u64,
    pub eval_refreshes: u64,
}

#[pyfunction]
#[pyo3(signature = (fen, pending=false, turn_count=0, seconds=1.0,
                   max_depth=12, node_limit=10000000, prior_positions=None, evaluator=None,
                   optimizations=true, extension_turns=0, collect_leaves=0, leaf_seed=9175,
                   pvs=false, fresh_tt=false, tt_capacity=32768, root_order=None, profile_search=false,
                   incremental_eval=false, collect_leaf_paths=false))]
fn alphabeta_search(py: Python<'_>, fen: &str, pending: bool, turn_count: u32,
                    seconds: f64, max_depth: u32, node_limit: u64,
                    prior_positions: Option<Vec<String>>, evaluator: Option<PyRef<'_, CheapValue>>,
                    optimizations: bool, extension_turns: u32, collect_leaves: usize,
                    leaf_seed: u64, pvs: bool, fresh_tt: bool, tt_capacity: usize,
                    root_order: Option<Vec<String>>, profile_search: bool, incremental_eval: bool,
                    collect_leaf_paths: bool) -> PyResult<AlphaBetaResult> {
    if tt_capacity>1_048_576 {return Err(PyValueError::new_err("tt_capacity <= 1048576 required"));}
    if collect_leaves>4096 {return Err(PyValueError::new_err("collect_leaves <= 4096 required"));}
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 86400.0
        || max_depth == 0 || max_depth > 150 || node_limit == 0 || extension_turns>8 {
        return Err(PyValueError::new_err("positive finite limits required; depth <= 150, seconds <= 86400, extensions <= 8"));
    }
    let game = Game::from_state(fen, pending, turn_count).map_err(PyValueError::new_err)?;
    let root_order=root_order.unwrap_or_default();
    let legal=if root_order.is_empty() {vec![]} else {moves(&game)};
    let mut hints=Vec::new();
    for uci in root_order {
        let mv=legal.iter().find(|m| m.uci()==uci)
            .ok_or_else(||PyValueError::new_err("root_order contains an illegal move"))?;
        if hints.contains(mv) {return Err(PyValueError::new_err("duplicate root_order move"));}
        hints.push(*mv);
    }
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
            eval_cache: EvalCache::new(), score_cache: ScoreCache::with_capacity(tt_capacity), optimizations, prune: true,
            evaluator, eval_scratch:Scratch::default(), extension_nodes:0, max_ply:0,
            recorder:crate::leaf_recorder::Recorder::new(collect_leaves,leaf_seed),
            pvs,pvs_scouts:0,pvs_researches:0,root_order:hints,phase_nodes:[0;3],phase_generated:[0;3],
            profile:crate::search_profile::Profile::new(profile_search),incremental_eval,
            trace_leaves:collect_leaf_paths && collect_leaves>0,action_path:vec![] };
        let mut result = AlphaBetaResult { action: game.search_actions_rust().first().cloned(),
            value: None, completed_depth: 0, nodes: 0, cutoffs: 0,
            elapsed_seconds: 0.0, interrupted: false, eval_cache_hits: 0, tt_hits: 0, tt_cutoffs: 0,
            extension_nodes:0, max_ply_reached:0, leaf_samples:vec![], leaf_paths:vec![], leaf_evaluations:0,
            pvs_scouts:0,pvs_researches:0,tt_rejected:0,phase_nodes:vec![],phase_generated:vec![],
            profile_labels:vec![],profile_seconds:vec![],profile_calls:vec![],
            incremental_updates:0,eval_refreshes:0 };
        for depth in 1..=max_depth {
            if fresh_tt {search.score_cache.next_iteration();}
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
        result.pvs_scouts=search.pvs_scouts;result.pvs_researches=search.pvs_researches;
        result.tt_rejected=search.score_cache.rejected;
        result.phase_nodes=search.phase_nodes.to_vec();result.phase_generated=search.phase_generated.to_vec();
        result.leaf_evaluations=search.recorder.seen;
        if collect_leaf_paths {result.leaf_paths=search.recorder.paths();}
        result.leaf_samples=search.recorder.finish();
        result.profile_labels=crate::search_profile::LABELS.iter().map(|s|s.to_string()).collect();
        result.profile_seconds=search.profile.nanos.iter().map(|n|*n as f64/1e9).collect();
        result.profile_calls=search.profile.calls.to_vec();
        result.incremental_updates=search.eval_scratch.incremental_updates;
        result.eval_refreshes=search.eval_scratch.refreshes;
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
        run_pvs(g,depth,prune,path,extensions,false)
    }
    fn run_pvs(g: &Game, depth: u32, prune: bool, path: Vec<PositionKey>, extensions: u32,pvs: bool)
        -> (f64, Option<String>) {
        let mut counts=HashMap::new();
        for k in path { *counts.entry(k).or_insert(0)+=1; }
        Search { deadline: Instant::now() + Duration::from_secs(30), node_limit: 1_000_000,
            nodes: 0, cutoffs: 0, ordering: Ordering::new(), path:vec![], counts, prune,
            eval_cache:EvalCache::new(),score_cache:ScoreCache::new(),optimizations:prune,evaluator: None,
            eval_scratch:Scratch::default(), extension_nodes:0, max_ply:0,
            recorder:crate::leaf_recorder::Recorder::new(0,0),pvs,pvs_scouts:0,pvs_researches:0,
            root_order:vec![],phase_nodes:[0;3],phase_generated:[0;3],
            profile:crate::search_profile::Profile::new(false),incremental_eval:false,
            trace_leaves:false,action_path:vec![] }
            .visit(g, depth, -2.0, 2.0,0,extensions).map(|(v,m)| (v,m.map(|x|x.uci()))).unwrap()
    }
    #[test]
    fn pvs_matches_exhaustive_with_consecutive_max_and_forced_responses() {
        for (fen,pending) in [
            ("k7/8/8/8/8/8/3P4/4K3 w - - 0 1",false),
            ("k7/8/8/8/8/8/3P4/4K3 w - - 0 1",true),
            ("k7/8/8/8/8/8/3P4/4K3 b - - 0 1",false),
            ("k3r3/8/8/8/8/8/8/4K3 w - - 0 1",false)] {
            let g=Game::from_state(fen,pending,0).unwrap();
            for depth in 1..=3 {
                assert_eq!(run_pvs(&g,depth,true,vec![],0,true).0,run(&g,depth,false,vec![]).0);
            }
        }
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
