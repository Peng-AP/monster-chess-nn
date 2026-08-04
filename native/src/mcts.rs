//! MCTS node arithmetic, ported from `src/mcts.py` (DIRECTIVE E3).
//!
//! This module carries the logic `CONTEXT.md` §1.2 calls the subtlest in the
//! engine, and the directive's risk table records that M2 misread it twice. The
//! two rules, stated once:
//!
//! 1. **Backprop stores value in the perspective of the side that moved *into*
//!    the node** — its parent's side to move — so the parent reads Q
//!    consistently while selecting. The root, having no parent, uses its own.
//!
//! 2. **FPU expresses parent-Q in the *selector's* perspective.** A node's Q is
//!    stored in its grandparent's frame, so it matches the selector only when
//!    grandparent and parent share a side to move. Under strict alternation the
//!    flip is every non-root ply — but **White's two half-moves do not change
//!    side**, so the grandparent test is required rather than a ply-parity
//!    shortcut. This is the specific thing a port written from chess intuition
//!    gets wrong.
//!
//! Nodes live in an arena indexed by `usize`, not a pointer tree: parent links
//! plus Rust ownership do not mix, and an arena also makes the whole tree one
//! contiguous allocation, which is most of the point of moving off Python.
//!
//! `tests/test_mcts_perspective_contract.py` states the expected numbers by
//! hand and runs against **both** engines.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::game::Game;

/// Mirrors `config.C_PUCT`, asserted against Python by the test suite.
pub const C_PUCT: f64 = 1.5;
/// Mirrors `config.FPU_REDUCTION`.
pub const FPU_REDUCTION: f64 = 0.30;
/// Mirrors `config.EXPLORATION_CONSTANT` (UCB1, heuristic mode).
pub const EXPLORATION_CONSTANT: f64 = 1.41;

pub struct Node {
    pub state: Game,
    pub parent: Option<usize>,
    pub action: Option<String>,
    pub prior: f64,
    pub visit_count: u32,
    pub total_value: f64,
    pub children: Vec<usize>,
    pub is_expanded: bool,
}

impl Node {
    fn new(state: Game, parent: Option<usize>, action: Option<String>, prior: f64) -> Self {
        Node {
            state,
            parent,
            action,
            prior,
            visit_count: 0,
            total_value: 0.0,
            children: Vec::new(),
            is_expanded: false,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f64
        }
    }
}

pub struct Arena {
    pub nodes: Vec<Node>,
}

impl Arena {
    pub fn new(root_state: Game) -> Self {
        Arena {
            nodes: vec![Node::new(root_state, None, None, 1.0)],
        }
    }

    pub fn add_child(&mut self, parent: usize, state: Game, action: String, prior: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::new(state, Some(parent), Some(action), prior));
        self.nodes[parent].children.push(idx);
        idx
    }

    /// PUCT: Q + c * P * sqrt(N_parent) / (1 + N), with FPU for unvisited children.
    pub fn puct_score(&self, idx: usize, c_puct: f64, fpu_reduction: f64) -> f64 {
        let node = &self.nodes[idx];
        let parent_visits = match node.parent {
            Some(p) => self.nodes[p].visit_count.max(1) as f64,
            None => 1.0,
        };
        if node.visit_count == 0 {
            let fpu_q = match node.parent {
                None => 0.0,
                Some(p) => {
                    let parent = &self.nodes[p];
                    let mut parent_q = parent.q_value();
                    // The flip: parent's Q lives in the grandparent's frame.
                    if let Some(gp) = parent.parent {
                        if self.nodes[gp].state.is_white_turn != parent.state.is_white_turn {
                            parent_q = -parent_q;
                        }
                    }
                    (parent_q - fpu_reduction).clamp(-1.0, 1.0)
                }
            };
            // The unvisited branch omits the (1 + N) divisor; N is 0, so it is 1.
            fpu_q + c_puct * node.prior * parent_visits.sqrt()
        } else {
            node.q_value()
                + c_puct * node.prior * parent_visits.sqrt() / (1.0 + node.visit_count as f64)
        }
    }

    /// UCB1, heuristic mode. Unvisited children sort first, as in Python's `inf`.
    pub fn ucb_score(&self, idx: usize, c: f64) -> f64 {
        let node = &self.nodes[idx];
        if node.visit_count == 0 {
            return f64::INFINITY;
        }
        let parent_visits = match node.parent {
            Some(p) => self.nodes[p].visit_count as f64,
            None => 1.0,
        };
        node.q_value() + c * (parent_visits.ln() / node.visit_count as f64).sqrt()
    }

    /// `max` over children, matching Python's `max()` tie-break: first wins.
    pub fn best_child_puct(&self, idx: usize, c_puct: f64, fpu_reduction: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for &child in &self.nodes[idx].children {
            let score = self.puct_score(child, c_puct, fpu_reduction);
            match best {
                Some((_, b)) if !(score > b) => {}
                _ => best = Some((child, score)),
            }
        }
        best.map(|(i, _)| i)
    }

    /// Propagate a White-perspective value to the root.
    pub fn backpropagate(&mut self, from: usize, value: f64) {
        let mut current = Some(from);
        while let Some(idx) = current {
            self.nodes[idx].visit_count += 1;
            let signed = match self.nodes[idx].parent {
                Some(p) => {
                    if self.nodes[p].state.is_white_turn {
                        value
                    } else {
                        -value
                    }
                }
                None => {
                    if self.nodes[idx].state.is_white_turn {
                        value
                    } else {
                        -value
                    }
                }
            };
            self.nodes[idx].total_value += signed;
            current = self.nodes[idx].parent;
        }
    }
}

// ---------------------------------------------------------------------------
// Sequential UCB1 search (heuristic mode)
// ---------------------------------------------------------------------------
//
// Ported from `_run_sequential` / `_select_ucb` / `_evaluate_and_expand_ucb`.
// Leaves are scored by the native heuristic, so this path never touches Python.
//
// D4 puts search under *statistical* parity, not bit-parity: Python's MT19937
// stream is not replicated, and `expand_one` shuffles untried actions. So the
// RNG here is our own (xorshift, seeded) and equivalence is established by the
// E3 gates rather than by identical playouts.

use crate::eval::evaluate as heuristic;

/// xorshift64*, so a seeded run is reproducible without pulling in a crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn shuffle<T>(&mut self, items: &mut Vec<T>) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

impl Arena {
    fn is_fully_expanded(&self, idx: usize) -> bool {
        self.nodes[idx].is_expanded
    }

    /// Descend by UCB1 until an unexpanded node or a terminal is reached.
    fn select_ucb(&self, root: usize, c: f64) -> usize {
        let mut node = root;
        while !self.nodes[node].state.is_terminal_rust() {
            if !self.is_fully_expanded(node) {
                return node;
            }
            if self.nodes[node].children.is_empty() {
                return node;
            }
            let mut best: Option<(usize, f64)> = None;
            for &child in &self.nodes[node].children {
                let score = self.ucb_score(child, c);
                match best {
                    Some((_, b)) if !(score > b) => {}
                    _ => best = Some((child, score)),
                }
            }
            node = match best {
                Some((i, _)) => i,
                None => return node,
            };
        }
        node
    }

    /// Expand one untried action, mirroring `expand_one`.
    fn expand_one(&mut self, idx: usize, untried: &mut Vec<Vec<String>>, rng: &mut Rng) -> Option<usize> {
        if untried[idx].is_empty() && !self.nodes[idx].is_expanded {
            let mut actions = self.nodes[idx].state.search_actions_rust();
            rng.shuffle(&mut actions);
            if actions.is_empty() {
                self.nodes[idx].is_expanded = true;
                return None;
            }
            untried[idx] = actions;
        }
        let action = match untried[idx].pop() {
            Some(a) => a,
            None => {
                self.nodes[idx].is_expanded = true;
                return None;
            }
        };
        let mut child_state = self.nodes[idx].state.clone();
        if child_state.apply_half(&action).is_err() {
            return None;
        }
        let child = self.add_child(idx, child_state, action, 1.0);
        untried.push(Vec::new());
        if untried[idx].is_empty() {
            self.nodes[idx].is_expanded = true;
        }
        Some(child)
    }

    fn leaf_value(&self, idx: usize) -> f64 {
        let node = &self.nodes[idx];
        if let Some(result) = node.state.result_rust() {
            return result;
        }
        heuristic(
            node.state.board_ref(),
            node.state.is_white_turn,
            node.state.white_half_pending,
        )
    }

    fn should_stop_early(&self, root: usize, sims_done: usize, total: usize) -> bool {
        if (sims_done as f64) < total as f64 * 0.3 {
            return false;
        }
        let children = &self.nodes[root].children;
        if children.is_empty() {
            return false;
        }
        if self.nodes[root].q_value().abs() > 0.95 {
            return true;
        }
        let mut visits: Vec<u32> = children.iter().map(|&c| self.nodes[c].visit_count).collect();
        visits.sort_unstable_by(|a, b| b.cmp(a));
        if visits.len() >= 2 {
            let remaining = (total - sims_done) as i64;
            if (visits[0] as i64 - visits[1] as i64) > remaining {
                return true;
            }
        }
        false
    }

    /// Run `simulations` sequential UCB1 iterations from the root.
    pub fn run_sequential(&mut self, simulations: usize, c: f64, allow_early_stop: bool, rng: &mut Rng) {
        let mut untried: Vec<Vec<String>> = vec![Vec::new(); self.nodes.len()];
        for i in 0..simulations {
            let node = self.select_ucb(0, c);
            let (leaf, value) = if self.nodes[node].state.is_terminal_rust() {
                (node, self.leaf_value(node))
            } else {
                while untried.len() < self.nodes.len() {
                    untried.push(Vec::new());
                }
                match self.expand_one(node, &mut untried, rng) {
                    None => (node, self.leaf_value(node)),
                    Some(child) => (child, self.leaf_value(child)),
                }
            };
            self.backpropagate(leaf, value);
            if allow_early_stop && i % 32 == 31 && self.should_stop_early(0, i + 1, simulations) {
                break;
            }
        }
    }

    /// Visit counts of the root's children, with their actions.
    pub fn root_visits(&self) -> Vec<(String, u32)> {
        self.nodes[0]
            .children
            .iter()
            .map(|&c| {
                (
                    self.nodes[c].action.clone().unwrap_or_default(),
                    self.nodes[c].visit_count,
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Python surface — enough to run the contract tests against the native arena.
// ---------------------------------------------------------------------------

#[pyclass(name = "Tree")]
pub struct PyTree {
    arena: Arena,
}

#[pymethods]
impl PyTree {
    #[new]
    #[pyo3(signature = (fen, white_half_pending=false, turn_count=0))]
    fn new(fen: &str, white_half_pending: bool, turn_count: u32) -> PyResult<Self> {
        // pending and turn_count are part of the state, not decoration: the
        // first carries a different action set, the second decides the cap.
        let root = Game::from_state(fen, white_half_pending, turn_count)
            .map_err(PyValueError::new_err)?;
        Ok(PyTree { arena: Arena::new(root) })
    }

    /// Add a child by applying one half-move action to the parent's state.
    #[pyo3(signature = (parent, action, prior=1.0))]
    fn add_child(&mut self, parent: usize, action: &str, prior: f64) -> PyResult<usize> {
        let mut state = self.arena.nodes[parent].state.clone();
        state
            .apply_half(action)
            .map_err(PyValueError::new_err)?;
        Ok(self.arena.add_child(parent, state, action.to_string(), prior))
    }

    fn set_stats(&mut self, idx: usize, visit_count: u32, total_value: f64) {
        self.arena.nodes[idx].visit_count = visit_count;
        self.arena.nodes[idx].total_value = total_value;
    }

    fn set_prior(&mut self, idx: usize, prior: f64) {
        self.arena.nodes[idx].prior = prior;
    }

    fn visit_count(&self, idx: usize) -> u32 {
        self.arena.nodes[idx].visit_count
    }

    fn total_value(&self, idx: usize) -> f64 {
        self.arena.nodes[idx].total_value
    }

    fn q_value(&self, idx: usize) -> f64 {
        self.arena.nodes[idx].q_value()
    }

    fn is_white_turn(&self, idx: usize) -> bool {
        self.arena.nodes[idx].state.is_white_turn
    }

    fn white_half_pending(&self, idx: usize) -> bool {
        self.arena.nodes[idx].state.white_half_pending
    }

    fn fen(&self, idx: usize) -> String {
        self.arena.nodes[idx].state.fen_string()
    }

    #[pyo3(signature = (idx, c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION))]
    fn puct_score(&self, idx: usize, c_puct: f64, fpu_reduction: f64) -> f64 {
        self.arena.puct_score(idx, c_puct, fpu_reduction)
    }

    #[pyo3(signature = (idx, c=EXPLORATION_CONSTANT))]
    fn ucb_score(&self, idx: usize, c: f64) -> f64 {
        self.arena.ucb_score(idx, c)
    }

    fn backpropagate(&mut self, from: usize, value: f64) {
        self.arena.backpropagate(from, value);
    }

    /// Sequential UCB1 with heuristic leaves, entirely inside the crate.
    #[pyo3(signature = (simulations, c=EXPLORATION_CONSTANT, allow_early_stop=true, seed=20260803))]
    fn run_sequential(
        &mut self,
        simulations: usize,
        c: f64,
        allow_early_stop: bool,
        seed: u64,
    ) -> PyResult<()> {
        let mut rng = Rng::new(seed);
        self.arena.run_sequential(simulations, c, allow_early_stop, &mut rng);
        Ok(())
    }

    fn root_visits(&self) -> Vec<(String, u32)> {
        self.arena.root_visits()
    }

    fn node_count(&self) -> usize {
        self.arena.nodes.len()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTree>()?;
    m.add("C_PUCT", C_PUCT)?;
    m.add("FPU_REDUCTION", FPU_REDUCTION)?;
    m.add("EXPLORATION_CONSTANT", EXPLORATION_CONSTANT)?;
    Ok(())
}
