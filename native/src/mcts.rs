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
// Python surface — enough to run the contract tests against the native arena.
// ---------------------------------------------------------------------------

#[pyclass(name = "Tree")]
pub struct PyTree {
    arena: Arena,
}

#[pymethods]
impl PyTree {
    #[new]
    fn new(fen: &str) -> PyResult<Self> {
        let root = Game::from_fen(fen).map_err(PyValueError::new_err)?;
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
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTree>()?;
    m.add("C_PUCT", C_PUCT)?;
    m.add("FPU_REDUCTION", FPU_REDUCTION)?;
    m.add("EXPLORATION_CONSTANT", EXPLORATION_CONSTANT)?;
    Ok(())
}
