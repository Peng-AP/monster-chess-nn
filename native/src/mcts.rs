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
use pyo3::types::PyBytes;

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
    /// Game-theoretic proof in **White's** perspective: `Some(1.0)` White wins
    /// with best play, `Some(-1.0)` Black does, `None` unknown.
    ///
    /// Only a king capture proves anything. The move-limit relabel (+-0.5 by
    /// heuristic sign) is an opinion about an unfinished game, so it must never
    /// become a proof -- treating it as one would let the search "prove" wins
    /// that were merely positions it liked when the clock ran out.
    pub proof: Option<f64>,
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
            proof: None,
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
    pub fn next_u64(&mut self) -> u64 {
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
    ///
    /// The solver is not wired into this path: UCB1 mode exists for heuristic
    /// play and diagnostics, and mixing a behaviour change into a path used as
    /// a reference would blur what the flag is being measured against.
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

    /// Depth in plies of the deepest node in the tree.
    pub fn max_depth(&self) -> usize {
        let mut depth = vec![0usize; self.nodes.len()];
        let mut best = 0;
        for idx in 1..self.nodes.len() {
            if let Some(p) = self.nodes[idx].parent {
                depth[idx] = depth[p] + 1;
                best = best.max(depth[idx]);
            }
        }
        best
    }

    /// Depth of the principal variation — follow the most-visited child.
    ///
    /// This is the honest "how deep does it actually look" number: a tree can
    /// be deep down a line it visited twice while the move it will play was
    /// resolved shallowly. The PV depth is the line the search is committing to.
    pub fn pv_depth(&self) -> usize {
        let mut node = 0usize;
        let mut depth = 0usize;
        loop {
            let children = &self.nodes[node].children;
            if children.is_empty() {
                return depth;
            }
            let mut best = children[0];
            for &c in children {
                if self.nodes[c].visit_count > self.nodes[best].visit_count {
                    best = c;
                }
            }
            if self.nodes[best].visit_count == 0 {
                return depth;
            }
            node = best;
            depth += 1;
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
// Batched PUCT with an NN bridge (D3 stage 1)
// ---------------------------------------------------------------------------
//
// The search stays native and calls Python once per *batch*: it fills an
// (N, C, 8, 8) f32 buffer, hands it over as raw bytes, and gets values and
// policy logits back the same way. Bytes rather than Python lists because a
// batch of 16 is ~17k floats, and list marshalling would cost more than the
// whole search it is meant to serve.
//
// Ported from `_run_batched_puct`. Two details that look like implementation
// noise and are not:
//   * the root is expanded and backpropagated **synchronously first**, so the
//     batch loop always descends into real children, and it counts as one
//     simulation;
//   * a `pending` set stops the batch early when re-selecting an already
//     selected node — virtual loss cannot diversify a frontier that has not
//     been expanded yet, so padding the batch would be duplicate work counted
//     as progress. Simulations are counted by completed backpropagations.

pub const VIRTUAL_LOSS: i64 = 3;

/// Softmax over a subset of logit indices, matching `_softmax_masked`.
/// Degenerate (all-zero) totals fall back to uniform rather than producing NaN.
fn softmax_masked(logits: &[f32], indices: &[usize]) -> Vec<f64> {
    if indices.is_empty() {
        return Vec::new();
    }
    let vals: Vec<f64> = indices
        .iter()
        .map(|&i| *logits.get(i).unwrap_or(&0.0) as f64)
        .collect();
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = vals.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = exp.iter().sum();
    if total == 0.0 {
        let uniform = 1.0 / indices.len() as f64;
        return vec![uniform; indices.len()];
    }
    exp.iter().map(|e| e / total).collect()
}

impl Arena {
    fn apply_virtual_loss(&mut self, node: usize) {
        let mut current = Some(node);
        while let Some(idx) = current {
            self.nodes[idx].visit_count =
                (self.nodes[idx].visit_count as i64 + VIRTUAL_LOSS) as u32;
            self.nodes[idx].total_value -= VIRTUAL_LOSS as f64;
            current = self.nodes[idx].parent;
        }
    }

    fn revert_virtual_loss(&mut self, node: usize) {
        let mut current = Some(node);
        while let Some(idx) = current {
            self.nodes[idx].visit_count =
                (self.nodes[idx].visit_count as i64 - VIRTUAL_LOSS) as u32;
            self.nodes[idx].total_value += VIRTUAL_LOSS as f64;
            current = self.nodes[idx].parent;
        }
    }

    /// Descend by PUCT to a terminal or unexpanded node.
    ///
    /// With the solver on, a proven-lost child is never descended into (its
    /// value is already known, so visits there buy nothing) and a proven node
    /// is a dead end rather than something to keep sampling.
    fn select_puct(&self, root: usize, c_puct: f64, fpu: f64, solver: bool) -> usize {
        let mut node = root;
        while !self.nodes[node].state.is_terminal_rust() {
            if solver && self.nodes[node].proof.is_some() {
                return node;
            }
            if !self.nodes[node].is_expanded {
                return node;
            }
            if self.nodes[node].children.is_empty() {
                return node;
            }
            let next = if solver {
                self.best_child_puct_unproven(node, c_puct, fpu)
            } else {
                self.best_child_puct(node, c_puct, fpu)
            };
            node = match next {
                Some(child) => child,
                None => return node,
            };
        }
        node
    }

    /// `best_child_puct`, skipping children already proven lost for the mover.
    fn best_child_puct_unproven(&self, idx: usize, c_puct: f64, fpu: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for &child in &self.nodes[idx].children {
            if self.is_proven_loss_for_mover(idx, child) {
                continue;
            }
            let score = self.puct_score(child, c_puct, fpu);
            match best {
                Some((_, b)) if !(score > b) => {}
                _ => best = Some((child, score)),
            }
        }
        // Every child refuted: fall back so selection still terminates.
        best.map(|(i, _)| i)
            .or_else(|| self.best_child_puct(idx, c_puct, fpu))
    }

    /// Expand every legal half-move with priors from the policy head.
    fn expand_with_policy(&mut self, node: usize, logits: Option<&[f32]>) {
        let actions = self.nodes[node].state.search_actions_rust();
        if actions.is_empty() {
            self.nodes[node].is_expanded = true;
            return;
        }
        let priors: Vec<f64> = match logits {
            None => vec![1.0 / actions.len() as f64; actions.len()],
            Some(l) => {
                let promotion_aware = l.len() >= crate::encoding::PROMOTION_AWARE_POLICY_SIZE;
                let indices: Vec<usize> = actions.iter().map(|a| {
                    crate::encoding::policy_index(a, promotion_aware).unwrap_or(0)
                }).collect();
                softmax_masked(l, &indices)
            }
        };
        for (action, prior) in actions.into_iter().zip(priors) {
            let mut child_state = self.nodes[node].state.clone();
            if child_state.apply_half(&action).is_err() {
                continue;
            }
            self.add_child(node, child_state, action, prior);
        }
        self.nodes[node].is_expanded = true;
    }

    /// (C, 8, 8) encoding of a node's state, appended to `buf`.
    fn encode_into(&self, node: usize, channels: usize, buf: &mut Vec<f32>) {
        let state = &self.nodes[node].state;
        let hwc = crate::encoding::encode(
            state.board_ref(),
            state.is_white_turn,
            state.white_half_pending,
            channels,
        )
        .unwrap_or_else(|_| vec![0.0; 8 * 8 * channels]);
        // (8, 8, C) -> (C, 8, 8): the bridge wants channels-first for torch.
        for c in 0..channels {
            for rank in 0..8 {
                for file in 0..8 {
                    buf.push(hwc[rank * 8 * channels + file * channels + c]);
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Certainty propagation (MCTS-Solver)
// ---------------------------------------------------------------------------
//
// Plain MCTS dilutes a forced win: the winning line is one path among
// thousands, its +1 is averaged into a mean, and the visit distribution never
// concentrates hard enough to play it. Measured on this engine: 29% of
// dominant-unfinished games held a forced king capture within 3 Black moves
// that 1600-sim search walked past, and endgame PV depth stays at 4 plies even
// at 51,200 sims.
//
// MCTS-Solver (Winands et al.) fixes the dilution rather than the depth: a
// proven result backs up as an exact, unaveraged fact.
//
//   * a node whose side to move has ANY child proven winning for it is proven
//     winning;
//   * a node ALL of whose children are proven losing for it is proven losing;
//   * proofs are expressed in White's perspective throughout, so White is the
//     maximiser and Black the minimiser -- and because White's two half-moves
//     do not change the side to move, that framing stays correct across the
//     half-pair where a ply-parity rule would not.
//
// Selection then refuses children proven lost and takes a proven win at once.

impl Arena {
    /// Only a king capture proves a result. See the note on `Node::proof`.
    fn terminal_proof(&self, idx: usize) -> Option<f64> {
        let state = &self.nodes[idx].state;
        if state.board_ref().king_square(crate::bitboard::WHITE).is_none() {
            return Some(-1.0);
        }
        if state.board_ref().king_square(crate::bitboard::BLACK).is_none() {
            return Some(1.0);
        }
        None
    }

    /// Recompute one node's proof from its children, returning true if it changed.
    fn update_proof(&mut self, idx: usize) -> bool {
        if self.nodes[idx].proof.is_some() {
            return false;
        }
        let children = self.nodes[idx].children.clone();
        if children.is_empty() {
            return false;
        }
        // White to move maximises in White perspective; Black minimises.
        let win_for_mover = if self.nodes[idx].state.is_white_turn { 1.0 } else { -1.0 };
        let mut all_lost = true;
        for &child in &children {
            match self.nodes[child].proof {
                Some(v) if v == win_for_mover => {
                    self.nodes[idx].proof = Some(win_for_mover);
                    return true;
                }
                Some(_) => {}
                None => all_lost = false,
            }
        }
        // "All children lose" only counts once every child is expanded and
        // proven; an unexpanded node is not a refutation.
        if all_lost && self.nodes[idx].is_expanded {
            self.nodes[idx].proof = Some(-win_for_mover);
            return true;
        }
        false
    }

    /// Propagate a newly proven leaf towards the root, stopping when nothing changes.
    fn propagate_proof(&mut self, from: usize) {
        let mut current = self.nodes[from].parent;
        while let Some(idx) = current {
            if !self.update_proof(idx) {
                return;
            }
            current = self.nodes[idx].parent;
        }
    }

    /// A child proven winning for the side to move at `idx`, if any.
    fn proven_winning_child(&self, idx: usize) -> Option<usize> {
        let win = if self.nodes[idx].state.is_white_turn { 1.0 } else { -1.0 };
        self.nodes[idx]
            .children
            .iter()
            .copied()
            .find(|&c| self.nodes[c].proof == Some(win))
    }

    fn is_proven_loss_for_mover(&self, parent: usize, child: usize) -> bool {
        let loss = if self.nodes[parent].state.is_white_turn { -1.0 } else { 1.0 };
        self.nodes[child].proof == Some(loss)
    }
}

// ---------------------------------------------------------------------------
// Dirichlet root noise (self-play exploration)
// ---------------------------------------------------------------------------
//
// Root-only and self-play-only: mixing noise into an evaluation or arena game
// would corrupt the thing being measured. alpha = 0.3, epsilon = 0.25.
//
// D4 puts RNG under statistical parity, so this need not reproduce numpy's
// stream -- only be a correct Dirichlet sample.

pub const DIRICHLET_ALPHA: f64 = 0.3;
pub const DIRICHLET_EPSILON: f64 = 0.25;

impl Rng {
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let mut u1 = self.next_f64();
        if u1 < 1e-300 {
            u1 = 1e-300;
        }
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Marsaglia-Tsang. For alpha < 1 it uses the boost
    /// Gamma(a) = Gamma(a+1) * U^(1/a), which matters here: alpha is 0.3.
    fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let u = self.next_f64().max(1e-300);
            return self.gamma(alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    fn dirichlet(&mut self, alpha: f64, n: usize) -> Vec<f64> {
        let samples: Vec<f64> = (0..n).map(|_| self.gamma(alpha)).collect();
        let total: f64 = samples.iter().sum();
        if total <= 0.0 {
            return vec![1.0 / n as f64; n];
        }
        samples.iter().map(|s| s / total).collect()
    }
}

impl Arena {
    /// Mix Dirichlet noise into the root children's priors.
    pub fn add_root_noise(&mut self, alpha: f64, epsilon: f64, rng: &mut Rng) {
        let children = self.nodes[0].children.clone();
        if children.is_empty() {
            return;
        }
        let noise = rng.dirichlet(alpha, children.len());
        for (&child, n) in children.iter().zip(noise) {
            self.nodes[child].prior = (1.0 - epsilon) * self.nodes[child].prior + epsilon * n;
        }
    }

    /// Keep only the subtree under `child`, making it the new root.
    ///
    /// **Rebasing, derived rather than guessed.** A node's Q is stored in its
    /// *parent's* side-to-move frame (see `backpropagate`), while a root's Q is
    /// in its *own*. Rerooting severs exactly one parent link and leaves every
    /// other parent-child relationship intact, so only the new root's frame
    /// changes -- its descendants keep theirs. The new root therefore needs its
    /// accumulated value negated exactly when the side to move differs from the
    /// old root's, and nothing else needs touching.
    ///
    /// The Python engine refuses reuse across a side change rather than rebase,
    /// which costs it the whole tree on every Black move. With this the tree
    /// survives the entire game, as LC0's does.
    pub fn reroot(&mut self, child: usize) {
        let old_root_white = self.nodes[0].state.is_white_turn;
        let new_root_white = self.nodes[child].state.is_white_turn;
        if old_root_white != new_root_white {
            self.nodes[child].total_value = -self.nodes[child].total_value;
        }
        let mut order = vec![child];
        let mut i = 0;
        while i < order.len() {
            let node = order[i];
            for &c in &self.nodes[node].children {
                order.push(c);
            }
            i += 1;
        }
        let mut mapping = std::collections::HashMap::new();
        for (new_idx, &old_idx) in order.iter().enumerate() {
            mapping.insert(old_idx, new_idx);
        }
        let mut fresh: Vec<Node> = Vec::with_capacity(order.len());
        for &old_idx in &order {
            let old = &self.nodes[old_idx];
            fresh.push(Node {
                state: old.state.clone(),
                parent: if old_idx == child {
                    None
                } else {
                    old.parent.and_then(|p| mapping.get(&p).copied())
                },
                action: if old_idx == child { None } else { old.action.clone() },
                prior: old.prior,
                visit_count: old.visit_count,
                total_value: old.total_value,
                children: old
                    .children
                    .iter()
                    .filter_map(|c| mapping.get(c).copied())
                    .collect(),
                is_expanded: old.is_expanded,
                proof: old.proof,
            });
        }
        self.nodes = fresh;
    }
}

// ---------------------------------------------------------------------------
// Selection layer: overrides, oscillation penalty, temperature
// ---------------------------------------------------------------------------
//
// These are the owner's product decisions (§0.3), not implementation detail,
// and they port verbatim:
//   * king safety (2026-07-12, engine-wide) -- never hand over an immediate
//     king capture when a searched alternative survives;
//   * White first half (2026-07-17) -- an m1 must keep at least one king-safe
//     completion whenever a searched alternative does, because the m2-level
//     override cannot repair an m1 blunder;
//   * oscillation penalty (2026-07-17) -- *penalise* exact reversals, do not
//     forbid them; the discount only decides ties.
//
// Both overrides re-rank among the search's OWN children by visit count, and
// touch nothing unless the chosen action is actually bad.

pub const OSCILLATION_VISIT_PENALTY: f64 = 0.10;

/// True if, after this turn-completing action, the opponent can capture the
/// mover's king immediately.
fn hangs_king(state: &Game, action: &str) -> bool {
    let mut tmp = state.clone();
    if tmp.apply_half(action).is_err() {
        return false;
    }
    if tmp.is_terminal_rust() {
        return false; // the action itself ended the game
    }
    if tmp.is_white_turn {
        crate::eval::white_threat_scan(tmp.board_ref(), tmp.white_half_pending)
    } else {
        crate::eval::black_threat_scan(tmp.board_ref())
    }
}

/// True if after this first half-move EVERY legal second half hangs the king.
fn m1_dooms_king(state: &Game, m1: &str) -> bool {
    let mut tmp = state.clone();
    if tmp.apply_half(m1).is_err() {
        return false;
    }
    if tmp.is_terminal_rust() {
        return false; // a winning m1 is never doomed
    }
    for m2 in tmp.search_actions_rust() {
        if !hangs_king(&tmp, &m2) {
            return false;
        }
    }
    true
}

fn is_reversal(action: &str, prev: &[(u8, u8)]) -> bool {
    let b = action.as_bytes();
    if b.len() < 4 {
        return false;
    }
    let from = (b[1] - b'1') * 8 + (b[0] - b'a');
    let to = (b[3] - b'1') * 8 + (b[2] - b'a');
    prev.iter().any(|&(pf, pt)| from == pt && to == pf)
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
    #[pyo3(signature = (fen, white_half_pending=false, turn_count=0, history=None))]
    fn new(
        fen: &str,
        white_half_pending: bool,
        turn_count: u32,
        history: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // None of these are decoration. `pending` selects a different action
        // set, `turn_count` decides the cap, and `history` is what the
        // oscillation penalty reads — a tree built without it silently stops
        // penalising reversals.
        let root = Game::from_state_with_history(
            fen,
            white_half_pending,
            turn_count,
            &history.unwrap_or_default(),
        )
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

    /// Batched PUCT. `eval_fn(batch_bytes, n, channels)` must return
    /// `(values_bytes, policy_bytes)` — n values and n*policy_width logits,
    /// little-endian. Bytes, not lists: a batch of 16 is ~17k floats and list
    /// marshalling would cost more than the search it serves.
    #[pyo3(signature = (simulations, eval_fn, batch_size=16, channels=17,
                        c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION,
                        allow_early_stop=true, root_noise=false, seed=20260803,
                        heuristic_values=false, solver=false))]
    fn run_batched_puct(
        &mut self,
        py: Python<'_>,
        simulations: usize,
        eval_fn: &Bound<'_, PyAny>,
        batch_size: usize,
        channels: usize,
        c_puct: f64,
        fpu_reduction: f64,
        allow_early_stop: bool,
        root_noise: bool,
        seed: u64,
        // HybridEvaluator takes VALUES from the heuristic and only the
        // POLICY from the network, and it skips the forward entirely when
        // |heuristic| >= 0.95. Without this the native search would use NN
        // values wherever generation uses hybrid ones -- same moves early,
        // different tree once values diverge.
        heuristic_values: bool,
        // Certainty propagation. OFF by default: it changes what the engine
        // plays, so under DIRECTIVE section 0.1 it lands behind a flag and is
        // measured on its own rather than folded into the port. Declared last
        // to match the pyo3 signature order exactly.
        solver: bool,
    ) -> PyResult<()> {
        let call = |py: Python<'_>, nodes: &[usize], arena: &Arena| -> PyResult<(Vec<f64>, Vec<f32>)> {
            let mut buf: Vec<f32> = Vec::with_capacity(nodes.len() * channels * 64);
            for &n in nodes {
                arena.encode_into(n, channels, &mut buf);
            }
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4)
            };
            let result = eval_fn.call1((PyBytes::new(py, bytes), nodes.len(), channels))?;
            let (vb, pb): (Vec<u8>, Vec<u8>) = result.extract()?;
            // The value head speaks in the SIDE-TO-MOVE perspective; the tree
            // backpropagates in White's. `NNEvaluator._to_white_perspective`
            // does this conversion in Python, and a bridge that forwards the
            // raw value silently flips the sign at every Black-to-move leaf.
            // The conversion lives here, not in the bridge, because the search
            // is what knows each leaf's side.
            let values: Vec<f64> = vb
                .chunks_exact(4)
                .zip(nodes.iter())
                .map(|(c, &node)| {
                    let raw = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                    if arena.nodes[node].state.is_white_turn {
                        raw
                    } else {
                        -raw
                    }
                })
                .collect();
            let policies: Vec<f32> = pb
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((values, policies))
        };

        if self.arena.nodes[0].state.is_terminal_rust() {
            let v = self.arena.nodes[0].state.result_rust().unwrap_or(0.0);
            self.arena.backpropagate(0, v);
            return Ok(());
        }

        // Root synchronously first, so the batch loop descends into real children.
        let (values, policies) = call(py, &[0], &self.arena)?;
        let root_value = if heuristic_values {
            let st = &self.arena.nodes[0].state;
            crate::eval::evaluate(st.board_ref(), st.is_white_turn, st.white_half_pending)
        } else {
            *values.first().unwrap_or(&0.0)
        };
        if !self.arena.nodes[0].is_expanded {
            let logits = if policies.len() >= crate::encoding::LEGACY_POLICY_SIZE {
                Some(policies.as_slice())
            } else {
                None
            };
            self.arena.expand_with_policy(0, logits);
        }
        self.arena.backpropagate(0, root_value);
        if root_noise && !self.arena.nodes[0].children.is_empty() {
            let mut rng = Rng::new(seed);
            self.arena
                .add_root_noise(DIRICHLET_ALPHA, DIRICHLET_EPSILON, &mut rng);
        }
        let mut sims_done = 1usize;

        while sims_done < simulations {
            if allow_early_stop && self.arena.should_stop_early(0, sims_done, simulations) {
                break;
            }
            let target = batch_size.min(simulations - sims_done);
            // (node, immediate value if decided, expand-with-uniform-priors)
            let mut leaves: Vec<(usize, Option<f64>, bool)> = Vec::with_capacity(target);
            let mut pending: Vec<usize> = Vec::with_capacity(target);
            while leaves.len() < target {
                let node = self.arena.select_puct(0, c_puct, fpu_reduction, solver);
                if pending.contains(&node) {
                    break; // frontier exhausted; padding would be duplicate work
                }
                pending.push(node);
                self.arena.apply_virtual_loss(node);
                let state = &self.arena.nodes[node].state;
                match state.result_rust() {
                    // Terminal: backprop only, never expanded.
                    Some(v) => leaves.push((node, Some(v), false)),
                    None => {
                        // Decided-by-clamp: the Python evaluator returns the
                        // clamp with policy=None and skips the forward, so the
                        // node is expanded with uniform priors.
                        if heuristic_values {
                            // Mirrors HybridEvaluator: the heuristic is the
                            // value, and |value| >= 0.95 means the position is
                            // decided, so no forward and uniform priors.
                            let h = crate::eval::evaluate(
                                state.board_ref(),
                                state.is_white_turn,
                                state.white_half_pending,
                            );
                            if h.abs() >= 0.95 {
                                leaves.push((node, Some(h), true));
                            } else {
                                leaves.push((node, None, true));
                            }
                        } else {
                            match crate::eval::pre_nn_clamp(
                                state.board_ref(),
                                state.is_white_turn,
                                state.white_half_pending,
                            ) {
                                Some(v) => leaves.push((node, Some(v), true)),
                                None => leaves.push((node, None, true)),
                            }
                        }
                    }
                }
            }
            if leaves.is_empty() {
                break;
            }

            let nn_nodes: Vec<usize> = leaves
                .iter()
                .filter(|(_, imm, _)| imm.is_none())
                .map(|(n, _, _)| *n)
                .collect();
            let (nn_values, nn_policies) = if nn_nodes.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                call(py, &nn_nodes, &self.arena)?
            };

            let mut nn_cursor = 0usize;
            for (node, immediate, expand) in &leaves {
                self.arena.revert_virtual_loss(*node);
                match immediate {
                    Some(v) => {
                        if *expand && !self.arena.nodes[*node].is_expanded {
                            self.arena.expand_with_policy(*node, None);
                        }
                        if solver {
                            // Only a king capture proves anything; the cap's
                            // +-0.5 relabel deliberately does not.
                            if let Some(p) = self.arena.terminal_proof(*node) {
                                self.arena.nodes[*node].proof = Some(p);
                            }
                        }
                        self.arena.backpropagate(*node, *v);
                        if solver && self.arena.nodes[*node].proof.is_some() {
                            self.arena.propagate_proof(*node);
                        }
                    }
                    None => {
                        let value = if heuristic_values {
                            let st = &self.arena.nodes[*node].state;
                            crate::eval::evaluate(
                                st.board_ref(),
                                st.is_white_turn,
                                st.white_half_pending,
                            )
                        } else {
                            *nn_values.get(nn_cursor).unwrap_or(&0.0)
                        };
                        let policy_width = if nn_nodes.is_empty() {
                            crate::encoding::LEGACY_POLICY_SIZE
                        } else {
                            nn_policies.len() / nn_nodes.len()
                        };
                        let start = nn_cursor * policy_width;
                        if !self.arena.nodes[*node].is_expanded {
                            let logits = if policy_width >= crate::encoding::LEGACY_POLICY_SIZE
                                && nn_policies.len() >= start + policy_width {
                                Some(&nn_policies[start..start + policy_width])
                            } else {
                                None
                            };
                            self.arena.expand_with_policy(*node, logits);
                        }
                        self.arena.backpropagate(*node, value);
                        if solver {
                            // A freshly expanded node may already be decided --
                            // e.g. every child hands over the king.
                            let children = self.arena.nodes[*node].children.clone();
                            for child in children {
                                if let Some(p) = self.arena.terminal_proof(child) {
                                    self.arena.nodes[child].proof = Some(p);
                                }
                            }
                            if self.arena.update_proof(*node) {
                                self.arena.propagate_proof(*node);
                            }
                        }
                        nn_cursor += 1;
                    }
                }
            }
            sims_done += leaves.len();
        }
        Ok(())
    }

    /// Selection, ported from `get_best_action`: raw visit distribution for the
    /// training target, oscillation-adjusted visits for the move actually
    /// played, then the two owner overrides, then the *selected child's* Q.
    ///
    /// Reporting the selected child's Q rather than the root average is an
    /// owner decision (2026-07-17): the root average is a visit-weighted mean
    /// including simulations spent refuting losing siblings, so a proven mate
    /// reads ~+0.7. The selected child sits at exactly ±1.0 for proven lines.
    /// NOTE: `seed` constructs a fresh RNG on every call. A caller that passes
    /// a constant gets an identical stream each time — temperature sampling
    /// returns the same move and Dirichlet noise the same vector, which looks
    /// random because different positions still differ. Callers must advance
    /// the seed per decision; `NativeMCTS` does.
    #[pyo3(signature = (temperature=1.0, seed=20260803))]
    fn best_action(
        &self,
        temperature: f64,
        seed: u64,
    ) -> (Option<String>, Vec<(String, f64)>, f64) {
        let root = &self.arena.nodes[0];
        if root.children.is_empty() {
            return (None, Vec::new(), 0.0);
        }
        // A proven win is not a thing to weigh against visit counts. Reported
        // with an exact value, which is also what the owner asked the search
        // value to say for proven lines (2026-07-17).
        if let Some(winner) = self.arena.proven_winning_child(0) {
            let probs: Vec<(String, f64)> = root
                .children
                .iter()
                .map(|&c| {
                    let v = self.arena.nodes[c].visit_count as f64;
                    (self.arena.nodes[c].action.clone().unwrap_or_default(), v)
                })
                .collect();
            let total: f64 = probs.iter().map(|(_, v)| *v).sum();
            let probs = if total > 0.0 {
                probs.into_iter().map(|(a, v)| (a, v / total)).collect()
            } else {
                let u = 1.0 / root.children.len() as f64;
                probs.into_iter().map(|(a, _)| (a, u)).collect()
            };
            // The reported value is in the ROOT'S SIDE-TO-MOVE perspective --
            // a root child's Q accumulates in exactly that frame, so the
            // non-solver path returns +1 for "the mover wins". Returning
            // White's perspective here instead would flip the sign for Black
            // and disagree with the very path this is meant to sharpen.
            return (self.arena.nodes[winner].action.clone(), probs, 1.0);
        }
        let info: Vec<(usize, String, f64)> = root
            .children
            .iter()
            .map(|&c| {
                (
                    c,
                    self.arena.nodes[c].action.clone().unwrap_or_default(),
                    self.arena.nodes[c].visit_count as f64,
                )
            })
            .collect();

        // Training target keeps the RAW distribution.
        let total: f64 = info.iter().map(|(_, _, v)| *v).sum();
        let probs: Vec<(String, f64)> = if total <= 0.0 {
            let uniform = 1.0 / info.len() as f64;
            info.iter().map(|(_, a, _)| (a.clone(), uniform)).collect()
        } else {
            info.iter()
                .map(|(_, a, v)| (a.clone(), v / total))
                .collect()
        };

        // Selection uses oscillation-adjusted visits.
        let state = &root.state;
        let prev = state.own_previous_moves();
        let adjusted: Vec<f64> = if state.turn_completing() && !prev.is_empty() {
            info.iter()
                .map(|(_, a, v)| {
                    if is_reversal(a, &prev) {
                        v * (1.0 - OSCILLATION_VISIT_PENALTY)
                    } else {
                        *v
                    }
                })
                .collect()
        } else {
            info.iter().map(|(_, _, v)| *v).collect()
        };

        let mut selected = if temperature < 0.01 {
            // Python's max() keeps the FIRST maximum; strict > matches it.
            let mut best = 0usize;
            for i in 1..adjusted.len() {
                if adjusted[i] > adjusted[best] {
                    best = i;
                }
            }
            info[best].1.clone()
        } else {
            let weights: Vec<f64> = adjusted
                .iter()
                .map(|v| v.max(0.0).powf(1.0 / temperature))
                .collect();
            let total_w: f64 = weights.iter().sum();
            let mut rng = Rng::new(seed);
            if total_w == 0.0 {
                let idx = (rng.next_u64() % info.len() as u64) as usize;
                info[idx].1.clone()
            } else {
                let mut draw = (rng.next_u64() as f64 / u64::MAX as f64) * total_w;
                let mut chosen = info.len() - 1;
                for (i, w) in weights.iter().enumerate() {
                    draw -= w;
                    if draw <= 0.0 {
                        chosen = i;
                        break;
                    }
                }
                info[chosen].1.clone()
            }
        };

        // Override 1: a first half-move must keep a king-safe completion.
        if state.is_white_turn && !state.white_half_pending && m1_dooms_king(state, &selected) {
            let mut order: Vec<&(usize, String, f64)> = info.iter().collect();
            order.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            for cand in order {
                if cand.1 != selected && !m1_dooms_king(state, &cand.1) {
                    selected = cand.1.clone();
                    break;
                }
            }
        }

        // Override 2: never hand over an immediate king capture.
        if state.turn_completing() && hangs_king(state, &selected) {
            let mut order: Vec<&(usize, String, f64)> = info.iter().collect();
            order.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            for cand in order {
                if cand.1 != selected && !hangs_king(state, &cand.1) {
                    selected = cand.1.clone();
                    break;
                }
            }
        }

        let value = info
            .iter()
            .find(|(_, a, v)| *a == selected && *v > 0.0)
            .map(|(c, _, _)| self.arena.nodes[*c].q_value())
            .unwrap_or_else(|| self.arena.nodes[0].q_value());

        (Some(selected), probs, value)
    }

    /// Reuse the subtree under `action` as the new root. False when that child
    /// does not exist. Only valid across White's first -> second half-move.
    fn reroot(&mut self, action: &str) -> bool {
        let target = self.arena.nodes[0]
            .children
            .iter()
            .copied()
            .find(|&c| self.arena.nodes[c].action.as_deref() == Some(action));
        match target {
            Some(child) => {
                self.arena.reroot(child);
                true
            }
            None => false,
        }
    }

    #[pyo3(signature = (alpha=DIRICHLET_ALPHA, epsilon=DIRICHLET_EPSILON, seed=20260803))]
    fn add_root_noise(&mut self, alpha: f64, epsilon: f64, seed: u64) {
        let mut rng = Rng::new(seed);
        self.arena.add_root_noise(alpha, epsilon, &mut rng);
    }

    /// Proof state of the root's children, in White's perspective.
    fn root_proofs(&self) -> Vec<(String, Option<f64>)> {
        self.arena.nodes[0]
            .children
            .iter()
            .map(|&c| {
                (
                    self.arena.nodes[c].action.clone().unwrap_or_default(),
                    self.arena.nodes[c].proof,
                )
            })
            .collect()
    }

    fn root_proof(&self) -> Option<f64> {
        self.arena.nodes[0].proof
    }

    fn root_priors(&self) -> Vec<(String, f64)> {
        self.arena.nodes[0]
            .children
            .iter()
            .map(|&c| {
                (
                    self.arena.nodes[c].action.clone().unwrap_or_default(),
                    self.arena.nodes[c].prior,
                )
            })
            .collect()
    }

    fn max_depth(&self) -> usize {
        self.arena.max_depth()
    }

    fn pv_depth(&self) -> usize {
        self.arena.pv_depth()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTree>()?;
    m.add("C_PUCT", C_PUCT)?;
    m.add("FPU_REDUCTION", FPU_REDUCTION)?;
    m.add("EXPLORATION_CONSTANT", EXPLORATION_CONSTANT)?;
    m.add("DIRICHLET_ALPHA", DIRICHLET_ALPHA)?;
    m.add("DIRICHLET_EPSILON", DIRICHLET_EPSILON)?;
    m.add("VIRTUAL_LOSS", VIRTUAL_LOSS)?;
    m.add("OSCILLATION_VISIT_PENALTY", OSCILLATION_VISIT_PENALTY)?;
    Ok(())
}
