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
    fn select_puct(&self, root: usize, c_puct: f64, fpu: f64) -> usize {
        let mut node = root;
        while !self.nodes[node].state.is_terminal_rust() {
            if !self.nodes[node].is_expanded {
                return node;
            }
            if self.nodes[node].children.is_empty() {
                return node;
            }
            node = match self.best_child_puct(node, c_puct, fpu) {
                Some(child) => child,
                None => return node,
            };
        }
        node
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
                let indices: Vec<usize> = actions.iter().map(|a| uci_to_index(a)).collect();
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

fn uci_to_index(uci: &str) -> usize {
    let b = uci.as_bytes();
    if b.len() < 4 {
        return 0;
    }
    let from = ((b[1] - b'1') * 8 + (b[0] - b'a')) as usize;
    let to = ((b[3] - b'1') * 8 + (b[2] - b'a')) as usize;
    from * 64 + to
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

    /// Batched PUCT. `eval_fn(batch_bytes, n, channels)` must return
    /// `(values_bytes, policy_bytes)` — n f32 values and n*4096 f32 logits,
    /// little-endian. Bytes, not lists: a batch of 16 is ~17k floats and list
    /// marshalling would cost more than the search it serves.
    #[pyo3(signature = (simulations, eval_fn, batch_size=16, channels=17,
                        c_puct=C_PUCT, fpu_reduction=FPU_REDUCTION,
                        allow_early_stop=true))]
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
            let values: Vec<f64> = vb
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
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
        let root_value = *values.first().unwrap_or(&0.0);
        if !self.arena.nodes[0].is_expanded {
            let logits = if policies.len() >= 4096 { Some(&policies[..4096]) } else { None };
            self.arena.expand_with_policy(0, logits);
        }
        self.arena.backpropagate(0, root_value);
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
                let node = self.arena.select_puct(0, c_puct, fpu_reduction);
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
                        self.arena.backpropagate(*node, *v);
                    }
                    None => {
                        let value = *nn_values.get(nn_cursor).unwrap_or(&0.0);
                        let start = nn_cursor * 4096;
                        if !self.arena.nodes[*node].is_expanded {
                            let logits = if nn_policies.len() >= start + 4096 {
                                Some(&nn_policies[start..start + 4096])
                            } else {
                                None
                            };
                            self.arena.expand_with_policy(*node, logits);
                        }
                        self.arena.backpropagate(*node, value);
                        nn_cursor += 1;
                    }
                }
            }
            sims_done += leaves.len();
        }
        Ok(())
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
    Ok(())
}
