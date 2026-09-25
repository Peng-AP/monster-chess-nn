//! Sparse piece-square value net, MCSV001 float format. No Python/GPU at leaves.
//! First correctness baseline recomputes sparse sums; incremental updates follow
//! only after profiling and parity tests justify the extra state machinery.
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::game::Game;
use crate::bitboard::WHITE;

pub const INPUTS: usize = 840;

pub fn features(g: &Game) -> Vec<(usize, f32)> {
    let mut out = Vec::with_capacity(40);
    visit_features(g, |i,v| out.push((i,v)));
    out
}

fn visit_features(g: &Game, mut add: impl FnMut(usize, f32)) {
    let b = g.board_ref();
    let mut occupied = b.occupied;
    while occupied != 0 {
        let sq = occupied.trailing_zeros() as u8;
        occupied &= occupied - 1;
        let color_offset = if b.occupied_co[WHITE] & (1u64 << sq) != 0 { 0 } else { 6 };
        add((color_offset + b.piece_type_at(sq).unwrap() as usize - 1)*64 + sq as usize, 1.0);
    }
    let phase = if !g.is_white_turn { 2 } else if g.white_half_pending { 1 } else { 0 };
    add(768 + phase, 1.0);
    for (i, sq) in [7, 0, 63, 56].iter().enumerate() {
        if b.castling & (1u64 << sq) != 0 { add(771+i, 1.0); }
    }
    if let Some(sq) = b.ep_square { add(775+sq as usize, 1.0); }
    add(839, (150u32.saturating_sub(g.turn_count)) as f32/150.0);
}

/// Owned by one search, never shared with another thread or recursive frame.
#[derive(Default)]
pub struct Scratch {
    activation: Vec<f32>, sums: Vec<f32>, active: [u64; 14], budget: f32,
    ready: bool, since_refresh: u32,
    pub incremental_updates: u64, pub refreshes: u64,
}

pub struct Weights {
    inputs: usize,
    width: usize,
    hidden: usize,
    w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32,
}

impl Weights {
    pub fn evaluate(&self, g: &Game) -> f64 {
        self.evaluate_with_scratch(g, &mut Scratch::default())
    }

    pub fn evaluate_with_scratch(&self, g: &Game, scratch: &mut Scratch) -> f64 {
        scratch.activation.clone_from(&self.b1);
        let a = &mut scratch.activation;
        let mut add = |feature: usize, value: f32| {
            for (v, w) in a.iter_mut().zip(&self.w1[feature*self.width..(feature+1)*self.width]) {
                *v += value * w;
            }
        };
        visit_features(g, &mut add);
        if self.inputs==crate::relative_features::INPUTS {
            crate::relative_features::visit(g, &mut add);
        }
        self.output(a)
    }

    /// Reuse the last EVALUATED position, not an assumed parent. Sparse XOR
    /// covers arbitrary DFS jumps, captures, promotions, phase, rights and EP.
    /// Raw sums are separate from the destructive ReLU scratch. Bounded refresh
    /// limits FP drift; this optional path is tolerance-equivalent, not bitwise.
    pub fn evaluate_incremental(&self, g: &Game, scratch: &mut Scratch) -> f64 {
        if self.inputs!=INPUTS {
            scratch.refreshes+=1;
            return self.evaluate_with_scratch(g,scratch);
        }
        let mut active=[0u64;14];
        let mut budget=0.0;
        let mut count=0;
        visit_features(g,|i,v| {
            if i==839 {budget=v;} else {active[i/64]|=1u64<<(i%64);count+=1;}
        });
        let changes: u32=active.iter().zip(scratch.active).map(|(a,b)|(a^b).count_ones()).sum();
        let budget_changed=budget!=scratch.budget;
        if !scratch.ready || scratch.since_refresh>=32 || changes+budget_changed as u32>=count+1 {
            scratch.sums.clone_from(&self.b1);
            visit_features(g,|i,v|self.add_column(&mut scratch.sums,i,v));
            scratch.ready=true;scratch.since_refresh=0;scratch.refreshes+=1;
        } else {
            for (word,(&new,&old)) in active.iter().zip(&scratch.active).enumerate() {
                let mut changed=new^old;
                while changed!=0 {
                    let bit=changed.trailing_zeros() as usize;
                    changed&=changed-1;
                    self.add_column(&mut scratch.sums,word*64+bit,if new&(1u64<<bit)!=0 {1.0} else {-1.0});
                }
            }
            if budget_changed {self.add_column(&mut scratch.sums,839,budget-scratch.budget);}
            scratch.since_refresh+=1;scratch.incremental_updates+=1;
        }
        scratch.active=active;scratch.budget=budget;
        scratch.activation.clone_from(&scratch.sums);
        self.output(&mut scratch.activation)
    }

    #[inline]
    fn add_column(&self, sums: &mut [f32], feature: usize, value: f32) {
        for (v,w) in sums.iter_mut().zip(&self.w1[feature*self.width..(feature+1)*self.width]) {
            *v+=value*w;
        }
    }

    fn output(&self, a: &mut [f32]) -> f64 {
        for v in a.iter_mut() { *v = v.max(0.0); }
        let mut output = self.b3;
        for j in 0..self.hidden {
            let v = self.b2[j] + crate::float_dot::dot(a, &self.w2[j*self.width..(j+1)*self.width]);
            output += v.max(0.0)*self.w3[j];
        }
        output.tanh() as f64
    }
}

#[pyclass]
pub struct CheapValue { pub weights: Arc<Weights> }

#[pymethods]
impl CheapValue {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let data = std::fs::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if data.len() < 20 || (&data[..8] != b"MCSV001\0" && &data[..8] != b"MCSV002\0") {
            return Err(PyValueError::new_err("invalid MCSV header"));
        }
        let number = |i| u32::from_le_bytes(data[i..i+4].try_into().unwrap()) as usize;
        let (inputs, width, hidden) = (number(8), number(12), number(16));
        let expected=if &data[..8]==b"MCSV001\0" {INPUTS} else {crate::relative_features::INPUTS};
        if inputs != expected || width == 0 || width > 4096 || hidden == 0 || hidden > 512 {
            return Err(PyValueError::new_err("invalid model dimensions"));
        }
        let count = inputs*width + width + width*hidden + hidden + hidden + 1;
        if data.len() != 20 + 4*count { return Err(PyValueError::new_err("invalid weight length")); }
        let floats: Vec<f32> = data[20..].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if floats.iter().any(|x| !x.is_finite()) { return Err(PyValueError::new_err("nonfinite weights")); }
        let mut offset = 0;
        let mut take = |n| { let v = floats[offset..offset+n].to_vec(); offset += n; v };
        let weights = Weights { inputs, width, hidden, w1: take(inputs*width), b1: take(width),
            w2: take(width*hidden), b2: take(hidden), w3: take(hidden), b3: take(1)[0] };
        Ok(Self { weights: Arc::new(weights) })
    }

    #[getter]
    fn input_count(&self) -> usize { self.weights.inputs }

    #[pyo3(signature = (fen, pending=false, turn_count=0))]
    fn evaluate(&self, fen: &str, pending: bool, turn_count: u32) -> PyResult<f64> {
        let g = Game::from_state(fen, pending, turn_count).map_err(PyValueError::new_err)?;
        Ok(self.weights.evaluate(&g))
    }

    /// Diagnostic path: one scratch across arbitrary positions, as in DFS.
    #[pyo3(signature = (states, incremental=true))]
    fn evaluate_sequence(&self, states: Vec<(String,bool,u32)>, incremental: bool)
        -> PyResult<(Vec<f64>,u64,u64)> {
        let mut scratch=Scratch::default();let mut values=Vec::with_capacity(states.len());
        for (fen,pending,count) in states {
            let g=Game::from_state(&fen,pending,count).map_err(PyValueError::new_err)?;
            values.push(if incremental {self.weights.evaluate_incremental(&g,&mut scratch)}
                        else {self.weights.evaluate_with_scratch(&g,&mut scratch)});
        }
        Ok((values,scratch.incremental_updates,scratch.refreshes))
    }

    #[staticmethod]
    #[pyo3(signature = (fen, pending=false, turn_count=0, relative=false))]
    fn features(fen: &str, pending: bool, turn_count: u32, relative: bool) -> PyResult<Vec<(usize, f32)>> {
        let g = Game::from_state(fen, pending, turn_count).map_err(PyValueError::new_err)?;
        let mut out=features(&g);
        if relative { crate::relative_features::visit(&g,|i,v|out.push((i,v))); }
        Ok(out)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CheapValue>()?;
    Ok(())
}
