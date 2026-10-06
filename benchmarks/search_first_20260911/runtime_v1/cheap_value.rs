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
    let b = g.board_ref();
    let mut out = Vec::with_capacity(40);
    let mut occupied = b.occupied;
    while occupied != 0 {
        let sq = occupied.trailing_zeros() as u8;
        occupied &= occupied - 1;
        let color_offset = if b.occupied_co[WHITE] & (1u64 << sq) != 0 { 0 } else { 6 };
        out.push(((color_offset + b.piece_type_at(sq).unwrap() as usize - 1)*64 + sq as usize, 1.0));
    }
    let phase = if !g.is_white_turn { 2 } else if g.white_half_pending { 1 } else { 0 };
    out.push((768 + phase, 1.0));
    for (i, sq) in [7, 0, 63, 56].iter().enumerate() {
        if b.castling & (1u64 << sq) != 0 { out.push((771+i, 1.0)); }
    }
    if let Some(sq) = b.ep_square { out.push((775+sq as usize, 1.0)); }
    out.push((839, (150u32.saturating_sub(g.turn_count)) as f32/150.0));
    out
}

pub struct Weights {
    width: usize,
    hidden: usize,
    w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32,
}

impl Weights {
    pub fn evaluate(&self, g: &Game) -> f64 {
        let mut a = self.b1.clone();
        for (feature, value) in features(g) {
            for (v, w) in a.iter_mut().zip(&self.w1[feature*self.width..(feature+1)*self.width]) {
                *v += value * w;
            }
        }
        for v in &mut a { *v = v.max(0.0); }
        let mut output = self.b3;
        for j in 0..self.hidden {
            let mut v = self.b2[j];
            for (x, w) in a.iter().zip(&self.w2[j*self.width..(j+1)*self.width]) { v += x*w; }
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
        if data.len() < 20 || &data[..8] != b"MCSV001\0" {
            return Err(PyValueError::new_err("invalid MCSV001 header"));
        }
        let number = |i| u32::from_le_bytes(data[i..i+4].try_into().unwrap()) as usize;
        let (inputs, width, hidden) = (number(8), number(12), number(16));
        if inputs != INPUTS || width == 0 || width > 4096 || hidden == 0 || hidden > 512 {
            return Err(PyValueError::new_err("invalid model dimensions"));
        }
        let count = INPUTS*width + width + width*hidden + hidden + hidden + 1;
        if data.len() != 20 + 4*count { return Err(PyValueError::new_err("invalid weight length")); }
        let floats: Vec<f32> = data[20..].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if floats.iter().any(|x| !x.is_finite()) { return Err(PyValueError::new_err("nonfinite weights")); }
        let mut offset = 0;
        let mut take = |n| { let v = floats[offset..offset+n].to_vec(); offset += n; v };
        let weights = Weights { width, hidden, w1: take(INPUTS*width), b1: take(width),
            w2: take(width*hidden), b2: take(hidden), w3: take(hidden), b3: take(1)[0] };
        Ok(Self { weights: Arc::new(weights) })
    }

    #[pyo3(signature = (fen, pending=false, turn_count=0))]
    fn evaluate(&self, fen: &str, pending: bool, turn_count: u32) -> PyResult<f64> {
        let g = Game::from_state(fen, pending, turn_count).map_err(PyValueError::new_err)?;
        Ok(self.weights.evaluate(&g))
    }

    #[staticmethod]
    #[pyo3(signature = (fen, pending=false, turn_count=0))]
    fn features(fen: &str, pending: bool, turn_count: u32) -> PyResult<Vec<(usize, f32)>> {
        let g = Game::from_state(fen, pending, turn_count).map_err(PyValueError::new_err)?;
        Ok(features(&g))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CheapValue>()?;
    Ok(())
}
