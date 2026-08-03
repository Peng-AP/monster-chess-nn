//! Native rules/search core for Monster Chess (DIRECTIVE E1-E4).
//!
//! Parity before improvement (DIRECTIVE §0.1): this crate reproduces the
//! measured behaviour of `src/monster_chess.py` and friends. Nothing here may
//! "fix" the Python engine's quirks — the quirks are the contract, and the
//! differential harness treats any divergence as a defect in *this* crate.

use pyo3::prelude::*;

mod bitboard;
mod monster;

/// Build identity, so the Python side can assert it loaded the crate it built.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn monster_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    bitboard::register(m)?;
    monster::register(m)?;
    Ok(())
}
