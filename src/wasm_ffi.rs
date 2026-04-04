//! C-ABI export layer for MoonBit wasm interop.
//!
//! Enable with `--features wasm-ffi`. Only meaningful on `wasm32-unknown-unknown`.
//!
//! # Memory layout
//!
//! All pointer arguments (`*_ptr`) are i32 linear-memory addresses.
//! Use `nuts_alloc` / `nuts_free` to manage buffers from MoonBit.
//!
//! # MoonBit side (moon.pkg.json)
//!
//! ```json
//! {
//!   "link": {
//!     "wasm": {
//!       "import-memory": { "module": "nuts_rs", "name": "memory" },
//!       "heap-start-address": 524288
//!     }
//!   }
//! }
//! ```
//!
//! # MoonBit FFI declarations
//!
//! ```moonbit
//! fn nuts_alloc(count : Int) -> Int = "nuts_rs" "nuts_alloc"
//! fn nuts_free(ptr : Int, count : Int) = "nuts_rs" "nuts_free"
//! fn nuts_sample(
//!   dim : Int, start_ptr : Int, num_tune : Int,
//!   num_draws : Int, seed : Int64, out_ptr : Int,
//! ) -> Int = "nuts_rs" "nuts_sample"
//!
//! // Must be exported so nuts-rs can call it
//! pub fn moonbit_logp(pos_ptr : Int, grad_ptr : Int, dim : Int) -> Double { ... }
//! ```

use std::collections::HashMap;

use anyhow::Result;
use nuts_storable::HasDims;
use rand::{SeedableRng, rngs::ChaCha8Rng};
use thiserror::Error;

use crate::{
    CpuMath, DiagGradNutsSettings,
    math::{CpuLogpFunc, CpuMathError, LogpError},
    sampler::sample_sequentially,
};

// ---------------------------------------------------------------------------
// MoonBit logp callback (imported from wasm module "moonbit")
//
// Wasm import: (import "moonbit" "moonbit_logp" (func (param i32 i32 i32) (result f64)))
//
// pos_ptr : linear-memory address of f64[dim] position (read-only)
// grad_ptr: linear-memory address of f64[dim] gradient (write-only output)
// dim     : number of dimensions
// returns : log density (finite), or any non-finite value to signal divergence
// ---------------------------------------------------------------------------
#[link(wasm_import_module = "moonbit")]
unsafe extern "C" {
    fn moonbit_logp(pos_ptr: i32, grad_ptr: i32, dim: i32) -> f64;
}

// ---------------------------------------------------------------------------
// LogpError
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum WasmLogpError {
    #[error("logp callback returned a non-finite value (treated as divergence)")]
    NonFinite,
}

impl LogpError for WasmLogpError {
    fn is_recoverable(&self) -> bool {
        // Non-finite logp is treated as a divergence; sampling continues.
        true
    }
}

// ---------------------------------------------------------------------------
// CpuLogpFunc implementation backed by the MoonBit callback
// ---------------------------------------------------------------------------

struct MoonBitLogpFunc {
    dim: usize,
}

impl HasDims for MoonBitLogpFunc {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [
            ("unconstrained_parameter".to_string(), self.dim as u64),
            ("dim".to_string(), self.dim as u64),
        ]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for MoonBitLogpFunc {
    type LogpError = WasmLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, WasmLogpError> {
        // SAFETY: pos_ptr and grad_ptr are valid f64 slices for the duration of the call.
        // moonbit_logp must not retain these pointers after it returns.
        let val = unsafe {
            moonbit_logp(
                position.as_ptr() as i32,
                gradient.as_mut_ptr() as i32,
                position.len() as i32,
            )
        };
        if val.is_finite() { Ok(val) } else { Err(WasmLogpError::NonFinite) }
    }

    fn expand_vector<R>(&mut self, _rng: &mut R, array: &[f64]) -> Result<Vec<f64>, CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        Ok(array.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Exported C-ABI functions
// ---------------------------------------------------------------------------

/// Allocate a contiguous buffer of `count` f64 values in Rust's heap.
///
/// Returns the linear-memory address as i32, or 0 on allocation failure.
#[unsafe(no_mangle)]
pub extern "C" fn nuts_alloc(count: i32) -> i32 {
    if count <= 0 {
        return 0;
    }
    let Ok(layout) = std::alloc::Layout::array::<f64>(count as usize) else {
        return 0;
    };
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() { 0 } else { ptr as i32 }
}

/// Free a buffer previously allocated by `nuts_alloc`.
///
/// `ptr` and `count` must match exactly what was passed to / returned from `nuts_alloc`.
#[unsafe(no_mangle)]
pub extern "C" fn nuts_free(ptr: i32, count: i32) {
    if ptr == 0 || count <= 0 {
        return;
    }
    let Ok(layout) = std::alloc::Layout::array::<f64>(count as usize) else {
        return;
    };
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) }
}

/// Run NUTS sampling using a MoonBit-provided logp function.
///
/// # Arguments
///
/// - `dim`       — problem dimensionality
/// - `start_ptr` — linear-memory address of `f64[dim]` initial position
/// - `num_tune`  — number of tuning (warm-up) draws
/// - `num_draws` — number of sampling draws
/// - `seed`      — random seed (i64)
/// - `out_ptr`   — linear-memory address of `f64[dim * (num_tune + num_draws)]`
///                 output buffer; caller must allocate with `nuts_alloc`
///
/// # Returns
///
/// `0` on success, `-1` on error.
///
/// Draws are written row-major: draw `i` occupies `out_ptr[i*dim .. (i+1)*dim]`.
/// Tuning draws come first, sampling draws follow.
#[unsafe(no_mangle)]
pub extern "C" fn nuts_sample(
    dim: i32,
    start_ptr: i32,
    num_tune: i32,
    num_draws: i32,
    seed: i64,
    out_ptr: i32,
) -> i32 {
    match run_sampling(dim, start_ptr, num_tune, num_draws, seed, out_ptr) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

fn run_sampling(
    dim: i32,
    start_ptr: i32,
    num_tune: i32,
    num_draws: i32,
    seed: i64,
    out_ptr: i32,
) -> Result<()> {
    let dim = dim as usize;
    let total = (num_tune + num_draws) as u64;

    // SAFETY: MoonBit guarantees start_ptr is a valid f64[dim] for this call.
    let start = unsafe { std::slice::from_raw_parts(start_ptr as *const f64, dim) };

    let math = CpuMath::new(MoonBitLogpFunc { dim });

    let settings = DiagGradNutsSettings {
        num_tune: num_tune as u64,
        num_draws: num_draws as u64,
        ..Default::default()
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
    let iter = sample_sequentially(math, settings, start, total, 0, &mut rng)?;

    // SAFETY: MoonBit guarantees out_ptr is a valid f64[dim * total] buffer.
    let out =
        unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f64, dim * total as usize) };

    for (i, draw) in iter.enumerate() {
        let (position, _progress) = draw?;
        let offset = i * dim;
        out[offset..offset + dim].copy_from_slice(&position);
    }

    Ok(())
}
