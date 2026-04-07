// Pure computation — no PyO3, no allocator awareness.
// Uses std::simd (portable SIMD) — compiles to NEON on ARM, AVX on x86.
// Requires nightly: rustup run nightly cargo build
//
// The commented alternatives auto-vectorize to identical assembly under -O.
// Verify with: rustc -O --crate-type lib --emit asm -C target-cpu=native

use std::simd::prelude::*;

const LANES: usize = 4;

/// Pad a slice to a multiple of LANES with a fill value.
fn pad_slice(a: &[f32], fill: f32) -> Vec<f32> {
    let rem = a.len() % LANES;
    if rem == 0 {
        return a.to_vec();
    }
    let mut v = Vec::with_capacity(a.len() + LANES - rem);
    v.extend_from_slice(a);
    v.resize(v.capacity(), fill);
    v
}

// auto-vectorizes: a.iter().zip(b).map(|(x, y)| x + y).collect()
pub(crate) fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len().min(b.len());
    let a = pad_slice(&a[..len], 0.0);
    let b = pad_slice(&b[..len], 0.0);

    let (chunks_a, _) = a.as_chunks::<LANES>();
    let (chunks_b, _) = b.as_chunks::<LANES>();

    let mut out = Vec::with_capacity(a.len());
    for (a, b) in chunks_a.iter().zip(chunks_b) {
        let r = f32x4::from_array(*a) + f32x4::from_array(*b);
        out.extend_from_slice(&r.to_array());
    }
    out.truncate(len);
    out
}

// auto-vectorizes: a.iter().zip(b).map(|(x, y)| x * y).collect()
pub(crate) fn mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len().min(b.len());
    let a = pad_slice(&a[..len], 0.0);
    let b = pad_slice(&b[..len], 0.0);

    let (chunks_a, _) = a.as_chunks::<LANES>();
    let (chunks_b, _) = b.as_chunks::<LANES>();

    let mut out = Vec::with_capacity(a.len());
    for (a, b) in chunks_a.iter().zip(chunks_b) {
        let r = f32x4::from_array(*a) * f32x4::from_array(*b);
        out.extend_from_slice(&r.to_array());
    }
    out.truncate(len);
    out
}

// auto-vectorizes: a.iter().map(|x| -x).collect()
pub(crate) fn neg_f32(a: &[f32]) -> Vec<f32> {
    let len = a.len();
    let a = pad_slice(a, 0.0);

    let (chunks, _) = a.as_chunks::<LANES>();

    let mut out = Vec::with_capacity(a.len());
    for chunk in chunks {
        let r = -f32x4::from_array(*chunk);
        out.extend_from_slice(&r.to_array());
    }
    out.truncate(len);
    out
}

// auto-vectorizes: a.iter().sum()
pub(crate) fn sum_f32(a: &[f32]) -> f32 {
    let a = pad_slice(a, 0.0);
    let (chunks, _) = a.as_chunks::<LANES>();

    let mut acc = f32x4::splat(0.0);
    for chunk in chunks {
        acc += f32x4::from_array(*chunk);
    }
    acc.reduce_sum()
}
