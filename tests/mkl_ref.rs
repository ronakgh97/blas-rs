#![allow(dead_code)]
use blas::{isamax, sasum, saxpy, scopy, sdot, sgemm, sgemv, snrm2, srot, srotg, sscal, sswap};

extern crate intel_mkl_src;

#[inline(always)]
/// Compare two slices of f32 values with a tolerance of 1e-5 (absolute or relative).
pub fn assert_eq_slices(ours: &[f32], mkl: &[f32], msg: &str) {
    assert_eq!(ours.len(), mkl.len(), "{msg}: len mismatch");
    for (i, (&o, &m)) in ours.iter().zip(mkl.iter()).enumerate() {
        let err = (o - m).abs();
        let rel = if m.abs() > 1.0 { err / m.abs() } else { err };
        assert!(
            err <= 1e-5 || rel <= 1e-5,
            "{msg}[{i}]: ours={o}, mkl={m}, err={err}"
        );
    }
}

#[inline(always)]
/// Compare two f32 values with a tolerance of 1e-5 (absolute or relative).
pub fn assert_eq_f32(ours: f32, mkl: f32, msg: &str) {
    let err = (ours - mkl).abs();
    let rel = if mkl.abs() > 1.0 {
        err / mkl.abs()
    } else {
        err
    };
    assert!(
        err <= 1e-5 || rel <= 1e-5,
        "{msg}: ours={ours}, mkl={mkl}, err={err}"
    );
}

// --- lvl1

pub fn mkl_axpy(n: usize, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe { saxpy(n as i32, alpha, x, incx, y, incy) }
}
pub fn mkl_scal(n: usize, alpha: f32, x: &mut [f32], incx: i32) {
    unsafe { sscal(n as i32, alpha, x, incx) }
}
pub fn mkl_copy(n: usize, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe { scopy(n as i32, x, incx, y, incy) }
}
pub fn mkl_swap(n: usize, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe { sswap(n as i32, x, incx, y, incy) }
}
pub fn mkl_dot(n: usize, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe { sdot(n as i32, x, incx, y, incy) }
}
pub fn mkl_nrm2(n: usize, x: &[f32], incx: i32) -> f32 {
    unsafe { snrm2(n as i32, x, incx) }
}
pub fn mkl_asum(n: usize, x: &[f32], incx: i32) -> f32 {
    unsafe { sasum(n as i32, x, incx) }
}
pub fn mkl_i_amax(n: usize, x: &[f32], incx: i32) -> usize {
    let idx = unsafe { isamax(n as i32, x, incx) };
    idx.saturating_sub(1) // MKL returns 1-indexed, our i_amax returns 0-indexed.
}
pub fn mkl_rot(n: usize, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, c: f32, s: f32) {
    unsafe { srot(n as i32, x, incx, y, incy, c, s) }
}
pub fn mkl_rotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    unsafe { srotg(a, b, c, s) }
}

// --- lvl2

#[allow(clippy::too_many_arguments)]
pub fn mkl_gemv(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
    trans: bool,
) {
    unsafe {
        sgemv(
            if trans { b'T' } else { b'N' },
            m as i32,
            n as i32,
            alpha,
            a,
            lda as i32,
            x,
            incx,
            beta,
            y,
            incy,
        );
    }
}

// --- lvl3

#[allow(clippy::too_many_arguments)]
pub fn mkl_gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
    trans_a: bool,
    trans_b: bool,
) {
    unsafe {
        sgemm(
            if trans_a { b'T' } else { b'N' },
            if trans_b { b'T' } else { b'N' },
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}
