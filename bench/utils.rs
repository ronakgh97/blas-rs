use blas::{isamax, sasum, saxpy, sdot, sgemm, sgemv, snrm2};
use blas_rs::utils::get_cache_size;
use std::sync::LazyLock;

extern crate intel_mkl_src;

/// [This CPU](https://www.intel.com/content/www/us/en/products/sku/235996/intel-core-i7-processor-14650hx-30m-cache-up-to-5-20-ghz/specifications.html)
pub static MAX_L1L2_KB: LazyLock<f64> = LazyLock::new(|| {
    let (l1, l2, _) = get_cache_size();
    (l1 + l2) as f64 // <- this assumption is a bit arbitrary but fine
});
#[inline(always)]
pub fn axpy_intel_mkl(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        saxpy(n, alpha, x, incx, y, incy);
    }
}
#[inline(always)]
pub fn nrm2_intel_mkl(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe { snrm2(n, x, incx) }
}
#[inline(always)]
pub fn asum_intel_mkl(n: i32, x: &[f32], incx: i32) -> f32 {
    unsafe { sasum(n, x, incx) }
}
#[inline(always)]
pub fn dot_intel_mkl(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    unsafe { sdot(n, x, incx, y, incy) }
}
#[inline(always)]
pub fn i_amax_intel_mkl(n: i32, x: &[f32], incx: i32) -> usize {
    let idx = unsafe { isamax(n, x, incx) };
    idx.saturating_sub(1) // MKL returns 1-indexed, ours returns 0-indexed
}
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn gemv_intel_mkl(
    m: i32,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
    is_trans_a: bool,
) {
    unsafe {
        sgemv(
            if is_trans_a { b'T' } else { b'N' },
            m,
            n,
            alpha,
            a,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        );
    }
}
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_intel_mkl(
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
    is_trans_a: bool,
    is_trans_b: bool,
) {
    unsafe {
        sgemm(
            if is_trans_a { b'T' } else { b'N' },
            if is_trans_b { b'T' } else { b'N' },
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
}
