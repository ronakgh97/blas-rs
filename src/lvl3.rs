//! Implementation of Level 3 BLAS routines
use crate::lvl1::scal;
use std::arch::x86_64::*;

// Micro-tile: 16 rows x 6 cols of C
const MR: usize = 16;
const NR: usize = 6;

// Cache blocking tuned for i7-14650HX
const GEMM_NC: usize = 1024; // B panel  (L3)
const GEMM_MC: usize = 512; // A block  (L2)
const GEMM_KC: usize = 256; // K block  (L1)

/// Micro-kernel: `C[16x6] += alpha * A_panel[16 x kc] * B_panel[kc x 6]`
/// - a: packed, k-major, 16 floats per k step
/// - b: packed, k-major, 4 floats per k step
/// - c: column-major tile with leading dimension ldc
///
/// C is touched exactly once and accumulators stay in registers for all the kc.
#[inline(always)]
#[allow(unused)]
unsafe fn kernel_16x4(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    unsafe {
        let mut c00 = _mm256_setzero_ps();
        let mut c10 = _mm256_setzero_ps();
        let mut c01 = _mm256_setzero_ps();
        let mut c11 = _mm256_setzero_ps();
        let mut c02 = _mm256_setzero_ps();
        let mut c12 = _mm256_setzero_ps();
        let mut c03 = _mm256_setzero_ps();
        let mut c13 = _mm256_setzero_ps();

        for p in 0..kc {
            let a0 = _mm256_loadu_ps(a.add(p * MR));
            let a1 = _mm256_loadu_ps(a.add(p * MR + 8));

            let b0 = _mm256_broadcast_ss(&*b.add(p * NR));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c10 = _mm256_fmadd_ps(a1, b0, c10);

            let b1 = _mm256_broadcast_ss(&*b.add(p * NR + 1));
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            let b2 = _mm256_broadcast_ss(&*b.add(p * NR + 2));
            c02 = _mm256_fmadd_ps(a0, b2, c02);
            c12 = _mm256_fmadd_ps(a1, b2, c12);

            let b3 = _mm256_broadcast_ss(&*b.add(p * NR + 3));
            c03 = _mm256_fmadd_ps(a0, b3, c03);
            c13 = _mm256_fmadd_ps(a1, b3, c13);
        }

        let va = _mm256_set1_ps(alpha);
        macro_rules! store_col {
            ($col:expr, $lo:expr, $hi:expr) => {{
                let p = c.add($col * ldc);
                _mm256_storeu_ps(p, _mm256_fmadd_ps(va, $lo, _mm256_loadu_ps(p)));
                _mm256_storeu_ps(
                    p.add(8),
                    _mm256_fmadd_ps(va, $hi, _mm256_loadu_ps(p.add(8))),
                );
            }};
        }
        store_col!(0, c00, c10);
        store_col!(1, c01, c11);
        store_col!(2, c02, c12);
        store_col!(3, c03, c13);
    }
}

/// Micro-kernel: `C[16x6] += alpha * A_panel[16 x kc] * B_panel[kc x 6]`
/// - a: packed, k-major, 16 floats per k step
/// - b: packed, k-major, 6 floats per k step
/// - c: column-major tile with leading dimension ldc
///
/// C is touched exactly once; accumulators stay in registers for all kc.
#[inline(always)]
unsafe fn kernel_16x6(
    kc: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    unsafe {
        let mut c00 = _mm256_setzero_ps();
        let mut c10 = _mm256_setzero_ps();
        let mut c01 = _mm256_setzero_ps();
        let mut c11 = _mm256_setzero_ps();
        let mut c02 = _mm256_setzero_ps();
        let mut c12 = _mm256_setzero_ps();
        let mut c03 = _mm256_setzero_ps();
        let mut c13 = _mm256_setzero_ps();
        let mut c04 = _mm256_setzero_ps();
        let mut c14 = _mm256_setzero_ps();
        let mut c05 = _mm256_setzero_ps();
        let mut c15 = _mm256_setzero_ps();

        for p in 0..kc {
            let a0 = _mm256_loadu_ps(a.add(p * MR));
            let a1 = _mm256_loadu_ps(a.add(p * MR + 8));

            let b0 = _mm256_broadcast_ss(&*b.add(p * NR));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c10 = _mm256_fmadd_ps(a1, b0, c10);

            let b1 = _mm256_broadcast_ss(&*b.add(p * NR + 1));
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            let b2 = _mm256_broadcast_ss(&*b.add(p * NR + 2));
            c02 = _mm256_fmadd_ps(a0, b2, c02);
            c12 = _mm256_fmadd_ps(a1, b2, c12);

            let b3 = _mm256_broadcast_ss(&*b.add(p * NR + 3));
            c03 = _mm256_fmadd_ps(a0, b3, c03);
            c13 = _mm256_fmadd_ps(a1, b3, c13);

            let b4 = _mm256_broadcast_ss(&*b.add(p * NR + 4));
            c04 = _mm256_fmadd_ps(a0, b4, c04);
            c14 = _mm256_fmadd_ps(a1, b4, c14);

            let b5 = _mm256_broadcast_ss(&*b.add(p * NR + 5));
            c05 = _mm256_fmadd_ps(a0, b5, c05);
            c15 = _mm256_fmadd_ps(a1, b5, c15);
        }

        let va = _mm256_set1_ps(alpha);
        macro_rules! store_col {
            ($col:expr, $lo:expr, $hi:expr) => {{
                let p = c.add($col * ldc);
                _mm256_storeu_ps(p, _mm256_fmadd_ps(va, $lo, _mm256_loadu_ps(p)));
                _mm256_storeu_ps(
                    p.add(8),
                    _mm256_fmadd_ps(va, $hi, _mm256_loadu_ps(p.add(8))),
                );
            }};
        }
        store_col!(0, c00, c10);
        store_col!(1, c01, c11);
        store_col!(2, c02, c12);
        store_col!(3, c03, c13);
        store_col!(4, c04, c14);
        store_col!(5, c05, c15);
    }
}

unsafe fn pack_a(mc: usize, kc: usize, a: *const f32, rs: usize, cs: usize, dst: *mut f32) {
    unsafe {
        let mut i0 = 0;
        while i0 < mc {
            let mr = MR.min(mc - i0);
            let strip = dst.add(i0 * kc);
            if rs == 1 {
                // column-major source, rows are contiguous
                for p in 0..kc {
                    let d = strip.add(p * MR);
                    std::ptr::copy_nonoverlapping(a.add(i0 + p * cs), d, mr);
                    for r in mr..MR {
                        *d.add(r) = 0.0;
                    }
                }
            } else {
                // transposed source, walk p innermost so reads are contiguous
                for r in 0..mr {
                    let src = a.add((i0 + r) * rs);
                    for p in 0..kc {
                        *strip.add(p * MR + r) = *src.add(p * cs);
                    }
                }
                for r in mr..MR {
                    for p in 0..kc {
                        *strip.add(p * MR + r) = 0.0;
                    }
                }
            }
            i0 += MR;
        }
    }
}

unsafe fn pack_b(kc: usize, nc: usize, b: *const f32, rs: usize, cs: usize, dst: *mut f32) {
    unsafe {
        let mut j0 = 0;
        while j0 < nc {
            let nr = NR.min(nc - j0);
            let strip = dst.add(j0 * kc);
            for p in 0..kc {
                let d = strip.add(p * NR);
                let src = b.add(p * rs + j0 * cs);
                if cs == 1 && nr == 6 {
                    _mm_storeu_ps(d, _mm_loadu_ps(src));
                    *d.add(4) = *src.add(4);
                    *d.add(5) = *src.add(5);
                } else {
                    for j in 0..nr {
                        *d.add(j) = *b.add(p * rs + (j0 + j) * cs);
                    }
                }

                for j in nr..NR {
                    *d.add(j) = 0.0;
                }
            }
            j0 += NR;
        }
    }
}

/// GEMM: `C += alpha * op(A) * op(B)`
///
/// A and B are described by (row stride and col stride) of op(X)
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_blocked(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    a_rs: usize,
    a_cs: usize,
    b: *const f32,
    b_rs: usize,
    b_cs: usize,
    c: *mut f32,
    ldc: usize,
) {
    // size packing buffers to what is actually needed (small matrices stay cheap)
    let n_blocks = n.div_ceil(GEMM_NC);
    let nc_step = n.div_ceil(n_blocks).div_ceil(NR) * NR;
    let mc_max = GEMM_MC.min(m).div_ceil(MR) * MR;
    let nc_max = nc_step;
    let kc_max = GEMM_KC.min(k);
    let mut a_packed = vec![0.0f32; mc_max * kc_max];
    let mut b_packed = vec![0.0f32; kc_max * nc_max];

    unsafe {
        let mut jc = 0;
        while jc < n {
            let nc = nc_step.min(n - jc);

            let mut pc = 0;
            while pc < k {
                let kc = GEMM_KC.min(k - pc);

                pack_b(
                    kc,
                    nc,
                    b.add(pc * b_rs + jc * b_cs),
                    b_rs,
                    b_cs,
                    b_packed.as_mut_ptr(),
                );

                let mut ic = 0;
                while ic < m {
                    let mc = GEMM_MC.min(m - ic);

                    pack_a(
                        mc,
                        kc,
                        a.add(ic * a_rs + pc * a_cs),
                        a_rs,
                        a_cs,
                        a_packed.as_mut_ptr(),
                    );

                    let mut j0 = 0;
                    while j0 < nc {
                        let nr = NR.min(nc - j0);
                        let bp = b_packed.as_ptr().add(j0 * kc);

                        let mut i0 = 0;
                        while i0 < mc {
                            let mr = MR.min(mc - i0);
                            let ap = a_packed.as_ptr().add(i0 * kc);
                            let cp = c.add((ic + i0) + (jc + j0) * ldc);

                            if mr == MR && nr == NR {
                                kernel_16x6(kc, alpha, ap, bp, cp, ldc);
                            } else {
                                // for edge tile, compute into a zeroed scratch tile, add the valid part
                                let mut tmp = [0.0f32; MR * NR];
                                kernel_16x6(kc, alpha, ap, bp, tmp.as_mut_ptr(), MR);
                                for j in 0..nr {
                                    for i in 0..mr {
                                        *cp.add(i + j * ldc) += tmp[i + j * MR];
                                    }
                                }
                            }
                            i0 += MR;
                        }
                        j0 += NR;
                    }

                    ic += GEMM_MC;
                }
                pc += GEMM_KC;
            }
            jc += nc_step;
        }
    }
}

use crate::GLOBAL_THREAD_POOL;
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
unsafe fn gemm_par(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    a_rs: usize,
    a_cs: usize,
    b: *const f32,
    b_rs: usize,
    b_cs: usize,
    c: *mut f32,
    ldc: usize,
) {
    let nt = GLOBAL_THREAD_POOL.current_num_threads();

    // too small (2*m*n*k flops), scheduling overhead would dominate
    if nt == 1 || 2 * m * n * k < (64 << 20) {
        unsafe { gemm_blocked(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, c, ldc) };
        return;
    }

    // split the larger of n / m; chunks are multiples of the micro-tile
    let split_n = n >= m;
    let (dim, unit) = if split_n { (n, NR) } else { (m, MR) };
    let chunk = dim.div_ceil(nt).div_ceil(unit) * unit;
    let nchunks = dim.div_ceil(chunk);

    // cast to usize to smuggle the raw pointers into the rayon task cuz `*mut f32` is not `Send`
    // SAFETY: A/B are read-only in every task and the C slabs are disjoint by construction of the 1D split
    let (ap, bp, cp) = (a as usize, b as usize, c as usize);

    GLOBAL_THREAD_POOL.install(|| {
        (0..nchunks).into_par_iter().for_each(|t| unsafe {
            let (a, b, c) = (ap as *const f32, bp as *const f32, cp as *mut f32);
            let start = t * chunk;
            let len = chunk.min(dim - start);
            if split_n {
                // columns [start, start+len) of C and op(B)
                gemm_blocked(
                    m,
                    len,
                    k,
                    alpha,
                    a,
                    a_rs,
                    a_cs,
                    b.add(start * b_cs),
                    b_rs,
                    b_cs,
                    c.add(start * ldc),
                    ldc,
                );
            } else {
                // rows [start, start+len) of C and op(A)
                gemm_blocked(
                    len,
                    n,
                    k,
                    alpha,
                    a.add(start * a_rs),
                    a_rs,
                    a_cs,
                    b,
                    b_rs,
                    b_cs,
                    c.add(start),
                    ldc,
                );
            }
        });
    });
}

/// C += alpha * A * B
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_nn(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    unsafe { gemm_par(m, n, k, alpha, a, 1, lda, b, 1, ldb, c, ldc) }
}

/// C += alpha * A * B^T
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    unsafe { gemm_par(m, n, k, alpha, a, 1, lda, b, ldb, 1, c, ldc) }
}

/// C += alpha * A^T * B
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tn(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    unsafe { gemm_par(m, n, k, alpha, a, lda, 1, b, 1, ldb, c, ldc) }
}

/// C += alpha * A^T * B^T
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    unsafe { gemm_par(m, n, k, alpha, a, lda, 1, b, ldb, 1, c, ldc) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
/// The gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemm.html)
pub fn gemm(
    m: usize, // rows of C and op(A)
    n: usize, // cols of C and op(B)
    k: usize, // inner dimension
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
    is_trans_a: bool,
    is_trans_b: bool,
) {
    if m == 0 || n == 0 {
        return;
    }

    // dimensions of A and B as "stored"
    let a_rows = if is_trans_a { k } else { m };
    let a_cols = if is_trans_a { m } else { k };
    let b_rows = if is_trans_b { n } else { k };
    let b_cols = if is_trans_b { k } else { n };

    // required slice length for a column-major rows x cols matrix with leading dim ld
    let need = |rows: usize, cols: usize, ld: usize| {
        if rows == 0 || cols == 0 {
            0
        } else {
            (cols - 1) * ld + rows
        }
    };

    if lda == 0 || lda < a_rows {
        panic!("lda must be >= rows of stored A and non-zero");
    }
    if ldb == 0 || ldb < b_rows {
        panic!("ldb must be >= rows of stored B and non-zero");
    }
    if ldc == 0 || ldc < m {
        panic!("ldc must be >= rows of C and non-zero");
    }
    if a.len() < need(a_rows, a_cols, lda) {
        panic!("Matrix A buffer too small");
    }
    if b.len() < need(b_rows, b_cols, ldb) {
        panic!("Matrix B buffer too small");
    }
    if c.len() < need(m, n, ldc) {
        panic!("Matrix C buffer too small");
    }

    // C = beta * C  (beta == 0 overwrites, so NaN/Inf already in C do not leak through)
    if beta == 0.0 {
        for j in 0..n {
            c[j * ldc..j * ldc + m].fill(0.0);
        }
    } else if beta != 1.0 {
        for j in 0..n {
            scal(m, beta, &mut c[j * ldc..j * ldc + m], 1);
        }
    }

    // nothing to accumulate (k == 0 still applied beta above)
    if k == 0 || alpha == 0.0 {
        return;
    }

    // if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
    //     gemm_fallback(
    //         m, n, k, alpha, a, lda, b, ldb, c, ldc, is_trans_a, is_trans_b,
    //     );
    //     return;
    // }

    let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    unsafe {
        match (is_trans_a, is_trans_b) {
            (false, false) => gemm_nn(m, n, k, alpha, ap, lda, bp, ldb, cp, ldc),
            (false, true) => gemm_nt(m, n, k, alpha, ap, lda, bp, ldb, cp, ldc),
            (true, false) => gemm_tn(m, n, k, alpha, ap, lda, bp, ldb, cp, ldc),
            (true, true) => gemm_tt(m, n, k, alpha, ap, lda, bp, ldb, cp, ldc),
        }
    }
}

/// Plain scalar GEMM with no SIMD
#[allow(unused)]
#[allow(clippy::too_many_arguments)]
fn gemm_fallback(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    ta: bool,
    tb: bool,
) {
    for j in 0..n {
        for p in 0..k {
            let bv = if tb { b[j + p * ldb] } else { b[p + j * ldb] };
            for i in 0..m {
                let av = if ta { a[p + i * lda] } else { a[i + p * lda] };
                c[i + j * ldc] += alpha * av * bv;
            }
        }
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn gemm_regress() {
    use crate::utils::Noise;
    use std::hint::black_box;
    use std::time::Instant;

    let warmup = 16;
    let runs = 24;
    let size = 1024;

    println!("size: {}", size);

    let mut noise = Noise::rng();
    let mut a = vec![0.0f32; size * size];
    let mut b = vec![0.0f32; size * size];
    let mut c = vec![0.0f32; size * size];

    black_box(&mut a);
    black_box(&mut b);
    black_box(&mut c);

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut b);
    c.fill(1.0);

    for _ in 0..warmup {
        c.fill(1.0);
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, false, false,
        );
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, true, true,
        );
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, false, true,
        );
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, true, false,
        );
    }

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut b);
    c.fill(1.0);

    let start = Instant::now();
    for _ in 0..runs {
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, false, false,
        );
    }
    let dur = start.elapsed();

    let total_flops = 2.0 * (size as f64).powi(3) * runs as f64;
    let gflops = total_flops / dur.as_secs_f64() / 1e9;

    if cfg!(not(debug_assertions)) {
        assert!(
            gflops > 108.6,
            "gemm_n_n perf regress: {:.2} GFLOPS",
            gflops
        );
    }

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut b);
    c.fill(1.0);

    let start = Instant::now();
    for _ in 0..runs {
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, false, true,
        );
    }
    let dur = start.elapsed();

    let total_flops = 2.0 * (size as f64).powi(3) * runs as f64;
    let gflops = total_flops / dur.as_secs_f64() / 1e9;

    if cfg!(not(debug_assertions)) {
        assert!(
            gflops > 108.6,
            "gemm_n_t perf regress: {:.2} GFLOPS",
            gflops
        );
    }

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut b);
    c.fill(1.0);

    let start = Instant::now();
    for _ in 0..runs {
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, true, true,
        );
    }
    let dur = start.elapsed();

    let total_flops = 2.0 * (size as f64).powi(3) * runs as f64;
    let gflops = total_flops / dur.as_secs_f64() / 1e9;

    if cfg!(not(debug_assertions)) {
        assert!(
            gflops > 108.6,
            "gemm_t_t perf regress: {:.2} GFLOPS",
            gflops
        );
    }

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut b);
    c.fill(1.0);

    let start = Instant::now();
    for _ in 0..runs {
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, true, false,
        );
    }
    let dur = start.elapsed();

    let total_flops = 2.0 * (size as f64).powi(3) * runs as f64;
    let gflops = total_flops / dur.as_secs_f64() / 1e9;

    if cfg!(not(debug_assertions)) {
        assert!(
            gflops > 108.6,
            "gemm_t_n perf regress: {:.2} GFLOPS",
            gflops
        );
    }
}

/// Parallel GEMM must be bit-identical to the serial path
#[test]
fn gemm_parallel_matches_serial() {
    use crate::utils::Noise;

    // 2*420*380*415 ≈ 132M flops, past the parallel threshold
    // sizes are non-multiples of MR/NR/GEMM_KC to hit every edge-tile path
    // with padded leading dims to hit the strided pack paths
    let (m, n, k) = (420, 380, 415);
    let (lda, ldb, ldc) = (m + 9, k + 7, m + 5);

    let mut noise = Noise::rng();
    let mut c_init = vec![0.0f32; ldc * n];
    noise.fill_f32(&mut c_init);

    for ta in [false, true] {
        for tb in [false, true] {
            let a_rows = if ta { k } else { m };
            let a_cols = if ta { m } else { k };
            let b_rows = if tb { n } else { k };
            let b_cols = if tb { k } else { n };

            let mut a = vec![0.0f32; (a_cols - 1) * lda + a_rows];
            noise.fill_f32(&mut a);
            let mut b = vec![0.0f32; (b_cols - 1) * ldb + b_rows];
            noise.fill_f32(&mut b);

            let (a_rs, a_cs) = if ta { (lda, 1) } else { (1, lda) };
            let (b_rs, b_cs) = if tb { (ldb, 1) } else { (1, ldb) };

            for beta in [0.0f32, 2.5, 1.0] {
                // parallel path, through the public entry point
                let mut c_par = c_init.clone();
                gemm(
                    m, n, k, 1.5, &a, lda, &b, ldb, beta, &mut c_par, ldc, ta, tb,
                );

                // serial reference with the same beta handling
                let mut c_ref = c_init.clone();
                if beta != 1.0 {
                    for j in 0..n {
                        if beta == 0.0 {
                            c_ref[j * ldc..j * ldc + m].fill(0.0);
                        } else {
                            scal(m, beta, &mut c_ref[j * ldc..j * ldc + m], 1);
                        }
                    }
                }
                unsafe {
                    gemm_blocked(
                        m,
                        n,
                        k,
                        1.5,
                        a.as_ptr(),
                        a_rs,
                        a_cs,
                        b.as_ptr(),
                        b_rs,
                        b_cs,
                        c_ref.as_mut_ptr(),
                        ldc,
                    );
                }

                for j in 0..n {
                    for i in 0..m {
                        let idx = i + j * ldc;
                        assert_eq!(
                            c_par[idx].to_bits(),
                            c_ref[idx].to_bits(),
                            "ta={ta} tb={tb} beta={beta} c[{i},{j}]: parallel={}, serial={}",
                            c_par[idx],
                            c_ref[idx],
                        );
                    }
                }
            }
        }
    }
}
