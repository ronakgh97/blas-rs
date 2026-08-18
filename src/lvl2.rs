//! Implementation of Level 2 BLAS routines
use crate::lvl1::scal;
use crate::reduce_add;
#[allow(unused)]
use std::arch::x86_64::{
    _MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _mm_prefetch, _mm256_add_ps, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// The gemv routines compute a scalar-matrix-vector product and add the result to a scalar-vector product, with a general matrix.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemv.html)
pub fn gemv(
    m: usize,       // rows of mat
    n: usize,       // cols of mat
    alpha: f32,     // scaling for product
    a: &[f32],      // input matrix buf
    lda: usize,     // leading dim of `a`, row or col major depends
    x: &[f32],      // mul vector buf
    incx: i32,      // strided access for x
    beta: f32,      // y scaling
    y: &mut [f32],  // resultant buf
    incy: i32,      // strided access for y
    is_trans: bool, // whether to treat `a` as transposed
) {
    let (x_len, y_len) = if is_trans { (m, n) } else { (n, m) };

    if incx == 0 || incy == 0 {
        panic!("incx and incy must be non-zero");
    }
    if lda == 0 || lda < m {
        panic!("lda must be >= m and non-zero");
    }
    if m == 0 || n == 0 {
        panic!("Matrix dimensions must be greater than zero");
    }

    // `(n - 1) * lda` is start of last col, since we are col major,
    // so we added m to get the last element of that col
    if a.len() < (n - 1) * lda + m {
        panic!("Matrix A is too short for the given dimensions and leading dimension");
    }

    // check inner dim
    if (x.len() < (1 + (x_len - 1) * incx.unsigned_abs() as usize))
        || (y.len() < (1 + (y_len - 1) * incy.unsigned_abs() as usize))
    {
        panic!("Vector x is too short for the given dimensions, increment and transposition");
    }

    if beta != 1.0 {
        scal(y_len, beta, y, incy);
    }
    // ret if 0
    if alpha == 0.0 {
        return;
    }

    let incx_isize = incx as isize;
    let incy_isize = incy as isize;

    const PREFETCH_DIST: usize = 128;

    // NOTE; bench insights TODO;

    // non-transposed case; axpy each col of A scaled by x[j] into y; prefetch next col of A
    // transposed case; dot product each col of A with x, scale by alpha, add to y[j]; prefetch next col of A
    if !is_trans {
        let ix_b = if incx < 0 {
            (1 - n as isize) * incx_isize
        } else {
            0
        };
        let iy_b = if incy < 0 {
            (1 - m as isize) * incy_isize
        } else {
            0
        };

        let a_ptr = a.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        // simd path, load columns of A and x, do axpy, store to y; full rizz
        if incx == 1 && incy == 1 {
            unsafe {
                // 6 col unrolling
                let mut j = 0usize;
                while j + 5 < n {
                    let alpha_x0 = alpha * *x_ptr.add(j);
                    let alpha_x1 = alpha * *x_ptr.add(j + 1);
                    let alpha_x2 = alpha * *x_ptr.add(j + 2);
                    let alpha_x3 = alpha * *x_ptr.add(j + 3);
                    let alpha_x4 = alpha * *x_ptr.add(j + 4);
                    let alpha_x5 = alpha * *x_ptr.add(j + 5);

                    // load x alpha's
                    let ax0 = _mm256_set1_ps(alpha_x0);
                    let ax1 = _mm256_set1_ps(alpha_x1);
                    let ax2 = _mm256_set1_ps(alpha_x2);
                    let ax3 = _mm256_set1_ps(alpha_x3);
                    let ax4 = _mm256_set1_ps(alpha_x4);
                    let ax5 = _mm256_set1_ps(alpha_x5);

                    let col0 = a_ptr.add(j * lda);
                    let col1 = a_ptr.add((j + 1) * lda);
                    let col2 = a_ptr.add((j + 2) * lda);
                    let col3 = a_ptr.add((j + 3) * lda);
                    let col4 = a_ptr.add((j + 4) * lda);
                    let col5 = a_ptr.add((j + 5) * lda);

                    // _mm_prefetch(a_ptr.add((j + 6) * lda) as *const i8, _MM_HINT_T0);
                    // _mm_prefetch(a_ptr.add((j + 12) * lda) as *const i8, _MM_HINT_T1);

                    // process 32 rows (load 4 reg for y and ~6 reg for a's col and 4 for store)
                    let mut i = 0usize;
                    while i + 32 <= m {
                        // load y write buf once
                        let mut y0 = _mm256_loadu_ps(y_ptr.add(i));
                        let mut y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                        let mut y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                        let mut y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                        // load 6 col of A and do 6 fma for each col;
                        // we are doing a lot of work with minimal loads/stores,
                        // just praying for cpu-chan to schedule well

                        _mm_prefetch(col0.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(col1.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(col2.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(col3.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(col4.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(col5.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);

                        // col 1
                        let c10 = _mm256_loadu_ps(col0.add(i));
                        let c11 = _mm256_loadu_ps(col0.add(i + 8));
                        let c12 = _mm256_loadu_ps(col0.add(i + 16));
                        let c13 = _mm256_loadu_ps(col0.add(i + 24));
                        y0 = _mm256_fmadd_ps(ax0, c10, y0);
                        y1 = _mm256_fmadd_ps(ax0, c11, y1);
                        y2 = _mm256_fmadd_ps(ax0, c12, y2);
                        y3 = _mm256_fmadd_ps(ax0, c13, y3);

                        // col 2
                        let c20 = _mm256_loadu_ps(col1.add(i));
                        let c21 = _mm256_loadu_ps(col1.add(i + 8));
                        let c22 = _mm256_loadu_ps(col1.add(i + 16));
                        let c23 = _mm256_loadu_ps(col1.add(i + 24));
                        y0 = _mm256_fmadd_ps(ax1, c20, y0);
                        y1 = _mm256_fmadd_ps(ax1, c21, y1);
                        y2 = _mm256_fmadd_ps(ax1, c22, y2);
                        y3 = _mm256_fmadd_ps(ax1, c23, y3);

                        // col 3
                        let c30 = _mm256_loadu_ps(col2.add(i));
                        let c31 = _mm256_loadu_ps(col2.add(i + 8));
                        let c32 = _mm256_loadu_ps(col2.add(i + 16));
                        let c33 = _mm256_loadu_ps(col2.add(i + 24));
                        y0 = _mm256_fmadd_ps(ax2, c30, y0);
                        y1 = _mm256_fmadd_ps(ax2, c31, y1);
                        y2 = _mm256_fmadd_ps(ax2, c32, y2);
                        y3 = _mm256_fmadd_ps(ax2, c33, y3);

                        // col 4
                        let c40 = _mm256_loadu_ps(col3.add(i));
                        let c41 = _mm256_loadu_ps(col3.add(i + 8));
                        let c42 = _mm256_loadu_ps(col3.add(i + 16));
                        let c43 = _mm256_loadu_ps(col3.add(i + 24));
                        y0 = _mm256_fmadd_ps(ax3, c40, y0);
                        y1 = _mm256_fmadd_ps(ax3, c41, y1);
                        y2 = _mm256_fmadd_ps(ax3, c42, y2);
                        y3 = _mm256_fmadd_ps(ax3, c43, y3);

                        // col 5
                        let c50 = _mm256_loadu_ps(col4.add(i));
                        let c51 = _mm256_loadu_ps(col4.add(i + 8));
                        let c52 = _mm256_loadu_ps(col4.add(i + 16));
                        let c53 = _mm256_loadu_ps(col4.add(i + 24));
                        y0 = _mm256_fmadd_ps(ax4, c50, y0);
                        y1 = _mm256_fmadd_ps(ax4, c51, y1);
                        y2 = _mm256_fmadd_ps(ax4, c52, y2);
                        y3 = _mm256_fmadd_ps(ax4, c53, y3);

                        // col 6
                        let c60 = _mm256_loadu_ps(col5.add(i));
                        let c61 = _mm256_loadu_ps(col5.add(i + 8));
                        let c62 = _mm256_loadu_ps(col5.add(i + 16));
                        let c63 = _mm256_loadu_ps(col5.add(i + 24));
                        y0 = _mm256_fmadd_ps(ax5, c60, y0);
                        y1 = _mm256_fmadd_ps(ax5, c61, y1);
                        y2 = _mm256_fmadd_ps(ax5, c62, y2);
                        y3 = _mm256_fmadd_ps(ax5, c63, y3);

                        // write back once per 6 col
                        // TODO; store/compute/load all iteration is SLOW HEAVY!!!,
                        //  we can load 6 col and just pray for cpu
                        _mm256_storeu_ps(y_ptr.add(i), y0);
                        _mm256_storeu_ps(y_ptr.add(i + 8), y1);
                        _mm256_storeu_ps(y_ptr.add(i + 16), y2);
                        _mm256_storeu_ps(y_ptr.add(i + 24), y3);
                        i += 32;
                    }

                    // squeeze out everything (1 y load per 6 col)
                    while i + 8 <= m {
                        let mut y = _mm256_loadu_ps(y_ptr.add(i));
                        y = _mm256_fmadd_ps(ax0, _mm256_loadu_ps(col0.add(i)), y);
                        y = _mm256_fmadd_ps(ax1, _mm256_loadu_ps(col1.add(i)), y);
                        y = _mm256_fmadd_ps(ax2, _mm256_loadu_ps(col2.add(i)), y);
                        y = _mm256_fmadd_ps(ax3, _mm256_loadu_ps(col3.add(i)), y);
                        y = _mm256_fmadd_ps(ax4, _mm256_loadu_ps(col4.add(i)), y);
                        y = _mm256_fmadd_ps(ax5, _mm256_loadu_ps(col5.add(i)), y);
                        _mm256_storeu_ps(y_ptr.add(i), y);
                        i += 8;
                    }

                    // scalar fallback
                    while i < m {
                        let mut v = *y_ptr.add(i);
                        v = alpha_x0.mul_add(*col0.add(i), v);
                        v = alpha_x1.mul_add(*col1.add(i), v);
                        v = alpha_x2.mul_add(*col2.add(i), v);
                        v = alpha_x3.mul_add(*col3.add(i), v);
                        v = alpha_x4.mul_add(*col4.add(i), v);
                        v = alpha_x5.mul_add(*col5.add(i), v);
                        *y_ptr.add(i) = v;
                        i += 1;
                    }

                    j += 6; // step by 6 col
                }

                // process 1 col if n is odd/not divisible by 6
                while j < n {
                    let alpha_x = alpha * *x_ptr.add(j);
                    let bx = _mm256_set1_ps(alpha_x);
                    let col = a_ptr.add(j * lda);

                    // load 8 reg for A's col and y, do fmadd, store to y;
                    let mut i = 0usize;
                    while i + 32 <= m {
                        let a0 = _mm256_loadu_ps(col.add(i));
                        let a1 = _mm256_loadu_ps(col.add(i + 8));
                        let a2 = _mm256_loadu_ps(col.add(i + 16));
                        let a3 = _mm256_loadu_ps(col.add(i + 24));

                        let y0 = _mm256_loadu_ps(y_ptr.add(i));
                        let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                        let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                        let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                        _mm256_storeu_ps(y_ptr.add(i), _mm256_fmadd_ps(bx, a0, y0));
                        _mm256_storeu_ps(y_ptr.add(i + 8), _mm256_fmadd_ps(bx, a1, y1));
                        _mm256_storeu_ps(y_ptr.add(i + 16), _mm256_fmadd_ps(bx, a2, y2));
                        _mm256_storeu_ps(y_ptr.add(i + 24), _mm256_fmadd_ps(bx, a3, y3));
                        i += 32;
                    }

                    // any leftovers
                    while i + 8 <= m {
                        let a = _mm256_loadu_ps(col.add(i));
                        let y = _mm256_loadu_ps(y_ptr.add(i));
                        _mm256_storeu_ps(y_ptr.add(i), _mm256_fmadd_ps(bx, a, y));
                        i += 8;
                    }

                    // scalar fallback
                    while i < m {
                        *y_ptr.add(i) = alpha_x.mul_add(*col.add(i), *y_ptr.add(i));
                        i += 1;
                    }
                    j += 1
                }
            }
            // if incx!=1 but incy is, we can still do the axpy with simd,
            // just load x with stride and prefetching, cpu rizz but less rizz
        } else if incy == 1 {
            unsafe {
                for j in 0..n {
                    // load x (strided)
                    let alpha_x = alpha * *x_ptr.offset(ix_b + j as isize * incx_isize);
                    let bx = _mm256_set1_ps(alpha_x);
                    let col = a_ptr.add(j * lda);

                    _mm_prefetch(a_ptr.add((j + 1) * lda) as *const i8, _MM_HINT_NTA);
                    _mm_prefetch(a_ptr.add((j + 8) * lda) as *const i8, _MM_HINT_NTA);

                    let mut i = 0usize;
                    while i + 32 <= m {
                        // load (8x4) block of A and y, do axpy, store to y; this is the rizz part
                        let a0 = _mm256_loadu_ps(col.add(i));
                        let a1 = _mm256_loadu_ps(col.add(i + 8));
                        let a2 = _mm256_loadu_ps(col.add(i + 16));
                        let a3 = _mm256_loadu_ps(col.add(i + 24));

                        let y0 = _mm256_loadu_ps(y_ptr.add(i));
                        let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                        let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                        let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                        // write back
                        _mm256_storeu_ps(y_ptr.add(i), _mm256_fmadd_ps(bx, a0, y0));
                        _mm256_storeu_ps(y_ptr.add(i + 8), _mm256_fmadd_ps(bx, a1, y1));
                        _mm256_storeu_ps(y_ptr.add(i + 16), _mm256_fmadd_ps(bx, a2, y2));
                        _mm256_storeu_ps(y_ptr.add(i + 24), _mm256_fmadd_ps(bx, a3, y3));
                        i += 32;
                    }

                    // if left any
                    while i + 8 <= m {
                        let a = _mm256_loadu_ps(col.add(i));
                        let y = _mm256_loadu_ps(y_ptr.add(i));
                        _mm256_storeu_ps(y_ptr.add(i), _mm256_fmadd_ps(bx, a, y));
                        i += 8;
                    }

                    // scalar
                    while i < m {
                        *y_ptr.add(i) = alpha_x.mul_add(*col.add(i), *y_ptr.add(i));
                        i += 1;
                    }
                }
            }
            // non-simd path, just do the axpy with prefetching, no rizz
        } else {
            unsafe {
                for j in 0..n {
                    // get strided x
                    let x_val = alpha * *x_ptr.offset(ix_b + j as isize * incx_isize);
                    let col = a_ptr.add(j * lda);

                    _mm_prefetch(a_ptr.add((j + 1) * lda) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_ptr.add((j + 8) * lda) as *const i8, _MM_HINT_T0);

                    // strided axpy into y
                    let mut iy = iy_b;
                    for i in 0..m {
                        *y_ptr.offset(iy) = x_val.mul_add(*col.add(i), *y_ptr.offset(iy));
                        iy += incy_isize;
                    }
                }
            }
        }
        // in_trans case;
    } else {
        let ix_b = if incx < 0 {
            (1 - m as isize) * incx_isize
        } else {
            0
        };
        let iy_b = if incy < 0 {
            (1 - n as isize) * incy_isize
        } else {
            0
        };

        let a_ptr = a.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        // simd path, load columns of A and x, do dot product, store to y;
        // prefetch/pray future columns of A for rizzing cpu-chan

        if incx == 1 && incy == 1 {
            unsafe {
                let mut j = 0usize;

                // process 6 columns to maximize X vector reuse
                while j + 5 < n {
                    let col0 = a_ptr.add(j * lda);
                    let col1 = a_ptr.add((j + 1) * lda);
                    let col2 = a_ptr.add((j + 2) * lda);
                    let col3 = a_ptr.add((j + 3) * lda);
                    let col4 = a_ptr.add((j + 4) * lda);
                    let col5 = a_ptr.add((j + 5) * lda);

                    // for prefetch
                    // let col6: *const f32 = a_ptr.add((j + 6) * lda);
                    // let col7 = a_ptr.add((j + 7) * lda);

                    _mm_prefetch(a_ptr.add((j + 6) * lda) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_ptr.add((j + 7) * lda) as *const i8, _MM_HINT_T0);

                    // set 2 acc from each column
                    let mut sum0_0 = _mm256_setzero_ps();
                    let mut sum0_1 = _mm256_setzero_ps();

                    let mut sum1_0 = _mm256_setzero_ps();
                    let mut sum1_1 = _mm256_setzero_ps();

                    let mut sum2_0 = _mm256_setzero_ps();
                    let mut sum2_1 = _mm256_setzero_ps();

                    let mut sum3_0 = _mm256_setzero_ps();
                    let mut sum3_1 = _mm256_setzero_ps();

                    let mut sum4_0 = _mm256_setzero_ps();
                    let mut sum4_1 = _mm256_setzero_ps();

                    let mut sum5_0 = _mm256_setzero_ps();
                    let mut sum5_1 = _mm256_setzero_ps();

                    let mut i = 0usize;

                    // TODO; 6 col unroling will stack pill?
                    while i + 32 <= m {
                        _mm_prefetch(col0.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(col2.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(col4.add(i + PREFETCH_DIST) as *const i8, _MM_HINT_T0);

                        // load x's once (32 floats)
                        let x0 = _mm256_loadu_ps(x_ptr.add(i));
                        let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                        let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                        let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                        // compute fma for col 0
                        sum0_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i)), x0, sum0_0);
                        sum0_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i + 8)), x1, sum0_1);
                        sum0_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i + 16)), x2, sum0_0);
                        sum0_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i + 24)), x3, sum0_1);

                        // compute fma for col 1
                        sum1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i)), x0, sum1_0);
                        sum1_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i + 8)), x1, sum1_1);
                        sum1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i + 16)), x2, sum1_0);
                        sum1_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i + 24)), x3, sum1_1);

                        //compute fma for col2
                        sum2_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col2.add(i)), x0, sum2_0);
                        sum2_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col2.add(i + 8)), x1, sum2_1);
                        sum2_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col2.add(i + 16)), x2, sum2_0);
                        sum2_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col2.add(i + 24)), x3, sum2_1);

                        //compute fma for col 3
                        sum3_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col3.add(i)), x0, sum3_0);
                        sum3_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col3.add(i + 8)), x1, sum3_1);
                        sum3_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col3.add(i + 16)), x2, sum3_0);
                        sum3_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col3.add(i + 24)), x3, sum3_1);

                        //compute fma for col 4
                        sum4_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col4.add(i)), x0, sum4_0);
                        sum4_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col4.add(i + 8)), x1, sum4_1);
                        sum4_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col4.add(i + 16)), x2, sum4_0);
                        sum4_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col4.add(i + 24)), x3, sum4_1);

                        //compute fma for col 4
                        sum5_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col5.add(i)), x0, sum5_0);
                        sum5_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col5.add(i + 8)), x1, sum5_1);
                        sum5_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col5.add(i + 16)), x2, sum5_0);
                        sum5_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col5.add(i + 24)), x3, sum5_1);
                        i += 32;
                    }

                    // if 8 rows remaining
                    while i + 8 <= m {
                        let x = _mm256_loadu_ps(x_ptr.add(i));
                        sum0_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i)), x, sum0_0);
                        sum1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i)), x, sum1_0);
                        sum2_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col2.add(i)), x, sum2_0);
                        sum3_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col3.add(i)), x, sum3_0);
                        sum4_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col4.add(i)), x, sum4_0);
                        sum5_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col5.add(i)), x, sum5_0);
                        i += 8;
                    }

                    // reduction of acc
                    let mut dot0 = reduce_add!(_mm256_add_ps(sum0_0, sum0_1));
                    let mut dot1 = reduce_add!(_mm256_add_ps(sum1_0, sum1_1));
                    let mut dot2 = reduce_add!(_mm256_add_ps(sum2_0, sum2_1));
                    let mut dot3 = reduce_add!(_mm256_add_ps(sum3_0, sum3_1));
                    let mut dot4 = reduce_add!(_mm256_add_ps(sum4_0, sum4_1));
                    let mut dot5 = reduce_add!(_mm256_add_ps(sum5_0, sum5_1));

                    // scalar fallback
                    while i < m {
                        let x_val = *x_ptr.add(i);
                        dot0 = x_val.mul_add(*col0.add(i), dot0);
                        dot1 = x_val.mul_add(*col1.add(i), dot1);
                        dot2 = x_val.mul_add(*col2.add(i), dot2);
                        dot3 = x_val.mul_add(*col3.add(i), dot3);
                        dot4 = x_val.mul_add(*col4.add(i), dot4);
                        dot5 = x_val.mul_add(*col5.add(i), dot5);
                        i += 1;
                    }

                    // write back with alpha scaling (contiguous)
                    *y_ptr.add(j) = alpha.mul_add(dot0, *y_ptr.add(j));
                    *y_ptr.add(j + 1) = alpha.mul_add(dot1, *y_ptr.add(j + 1));
                    *y_ptr.add(j + 2) = alpha.mul_add(dot2, *y_ptr.add(j + 2));
                    *y_ptr.add(j + 3) = alpha.mul_add(dot3, *y_ptr.add(j + 3));
                    *y_ptr.add(j + 4) = alpha.mul_add(dot4, *y_ptr.add(j + 4));
                    *y_ptr.add(j + 5) = alpha.mul_add(dot5, *y_ptr.add(j + 5));
                    j += 6;
                }

                // if n is not divisible by 6
                // process remaining columns; no reuse but still simd dot product
                while j < n {
                    let col = a_ptr.add(j * lda);
                    let mut sum0 = _mm256_setzero_ps();
                    let mut sum1 = _mm256_setzero_ps();
                    let mut sum2 = _mm256_setzero_ps();
                    let mut sum3 = _mm256_setzero_ps();

                    let mut i = 0usize;
                    // do four cols of 8 (4x8=32) elements at a time, then reduce to scalar
                    while i + 32 <= m {
                        sum0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(col.add(i)),
                            _mm256_loadu_ps(x_ptr.add(i)),
                            sum0,
                        );
                        sum1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(col.add(i + 8)),
                            _mm256_loadu_ps(x_ptr.add(i + 8)),
                            sum1,
                        );
                        sum2 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(col.add(i + 16)),
                            _mm256_loadu_ps(x_ptr.add(i + 16)),
                            sum2,
                        );
                        sum3 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(col.add(i + 24)),
                            _mm256_loadu_ps(x_ptr.add(i + 24)),
                            sum3,
                        );
                        i += 32;
                    }

                    while i + 8 <= m {
                        sum0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(col.add(i)),
                            _mm256_loadu_ps(x_ptr.add(i)),
                            sum0,
                        );
                        i += 8;
                    }

                    let dot0 = _mm256_add_ps(sum0, sum1);
                    let dot1 = _mm256_add_ps(sum2, sum3);
                    let mut dot = reduce_add!(_mm256_add_ps(dot0, dot1));
                    while i < m {
                        dot = (*col.add(i)).mul_add(*x_ptr.add(i), dot);
                        i += 1;
                    }

                    *y_ptr.add(j) = alpha.mul_add(dot, *y_ptr.add(j));
                    j += 1;
                }
            }
        } else if incx == 1 {
            unsafe {
                for j in 0..n {
                    let col = a_ptr.add(j * lda);

                    _mm_prefetch(a_ptr.add((j + 1) * lda) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_ptr.add((j + 8) * lda) as *const i8, _MM_HINT_T0);

                    // four accumulators for the dot product,
                    // we will reduce them at the end
                    let mut sum0 = _mm256_setzero_ps();
                    let mut sum1 = _mm256_setzero_ps();
                    let mut sum2 = _mm256_setzero_ps();
                    let mut sum3 = _mm256_setzero_ps();

                    // load (8x4) block of A and x, do fmadd, accumulate into sums;
                    // this is the rizz part
                    let mut i = 0usize;
                    while i + 32 <= m {
                        let a0 = _mm256_loadu_ps(col.add(i));
                        let a1 = _mm256_loadu_ps(col.add(i + 8));
                        let a2 = _mm256_loadu_ps(col.add(i + 16));
                        let a3 = _mm256_loadu_ps(col.add(i + 24));

                        let x0 = _mm256_loadu_ps(x_ptr.add(i));
                        let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                        let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                        let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                        sum0 = _mm256_fmadd_ps(a0, x0, sum0);
                        sum1 = _mm256_fmadd_ps(a1, x1, sum1);
                        sum2 = _mm256_fmadd_ps(a2, x2, sum2);
                        sum3 = _mm256_fmadd_ps(a3, x3, sum3);
                        i += 32;
                    }

                    // leftovers
                    while i + 8 <= m {
                        let a = _mm256_loadu_ps(col.add(i));
                        let x = _mm256_loadu_ps(x_ptr.add(i));
                        sum0 = _mm256_fmadd_ps(a, x, sum0);
                        i += 8;
                    }

                    // reduce to scalar
                    let sum01 = _mm256_add_ps(sum0, sum1);
                    let sum23 = _mm256_add_ps(sum2, sum3);
                    let mut dot_val = reduce_add!(_mm256_add_ps(sum01, sum23));

                    // any remaining
                    while i < m {
                        dot_val += *col.add(i) * *x_ptr.add(i);
                        i += 1;
                    }

                    // write back to y with alpha scaling, strided if needed
                    if incy == 1 {
                        *y_ptr.add(j) = alpha.mul_add(dot_val, *y_ptr.add(j));
                    } else {
                        let iy = iy_b + j as isize * incy_isize;
                        *y_ptr.offset(iy) = alpha.mul_add(dot_val, *y_ptr.offset(iy));
                    }
                }
            }
            // non-simd path, just do the dot product with prefetching, no rizz
        } else {
            unsafe {
                for j in 0..n {
                    let col = a_ptr.add(j * lda);

                    _mm_prefetch(a_ptr.add((j + 1) * lda) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_ptr.add((j + 8) * lda) as *const i8, _MM_HINT_T0);

                    // compute dot
                    let mut ix = ix_b;
                    let mut dot_val = 0.0f32;
                    for i in 0..m {
                        dot_val = (*col.add(i)).mul_add(*x_ptr.offset(ix), dot_val);
                        ix += incx_isize;
                    }

                    // update y strided with alpha*dot_val + y
                    let iy = iy_b + j as isize * incy_isize;
                    *y_ptr.offset(iy) = alpha.mul_add(dot_val, *y_ptr.offset(iy));
                }
            }
        }
    }
}

#[test]
fn gemv_test() {
    use crate::utils::Noise;
    use std::hint::black_box;
    use std::time::Instant;

    let warmup_count = 64;
    let run_count = 256;
    let size = 1256;

    println!("size: {}", size);

    let mut noise = Noise::rng();
    let mut a = vec![1.0f32; size * size];
    let mut x = vec![1.0f32; size];
    let mut y = vec![0.0f32; size];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut x);
    y.fill(1.0);

    for _ in 0..warmup_count {
        y.fill(1.0);
        gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y, 1, false);
        gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y, 1, true);
    }

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut x);
    y.fill(1.0);

    let start = Instant::now();
    for _ in 0..run_count {
        gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y, 1, false);
    }
    let dur = start.elapsed();

    let total_flops = 2.0 * (size.pow(2) as f64) * (run_count as f64);
    let gflops = total_flops / dur.as_secs_f64() / 1e9;

    println!(
        "gemv: {:?} seconds, {:.2} GFLOPS",
        dur.as_secs_f64(),
        gflops
    );

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut x);
    y.fill(1.0);

    for _ in 0..warmup_count {
        y.fill(1.0);
        gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y, 1, true);
    }

    y.fill(1.0);

    let start = Instant::now();
    for _ in 0..run_count {
        gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y, 1, true);
    }
    let dur = start.elapsed();

    let total_flops = 2.0 * (size.pow(2) as f64) * (run_count as f64);
    let gflops = total_flops / dur.as_secs_f64() / 1e9;

    println!(
        "gemv_t: {:?} seconds, {:.2} GFLOPS",
        dur.as_secs_f64(),
        gflops
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// The symv routines compute a scalar-matrix-vector product and add the result to a scalar-vector product, with a symmetric matrix.
/// [ref](http://intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/symv.html)
pub fn symv(
    n: usize,   // col,rows of mat
    alpha: f32, // scaling for product
    a: &[f32],  // input matrix buf
    lda: usize, // leading dim of a, row or col major depends, but we follow `column major`
    x: &[f32],  // mul vector buf
    incx: i32,
    beta: f32,     // y scaling
    y: &mut [f32], // resultant buf
    incy: i32,
    uplo: bool, // `true` for upper, `false` for lower
) {
    if incx == 0 || incy == 0 {
        panic!("incx and incy must be non-zero");
    }
    if lda == 0 || lda < n {
        panic!("lda must be >= m and non-zero");
    }

    if n == 0 {
        panic!("Matrix dimensions must be greater than zero");
    }

    // `(n - 1) * lda` is start of last col, since we are col major,
    // so we added n to get the last element of that col
    if a.len() < (n - 1) * lda + n {
        panic!("Matrix A is too short for the given dimensions and leading dimension");
    }

    // check inner dim
    if (x.len() < (1 + (n - 1) * incx.unsigned_abs() as usize))
        || (y.len() < (1 + (n - 1) * incy.unsigned_abs() as usize))
    {
        panic!("Vector x is too short for the given dimensions, increment and transposition");
    }

    // we use `scal` to handle the beta scaling and zeroing out y if beta is 0, as per BLAS spec
    scal(n, beta, y, incy);

    if alpha == 0.0 {
        return;
    }

    if !uplo {}
}
