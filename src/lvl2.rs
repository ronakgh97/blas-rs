use crate::lvl1::scal;
use crate::utils::from_m256;
use std::arch::x86_64::{
    _MM_HINT_NTA, _mm_prefetch, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps,
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

    // non-transposed case; axpy each col of A scaled by x[j] into y; prefetch next col of A
    // transposed case; dot product each col of A with x, scale by alpha, add to y[j]; prefetch next col of A
    if !is_trans {
        let ix_b = if incx < 0 {
            (1 - n as isize) * incx_isize
        } else {
            0
        };

        let a_ptr = a.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        // simd path, load columns of A and x, do axpy, store to y; full rizz
        if incx == 1 && incy == 1 {
            unsafe {
                let mut j = 0usize;
                while j + 1 < n {
                    let alpha_x0 = alpha * *x_ptr.add(j);
                    let alpha_x1 = alpha * *x_ptr.add(j + 1);
                    let bx0 = _mm256_set1_ps(alpha_x0);
                    let bx1 = _mm256_set1_ps(alpha_x1);
                    let col0 = a_ptr.add(j * lda);
                    let col1 = a_ptr.add((j + 1) * lda);

                    let mut i = 0usize;
                    while i + 32 <= m {
                        // load y write buf
                        let y0 = _mm256_loadu_ps(y_ptr.add(i));
                        let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                        let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                        let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                        // load two column (64) of A
                        let a00 = _mm256_loadu_ps(col0.add(i));
                        let a01 = _mm256_loadu_ps(col0.add(i + 8));
                        let a02 = _mm256_loadu_ps(col0.add(i + 16));
                        let a03 = _mm256_loadu_ps(col0.add(i + 24));
                        let a10 = _mm256_loadu_ps(col1.add(i));
                        let a11 = _mm256_loadu_ps(col1.add(i + 8));
                        let a12 = _mm256_loadu_ps(col1.add(i + 16));
                        let a13 = _mm256_loadu_ps(col1.add(i + 24));

                        let y0 = _mm256_fmadd_ps(bx1, a10, _mm256_fmadd_ps(bx0, a00, y0));
                        let y1 = _mm256_fmadd_ps(bx1, a11, _mm256_fmadd_ps(bx0, a01, y1));
                        let y2 = _mm256_fmadd_ps(bx1, a12, _mm256_fmadd_ps(bx0, a02, y2));
                        let y3 = _mm256_fmadd_ps(bx1, a13, _mm256_fmadd_ps(bx0, a03, y3));

                        // write back
                        _mm256_storeu_ps(y_ptr.add(i), y0);
                        _mm256_storeu_ps(y_ptr.add(i + 8), y1);
                        _mm256_storeu_ps(y_ptr.add(i + 16), y2);
                        _mm256_storeu_ps(y_ptr.add(i + 24), y3);

                        i += 32;
                    }

                    while i + 8 <= m {
                        let y0 = _mm256_loadu_ps(y_ptr.add(i));
                        let a00 = _mm256_loadu_ps(col0.add(i));
                        let a10 = _mm256_loadu_ps(col1.add(i));
                        let y0 = _mm256_fmadd_ps(bx1, a10, _mm256_fmadd_ps(bx0, a00, y0));
                        _mm256_storeu_ps(y_ptr.add(i), y0);
                        i += 8;
                    }

                    while i < m {
                        let v = alpha_x0.mul_add(*col0.add(i), *y_ptr.add(i));
                        *y_ptr.add(i) = alpha_x1.mul_add(*col1.add(i), v);
                        i += 1;
                    }

                    j += 2;
                }

                // process 1 col if n is odd
                if j < n {
                    let alpha_x = alpha * *x_ptr.add(j);
                    let bx = _mm256_set1_ps(alpha_x);
                    let col = a_ptr.add(j * lda);

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
                        let a0 = _mm256_loadu_ps(col.add(i));
                        let y0 = _mm256_loadu_ps(y_ptr.add(i));
                        _mm256_storeu_ps(y_ptr.add(i), _mm256_fmadd_ps(bx, a0, y0));
                        i += 8;
                    }

                    // scalar fallback
                    while i < m {
                        *y_ptr.add(i) = alpha_x.mul_add(*col.add(i), *y_ptr.add(i));
                        i += 1;
                    }
                }
            }
            // if incx!=1 but incy is, we can still do the axpy with simd,
            // just load x with stride and prefetching, rizz but less rizz
        } else if incy == 1 {
            unsafe {
                for j in 0..n {
                    // load x (strided)
                    let alpha_x = alpha * *x_ptr.offset(ix_b + j as isize * incx_isize);
                    let bx = _mm256_set1_ps(alpha_x);
                    let col = a_ptr.add(j * lda);

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
                        let a0 = _mm256_loadu_ps(col.add(i));
                        let y0 = _mm256_loadu_ps(y_ptr.add(i));
                        _mm256_storeu_ps(y_ptr.add(i), _mm256_fmadd_ps(bx, a0, y0));
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
            let iy_b = if incy < 0 {
                (1 - m as isize) * incy_isize
            } else {
                0
            };
            unsafe {
                for j in 0..n {
                    // get strided x
                    let x_val = alpha * *x_ptr.offset(ix_b + j as isize * incx_isize);
                    let col = a_ptr.add(j * lda);

                    if j + 1 < n {
                        _mm_prefetch(a_ptr.add((j + 1) * lda) as *const i8, _MM_HINT_NTA);
                    }

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
        unsafe {
            if incx == 1 && incy == 1 {
                let mut j = 0usize;

                // process 2 columns to maximize X vector reuse
                while j + 1 < n {
                    let col0 = a_ptr.add(j * lda);
                    let col1 = a_ptr.add((j + 1) * lda);

                    if j + 2 < n {
                        _mm_prefetch(a_ptr.add((j + 2) * lda) as *const i8, _MM_HINT_NTA);
                    }

                    // set 4 acc from each column
                    let mut sum0_0 = _mm256_setzero_ps();
                    let mut sum0_1 = _mm256_setzero_ps();
                    let mut sum0_2 = _mm256_setzero_ps();
                    let mut sum0_3 = _mm256_setzero_ps();

                    let mut sum1_0 = _mm256_setzero_ps();
                    let mut sum1_1 = _mm256_setzero_ps();
                    let mut sum1_2 = _mm256_setzero_ps();
                    let mut sum1_3 = _mm256_setzero_ps();

                    let mut i = 0usize;

                    while i + 32 <= m {
                        // load x once (32 floats)
                        let x0 = _mm256_loadu_ps(x_ptr.add(i));
                        let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                        let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                        let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                        // compute fma for col 0
                        sum0_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i)), x0, sum0_0);
                        sum0_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i + 8)), x1, sum0_1);
                        sum0_2 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i + 16)), x2, sum0_2);
                        sum0_3 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i + 24)), x3, sum0_3);

                        // compute fma for col 1
                        sum1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i)), x0, sum1_0);
                        sum1_1 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i + 8)), x1, sum1_1);
                        sum1_2 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i + 16)), x2, sum1_2);
                        sum1_3 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i + 24)), x3, sum1_3);

                        i += 32;
                    }

                    while i + 8 <= m {
                        let x0 = _mm256_loadu_ps(x_ptr.add(i));
                        sum0_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col0.add(i)), x0, sum0_0);
                        sum1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(col1.add(i)), x0, sum1_0);
                        i += 8;
                    }

                    // combine acc for col 2
                    let sum0_01 = _mm256_add_ps(sum0_0, sum0_1);
                    let sum0_23 = _mm256_add_ps(sum0_2, sum0_3);
                    let mut dot0 = from_m256(_mm256_add_ps(sum0_01, sum0_23));

                    // combine acc for col 1
                    let sum1_01 = _mm256_add_ps(sum1_0, sum1_1);
                    let sum1_23 = _mm256_add_ps(sum1_2, sum1_3);
                    let mut dot1 = from_m256(_mm256_add_ps(sum1_01, sum1_23));

                    // scalar fallback
                    while i < m {
                        let x_val = *x_ptr.add(i);
                        dot0 += *col0.add(i) * x_val;
                        dot1 += *col1.add(i) * x_val;
                        i += 1;
                    }

                    // write back with alpha scaling (contiguous)
                    *y_ptr.add(j) = alpha.mul_add(dot0, *y_ptr.add(j));
                    *y_ptr.add(j + 1) = alpha.mul_add(dot1, *y_ptr.add(j + 1));

                    j += 2;
                }

                // if n is odd, process last column; no reuse but still simd dot product
                if j < n {
                    let col = a_ptr.add(j * lda);
                    let mut sum0 = _mm256_setzero_ps();
                    let mut sum1 = _mm256_setzero_ps();
                    let mut i = 0usize;

                    while i + 16 <= m {
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
                        i += 16;
                    }
                    while i + 8 <= m {
                        sum0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(col.add(i)),
                            _mm256_loadu_ps(x_ptr.add(i)),
                            sum0,
                        );
                        i += 8;
                    }

                    let mut dot_val = from_m256(_mm256_add_ps(sum0, sum1));
                    while i < m {
                        dot_val += *col.add(i) * *x_ptr.add(i);
                        i += 1;
                    }
                    *y_ptr.add(j) = alpha.mul_add(dot_val, *y_ptr.add(j));
                }
            } else if incx == 1 {
                for j in 0..n {
                    let col = a_ptr.add(j * lda);

                    if j + 1 < n {
                        _mm_prefetch(a_ptr.add((j + 1) * lda) as *const i8, _MM_HINT_NTA);
                    }

                    // four accumulators for the dot product,
                    // we will reduce them at the end
                    let mut sum0 = _mm256_setzero_ps();
                    let mut sum1 = _mm256_setzero_ps();
                    let mut sum2 = _mm256_setzero_ps();
                    let mut sum3 = _mm256_setzero_ps();

                    let mut i = 0usize;
                    while i + 32 <= m {
                        // load (8x4) block of A and x, do fmadd, accumulate into sums; this is the rizz part
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
                        let a0 = _mm256_loadu_ps(col.add(i));
                        let x0 = _mm256_loadu_ps(x_ptr.add(i));
                        sum0 = _mm256_fmadd_ps(a0, x0, sum0);
                        i += 8;
                    }

                    // reduce to scalar
                    let sum01 = _mm256_add_ps(sum0, sum1);
                    let sum23 = _mm256_add_ps(sum2, sum3);
                    let sum = _mm256_add_ps(sum01, sum23);
                    let mut dot_val = from_m256(sum);

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
                // non-simd path, just do the dot product with prefetching, no rizz
            } else {
                for j in 0..n {
                    let col = a_ptr.add(j * lda);

                    if j + 1 < n {
                        _mm_prefetch(a_ptr.add((j + 1) * lda) as *const i8, _MM_HINT_NTA);
                    }
                    if j + 8 < n {
                        _mm_prefetch(a_ptr.add((j + 8) * lda) as *const i8, _MM_HINT_NTA);
                    }

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
    use crate::utils::gen_fill;
    use std::hint::black_box;

    let warmup_count = 32;
    let run_count = 256;
    let size = 4196;

    let mut a = vec![1.0f32; size * size];
    let mut x = vec![1.0f32; size];
    let mut y = vec![0.0f32; size];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    gen_fill(&mut a);
    gen_fill(&mut x);

    for _ in 0..warmup_count {
        y.fill(1.0);
        gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y, 1, false);
    }

    y.fill(1.0);

    let start = std::time::Instant::now();
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
