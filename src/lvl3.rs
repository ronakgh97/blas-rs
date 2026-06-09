use crate::lvl1::scal;
use crate::utils::reduce_f32;
use std::arch::x86_64::{
    _MM_HINT_T0, _MM_HINT_T1, _mm_prefetch, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
    _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};
use std::slice::from_raw_parts_mut;

// TODO; too slow, i know there is rooms for perf!!!, and i have no idea what i am doing, REALLY

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// The gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemm.html)
pub fn gemm(
    m: usize,      // rows of C (and A when not transposed)
    n: usize,      // cols of C (and B when not transposed)
    k: usize,      // inner dimension (cols of A when not transposed, rows of B when not transposed)
    alpha: f32,    // scaling for product
    a: &[f32],     // matrix A: m×k when is_trans_a=false, k×m when is_trans_a=true (column major)
    lda: usize,    // leading dim of A (column major storage)
    b: &[f32],     // matrix B: k×n when is_trans_b=false, n×k when is_trans_b=true (column major)
    ldb: usize,    // leading dim of B (column major storage)
    beta: f32,     // result scaling
    c: &mut [f32], // resultant mat C: m×n (column major)
    ldc: usize,    // leading dim of C (column major storage)
    is_trans_a: bool,
    is_trans_b: bool,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let a_rows = if is_trans_a { k } else { m };
    let a_cols = if is_trans_a { m } else { k };

    let b_rows = if is_trans_b { n } else { k };
    let b_cols = if is_trans_b { k } else { n };

    if lda == 0 || lda < a_rows {
        panic!("lda must be >= rows of stored A and non-zero");
    }

    if ldb == 0 || ldb < b_rows {
        panic!("ldb must be >= rows of stored B and non-zero");
    }

    if ldc == 0 || ldc < m {
        panic!("ldc must be >= rows of C and non-zero");
    }

    if a.len() < (a_cols - 1) * lda + a_rows {
        panic!("Matrix A buffer too small");
    }
    if b.len() < (b_cols - 1) * ldb + b_rows {
        panic!("Matrix B buffer too small");
    }
    if c.len() < (n - 1) * ldc + m {
        panic!("Matrix C buffer too small");
    }

    // pre-scale
    if beta != 1.0 {
        for j in 0..n {
            let start = j * ldc;
            let col = unsafe { from_raw_parts_mut(c.as_mut_ptr().add(start), m) };
            scal(m, beta, col, 1);
        }
    }

    // ret if 0 alpha
    if alpha == 0.0 {
        return;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    match is_trans_a {
        // C = alpha * A * B + beta * C; gemm_f_f
        // C = alpha * A * B^T + beta * C; gemm_f_t
        false => unsafe {
            let mut j = 0usize;
            // iter over 4 columns of C at a time (8x4)
            while j + 3 < n {
                let c0_ptr = c_ptr.add(j * ldc);
                let c1_ptr = c_ptr.add((j + 1) * ldc);
                let c2_ptr = c_ptr.add((j + 2) * ldc);
                let c3_ptr = c_ptr.add((j + 3) * ldc);

                let mut p = 0usize;
                // inner k loop; Unrolled by 4 to maximize FMA throughput
                while p + 3 < k {
                    // load and scale all 16 B scalars for this 4x4 K-N tile
                    #[rustfmt::skip]
                    let b0_0 = alpha* if is_trans_b { *b_ptr.add(j + p * ldb) } else {*b_ptr.add(p + j * ldb) };
                    #[rustfmt::skip]
                    let b0_1 = alpha * if is_trans_b { *b_ptr.add(j + 1 + p * ldb) } else { *b_ptr.add(p + (j + 1) * ldb) };
                    #[rustfmt::skip]
                    let b0_2 = alpha * if is_trans_b { *b_ptr.add(j + 2 + p * ldb) } else {*b_ptr.add(p + (j + 2) * ldb) };
                    #[rustfmt::skip]
                    let b0_3 = alpha * if is_trans_b { *b_ptr.add(j + 3 + p * ldb) } else { *b_ptr.add(p + (j + 3) * ldb) };

                    #[rustfmt::skip]
                    let b1_0 = alpha * if is_trans_b { *b_ptr.add(j + (p + 1) * ldb) } else {*b_ptr.add(p + 1 + j * ldb) };
                    #[rustfmt::skip]
                    let b1_1 = alpha * if is_trans_b { *b_ptr.add(j + 1 + (p + 1) * ldb) } else { *b_ptr.add(p + 1 + (j + 1) * ldb) };
                    #[rustfmt::skip]
                    let b1_2 = alpha * if is_trans_b { *b_ptr.add(j + 2 + (p + 1) * ldb) } else { *b_ptr.add(p + 1 + (j + 2) * ldb) };
                    #[rustfmt::skip]
                    let b1_3 = alpha * if is_trans_b { *b_ptr.add(j + 3 + (p + 1) * ldb) } else { *b_ptr.add(p + 1 + (j + 3) * ldb) };

                    #[rustfmt::skip]
                    let b2_0 = alpha * if is_trans_b { *b_ptr.add(j + (p + 2) * ldb) } else { *b_ptr.add(p + 2 + j * ldb) };
                    #[rustfmt::skip]
                    let b2_1 = alpha * if is_trans_b { *b_ptr.add(j + 1 + (p + 2) * ldb) } else { *b_ptr.add(p + 2 + (j + 1) * ldb) };
                    #[rustfmt::skip]
                    let b2_2 = alpha * if is_trans_b { *b_ptr.add(j + 2 + (p + 2) * ldb) } else { *b_ptr.add(p + 2 + (j + 2) * ldb) };
                    #[rustfmt::skip]
                    let b2_3 = alpha * if is_trans_b { *b_ptr.add(j + 3 + (p + 2) * ldb) } else { *b_ptr.add(p + 2 + (j + 3) * ldb) };

                    #[rustfmt::skip]
                    let b3_0 = alpha * if is_trans_b { *b_ptr.add(j + (p + 3) * ldb) } else { *b_ptr.add(p + 3 + j * ldb) };
                    #[rustfmt::skip]
                    let b3_1 = alpha * if is_trans_b { *b_ptr.add(j + 1 + (p + 3) * ldb) } else { *b_ptr.add(p + 3 + (j + 1) * ldb) };
                    #[rustfmt::skip]
                    let b3_2 = alpha * if is_trans_b { *b_ptr.add(j + 2 + (p + 3) * ldb) } else { *b_ptr.add(p + 3 + (j + 2) * ldb) };
                    #[rustfmt::skip]
                    let b3_3 = alpha * if is_trans_b { *b_ptr.add(j + 3 + (p + 3) * ldb) } else { *b_ptr.add(p + 3 + (j + 3) * ldb) };

                    let mut i = 0usize;

                    // prefetch & pray
                    {
                        // prefetch p+4, p+5, p+6, p+7
                        for offset in 0..4 {
                            let col_ptr = a_ptr.add((p + 4 + offset) * lda);
                            _mm_prefetch::<_MM_HINT_T0>(col_ptr as *const i8); // prefetch the first cache line of
                            _mm_prefetch::<_MM_HINT_T0>(col_ptr.add(16) as *const i8); // next cache line
                            _mm_prefetch::<_MM_HINT_T1>(col_ptr.add(32) as *const i8); // maybe to L2
                        }
                    }

                    // inner M Loop; Sequential memory access for Matrix A
                    while i + 16 <= m {
                        let mut c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        let mut c01 = _mm256_loadu_ps(c0_ptr.add(i + 8));
                        let mut c10 = _mm256_loadu_ps(c1_ptr.add(i));
                        let mut c11 = _mm256_loadu_ps(c1_ptr.add(i + 8));
                        let mut c20 = _mm256_loadu_ps(c2_ptr.add(i));
                        let mut c21 = _mm256_loadu_ps(c2_ptr.add(i + 8));
                        let mut c30 = _mm256_loadu_ps(c3_ptr.add(i));
                        let mut c31 = _mm256_loadu_ps(c3_ptr.add(i + 8));

                        // K step 0
                        let a0 = _mm256_loadu_ps(a_ptr.add(p * lda + i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(p * lda + i + 8));
                        c00 = _mm256_fmadd_ps(_mm256_set1_ps(b0_0), a0, c00);
                        c01 = _mm256_fmadd_ps(_mm256_set1_ps(b0_0), a1, c01);
                        c10 = _mm256_fmadd_ps(_mm256_set1_ps(b0_1), a0, c10);
                        c11 = _mm256_fmadd_ps(_mm256_set1_ps(b0_1), a1, c11);
                        c20 = _mm256_fmadd_ps(_mm256_set1_ps(b0_2), a0, c20);
                        c21 = _mm256_fmadd_ps(_mm256_set1_ps(b0_2), a1, c21);
                        c30 = _mm256_fmadd_ps(_mm256_set1_ps(b0_3), a0, c30);
                        c31 = _mm256_fmadd_ps(_mm256_set1_ps(b0_3), a1, c31);

                        // K step 1
                        let a0 = _mm256_loadu_ps(a_ptr.add((p + 1) * lda + i));
                        let a1 = _mm256_loadu_ps(a_ptr.add((p + 1) * lda + i + 8));
                        c00 = _mm256_fmadd_ps(_mm256_set1_ps(b1_0), a0, c00);
                        c01 = _mm256_fmadd_ps(_mm256_set1_ps(b1_0), a1, c01);
                        c10 = _mm256_fmadd_ps(_mm256_set1_ps(b1_1), a0, c10);
                        c11 = _mm256_fmadd_ps(_mm256_set1_ps(b1_1), a1, c11);
                        c20 = _mm256_fmadd_ps(_mm256_set1_ps(b1_2), a0, c20);
                        c21 = _mm256_fmadd_ps(_mm256_set1_ps(b1_2), a1, c21);
                        c30 = _mm256_fmadd_ps(_mm256_set1_ps(b1_3), a0, c30);
                        c31 = _mm256_fmadd_ps(_mm256_set1_ps(b1_3), a1, c31);

                        // K step 2
                        let a0 = _mm256_loadu_ps(a_ptr.add((p + 2) * lda + i));
                        let a1 = _mm256_loadu_ps(a_ptr.add((p + 2) * lda + i + 8));
                        c00 = _mm256_fmadd_ps(_mm256_set1_ps(b2_0), a0, c00);
                        c01 = _mm256_fmadd_ps(_mm256_set1_ps(b2_0), a1, c01);
                        c10 = _mm256_fmadd_ps(_mm256_set1_ps(b2_1), a0, c10);
                        c11 = _mm256_fmadd_ps(_mm256_set1_ps(b2_1), a1, c11);
                        c20 = _mm256_fmadd_ps(_mm256_set1_ps(b2_2), a0, c20);
                        c21 = _mm256_fmadd_ps(_mm256_set1_ps(b2_2), a1, c21);
                        c30 = _mm256_fmadd_ps(_mm256_set1_ps(b2_3), a0, c30);
                        c31 = _mm256_fmadd_ps(_mm256_set1_ps(b2_3), a1, c31);

                        // K step 3
                        let a0 = _mm256_loadu_ps(a_ptr.add((p + 3) * lda + i));
                        let a1 = _mm256_loadu_ps(a_ptr.add((p + 3) * lda + i + 8));
                        c00 = _mm256_fmadd_ps(_mm256_set1_ps(b3_0), a0, c00);
                        c01 = _mm256_fmadd_ps(_mm256_set1_ps(b3_0), a1, c01);
                        c10 = _mm256_fmadd_ps(_mm256_set1_ps(b3_1), a0, c10);
                        c11 = _mm256_fmadd_ps(_mm256_set1_ps(b3_1), a1, c11);
                        c20 = _mm256_fmadd_ps(_mm256_set1_ps(b3_2), a0, c20);
                        c21 = _mm256_fmadd_ps(_mm256_set1_ps(b3_2), a1, c21);
                        c30 = _mm256_fmadd_ps(_mm256_set1_ps(b3_3), a0, c30);
                        c31 = _mm256_fmadd_ps(_mm256_set1_ps(b3_3), a1, c31);

                        // write to registers
                        _mm256_storeu_ps(c0_ptr.add(i), c00);
                        _mm256_storeu_ps(c0_ptr.add(i + 8), c01);
                        _mm256_storeu_ps(c1_ptr.add(i), c10);
                        _mm256_storeu_ps(c1_ptr.add(i + 8), c11);
                        _mm256_storeu_ps(c2_ptr.add(i), c20);
                        _mm256_storeu_ps(c2_ptr.add(i + 8), c21);
                        _mm256_storeu_ps(c3_ptr.add(i), c30);
                        _mm256_storeu_ps(c3_ptr.add(i + 8), c31);

                        i += 16;
                    }

                    // remainder M loop (8x4 chunk) inside unrolled K
                    while i + 8 <= m {
                        let mut c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        let mut c10 = _mm256_loadu_ps(c1_ptr.add(i));
                        let mut c20 = _mm256_loadu_ps(c2_ptr.add(i));
                        let mut c30 = _mm256_loadu_ps(c3_ptr.add(i));

                        for k_offset in 0..4 {
                            let b_sel = match k_offset {
                                0 => (b0_0, b0_1, b0_2, b0_3),
                                1 => (b1_0, b1_1, b1_2, b1_3),
                                2 => (b2_0, b2_1, b2_2, b2_3),
                                _ => (b3_0, b3_1, b3_2, b3_3),
                            };

                            let a0 = _mm256_loadu_ps(a_ptr.add((p + k_offset) * lda + i));
                            c00 = _mm256_fmadd_ps(_mm256_set1_ps(b_sel.0), a0, c00);
                            c10 = _mm256_fmadd_ps(_mm256_set1_ps(b_sel.1), a0, c10);
                            c20 = _mm256_fmadd_ps(_mm256_set1_ps(b_sel.2), a0, c20);
                            c30 = _mm256_fmadd_ps(_mm256_set1_ps(b_sel.3), a0, c30);
                        }

                        _mm256_storeu_ps(c0_ptr.add(i), c00);
                        _mm256_storeu_ps(c1_ptr.add(i), c10);
                        _mm256_storeu_ps(c2_ptr.add(i), c20);
                        _mm256_storeu_ps(c3_ptr.add(i), c30);
                        i += 8;
                    }

                    // remainder M loop (scalars 1-7) inside unrolled K
                    while i < m {
                        for k_offset in 0..4 {
                            let b_sel = match k_offset {
                                0 => (b0_0, b0_1, b0_2, b0_3),
                                1 => (b1_0, b1_1, b1_2, b1_3),
                                2 => (b2_0, b2_1, b2_2, b2_3),
                                _ => (b3_0, b3_1, b3_2, b3_3),
                            };
                            let a_val = *a_ptr.add((p + k_offset) * lda + i);
                            *c0_ptr.add(i) = b_sel.0.mul_add(a_val, *c0_ptr.add(i));
                            *c1_ptr.add(i) = b_sel.1.mul_add(a_val, *c1_ptr.add(i));
                            *c2_ptr.add(i) = b_sel.2.mul_add(a_val, *c2_ptr.add(i));
                            *c3_ptr.add(i) = b_sel.3.mul_add(a_val, *c3_ptr.add(i));
                        }
                        i += 1;
                    }

                    p += 4;
                }

                // rest of K loop for 4 cols of C
                while p < k {
                    #[rustfmt::skip]
                    let b0 = alpha * if is_trans_b { *b_ptr.add(j + p * ldb) } else { *b_ptr.add(p + j * ldb) };
                    #[rustfmt::skip]
                    let b1 = alpha * if is_trans_b { *b_ptr.add(j + 1 + p * ldb) } else { *b_ptr.add(p + (j + 1) * ldb) };
                    #[rustfmt::skip]
                    let b2 = alpha * if is_trans_b { *b_ptr.add(j + 2 + p * ldb) } else { *b_ptr.add(p + (j + 2) * ldb) };
                    #[rustfmt::skip]
                    let b3 = alpha * if is_trans_b { *b_ptr.add(j + 3 + p * ldb) } else { *b_ptr.add(p + (j + 3) * ldb) };

                    let mut i = 0usize;
                    while i + 16 <= m {
                        let a0 = _mm256_loadu_ps(a_ptr.add(p * lda + i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(p * lda + i + 8));

                        _mm256_storeu_ps(
                            c0_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b0), a0, _mm256_loadu_ps(c0_ptr.add(i))),
                        );
                        _mm256_storeu_ps(
                            c0_ptr.add(i + 8),
                            _mm256_fmadd_ps(
                                _mm256_set1_ps(b0),
                                a1,
                                _mm256_loadu_ps(c0_ptr.add(i + 8)),
                            ),
                        );
                        _mm256_storeu_ps(
                            c1_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b1), a0, _mm256_loadu_ps(c1_ptr.add(i))),
                        );
                        _mm256_storeu_ps(
                            c1_ptr.add(i + 8),
                            _mm256_fmadd_ps(
                                _mm256_set1_ps(b1),
                                a1,
                                _mm256_loadu_ps(c1_ptr.add(i + 8)),
                            ),
                        );
                        _mm256_storeu_ps(
                            c2_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b2), a0, _mm256_loadu_ps(c2_ptr.add(i))),
                        );
                        _mm256_storeu_ps(
                            c2_ptr.add(i + 8),
                            _mm256_fmadd_ps(
                                _mm256_set1_ps(b2),
                                a1,
                                _mm256_loadu_ps(c2_ptr.add(i + 8)),
                            ),
                        );
                        _mm256_storeu_ps(
                            c3_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b3), a0, _mm256_loadu_ps(c3_ptr.add(i))),
                        );
                        _mm256_storeu_ps(
                            c3_ptr.add(i + 8),
                            _mm256_fmadd_ps(
                                _mm256_set1_ps(b3),
                                a1,
                                _mm256_loadu_ps(c3_ptr.add(i + 8)),
                            ),
                        );
                        i += 16;
                    }
                    // fallback
                    while i + 8 <= m {
                        let a0 = _mm256_loadu_ps(a_ptr.add(p * lda + i));
                        _mm256_storeu_ps(
                            c0_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b0), a0, _mm256_loadu_ps(c0_ptr.add(i))),
                        );
                        _mm256_storeu_ps(
                            c1_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b1), a0, _mm256_loadu_ps(c1_ptr.add(i))),
                        );
                        _mm256_storeu_ps(
                            c2_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b2), a0, _mm256_loadu_ps(c2_ptr.add(i))),
                        );
                        _mm256_storeu_ps(
                            c3_ptr.add(i),
                            _mm256_fmadd_ps(_mm256_set1_ps(b3), a0, _mm256_loadu_ps(c3_ptr.add(i))),
                        );
                        i += 8;
                    }
                    // cleanup
                    while i < m {
                        let a_val = *a_ptr.add(p * lda + i);
                        *c0_ptr.add(i) = b0.mul_add(a_val, *c0_ptr.add(i));
                        *c1_ptr.add(i) = b1.mul_add(a_val, *c1_ptr.add(i));
                        *c2_ptr.add(i) = b2.mul_add(a_val, *c2_ptr.add(i));
                        *c3_ptr.add(i) = b3.mul_add(a_val, *c3_ptr.add(i));
                        i += 1;
                    }
                    p += 1;
                }
                j += 4;
            }

            // col reminder loop
            while j < n {
                let c_col_ptr = c_ptr.add(j * ldc);
                let mut p = 0usize;
                while p < k {
                    let b_val = alpha
                        * if is_trans_b {
                            *b_ptr.add(j + p * ldb)
                        } else {
                            *b_ptr.add(p + j * ldb)
                        };
                    let mut i = 0usize;
                    while i + 8 <= m {
                        let a0 = _mm256_loadu_ps(a_ptr.add(p * lda + i));
                        _mm256_storeu_ps(
                            c_col_ptr.add(i),
                            _mm256_fmadd_ps(
                                _mm256_set1_ps(b_val),
                                a0,
                                _mm256_loadu_ps(c_col_ptr.add(i)),
                            ),
                        );
                        i += 8;
                    }

                    // scalar fallback
                    while i < m {
                        *c_col_ptr.add(i) =
                            b_val.mul_add(*a_ptr.add(p * lda + i), *c_col_ptr.add(i));
                        i += 1;
                    }
                    p += 1;
                }
                j += 1;
            }
        },
        // C = alpha * A^T * B^T + beta * C; gemm_t_t
        // C = alpha * A^T * B + beta * C; gemm_t_f
        true => unsafe {
            if is_trans_b {
                let mut b_packed = vec![0.0f32; k * 8];
                let mut j = 0usize;
                // process 8 cols of B at a time (8xk block), pack into contiguous for reuse in inner loop
                while j + 7 < n {
                    let mut p = 0;
                    // `gather load` 8 cols of B (strided) and pack into contiguous for reuse in inner loop
                    while p < k {
                        _mm256_storeu_ps(
                            b_packed.as_mut_ptr().add(p * 8),
                            _mm256_loadu_ps(b_ptr.add(j + p * ldb)),
                        );
                        p += 1;
                    }

                    // 8 col ptrs
                    let c0_ptr = c_ptr.add(j * ldc);
                    let c1_ptr = c_ptr.add((j + 1) * ldc);
                    let c2_ptr = c_ptr.add((j + 2) * ldc);
                    let c3_ptr = c_ptr.add((j + 3) * ldc);
                    let c4_ptr = c_ptr.add((j + 4) * ldc);
                    let c5_ptr = c_ptr.add((j + 5) * ldc);
                    let c6_ptr = c_ptr.add((j + 6) * ldc);
                    let c7_ptr = c_ptr.add((j + 7) * ldc);

                    let mut i = 0usize;
                    // process 4 rows of A at a time (4xk block) to reuse broadcasted A in inner loop
                    while i + 3 < m {
                        // 4 col ptrs for current 4 rows
                        let a_col0 = a_ptr.add(i * lda);
                        let a_col1 = a_ptr.add((i + 1) * lda);
                        let a_col2 = a_ptr.add((i + 2) * lda);
                        let a_col3 = a_ptr.add((i + 3) * lda);

                        // set up acc for each col of C
                        // acc round 1
                        let mut c0_0 = _mm256_setzero_ps();
                        let mut c1_0 = _mm256_setzero_ps();
                        let mut c2_0 = _mm256_setzero_ps();
                        let mut c3_0 = _mm256_setzero_ps();

                        // acc round 2
                        let mut c0_1 = _mm256_setzero_ps();
                        let mut c1_1 = _mm256_setzero_ps();
                        let mut c2_1 = _mm256_setzero_ps();
                        let mut c3_1 = _mm256_setzero_ps();

                        let mut p = 0usize;
                        // inner k dim loop
                        while p + 1 < k {
                            // load two packed_b block of 8 elements (2 cols)
                            // broadcast two scalars from A, compute 8x2 block of C
                            let b0 = _mm256_loadu_ps(b_packed.as_ptr().add(p * 8));
                            let b1 = _mm256_loadu_ps(b_packed.as_ptr().add((p + 1) * 8));

                            // broadcast
                            let a0 = _mm256_set1_ps(*a_col0.add(p));
                            let a1 = _mm256_set1_ps(*a_col1.add(p));
                            let a2 = _mm256_set1_ps(*a_col2.add(p));
                            let a3 = _mm256_set1_ps(*a_col3.add(p));

                            // fmadd; acc <= (A x B)
                            c0_0 = _mm256_fmadd_ps(a0, b0, c0_0);
                            c1_0 = _mm256_fmadd_ps(a1, b0, c1_0);
                            c2_0 = _mm256_fmadd_ps(a2, b0, c2_0);
                            c3_0 = _mm256_fmadd_ps(a3, b0, c3_0);

                            // 2nd round
                            let a0_1 = _mm256_set1_ps(*a_col0.add(p + 1));
                            let a1_1 = _mm256_set1_ps(*a_col1.add(p + 1));
                            let a2_1 = _mm256_set1_ps(*a_col2.add(p + 1));
                            let a3_1 = _mm256_set1_ps(*a_col3.add(p + 1));

                            c0_1 = _mm256_fmadd_ps(a0_1, b1, c0_1);
                            c1_1 = _mm256_fmadd_ps(a1_1, b1, c1_1);
                            c2_1 = _mm256_fmadd_ps(a2_1, b1, c2_1);
                            c3_1 = _mm256_fmadd_ps(a3_1, b1, c3_1);

                            p += 2;
                        }

                        // reminder handling for odd k, process 1 col of B
                        if p < k {
                            let b0 = _mm256_loadu_ps(b_packed.as_ptr().add(p * 8));

                            let a0 = _mm256_set1_ps(*a_col0.add(p));
                            let a1 = _mm256_set1_ps(*a_col1.add(p));
                            let a2 = _mm256_set1_ps(*a_col2.add(p));
                            let a3 = _mm256_set1_ps(*a_col3.add(p));

                            c0_0 = _mm256_fmadd_ps(a0, b0, c0_0);
                            c1_0 = _mm256_fmadd_ps(a1, b0, c1_0);
                            c2_0 = _mm256_fmadd_ps(a2, b0, c2_0);
                            c3_0 = _mm256_fmadd_ps(a3, b0, c3_0);
                        }

                        // reduce
                        c0_0 = _mm256_add_ps(c0_0, c0_1);
                        c1_0 = _mm256_add_ps(c1_0, c1_1);
                        c2_0 = _mm256_add_ps(c2_0, c2_1);
                        c3_0 = _mm256_add_ps(c3_0, c3_1);

                        // scale by alpha
                        let v_alpha = _mm256_set1_ps(alpha);
                        c0_0 = _mm256_mul_ps(c0_0, v_alpha);
                        c1_0 = _mm256_mul_ps(c1_0, v_alpha);
                        c2_0 = _mm256_mul_ps(c2_0, v_alpha);
                        c3_0 = _mm256_mul_ps(c3_0, v_alpha);

                        // since c is column-major by "choice"
                        // we do `scatter store`,i.e. write to non-contiguous mem from avx reg
                        let mut tmp = [0.0; 8];

                        // store back to C with scatter pattern
                        _mm256_storeu_ps(tmp.as_mut_ptr(), c0_0);
                        *c0_ptr.add(i) += tmp[0];
                        *c1_ptr.add(i) += tmp[1];
                        *c2_ptr.add(i) += tmp[2];
                        *c3_ptr.add(i) += tmp[3];
                        *c4_ptr.add(i) += tmp[4];
                        *c5_ptr.add(i) += tmp[5];
                        *c6_ptr.add(i) += tmp[6];
                        *c7_ptr.add(i) += tmp[7];

                        // store back to C for next row
                        _mm256_storeu_ps(tmp.as_mut_ptr(), c1_0);
                        *c0_ptr.add(i + 1) += tmp[0];
                        *c1_ptr.add(i + 1) += tmp[1];
                        *c2_ptr.add(i + 1) += tmp[2];
                        *c3_ptr.add(i + 1) += tmp[3];
                        *c4_ptr.add(i + 1) += tmp[4];
                        *c5_ptr.add(i + 1) += tmp[5];
                        *c6_ptr.add(i + 1) += tmp[6];
                        *c7_ptr.add(i + 1) += tmp[7];

                        // next row
                        _mm256_storeu_ps(tmp.as_mut_ptr(), c2_0);
                        *c0_ptr.add(i + 2) += tmp[0];
                        *c1_ptr.add(i + 2) += tmp[1];
                        *c2_ptr.add(i + 2) += tmp[2];
                        *c3_ptr.add(i + 2) += tmp[3];
                        *c4_ptr.add(i + 2) += tmp[4];
                        *c5_ptr.add(i + 2) += tmp[5];
                        *c6_ptr.add(i + 2) += tmp[6];
                        *c7_ptr.add(i + 2) += tmp[7];

                        // next row
                        _mm256_storeu_ps(tmp.as_mut_ptr(), c3_0);
                        *c0_ptr.add(i + 3) += tmp[0];
                        *c1_ptr.add(i + 3) += tmp[1];
                        *c2_ptr.add(i + 3) += tmp[2];
                        *c3_ptr.add(i + 3) += tmp[3];
                        *c4_ptr.add(i + 3) += tmp[4];
                        *c5_ptr.add(i + 3) += tmp[5];
                        *c6_ptr.add(i + 3) += tmp[6];
                        *c7_ptr.add(i + 3) += tmp[7];

                        i += 4;
                    }

                    // process 1 row, if m is odd/not divisible by 4
                    while i < m {
                        let a_col0 = a_ptr.add(i * lda);

                        let mut c0_0 = _mm256_setzero_ps();
                        let mut p = 0usize;
                        while p + 1 < k {
                            let b0 = _mm256_loadu_ps(b_packed.as_ptr().add(p * 8));
                            let b1 = _mm256_loadu_ps(b_packed.as_ptr().add((p + 1) * 8));

                            let a0 = _mm256_set1_ps(*a_col0.add(p));
                            c0_0 = _mm256_fmadd_ps(a0, b0, c0_0);

                            let a0_1 = _mm256_set1_ps(*a_col0.add(p + 1));
                            c0_0 = _mm256_fmadd_ps(a0_1, b1, c0_0);

                            p += 2;
                        }

                        // handle reminders
                        if p < k {
                            let b0 = _mm256_loadu_ps(b_packed.as_ptr().add(p * 8));
                            let a0 = _mm256_set1_ps(*a_col0.add(p));
                            c0_0 = _mm256_fmadd_ps(a0, b0, c0_0);
                        }

                        // scale by alpha, write back
                        let v_alpha = _mm256_set1_ps(alpha);
                        c0_0 = _mm256_mul_ps(c0_0, v_alpha);

                        // `scatter load`
                        let mut tmp = [0.0; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), c0_0);
                        *c0_ptr.add(i) += tmp[0];
                        *c1_ptr.add(i) += tmp[1];
                        *c2_ptr.add(i) += tmp[2];
                        *c3_ptr.add(i) += tmp[3];
                        *c4_ptr.add(i) += tmp[4];
                        *c5_ptr.add(i) += tmp[5];
                        *c6_ptr.add(i) += tmp[6];
                        *c7_ptr.add(i) += tmp[7];

                        i += 1;
                    }

                    j += 8;
                }

                // final scalar fallback (no simd, no rizz, no cpu)
                while j < n {
                    let c0_ptr = c_ptr.add(j * ldc);
                    let mut i = 0usize;

                    // process remaining col, 4 rows at a time here
                    while i + 3 < m {
                        let a_col0 = a_ptr.add(i * lda);
                        let a_col1 = a_ptr.add((i + 1) * lda);
                        let a_col2 = a_ptr.add((i + 2) * lda);
                        let a_col3 = a_ptr.add((i + 3) * lda);

                        // four acc
                        let mut sum0 = 0.0;
                        let mut sum1 = 0.0;
                        let mut sum2 = 0.0;
                        let mut sum3 = 0.0;

                        let mut p = 0usize;
                        while p + 3 < k {
                            let b0 = *b_ptr.add(j + p * ldb);
                            let b1 = *b_ptr.add(j + (p + 1) * ldb);
                            let b2 = *b_ptr.add(j + (p + 2) * ldb);
                            let b3 = *b_ptr.add(j + (p + 3) * ldb);

                            // mul_add 'em
                            sum0 = (*a_col0.add(p)).mul_add(b0, sum0);
                            sum1 = (*a_col1.add(p)).mul_add(b0, sum1);
                            sum2 = (*a_col2.add(p)).mul_add(b0, sum2);
                            sum3 = (*a_col3.add(p)).mul_add(b0, sum3);

                            sum0 = (*a_col0.add(p + 1)).mul_add(b1, sum0);
                            sum1 = (*a_col1.add(p + 1)).mul_add(b1, sum1);
                            sum2 = (*a_col2.add(p + 1)).mul_add(b1, sum2);
                            sum3 = (*a_col3.add(p + 1)).mul_add(b1, sum3);

                            sum0 = (*a_col0.add(p + 2)).mul_add(b2, sum0);
                            sum1 = (*a_col1.add(p + 2)).mul_add(b2, sum1);
                            sum2 = (*a_col2.add(p + 2)).mul_add(b2, sum2);
                            sum3 = (*a_col3.add(p + 2)).mul_add(b2, sum3);

                            sum0 = (*a_col0.add(p + 3)).mul_add(b3, sum0);
                            sum1 = (*a_col1.add(p + 3)).mul_add(b3, sum1);
                            sum2 = (*a_col2.add(p + 3)).mul_add(b3, sum2);
                            sum3 = (*a_col3.add(p + 3)).mul_add(b3, sum3);

                            p += 4;
                        }

                        // handle remaining
                        while p < k {
                            let b0 = *b_ptr.add(j + p * ldb);
                            sum0 = (*a_col0.add(p)).mul_add(b0, sum0);
                            sum1 = (*a_col1.add(p)).mul_add(b0, sum1);
                            sum2 = (*a_col2.add(p)).mul_add(b0, sum2);
                            sum3 = (*a_col3.add(p)).mul_add(b0, sum3);
                            p += 1;
                        }

                        // scale and write to C
                        *c0_ptr.add(i) += alpha * sum0;
                        *c0_ptr.add(i + 1) += alpha * sum1;
                        *c0_ptr.add(i + 2) += alpha * sum2;
                        *c0_ptr.add(i + 3) += alpha * sum3;

                        i += 4;
                    }

                    // last final fallback, m < 4, n < 8
                    while i < m {
                        let a_col0 = a_ptr.add(i * lda);
                        let mut sum0 = 0.0;
                        let mut p = 0usize;
                        // unroll 4 for a_col0 and b, accumulate into sum0 for one element in C
                        while p + 3 < k {
                            let b0 = *b_ptr.add(j + p * ldb);
                            let b1 = *b_ptr.add(j + (p + 1) * ldb);
                            let b2 = *b_ptr.add(j + (p + 2) * ldb);
                            let b3 = *b_ptr.add(j + (p + 3) * ldb);
                            sum0 = (*a_col0.add(p)).mul_add(b0, sum0);
                            sum0 = (*a_col0.add(p + 1)).mul_add(b1, sum0);
                            sum0 = (*a_col0.add(p + 2)).mul_add(b2, sum0);
                            sum0 = (*a_col0.add(p + 3)).mul_add(b3, sum0);
                            p += 4;
                        }
                        // remaining
                        while p < k {
                            let b0 = *b_ptr.add(j + p * ldb);
                            sum0 = (*a_col0.add(p)).mul_add(b0, sum0);
                            p += 1;
                        }
                        *c0_ptr.add(i) += alpha * sum0;
                        i += 1;
                    }

                    j += 1;
                }
            } else {
                let mut j = 0usize;
                // process 4 cols of C at a time
                while j + 3 < n {
                    let c0_ptr = c_ptr.add(j * ldc);
                    let c1_ptr = c_ptr.add((j + 1) * ldc);
                    let c2_ptr = c_ptr.add((j + 2) * ldc);
                    let c3_ptr = c_ptr.add((j + 3) * ldc);

                    let b0_ptr = b_ptr.add(j * ldb);
                    let b1_ptr = b_ptr.add((j + 1) * ldb);
                    let b2_ptr = b_ptr.add((j + 2) * ldb);
                    let b3_ptr = b_ptr.add((j + 3) * ldb);

                    let mut i = 0usize;
                    // process 4 rows of A at a time
                    while i + 3 < m {
                        let a0_ptr = a_ptr.add(i * lda);
                        let a1_ptr = a_ptr.add((i + 1) * lda);
                        let a2_ptr = a_ptr.add((i + 2) * lda);
                        let a3_ptr = a_ptr.add((i + 3) * lda);

                        // 16 accumulators for a 4x4 block of C
                        let mut sum00 = _mm256_setzero_ps();
                        let mut sum01 = _mm256_setzero_ps();
                        let mut sum02 = _mm256_setzero_ps();
                        let mut sum03 = _mm256_setzero_ps();

                        let mut sum10 = _mm256_setzero_ps();
                        let mut sum11 = _mm256_setzero_ps();
                        let mut sum12 = _mm256_setzero_ps();
                        let mut sum13 = _mm256_setzero_ps();

                        let mut sum20 = _mm256_setzero_ps();
                        let mut sum21 = _mm256_setzero_ps();
                        let mut sum22 = _mm256_setzero_ps();
                        let mut sum23 = _mm256_setzero_ps();

                        let mut sum30 = _mm256_setzero_ps();
                        let mut sum31 = _mm256_setzero_ps();
                        let mut sum32 = _mm256_setzero_ps();
                        let mut sum33 = _mm256_setzero_ps();

                        let mut p = 0usize;
                        // inner K loop; step by 8 (1 AVX register)
                        while p + 7 < k {
                            let b0 = _mm256_loadu_ps(b0_ptr.add(p));
                            let b1 = _mm256_loadu_ps(b1_ptr.add(p));
                            let b2 = _mm256_loadu_ps(b2_ptr.add(p));
                            let b3 = _mm256_loadu_ps(b3_ptr.add(p));

                            let a0 = _mm256_loadu_ps(a0_ptr.add(p));
                            sum00 = _mm256_fmadd_ps(a0, b0, sum00);
                            sum01 = _mm256_fmadd_ps(a0, b1, sum01);
                            sum02 = _mm256_fmadd_ps(a0, b2, sum02);
                            sum03 = _mm256_fmadd_ps(a0, b3, sum03);

                            let a1 = _mm256_loadu_ps(a1_ptr.add(p));
                            sum10 = _mm256_fmadd_ps(a1, b0, sum10);
                            sum11 = _mm256_fmadd_ps(a1, b1, sum11);
                            sum12 = _mm256_fmadd_ps(a1, b2, sum12);
                            sum13 = _mm256_fmadd_ps(a1, b3, sum13);

                            let a2 = _mm256_loadu_ps(a2_ptr.add(p));
                            sum20 = _mm256_fmadd_ps(a2, b0, sum20);
                            sum21 = _mm256_fmadd_ps(a2, b1, sum21);
                            sum22 = _mm256_fmadd_ps(a2, b2, sum22);
                            sum23 = _mm256_fmadd_ps(a2, b3, sum23);

                            let a3 = _mm256_loadu_ps(a3_ptr.add(p));
                            sum30 = _mm256_fmadd_ps(a3, b0, sum30);
                            sum31 = _mm256_fmadd_ps(a3, b1, sum31);
                            sum32 = _mm256_fmadd_ps(a3, b2, sum32);
                            sum33 = _mm256_fmadd_ps(a3, b3, sum33);

                            p += 8;
                        }

                        // reduce scalar
                        let mut dot00 = reduce_f32(sum00);
                        let mut dot01 = reduce_f32(sum01);
                        let mut dot02 = reduce_f32(sum02);
                        let mut dot03 = reduce_f32(sum03);

                        let mut dot10 = reduce_f32(sum10);
                        let mut dot11 = reduce_f32(sum11);
                        let mut dot12 = reduce_f32(sum12);
                        let mut dot13 = reduce_f32(sum13);

                        let mut dot20 = reduce_f32(sum20);
                        let mut dot21 = reduce_f32(sum21);
                        let mut dot22 = reduce_f32(sum22);
                        let mut dot23 = reduce_f32(sum23);

                        let mut dot30 = reduce_f32(sum30);
                        let mut dot31 = reduce_f32(sum31);
                        let mut dot32 = reduce_f32(sum32);
                        let mut dot33 = reduce_f32(sum33);

                        // scalar fallback for remaining K
                        while p < k {
                            let a0_val = *a0_ptr.add(p);
                            let a1_val = *a1_ptr.add(p);
                            let a2_val = *a2_ptr.add(p);
                            let a3_val = *a3_ptr.add(p);

                            let b0_val = *b0_ptr.add(p);
                            let b1_val = *b1_ptr.add(p);
                            let b2_val = *b2_ptr.add(p);
                            let b3_val = *b3_ptr.add(p);

                            dot00 = a0_val.mul_add(b0_val, dot00);
                            dot01 = a0_val.mul_add(b1_val, dot01);
                            dot02 = a0_val.mul_add(b2_val, dot02);
                            dot03 = a0_val.mul_add(b3_val, dot03);

                            dot10 = a1_val.mul_add(b0_val, dot10);
                            dot11 = a1_val.mul_add(b1_val, dot11);
                            dot12 = a1_val.mul_add(b2_val, dot12);
                            dot13 = a1_val.mul_add(b3_val, dot13);

                            dot20 = a2_val.mul_add(b0_val, dot20);
                            dot21 = a2_val.mul_add(b1_val, dot21);
                            dot22 = a2_val.mul_add(b2_val, dot22);
                            dot23 = a2_val.mul_add(b3_val, dot23);

                            dot30 = a3_val.mul_add(b0_val, dot30);
                            dot31 = a3_val.mul_add(b1_val, dot31);
                            dot32 = a3_val.mul_add(b2_val, dot32);
                            dot33 = a3_val.mul_add(b3_val, dot33);

                            p += 1;
                        }

                        // scale and store back to C
                        *c0_ptr.add(i) = alpha.mul_add(dot00, *c0_ptr.add(i));
                        *c1_ptr.add(i) = alpha.mul_add(dot01, *c1_ptr.add(i));
                        *c2_ptr.add(i) = alpha.mul_add(dot02, *c2_ptr.add(i));
                        *c3_ptr.add(i) = alpha.mul_add(dot03, *c3_ptr.add(i));

                        *c0_ptr.add(i + 1) = alpha.mul_add(dot10, *c0_ptr.add(i + 1));
                        *c1_ptr.add(i + 1) = alpha.mul_add(dot11, *c1_ptr.add(i + 1));
                        *c2_ptr.add(i + 1) = alpha.mul_add(dot12, *c2_ptr.add(i + 1));
                        *c3_ptr.add(i + 1) = alpha.mul_add(dot13, *c3_ptr.add(i + 1));

                        *c0_ptr.add(i + 2) = alpha.mul_add(dot20, *c0_ptr.add(i + 2));
                        *c1_ptr.add(i + 2) = alpha.mul_add(dot21, *c1_ptr.add(i + 2));
                        *c2_ptr.add(i + 2) = alpha.mul_add(dot22, *c2_ptr.add(i + 2));
                        *c3_ptr.add(i + 2) = alpha.mul_add(dot23, *c3_ptr.add(i + 2));

                        *c0_ptr.add(i + 3) = alpha.mul_add(dot30, *c0_ptr.add(i + 3));
                        *c1_ptr.add(i + 3) = alpha.mul_add(dot31, *c1_ptr.add(i + 3));
                        *c2_ptr.add(i + 3) = alpha.mul_add(dot32, *c2_ptr.add(i + 3));
                        *c3_ptr.add(i + 3) = alpha.mul_add(dot33, *c3_ptr.add(i + 3));

                        i += 4;
                    }

                    // remainder M (1-3 rows)
                    while i < m {
                        let a_ptr_i = a_ptr.add(i * lda);
                        let mut sum0 = _mm256_setzero_ps();
                        let mut sum1 = _mm256_setzero_ps();
                        let mut sum2 = _mm256_setzero_ps();
                        let mut sum3 = _mm256_setzero_ps();

                        let mut p = 0usize;
                        while p + 7 < k {
                            let a0 = _mm256_loadu_ps(a_ptr_i.add(p));
                            sum0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0_ptr.add(p)), sum0);
                            sum1 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b1_ptr.add(p)), sum1);
                            sum2 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b2_ptr.add(p)), sum2);
                            sum3 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b3_ptr.add(p)), sum3);
                            p += 8;
                        }

                        let mut dot0 = reduce_f32(sum0);
                        let mut dot1 = reduce_f32(sum1);
                        let mut dot2 = reduce_f32(sum2);
                        let mut dot3 = reduce_f32(sum3);

                        while p < k {
                            let a_val = *a_ptr_i.add(p);
                            dot0 = a_val.mul_add(*b0_ptr.add(p), dot0);
                            dot1 = a_val.mul_add(*b1_ptr.add(p), dot1);
                            dot2 = a_val.mul_add(*b2_ptr.add(p), dot2);
                            dot3 = a_val.mul_add(*b3_ptr.add(p), dot3);
                            p += 1;
                        }

                        *c0_ptr.add(i) = alpha.mul_add(dot0, *c0_ptr.add(i));
                        *c1_ptr.add(i) = alpha.mul_add(dot1, *c1_ptr.add(i));
                        *c2_ptr.add(i) = alpha.mul_add(dot2, *c2_ptr.add(i));
                        *c3_ptr.add(i) = alpha.mul_add(dot3, *c3_ptr.add(i));

                        i += 1;
                    }
                    j += 4;
                }

                // remainder N (1-3 cols)
                while j < n {
                    let c0_ptr = c_ptr.add(j * ldc);
                    let b0_ptr = b_ptr.add(j * ldb);
                    let mut i = 0usize;

                    while i + 3 < m {
                        let a0_ptr = a_ptr.add(i * lda);
                        let a1_ptr = a_ptr.add((i + 1) * lda);
                        let a2_ptr = a_ptr.add((i + 2) * lda);
                        let a3_ptr = a_ptr.add((i + 3) * lda);

                        let mut sum0 = _mm256_setzero_ps();
                        let mut sum1 = _mm256_setzero_ps();
                        let mut sum2 = _mm256_setzero_ps();
                        let mut sum3 = _mm256_setzero_ps();

                        let mut p = 0usize;
                        while p + 7 < k {
                            let b0 = _mm256_loadu_ps(b0_ptr.add(p));
                            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0_ptr.add(p)), b0, sum0);
                            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1_ptr.add(p)), b0, sum1);
                            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2_ptr.add(p)), b0, sum2);
                            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3_ptr.add(p)), b0, sum3);
                            p += 8;
                        }

                        let mut dot0 = reduce_f32(sum0);
                        let mut dot1 = reduce_f32(sum1);
                        let mut dot2 = reduce_f32(sum2);
                        let mut dot3 = reduce_f32(sum3);

                        while p < k {
                            let b_val = *b0_ptr.add(p);
                            dot0 = b_val.mul_add(*a0_ptr.add(p), dot0);
                            dot1 = b_val.mul_add(*a1_ptr.add(p), dot1);
                            dot2 = b_val.mul_add(*a2_ptr.add(p), dot2);
                            dot3 = b_val.mul_add(*a3_ptr.add(p), dot3);
                            p += 1;
                        }

                        *c0_ptr.add(i) = alpha.mul_add(dot0, *c0_ptr.add(i));
                        *c0_ptr.add(i + 1) = alpha.mul_add(dot1, *c0_ptr.add(i + 1));
                        *c0_ptr.add(i + 2) = alpha.mul_add(dot2, *c0_ptr.add(i + 2));
                        *c0_ptr.add(i + 3) = alpha.mul_add(dot3, *c0_ptr.add(i + 3));

                        i += 4;
                    }

                    while i < m {
                        let a_ptr_i = a_ptr.add(i * lda);
                        let mut sum0 = _mm256_setzero_ps();
                        let mut p = 0usize;
                        while p + 7 < k {
                            sum0 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(a_ptr_i.add(p)),
                                _mm256_loadu_ps(b0_ptr.add(p)),
                                sum0,
                            );
                            p += 8;
                        }
                        let mut dot0 = reduce_f32(sum0);
                        while p < k {
                            dot0 = (*a_ptr_i.add(p)).mul_add(*b0_ptr.add(p), dot0);
                            p += 1;
                        }
                        *c0_ptr.add(i) = alpha.mul_add(dot0, *c0_ptr.add(i));
                        i += 1;
                    }
                    j += 1;
                }
            }
        },
    }
}

#[test]
fn gemm_test() {
    use crate::utils::Noise;
    use std::hint::black_box;
    use std::time::Instant;

    let warmup = 16;
    let runs = 24;
    let size = 1024;

    println!("size: {}", size);

    let mut noise = Noise::init();
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

    println!(
        "gemm_f_f: {:?} seconds, {:.2} GFLOPS",
        dur.as_secs_f64(),
        gflops
    );

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

    println!(
        "gemm_f_t: {:?} seconds, {:.2} GFLOPS",
        dur.as_secs_f64(),
        gflops
    );

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

    println!(
        "gemm_t_t: {:?} seconds, {:.2} GFLOPS",
        dur.as_secs_f64(),
        gflops
    );

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

    println!(
        "gemm_t_f: {:?} seconds, {:.2} GFLOPS",
        dur.as_secs_f64(),
        gflops
    );
}
