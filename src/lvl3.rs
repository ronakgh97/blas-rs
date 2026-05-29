use crate::lvl1::scal;
use crate::utils::from_m256;
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
use std::slice::from_raw_parts_mut;

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
        // C = alpha * A * B + beta * C; C = alpha * A * B^T + beta * C
        false => unsafe {
            let mut j = 0usize;
            // iter over 4 col at a time
            while j + 3 < n {
                let c0_ptr = c_ptr.add(j * ldc);
                let c1_ptr = c_ptr.add((j + 1) * ldc);
                let c2_ptr = c_ptr.add((j + 2) * ldc);
                let c3_ptr = c_ptr.add((j + 3) * ldc);

                let mut p = 0usize;
                // inner k dim loop
                while p < k {
                    let a_col_ptr = a_ptr.add(p * lda);

                    // load scalar from B (strided if) and broadcast to vector
                    let b0 = if is_trans_b {
                        *b_ptr.add(j + p * ldb)
                    } else {
                        *b_ptr.add(p + j * ldb)
                    };
                    let b1 = if is_trans_b {
                        *b_ptr.add(j + 1 + p * ldb)
                    } else {
                        *b_ptr.add(p + (j + 1) * ldb)
                    };
                    let b2 = if is_trans_b {
                        *b_ptr.add(j + 2 + p * ldb)
                    } else {
                        *b_ptr.add(p + (j + 2) * ldb)
                    };
                    let b3 = if is_trans_b {
                        *b_ptr.add(j + 3 + p * ldb)
                    } else {
                        *b_ptr.add(p + (j + 3) * ldb)
                    };

                    // alpha scale
                    let scale0 = alpha * b0;
                    let scale1 = alpha * b1;
                    let scale2 = alpha * b2;
                    let scale3 = alpha * b3;

                    // 8 reg of B
                    let bx0 = _mm256_set1_ps(scale0);
                    let bx1 = _mm256_set1_ps(scale1);
                    let bx2 = _mm256_set1_ps(scale2);
                    let bx3 = _mm256_set1_ps(scale3);

                    let mut i = 0usize;
                    while i + 16 <= m {
                        // load two avx from col p of A
                        let a0 = _mm256_loadu_ps(a_col_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_col_ptr.add(i + 8));

                        // load 16 for each 4 cols
                        let c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        let c01 = _mm256_loadu_ps(c0_ptr.add(i + 8));

                        let c10 = _mm256_loadu_ps(c1_ptr.add(i));
                        let c11 = _mm256_loadu_ps(c1_ptr.add(i + 8));

                        let c20 = _mm256_loadu_ps(c2_ptr.add(i));
                        let c21 = _mm256_loadu_ps(c2_ptr.add(i + 8));

                        let c30 = _mm256_loadu_ps(c3_ptr.add(i));
                        let c31 = _mm256_loadu_ps(c3_ptr.add(i + 8));

                        // fmadd; c_col <= a_col * b_scalar
                        let c00 = _mm256_fmadd_ps(bx0, a0, c00);
                        let c01 = _mm256_fmadd_ps(bx0, a1, c01);
                        let c10 = _mm256_fmadd_ps(bx1, a0, c10);
                        let c11 = _mm256_fmadd_ps(bx1, a1, c11);
                        let c20 = _mm256_fmadd_ps(bx2, a0, c20);
                        let c21 = _mm256_fmadd_ps(bx2, a1, c21);
                        let c30 = _mm256_fmadd_ps(bx3, a0, c30);
                        let c31 = _mm256_fmadd_ps(bx3, a1, c31);

                        // write back
                        _mm256_storeu_ps(c0_ptr.add(i), c00);
                        _mm256_storeu_ps(c0_ptr.add(i + 8), c01);
                        _mm256_storeu_ps(c1_ptr.add(i), c10);
                        _mm256_storeu_ps(c1_ptr.add(i + 8), c11);
                        _mm256_storeu_ps(c2_ptr.add(i), c20);
                        _mm256_storeu_ps(c2_ptr.add(i + 8), c21);
                        _mm256_storeu_ps(c3_ptr.add(i), c30);
                        _mm256_storeu_ps(c3_ptr.add(i + 8), c31);

                        i += 16; // step by
                    }

                    // remaining (8x4)
                    while i + 8 <= m {
                        let a0 = _mm256_loadu_ps(a_col_ptr.add(i));

                        let c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        let c10 = _mm256_loadu_ps(c1_ptr.add(i));
                        let c20 = _mm256_loadu_ps(c2_ptr.add(i));
                        let c30 = _mm256_loadu_ps(c3_ptr.add(i));

                        _mm256_storeu_ps(c0_ptr.add(i), _mm256_fmadd_ps(bx0, a0, c00));
                        _mm256_storeu_ps(c1_ptr.add(i), _mm256_fmadd_ps(bx1, a0, c10));
                        _mm256_storeu_ps(c2_ptr.add(i), _mm256_fmadd_ps(bx2, a0, c20));
                        _mm256_storeu_ps(c3_ptr.add(i), _mm256_fmadd_ps(bx3, a0, c30));

                        i += 8;
                    }

                    // leftovers (1-7)
                    while i < m {
                        let a_val = *a_col_ptr.add(i);
                        *c0_ptr.add(i) = scale0.mul_add(a_val, *c0_ptr.add(i));
                        *c1_ptr.add(i) = scale1.mul_add(a_val, *c1_ptr.add(i));
                        *c2_ptr.add(i) = scale2.mul_add(a_val, *c2_ptr.add(i));
                        *c3_ptr.add(i) = scale3.mul_add(a_val, *c3_ptr.add(i));
                        i += 1;
                    }

                    p += 1;
                }

                j += 4;
            }

            // if n does not fit
            if j + 1 < n {
                let c0_ptr = c_ptr.add(j * ldc);
                let c1_ptr = c_ptr.add((j + 1) * ldc);
                let mut p = 0usize;
                while p < k {
                    let a_col_ptr = a_ptr.add(p * lda);

                    let b0 = if is_trans_b {
                        *b_ptr.add(j + p * ldb)
                    } else {
                        *b_ptr.add(p + j * ldb)
                    };
                    let b1 = if is_trans_b {
                        *b_ptr.add(j + 1 + p * ldb)
                    } else {
                        *b_ptr.add(p + (j + 1) * ldb)
                    };

                    let scale0 = alpha * b0;
                    let scale1 = alpha * b1;
                    let bx0 = _mm256_set1_ps(scale0);
                    let bx1 = _mm256_set1_ps(scale1);

                    let mut i = 0usize;
                    // load four vectors from A and C, compute and store back (32x2)
                    while i + 32 <= m {
                        let a0 = _mm256_loadu_ps(a_col_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_col_ptr.add(i + 8));
                        let a2 = _mm256_loadu_ps(a_col_ptr.add(i + 16));
                        let a3 = _mm256_loadu_ps(a_col_ptr.add(i + 24));

                        let c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        let c01 = _mm256_loadu_ps(c0_ptr.add(i + 8));
                        let c02 = _mm256_loadu_ps(c0_ptr.add(i + 16));
                        let c03 = _mm256_loadu_ps(c0_ptr.add(i + 24));

                        let c10 = _mm256_loadu_ps(c1_ptr.add(i));
                        let c11 = _mm256_loadu_ps(c1_ptr.add(i + 8));
                        let c12 = _mm256_loadu_ps(c1_ptr.add(i + 16));
                        let c13 = _mm256_loadu_ps(c1_ptr.add(i + 24));

                        let c00 = _mm256_fmadd_ps(bx0, a0, c00);
                        let c01 = _mm256_fmadd_ps(bx0, a1, c01);
                        let c02 = _mm256_fmadd_ps(bx0, a2, c02);
                        let c03 = _mm256_fmadd_ps(bx0, a3, c03);

                        let c10 = _mm256_fmadd_ps(bx1, a0, c10);
                        let c11 = _mm256_fmadd_ps(bx1, a1, c11);
                        let c12 = _mm256_fmadd_ps(bx1, a2, c12);
                        let c13 = _mm256_fmadd_ps(bx1, a3, c13);

                        _mm256_storeu_ps(c0_ptr.add(i), c00);
                        _mm256_storeu_ps(c0_ptr.add(i + 8), c01);
                        _mm256_storeu_ps(c0_ptr.add(i + 16), c02);
                        _mm256_storeu_ps(c0_ptr.add(i + 24), c03);

                        _mm256_storeu_ps(c1_ptr.add(i), c10);
                        _mm256_storeu_ps(c1_ptr.add(i + 8), c11);
                        _mm256_storeu_ps(c1_ptr.add(i + 16), c12);
                        _mm256_storeu_ps(c1_ptr.add(i + 24), c13);

                        i += 32;
                    }

                    // fallback
                    while i + 8 <= m {
                        let a0 = _mm256_loadu_ps(a_col_ptr.add(i));
                        let c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        let c10 = _mm256_loadu_ps(c1_ptr.add(i));

                        _mm256_storeu_ps(c0_ptr.add(i), _mm256_fmadd_ps(bx0, a0, c00));
                        _mm256_storeu_ps(c1_ptr.add(i), _mm256_fmadd_ps(bx1, a0, c10));

                        i += 8;
                    }

                    // cleanup
                    while i < m {
                        let a_val = *a_col_ptr.add(i);
                        *c0_ptr.add(i) = scale0.mul_add(a_val, *c0_ptr.add(i));
                        *c1_ptr.add(i) = scale1.mul_add(a_val, *c1_ptr.add(i));
                        i += 1;
                    }

                    p += 1;
                }

                j += 2;
            }

            // last fallback
            if j < n {
                let c0_ptr = c_ptr.add(j * ldc);
                let mut p = 0usize;
                while p < k {
                    let a_col_ptr = a_ptr.add(p * lda);

                    let b0 = if is_trans_b {
                        *b_ptr.add(j + p * ldb)
                    } else {
                        *b_ptr.add(p + j * ldb)
                    };

                    let scale0 = alpha * b0;
                    let bx0 = _mm256_set1_ps(scale0);

                    let mut i = 0usize;
                    // load four vectors from A and C, compute and store back (32x1)
                    while i + 32 <= m {
                        let a0 = _mm256_loadu_ps(a_col_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_col_ptr.add(i + 8));
                        let a2 = _mm256_loadu_ps(a_col_ptr.add(i + 16));
                        let a3 = _mm256_loadu_ps(a_col_ptr.add(i + 24));

                        let c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        let c01 = _mm256_loadu_ps(c0_ptr.add(i + 8));
                        let c02 = _mm256_loadu_ps(c0_ptr.add(i + 16));
                        let c03 = _mm256_loadu_ps(c0_ptr.add(i + 24));

                        _mm256_storeu_ps(c0_ptr.add(i), _mm256_fmadd_ps(bx0, a0, c00));
                        _mm256_storeu_ps(c0_ptr.add(i + 8), _mm256_fmadd_ps(bx0, a1, c01));
                        _mm256_storeu_ps(c0_ptr.add(i + 16), _mm256_fmadd_ps(bx0, a2, c02));
                        _mm256_storeu_ps(c0_ptr.add(i + 24), _mm256_fmadd_ps(bx0, a3, c03));

                        i += 32;
                    }

                    // if doesnt fit
                    while i + 8 <= m {
                        let a0 = _mm256_loadu_ps(a_col_ptr.add(i));
                        let c00 = _mm256_loadu_ps(c0_ptr.add(i));
                        _mm256_storeu_ps(c0_ptr.add(i), _mm256_fmadd_ps(bx0, a0, c00));
                        i += 8;
                    }

                    // scalar fallback
                    while i < m {
                        let a_val = *a_col_ptr.add(i);
                        *c0_ptr.add(i) = scale0.mul_add(a_val, *c0_ptr.add(i));
                        i += 1;
                    }

                    p += 1;
                }
            }
        },
        // C = alpha * A^T * B + beta * C; C = alpha * A^T * B^T + beta * C
        true => unsafe {
            if is_trans_b {
                // TODO; this branch is bottleneck!!!, find a way to vectorized
                for j in 0..n {
                    let c0_ptr = c_ptr.add(j * ldc);
                    for i in 0..m {
                        let a_col_ptr = a_ptr.add(i * lda);
                        let mut sum0 = 0.0f32;
                        let mut sum1 = 0.0f32;
                        let mut p = 0usize;
                        while p + 1 < k {
                            let b0 = *b_ptr.add(j + p * ldb);
                            let b1 = *b_ptr.add(j + (p + 1) * ldb);
                            let a0 = *a_col_ptr.add(p);
                            let a1 = *a_col_ptr.add(p + 1);
                            sum0 = a0.mul_add(b0, sum0);
                            sum1 = a1.mul_add(b1, sum1);
                            p += 2;
                        }
                        if p < k {
                            let b0 = *b_ptr.add(j + p * ldb);
                            let a0 = *a_col_ptr.add(p);
                            sum0 = a0.mul_add(b0, sum0);
                        }
                        let dot0 = sum0 + sum1;
                        *c0_ptr.add(i) = alpha.mul_add(dot0, *c0_ptr.add(i));
                    }
                }
            } else {
                let mut j = 0usize;
                // process 2 cols of C
                while j + 1 < n {
                    let c0_ptr = c_ptr.add(j * ldc);
                    let c1_ptr = c_ptr.add((j + 1) * ldc);
                    let b0_ptr = b_ptr.add(j * ldb);
                    let b1_ptr = b_ptr.add((j + 1) * ldb);

                    for i in 0..m {
                        let a_col_ptr = a_ptr.add(i * lda);

                        // 8 a and 8 b vectors to accumulate into 16 dot products for 2 cols
                        let mut sum0_0 = _mm256_setzero_ps();
                        let mut sum0_1 = _mm256_setzero_ps();
                        let mut sum0_2 = _mm256_setzero_ps();
                        let mut sum0_3 = _mm256_setzero_ps();

                        let mut sum1_0 = _mm256_setzero_ps();
                        let mut sum1_1 = _mm256_setzero_ps();
                        let mut sum1_2 = _mm256_setzero_ps();
                        let mut sum1_3 = _mm256_setzero_ps();

                        let mut p = 0usize;
                        // process inner k dim in 4 avx
                        while p + 32 <= k {
                            // load a^t (contiguous case)
                            let a0 = _mm256_loadu_ps(a_col_ptr.add(p));
                            let a1 = _mm256_loadu_ps(a_col_ptr.add(p + 8));
                            let a2 = _mm256_loadu_ps(a_col_ptr.add(p + 16));
                            let a3 = _mm256_loadu_ps(a_col_ptr.add(p + 24));

                            // load b; 4 vectors for each col

                            // col1
                            let b00 = _mm256_loadu_ps(b0_ptr.add(p));
                            let b01 = _mm256_loadu_ps(b0_ptr.add(p + 8));
                            let b02 = _mm256_loadu_ps(b0_ptr.add(p + 16));
                            let b03 = _mm256_loadu_ps(b0_ptr.add(p + 24));

                            // col2
                            let b10 = _mm256_loadu_ps(b1_ptr.add(p));
                            let b11 = _mm256_loadu_ps(b1_ptr.add(p + 8));
                            let b12 = _mm256_loadu_ps(b1_ptr.add(p + 16));
                            let b13 = _mm256_loadu_ps(b1_ptr.add(p + 24));

                            // accumulate fma 8 of them, independently
                            sum0_0 = _mm256_fmadd_ps(a0, b00, sum0_0);
                            sum0_1 = _mm256_fmadd_ps(a1, b01, sum0_1);
                            sum0_2 = _mm256_fmadd_ps(a2, b02, sum0_2);
                            sum0_3 = _mm256_fmadd_ps(a3, b03, sum0_3);

                            sum1_0 = _mm256_fmadd_ps(a0, b10, sum1_0);
                            sum1_1 = _mm256_fmadd_ps(a1, b11, sum1_1);
                            sum1_2 = _mm256_fmadd_ps(a2, b12, sum1_2);
                            sum1_3 = _mm256_fmadd_ps(a3, b13, sum1_3);

                            p += 32;
                        }

                        // fallback
                        while p + 8 <= k {
                            let a0 = _mm256_loadu_ps(a_col_ptr.add(p));
                            sum0_0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0_ptr.add(p)), sum0_0);
                            sum1_0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b1_ptr.add(p)), sum1_0);
                            p += 8;
                        }

                        // reduction
                        let sum0 = _mm256_add_ps(
                            _mm256_add_ps(sum0_0, sum0_1),
                            _mm256_add_ps(sum0_2, sum0_3),
                        );
                        let sum1 = _mm256_add_ps(
                            _mm256_add_ps(sum1_0, sum1_1),
                            _mm256_add_ps(sum1_2, sum1_3),
                        );

                        let mut dot0 = from_m256(sum0);
                        let mut dot1 = from_m256(sum1);

                        // scalar fallback
                        while p < k {
                            let a_val = *a_col_ptr.add(p);
                            dot0 = a_val.mul_add(*b0_ptr.add(p), dot0);
                            dot1 = a_val.mul_add(*b1_ptr.add(p), dot1);
                            p += 1;
                        }

                        *c0_ptr.add(i) = alpha.mul_add(dot0, *c0_ptr.add(i));
                        *c1_ptr.add(i) = alpha.mul_add(dot1, *c1_ptr.add(i));
                    }

                    j += 2;
                }

                // final process leftover columns; add further similar computations
                if j < n {
                    let c0_ptr = c_ptr.add(j * ldc);
                    let b0_ptr = b_ptr.add(j * ldb);
                    for i in 0..m {
                        let a_col_ptr = a_ptr.add(i * lda);

                        let mut sum0 = _mm256_setzero_ps();
                        let mut sum1 = _mm256_setzero_ps();
                        let mut sum2 = _mm256_setzero_ps();
                        let mut sum3 = _mm256_setzero_ps();

                        let mut p = 0usize;
                        while p + 32 <= k {
                            let a0 = _mm256_loadu_ps(a_col_ptr.add(p));
                            let a1 = _mm256_loadu_ps(a_col_ptr.add(p + 8));
                            let a2 = _mm256_loadu_ps(a_col_ptr.add(p + 16));
                            let a3 = _mm256_loadu_ps(a_col_ptr.add(p + 24));

                            let b0 = _mm256_loadu_ps(b0_ptr.add(p));
                            let b1 = _mm256_loadu_ps(b0_ptr.add(p + 8));
                            let b2 = _mm256_loadu_ps(b0_ptr.add(p + 16));
                            let b3 = _mm256_loadu_ps(b0_ptr.add(p + 24));

                            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
                            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
                            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
                            sum3 = _mm256_fmadd_ps(a3, b3, sum3);

                            p += 32;
                        }

                        while p + 8 <= k {
                            let a0 = _mm256_loadu_ps(a_col_ptr.add(p));
                            sum0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0_ptr.add(p)), sum0);
                            p += 8;
                        }

                        let sum01 = _mm256_add_ps(sum0, sum1);
                        let sum23 = _mm256_add_ps(sum2, sum3);
                        let mut dot0 = from_m256(_mm256_add_ps(sum01, sum23));

                        while p < k {
                            let a_val = *a_col_ptr.add(p);
                            dot0 = a_val.mul_add(*b0_ptr.add(p), dot0);
                            p += 1;
                        }

                        *c0_ptr.add(i) = alpha.mul_add(dot0, *c0_ptr.add(i));
                    }
                }
            }
        },
    }
}

#[test]
fn gemm_test() {
    use crate::utils::gen_fill;
    use std::hint::black_box;
    use std::time::Instant;

    let warmup = 12;
    let runs = 24;
    let size = 1024;

    println!("size: {}", size);

    let mut a = vec![0.0f32; size * size];
    let mut b = vec![0.0f32; size * size];
    let mut c = vec![0.0f32; size * size];

    black_box(&mut a);
    black_box(&mut b);
    black_box(&mut c);

    gen_fill(&mut a);
    gen_fill(&mut b);
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

    gen_fill(&mut a);
    gen_fill(&mut b);
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

    gen_fill(&mut a);
    gen_fill(&mut b);
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

    gen_fill(&mut a);
    gen_fill(&mut b);
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

    gen_fill(&mut a);
    gen_fill(&mut b);
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
