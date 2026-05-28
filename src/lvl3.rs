use crate::lvl1::{axpy, axpy_unsafe, dot, dot_unsafe, scal};
use std::slice::{from_raw_parts, from_raw_parts_mut};

// TODO: minimal branching, checks and (buffered) fn call overhead, clean doc

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
    // README: Go through `gemv` implementation in lvl2.rs before reading this

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

    // Properly scale C by beta
    for j in 0..n {
        let start = j * ldc;
        let col = unsafe { from_raw_parts_mut(c.as_mut_ptr().add(start), m) };
        scal(m, beta, col, 1);
    }

    if alpha == 0.0 {
        return;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    // Number of columns of B to process in one block
    let block_n = 160;
    // Number of rows of A to process in one block
    let block_m = 128;
    // Number of elements in the inner dimension to process in one block,
    let block_k = 64;

    match (is_trans_a, is_trans_b) {
        // C = alpha * A * B + beta * C
        // C = alpha * A * B^T + beta * C (B transposed)
        // we need JUST a scalar from b, either B[i][j] or B[j][i] and use do OUTER PRODUCT MATMUL
        // while A & C are column contiguous, so using axpy is the right choice here
        (false, _) => {
            // iter over block of col of B and C, then inner dim and finally block of row of A and C
            for j_b in (0..n).step_by(block_n) {
                let j_max = (j_b + block_n).min(n); // <- max col of B and C to process in this block
                for k_b in (0..k).step_by(block_k) {
                    let k_max = (k_b + block_k).min(k); // shared dim, max col of A and row of B to process in this block
                    for i_b in (0..m).step_by(block_m) {
                        let i_max = (i_b + block_m).min(m); // <- max row of A and C to process in this block
                        // num of element in col of A and C to process in this block
                        let curr_e = i_max - i_b;
                        // iter over col of B and C in block,
                        // for each col, get col_buf of C to update in INNER loop (cached!!)
                        for j in j_b..j_max {
                            // MATRIX C: row (i_b..i_max) + col (j) * ldc
                            let c_idx = i_b + (j * ldc); // <- column index, c0, c1, ... c(n-1)
                            let c_col_ptr = unsafe { c_ptr.add(c_idx) };
                            // get mut col_buf by reading `curr_e` i.e no of current element in that rows
                            let c_col_buf = unsafe { from_raw_parts_mut(c_col_ptr, curr_e) };

                            // iter over shared dim in block, for each col in C
                            // for each element, scale by alpha and perform `axpy` on col of A and C
                            for kk in k_b..k_max {
                                // grab each element in this col and scale
                                let b_idx = if is_trans_b {
                                    j + (kk * ldb) // element B[j][k] in n×k storage (transposed)
                                } else {
                                    kk + (j * ldb) // element B[k][j] in k×n storage (not transposed)
                                };
                                let b_val = unsafe { *b_ptr.add(b_idx) };

                                // save compute, if zero
                                if b_val != 0.0 {
                                    let scale_b = b_val * alpha;

                                    // grab an entire col buf, using k since we are moving down in THAT BLOCK
                                    let a_idx = i_b + (kk * lda); // points to starting A[i_b][k], stride by `lda` since we are moving down in col of A
                                    let a_col_ptr = unsafe { a_ptr.add(a_idx) };
                                    let a_col_buf = unsafe { from_raw_parts(a_col_ptr, curr_e) }; // <-- MATRIX A: row (i_b..i_max) + col (k) * lda

                                    // `axpy` C_col = C_col + (scaled_b * A_col)
                                    unsafe {
                                        axpy_unsafe(curr_e, scale_b, a_col_buf, 1, c_col_buf, 1); // <- inc* are 1, since cols of A & C are contiguous no matter
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // C = alpha * A^T * B + beta * C (A transposed)
        // compute dot between a row of A^T (contiguous) and a column of B, (INNER PRODUCT)
        // C = alpha * A^T * B^T + beta * C (both transposed) (OUTER PRODUCT)
        // well, dot can use here with stride, using col of A^T with stride, and a scalar element from B (INNER PRODUCT)
        (true, _) => {
            for j_b in (0..n).step_by(block_n) {
                let j_max = (j_b + block_n).min(n);
                for k_b in (0..k).step_by(block_k) {
                    let k_max = (k_b + block_k).min(k);
                    for i_b in (0..m).step_by(block_m) {
                        let i_max = (i_b + block_m).min(m);
                        let curr_r = i_max - i_b;

                        // iter over all column of B & C in block
                        for j in j_b..j_max {
                            // C [i_b][j], since we are moving down in col of C, and col is j, so row is i_b
                            let c_idx = i_b + (j * ldc); // <- column index, c0, c1, ... c(n-1)
                            let c_col_ptr = unsafe { c_ptr.add(c_idx) };

                            // for each row r in i_b block,
                            // we compute partial dot over shared dim (k_b...k_max), and update c[r][j]
                            // compute buf len is same for both buf, since its inner dim, and for dot
                            let buf_len = k_max - k_b;
                            for r_off in 0..curr_r {
                                // absolute row index = start row index of the block + local offset (0...curr_r)
                                let i = i_b + r_off;

                                if !is_trans_b {
                                    // get a row from A in this BLOCK, since A is transposed, we are moving down in row,
                                    // reading ahead from starting index of col i.e. first row index, gives us entire row_buf in current BLOCK
                                    let a_idx = k_b + (i * lda); // <- points to starting element of col
                                    let a_row_ptr = unsafe { a_ptr.add(a_idx) }; // pointer to the start of row r within this k-block for A^T
                                    let a_row_buf = unsafe { from_raw_parts(a_row_ptr, buf_len) };

                                    // B stored as k×n (not transposed), need column j of B (contiguous)
                                    let b_idx = k_b + (j * ldb); // element B[k][j] in k×n storage (not transposed)
                                    let b_col_ptr = unsafe { b_ptr.add(b_idx) }; // pointer to the start of column j within this k-block for B
                                    let b_col_buf = unsafe { from_raw_parts(b_col_ptr, buf_len) };

                                    // compute & apply, using `mul_add` FMA
                                    unsafe {
                                        let partial =
                                            dot_unsafe(buf_len, a_row_buf, 1, b_col_buf, 1);
                                        *c_col_ptr.add(r_off) =
                                            alpha.mul_add(partial, *c_col_ptr.add(r_off)); // use rdx because c_col_ptr already points at row i_b
                                    }
                                } else {
                                    // Row i of A^T = column i of stored A (contiguous)
                                    let a_idx = k_b + (i * lda);
                                    let a_row_buf =
                                        unsafe { from_raw_parts(a_ptr.add(a_idx), buf_len) };

                                    // B stored as n×k (transposed), need row j of stored B (strided by ldb)
                                    let b_start = j + (k_b * ldb);
                                    let b_buf = unsafe {
                                        from_raw_parts(b_ptr.add(b_start), (buf_len - 1) * ldb + 1)
                                    };

                                    unsafe {
                                        // `dot` that son of a B...
                                        let partial =
                                            dot_unsafe(buf_len, a_row_buf, 1, b_buf, ldb as i32); // <- include stride for col aka, y_buf
                                        *c_col_ptr.add(r_off) =
                                            alpha.mul_add(partial, *c_col_ptr.add(r_off));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// The gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemm.html)
pub fn gemm_native(
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

    // Scale C by beta
    for j in 0..n {
        let start = j * ldc;
        let col = unsafe { from_raw_parts_mut(c.as_mut_ptr().add(start), m) };
        scal(m, beta, col, 1);
    }

    if alpha == 0.0 {
        return;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    match (is_trans_a, is_trans_b) {
        // Case 1 & 2: A is NOT transposed
        // Outer Product / Rank-1 Update approach.
        // We iterate over the inner dimension k, and for each slice,
        // perform an AXPY operation C_col += (alpha * B_val) * A_col
        (false, _) => {
            for j in 0..n {
                // Pointer to the start of column j in C
                let c_col_ptr = unsafe { c_ptr.add(j * ldc) };
                // Mutable slice for the entire column j (length m)
                let c_col = unsafe { from_raw_parts_mut(c_col_ptr, m) };

                for p in 0..k {
                    // Determine the scalar from B(p, j)
                    let b_val = if is_trans_b {
                        // B stored as n×k. We want B(p, j).
                        // Logical B is k×n. Stored B^T is n×k.
                        // Element is at row j, col p.
                        unsafe { *b_ptr.add(j + p * ldb) }
                    } else {
                        // B stored as k×n. We want B(p, j).
                        // Element is at row p, col j.
                        unsafe { *b_ptr.add(p + j * ldb) }
                    };

                    // Skip zero values
                    if b_val != 0.0 {
                        let scale = alpha * b_val;

                        // Pointer to column p of A (A is m×k, not transposed)
                        let a_col_ptr = unsafe { a_ptr.add(p * lda) };
                        let a_col = unsafe { from_raw_parts(a_col_ptr, m) };

                        // `axpy` that bitch
                        // C[:, j] += scale * A[:, p]
                        axpy(m, scale, a_col, 1, c_col, 1);
                    }
                }
            }
        }

        // Case 3 & 4: A IS transposed
        // Inner Product / Dot Product approach.
        // we compute each element C(i, j) individually using DOT.
        // C(i, j) = A_col_i . B_vec_j
        (true, _) => {
            for j in 0..n {
                // Pointer to column j of C
                let c_col_ptr = unsafe { c_ptr.add(j * ldc) };
                let c_col = unsafe { from_raw_parts_mut(c_col_ptr, m) };

                // prepare the vector from B (length k)
                // we need either a column of B or a row of B (strided)
                let (b_vec_ptr, b_stride, b_mem_len) = if is_trans_b {
                    // B stored as n×k. op(B) is B^T (k×n).
                    // we need col j of op(B) -> Row j of stored B, strided access.
                    (
                        unsafe { b_ptr.add(j) },
                        ldb as i32,
                        // Memory span required for the buffer,
                        // from first element at b_ptr + j to last element at b_ptr + j + (k-1)*ldb
                        if k == 0 { 0 } else { (k - 1) * ldb + 1 },
                    )
                } else {
                    // B stored as k×n. op(B) is B.
                    // we need col j of B, contiguous access.
                    (unsafe { b_ptr.add(j * ldb) }, 1, k)
                };

                // get buf for B vector covering the memory range
                let b_vec = unsafe { from_raw_parts(b_vec_ptr, b_mem_len) };

                for (i, col) in c_col.iter_mut().enumerate().take(m) {
                    // A is stored as k×m. op(A) is A^T (m×k).
                    // we need row i of op(A) -> Col i of stored A, contiguous access.
                    let a_col_ptr = unsafe { a_ptr.add(i * lda) };
                    let a_col = unsafe { from_raw_parts(a_col_ptr, k) };

                    // `dot` that bitch
                    // compute dot product A[:, i] . B[:][j]
                    let prod = dot(k, a_col, 1, b_vec, b_stride);

                    // accumulate C[i, j] += alpha * prod
                    *col += alpha * prod;
                }
            }
        }
    }
}

#[test]
fn gemm_native_test() {
    use crate::utils::gen_fill;
    use rayon::join;
    use std::hint::black_box;
    use std::time::Instant;

    let size = 2048;
    let runs = 32;
    let warmup = 12;

    let mut a = vec![0.0f32; size * size];
    let mut b = vec![0.0f32; size * size];
    let mut c1 = vec![0.0f32; size * size];
    let mut c2 = vec![0.0f32; size * size];

    black_box(&mut a);
    black_box(&mut b);
    black_box(&mut c1);
    black_box(&mut c2);

    gen_fill(&mut a);
    gen_fill(&mut b);

    for _ in 0..warmup {
        c1.fill(1.0);
        c2.fill(1.0);
        join(
            || {
                gemm(
                    size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c1, size, false, false,
                )
            },
            || {
                gemm(
                    size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c2, size, false, false,
                )
            },
        );
    }

    assert!(
        c1.iter()
            .zip(c2.iter())
            .all(|(&x, &y)| (x - y).abs() < 1e-3)
    );

    c1.fill(1.0);
    c2.fill(1.0);

    let (dur_native, dur_opt) = join(
        || {
            let start = Instant::now();
            for _ in 0..runs {
                gemm(
                    size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c1, size, false, false,
                );
            }
            start.elapsed()
        },
        || {
            let start = Instant::now();
            for _ in 0..runs {
                gemm_native(
                    size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c2, size, false, false,
                );
            }
            start.elapsed()
        },
    );

    let total_flops = 2.0 * (size as f64).powi(3) * runs as f64;
    let gflops_native = total_flops / dur_native.as_secs_f64() / 1e9;
    let gflops_opt = total_flops / dur_opt.as_secs_f64() / 1e9;

    println!(
        "gemm_native: {:?} seconds, {:.2} GFLOPS",
        dur_native.as_secs_f64(),
        gflops_native
    );
    println!(
        "gemm_opt: {:?} seconds, {:.2} GFLOPS",
        dur_opt.as_secs_f64(),
        gflops_opt
    );
}
