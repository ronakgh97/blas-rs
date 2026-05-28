use crate::lvl1::{axpy, axpy_unsafe, dot, dot_unsafe, scal};
use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
use std::slice::{from_raw_parts, from_raw_parts_mut};

// TODO: the order of checks can be improved, if done smartly to avoid unnecessary compute
// TODO: minimal branching, checks and (buffered) fn call overhead, clean doc

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// The gemv routines compute a scalar-matrix-vector product and add the result to a scalar-vector product, with a general matrix.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemv.html)
pub fn gemv(
    m: usize,   // row of mat
    n: usize,   // col of mat
    alpha: f32, // scaling for product
    a: &[f32],  // matrix buf
    lda: usize, // leading dim of a, row or col major depends, we follow `column major`
    x: &[f32],  // mul vector buf
    incx: i32,
    beta: f32,     // y scaling
    y: &mut [f32], // resultant vec_buf
    incy: i32,
    is_trans: bool, // Fortran uses 'T', 'N', but we will use bool
) {
    // get the effective dimensions of x and y based on transposition
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

    // we use `scal` to handle the beta scaling and zeroing out y if beta is 0, as per BLAS spec
    scal(y_len, beta, y, incy);

    // y is already scaled with beta, we are done here, no need to do any mul computation
    // also returning here means `gemv` just became `scal` :0
    if alpha == 0.0 {
        return;
    }

    // How many columns to process before we proceed to next block
    const COL_BLOCK: usize = 128;

    // How many rows to proceed at once
    const ROW_BLOCK: usize = 64;

    // We have taken `!trans` because, default is column major,
    // so for simd we need contigous memory, and this is fine place to use `axpy` from lvl1.
    // In this case, we pin/reuse Y cache, because y is getting written and loaded multiple times,
    if !is_trans {
        // this is typical base stride logic, for Blas
        // we know if -ve we start from highest index, logically the first element, and we go back, visa verse for +ve
        let ix_b = if incx < 0 {
            (1 - n as isize) * incx as isize
        } else {
            0
        };
        let iy_b = if incy < 0 {
            (1 - m as isize) * incy as isize
        } else {
            0
        };

        // Step through rows in chunks of `row_block`,
        // we will process a block of cols for each row block,
        // so that we can reuse the data in cache, and also apply simd on that block of rows
        for row in (0..m).step_by(ROW_BLOCK) {
            let row_end = (row + ROW_BLOCK).min(m); // <- handle last block (same reason)
            let curr_m = row_end - row; // <- current element in the row block, we compute on this many element only for this block

            // Calculate exact memory bounds for y_buf and get local len of the y_buf for this block,
            // since we are processing `curr_m` rows, and each row is `incy` apart,
            // so we need to account for that in the length calculation, and also `axpy` need this, so check it out first
            let y_stride = incy.unsigned_abs() as usize;
            let y_buf_len = 1 + (curr_m - 1) * y_stride;

            // Find the starting address for this chunk
            // since `from_raw_parts` points to first element, reads in increasing order,
            // for -ve, because logically in `axpy` we are going backward,
            // so for getting the slice, we need lowest addr (i.e. row + curr_m -1 is the last in this block), so we can `offset` and `from_raw_parts_mut` reads ahead
            // in way that we cover the entire slice, and can safely send to `axpy`, rest striding, inc are handled by it
            // e.g: axpy needs this [-4, -3, -2, -1, 0], not [0, -2, -4],
            // for +ve, we take first logical element/lowest addr and move forward up the memory,
            // for -ve, we take ALSO take lowest addr, (logically last) and move forward, or rather we OFFSET!!
            // final YAP: we are pointing to addr of first logical element, WHICH IS ALSO THE LAST, MAN FUCK YOU IF YOU DON'T UNDERSTAND ...*sighh*
            let y_addr = if incy < 0 {
                iy_b + (row as isize + curr_m as isize - 1) * incy as isize
            } else {
                iy_b + (row as isize * incy as isize)
            };

            // Experiment
            unsafe {
                _mm_prefetch(
                    y.as_ptr().add((row + 2 * ROW_BLOCK).min(y.len())) as *const i8,
                    _MM_HINT_ET0,
                );
            }

            // Get mut buf from y for this "fixed" row chunk, this stay in cache
            let y_ptr = unsafe { y.as_mut_ptr().offset(y_addr) };
            let y_buf = unsafe { from_raw_parts_mut(y_ptr, y_buf_len) };

            // Step through columns in chunks of `col_block`
            for col in (0..n).step_by(COL_BLOCK) {
                let col_end = (col + COL_BLOCK).min(n); // <- handle last block which might be smaller than col_block

                unsafe {
                    // iter over all element in col 'tile'
                    for idx in col..col_end {
                        // Get base pointer, we wil move it down the column for each iteration to row r,
                        // using lda is the leading dimension (number of elements to skip to get to the next column)
                        let col_ptr = a.as_ptr().add(idx * lda + row); // <- e.g. c1, c2, c3...to end title
                        let col_buf = from_raw_parts(col_ptr, curr_m); // every col has "m" elements, and they are contiguous in memory, so we can create a slice buf for it

                        // Get a scalar buf from x for this column (idx'th)
                        let ix = ix_b + (idx as isize * incx as isize);
                        let x_val = *x.as_ptr().offset(ix);

                        // Compute this chunk
                        // Partial result for this column,
                        // res_buf <- |alpha * col[0] * x[0]| + ... + |alpha * col[n] * x[n]|.
                        axpy_unsafe(curr_m, alpha * x_val, col_buf, 1, y_buf, incy); // <-- we put incx as 1 because x which is col is contiguous, go through cmt if you down get it
                    }
                }
            }
        }
    } else {
        // SYMMETRY!!!
        // ok now it transposed, same logic but opposite behavior, we have "row-major" layout and since its row are contiguous in memory,
        // we can apply simd, and this is good place to use `dot` from lvl1.
        // We can keep x in cache, since it's used across tha columns
        let ix_b = if incx < 0 {
            (1 - m as isize) * incx as isize
        } else {
            0
        }; // <- m & n are interchanged
        let iy_b = if incy < 0 {
            (1 - n as isize) * incy as isize
        } else {
            0
        };

        // outer loop (X-vector)
        for row in (0..m).step_by(ROW_BLOCK) {
            let row_end = (row + ROW_BLOCK).min(m);
            let curr_m = row_end - row;

            // Calculate exact memory bounds and starting addr for x_buf
            let x_stride = incx.unsigned_abs() as usize;
            let x_buf_len = 1 + (curr_m - 1) * x_stride;
            let x_addr = if incx < 0 {
                ix_b + (row as isize + curr_m as isize - 1) * incx as isize
            } else {
                ix_b + (row as isize * incx as isize)
            };

            // Experiment
            unsafe {
                _mm_prefetch(
                    x.as_ptr().add((row + 2 * ROW_BLOCK).min(x.len())) as *const i8,
                    _MM_HINT_ET0,
                );
            }

            // Get immut buf from x for this "fixed" row chunk
            // this stay in cache, same logic as above but for y this time
            let x_ptr = unsafe { x.as_ptr().offset(x_addr) }; // <- used offset here
            let x_buf = unsafe { from_raw_parts(x_ptr, x_buf_len) };

            // inner loop (Y-vector)
            for col in (0..n).step_by(COL_BLOCK) {
                let col_end = (col + COL_BLOCK).min(n); // <- handle last

                unsafe {
                    let yb_ptr = y.as_mut_ptr();
                    // iter over cols in the block
                    for i in col..col_end {
                        // Get base pointer for the current col, we will move it across the col for each iteration,
                        let col_ptr = a.as_ptr().add(i * lda + row);
                        // Get buf from on that ith, we use `from_raw_parts` since it contiguous and look `curr_m` ahead
                        let col_buf = from_raw_parts(col_ptr, curr_m);

                        // use `dot` from lvl1, performs col_buf * x_buf
                        let dot_val = dot_unsafe(curr_m, col_buf, 1, x_buf, incx); // <- again incx is 1 here, same reason

                        // Get location for y[i] and partial apply dot
                        let iy = iy_b + (i as isize * incy as isize);
                        let ty_ptr = yb_ptr.offset(iy);
                        *ty_ptr = alpha.mul_add(dot_val, *ty_ptr); // Y + alpha * dot -> Y
                    } // move to the next row
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// The gemv routines compute a scalar-matrix-vector product and add the result to a scalar-vector product, with a general matrix.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemv.html)
pub fn gemv_native(
    m: usize,   // row of mat
    n: usize,   // col of mat
    alpha: f32, // scaling for product
    a: &[f32],  // matrix buf
    lda: usize, // leading dim of a row or col major depends, we follow `column major`
    x: &[f32],  // mul vector buf
    incx: i32,
    beta: f32,     // y scaling
    y: &mut [f32], // resultant buf
    incy: i32,
    is_trans: bool, // Fortran uses 'T', 'N', but we will use bool
) {
    // get the effective dimensions of x and y based on transposition
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

    // we use `scal` to handle the beta scaling and zeroing out y if beta is 0, as per BLAS spec
    scal(y_len, beta, y, incy);

    // y is already scaled with beta, we are done here, no need to do any mul computation
    // also returning here means `gemv` just became `scal` :0
    if alpha == 0.0 {
        return;
    }

    // We have taken `!trans` because, default is column major,
    // so for simd we need contigous memory, and this is fine place to use `axpy` from lvl1
    if !is_trans {
        let incx = incx as isize;
        unsafe {
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            // iter all col here
            for idx in 0..n {
                // Get base pointer, we wil move it down the column for each iteration,
                // using lda is the leading dimension (row stride for column major)
                let col_ptr = a.as_ptr().add(idx * lda); // <- e.g c1, c2, c3..n times
                let col_buf = from_raw_parts(col_ptr, m); // every col has m elements, and they are contiguous in memory, so we can create a slice buf for it
                // Partial result for this column,
                // res_buf <- alpha * col[0] * x[0] + ... + alpha * col[n] * x[n].
                let x_val = *x.as_ptr().offset(ix);
                axpy(m, alpha * x_val, col_buf, 1, y, incy); // <- we take incx as 1, because its column major, simd can be applied :9
                ix += incx;
            }
        }
    } else {
        // ok now it transposed, same logic but opposite behavior, we have "row-major" and since its row are contiguous in memory,
        // we can apply simd, and this is good place to use `dot` from lvl1
        let incy = incy as isize;
        let mut iy = if incy < 0 {
            (1 - y_len as isize) * incy
        } else {
            0
        };

        // base pointer for y, we will offset it for each row
        let y_ptr = y.as_mut_ptr();

        unsafe {
            // iter over all rows
            for i in 0..n {
                // Get base pointer for the current row, we will move it across the row for each iteration,
                let col_ptr = a.as_ptr().add(i * lda);
                // Get buf from on that idx, we use `from_raw_parts` since it contiguous and look `n` ahead
                let col_buf = from_raw_parts(col_ptr, m);

                // use `dot` from lvl1
                let dot_val = dot(m, col_buf, 1, x, incx); // <- again incx is 1 here, same reason

                // Apply alpha and add to y
                *y_ptr.offset(iy) = alpha.mul_add(dot_val, *y_ptr.offset(iy));

                iy += incy;
            } // move to the next row
        }
    }
}

// my over engineering gemv, did 18% lesser in gflops 😭🌹
#[test]
fn gemv_native_test() {
    use crate::utils::gen_fill;
    use rayon::join;
    use std::hint::black_box;

    let warmup_count = 32;
    let run_count = 256;

    let size = 4096;

    let mut a = vec![1.0f32; size * size];
    let mut x = vec![1.0f32; size];
    let mut y1 = vec![0.0f32; size];
    let mut y2 = vec![0.0f32; size];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y1);
    black_box(&mut y2);

    gen_fill(&mut a);
    gen_fill(&mut x);

    y1.fill(1.0);
    y2.fill(1.0);

    for _ in 0..warmup_count {
        y1.fill(1.0);
        y2.fill(1.0);
        join(
            || {
                gemv_native(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y1, 1, false);
            },
            || {
                gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y2, 1, false);
            },
        );
    }
    assert!(
        y1.iter()
            .zip(y2.iter())
            .all(|(&v1, &v2)| (v1 - v2).abs() < 1e-3)
    );

    y1.fill(1.0);
    y2.fill(1.0);

    let (dur_native, dur_opt) = join(
        || {
            let start = std::time::Instant::now();
            for _ in 0..run_count {
                gemv_native(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y1, 1, false);
            }
            start.elapsed()
        },
        || {
            let start = std::time::Instant::now();
            for _ in 0..run_count {
                gemv(size, size, 5.0, &a, size, &x, 1, 7.0, &mut y2, 1, false);
            }
            start.elapsed()
        },
    );

    let total_flops = 2.0 * (size.pow(2) as f64) * (run_count as f64);
    let gflops_native = total_flops / dur_native.as_secs_f64() / 1e9;
    let gflops_opt = total_flops / dur_opt.as_secs_f64() / 1e9;

    println!(
        "gemv_native: {:?} seconds, {:.2} GFLOPS",
        dur_native.as_secs_f64(),
        gflops_native
    );
    println!(
        "gemv_opt: {:?} seconds, {:.2} GFLOPS",
        dur_opt.as_secs_f64(),
        gflops_opt
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
