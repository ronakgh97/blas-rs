//! Implementation of Level 1 BLAS routines
use crate::{reduce_add, reduce_max};
use std::arch::x86_64::_mm256_setr_epi32;
#[allow(unused)]
use std::arch::x86_64::{
    __m256i, _CMP_GT_OQ, _CMP_LT_OQ, _MM_HINT_ET0, _MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T2,
    _mm_prefetch, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps, _mm256_blendv_epi8,
    _mm256_blendv_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set_epi32, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256, _mm256_storeu_ps, _mm256_storeu_si256,
    _mm256_stream_ps,
};
use std::ptr::{copy_nonoverlapping, swap_nonoverlapping};
// TODO: x[ix], x[ix + incx], x[ix + 2*incx], ..., x[ix + (n-1)*incx]
// TODO: take raw ptr from unsafe fn, let caller adjust
// TODO; Boilerplate, use MACRO EXPANSIONS

#[inline(always)]
/// The axpy routines compute a scalar-vector product and add the result to a vector.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/axpy.htmll) for more details
pub fn axpy(n: usize, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 || alpha == 0.0 {
        return;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    // Bound checks
    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    // Bound checks
    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    unsafe {
        axpy_unsafe(n, alpha, x, incx, y, incy);
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn axpy_unsafe(
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: i32,
    y: &mut [f32],
    incy: i32,
) {
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    unsafe {
        if incx == 1 && incy == 1 {
            let mut i = 0;

            // Handle 4 AVX registers at a time
            let alpha_x8 = _mm256_set1_ps(alpha);
            while i + 32 <= n {
                // Load from x
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Load from y
                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                // FMA, Y += alpha * X
                let r0 = _mm256_fmadd_ps(alpha_x8, x0, y0);
                let r1 = _mm256_fmadd_ps(alpha_x8, x1, y1);
                let r2 = _mm256_fmadd_ps(alpha_x8, x2, y2);
                let r3 = _mm256_fmadd_ps(alpha_x8, x3, y3);

                // Store results back to y
                _mm256_storeu_ps(y_ptr.add(i), r0);
                _mm256_storeu_ps(y_ptr.add(i + 8), r1);
                _mm256_storeu_ps(y_ptr.add(i + 16), r2);
                _mm256_storeu_ps(y_ptr.add(i + 24), r3);

                i += 32;

                // {
                //     _mm_prefetch(x_ptr.add(i + 128) as *const i8, _MM_HINT_NTA);
                //     _mm_prefetch(y_ptr.add(i + 128) as *const i8, _MM_HINT_NTA);
                // }
            }

            // Handle one AVX register at a time.
            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let y = _mm256_loadu_ps(y_ptr.add(i));
                let res = _mm256_fmadd_ps(alpha_x8, x, y);
                _mm256_storeu_ps(y_ptr.add(i), res);
                i += 8;
            }

            // Handle remaining elements
            while i < n {
                let x_val = *x_ptr.add(i);
                let y_val = *y_ptr.add(i);
                *y_ptr.add(i) = alpha.mul_add(x_val, y_val);
                i += 1;
            }
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 {
                (n as isize - 1) * -incx
            } else {
                0
            };
            let mut iy = if incy < 0 {
                (n as isize - 1) * -incy
            } else {
                0
            };

            // Y += alpha * X
            let mut i = 0;
            while i + 4 <= n {
                // process 4 elements per iter
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = alpha.mul_add(x_val, y_val);
                ix += incx;
                iy += incy;

                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = alpha.mul_add(x_val, y_val);
                ix += incx;
                iy += incy;

                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = alpha.mul_add(x_val, y_val);
                ix += incx;
                iy += incy;

                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = alpha.mul_add(x_val, y_val);
                ix += incx;
                iy += incy;

                i += 4;
            }

            while i < n {
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = alpha.mul_add(x_val, y_val);
                ix += incx;
                iy += incy;

                i += 1;
            }
        }
    }
}

#[inline(always)]
/// The scal routines computes a scalar-vector product.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/scal.html) for more details
pub fn scal(n: usize, alpha: f32, x: &mut [f32], incx: i32) {
    if n == 0 || alpha == 1.0 {
        return;
    }

    if incx == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    unsafe {
        scal_unsafe(n, alpha, x, incx);
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn scal_unsafe(n: usize, alpha: f32, x: &mut [f32], incx: i32) {
    let x_ptr = x.as_mut_ptr();

    unsafe {
        if incx == 1 {
            if alpha == 0.0 {
                std::ptr::write_bytes(x_ptr, 0u8, n);
                return;
            }

            let alpha_x8 = _mm256_set1_ps(alpha);
            let mut i = 0;

            // For large n, use non-temporal stores to avoid
            // polluting cache with data that won't be reused soon.
            // `_mm256_stream_ps` requires 32-byte alignment.
            let is_aligned = (x_ptr as usize).is_multiple_of(32);
            if is_aligned {
                while i + 32 <= n {
                    let v0 = _mm256_mul_ps(alpha_x8, _mm256_loadu_ps(x_ptr.add(i)));
                    let v1 = _mm256_mul_ps(alpha_x8, _mm256_loadu_ps(x_ptr.add(i + 8)));
                    let v2 = _mm256_mul_ps(alpha_x8, _mm256_loadu_ps(x_ptr.add(i + 16)));
                    let v3 = _mm256_mul_ps(alpha_x8, _mm256_loadu_ps(x_ptr.add(i + 24)));

                    _mm256_stream_ps(x_ptr.add(i), v0);
                    _mm256_stream_ps(x_ptr.add(i + 8), v1);
                    _mm256_stream_ps(x_ptr.add(i + 16), v2);
                    _mm256_stream_ps(x_ptr.add(i + 24), v3);

                    i += 32;
                }

                while i + 8 <= n {
                    let v = _mm256_mul_ps(alpha_x8, _mm256_loadu_ps(x_ptr.add(i)));
                    _mm256_stream_ps(x_ptr.add(i), v);
                    i += 8;
                }
            } else {
                // unaligned case, use regular loadu, storeu, 32 elements at time
                while i + 32 <= n {
                    let mut v0 = _mm256_loadu_ps(x_ptr.add(i));
                    let mut v1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                    let mut v2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                    let mut v3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                    v0 = _mm256_mul_ps(alpha_x8, v0);
                    v1 = _mm256_mul_ps(alpha_x8, v1);
                    v2 = _mm256_mul_ps(alpha_x8, v2);
                    v3 = _mm256_mul_ps(alpha_x8, v3);

                    _mm256_storeu_ps(x_ptr.add(i), v0);
                    _mm256_storeu_ps(x_ptr.add(i + 8), v1);
                    _mm256_storeu_ps(x_ptr.add(i + 16), v2);
                    _mm256_storeu_ps(x_ptr.add(i + 24), v3);

                    // next 32 elements
                    i += 32;

                    _mm_prefetch(x_ptr.add(i + 32) as *const i8, _MM_HINT_NTA);
                }

                while i + 8 <= n {
                    let v = _mm256_loadu_ps(x_ptr.add(i));
                    let res = _mm256_mul_ps(alpha_x8, v);
                    _mm256_storeu_ps(x_ptr.add(i), res);
                    i += 8;
                }
            }

            // handle leftovers
            while i < n {
                *x_ptr.add(i) *= alpha;
                i += 1;
            }
        } else {
            // stride case, we can't use SIMD, just do a simple loop
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

            let mut i = 0;
            while i + 4 <= n {
                *x_ptr.offset(ix) *= alpha;
                *x_ptr.offset(ix + incx) *= alpha;
                *x_ptr.offset(ix + 2 * incx) *= alpha;
                *x_ptr.offset(ix + 3 * incx) *= alpha;
                ix += 4 * incx;
                i += 4;
            }

            // fallback
            while i < n {
                *x_ptr.offset(ix) *= alpha;
                ix += incx;
                i += 1;
            }
        }
    }
}

#[inline(always)]
/// The copy routines copy one vector to another.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/copy.html) for more details
pub fn copy(n: usize, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 {
        return;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    unsafe {
        copy_unsafe(n, x, incx, y, incy);
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn copy_unsafe(n: usize, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    unsafe {
        if incx == 1 && incy == 1 {
            // contiguous memory allows for a simple bulk copy
            copy_nonoverlapping(x_ptr, y_ptr, n); // same as memcpy
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

            // unroll the loop to copy 4 elements at a time for better perf?
            let mut i = 0;
            while i + 4 <= n {
                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;

                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;

                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;

                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;

                i += 4;
            }

            // copy remaining elements
            while i < n {
                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;
                i += 1;
            }
        }
    }
}

#[inline(always)]
/// Given two vectors of n elements, x and y, the swap routines return vectors y and x swapped, each replacing the other.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/swap.html) for more details
pub fn swap(n: usize, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 {
        return;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    unsafe {
        swap_unsafe(n, x, incx, y, incy);
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn swap_unsafe(n: usize, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();

    unsafe {
        if incx == 1 && incy == 1 {
            let x_addr = x_ptr as usize;
            let y_addr = y_ptr as usize;
            let byte_len = n * size_of::<f32>();

            if x_addr == y_addr {
            } else if x_addr + byte_len <= y_addr || y_addr + byte_len <= x_addr {
                swap_nonoverlapping(x_ptr, y_ptr, n);
            } else {
                for i in 0..n {
                    std::ptr::swap(x_ptr.add(i), y_ptr.add(i));
                }
            }
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

            // swap 4 elements at a time
            let mut i = 0;
            while i + 4 <= n {
                let tmp0 = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp0;
                ix += incx;
                iy += incy;

                let tmp1 = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp1;
                ix += incx;
                iy += incy;

                let tmp2 = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp2;
                ix += incx;
                iy += incy;

                let tmp3 = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp3;
                ix += incx;
                iy += incy;

                i += 4;
            }

            while i < n {
                let tmp = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp;
                ix += incx;
                iy += incy;
                i += 1;
            }
        }
    }
}

#[inline(always)]
/// The dot routines perform a dot product between two vectors.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/dot.html) for more details
pub fn dot(n: usize, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    if n == 0 {
        return 0.0;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    unsafe { dot_unsafe(n, x, incx, y, incy) }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn dot_unsafe(n: usize, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    unsafe {
        if incx == 1 && incy == 1 {
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            let mut i = 0;

            // Load 32 elements (4 AVX registers) at a time
            // compute partial dot products in 'parallel' (ILP)
            while i + 32 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                // 4 accumulators to allow for some instruction-level parallelism,
                // we will sum them up at the end
                sum0 = _mm256_fmadd_ps(x0, y0, sum0);
                sum1 = _mm256_fmadd_ps(x1, y1, sum1);
                sum2 = _mm256_fmadd_ps(x2, y2, sum2);
                sum3 = _mm256_fmadd_ps(x3, y3, sum3);

                i += 32;

                // {
                //     _mm_prefetch(x_ptr.add(i + 128) as *const i8, _MM_HINT_T0);
                //     _mm_prefetch(y_ptr.add(i + 128) as *const i8, _MM_HINT_T0);
                // }
            }

            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let y = _mm256_loadu_ps(y_ptr.add(i));
                sum0 = _mm256_fmadd_ps(x, y, sum0);
                i += 8;
            }

            // add & reduce
            let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
            let mut result = reduce_add!(sum);

            while i < n {
                let x_val = *x_ptr.add(i);
                let y_val = *y_ptr.add(i);
                result = x_val.mul_add(y_val, result);
                i += 1;
            }

            result
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;

            // Have four accumulators to allow for some level of parallelism
            let mut i = 0;
            while i + 4 <= n {
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                sum0 = x_val.mul_add(y_val, sum0);
                ix += incx;
                iy += incy;

                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                sum1 = x_val.mul_add(y_val, sum1);
                ix += incx;
                iy += incy;

                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                sum2 = x_val.mul_add(y_val, sum2);
                ix += incx;
                iy += incy;

                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                sum3 = x_val.mul_add(y_val, sum3);
                ix += incx;
                iy += incy;

                i += 4;
            }

            // Handle remaining elements
            while i < n {
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                sum0 = x_val.mul_add(y_val, sum0);
                ix += incx;
                iy += incy;

                i += 4;
            }

            (sum0 + sum1) + (sum2 + sum3)
        }
    }
}

#[inline(always)]
/// The nrm2 routines compute Euclidean norm of a vector.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/nrm2.html) for more details
pub fn nrm2(n: usize, x: &[f32], incx: i32) -> f32 {
    if n == 0 {
        return 0.0;
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    unsafe { nrm2_unsafe(n, x, incx) }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn nrm2_unsafe(n: usize, x: &[f32], incx: i32) -> f32 {
    let x_ptr = x.as_ptr();

    unsafe {
        if incx == 1 {
            if n == 0 {
                return 0.0;
            }

            // Mask to clear sign bit for absolute value
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

            // find maximum absolute value (scale) to prevent overflow
            // nrm2 = scale * sqrt(sum((x[i]/scale)^2))
            let mut i: usize = 0;
            let mut scale0 = _mm256_setzero_ps();
            let mut scale1 = _mm256_setzero_ps();
            let mut scale2 = _mm256_setzero_ps();
            let mut scale3 = _mm256_setzero_ps();

            while i + 32 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // mask sign bit using AND and do max to find scale
                scale0 = _mm256_max_ps(scale0, _mm256_and_ps(x0, mask));
                scale1 = _mm256_max_ps(scale1, _mm256_and_ps(x1, mask));
                scale2 = _mm256_max_ps(scale2, _mm256_and_ps(x2, mask));
                scale3 = _mm256_max_ps(scale3, _mm256_and_ps(x3, mask));

                i += 32;
            }

            // find max
            let mut combined =
                _mm256_max_ps(_mm256_max_ps(scale0, scale1), _mm256_max_ps(scale2, scale3));
            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                combined = _mm256_max_ps(combined, _mm256_and_ps(x, mask));
                i += 8;
            }

            // reduce max
            let mut scale = reduce_max!(combined);
            // handle remaining elements (when n % 8 != 0 and n < 32)
            while i < n {
                scale = scale.max((*x_ptr.add(i)).abs());
                i += 1;
            }

            if scale == 0.0 && scale == f32::INFINITY {
                return 0.0;
            }

            // compute sum of (x[i]/scale)^2
            let mut i: usize = 0;
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            let inv_scale = _mm256_set1_ps(1.0 / scale); // inv, mul > div?

            while i + 32 <= n {
                // Load & 'scale' 4 AVX reg at time
                let x0 = _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i)), inv_scale);
                let x1 = _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i + 8)), inv_scale);
                let x2 = _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i + 16)), inv_scale);
                let x3 = _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i + 24)), inv_scale);

                // square and accumulate
                sum0 = _mm256_fmadd_ps(x0, x0, sum0);
                sum1 = _mm256_fmadd_ps(x1, x1, sum1);
                sum2 = _mm256_fmadd_ps(x2, x2, sum2);
                sum3 = _mm256_fmadd_ps(x3, x3, sum3);

                i += 32;

                // {
                //     _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_T2);
                // }
            }

            let mut sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
            while i + 8 <= n {
                let x = _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i)), inv_scale);
                sum = _mm256_fmadd_ps(x, x, sum);
                i += 8;
            }

            let mut result = reduce_add!(sum);

            let inv_scale = 1.0 / scale;

            // handle remaining
            while i < n {
                let xi = *x_ptr.add(i) * inv_scale;
                result = xi.mul_add(xi, result);
                i += 1;
            }

            scale * result.sqrt()
        } else {
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

            // find max absolute value
            let mut scale = 0.0f32;
            let mut i = 0usize;
            while i < n {
                let val = (*x_ptr.offset(ix)).abs();
                scale = scale.max(val);
                ix += incx;
                i += 1;
            }

            if scale == 0.0 {
                return 0.0;
            }

            // compute sum of (x[i]/scale)^2
            ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut sum: f32 = 0.0;

            let mut i = 0usize;
            while i + 4 <= n {
                let x0 = *x_ptr.offset(ix) / scale;
                let x1 = *x_ptr.offset(ix + incx) / scale;
                let x2 = *x_ptr.offset(ix + 2 * incx) / scale;
                let x3 = *x_ptr.offset(ix + 3 * incx) / scale;

                sum = x0.mul_add(x0, sum);
                sum = x1.mul_add(x1, sum);
                sum = x2.mul_add(x2, sum);
                sum = x3.mul_add(x3, sum);

                ix += 4 * incx;
                i += 4;
            }

            while i < n {
                let x_val = *x_ptr.offset(ix) / scale;
                sum = x_val.mul_add(x_val, sum);
                ix += incx;
                i += 1;
            }

            scale * sum.sqrt()
        }
    }
}

#[inline(always)]
/// The asum routine computes the sum of the magnitudes of elements of a real vector, or the sum of magnitudes of the real and imaginary parts of elements of a complex vector.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/asum.html) for more details
pub fn asum(n: usize, x: &[f32], incx: i32) -> f32 {
    if n == 0 {
        return 0.0;
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    unsafe { asum_unsafe(n, x, incx) }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn asum_unsafe(n: usize, x: &[f32], incx: i32) -> f32 {
    let x_ptr = x.as_ptr();

    unsafe {
        if incx == 1 {
            let mut i = 0;

            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();

            // Mask to clear the sign bit, effectively computing absolute value: [0x7fffffff, 0x7fffffff, ...8 times]
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

            while i + 32 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Compute absolute values using AND with mask
                let abs_x0 = _mm256_and_ps(x0, mask);
                let abs_x1 = _mm256_and_ps(x1, mask);
                let abs_x2 = _mm256_and_ps(x2, mask);
                let abs_x3 = _mm256_and_ps(x3, mask);

                sum0 = _mm256_add_ps(sum0, abs_x0);
                sum1 = _mm256_add_ps(sum1, abs_x1);
                sum2 = _mm256_add_ps(sum2, abs_x2);
                sum3 = _mm256_add_ps(sum3, abs_x3);

                i += 32;

                // {
                //     _mm_prefetch(x_ptr.add(i + 256) as *const i8, _MM_HINT_NTA);
                // }
            }

            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let abs_x = _mm256_and_ps(x, _mm256_set1_ps(f32::from_bits(0x7FFFFFFF)));
                sum0 = _mm256_add_ps(sum0, abs_x);
                i += 8;
            }

            // sum them up!
            let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

            let mut result = reduce_add!(sum); // reduce
            while i < n {
                result += (*x_ptr.add(i)).abs();
                i += 1;
            }
            result
        } else {
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let x_ptr = x.as_ptr();
            let mut sum = 0.0f32;

            let mut i = 0usize;
            while i + 4 <= n {
                let v0 = *x_ptr.offset(ix);
                let v1 = *x_ptr.offset(ix + incx);
                let v2 = *x_ptr.offset(ix + 2 * incx);
                let v3 = *x_ptr.offset(ix + 3 * incx);
                sum += v0.abs();
                sum += v1.abs();
                sum += v2.abs();
                sum += v3.abs();
                ix += 4 * incx;
                i += 4;
            }

            while i < n {
                sum += (*x_ptr.offset(ix)).abs();
                ix += incx;
                i += 1;
            }

            sum
        }
    }
}

#[inline(always)]
/// The iamax routines return an index i such that 'x\[i\]' has the maximum absolute value of all elements in vector x. _(DOES NOT HANDLE NAN)_
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/iamax.html) for more details
pub fn i_amax(n: usize, x: &[f32], incx: i32) -> usize {
    if n == 0 {
        panic!("n must be greater than 0");
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    unsafe { i_amax_unsafe(n, x, incx) }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn i_amax_unsafe(n: usize, x: &[f32], incx: i32) -> usize {
    unsafe {
        if incx == 1 {
            let x_ptr = x.as_ptr();

            // Create mask [0x7fffffff, 0x7fffffff, ...8 times]
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

            // Lane order for indices, will be added to base index to get the actual index of the max value
            let base_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

            // Init max trackers to -inf so any valid absolute value overwrites it,
            // I hope this doesn't cause any issues later :(
            let neg_inf_lane = _mm256_set1_ps(-f32::INFINITY);

            let mut la_vals0 = neg_inf_lane;
            let mut la_idxs0 = _mm256_setzero_si256();

            let mut la_vals1 = neg_inf_lane;
            let mut la_idxs1 = _mm256_setzero_si256();

            let mut la_vals2 = neg_inf_lane;
            let mut la_idxs2 = _mm256_setzero_si256();

            let mut la_vals3 = neg_inf_lane;
            let mut la_idxs3 = _mm256_setzero_si256();

            // pre-index to reduce broadcast
            let mut i = 0usize;
            let mut idx0 = base_idx;
            let mut idx1 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(8));
            let mut idx2 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(16));
            let mut idx3 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(24));
            let thirty_two = _mm256_set1_epi32(32);

            while i + 32 <= n {
                // Load Values
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Compute absolute values using AND with mask
                let xabs0 = _mm256_and_ps(x0, mask);
                let xabs1 = _mm256_and_ps(x1, mask);
                let xabs2 = _mm256_and_ps(x2, mask);
                let xabs3 = _mm256_and_ps(x3, mask);

                // Create comparison masks > current max for each lane
                // Either returns 0xFFFFFFFF (true) or 0x00000000 (false) for each lane
                let cmp0 = _mm256_cmp_ps(xabs0, la_vals0, _CMP_GT_OQ);
                let cmp1 = _mm256_cmp_ps(xabs1, la_vals1, _CMP_GT_OQ);
                let cmp2 = _mm256_cmp_ps(xabs2, la_vals2, _CMP_GT_OQ);
                let cmp3 = _mm256_cmp_ps(xabs3, la_vals3, _CMP_GT_OQ);

                // Blend values and indices using the mask,
                // take compute cmp mask for each lane, check against abs lane,
                // keep either the old max or update with new value and index.
                // For example
                //  old val = [5.0, 3.0, 6.0, 2.0, 4.0, 1.0, 7.0, 0.5], new val = [4.0, 8.0, 5.0, 1.0, 3.0, 2.0, 6.0, 9.0]
                //  cmp mask = [0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0xFFFFFFFF]
                //  result = [5.0, 8.0, 6.0, 2.0, 4.0, 1.0, 7.0, 9.0]
                // Same goes for indices
                la_vals0 = _mm256_blendv_ps(la_vals0, xabs0, cmp0);
                la_idxs0 = _mm256_blendv_epi8(la_idxs0, idx0, _mm256_castps_si256(cmp0));

                la_vals1 = _mm256_blendv_ps(la_vals1, xabs1, cmp1);
                la_idxs1 = _mm256_blendv_epi8(la_idxs1, idx1, _mm256_castps_si256(cmp1));

                la_vals2 = _mm256_blendv_ps(la_vals2, xabs2, cmp2);
                la_idxs2 = _mm256_blendv_epi8(la_idxs2, idx2, _mm256_castps_si256(cmp2));

                la_vals3 = _mm256_blendv_ps(la_vals3, xabs3, cmp3);
                la_idxs3 = _mm256_blendv_epi8(la_idxs3, idx3, _mm256_castps_si256(cmp3));

                i += 32;
                // Move indices up by 32 for the next iteration
                idx0 = _mm256_add_epi32(idx0, thirty_two);
                idx1 = _mm256_add_epi32(idx1, thirty_two);
                idx2 = _mm256_add_epi32(idx2, thirty_two);
                idx3 = _mm256_add_epi32(idx3, thirty_two);

                // {
                //     _mm_prefetch(x_ptr.add(i + 256) as *const i8, _MM_HINT_NTA);
                // }
            }

            // Tree reduction of the 4 max values and their indices to a single max value and index
            let cmp01 = _mm256_cmp_ps(la_vals1, la_vals0, _CMP_GT_OQ);
            let m01_vals = _mm256_blendv_ps(la_vals0, la_vals1, cmp01);
            let m01_idxs = _mm256_blendv_epi8(la_idxs0, la_idxs1, _mm256_castps_si256(cmp01));

            let cmp23 = _mm256_cmp_ps(la_vals3, la_vals2, _CMP_GT_OQ);
            let m23_vals = _mm256_blendv_ps(la_vals2, la_vals3, cmp23);
            let m23_idxs = _mm256_blendv_epi8(la_idxs2, la_idxs3, _mm256_castps_si256(cmp23));

            let cmp_final = _mm256_cmp_ps(m23_vals, m01_vals, _CMP_GT_OQ);
            let mut la_vals = _mm256_blendv_ps(m01_vals, m23_vals, cmp_final);
            let mut la_idxs =
                _mm256_blendv_epi8(m01_idxs, m23_idxs, _mm256_castps_si256(cmp_final));

            let mut rem_idx = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32));
            let eight = _mm256_set1_epi32(8); // maintain 8-lane running index for the next loop

            // Reminder loop for any remaining elements that didn't fit into the 32-element blocks
            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let abs = _mm256_and_ps(x, mask);

                let cmp = _mm256_cmp_ps(abs, la_vals, _CMP_GT_OQ);
                la_vals = _mm256_blendv_ps(la_vals, abs, cmp);
                la_idxs = _mm256_blendv_epi8(la_idxs, rem_idx, _mm256_castps_si256(cmp));

                rem_idx = _mm256_add_epi32(rem_idx, eight);

                i += 8;
            }

            // TODO; these tmp alloc is fine, since it not inside any hot loop

            let mut tmp_vals = [0.0f32; 8];
            let mut tmp_idxs = [0i32; 8];
            // Store the tree-reduced values to temporary arrays for final scalar reduction
            _mm256_storeu_ps(tmp_vals.as_mut_ptr(), la_vals);
            _mm256_storeu_si256(tmp_idxs.as_mut_ptr() as *mut __m256i, la_idxs);

            let mut la_val = -f32::INFINITY;
            let mut la_idx = 0usize;

            for j in 0..8 {
                // _CMP_GT_OQ is strict (>) and ignores equality, so the SIMD registers
                // inherently preserve the first occurrence (lowest index)
                if tmp_vals.get_unchecked(j) > &la_val {
                    la_val = *tmp_vals.get_unchecked(j);
                    la_idx = *tmp_idxs.get_unchecked(j) as usize;
                } // On tie, we ignore the new index since
                // we want the first occurrence (lower index) of the max value
            }

            while i < n {
                let val = (*x_ptr.add(i)).abs();
                if val > la_val || (val == la_val && i < la_idx) {
                    la_val = val;
                    la_idx = i;
                }
                i += 1;
            }

            la_idx
        } else {
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let x_ptr = x.as_ptr();
            let mut la_idx: usize = 0;
            let mut la_val = -f32::INFINITY;
            for i in 0..n {
                let val = (*x_ptr.offset(ix)).abs();
                if val > la_val {
                    la_val = val;
                    la_idx = i;
                }
                ix += incx;
            }
            la_idx
        }
    }
}

#[inline(always)]
/// The iamin routines return an index i such that 'x\[i\]' has the minimum absolute value of all elements in vector x. _(DOES NOT HANDLE NAN)_
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/iamin.html) for more details
pub fn i_amin(n: usize, x: &[f32], incx: i32) -> usize {
    if n == 0 {
        panic!("n must be greater than 0");
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    unsafe { i_amin_unsafe(n, x, incx) }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn i_amin_unsafe(n: usize, x: &[f32], incx: i32) -> usize {
    unsafe {
        if incx == 1 {
            let x_ptr = x.as_ptr();

            // Create mask [0x7fffffff, 0x7fffffff, ...8 times]
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

            let base_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

            // Init min trackers to +inf so any valid absolute value overwrites it,
            let pos_inf_lane = _mm256_set1_ps(f32::INFINITY);
            let mut sm_vals0 = pos_inf_lane;
            let mut sm_vals1 = pos_inf_lane;
            let mut sm_vals2 = pos_inf_lane;
            let mut sm_vals3 = pos_inf_lane;

            let mut sm_idxs0 = _mm256_setzero_si256();
            let mut sm_idxs1 = _mm256_setzero_si256();
            let mut sm_idxs2 = _mm256_setzero_si256();
            let mut sm_idxs3 = _mm256_setzero_si256();

            // Index once b4 loop for less `vpbroadcastd`
            let mut i = 0usize;
            let mut idx0 = base_idx;
            let mut idx1 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(8));
            let mut idx2 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(16));
            let mut idx3 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(24));
            let thirty_two = _mm256_set1_epi32(32);

            while i + 32 <= n {
                // Load Values
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Compute absolute values using AND with mask
                let xabs0 = _mm256_and_ps(x0, mask);
                let xabs1 = _mm256_and_ps(x1, mask);
                let xabs2 = _mm256_and_ps(x2, mask);
                let xabs3 = _mm256_and_ps(x3, mask);

                // Create comparison masks < current min for each lane
                // Either returns 0xFFFFFFFF (true) or 0x00000000 (false) for each lane
                let cmp0 = _mm256_cmp_ps(xabs0, sm_vals0, _CMP_LT_OQ);
                let cmp1 = _mm256_cmp_ps(xabs1, sm_vals1, _CMP_LT_OQ);
                let cmp2 = _mm256_cmp_ps(xabs2, sm_vals2, _CMP_LT_OQ);
                let cmp3 = _mm256_cmp_ps(xabs3, sm_vals3, _CMP_LT_OQ);

                // Blend values and indices using the mask,
                // take compute cmp mask for each lane, check against abs lane,
                // keep either the old min or update with new value and index.
                sm_vals0 = _mm256_blendv_ps(sm_vals0, xabs0, cmp0);
                sm_idxs0 = _mm256_blendv_epi8(sm_idxs0, idx0, _mm256_castps_si256(cmp0));

                sm_vals1 = _mm256_blendv_ps(sm_vals1, xabs1, cmp1);
                sm_idxs1 = _mm256_blendv_epi8(sm_idxs1, idx1, _mm256_castps_si256(cmp1));

                sm_vals2 = _mm256_blendv_ps(sm_vals2, xabs2, cmp2);
                sm_idxs2 = _mm256_blendv_epi8(sm_idxs2, idx2, _mm256_castps_si256(cmp2));

                sm_vals3 = _mm256_blendv_ps(sm_vals3, xabs3, cmp3);
                sm_idxs3 = _mm256_blendv_epi8(sm_idxs3, idx3, _mm256_castps_si256(cmp3));

                // Process next 32 elements
                i += 32;
                idx0 = _mm256_add_epi32(idx0, thirty_two);
                idx1 = _mm256_add_epi32(idx1, thirty_two);
                idx2 = _mm256_add_epi32(idx2, thirty_two);
                idx3 = _mm256_add_epi32(idx3, thirty_two);

                // {
                //     _mm_prefetch(x_ptr.add(i + 256) as *const i8, _MM_HINT_NTA);
                // }
            }

            // Tree reduction for merging acc
            let cmp01 = _mm256_cmp_ps(sm_vals1, sm_vals0, _CMP_LT_OQ);
            let m01_vals = _mm256_blendv_ps(sm_vals0, sm_vals1, cmp01);
            let m01_idxs = _mm256_blendv_epi8(sm_idxs0, sm_idxs1, _mm256_castps_si256(cmp01));

            let cmp23 = _mm256_cmp_ps(sm_vals3, sm_vals2, _CMP_LT_OQ);
            let m23_vals = _mm256_blendv_ps(sm_vals2, sm_vals3, cmp23);
            let m23_idxs = _mm256_blendv_epi8(sm_idxs2, sm_idxs3, _mm256_castps_si256(cmp23));

            let cmp_final = _mm256_cmp_ps(m23_vals, m01_vals, _CMP_LT_OQ);
            let mut sm_vals = _mm256_blendv_ps(m01_vals, m23_vals, cmp_final);
            let mut sm_idxs =
                _mm256_blendv_epi8(m01_idxs, m23_idxs, _mm256_castps_si256(cmp_final));

            // Running index for the remaining elements, starting from the last processed index
            let mut rem_idx = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32));
            let eight = _mm256_set1_epi32(8);

            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let abs = _mm256_and_ps(x, mask);

                let cmp = _mm256_cmp_ps(abs, sm_vals, _CMP_LT_OQ);
                sm_vals = _mm256_blendv_ps(sm_vals, abs, cmp);
                sm_idxs = _mm256_blendv_epi8(sm_idxs, rem_idx, _mm256_castps_si256(cmp));

                rem_idx = _mm256_add_epi32(rem_idx, eight);

                i += 8;
            }

            let mut tmp_vals = [0.0f32; 8];
            let mut tmp_idxs = [0i32; 8];
            // Store the tree-reduced values to temporary arrays for final scalar reduction
            _mm256_storeu_ps(tmp_vals.as_mut_ptr(), sm_vals);
            _mm256_storeu_si256(tmp_idxs.as_mut_ptr() as *mut __m256i, sm_idxs);

            let mut sm_val = f32::INFINITY;
            let mut sm_idx = 0usize;

            for j in 0..8 {
                // Simplified tie-breaking (strict < keeps the first occurrence)
                if tmp_vals.get_unchecked(j) < &sm_val {
                    sm_val = *tmp_vals.get_unchecked(j);
                    sm_idx = *tmp_idxs.get_unchecked(j) as usize;
                } // On tie, we ignore the new index since
                // we want the first occurrence (lower index) of the min value
            }

            while i < n {
                let val = (*x_ptr.add(i)).abs();
                if val < sm_val || (val == sm_val && i < sm_idx) {
                    sm_val = val;
                    sm_idx = i;
                }
                i += 1;
            }

            sm_idx
        } else {
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let x_ptr = x.as_ptr();
            let mut sm_idx: usize = 0;
            let mut sm_val = f32::INFINITY;
            for i in 0..n {
                let val = (*x_ptr.offset(ix)).abs();
                if val < sm_val {
                    sm_val = val;
                    sm_idx = i;
                }
                ix += incx;
            }
            sm_idx
        }
    }
}

#[inline(always)]
/// Given two vectors x and y of n elements, the rot routines compute four scalar-vector products and update the input vectors with the sum of two of these scalar-vector products.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/rot.html)
pub fn rot(n: usize, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32, c: f32, s: f32) {
    if n == 0 {
        panic!("n must be greater than 0");
    }

    if incx == 0 || incy == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();
    if incx == 1 && incy == 1 {
        unsafe {
            let c_x8 = _mm256_set1_ps(c);
            let s_x8 = _mm256_set1_ps(s);
            let neg_s_x8 = _mm256_set1_ps(-s);

            let mut i = 0;

            while i + 32 <= n {
                // Load from x
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Load from y
                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                // Updated rotated x
                let rx0 = _mm256_fmadd_ps(c_x8, x0, _mm256_mul_ps(s_x8, y0));
                let rx1 = _mm256_fmadd_ps(c_x8, x1, _mm256_mul_ps(s_x8, y1));
                let rx2 = _mm256_fmadd_ps(c_x8, x2, _mm256_mul_ps(s_x8, y2));
                let rx3 = _mm256_fmadd_ps(c_x8, x3, _mm256_mul_ps(s_x8, y3));

                // Updated rotated y
                let ry0 = _mm256_fmadd_ps(c_x8, y0, _mm256_mul_ps(neg_s_x8, x0));
                let ry1 = _mm256_fmadd_ps(c_x8, y1, _mm256_mul_ps(neg_s_x8, x1));
                let ry2 = _mm256_fmadd_ps(c_x8, y2, _mm256_mul_ps(neg_s_x8, x2));
                let ry3 = _mm256_fmadd_ps(c_x8, y3, _mm256_mul_ps(neg_s_x8, x3));

                // Write back
                _mm256_storeu_ps(x_ptr.add(i), rx0);
                _mm256_storeu_ps(x_ptr.add(i + 8), rx1);
                _mm256_storeu_ps(x_ptr.add(i + 16), rx2);
                _mm256_storeu_ps(x_ptr.add(i + 24), rx3);

                _mm256_storeu_ps(y_ptr.add(i), ry0);
                _mm256_storeu_ps(y_ptr.add(i + 8), ry1);
                _mm256_storeu_ps(y_ptr.add(i + 16), ry2);
                _mm256_storeu_ps(y_ptr.add(i + 24), ry3);
                i += 32;

                // {
                //     _mm_prefetch(x_ptr.add(i + 128) as *const i8, _MM_HINT_ET0);
                //     _mm_prefetch(y_ptr.add(i + 128) as *const i8, _MM_HINT_ET0);
                // }
            }
            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let y = _mm256_loadu_ps(y_ptr.add(i));

                let rx = _mm256_fmadd_ps(c_x8, x, _mm256_mul_ps(s_x8, y));
                let ry = _mm256_fmadd_ps(c_x8, y, _mm256_mul_ps(neg_s_x8, x));

                _mm256_storeu_ps(x_ptr.add(i), rx);
                _mm256_storeu_ps(y_ptr.add(i), ry);
                i += 8;
            }
            while i < n {
                let x_val = *x_ptr.add(i);
                let y_val = *y_ptr.add(i);
                *x_ptr.add(i) = c * x_val + s * y_val;
                *y_ptr.add(i) = c * y_val - s * x_val;
                i += 1;
            }
        }
    } else {
        let incx = incx as isize;
        let incy = incy as isize;
        let mut ix = if incx < 0 {
            (n as isize - 1) * -incx
        } else {
            0
        };
        let mut iy = if incy < 0 {
            (n as isize - 1) * -incy
        } else {
            0
        };

        unsafe {
            for _ in 0..n {
                // x' = C⋅x+S⋅y / y' = C⋅y−S⋅x
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *x_ptr.offset(ix) = c.mul_add(x_val, s * y_val);
                *y_ptr.offset(iy) = c.mul_add(y_val, -s * x_val);

                ix += incx;
                iy += incy;
            }
        }
    }
}

#[inline(always)]
/// Given the Cartesian coordinates (a, b) of a point, the rotg routines return the parameters c, s, r, and z associated with the Givens rotation.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/rotg.html)
pub fn rotg(a: &mut f32, b: &mut f32, c: &mut f32, s: &mut f32) {
    // Handle the zero scale case early
    if a.abs() + b.abs() == 0.0 {
        *c = 1.0;
        *s = 0.0;
        *a = 0.0;
        *b = 0.0;
        return;
    }
    // rot is calculate based on mag, because we want shortest rotation
    // For example, if we have (-999, 1), we take something like ~(-999.0005, 0),
    // so r copy sign of a, so that cos and sin stays in second quadrant
    // TODO: BUT I can't confirm/find how BLAS does, they ALWAYS take the sign of *a, [think later]
    let rot = if a.abs() > b.abs() { *a } else { *b };
    let mut r = a.hypot(*b);
    r = r.copysign(rot);

    // Calculate sin and cos theta of rot matrix
    // This is trivial, if solve the equation taking second component of given vector as zero,
    // we would get this relation
    // From here we can say cos and sine is proportional to a and b,
    *c = *a / r;
    *s = *b / r;

    // This is by the intel docs ref to prevent floating point precision issues like (1 - (99999)^2) since s^2 + c^2 = 1
    // What/How this does? its compression, if (a) |cos| is larger than (b) |sin|, so we store smaller value, here means |sin| < 1
    // and if not we CANT return smaller this time, because caller won't be able to know, Z is c or s?,
    // so we return 1/c, since |cos| < 1, so inverse be > 1, now caller can distinguish, |Z| > 1 -> 1/c or |Z| < 1 -> s
    // Also we need check for c == 0, so we send s = 1 (sin[pi/2]), means vector is align with y-axis, we just need +90 rot, that's it
    let z = if a.abs() > b.abs() {
        *s
    } else if *c != 0.0 {
        1.0 / *c
    } else {
        1.0
    };

    *a = r;
    *b = z;
}

//TODO: rotmg and rotm here
