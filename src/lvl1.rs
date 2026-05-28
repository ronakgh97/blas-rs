use crate::utils::from_m256;
#[allow(unused_imports)]
use std::arch::x86_64::{
    __m256i, _CMP_GT_OQ, _CMP_LT_OQ, _MM_HINT_ET0, _MM_HINT_NTA, _MM_HINT_T1, _MM_HINT_T2,
    _mm_prefetch, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps, _mm256_blendv_epi8,
    _mm256_blendv_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_fmadd_ps,
    _mm256_hadd_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
    _mm256_set_epi32, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_shuffle_epi32, _mm256_storeu_ps, _mm256_storeu_si256,
};
// TODO: x[ix], x[ix + incx], x[ix + 2*incx], ..., x[ix + (n-1)*incx]
// TODO: Need to handle to overflows for f32, using scale^2 * ( (x1/scale)^2 + (x1/scale)^2 + ... )
// TODO: take raw ptr from unsafe fn

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

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    if incx == 1 && incy == 1 {
        let mut i = 0;

        // Handle 4 AVX registers at a time
        unsafe {
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

                // I don't know, how to rightfully use this
                if i + 96 < n {
                    _mm_prefetch(x_ptr.add(i + 96) as *const i8, _MM_HINT_T2);
                    _mm_prefetch(y_ptr.add(i + 96) as *const i8, _MM_HINT_NTA);
                }
            }

            // Handle one AVX register at a time.
            while i + 8 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let res0 = _mm256_fmadd_ps(alpha_x8, x0, y0);
                _mm256_storeu_ps(y_ptr.add(i), res0);
                i += 8;
            }

            // Handle remaining elements
            while i < n {
                let x_val = *x_ptr.add(i);
                let y_val = *y_ptr.add(i);
                *y_ptr.add(i) = alpha.mul_add(x_val, y_val);
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
                // Y += alpha * X
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = alpha.mul_add(x_val, y_val);
                ix += incx;
                iy += incy;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// The axpy routines compute a scalar-vector product and add the result to a vector.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/axpy.htmll) for more details
pub unsafe fn axpy_unsafe(n: usize, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    if incx == 1 && incy == 1 {
        let mut i = 0;

        // Handle 4 AVX registers at a time
        unsafe {
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

                // I don't know, how to rightfully use this
                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_T2);
                    _mm_prefetch(y_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
                }
            }

            // Handle one AVX register at a time.
            while i + 8 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let res0 = _mm256_fmadd_ps(alpha_x8, x0, y0);
                _mm256_storeu_ps(y_ptr.add(i), res0);
                i += 8;
            }

            // Handle remaining elements
            while i < n {
                let x_val = *x_ptr.add(i);
                let y_val = *y_ptr.add(i);
                *y_ptr.add(i) = alpha.mul_add(x_val, y_val);
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
                // Y += alpha * X
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = alpha.mul_add(x_val, y_val);
                ix += incx;
                iy += incy;
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
        let x_ptr = x.as_mut_ptr();

        if incx == 1 {
            if alpha == 0.0 {
                std::ptr::write_bytes(x_ptr, 0u8, n); // memset to zero
                // for i in 0..n {
                //     *x_ptr.add(i) = 0.0  ;
                // }

                return;
            }

            let alpha_x8 = _mm256_set1_ps(alpha);
            let mut i = 0;

            while i + 32 <= n {
                // Load 4 AVX registers at a time
                let mut v0 = _mm256_loadu_ps(x_ptr.add(i));
                let mut v1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let mut v2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let mut v3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Scale the vectors by alpha
                v0 = _mm256_mul_ps(alpha_x8, v0);
                v1 = _mm256_mul_ps(alpha_x8, v1);
                v2 = _mm256_mul_ps(alpha_x8, v2);
                v3 = _mm256_mul_ps(alpha_x8, v3);

                // Write back
                _mm256_storeu_ps(x_ptr.add(i), v0);
                _mm256_storeu_ps(x_ptr.add(i + 8), v1);
                _mm256_storeu_ps(x_ptr.add(i + 16), v2);
                _mm256_storeu_ps(x_ptr.add(i + 24), v3);

                // We processed 4 registers of 8 lanes each -> 32 elements
                i += 32;

                // Experimental
                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 32) as *const i8, _MM_HINT_NTA);
                }
            }

            // Handle leftovers
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x_ptr.add(i));
                let res = _mm256_mul_ps(alpha_x8, v);
                _mm256_storeu_ps(x_ptr.add(i), res);
                i += 8;
            }

            while i < n {
                *x_ptr.add(i) *= alpha;
                i += 1;
            }
        } else {
            // Stride case, we can't use SIMD, just do a simple loop
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

            for _ in 0..n {
                *x_ptr.offset(ix) *= alpha;
                ix += incx;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// The scal routines computes a scalar-vector product.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/scal.html) for more details
pub unsafe fn scal_unsafe(n: usize, alpha: f32, x: &mut [f32], incx: i32) {
    unsafe {
        let x_ptr = x.as_mut_ptr();

        if incx == 1 {
            if alpha == 0.0 {
                std::ptr::write_bytes(x_ptr, 0u8, n); // memset to zero
                // for i in 0..n {
                //     *x_ptr.add(i) = 0.0  ;
                // }

                return;
            }

            let alpha_x8 = _mm256_set1_ps(alpha);
            let mut i = 0;

            while i + 32 <= n {
                // Load 4 AVX registers at a time
                let mut v0 = _mm256_loadu_ps(x_ptr.add(i));
                let mut v1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let mut v2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let mut v3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Scale the vectors by alpha
                v0 = _mm256_mul_ps(alpha_x8, v0);
                v1 = _mm256_mul_ps(alpha_x8, v1);
                v2 = _mm256_mul_ps(alpha_x8, v2);
                v3 = _mm256_mul_ps(alpha_x8, v3);

                // Write back
                _mm256_storeu_ps(x_ptr.add(i), v0);
                _mm256_storeu_ps(x_ptr.add(i + 8), v1);
                _mm256_storeu_ps(x_ptr.add(i + 16), v2);
                _mm256_storeu_ps(x_ptr.add(i + 24), v3);

                // We processed 4 registers of 8 lanes each -> 32 elements
                i += 32;

                // Experimental
                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 32) as *const i8, _MM_HINT_NTA);
                }
            }

            // Handle leftovers
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x_ptr.add(i));
                let res = _mm256_mul_ps(alpha_x8, v);
                _mm256_storeu_ps(x_ptr.add(i), res);
                i += 8;
            }

            while i < n {
                *x_ptr.add(i) *= alpha;
                i += 1;
            }
        } else {
            // Stride case, we can't use SIMD, just do a simple loop
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

            for _ in 0..n {
                *x_ptr.offset(ix) *= alpha;
                ix += incx;
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
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        if incx == 1 && incy == 1 {
            // Contiguous memory allows for a simple bulk copy
            std::ptr::copy_nonoverlapping(x_ptr, y_ptr, n);
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

            for _ in 0..n {
                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// The copy routines copy one vector to another.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/copy.html) for more details
pub fn copy_unsafe(n: usize, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        if incx == 1 && incy == 1 {
            // Contiguous memory allows for a simple bulk copy
            std::ptr::copy_nonoverlapping(x_ptr, y_ptr, n);
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

            for _ in 0..n {
                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;
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
        let x_ptr = x.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();

        if incx == 1 && incy == 1 {
            let x_addr = x_ptr as usize;
            let y_addr = y_ptr as usize;
            let byte_len = n * size_of::<f32>();

            if x_addr == y_addr {
            } else if x_addr + byte_len <= y_addr || y_addr + byte_len <= x_addr {
                std::ptr::swap_nonoverlapping(x_ptr, y_ptr, n);
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

            for _ in 0..n {
                let tmp = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp;
                ix += incx;
                iy += incy;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// Given two vectors of n elements, x and y, the swap routines return vectors y and x swapped, each replacing the other.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/swap.html) for more details
pub fn swap_unsafe(n: usize, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    unsafe {
        let x_ptr = x.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();

        if incx == 1 && incy == 1 {
            let x_addr = x_ptr as usize;
            let y_addr = y_ptr as usize;
            let byte_len = n * size_of::<f32>();

            if x_addr == y_addr {
            } else if x_addr + byte_len <= y_addr || y_addr + byte_len <= x_addr {
                std::ptr::swap_nonoverlapping(x_ptr, y_ptr, n);
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

            for _ in 0..n {
                let tmp = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp;
                ix += incx;
                iy += incy;
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

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    if incx == 1 && incy == 1 {
        unsafe {
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            let mut i = 0;

            // Load 32 elements (4 AVX registers) at a time and compute partial dot products in parallel
            while i + 32 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                // 4 accumulators to allow for some instruction-level parallelism, we will sum them up at the end
                sum0 = _mm256_fmadd_ps(x0, y0, sum0);
                sum1 = _mm256_fmadd_ps(x1, y1, sum1);
                sum2 = _mm256_fmadd_ps(x2, y2, sum2);
                sum3 = _mm256_fmadd_ps(x3, y3, sum3);

                i += 32;

                if i + 48 < n {
                    _mm_prefetch(x_ptr.add(i + 48) as *const i8, _MM_HINT_T1);
                    _mm_prefetch(y_ptr.add(i + 48) as *const i8, _MM_HINT_T1);
                }
            }

            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let y = _mm256_loadu_ps(y_ptr.add(i));
                sum0 = _mm256_fmadd_ps(x, y, sum0);
                i += 8;
            }

            let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

            let mut result = from_m256(sum);
            while i < n {
                result += x[i] * y[i];
                i += 1;
            }

            result
        }
    } else {
        let incx = incx as isize;
        let incy = incy as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;

        // Have two accumulators for `dot` to allow for some instruction-level parallelism
        for _ in 0..n / 2 {
            unsafe {
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
            }
        }

        // Handle the last element if n is odd
        if n % 2 == 1 {
            unsafe {
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                sum0 = x_val.mul_add(y_val, sum0);
            }
        }

        sum0 + sum1
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// The dot routines perform a dot product between two vectors.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/dot.html) for more details
pub unsafe fn dot_unsafe(n: usize, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    if incx == 1 && incy == 1 {
        unsafe {
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            let mut i = 0;

            // Load 32 elements (4 AVX registers) at a time and compute partial dot products in parallel
            while i + 32 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                // 4 accumulators to allow for some instruction-level parallelism, we will sum them up at the end
                sum0 = _mm256_fmadd_ps(x0, y0, sum0);
                sum1 = _mm256_fmadd_ps(x1, y1, sum1);
                sum2 = _mm256_fmadd_ps(x2, y2, sum2);
                sum3 = _mm256_fmadd_ps(x3, y3, sum3);

                i += 32;

                if i + 48 < n {
                    _mm_prefetch(x_ptr.add(i + 48) as *const i8, _MM_HINT_T1);
                    _mm_prefetch(y_ptr.add(i + 48) as *const i8, _MM_HINT_T1);
                }
            }

            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let y = _mm256_loadu_ps(y_ptr.add(i));
                sum0 = _mm256_fmadd_ps(x, y, sum0);
                i += 8;
            }

            let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

            let mut result = from_m256(sum);
            while i < n {
                result += x[i] * y[i];
                i += 1;
            }

            result
        }
    } else {
        let incx = incx as isize;
        let incy = incy as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;

        // Have two accumulators for `dot` to allow for some instruction-level parallelism
        for _ in 0..n / 2 {
            unsafe {
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
            }
        }

        // Handle the last element if n is odd
        if n % 2 == 1 {
            unsafe {
                let x_val = *x_ptr.offset(ix);
                let y_val = *y_ptr.offset(iy);
                sum0 = x_val.mul_add(y_val, sum0);
            }
        }

        sum0 + sum1
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

    if incx == 1 {
        let mut i = 0;
        let mut sum = unsafe { _mm256_setzero_ps() };

        let x_ptr = x.as_ptr();

        while i + 32 <= n {
            unsafe {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // I don't think, this will do what I intended too
                sum = _mm256_fmadd_ps(x0, x0, sum);
                sum = _mm256_fmadd_ps(x1, x1, sum);
                sum = _mm256_fmadd_ps(x2, x2, sum);
                sum = _mm256_fmadd_ps(x3, x3, sum);

                i += 32;

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_T2);
                }
            }
        }

        while i + 8 <= n {
            unsafe {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                sum = _mm256_fmadd_ps(x, x, sum);
                i += 8;
            }
        }

        let mut result = from_m256(sum);
        while i < n {
            result += x[i] * x[i];
            i += 1;
        }

        result.sqrt()
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let mut sum: f32 = 0.0;
        let x_ptr = x.as_ptr();
        for _ in 0..n {
            unsafe {
                let x_val = *x_ptr.offset(ix);
                sum = x_val.mul_add(x_val, sum);
                ix += incx;
            }
        }
        sum.sqrt()
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// The nrm2 routines compute Euclidean norm of a vector.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/nrm2.html) for more details
pub fn nrm2_unsafe(n: usize, x: &[f32], incx: i32) -> f32 {
    if incx == 1 {
        let mut i = 0;
        let mut sum = unsafe { _mm256_setzero_ps() };

        let x_ptr = x.as_ptr();

        while i + 32 <= n {
            unsafe {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // I don't think, this will do what I intended too
                sum = _mm256_fmadd_ps(x0, x0, sum);
                sum = _mm256_fmadd_ps(x1, x1, sum);
                sum = _mm256_fmadd_ps(x2, x2, sum);
                sum = _mm256_fmadd_ps(x3, x3, sum);

                i += 32;

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_T2);
                }
            }
        }

        while i + 8 <= n {
            unsafe {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                sum = _mm256_fmadd_ps(x, x, sum);
                i += 8;
            }
        }

        let mut result = from_m256(sum);
        while i < n {
            result += x[i] * x[i];
            i += 1;
        }

        result.sqrt()
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let mut sum: f32 = 0.0;
        let x_ptr = x.as_ptr();
        for _ in 0..n {
            unsafe {
                let x_val = *x_ptr.offset(ix);
                sum = x_val.mul_add(x_val, sum);
                ix += incx;
            }
        }
        sum.sqrt()
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

    if incx == 1 {
        let mut i = 0;
        let x_ptr = x.as_ptr();
        let mut sum0 = unsafe { _mm256_setzero_ps() };
        let mut sum1 = unsafe { _mm256_setzero_ps() };
        let mut sum2 = unsafe { _mm256_setzero_ps() };
        let mut sum3 = unsafe { _mm256_setzero_ps() };

        // Mask to clear the sign bit, effectively computing absolute value: [0x7fffffff, 0x7fffffff, ...8 times]
        let mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)) };

        while i + 32 <= n {
            unsafe {
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

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
                }
            }
        }

        while i + 8 <= n {
            unsafe {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let abs_x = _mm256_and_ps(x, _mm256_set1_ps(f32::from_bits(0x7FFFFFFF)));
                sum0 = _mm256_add_ps(sum0, abs_x);
                i += 8;
            }
        }

        // sum up
        let sum = unsafe { _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)) };

        let mut result = from_m256(sum); // reduce
        while i < n {
            result += x[i].abs();
            i += 1;
        }
        result
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

        let x_ptr = x.as_ptr();

        let mut sum = 0.0f32;

        for _ in 0..n {
            unsafe {
                sum += (*x_ptr.offset(ix)).abs();
                ix += incx;
            }
        }
        sum
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// The asum routine computes the sum of the magnitudes of elements of a real vector, or the sum of magnitudes of the real and imaginary parts of elements of a complex vector.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/asum.html) for more details
pub fn asum_unsafe(n: usize, x: &[f32], incx: i32) -> f32 {
    if incx == 1 {
        let mut i = 0;
        let x_ptr = x.as_ptr();
        let mut sum0 = unsafe { _mm256_setzero_ps() };
        let mut sum1 = unsafe { _mm256_setzero_ps() };
        let mut sum2 = unsafe { _mm256_setzero_ps() };
        let mut sum3 = unsafe { _mm256_setzero_ps() };

        // Mask to clear the sign bit, effectively computing absolute value: [0x7fffffff, 0x7fffffff, ...8 times]
        let mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)) };

        while i + 32 <= n {
            unsafe {
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

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
                }
            }
        }

        while i + 8 <= n {
            unsafe {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let abs_x = _mm256_and_ps(x, _mm256_set1_ps(f32::from_bits(0x7FFFFFFF)));
                sum0 = _mm256_add_ps(sum0, abs_x);
                i += 8;
            }
        }

        // sum up
        let sum = unsafe { _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)) };

        let mut result = from_m256(sum); // reduce
        while i < n {
            result += x[i].abs();
            i += 1;
        }
        result
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

        let x_ptr = x.as_ptr();

        let mut sum = 0.0f32;

        for _ in 0..n {
            unsafe {
                sum += (*x_ptr.offset(ix)).abs();
                ix += incx;
            }
        }
        sum
    }
}

#[inline(always)]
/// The iamax routines return an index i such that x\[i\] has the maximum absolute value of all elements in vector x.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/iamax.html) for more details
pub fn i_amax(n: usize, x: &[f32], incx: i32) -> usize {
    if n == 0 {
        return 0;
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if incx == 1 {
        unsafe {
            let x_ptr = x.as_ptr();

            // Create mask [0x7fffffff, 0x7fffffff, ...8 times]
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

            let base_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            // Init max trackers to -inf so any valid absolute value overwrites it,
            // I hope this doesn't cause any issues later :(
            let mut la_vals0 = _mm256_set1_ps(-f32::INFINITY);
            let mut la_idxs0 = _mm256_setzero_si256();

            let mut la_vals1 = _mm256_set1_ps(-f32::INFINITY);
            let mut la_idxs1 = _mm256_setzero_si256();

            let mut la_vals2 = _mm256_set1_ps(-f32::INFINITY);
            let mut la_idxs2 = _mm256_setzero_si256();

            let mut la_vals3 = _mm256_set1_ps(-f32::INFINITY);
            let mut la_idxs3 = _mm256_setzero_si256();

            let mut i = 0usize;

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

                // Create index vectors for current chunks by adding 'i' to base index
                // Each contains [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]...
                let idx0 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32)); // TODO: Broadcast overhead
                let idx1 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 8) as i32));
                let idx2 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 16) as i32));
                let idx3 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 24) as i32));

                // Create comparison masks > current max for each lane
                // Either returns 0xFFFFFFFF (true) or 0x00000000 (false) for each lane
                let cmp0 = _mm256_cmp_ps(xabs0, la_vals0, _CMP_GT_OQ);
                let cmp1 = _mm256_cmp_ps(xabs1, la_vals1, _CMP_GT_OQ);
                let cmp2 = _mm256_cmp_ps(xabs2, la_vals2, _CMP_GT_OQ);
                let cmp3 = _mm256_cmp_ps(xabs3, la_vals3, _CMP_GT_OQ);

                // Blend values and indices using the mask,
                // take compute cmp mask for each lane, check against abs lane,
                // keep either the old max or update with new value and index
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

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
                }
            }

            let cmp01 = _mm256_cmp_ps(la_vals1, la_vals0, _CMP_GT_OQ);
            la_vals0 = _mm256_blendv_ps(la_vals0, la_vals1, cmp01);
            la_idxs0 = _mm256_blendv_epi8(la_idxs0, la_idxs1, _mm256_castps_si256(cmp01));

            let cmp02 = _mm256_cmp_ps(la_vals2, la_vals0, _CMP_GT_OQ);
            la_vals0 = _mm256_blendv_ps(la_vals0, la_vals2, cmp02);
            la_idxs0 = _mm256_blendv_epi8(la_idxs0, la_idxs2, _mm256_castps_si256(cmp02));

            let cmp03 = _mm256_cmp_ps(la_vals3, la_vals0, _CMP_GT_OQ);
            la_vals0 = _mm256_blendv_ps(la_vals0, la_vals3, cmp03);
            la_idxs0 = _mm256_blendv_epi8(la_idxs0, la_idxs3, _mm256_castps_si256(cmp03));

            while i + 8 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let abs0 = _mm256_and_ps(x0, mask);
                let idx0 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32));

                let cmp0 = _mm256_cmp_ps(abs0, la_vals0, _CMP_GT_OQ);
                la_vals0 = _mm256_blendv_ps(la_vals0, abs0, cmp0);
                la_idxs0 = _mm256_blendv_epi8(la_idxs0, idx0, _mm256_castps_si256(cmp0));

                i += 8;
            }

            let mut tmp_vals = [0.0f32; 8];
            let mut tmp_idxs = [0i32; 8];
            // Store the registers to temporary arrays for reduction
            _mm256_storeu_ps(tmp_vals.as_mut_ptr(), la_vals0);
            _mm256_storeu_si256(tmp_idxs.as_mut_ptr() as *mut __m256i, la_idxs0);

            let mut la_val = -1.0f32;
            let mut la_idx = 0usize;

            for j in 0..8 {
                // Check if the current value is greater than the max found so far, or if it's a tie,
                // check if the index is smaller (to ensure we return the first occurrence of the max value)
                if tmp_vals[j] > la_val
                    || (tmp_vals[j] == la_val && (tmp_idxs[j] as usize) < la_idx)
                {
                    la_val = tmp_vals[j];
                    la_idx = tmp_idxs[j] as usize;
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
        }
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let x_ptr = x.as_ptr();
        let mut la_idx: usize = 0;
        let mut la_val = -f32::INFINITY;
        for _ in 0..n {
            let val = unsafe { (*x_ptr.offset(ix)).abs() };
            if val > la_val {
                la_val = val;
                la_idx = ix as usize;
            }
            ix += incx;
        }
        la_idx
    }
}

#[inline(always)]
/// The iamin routines return an index i such that x\[i\] has the minimum absolute value of all elements in vector x.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/iamin.html) for more details
pub fn i_amin(n: usize, x: &[f32], incx: i32) -> usize {
    if n == 0 {
        return 0;
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if incx == 1 {
        unsafe {
            let x_ptr = x.as_ptr();

            // Create mask [0x7fffffff, 0x7fffffff, ...8 times]
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

            let base_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            // Init min trackers to +inf so any valid absolute value overwrites it,
            let mut sm_vals0 = _mm256_set1_ps(f32::INFINITY);
            let mut sm_vals1 = _mm256_set1_ps(f32::INFINITY);
            let mut sm_vals2 = _mm256_set1_ps(f32::INFINITY);
            let mut sm_vals3 = _mm256_set1_ps(f32::INFINITY);

            let mut sm_idxs0 = _mm256_setzero_si256();
            let mut sm_idxs1 = _mm256_setzero_si256();
            let mut sm_idxs2 = _mm256_setzero_si256();
            let mut sm_idxs3 = _mm256_setzero_si256();

            let mut i = 0usize;

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

                // Create index vectors for current chunks by adding 'i' to base index
                // Each contains [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]...
                let idx0 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32)); // TODO: Broadcast overhead
                let idx1 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 8) as i32));
                let idx2 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 16) as i32));
                let idx3 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 24) as i32));

                // Create comparison masks < current min for each lane
                // Either returns 0xFFFFFFFF (true) or 0x00000000 (false) for each lane
                let cmp0 = _mm256_cmp_ps(xabs0, sm_vals0, _CMP_LT_OQ);
                let cmp1 = _mm256_cmp_ps(xabs1, sm_vals1, _CMP_LT_OQ);
                let cmp2 = _mm256_cmp_ps(xabs2, sm_vals2, _CMP_LT_OQ);
                let cmp3 = _mm256_cmp_ps(xabs3, sm_vals3, _CMP_LT_OQ);

                // Blend values and indices using the mask,
                // take compute cmp mask for each lane, check against abs lane,
                // keep either the old max or update with new value and index
                // For example
                //  old val = [5.0, 3.0, 6.0, 2.0, 4.0, 1.0, 7.0, 0.5], new val = [4.0, 8.0, 5.0, 1.0, 3.0, 2.0, 6.0, 9.0]
                //  cmp mask = [0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0xFFFFFFFF]
                //  result = [5.0, 8.0, 6.0, 2.0, 4.0, 1.0, 7.0, 9.0]
                // Same goes for indices
                sm_vals0 = _mm256_blendv_ps(sm_vals0, xabs0, cmp0);
                sm_idxs0 = _mm256_blendv_epi8(sm_idxs0, idx0, _mm256_castps_si256(cmp0));

                sm_vals1 = _mm256_blendv_ps(sm_vals1, xabs1, cmp1);
                sm_idxs1 = _mm256_blendv_epi8(sm_idxs1, idx1, _mm256_castps_si256(cmp1));

                sm_vals2 = _mm256_blendv_ps(sm_vals2, xabs2, cmp2);
                sm_idxs2 = _mm256_blendv_epi8(sm_idxs2, idx2, _mm256_castps_si256(cmp2));

                sm_vals3 = _mm256_blendv_ps(sm_vals3, xabs3, cmp3);
                sm_idxs3 = _mm256_blendv_epi8(sm_idxs3, idx3, _mm256_castps_si256(cmp3));

                i += 32;

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
                }
            }

            let cmp01 = _mm256_cmp_ps(sm_vals1, sm_vals0, _CMP_LT_OQ);
            sm_vals0 = _mm256_blendv_ps(sm_vals0, sm_vals1, cmp01);
            sm_idxs0 = _mm256_blendv_epi8(sm_idxs0, sm_idxs1, _mm256_castps_si256(cmp01));

            let cmp02 = _mm256_cmp_ps(sm_vals2, sm_vals0, _CMP_LT_OQ);
            sm_vals0 = _mm256_blendv_ps(sm_vals0, sm_vals2, cmp02);
            sm_idxs0 = _mm256_blendv_epi8(sm_idxs0, sm_idxs2, _mm256_castps_si256(cmp02));

            let cmp03 = _mm256_cmp_ps(sm_vals3, sm_vals0, _CMP_LT_OQ);
            sm_vals0 = _mm256_blendv_ps(sm_vals0, sm_vals3, cmp03);
            sm_idxs0 = _mm256_blendv_epi8(sm_idxs0, sm_idxs3, _mm256_castps_si256(cmp03));

            while i + 8 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let abs0 = _mm256_and_ps(x0, mask);
                let idx0 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32));

                let cmp0 = _mm256_cmp_ps(abs0, sm_vals0, _CMP_LT_OQ);
                sm_vals0 = _mm256_blendv_ps(sm_vals0, abs0, cmp0);
                sm_idxs0 = _mm256_blendv_epi8(sm_idxs0, idx0, _mm256_castps_si256(cmp0));

                i += 8;
            }

            let mut tmp_vals = [0.0f32; 8];
            let mut tmp_idxs = [0i32; 8];
            // Store the SIMD registers to temporary arrays for reduction
            _mm256_storeu_ps(tmp_vals.as_mut_ptr(), sm_vals0);
            _mm256_storeu_si256(tmp_idxs.as_mut_ptr() as *mut __m256i, sm_idxs0);

            let mut sm_val = f32::INFINITY;
            let mut sm_idx = 0usize;

            for j in 0..8 {
                // Check if the current value is greater than the max found so far, or if it's a tie,
                // check if the index is smaller (to ensure we return the first occurrence of the max value)
                if tmp_vals[j] < sm_val
                    || (tmp_vals[j] == sm_val && (tmp_idxs[j] as usize) < sm_idx)
                {
                    sm_val = tmp_vals[j];
                    sm_idx = tmp_idxs[j] as usize;
                } // On tie, we ignore the new index since
                // we want the first occurrence (lower index) of the max value
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
        }
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let x_ptr = x.as_ptr();
        let mut sm_idx: usize = 0;
        let mut sm_val = f32::INFINITY;
        for _ in 0..n {
            let val = unsafe { (*x_ptr.offset(ix)).abs() };
            if val < sm_val {
                sm_val = val;
                sm_idx = ix as usize;
            }
            ix += incx;
        }
        sm_idx
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

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_ET0);
                    _mm_prefetch(y_ptr.add(i + 64) as *const i8, _MM_HINT_ET0);
                }
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
