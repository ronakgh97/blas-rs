use crate::lvl1::copy_unsafe;
use std::arch::x86_64::{
    __cpuid_count, __m256, _mm256_add_ps, _mm256_cvtss_f32, _mm256_permute_ps,
    _mm256_permute2f128_ps,
};

/// Fills an existing buffer with random values in range `[-1.0, 1.0]`
#[inline(always)]
pub fn gen_fill(buf: &mut [f32]) {
    let mut rng = fastrand::Rng::new();
    for x in buf {
        *x = rng.f32_inclusive() * 2.0 - 1.0;
    }
}

#[test]
fn test_gen_fill() {
    let mut buf = vec![0.0f32; 999_999_999];
    let strt = std::time::Instant::now();
    gen_fill(&mut buf);
    let elp = strt.elapsed();
    println!(
        "Generated {} random numbers in {:?} seconds",
        buf.len(),
        elp.as_secs_f32()
    );
    assert!(buf.iter().all(|&x| (-1.0..=1.0).contains(&x)));
}

/// Performs reduction of `_m256` vector and returns as a `f32`
#[inline(always)]
pub fn reduce_f32(v: __m256) -> f32 {
    unsafe {
        // first pass; [a  b  c  d | e  f  g  h]
        let hi = _mm256_permute2f128_ps(v, v, 1); // [e  f  g  h | a  b  c  d]
        let sum = _mm256_add_ps(v, hi); // [a+e  b+f  c+g  d+h | e+a  f+b  g+c  h+d]
        // [x0 x1 x2 x3] -> [x2 x3 x0 x1]
        let hi = _mm256_permute_ps(sum, 0b01001110); // [c+g  d+h  a+e  b+f | g+c  h+d  e+a  f+b]

        // second pass to add the two halves together
        let sum = _mm256_add_ps(sum, hi); // [a+e+b+f  c+g+d+h  e+a+f+b  g+c+h+d]
        let hi = _mm256_permute_ps(sum, 0b10110001); // [c+g+d+h  a+e+b+f  g+c+h+d  e+a+f+b]
        let sum = _mm256_add_ps(sum, hi); // [a+e+b+f+c+g+d+h  a+e+b+f+c+g+d+h  e+a+f+b+g+c+h+d  e+a+f+b+g+c+h+d]
        _mm256_cvtss_f32(sum) // just extract one
    }
}

#[inline(always)]
/// Transposes a matrix `src` of dimensions `rows x cols` into `dest` of dimensions `cols x rows` using a blocked approach for cache efficiency
pub fn mat_transpose(src: &[f32], dest: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dest.len(), rows * cols);

    let block = { (rows / 4).min(cols / 4).clamp(1, 64) };

    for rb in (0..rows).step_by(block) {
        let r_max = (rb + block).min(rows);

        for cb in (0..cols).step_by(block) {
            let cmax = (cb + block).min(cols);

            for r in rb..r_max {
                copy_unsafe(
                    cmax - cb,
                    &src[r * cols + cb..],
                    1,
                    &mut dest[cb * rows + r..],
                    rows as i32,
                );
            }
        }
    }
}

#[test]
fn test_mat_transpose() {
    let rows = 1024;
    let cols = 1536;
    let size = rows * cols;

    let mut src = vec![0.0f32; size];
    gen_fill(&mut src);
    let mut dest = vec![0.0f32; size];

    mat_transpose(&src, &mut dest, rows, cols);

    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(src[r * cols + c], dest[c * rows + r]);
        }
    }
}

#[inline]
/// Returns the cache sizes (L1, L2, L3) in KB for the current CPU using CPUID `X86`
pub fn get_cache_size() -> (usize, usize, usize) {
    let mut l1 = 0;
    let mut l2 = 0;
    let mut l3 = 0;

    let mut i = 0;

    loop {
        let res = __cpuid_count(4, i);

        let cache_type = res.eax & 0x1F;
        if cache_type == 0 {
            break;
        }

        let level = (res.eax >> 5) & 0x7;

        let ways = ((res.ebx >> 22) & 0x3FF) + 1;
        let partitions = ((res.ebx >> 12) & 0x3FF) + 1;
        let line_size = (res.ebx & 0xFFF) + 1;
        let sets = res.ecx + 1;

        let size_kb = (ways * partitions * line_size * sets) as usize / 1024;

        match (level, cache_type) {
            (1, 1) => l1 = size_kb,
            (2, 3) => l2 = size_kb,
            (3, 3) => l3 = size_kb,
            _ => {}
        }

        i += 1;
    }

    (l1, l2, l3)
}
