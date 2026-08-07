//! Utility functions for random noise generation and matrix helper operations
use rand::distr::uniform::SampleUniform;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use rand_xoshiro::rand_core::Rng;
use std::arch::x86_64::{
    __cpuid_count, __m256, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm_max_ps, _mm_min_ps,
    _mm_shuffle_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
    _mm256_permute2f128_ps, _mm256_shuffle_ps, _mm256_storeu_ps, _mm256_unpackhi_ps,
    _mm256_unpacklo_ps,
};

/// A simple random noise generator using the `Xoroshiro128PlusPlus` algorithm from the `rand_xoshiro` crate.
#[derive(Clone)]
pub struct Noise {
    rng: Xoroshiro128PlusPlus,
}

impl Noise {
    /// Initializes the noise generator with a random seed from the OS RNG
    pub fn init() -> Self {
        Self {
            rng: Xoroshiro128PlusPlus::from_rng(&mut rand::rng()),
        }
    }

    /// Generates a random number of type `T` in the inclusive range `[min, max]`
    #[inline(always)]
    pub fn rand_range<T: PartialOrd + SampleUniform>(&mut self, min: T, max: T) -> T {
        self.rng.random_range(min..=max)
    }

    /// Generates a random boolean with probability `p` of being `true`
    #[inline(always)]
    pub fn bool(&mut self, p: f64) -> bool {
        self.rng.random_bool(p)
    }

    /// Fills a buffer with random `i32` values `[-1, 0, 1]`
    #[inline(always)]
    pub fn fill_i32(&mut self, buf: &mut [i32]) {
        for x in buf {
            *x = self.rng.random_range(-1i32..=1i32);
        }
    }

    /// Fills a buffer with random `f32` values in the range `[-1.0, 1.0]`
    #[inline(always)]
    pub fn fill_f32(&mut self, buf: &mut [f32]) {
        for x in buf {
            *x = self.rng.random_range(-1.0..=1.0)
        }
    }

    /// Fills a buffer with random bytes
    #[inline(always)]
    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        self.rng.fill_bytes(buf);
    }
}

#[test]
fn test_f32_fill() {
    let mut noise = Noise::init();
    let mut buf = vec![0.0f32; 999_999_999];
    let strt = std::time::Instant::now();
    noise.fill_f32(&mut buf);
    let elp = strt.elapsed();
    println!(
        "Generated {} random numbers in {:?} seconds",
        buf.len(),
        elp.as_secs_f32()
    );
    assert!(buf.iter().all(|&x| (-1.0..=1.0).contains(&x)));
}

#[inline(always)]
/// Performs horizontal add reduction of `__m256` vector and returns as a `f32`
pub fn reduce_add(v: __m256) -> f32 {
    unsafe {
        // first pass; [a,b,c,d | e,f,g,h]
        let hi = _mm256_extractf128_ps(v, 1); // [e,f,g,h]
        let lo = _mm256_castps256_ps128(v); // [a,b,c,d]
        let hsum = _mm_add_ps(lo, hi); // [a+b, c+d, e+f, g+h]

        // second pass; add the two halves together
        let sum = _mm_hadd_ps(hsum, hsum); // [a+e+b+f c+g+d+h a+e+b+f c+g+d+h]
        let sum = _mm_hadd_ps(sum, sum); // [a+e+b+f+c+g+d+h,_,_,_]
        _mm_cvtss_f32(sum) // just extract one
    }
}

#[inline(always)]
/// Performs horizontal min reduction of `__m256` vector and returns as a `f32`
pub fn reduce_min(v: __m256) -> f32 {
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1); // upper 128 bit
        let lo = _mm256_castps256_ps128(v); // lower 128 bit
        let m = _mm_min_ps(lo, hi); // min of lower/upper, 256 -> 128 bit

        let shuffle = _mm_shuffle_ps(m, m, 0b01_00_11_10); // [a,b,c,d] -> [c,d,a,b]
        let m = _mm_min_ps(m, shuffle); // [x,y,x,y]

        let shuffle = _mm_shuffle_ps(m, m, 0b10_11_00_01); // [x,y,x,y] -> [y,x,y,x]
        let fm = _mm_min_ps(m, shuffle); // [m,m,m,m]
        _mm_cvtss_f32(fm) // extract first
    }
}

#[inline(always)]
/// Performs horizontal max reduction of `__m256` vector and returns as a `f32`
pub fn reduce_max(v: __m256) -> f32 {
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1); // upper 128 bit
        let lo = _mm256_castps256_ps128(v); // lower 128 bit
        let m = _mm_max_ps(lo, hi); // max of lower/upper, 256 -> 128 bit

        let shuffle = _mm_shuffle_ps(m, m, 0b01_00_11_10); // [a,b,c,d] -> [c,d,a,b]
        let m = _mm_max_ps(m, shuffle); // [x,y,x,y]

        let shuffle = _mm_shuffle_ps(m, m, 0b10_11_00_01); // [x,y,x,y] -> [y,x,y,x]
        let fm = _mm_max_ps(m, shuffle); // [m,m,m,m]
        _mm_cvtss_f32(fm) // extract first
    }
}

#[test]
fn test_simd_reduce() {
    use core::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    const LANE: usize = 8;
    const RUN: usize = 512 * 512; // ~2MB

    let mut noise = Noise::init();

    let sample = {
        let mut vec = vec![0.0f32; LANE * RUN]; // heap is fine here
        noise.fill_f32(&mut vec);
        vec
    };

    unsafe {
        // native way
        let (ntv_cycle_c, ntv_elapsed) = {
            // start
            let clock_start_mark = {
                _mm_lfence();
                _rdtsc()
            };
            let time = Instant::now();

            for i in 0..RUN {
                // for simplicity and avoiding obvious `stack-spill`
                // we use rust idiomatic iterator
                let max = sample[i * LANE..(i + 1) * LANE]
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .unwrap_unchecked();

                black_box(max);
            }

            // end
            let clock_end_mark = {
                let e = __rdtscp(&mut 0);
                _mm_lfence();
                e
            };
            let elapsed = time.elapsed();

            (clock_end_mark - clock_start_mark, elapsed)
        };

        // opt way
        let (opt_cycle_c, opt_elapsed) = {
            // start
            let clock_start_mark = {
                _mm_lfence();
                _rdtsc()
            };
            let time = Instant::now();

            for i in 0..RUN {
                let load_reg = _mm256_loadu_ps(sample.as_ptr().add(i * LANE));
                let max = reduce_max(load_reg); // in-reg
                black_box(max);
            }

            // end
            let clock_end_mark = {
                let e = __rdtscp(&mut 0);
                _mm_lfence();
                e
            };
            let elapsed = time.elapsed();

            (clock_end_mark - clock_start_mark, elapsed)
        };

        let nvt_cycle_per_ops = ntv_cycle_c / (RUN as u64);
        let opt_cycle_per_ops = opt_cycle_c / (RUN as u64);

        println!(
            "Native: cycles/ops: {}, elapsed: {:?}\nOpt: cycles/ops: {}, elapsed: {:?}",
            nvt_cycle_per_ops, ntv_elapsed, opt_cycle_per_ops, opt_elapsed
        );
    }
}

#[inline(always)]
/// Transposes a matrix `src` of dimensions `rows x cols` into `dest` of dimensions `cols x rows` using a blocked approach for cache efficiency
pub fn mat_transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), rows * cols);

    const TILE: usize = 64;
    const LANE: usize = 8;

    for ii in (0..rows).step_by(TILE) {
        for jj in (0..cols).step_by(TILE) {
            let i_max = (ii + TILE).min(rows);
            let j_max = (jj + TILE).min(cols);

            unsafe {
                for i in (ii..i_max).step_by(LANE) {
                    for j in (jj..j_max).step_by(LANE) {
                        let i_end = (i + LANE).min(i_max);
                        let j_end = (j + LANE).min(j_max);

                        if i_end - i == LANE && j_end - j == LANE {
                            // 8x8 micro transpose
                            let src = src.as_ptr().add(i * cols + j);
                            let dst = dst.as_mut_ptr().add(j * rows + i);
                            transpose_8x8_avx2(src, cols, dst, rows);
                        } else {
                            // remaining
                            for x in i..i_end {
                                for y in j..j_end {
                                    *dst.get_unchecked_mut(y * rows + x) =
                                        *src.get_unchecked(x * cols + y);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
/// 8×8 in-register matrix transpose micro-kernel using `AVX2`. _MY HANDS ARE UP, I JUST COPY-PASTE FROM [Here](https://docs.rs/aprender-compute/0.63.0/src/trueno/blis/transpose.rs.html#50)_
unsafe fn transpose_8x8_avx2(src: *const f32, src_stride: usize, dst: *mut f32, dst_stride: usize) {
    unsafe {
        // load 8 source rows
        let r0 = _mm256_loadu_ps(src);
        let r1 = _mm256_loadu_ps(src.add(src_stride));
        let r2 = _mm256_loadu_ps(src.add(src_stride * 2));
        let r3 = _mm256_loadu_ps(src.add(src_stride * 3));
        let r4 = _mm256_loadu_ps(src.add(src_stride * 4));
        let r5 = _mm256_loadu_ps(src.add(src_stride * 5));
        let r6 = _mm256_loadu_ps(src.add(src_stride * 6));
        let r7 = _mm256_loadu_ps(src.add(src_stride * 7));

        // interleave adjacent pairs
        let t0 = _mm256_unpacklo_ps(r0, r1); // take first 4 elements of r0 and r1
        let t1 = _mm256_unpackhi_ps(r0, r1); // take last 4 elements of r0 and r1
        let t2 = _mm256_unpacklo_ps(r2, r3); // repeat...
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        // shuffle 64-bit pairs inside 128-bit lanes using 0x44 and 0xEE
        let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
        let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
        let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
        let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
        let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

        // cross 128-bit lane permute
        let v0 = _mm256_permute2f128_ps(u0, u4, 0x20);
        let v1 = _mm256_permute2f128_ps(u1, u5, 0x20);
        let v2 = _mm256_permute2f128_ps(u2, u6, 0x20);
        let v3 = _mm256_permute2f128_ps(u3, u7, 0x20);
        let v4 = _mm256_permute2f128_ps(u0, u4, 0x31);
        let v5 = _mm256_permute2f128_ps(u1, u5, 0x31);
        let v6 = _mm256_permute2f128_ps(u2, u6, 0x31);
        let v7 = _mm256_permute2f128_ps(u3, u7, 0x31);

        // store the 8 transposed rows
        _mm256_storeu_ps(dst, v0);
        _mm256_storeu_ps(dst.add(dst_stride), v1);
        _mm256_storeu_ps(dst.add(dst_stride * 2), v2);
        _mm256_storeu_ps(dst.add(dst_stride * 3), v3);
        _mm256_storeu_ps(dst.add(dst_stride * 4), v4);
        _mm256_storeu_ps(dst.add(dst_stride * 5), v5);
        _mm256_storeu_ps(dst.add(dst_stride * 6), v6);
        _mm256_storeu_ps(dst.add(dst_stride * 7), v7);
    }
}

#[test]
fn test_mat_transpose() {
    let rows = 1024;
    let cols = 1536;
    let size = rows * cols;

    let mut noise = Noise::init();
    let mut src = vec![0.0f32; size];
    noise.fill_f32(&mut src);
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
