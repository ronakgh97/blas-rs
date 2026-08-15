//! Utility functions for random noise generation and matrix helper operations

use std::arch::x86_64::{
    __cpuid_count, __m256i, _mm_cvtsi128_si64, _mm_unpackhi_epi64, _mm_xor_si128, _mm256_add_epi64,
    _mm256_castsi256_si128, _mm256_cvtepi32_ps, _mm256_extract_epi64, _mm256_extracti128_si256,
    _mm256_loadu_ps, _mm256_mul_epu32, _mm256_mul_ps, _mm256_permute2f128_ps, _mm256_set_epi64x,
    _mm256_set1_epi64x, _mm256_set1_ps, _mm256_shuffle_ps, _mm256_slli_epi64, _mm256_srli_epi64,
    _mm256_storeu_ps, _mm256_storeu_si256, _mm256_unpackhi_ps, _mm256_unpacklo_ps,
    _mm256_xor_si256,
};

/// A simple random noise generator using [`splitmix64`](https://xoshiro.di.unimi.it/splitmix64.c) algorithm with AVX2 support.
/// It generates random numbers in parallel using SIMD instructions.
#[derive(Clone)]
pub struct Noise {
    state: __m256i,
}

impl Noise {
    /// Initializes the noise generator with a random seed from the OS RNG
    pub fn init() -> Self {
        let rand_seed0 = rand::random::<i64>();
        let rand_seed1 = rand::random::<i64>();
        let rand_seed2 = rand::random::<i64>();
        let rand_seed3 = rand::random::<i64>();

        Self {
            state: unsafe { _mm256_set_epi64x(rand_seed3, rand_seed2, rand_seed1, rand_seed0) },
        }
    }

    /// Initializes the noise generator with a given u64 seed
    pub fn with_seed(seed: u64) -> Self {
        let seed0 = seed as i64;
        let seed1 = seed.wrapping_mul(0x9e3779b97f4a7c15) as i64;
        let seed2 = seed.wrapping_mul(0xbf58476d1ce4e5b9) as i64;
        let seed3 = seed.wrapping_mul(0x94d049bb133111eb) as i64;

        Self {
            state: unsafe { _mm256_set_epi64x(seed3, seed2, seed1, seed0) },
        }
    }

    /// Advances the internal `__m256i` state and returns four random `u64` values in a tuple.
    #[inline(always)]
    pub fn next_u64s(&mut self) -> (u64, u64, u64, u64) {
        unsafe {
            let m0 = _mm256_set1_epi64x(0x9e3779b97f4a7c15_u64 as i64);
            let m1 = _mm256_set1_epi64x(0xbf58476d1ce4e5b9_u64 as i64);
            let m2 = _mm256_set1_epi64x(0x94d049bb133111eb_u64 as i64);

            // z = state += m0
            self.state = _mm256_add_epi64(self.state, m0);
            let mut z = self.state;

            // z = (z ^ (z >> 30)) * m1
            let shift30 = _mm256_srli_epi64(z, 30);
            z = _mm256_xor_si256(z, shift30);
            z = Self::mul_u64x4(z, m1);

            // z = (z ^ (z >> 27)) * m2
            let shift27 = _mm256_srli_epi64(z, 27);
            z = _mm256_xor_si256(z, shift27);
            z = Self::mul_u64x4(z, m2);

            // z ^ (z >> 31)
            let shift31 = _mm256_srli_epi64(z, 31);
            z = _mm256_xor_si256(z, shift31);

            // extract the four results (extracts are slower, but fine for now)
            let r0 = _mm256_extract_epi64::<0>(z) as u64;
            let r1 = _mm256_extract_epi64::<1>(z) as u64;
            let r2 = _mm256_extract_epi64::<2>(z) as u64;
            let r3 = _mm256_extract_epi64::<3>(z) as u64;

            (r0, r1, r2, r3)
        }
    }

    /// Advances the internal `__m256i` state and returns the updated state.
    #[inline(always)]
    pub fn next_state(&mut self) -> __m256i {
        unsafe {
            let m0 = _mm256_set1_epi64x(0x9e3779b97f4a7c15_u64 as i64);
            let m1 = _mm256_set1_epi64x(0xbf58476d1ce4e5b9_u64 as i64);
            let m2 = _mm256_set1_epi64x(0x94d049bb133111eb_u64 as i64);

            // z = state += m0
            self.state = _mm256_add_epi64(self.state, m0);
            let mut z = self.state;

            // z = (z ^ (z >> 30)) * m1
            let shift30 = _mm256_srli_epi64(z, 30);
            z = _mm256_xor_si256(z, shift30);
            z = Self::mul_u64x4(z, m1);

            // z = (z ^ (z >> 27)) * m2
            let shift27 = _mm256_srli_epi64(z, 27);
            z = _mm256_xor_si256(z, shift27);
            z = Self::mul_u64x4(z, m2);

            // z ^ (z >> 31)
            let shift31 = _mm256_srli_epi64(z, 31);
            z = _mm256_xor_si256(z, shift31);
            z
        }
    }

    // TODO: THIS SHIT IS SO WEIRD!!!
    /// Wrap-multiplies two `__m256i` vectors containing 4 packed `u64` integers each by `mod(2^64)`, returning `__m256i` vector with the results.
    #[inline(always)]
    fn mul_u64x4(a: __m256i, b: __m256i) -> __m256i {
        // ab = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)2^32 + [- (a_hi*b_hi)2^64 -]
        unsafe {
            // a_lo * b_lo
            let alo_blo = _mm256_mul_epu32(a, b);

            // shift both right by 32 bits to get the upper 32 bits
            let a_hi = _mm256_srli_epi64(a, 32);
            let b_hi = _mm256_srli_epi64(b, 32);

            // a_lo * b_hi
            let cross1 = _mm256_mul_epu32(a, b_hi);

            // a_hi * b_lo
            let cross2 = _mm256_mul_epu32(a_hi, b);

            // a_lo*b_hi + a_hi*b_lo
            let cross = _mm256_add_epi64(cross1, cross2);

            // shift left by 32 bits to place in the correct position
            let cross = _mm256_slli_epi64(cross, 32);

            // low 64 bits of a*b
            _mm256_add_epi64(alo_blo, cross)
        }
    }

    /// Generates a random boolean with probability `p` of being `true`
    #[inline(always)]
    pub fn bool(&mut self, p: f64) -> bool {
        if p <= 0.0 {
            return false;
        }
        if p >= 1.0 {
            return true;
        }

        let rng_u64 = unsafe {
            let curr_state = self.next_state(); // [A, B, C, D]
            let hi128 = _mm256_extracti128_si256::<1>(curr_state); // [C, D]
            let lo128 = _mm256_castsi256_si128(curr_state); // [A, B]
            let x = _mm_xor_si128(lo128, hi128); // [A ^ C, B ^ D]
            let s = _mm_unpackhi_epi64(x, x); // [B ^ D, A ^ C]
            let x = _mm_xor_si128(x, s); // [A ^ C ^ B ^ D, A ^ C ^ B ^ D]

            _mm_cvtsi128_si64(x) as u64 // grab first
        };

        // casting to f64 and scaling to 2^64,
        // then shifting left by 1 to get a threshold for comparison
        let threshold = ((p * (1u64 << 63) as f64) as u64) << 1;
        rng_u64 < threshold
    }

    /// Fills a buffer with random `f32` values in the range `[-1.0, 1.0]`
    #[inline(always)]
    pub fn fill_f32(&mut self, buf: &mut [f32]) {
        // fill the buffer in chunks of 32 floats (8 * 4 floats)
        let scale = unsafe { _mm256_set1_ps(1.0 / (1u64 << 31) as f32) };
        let mut chunks_32 = buf.chunks_exact_mut(32);
        while let Some(chunk) = chunks_32.next() {
            unsafe {
                let l0 = self.next_state();
                let l1 = self.next_state();
                let l2 = self.next_state();
                let l3 = self.next_state();

                // `_mm256_cvtepi32_ps` treats the lanes as *signed*
                // i32 already, so leaving the sign bit random gives us the full
                // [-2^31, 2^31) range, which scales to [-1.0, 1.0) below.

                // convert to f32
                let l0_f32 = _mm256_cvtepi32_ps(l0);
                let l1_f32 = _mm256_cvtepi32_ps(l1);
                let l2_f32 = _mm256_cvtepi32_ps(l2);
                let l3_f32 = _mm256_cvtepi32_ps(l3);

                // clamp to [-1.0, 1.0] by scaling down by 2^31
                let l0_f32 = _mm256_mul_ps(l0_f32, scale);
                let l1_f32 = _mm256_mul_ps(l1_f32, scale);
                let l2_f32 = _mm256_mul_ps(l2_f32, scale);
                let l3_f32 = _mm256_mul_ps(l3_f32, scale);

                // write back
                _mm256_storeu_ps(chunk.as_mut_ptr(), l0_f32);
                _mm256_storeu_ps(chunk.as_mut_ptr().add(8), l1_f32);
                _mm256_storeu_ps(chunk.as_mut_ptr().add(16), l2_f32);
                _mm256_storeu_ps(chunk.as_mut_ptr().add(24), l3_f32);
            }
        }

        // fill the remaining buffer in chunks of 8 floats (2 * 4 floats)
        let mut chunk_8 = chunks_32.into_remainder().chunks_exact_mut(8);
        while let Some(chunk) = chunk_8.next() {
            unsafe {
                let l0 = self.next_state();
                let l0_f32 = _mm256_cvtepi32_ps(l0);
                let l0_f32 = _mm256_mul_ps(l0_f32, scale);
                _mm256_storeu_ps(chunk.as_mut_ptr(), l0_f32);
            }
        }

        // fill any remaining floats (less than 8) using a single u64
        let rem = chunk_8.into_remainder();
        if !rem.is_empty() {
            let scale_scalar = 1.0 / (1u64 << 31) as f32;
            let (r0, r1, r2, r3) = self.next_u64s();

            // u64x4 -> i32x8
            let split = [
                r0 as u32 as i32,
                (r0 >> 32) as i32,
                r1 as u32 as i32,
                (r1 >> 32) as i32,
                r2 as u32 as i32,
                (r2 >> 32) as i32,
                r3 as u32 as i32,
                (r3 >> 32) as i32,
            ];

            // fill the remaining floats with scaled values
            for i in 0..rem.len() {
                rem[i] = split[i] as f32 * scale_scalar;
            }
        }
    }

    /// Fills a buffer with random bytes
    #[inline(always)]
    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        // fill the buffer in chunks of 128 bytes (4 * 32 bytes)
        let mut chunks_128 = buf.chunks_exact_mut(128);
        while let Some(chunk) = chunks_128.next() {
            unsafe {
                let l0 = self.next_state();
                let l1 = self.next_state();
                let l2 = self.next_state();
                let l3 = self.next_state();

                _mm256_storeu_si256(chunk.as_mut_ptr() as *mut __m256i, l0);
                _mm256_storeu_si256(chunk.as_mut_ptr().offset(32) as *mut __m256i, l1);
                _mm256_storeu_si256(chunk.as_mut_ptr().offset(64) as *mut __m256i, l2);
                _mm256_storeu_si256(chunk.as_mut_ptr().offset(96) as *mut __m256i, l3);
            }
        }

        // fill the remaining bytes in chunks of 32 bytes
        let mut chunks_32 = chunks_128.into_remainder().chunks_exact_mut(32);
        while let Some(chunk) = chunks_32.next() {
            unsafe {
                let l0 = self.next_state();
                _mm256_storeu_si256(chunk.as_mut_ptr() as *mut __m256i, l0);
            }
        }

        // fill any remaining bytes (less than 32) using a single u64
        let rem = chunks_32.into_remainder();
        if !rem.is_empty() {
            let (r, _, _, _) = self.next_u64s();
            let bytes = r.to_ne_bytes();
            rem.copy_from_slice(&bytes[..rem.len()]);
        }
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

/// Performs horizontal add reduction of `__m256` vector and returns as a `f32`
#[macro_export]
macro_rules! reduce_add {
    ($v:expr) => {{
        use core::arch::x86_64::*;
        // first pass; [a,b,c,d | e,f,g,h]
        let hi = _mm256_extractf128_ps($v, 1); // [e,f,g,h]
        let lo = _mm256_castps256_ps128($v); // [a,b,c,d]
        let hsum = _mm_add_ps(lo, hi); // [a+b, c+d, e+f, g+h]

        // second pass; add the two halves together
        let sum = _mm_hadd_ps(hsum, hsum); // [a+e+b+f c+g+d+h a+e+b+f c+g+d+h]
        let sum = _mm_hadd_ps(sum, sum); // [a+e+b+f+c+g+d+h,_,_,_]
        _mm_cvtss_f32(sum) // just extract one
    }};
}

/// Performs horizontal sub reduction of `__m256` vector and returns as a `f32`
#[macro_export]
macro_rules! reduce_sub {
    ($v:expr) => {{
        use core::arch::x86_64::*;
        // first pass; [a,b,c,d | e,f,g,h]
        let hi = _mm256_extractf128_ps($v, 1); // [e,f,g,h]
        let lo = _mm256_castsi256_ps($v); // [a,b,c,d]
        let hsum = _mm256_sub_ps(lo, hi); // [a-b, c-d, e-f, g-h]

        // second pass; add the two halves together
        let sum = _mm_hsub_ps(hsum, hsum); // [a-b+e-f, c-d+g-h, a-b+e-f, c-d+g-h]
        let sum = _mm_hsub_ps(sum, sum); // [a-b+e-f+c-d+g-h, _, _, _]
        _mm_cvtss_f32(sum) // just extract one
    }};
}

/// Performs horizontal min reduction of `__m256` vector and returns as a `f32`
#[macro_export]
macro_rules! reduce_min {
    ($v:expr) => {{
        use core::arch::x86_64::*;
        let hi = _mm256_extractf128_ps($v, 1); // upper 128 bit
        let lo = _mm256_castps256_ps128($v); // lower 128 bit
        let m = _mm_min_ps(lo, hi); // min of lower/upper, 256 -> 128 bit

        let shuffle = _mm_shuffle_ps(m, m, 0b01_00_11_10); // [a,b,c,d] -> [c,d,a,b]
        let m = _mm_min_ps(m, shuffle); // [x,y,x,y]

        let shuffle = _mm_shuffle_ps(m, m, 0b10_11_00_01); // [x,y,x,y] -> [y,x,y,x]
        let fm = _mm_min_ps(m, shuffle); // [m,m,m,m]
        _mm_cvtss_f32(fm) // extract first
    }};
}

/// Performs horizontal max reduction of `__m256` vector and returns as a `f32`
#[macro_export]
macro_rules! reduce_max {
    ($v:expr) => {{
        use core::arch::x86_64::*;
        let hi = _mm256_extractf128_ps($v, 1); // upper 128 bit
        let lo = _mm256_castps256_ps128($v); // lower 128 bit
        let m = _mm_max_ps(lo, hi); // max of lower/upper, 256 -> 128 bit

        let shuffle = _mm_shuffle_ps(m, m, 0b01_00_11_10); // [a,b,c,d] -> [c,d,a,b]
        let m = _mm_max_ps(m, shuffle); // [x,y,x,y]

        let shuffle = _mm_shuffle_ps(m, m, 0b10_11_00_01); // [x,y,x,y] -> [y,x,y,x]
        let fm = _mm_max_ps(m, shuffle); // [m,m,m,m]
        _mm_cvtss_f32(fm) // extract first
    }};
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
                    .unwrap_unchecked(); // iter is not empty

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
                let max = reduce_max!(load_reg); // in-reg
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
/// Returns the cache sizes (L1, L2, L3) in KB for the current `X86` CPU using CPUID
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
