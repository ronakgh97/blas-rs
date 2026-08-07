//! # `blas-rs`
//!
//! Experimental BLAS kernels written in Rust for **`x86_64`**.
//!
//! This crate currently focuses on ALL Level [BLAS](https://www.netlib.org/blas/blas.pdf) kernels for `f32` only,
//! with SIMD-heavy implementations (`AVX2 only, so having AVX-512 CPU won't benefit`) where applicable.
//! All implementations are in native **RUST**, no external dependencies or FFI calls to C/C++ libraries.
//! The crate is intended for learning and experimentation purposes, not for production use.
//!
//! ## Implemented kernels
//!
//! - lvl1: `axpy`, `scal`, `copy`, `swap`, `dot`, `nrm2`, `asum`, `i_amax`, `rot`, `rotg`.
//! - lvl2: `gemv`.
//! - lvl3: `gemm`.
//!
//! ## Usage
//!
//! ```rust
//! use blas_rs::lvl1;
//!
//! let n: usize = 4;
//! let alpha: f32 = 2.0;
//! let x = vec![1.0, 2.0, 3.0, 4.0];
//! let mut y = vec![10.0, 20.0, 30.0, 40.0];
//!
//! lvl1::axpy(n, alpha, &x, 1, &mut y, 1);
//! assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
//! assert_eq!(lvl1::dot(n, &x, 1, &y, 1), 110.0);
//! ```
//!
//! ### Notes
//!
//! - APIs mirror BLAS-style signatures (`n`, raw increments, and slice buffers).
//! - Most routines panic on invalid increments (`incx == 0`, `incy == 0`, `n ==0` etc.) or insufficient slice length for the requested stride.
//! - This crate is not for a complete BLAS replacement; its purely learning focused and improve my understanding about x86, HPC, etc. So behavior and performance may change as more kernels are added.
//!
//!
//! ### References
//!
//! - [Netlib](https://www.netlib.org/blas/)
//! - [Cache blocking techniques](https://suif.stanford.edu/papers/lam-asplos91.pdf)
//! - [Intel doc](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/blas-routines.html)
//! - [intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
//!
//! ### Checkout benchmarks: [GitHub](https://github.com/ronakgh97/blas-rs/blob/master/README.md)
//!

#[cfg(not(target_arch = "x86_64"))]
compile_error!("blas_rs is only supported on x86_64 architectures");

#[cfg(all(
    not(doc),
    any(not(target_feature = "avx2"), not(target_feature = "fma"))
))]
compile_error!(
    "blas_rs requires some x86-64-v3 CPU capable features. Try compiling with `-C target-cpu=x86-64-v3`"
);

pub mod gpu;
pub mod lvl1;
pub mod lvl2;
pub mod lvl3;
pub mod utils;
