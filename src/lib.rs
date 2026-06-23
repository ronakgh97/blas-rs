//! # `blas-rs`
//!
//! Experimental BLAS kernels written in Rust for **`x86_64`**.
//!
//! This crate currently focuses on ALL Level BLAS operations for `f32` only,
//! with SIMD-heavy implementations (`AVX2 only, so having AVX-512 cpu won't benefit`) where applicable.
//!
//! ## Modules Structure
//!
//! - [`lvl1`]: implements vector-vector and vector-scalar routines.
//! - [`lvl2`]: implements for matrix-vector routines.
//! - [`lvl3`]: implements for matrix-matrix routines.
//! - [`utils`]: internal helpers used by kernels and tests.
//!
//! ## Implemented kernels [WIP](https://github.com/ronakgh97/blas_rs)
//!
//! - lvl1: `axpy`, `scal`, `copy`, `swap`, `dot`, `nrm2`, `asum`, `i_amax`, `rot`, `rotg`.
//! - lvl2: `gemv`
//! - lvl3: `gemm`
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
//! unsafe { lvl1::axpy_unsafe(n, alpha, &x, 1, &mut y, 1) };
//! assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
//! assert_eq!(unsafe { lvl1::dot_unsafe(n, &x, 1, &y, 1) }, 110.0);
//! ```
//!
//! ### Notes
//!
//! - APIs mirror BLAS-style signatures (`n`, raw increments, and slice buffers).
//! - Most routines panic on invalid increments (`incx == 0`, `incy == 0`, `n ==0` etc.) or insufficient slice length for the requested stride. Alternatively there are unsafe versions of the routines that bypass for performance.
//! - This crate is not for a complete BLAS replacement; its purely learning focused and improve my understanding about x86, HPC, etc. So behavior and performance may change as more kernels are added.
//!
//!
//! ### Ref
//! - [Netlib](https://www.netlib.org/blas/)
//! - [Intel doc](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/blas-routines.html)
//! - [intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
//!
//! ### Checkout benchmarks: [GitHub](https://github.com/ronakgh97/blas_rs)
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

pub mod lvl1;
pub mod lvl2;
pub mod lvl3;
pub mod utils;
