use blas_rs::lvl2::gemv;
use std::panic::catch_unwind;

mod mkl_ref;
use mkl_ref::*;

#[inline]
fn make_a(m: usize, n: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * n];
    for j in 0..n {
        for i in 0..m {
            a[i + j * lda] = ((i * 31 + j * 17 + 1) as f32) * 0.1;
        }
    }
    a
}

#[test]
fn gemv_basic_and_sizes() {
    // no-transpose
    for (m, n) in [
        (1, 1),
        (2, 3),
        (3, 2),
        (7, 7),
        (8, 8),
        (9, 9),
        (16, 16),
        (32, 32),
        (128, 128),
        (1024, 1024),
    ] {
        let a = make_a(m, n, m);
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let mut y_o = vec![0.0f32; m];
        let mut y_m = vec![0.0f32; m];
        gemv(m, n, 1.0, &a, m, &x, 1, 0.0, &mut y_o, 1, false);
        mkl_gemv(m, n, 1.0, &a, m, &x, 1, 0.0, &mut y_m, 1, false);
        assert_eq_slices(&y_o, &y_m, &format!("gemv no-trans m={m} n={n}"));
    }

    // transpose
    for (m, n) in [
        (1, 1),
        (2, 3),
        (3, 2),
        (7, 7),
        (8, 8),
        (9, 9),
        (16, 16),
        (32, 32),
        (128, 128),
        (1024, 1024),
    ] {
        let a = make_a(m, n, m);
        let x: Vec<f32> = (0..m).map(|i| (i + 1) as f32 * 0.1).collect();
        let mut y_o = vec![0.0f32; n];
        let mut y_m = vec![0.0f32; n];
        gemv(m, n, 1.0, &a, m, &x, 1, 0.0, &mut y_o, 1, true);
        mkl_gemv(m, n, 1.0, &a, m, &x, 1, 0.0, &mut y_m, 1, true);
        assert_eq_slices(&y_o, &y_m, &format!("gemv trans m={m} n={n}"));
    }
}

#[test]
fn gemv_neg_stride() {
    let a = make_a(4, 3, 4);

    // neg incx, no-trans: x[2,1,0], A*x = ...
    let x = [1.0f32, 2.0, 3.0];
    let mut y_o = [0.0f32; 4];
    let mut y_m = [0.0f32; 4];
    gemv(4, 3, 1.0, &a, 4, &x, -1, 0.0, &mut y_o, 1, false);
    mkl_gemv(4, 3, 1.0, &a, 4, &x, -1, 0.0, &mut y_m, 1, false);
    assert_eq_slices(&y_o, &y_m, "gemv neg incx no-trans");

    // neg incx, trans
    let x = [1.0f32, 2.0];
    let mut y_o = [0.0f32; 3];
    let mut y_m = [0.0f32; 3];
    gemv(2, 3, 1.0, &a, 2, &x, -1, 0.0, &mut y_o, 1, true);
    mkl_gemv(2, 3, 1.0, &a, 2, &x, -1, 0.0, &mut y_m, 1, true);
    assert_eq_slices(&y_o, &y_m, "gemv neg incx trans");
}

#[test]
fn gemv_nonunit_stride() {
    let a = make_a(4, 3, 4);

    // incx=2, incy=2, no-trans
    let x = [1.0f32, 99.0, 2.0, 99.0, 3.0];
    let mut y_o = [0.0f32; 7];
    let mut y_m = [0.0f32; 7];
    gemv(4, 3, 1.0, &a, 4, &x, 2, 0.0, &mut y_o, 2, false);
    mkl_gemv(4, 3, 1.0, &a, 4, &x, 2, 0.0, &mut y_m, 2, false);
    assert_eq_slices(&y_o, &y_m, "gemv nonunit stride no-trans");

    // incx=2, incy=2, trans (m=2, n=3)
    let a2 = make_a(2, 3, 2);
    let x = [1.0f32, 99.0, 2.0];
    let mut y_o = [0.0f32; 5];
    let mut y_m = [0.0f32; 5];
    gemv(2, 3, 1.0, &a2, 2, &x, 2, 0.0, &mut y_o, 2, true);
    mkl_gemv(2, 3, 1.0, &a2, 2, &x, 2, 0.0, &mut y_m, 2, true);
    assert_eq_slices(&y_o, &y_m, "gemv nonunit stride trans");
}

#[test]
fn gemv_alpha_neg_pos() {
    let m = 5;
    let n = 4;
    let a = make_a(m, n, m);
    let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();

    // alpha=0, beta scales y
    let mut y_o = [3.0f32; 5];
    let mut y_m = [3.0f32; 5];
    gemv(m, n, 0.0, &a, m, &x, 1, 2.0, &mut y_o, 1, false);
    mkl_gemv(m, n, 0.0, &a, m, &x, 1, 2.0, &mut y_m, 1, false);
    assert_eq_slices(&y_o, &y_m, "gemv alpha=0");

    // alpha=-1, beta=1 (accumulate negative)
    let mut y_o = [1.0f32; 5];
    let mut y_m = [1.0f32; 5];
    gemv(m, n, -1.0, &a, m, &x, 1, 1.0, &mut y_o, 1, false);
    mkl_gemv(m, n, -1.0, &a, m, &x, 1, 1.0, &mut y_m, 1, false);
    assert_eq_slices(&y_o, &y_m, "gemv alpha=-1 beta=1");

    // alpha=2.5, beta=0.5
    let mut y_o = [1.0f32; 5];
    let mut y_m = [1.0f32; 5];
    gemv(m, n, 2.5, &a, m, &x, 1, 0.5, &mut y_o, 1, false);
    mkl_gemv(m, n, 2.5, &a, m, &x, 1, 0.5, &mut y_m, 1, false);
    assert_eq_slices(&y_o, &y_m, "gemv alpha=2.5 beta=0.5");

    // transposed, alpha=-1
    let mut y_o = [1.0f32; 4];
    let mut y_m = [1.0f32; 4];
    let xt: Vec<f32> = (0..m).map(|i| (i + 1) as f32 * 0.1).collect();
    gemv(m, n, -1.0, &a, m, &xt, 1, 0.0, &mut y_o, 1, true);
    mkl_gemv(m, n, -1.0, &a, m, &xt, 1, 0.0, &mut y_m, 1, true);
    assert_eq_slices(&y_o, &y_m, "gemv alpha=-1 trans");
}

#[test]
#[should_panic]
fn gemv_panic() {
    gemv(
        2,
        2,
        1.0,
        &[1.0; 4],
        2,
        &[1.0; 2],
        0,
        0.0,
        &mut [0.0; 2],
        1,
        false,
    );
}

#[test]
fn gemv_bounds_error() {
    assert!(
        catch_unwind(|| {
            gemv(
                2,
                3,
                1.0,
                &[1.0; 2],
                2,
                &[1.0; 3],
                1,
                0.0,
                &mut [0.0; 2],
                1,
                false,
            );
        })
        .is_err()
    );
    assert!(
        catch_unwind(|| {
            gemv(
                2,
                3,
                1.0,
                &[1.0; 6],
                2,
                &[1.0; 2],
                1,
                0.0,
                &mut [0.0; 2],
                1,
                false,
            );
        })
        .is_err()
    );
}
