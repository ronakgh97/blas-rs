use blas_rs::lvl3::gemm;

mod mkl_ref;
use mkl_ref::*;

#[inline]
fn make_a(rows: usize, cols: usize, lda: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; lda * cols];
    for j in 0..cols {
        for i in 0..rows {
            a[i + j * lda] = ((i * 31 + j * 17 + 1) as f32) * 0.1;
        }
    }
    a
}

#[inline]
fn assert_gemm_eq(ours: &[f32], mkl: &[f32], m: usize, n: usize, ldc: usize, label: &str) {
    for j in 0..n {
        for i in 0..m {
            let idx = i + j * ldc;
            let (o, mv) = (ours[idx], mkl[idx]);
            let err = (o - mv).abs();
            let rel = if mv.abs() > 1.0 { err / mv.abs() } else { err };
            assert!(
                err <= 1e-3 || rel <= 1e-4,
                "{label}[{i},{j}]: ours={o}, mkl={mv}, err={err}"
            );
        }
    }
}

#[test]
fn gemm_basic_and_sizes() {
    for sz in [1, 2, 3, 4, 7, 8, 9, 16, 32, 64, 128, 256, 512] {
        let a = make_a(sz, sz, sz);
        let b = make_a(sz, sz, sz);
        let mut c_o = vec![0.0f32; sz * sz];
        let mut c_m = vec![0.0f32; sz * sz];
        gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_o, sz, false, false,
        );
        mkl_gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_m, sz, false, false,
        );
        assert_gemm_eq(&c_o, &c_m, sz, sz, sz, &format!("gemm n={sz}"));
    }

    // non-square
    for (m, n, k) in [(3, 7, 5), (7, 3, 5), (5, 3, 7), (1, 10, 1), (10, 1, 1)] {
        let a = make_a(m, k, m);
        let b = make_a(k, n, k);
        let mut c_o = vec![0.0f32; m * n];
        let mut c_m = vec![0.0f32; m * n];
        gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c_o, m, false, false);
        mkl_gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c_m, m, false, false);
        assert_gemm_eq(&c_o, &c_m, m, n, m, &format!("gemm {m}x{n}x{k}"));
    }
}

#[test]
fn gemm_trans_a() {
    for sz in [1, 2, 3, 7, 8, 9, 16, 32, 64, 128] {
        let a = make_a(sz, sz, sz);
        let b = make_a(sz, sz, sz);
        let mut c_o = vec![0.0f32; sz * sz];
        let mut c_m = vec![0.0f32; sz * sz];
        gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_o, sz, true, false,
        );
        mkl_gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_m, sz, true, false,
        );
        assert_gemm_eq(&c_o, &c_m, sz, sz, sz, &format!("gemm trans_a n={sz}"));
    }
}

#[test]
fn gemm_trans_b() {
    for sz in [1, 2, 3, 7, 8, 9, 16, 32, 64, 128] {
        let a = make_a(sz, sz, sz);
        let b = make_a(sz, sz, sz);
        let mut c_o = vec![0.0f32; sz * sz];
        let mut c_m = vec![0.0f32; sz * sz];
        gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_o, sz, false, true,
        );
        mkl_gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_m, sz, false, true,
        );
        assert_gemm_eq(&c_o, &c_m, sz, sz, sz, &format!("gemm trans_b n={sz}"));
    }
}

#[test]
fn gemm_trans_ab() {
    for sz in [1, 2, 3, 7, 8, 9, 16, 32, 64, 128] {
        let a = make_a(sz, sz, sz);
        let b = make_a(sz, sz, sz);
        let mut c_o = vec![0.0f32; sz * sz];
        let mut c_m = vec![0.0f32; sz * sz];
        gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_o, sz, true, true,
        );
        mkl_gemm(
            sz, sz, sz, 1.0, &a, sz, &b, sz, 0.0, &mut c_m, sz, true, true,
        );
        assert_gemm_eq(&c_o, &c_m, sz, sz, sz, &format!("gemm trans_ab n={sz}"));
    }
}

#[test]
fn gemm_lda_padding() {
    let (m, n, k, lda, ldb) = (3, 3, 3, 5, 5);
    let a = make_a(m, k, lda);
    let b = make_a(k, n, ldb);
    let mut c_o = vec![0.0f32; m * n];
    let mut c_m = vec![0.0f32; m * n];
    gemm(
        m, n, k, 1.0, &a, lda, &b, ldb, 0.0, &mut c_o, m, false, false,
    );
    mkl_gemm(
        m, n, k, 1.0, &a, lda, &b, ldb, 0.0, &mut c_m, m, false, false,
    );
    assert_gemm_eq(&c_o, &c_m, m, n, m, "gemm lda padding");

    // trans_a with padding
    let a_t = make_a(k, m, lda);
    let b2 = make_a(k, n, k);
    let mut c_o2 = vec![0.0f32; m * n];
    let mut c_m2 = vec![0.0f32; m * n];
    gemm(
        m, n, k, 1.0, &a_t, lda, &b2, k, 0.0, &mut c_o2, m, true, false,
    );
    mkl_gemm(
        m, n, k, 1.0, &a_t, lda, &b2, k, 0.0, &mut c_m2, m, true, false,
    );
    assert_gemm_eq(&c_o2, &c_m2, m, n, m, "gemm trans_a lda padding");
}

#[test]
fn gemm_alpha_neg_pos() {
    let (m, n, k) = (8, 6, 5);
    let a = make_a(m, k, m);
    let b = make_a(k, n, k);

    // alpha=0, beta scales C
    let mut c_o = vec![5.0f32; m * n];
    let mut c_m = c_o.clone();
    gemm(m, n, k, 0.0, &a, m, &b, k, 2.0, &mut c_o, m, false, false);
    mkl_gemm(m, n, k, 0.0, &a, m, &b, k, 2.0, &mut c_m, m, false, false);
    assert_gemm_eq(&c_o, &c_m, m, n, m, "gemm alpha=0");

    // beta=0, overwrite C
    let mut c_o = vec![999.0f32; m * n];
    let mut c_m = c_o.clone();
    gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c_o, m, false, false);
    mkl_gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c_m, m, false, false);
    assert_gemm_eq(&c_o, &c_m, m, n, m, "gemm beta=0");

    // accumulate
    let mut c_o: Vec<f32> = (0..(m * n)).map(|i| i as f32 * 0.01).collect();
    let mut c_m = c_o.clone();
    gemm(m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c_o, m, false, false);
    mkl_gemm(m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c_m, m, false, false);
    assert_gemm_eq(&c_o, &c_m, m, n, m, "gemm accumulate");

    // alpha=-1, beta=2
    let mut c_o: Vec<f32> = (0..(m * n)).map(|i| i as f32 * 0.01).collect();
    let mut c_m = c_o.clone();
    gemm(m, n, k, -1.0, &a, m, &b, k, 2.0, &mut c_o, m, false, false);
    mkl_gemm(m, n, k, -1.0, &a, m, &b, k, 2.0, &mut c_m, m, false, false);
    assert_gemm_eq(&c_o, &c_m, m, n, m, "gemm alpha=-1 beta=2");

    // trans_a, alpha=3, beta=-1
    let a_t = make_a(k, m, m);
    let mut c_o: Vec<f32> = (0..(m * n)).map(|i| i as f32 * 0.01).collect();
    let mut c_m = c_o.clone();
    gemm(m, n, k, 3.0, &a_t, m, &b, k, -1.0, &mut c_o, m, true, false);
    mkl_gemm(m, n, k, 3.0, &a_t, m, &b, k, -1.0, &mut c_m, m, true, false);
    assert_gemm_eq(&c_o, &c_m, m, n, m, "gemm trans_a alpha=3 beta=-1");
}

#[test]
#[should_panic]
fn gemm_panic() {
    gemm(2, 2, 2, 1.0, &[], 0, &[], 2, 0.0, &mut [], 2, false, false);
}

#[test]
fn gemm_zero_dims() {
    let mut c = vec![1.0f32; 4];
    let orig = c.clone();
    gemm(0, 2, 2, 1.0, &[], 0, &[], 0, 2.0, &mut c, 2, false, false);
    assert_eq!(c, orig);

    gemm(2, 0, 2, 1.0, &[], 0, &[], 0, 2.0, &mut c, 2, false, false);
    assert_eq!(c, orig);

    gemm(2, 2, 0, 1.0, &[], 0, &[], 0, 2.0, &mut c, 2, false, false);
    assert_eq!(c, orig);
}
