use blas_rs::lvl1::*;

mod mkl_ref;
use mkl_ref::*;

// ─── axpy ───────────────────────────────────────────────────────────────────

#[test]
fn axpy_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let mut y_o = vec![0.0f32; n];
        let mut y_m = vec![0.0f32; n];
        axpy(n, 2.0, &x, 1, &mut y_o, 1);
        mkl_axpy(n, 2.0, &x, 1, &mut y_m, 1);
        assert_eq_slices(&y_o, &y_m, &format!("axpy n={n}"));
    }
}

#[test]
fn axpy_neg_stride() {
    let x = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut y_o = [0.0f32; 8];
    let mut y_m = [0.0f32; 8];
    axpy(4, 1.0, &x, -1, &mut y_o, -1);
    mkl_axpy(4, 1.0, &x, -1, &mut y_m, -1);
    assert_eq_slices(&y_o, &y_m, "axpy neg stride");
}

#[test]
fn axpy_nonunit_stride() {
    let x = [1.0f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let mut y_o = [0.0f32; 8];
    let mut y_m = [0.0f32; 8];
    axpy(4, 3.0, &x, 2, &mut y_o, 2);
    mkl_axpy(4, 3.0, &x, 2, &mut y_m, 2);
    assert_eq_slices(&y_o, &y_m, "axpy nonunit stride");
}

#[test]
fn axpy_alpha_neg_pos() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    for alpha in [-1.0, 0.0, 0.5, 2.5] {
        let mut y_o = [5.0f32; 4];
        let mut y_m = [5.0f32; 4];
        axpy(4, alpha, &x, 1, &mut y_o, 1);
        mkl_axpy(4, alpha, &x, 1, &mut y_m, 1);
        assert_eq_slices(&y_o, &y_m, &format!("axpy alpha={alpha}"));
    }
}

#[test]
#[should_panic]
fn axpy_panic() {
    let x = [1.0f32; 4];
    let mut y = [0.0f32; 4];
    axpy(4, 1.0, &x, 0, &mut y, 1);
}

// ─── scal ───────────────────────────────────────────────────────────────────

#[test]
fn scal_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let mut x_o: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let mut x_m = x_o.clone();
        scal(n, 3.0, &mut x_o, 1);
        mkl_scal(n, 3.0, &mut x_m, 1);
        assert_eq_slices(&x_o, &x_m, &format!("scal n={n}"));
    }
}

#[test]
fn scal_neg_stride() {
    let mut x_o = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x_m = x_o;
    scal(4, -1.0, &mut x_o, -1);
    mkl_scal(4, -1.0, &mut x_m, -1);
    assert_eq_slices(&x_o, &x_m, "scal neg stride");
}

#[test]
fn scal_nonunit_stride() {
    let mut x_o = [1.0f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let mut x_m = x_o;
    scal(4, 5.0, &mut x_o, 2);
    mkl_scal(4, 5.0, &mut x_m, 2);
    assert_eq_slices(&x_o, &x_m, "scal nonunit stride");
}

#[test]
fn scal_alpha_neg_pos() {
    for alpha in [-1.0, 0.0, 0.5, 2.5] {
        let mut x_o = [1.0f32, 2.0, 3.0, 4.0];
        let mut x_m = x_o;
        scal(4, alpha, &mut x_o, 1);
        mkl_scal(4, alpha, &mut x_m, 1);
        assert_eq_slices(&x_o, &x_m, &format!("scal alpha={alpha}"));
    }
}

#[test]
#[should_panic]
fn scal_panic() {
    let mut x = [1.0f32; 4];
    scal(4, 2.0, &mut x, 0);
}

// ─── copy ───────────────────────────────────────────────────────────────────

#[test]
fn copy_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y_o = vec![0.0f32; n];
        let mut y_m = vec![0.0f32; n];
        copy(n, &x, 1, &mut y_o, 1);
        mkl_copy(n, &x, 1, &mut y_m, 1);
        assert_eq_slices(&y_o, &y_m, &format!("copy n={n}"));
    }
}

#[test]
fn copy_neg_stride() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y_o = [0.0f32; 4];
    let mut y_m = [0.0f32; 4];
    copy(4, &x, -1, &mut y_o, -1);
    mkl_copy(4, &x, -1, &mut y_m, -1);
    assert_eq_slices(&y_o, &y_m, "copy neg stride");
}

#[test]
fn copy_nonunit_stride() {
    let x = [1.0f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let mut y_o = [0.0f32; 8];
    let mut y_m = [0.0f32; 8];
    copy(4, &x, 2, &mut y_o, 2);
    mkl_copy(4, &x, 2, &mut y_m, 2);
    assert_eq_slices(&y_o, &y_m, "copy nonunit stride");
}

#[test]
#[should_panic]
fn copy_panic() {
    let x = [1.0f32; 4];
    let mut y = [0.0f32; 4];
    copy(4, &x, 0, &mut y, 1);
}

// ─── swap ───────────────────────────────────────────────────────────────────

#[test]
fn swap_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 128, 1024] {
        let mut x_o: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y_o: Vec<f32> = (0..n).map(|i| (i + 100) as f32).collect();
        let mut x_m = x_o.clone();
        let mut y_m = y_o.clone();
        swap(n, &mut x_o, 1, &mut y_o, 1);
        mkl_swap(n, &mut x_m, 1, &mut y_m, 1);
        assert_eq_slices(&x_o, &x_m, &format!("swap_x n={n}"));
        assert_eq_slices(&y_o, &y_m, &format!("swap_y n={n}"));
    }
}

#[test]
fn swap_neg_stride() {
    let mut x_o = [1.0f32, 2.0, 3.0, 4.0];
    let mut y_o = [5.0f32, 6.0, 7.0, 8.0];
    let mut x_m = x_o;
    let mut y_m = y_o;
    swap(4, &mut x_o, -1, &mut y_o, -1);
    mkl_swap(4, &mut x_m, -1, &mut y_m, -1);
    assert_eq_slices(&x_o, &x_m, "swap neg stride x");
    assert_eq_slices(&y_o, &y_m, "swap neg stride y");
}

#[test]
fn swap_nonunit_stride() {
    let mut x_o = [1.0f32, 99.0, 2.0, 99.0, 3.0];
    let mut y_o = [5.0f32, 99.0, 6.0, 99.0, 7.0];
    let mut x_m = x_o;
    let mut y_m = y_o;
    swap(3, &mut x_o, 2, &mut y_o, 2);
    mkl_swap(3, &mut x_m, 2, &mut y_m, 2);
    assert_eq_slices(&x_o, &x_m, "swap nonunit stride x");
    assert_eq_slices(&y_o, &y_m, "swap nonunit stride y");
}

#[test]
#[should_panic]
fn swap_panic() {
    let mut x = [1.0f32; 4];
    let mut y = [0.0f32; 4];
    swap(4, &mut x, 0, &mut y, 1);
}

// ─── dot ────────────────────────────────────────────────────────────────────

#[test]
fn dot_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let y: Vec<f32> = (0..n).map(|i| (i + 2) as f32 * 0.1).collect();
        assert_eq_f32(
            dot(n, &x, 1, &y, 1),
            mkl_dot(n, &x, 1, &y, 1),
            &format!("dot n={n}"),
        );
    }
}

#[test]
fn dot_neg_stride() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = [5.0f32, 6.0, 7.0, 8.0];
    assert_eq_f32(
        dot(4, &x, -1, &y, -1),
        mkl_dot(4, &x, -1, &y, -1),
        "dot neg stride",
    );
}

#[test]
fn dot_nonunit_stride() {
    let x = [1.0f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let y = [4.0f32, 99.0, 5.0, 99.0, 6.0, 99.0, 7.0];
    assert_eq_f32(
        dot(4, &x, 2, &y, 2),
        mkl_dot(4, &x, 2, &y, 2),
        "dot nonunit stride",
    );
}

#[test]
#[should_panic]
fn dot_panic() {
    let x = [1.0f32; 4];
    let y = [1.0f32; 4];
    dot(4, &x, 0, &y, 1);
}

// ─── nrm2 ───────────────────────────────────────────────────────────────────

#[test]
fn nrm2_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        assert_eq_f32(nrm2(n, &x, 1), mkl_nrm2(n, &x, 1), &format!("nrm2 n={n}"));
    }
}

#[test]
fn nrm2_neg_stride() {
    let x = [4.0f32, 3.0, 2.0, 1.0];
    assert_eq_f32(nrm2(4, &x, -1), mkl_nrm2(4, &x, -1), "nrm2 neg stride");
}

#[test]
#[should_panic]
fn nrm2_panic() {
    let x = [1.0f32; 2];
    nrm2(4, &x, 1);
}

// ─── asum ───────────────────────────────────────────────────────────────────

#[test]
fn asum_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32) - (n as f32) / 2.0) * 0.1)
            .collect();
        assert_eq_f32(asum(n, &x, 1), mkl_asum(n, &x, 1), &format!("asum n={n}"));
    }
}

#[test]
fn asum_neg_stride() {
    let x = [4.0f32, 3.0, 2.0, 1.0];
    assert_eq_f32(asum(4, &x, -1), mkl_asum(4, &x, -1), "asum neg stride");
}

#[test]
#[should_panic]
fn asum_panic() {
    let x = [1.0f32; 4];
    asum(4, &x, 0);
}

// ─── i_amax ─────────────────────────────────────────────────────────────────

#[test]
fn i_amax_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        assert_eq!(i_amax(n, &x, 1), mkl_i_amax(n, &x, 1), "i_amax n={n}");
    }
}

#[test]
fn i_amax_neg_stride() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    // our impl returns 0-based index of max abs; MKL returns 1-based converted to 0-based
    // both should agree on the actual array index of max abs element
    let ours = i_amax(4, &x, -1);
    let mkl = mkl_i_amax(4, &x, -1);
    assert_eq!(ours, mkl, "i_amax neg stride: ours={ours}, mkl={mkl}");
}

#[test]
#[should_panic]
fn i_amax_panic() {
    let x = [1.0f32; 4];
    i_amax(0, &x, 1);
}

// ─── i_amin ─────────────────────────────────────────────────────────────────

#[test]
fn i_amin_basic_and_sizes() {
    // no MKL isamin; test against our own definition
    for n in [1, 7, 8, 9, 16, 32, 128, 256, 1024] {
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let expected = x
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        assert_eq!(i_amin(n, &x, 1), expected, "i_amin n={n}");
    }
}

#[test]
#[should_panic]
fn i_amin_panic() {
    let x = [1.0f32; 4];
    i_amin(0, &x, 1);
}

// ─── rot ────────────────────────────────────────────────────────────────────

#[test]
fn rot_basic_and_sizes() {
    for n in [1, 7, 8, 9, 16, 128, 1024] {
        let mut x_o: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let mut y_o: Vec<f32> = (0..n).map(|i| (i + 10) as f32 * 0.1).collect();
        let mut x_m = x_o.clone();
        let mut y_m = y_o.clone();
        rot(n, &mut x_o, 1, &mut y_o, 1, 0.6, 0.8);
        mkl_rot(n, &mut x_m, 1, &mut y_m, 1, 0.6, 0.8);
        assert_eq_slices(&x_o, &x_m, &format!("rot_x n={n}"));
        assert_eq_slices(&y_o, &y_m, &format!("rot_y n={n}"));
    }
}

#[test]
fn rot_neg_stride() {
    let mut x_o = [1.0f32, 2.0, 3.0, 4.0];
    let mut y_o = [4.0f32, 3.0, 2.0, 1.0];
    let mut x_m = x_o;
    let mut y_m = y_o;
    rot(4, &mut x_o, -1, &mut y_o, -1, 0.6, 0.8);
    mkl_rot(4, &mut x_m, -1, &mut y_m, -1, 0.6, 0.8);
    assert_eq_slices(&x_o, &x_m, "rot neg stride x");
    assert_eq_slices(&y_o, &y_m, "rot neg stride y");
}

#[test]
fn rot_nonunit_stride() {
    let mut x_o = [1.0f32, 99.0, 2.0, 99.0, 3.0];
    let mut y_o = [4.0f32, 99.0, 5.0, 99.0, 6.0];
    let mut x_m = x_o;
    let mut y_m = y_o;
    rot(3, &mut x_o, 2, &mut y_o, 2, 0.6, 0.8);
    mkl_rot(3, &mut x_m, 2, &mut y_m, 2, 0.6, 0.8);
    assert_eq_slices(&x_o, &x_m, "rot nonunit stride x");
    assert_eq_slices(&y_o, &y_m, "rot nonunit stride y");
}

#[test]
#[should_panic]
fn rot_panic() {
    let mut x = [1.0f32; 2];
    let mut y = [1.0f32; 2];
    rot(2, &mut x, 0, &mut y, 1, 1.0, 0.0);
}

// ─── rotg ───────────────────────────────────────────────────────────────────

#[test]
fn rotg_basic_and_sizes() {
    let cases = [
        (3.0f32, 4.0f32),
        (0.0, 4.0),
        (3.0, 0.0),
        (0.0, 0.0),
        (-3.0, 4.0),
        (4.0, 3.0),
        (5.0, -12.0),
        (1.0, 1.0),
        (-7.0, -24.0),
    ];
    for (a_in, b_in) in cases {
        let (mut ao, mut bo, mut co, mut so) = (a_in, b_in, 0.0, 0.0);
        let (mut am, mut bm, mut cm, mut sm) = (a_in, b_in, 0.0, 0.0);
        rotg(&mut ao, &mut bo, &mut co, &mut so);
        mkl_rotg(&mut am, &mut bm, &mut cm, &mut sm);
        assert_eq_f32(ao, am, &format!("rotg({a_in},{b_in}) a"));
        assert_eq_f32(bo, bm, &format!("rotg({a_in},{b_in}) b"));
        assert_eq_f32(co, cm, &format!("rotg({a_in},{b_in}) c"));
        assert_eq_f32(so, sm, &format!("rotg({a_in},{b_in}) s"));
    }
}
