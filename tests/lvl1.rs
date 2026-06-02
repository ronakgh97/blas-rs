use blas_rs::lvl1::*;
use blas_rs::utils::Noise;

#[test]
fn test_axpy() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut y = vec![0.0; 8];

    axpy(8, 2.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    y.fill(0.0);

    axpy(8, 2.0, &x, 1, &mut y, -1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);

    axpy(8, 2.0, &x, -1, &mut y, 1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);

    axpy(4, 2.0, &x, 2, &mut y, 2);
    assert_eq!(y, vec![0.0, 0.0, 4.0, 0.0, 8.0, 0.0, 12.0, 0.0]);
    y.fill(0.0);

    axpy(0, 2.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
    y.fill(0.0);

    axpy(8, 0.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
    y.fill(0.0);

    axpy(8, -1.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]);
    y.fill(0.0);

    let mut gen_x = vec![0.0f32; 8192];
    let mut gen_y = vec![0.0f32; 8192];

    let mut noise = Noise::init();
    noise.fill_f32(&mut gen_x);
    noise.fill_f32(&mut gen_y);

    let r = 1024;

    let start = std::time::Instant::now();
    for _ in 0..r {
        axpy(gen_x.len(), 4.0, &gen_x, 1, &mut gen_y, 1);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let gflops = (2.0 * gen_x.len() as f64 * r as f64) / (elapsed * 1e9);

    println!(
        "Elapsed time: {:.6} seconds, GFLOPS: {:.2}",
        elapsed, gflops
    );
}

#[test]
#[should_panic]
fn test_panic_axpy() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    axpy(4, 2.0, &x, 0, &mut y, 1);

    let mut y = vec![0.0; 4];
    axpy(4, 2.0, &x, 1, &mut y, 0);

    let x = vec![1.0, 2.0];
    let mut y = vec![0.0; 4];
    axpy(4, 2.0, &x, 1, &mut y, 1);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 2];
    axpy(4, 2.0, &x, 1, &mut y, 1);
}

#[test]
fn test_scal() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 2.0, &mut x, 1);
    assert_eq!(x, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 1.0, &mut x, 1);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(5, -1.0, &mut x, -1);
    assert_eq!(x, vec![-1.0, -2.0, -3.0, -4.0, -5.0, 6.0, 7.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(4, 3.0, &mut x, 2);
    assert_eq!(x, vec![3.0, 2.0, 9.0, 4.0, 15.0, 6.0, 21.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(0, 2.0, &mut x, 1);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 0.0, &mut x, 1);
    assert_eq!(x, vec![0.0; 8]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 0.5, &mut x, 1);
    assert_eq!(x, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
}

#[test]
#[should_panic]
fn test_scal_panic() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    scal(4, 2.0, &mut x, 0);

    let mut x = vec![1.0, 2.0];
    scal(4, 2.0, &mut x, 1);
}

#[test]
fn test_copy() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut y = vec![0.0; 8];
    copy(8, &x, 1, &mut y, 1);
    assert_eq!(y, x);

    let mut y = vec![0.0; 8];
    copy(8, &x, -1, &mut y, 1);
    assert_eq!(y, vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

    let mut y = vec![0.0; 8];
    copy(4, &x, 2, &mut y, 2);
    assert_eq!(y, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(4, &x, -2, &mut y, 2);
    assert_eq!(y, vec![7.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(4, &x, 2, &mut y, -1);
    assert_eq!(y, vec![7.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(3, &x, -1, &mut y, -1);
    assert_eq!(y, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
}

#[test]
#[should_panic]
fn test_panic_copy() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    copy(4, &x, 0, &mut y, 1);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    copy(4, &x, 1, &mut y, 0);

    let x = vec![1.0, 2.0];
    let mut y = vec![0.0; 4];
    copy(4, &x, 1, &mut y, 1);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 2];
    copy(4, &x, 1, &mut y, 1);
}

#[test]
fn test_swap() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    swap(8, &mut x, 1, &mut y, 1);
    assert_eq!(x, vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    swap(4, &mut x, 2, &mut y, 2);
    assert_eq!(x, vec![9.0, 2.0, 11.0, 4.0, 13.0, 6.0, 15.0, 8.0]);
    assert_eq!(y, vec![1.0, 10.0, 3.0, 12.0, 5.0, 14.0, 7.0, 16.0]);

    x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    swap(0, &mut x, 1, &mut y, 1);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y, vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

    let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (x_slice, y_slice) = data.split_at_mut(4);
    swap(4, x_slice, 1, y_slice, 1);
    assert_eq!(x_slice, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y_slice, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[should_panic]
fn test_panic_swap() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    swap(4, &mut x, 0, &mut y, 1);

    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    swap(4, &mut x, 1, &mut y, 0);

    let mut x = vec![1.0, 2.0];
    let mut y = vec![0.0; 4];
    swap(4, &mut x, 1, &mut y, 1);

    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 2];
    swap(4, &mut x, 1, &mut y, 1);
}

#[test]
fn test_dot() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    let res = dot(8, &x, 1, &y, 1);
    let exp = dot_checker(&x, &y);
    assert_eq!(res, exp);

    let res = dot(4, &x, 2, &y, 2);
    let exp = x[0] * y[0] + x[2] * y[2] + x[4] * y[4] + x[6] * y[6];
    assert_eq!(res, exp);

    let res = dot(0, &x, 1, &y, 1);
    assert_eq!(res, 0.0);

    let res = dot(4, &x, -1, &y, 1);
    let exp = x[3] * y[0] + x[2] * y[1] + x[1] * y[2] + x[0] * y[3];
    assert_eq!(res, exp);

    let res = dot(4, &x, 1, &y, -1);
    let exp = x[0] * y[3] + x[1] * y[2] + x[2] * y[1] + x[3] * y[0];
    assert_eq!(res, exp);

    let mut gen_x = vec![0.0f32; 8192];
    let mut gen_y = vec![0.0f32; 8192];

    let mut noise = Noise::init();

    noise.fill_f32(&mut gen_x);
    noise.fill_f32(&mut gen_y);

    let r = 1024;

    let start = std::time::Instant::now();
    for _ in 0..r {
        dot(gen_x.len(), &gen_x, 1, &gen_y, 1);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let gflops = ((2.0 * gen_x.len() as f64 - 1.0) * r as f64) / (elapsed * 1e9);

    println!(
        "Elapsed time: {:.6} seconds, GFLOPS: {:.2}",
        elapsed, gflops
    );
}

#[test]
#[should_panic]
fn test_panic_dot() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    dot(4, &x, 0, &y, 1);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    dot(4, &x, 1, &y, 0);

    let x = vec![1.0, 2.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    dot(4, &x, 1, &y, 1);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 2.0];
    dot(4, &x, 1, &y, 1);
}

fn dot_checker(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let common = a.len().min(b.len());
    for i in 0..common {
        sum += a[i] * b[i];
    }
    sum
}

#[test]
fn test_nrm2() {
    let x = vec![3.0, 4.0];
    let res = nrm2(2, &x, 1);
    assert_eq!(res, 5.0);

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = nrm2(8, &x, 1);
    let exp: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert_eq!(res, exp);

    let res = nrm2(0, &x, 1);
    assert_eq!(res, 0.0);

    let x = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let res = nrm2(4, &x, 2);
    let exp = (1.0_f32.powi(2) + 2.0_f32.powi(2) + 3.0_f32.powi(2) + 4.0_f32.powi(2)).sqrt();
    assert_eq!(res, exp);

    let x = vec![4.0, 3.0, 2.0, 1.0];
    let res = nrm2(4, &x, -1);
    let exp = (4.0_f32.powi(2) + 3.0_f32.powi(2) + 2.0_f32.powi(2) + 1.0_f32.powi(2)).sqrt();
    assert_eq!(res, exp);
}

#[test]
#[should_panic]
fn test_nrm2_panic() {
    let x = vec![1.0, 2.0];
    nrm2(4, &x, 1);
}

#[test]
fn test_asum() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = asum(8, &x, 1);
    let exp: f32 = x.iter().map(|v| v.abs()).sum();
    assert_eq!(res, exp);

    let x = vec![-1.0, -2.0, -3.0, -4.0];
    let res = asum(4, &x, 1);
    assert_eq!(res, 10.0);

    let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let res = asum(8, &x, 1);
    let exp: f32 = x.iter().map(|v| v.abs()).sum();
    assert_eq!(res, exp);

    let x = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let res = asum(4, &x, 2);
    assert_eq!(res, 10.0);

    let x = vec![4.0, 3.0, 2.0, 1.0];
    let res = asum(4, &x, -1);
    assert_eq!(res, 10.0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = asum(0, &x, 1);
    assert_eq!(res, 0.0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = asum(4, &x, 1);
    assert_eq!(res, 10.0);
}

#[test]
#[should_panic]
fn test_asum_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    asum(4, &x, 0);
    let x = vec![1.0, 2.0];
    asum(4, &x, 1);
}

#[test]
fn test_i_amax() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let res = i_amax(4, &x, 2);
    assert_eq!(res, 6);

    let x = vec![4.0, 3.0, 2.0, 1.0];
    let res = i_amax(4, &x, -1);
    assert_eq!(res, 0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = i_amax(0, &x, 1);
    assert_eq!(res, 0);

    let x = vec![5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);
}

#[test]
#[should_panic]
fn test_i_amax_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    i_amax(4, &x, 0);
    let x = vec![1.0, 2.0];
    i_amax(4, &x, 1);
}

#[test]
fn test_i_amin() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0];
    let res = i_amin(4, &x, 2);
    assert_eq!(res, 6);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = i_amin(4, &x, -1);
    assert_eq!(res, 0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = i_amin(0, &x, 1);
    assert_eq!(res, 0);

    let x = vec![5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![2.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 1);

    let x = vec![-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![-8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0];
    let res = i_amin(8, &x, 1);
    assert_eq!(res, 0);
}

#[test]
#[should_panic]
fn test_i_amin_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    i_amin(4, &x, 0);
    let x = vec![1.0, 2.0];
    i_amin(4, &x, 1);
}

#[test]
fn test_rot() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let orig_x = x.clone();
    let orig_y = y.clone();
    let (c, s) = (1.0, 0.0);

    rot(8, &mut x, 1, &mut y, 1, c, s);
    assert_eq!(x, orig_x);
    assert_eq!(y, orig_y);
    x.fill(0.0);
    y.fill(0.0);

    x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (c, s) = (-1.0, 0.0);

    rot(8, &mut x, 1, &mut y, 1, c, s);
    assert_eq!(x, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
    assert_eq!(y, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
    x.fill(0.0);
    y.fill(0.0);

    x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (c, s) = (0.0, 1.0);

    rot(8, &mut x, 1, &mut y, 1, c, s);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
    x.fill(0.0);
    y.fill(0.0);

    x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    rot(4, &mut x, 2, &mut y, 2, 1.0, 0.0);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
#[should_panic]
fn test_rot_panic() {
    let mut x = vec![1.0, 2.0];
    let mut y = vec![1.0, 2.0];
    rot(0, &mut x, 1, &mut y, 1, 1.0, 0.0);
    let mut x = vec![1.0, 2.0];
    let mut y = vec![1.0, 2.0];
    rot(2, &mut x, 0, &mut y, 1, 1.0, 0.0);

    let mut x = vec![1.0, 2.0];
    let mut y = vec![1.0, 2.0];
    rot(2, &mut x, 1, &mut y, 0, 1.0, 0.0);

    let mut x = vec![1.0, 2.0];
    let mut y = vec![1.0, 2.0, 3.0, 4.0];
    rot(4, &mut x, 1, &mut y, 1, 1.0, 0.0);

    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![1.0, 2.0];
    rot(4, &mut x, 1, &mut y, 1, 1.0, 0.0);
}

#[test]
fn test_rotg() {
    let mut a = 3.0f32;
    let mut b = 4.0f32;
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    rotg(&mut a, &mut b, &mut c, &mut s);
    assert_eq!(a, 5.0);
    assert_eq!(c, 0.6);
    assert_eq!(s, 0.8);
    assert!((b - 1.6666666).abs() < 1e-5);

    let mut a = 0.0f32;
    let mut b = 4.0f32;
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    rotg(&mut a, &mut b, &mut c, &mut s);
    assert_eq!(a, 4.0);
    assert_eq!(c, 0.0);
    assert_eq!(s, 1.0);
    assert_eq!(b, 1.0);

    let mut a = 3.0f32;
    let mut b = 0.0f32;
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    rotg(&mut a, &mut b, &mut c, &mut s);
    assert_eq!(a, 3.0);
    assert_eq!(c, 1.0);
    assert_eq!(s, 0.0);
    assert_eq!(b, 0.0);

    let mut a = 0.0f32;
    let mut b = 0.0f32;
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    rotg(&mut a, &mut b, &mut c, &mut s);
    assert_eq!(c, 1.0);
    assert_eq!(s, 0.0);
    assert_eq!(a, 0.0);
    assert_eq!(b, 0.0);

    let mut a = -3.0f32;
    let mut b = 4.0f32;
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    rotg(&mut a, &mut b, &mut c, &mut s);
    assert_eq!(a, 5.0);
    assert_eq!(c, -0.6);
    assert_eq!(s, 0.8);

    let mut a = 4.0f32;
    let mut b = 3.0f32;
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    rotg(&mut a, &mut b, &mut c, &mut s);
    assert_eq!(a, 5.0);
    assert_eq!(c, 0.8);
    assert_eq!(s, 0.6);
    assert_eq!(b, 0.6);

    let mut a = 3.0f32;
    let mut b = 4.0f32;
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    rotg(&mut a, &mut b, &mut c, &mut s);
    let r = a;
    assert!((c * c + s * s - 1.0).abs() < 1e-5, "c^2 + s^2 should be ~1");
    assert!((c * r - 3.0).abs() < 1e-5, "c*r should be ~3");
    assert!((s * r - 4.0).abs() < 1e-5, "s*r should be ~4");
}
