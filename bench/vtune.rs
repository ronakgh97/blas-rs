use blas_rs::lvl1::{axpy, dot};
use blas_rs::lvl2::gemv;
use blas_rs::lvl3::gemm;
use blas_rs::utils::Noise;
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn usage() {
    eprintln!("Usage: vtune <kernel> <size> [time_secs]");
    eprintln!();
    eprintln!("Kernels:");
    eprintln!("  axpy        Y = aX + Y");
    eprintln!("  dot         X . Y");
    eprintln!("  gemv        Y = aA*X + bY");
    eprintln!("  gemv_t      Y = aA^T*X + bY");
    eprintln!("  gemm_f_f    C = aAB + bC");
    eprintln!("  gemm_t_f    C = aA^TB + bC");
    eprintln!("  gemm_f_t    C = aAB^T + bC");
    eprintln!("  gemm_t_t    C = aA^TB^T + bC");
    eprintln!();
    eprintln!("size: matrix/vector dimension (N for lvl1, M=N=K for gemm)");
    eprintln!("time_secs: profiling duration in seconds (default: 10)");
    eprintln!();
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        usage();
        std::process::exit(1);
    }

    let kernel = args[1].as_str();
    let size: usize = args[2].parse().unwrap_or_else(|e| {
        eprintln!("invalid size '{}': {}", args[2], e);
        std::process::exit(1);
    });
    let time_secs: f64 = if args.len() > 3 {
        args[3].parse().unwrap_or_else(|e| {
            eprintln!("invalid time '{}': {}", args[3], e);
            std::process::exit(1);
        })
    } else {
        10.0
    };

    let mut noise = Noise::init();

    // allocate once, fill with random data
    let n2 = size * size;
    let mut a = vec![0.0f32; n2];
    let mut b = vec![0.0f32; n2];
    let mut c = vec![0.0f32; n2];
    let mut x = vec![0.0f32; size];
    let mut y = vec![0.0f32; size];

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut b);
    noise.fill_f32(&mut x);
    y.fill(1.0);
    c.fill(0.0);

    let flops_per_call: f64 = match kernel {
        "axpy" | "dot" => 2.0 * size as f64,
        "gemv" | "gemv_t" => 2.0 * size as f64 * size as f64,
        "gemm_f_f" | "gemm_t_f" | "gemm_f_t" | "gemm_t_t" => {
            2.0 * size as f64 * size as f64 * size as f64
        }
        _ => {
            eprintln!("unknown kernel: '{}'", kernel);
            usage();
            std::process::exit(1);
        }
    };

    // warmup
    eprintln!("warming up (1s)...");
    let warmup_start = Instant::now();
    let mut warmup_count: u64 = 0;
    while warmup_start.elapsed().as_secs_f64() < 1.0 {
        match kernel {
            "axpy" => {
                axpy(size, 3.0, &x, 1, black_box(&mut y), 1);
                black_box(());
            }
            "dot" => {
                black_box(dot(size, &x, 1, &y, 1));
            }
            "gemv" => {
                gemv(
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &x,
                    1,
                    7.0,
                    black_box(&mut y),
                    1,
                    false,
                );
                black_box(());
            }
            "gemv_t" => {
                gemv(
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &x,
                    1,
                    7.0,
                    black_box(&mut y),
                    1,
                    true,
                );
                black_box(());
            }
            "gemm_f_f" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    false,
                    false,
                );
                black_box(());
            }
            "gemm_t_f" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    true,
                    false,
                );
                black_box(());
            }
            "gemm_f_t" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    false,
                    true,
                );
                black_box(());
            }
            "gemm_t_t" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    true,
                    true,
                );
                black_box(());
            }
            _ => unreachable!(),
        }
        warmup_count += 1;
        // re-randomize y/c between warmup iterations so the kernel does real work
        if matches!(kernel, "axpy" | "gemv" | "gemv_t") {
            y.fill(1.0);
        }
        if matches!(kernel, "gemm_f_f" | "gemm_t_f" | "gemm_f_t" | "gemm_t_t") {
            c.fill(0.0);
            noise.fill_f32(&mut a);
            noise.fill_f32(&mut b);
        }
    }
    eprintln!(
        "warmup: {} calls in {:.1}s",
        warmup_count,
        warmup_start.elapsed().as_secs_f64()
    );

    eprintln!(
        "profiling: {} for {:.0}s (size={})...",
        kernel, time_secs, size
    );
    let start = Instant::now();
    let mut runs: u64 = 0;

    // main profiling loop
    while start.elapsed().as_secs_f64() < time_secs {
        match kernel {
            "axpy" => {
                axpy(size, 3.0, &x, 1, black_box(&mut y), 1);
                black_box(());
                y.fill(1.0);
            }
            "dot" => {
                black_box(dot(size, &x, 1, &y, 1));
            }
            "gemv" => {
                gemv(
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &x,
                    1,
                    7.0,
                    black_box(&mut y),
                    1,
                    false,
                );
                black_box(());
                y.fill(1.0);
            }
            "gemv_t" => {
                gemv(
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &x,
                    1,
                    7.0,
                    black_box(&mut y),
                    1,
                    true,
                );
                black_box(());
                y.fill(1.0);
            }
            "gemm_f_f" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    false,
                    false,
                );
                black_box(());
                c.fill(0.0);
                noise.fill_f32(&mut a);
                noise.fill_f32(&mut b);
            }
            "gemm_t_f" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    true,
                    false,
                );
                black_box(());
                c.fill(0.0);
                noise.fill_f32(&mut a);
                noise.fill_f32(&mut b);
            }
            "gemm_f_t" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    false,
                    true,
                );
                black_box(());
                c.fill(0.0);
                noise.fill_f32(&mut a);
                noise.fill_f32(&mut b);
            }
            "gemm_t_t" => {
                gemm(
                    size,
                    size,
                    size,
                    5.0,
                    &a,
                    size,
                    &b,
                    size,
                    7.0,
                    black_box(&mut c),
                    size,
                    true,
                    true,
                );
                black_box(());
                c.fill(0.0);
                noise.fill_f32(&mut a);
                noise.fill_f32(&mut b);
            }
            _ => unreachable!(),
        }
        runs += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_flops = flops_per_call * runs as f64;
    let gflops = total_flops / elapsed / 1e9;

    println!("---");
    println!("kernel:   {}", kernel);
    println!("size:     {}", size);
    println!("runs:     {}", runs);
    println!("elapsed:  {:.3}s", elapsed);
    println!("flops/c:  {:.0}", flops_per_call);
    println!("gflops:    {:.2}", gflops);
}
