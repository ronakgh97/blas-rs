use blas_rs::lvl1::{axpy_unsafe, dot_unsafe, nrm2_unsafe};
use blas_rs::lvl2::gemv;
use blas_rs::lvl3::gemm;
use blas_rs::utils::Noise;
use std::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};
use std::env;
use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Instant;

fn usage() {
    eprintln!("Usage: vtune <kernel> <size> [time_secs]");
    eprintln!();
    eprintln!("Kernels:");
    eprintln!("  axpy        Y = aX + Y");
    eprintln!("  dot         X . Y");
    eprintln!("  nrm2        ||X||_2^2   ");
    eprintln!("  asum        ||X||_1     ");
    eprintln!("  gemv        Y = aA*X + bY");
    eprintln!("  gemv_t      Y = aA^T*X + bY");
    eprintln!("  gemm_f_f    C = aAB + bC");
    eprintln!("  gemm_t_f    C = aA^TB + bC");
    eprintln!("  gemm_f_t    C = aAB^T + bC");
    eprintln!("  gemm_t_t    C = aA^TB^T + bC");
    eprintln!();
    eprintln!("size: matrix/vector dimension (N for lvl1, M=N=K for gemm)");
    eprintln!("time_secs: profiling duration in seconds (default: 10)");
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
    let time_secs: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10.0);

    // pin core
    let cores = core_affinity::get_core_ids().expect("Failed to get Core IDs");
    println!("Available cores: {}", cores.len());
    let core = cores[4];
    core_affinity::set_for_current(core);

    let mut noise = Noise::init();
    let n2 = size * size;
    let mut a = vec![0.0f32; n2];
    let mut b = vec![0.0f32; n2];
    let mut c = vec![0.0f32; n2];
    let mut x = vec![0.0f32; size];
    let mut y = vec![0.0f32; size];

    noise.fill_f32(&mut a);
    noise.fill_f32(&mut b);
    noise.fill_f32(&mut x);

    let flops_per_call: f64 = match kernel {
        "axpy" | "dot" => 2.0 * size as f64,
        "nrm2" => 2.0 * size as f64 + 1.0,
        "asum" => size as f64,
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

    // warmup loop
    eprintln!("warming up (3.14s)...");
    let warmup_start = Instant::now();
    let mut warmup_count: u64 = 0;
    while warmup_start.elapsed().as_secs_f64() < PI {
        match kernel {
            "axpy" => {
                let _: () = unsafe { axpy_unsafe(size, 2.0, &x, 1, &mut y, 1) };
                black_box(());
            }
            "dot" => {
                black_box(unsafe { dot_unsafe(size, &x, 1, &y, 1) });
            }
            "nrm2" => {
                black_box(unsafe { nrm2_unsafe(size, &x, 1) });
            }
            "asum" => {
                black_box(unsafe { nrm2_unsafe(size, &x, 1) });
            }
            "gemv" => {
                gemv(size, size, 2.0, &a, size, &x, 1, 3.0, &mut y, 1, false);
                black_box(());
            }
            "gemv_t" => {
                gemv(size, size, 2.0, &a, size, &x, 1, 3.0, &mut y, 1, true);
                black_box(());
            }
            "gemm_f_f" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, false, false,
                );
                black_box(());
            }
            "gemm_t_f" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, true, false,
                );
                black_box(());
            }
            "gemm_f_t" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, false, true,
                );
                black_box(());
            }
            "gemm_t_t" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, true, true,
                );
                black_box(());
            }
            _ => unreachable!(),
        }
        warmup_count += 1;
    }
    eprintln!(
        "warmup: {} calls in {:.4}s",
        warmup_count,
        warmup_start.elapsed().as_secs_f64()
    );

    // main profiling loop
    eprintln!(
        "profiling: {} for {:.0}s (size={})...",
        kernel, time_secs, size
    );

    let clock_start_mark = unsafe {
        _mm_lfence();
        _rdtsc()
    };

    let start = Instant::now();
    let mut runs: u64 = 0;

    while start.elapsed().as_secs_f64() < time_secs {
        match kernel {
            "axpy" => {
                let _: () = unsafe { axpy_unsafe(size, 2.0, &x, 1, &mut y, 1) };
                black_box(());
            }
            "dot" => {
                black_box(unsafe { dot_unsafe(size, &x, 1, &y, 1) });
            }
            "nrm2" => {
                black_box(unsafe { nrm2_unsafe(size, &x, 1) });
            }
            "asum" => {
                black_box(unsafe { nrm2_unsafe(size, &x, 1) });
            }
            "gemv" => {
                gemv(size, size, 2.0, &a, size, &x, 1, 3.0, &mut y, 1, false);
                black_box(());
            }
            "gemv_t" => {
                gemv(size, size, 2.0, &a, size, &x, 1, 3.0, &mut y, 1, true);
                black_box(());
            }
            "gemm_f_f" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, false, false,
                );
                black_box(());
            }
            "gemm_t_f" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, true, false,
                );
                black_box(());
            }
            "gemm_f_t" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, false, true,
                );
                black_box(());
            }
            "gemm_t_t" => {
                gemm(
                    size, size, size, 2.0, &a, size, &b, size, 3.0, &mut c, size, true, true,
                );
                black_box(());
            }
            _ => unreachable!(),
        }
        runs += 1;
    }

    let clock_end_mark = unsafe {
        let e = __rdtscp(&mut 0);
        _mm_lfence();
        e
    };

    let elapsed = start.elapsed().as_secs_f64();
    let total_flops = flops_per_call * runs as f64;
    let gflops = total_flops / elapsed / 1e9;
    let cycles = (clock_end_mark - clock_start_mark) as f64;

    println!("---");
    println!("kernel:   {}", kernel);
    println!("size:     {}", size);
    println!("runs:     {}", runs);
    println!("elapsed:  {:.3}s", elapsed);
    println!("flops/c:  {:.0}", flops_per_call);
    println!("cycles:   {:.0}", cycles);
    println!("gflops:   {:.3}", gflops);
}
