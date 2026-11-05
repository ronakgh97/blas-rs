mod shared;

use crate::shared::harness::{MetricSet, run_bench};
use crate::shared::utils::*;
use blas_rs::lvl1::{asum, axpy, dot, i_amax, nrm2};
use blas_rs::lvl2::gemv;
use blas_rs::lvl3::gemm;
use blas_rs::utils::*;
use clap::Parser;
use colored::Colorize;
use std::error::Error;
use std::f64::consts::PI;
use std::hint::black_box;
use std::path::{Path, PathBuf};

extern crate intel_mkl_src;

// TODO: bench is doing some weird thing, need to investigate
// Bench uses this as ref: https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark

// TODO; add arithmetic intensity
enum BenchMetrics {
    Gflops(Vec<(f64, f64)>),
    Latency(Vec<(f64, f64)>),
    CompareGflops(Vec<(f64, f64)>),
    CompareLatency(Vec<(f64, f64)>),
    CacheEfficiency(Vec<(f64, f64)>), // LLC fit proxy
    CycleCost(Vec<(f64, f64)>), // TSC delta: invariant reference cycles, not turbo core clocks.
}

#[derive(Parser)]
struct CLi {
    #[arg(long, hide = true)]
    bench: bool, // stupid fuckass cargo injects this flag when running `cargo bench`
    // and it breaks clap parsing, so we hide it
    #[arg(short, long, num_args = 1..)]
    kernels: Vec<String>,
    #[arg(short, long, default_value_t = 32)]
    warmup_count: usize,
    #[arg(short, long, default_value_t = PI)]
    target_time: f64,
}

fn main() {
    unsafe {
        std::env::set_var("MKL_NUM_THREADS", "1");
    }
    unsafe {
        std::env::set_var("BLAS_RS_NUM_THREADS", "1");
    }

    let cli = CLi::parse();
    // bench will take minimum time of `target time * sample_len * kernel_len`
    let target_time = cli.target_time;
    let size_sample = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];

    // try pinning core
    if let Some(cores) = core_affinity::get_core_ids() {
        core_affinity::set_for_current(cores[4]);
    } else {
        eprintln!("Failed to get core ids, not pinning core");
    };

    println!("Heating up!!!");
    let mut noise = Noise::rng();
    warmup(&mut noise, cli.warmup_count, 1024);

    let plot_path = PathBuf::from("bench/plots");
    std::fs::create_dir_all(&plot_path).ok();
    let plot_path = &plot_path;
    for k in cli.kernels {
        match k.trim().to_lowercase().as_str() {
            "axpy" => {
                println!("Running AXPY benchmark...");
                bench_axpy(&mut noise, &size_sample, target_time, plot_path)
            }
            "dot" => {
                println!("Running DOT benchmark...");
                bench_dot(&mut noise, &size_sample, target_time, plot_path)
            }
            "nrm2" => {
                println!("Running NRM2 benchmark...");
                bench_nrm2(&mut noise, &size_sample, target_time, plot_path)
            }
            "asum" => {
                println!("Running ASUM benchmark...");
                bench_asum(&mut noise, &size_sample, target_time, plot_path)
            }
            "i_amax" => {
                println!("Running I_AMAX benchmark...");
                bench_i_amax(&mut noise, &size_sample, target_time, plot_path)
            }
            "gemv_n" => {
                println!("Running GEMV_N benchmark...");
                bench_gemv_n(&mut noise, &size_sample, target_time, plot_path)
            }
            "gemv_t" => {
                println!("Running GEMV_T benchmark...");
                bench_gemv_t(&mut noise, &size_sample, target_time, plot_path)
            }
            "gemm_n_n" => {
                println!("Running GEMM_N_N benchmark...");
                bench_gemm_n_n(&mut noise, &size_sample, target_time, plot_path)
            }
            "gemm_t_n" => {
                println!("Running GEMM_T_N benchmark...");
                bench_gemm_t_n(&mut noise, &size_sample, target_time, plot_path)
            }
            _ => {}
        }
    }
}

fn bench_axpy(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        noise.fill_f32(&mut x_buf);

        y1_buf.fill(1.0);
        let (rc, toc, tsc) = run_bench(|| axpy(i, 3.0, &x_buf, 1, &mut y1_buf, 1), target_time);

        y2_buf.fill(1.0);
        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || axpy_intel_mkl(i as i32, 3.0, &x_buf, 1, &mut y2_buf, 1),
            target_time,
        );

        let working_kb = 3.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read/write
        let total_flops = 2.0 * i as f64 * rc; // 2n FLOPs per axpy call
        let total_flops_intel = 2.0 * i as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kb,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        // reset, avoid hot cache possibility, we don't do TRUST ME BRO BENCH
        noise.fill_f32(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("axpy.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_dot(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        noise.fill_f32(&mut x_buf);

        y1_buf.fill(1.0);
        let (rc, toc, tsc) = run_bench(
            || {
                dot(i, &x_buf, 1, &y1_buf, 1);
            },
            target_time,
        );

        y2_buf.fill(1.0);
        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                dot_intel_mkl(i as i32, &x_buf, 1, &y2_buf, 1);
            },
            target_time,
        );

        let working_kb = 2.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read/write
        let total_flops = 2.0 * i as f64 * rc; // 2n FLOPs per dot call
        let total_flops_intel = 2.0 * i as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kb,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        // reset, avoid hot cache possibility, we don't do 'TRUST ME BRO' BENCH
        noise.fill_f32(&mut x_buf);
        noise.fill_f32(&mut y1_buf);
        noise.fill_f32(&mut y2_buf);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("dot.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_nrm2(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];

        black_box(&mut x_buf);

        noise.fill_f32(&mut x_buf);

        let (rc, toc, tsc) = run_bench(
            || {
                black_box(nrm2(i, &x_buf, 1));
            },
            target_time,
        );

        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                black_box(nrm2_intel_mkl(i as i32, &x_buf, 1));
            },
            target_time,
        );

        let working_kb = i as f64 * size_of::<f32>() as f64 / 1024.0;
        let total_flops = 2.0 * i as f64 * rc;
        let total_flops_intel = 2.0 * i as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kb,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), 0.0),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        noise.fill_f32(&mut x_buf);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("nrm2.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_asum(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];

        black_box(&mut x_buf);

        noise.fill_f32(&mut x_buf);

        let (rc, toc, tsc) = run_bench(
            || {
                black_box(asum(i, &x_buf, 1));
            },
            target_time,
        );

        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                black_box(asum_intel_mkl(i as i32, &x_buf, 1));
            },
            target_time,
        );

        let working_kb = i as f64 * size_of::<f32>() as f64 / 1024.0;
        let total_flops = i as f64 * rc;
        let total_flops_intel = i as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kb,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), 0.0),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        noise.fill_f32(&mut x_buf);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("asum.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_i_amax(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];

        black_box(&mut x_buf);

        noise.fill_f32(&mut x_buf);

        let (rc, toc, tsc) = run_bench(
            || {
                black_box(i_amax(i, &x_buf, 1));
            },
            target_time,
        );

        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                black_box(i_amax_intel_mkl(i as i32, &x_buf, 1));
            },
            target_time,
        );

        let working_kb = i as f64 * size_of::<f32>() as f64 / 1024.0;
        let total_flops = i as f64 * rc;
        let total_flops_intel = i as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kb,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), 0.0),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        noise.fill_f32(&mut x_buf);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("i_amax.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_gemv_n(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        black_box(&mut a_buf);
        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);

        y1_buf.fill(1.0);
        let (rc, toc, tsc) = run_bench(
            || {
                gemv(
                    i, // m
                    i, // n
                    5.0,
                    &a_buf, // matrix
                    i,      // lda
                    &x_buf, // vec
                    1,
                    7.0,
                    &mut y1_buf, // result y
                    1,
                    false,
                );
            },
            target_time,
        );

        y2_buf.fill(1.0);
        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                gemv_intel_mkl(
                    i as i32, // m
                    i as i32, // n
                    5.0,
                    &a_buf,   // matrix
                    i as i32, // lda
                    &x_buf,   // vec
                    1,
                    7.0,
                    &mut y2_buf, // result y
                    1,
                    false,
                );
            },
            target_time,
        );

        // no of (matrix + vector) element * write/read ops (2)
        let working_kbyte = (i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64 / 1024.0;
        let total_flops = 2.0 * i.pow(2) as f64 * rc;
        let total_flops_intel = 2.0 * i.pow(2) as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kbyte,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        // reset, avoid hot cache possibility
        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("gemv_n.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_gemv_t(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        black_box(&mut a_buf);
        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);

        y1_buf.fill(1.0);
        let (rc, toc, tsc) = run_bench(
            || {
                gemv(
                    i, // m
                    i, // n
                    5.0,
                    &a_buf, // matrix
                    i,      // lda
                    &x_buf, // vec
                    1,
                    7.0,
                    &mut y1_buf, // result y
                    1,
                    true,
                );
            },
            target_time,
        );

        y2_buf.fill(1.0);
        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                gemv_intel_mkl(
                    i as i32, // m
                    i as i32, // n
                    5.0,
                    &a_buf,   // matrix
                    i as i32, // lda
                    &x_buf,   // vec
                    1,
                    7.0,
                    &mut y2_buf, // result y
                    1,
                    true,
                );
            },
            target_time,
        );

        let working_kbyte = (i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64 / 1024.0;
        let total_flops = 2.0 * i.pow(2) as f64 * rc;
        let total_flops_intel = 2.0 * i.pow(2) as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kbyte,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        // reset, avoid hot cache possibility
        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("gemv_t.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_gemm_n_n(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i * i];
        let mut y1_buf = vec![0.0f32; i * i];
        let mut y2_buf = vec![0.0f32; i * i];

        black_box(&mut a_buf);
        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);

        y1_buf.fill(1.0);
        let (rc, toc, tsc) = run_bench(
            || {
                gemm(
                    i, // m
                    i, // n
                    i, // k
                    5.0,
                    &a_buf, // matrix A
                    i,      // lda
                    &x_buf, // matrix B
                    i,      // ldb
                    7.0,
                    &mut y1_buf, // result C
                    i,           // ldc
                    false,
                    false,
                );
            },
            target_time,
        );

        y2_buf.fill(1.0);
        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                gemm_intel_mkl(
                    i as i32, // m
                    i as i32, // n
                    i as i32, // k
                    5.0,
                    &a_buf,   // matrix A
                    i as i32, // lda
                    &x_buf,   // matrix B
                    i as i32, // ldb
                    7.0,
                    &mut y2_buf, // result C
                    i as i32,    // ldc
                    false,
                    false,
                );
            },
            target_time,
        );

        let working_kbyte =
            (2.0 * i.pow(2) as f64 + 3.0 * i.pow(2) as f64) * size_of::<f32>() as f64 / 1024.0;
        let total_flops = 2.0 * i.pow(3) as f64 * rc;
        let total_flops_intel = 2.0 * i.pow(3) as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kbyte,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("gemm_n_n.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_gemm_t_n(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i * i];
        let mut y1_buf = vec![0.0f32; i * i];
        let mut y2_buf = vec![0.0f32; i * i];

        black_box(&mut a_buf);
        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);

        y1_buf.fill(1.0);
        let (rc, toc, tsc) = run_bench(
            || {
                gemm(
                    i, // m
                    i, // n
                    i, // k
                    5.0,
                    &a_buf, // matrix A
                    i,      // lda
                    &x_buf, // matrix B
                    i,      // ldb
                    7.0,
                    &mut y1_buf, // result C
                    i,           // ldc
                    true,
                    false,
                );
            },
            target_time,
        );

        y2_buf.fill(1.0);
        let (rc_intel, toc_intel, tsc_intel) = run_bench(
            || {
                gemm_intel_mkl(
                    i as i32, // m
                    i as i32, // n
                    i as i32, // k
                    5.0,
                    &a_buf,   // matrix A
                    i as i32, // lda
                    &x_buf,   // matrix B
                    i as i32, // ldb
                    7.0,
                    &mut y2_buf, // result C
                    i as i32,    // ldc
                    true,
                    false,
                );
            },
            target_time,
        );

        let working_kbyte =
            (2.0 * i.pow(2) as f64 + 3.0 * i.pow(2) as f64) * size_of::<f32>() as f64 / 1024.0;
        let total_flops = 2.0 * i.pow(3) as f64 * rc;
        let total_flops_intel = 2.0 * i.pow(3) as f64 * rc_intel;

        let (
            gflops,
            gflops_intel,
            latency,
            latency_intel,
            cache_eff,
            cyc_per_flop,
            _cyc_per_flop_intel,
        ) = MetricSet::derive(
            rc,
            rc_intel,
            toc,
            toc_intel,
            tsc,
            tsc_intel,
            total_flops,
            total_flops_intel,
            working_kbyte,
        );

        let gflops_rel = (gflops - gflops_intel) / gflops_intel * 100.0;
        let latency_rel = (latency - latency_intel) / latency_intel * 100.0;

        collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            ((i as f64).log10(), cyc_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "| {:>.0} | {:>.0} | {:>.4} | {} | {} | {:>.2}% | {:>.4} |",
            i,
            rc,
            gflops,
            fmt_rel(gflops_rel, false),
            fmt_rel(latency_rel, true),
            cache_eff,
            cyc_per_flop
        );

        noise.fill_f32(&mut a_buf);
        noise.fill_f32(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }
    collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("gemm_t_n.png")) {
        Ok(_) => println!(),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

#[inline]
fn plot_bench(bench_metrics: &[BenchMetrics], output: &Path) -> Result<(), Box<dyn Error>> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(&output, (1200, 900)).into_drawing_area();
    root.fill(&BLACK)?;

    let chart_area = root.split_evenly((3, 2)).into_iter().enumerate();

    for (idx, area) in chart_area {
        let (x_label, y_label, color, label, points): (&str, &str, RGBColor, &str, &[(f64, f64)]) =
            match bench_metrics.get(idx) {
                Some(BenchMetrics::Gflops(points)) => {
                    ("log(Size)", "GFLOPS", RED, "GFLOPS", points.as_slice())
                }
                Some(BenchMetrics::Latency(points)) => (
                    "log(Size)",
                    "log(Time per Call (ns))",
                    BLUE,
                    "Latency",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CacheEfficiency(points)) => (
                    "log(Size)",
                    "L1+L2 Fit Proxy (%)",
                    GREEN,
                    "Heuristic Cache Fit",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CycleCost(points)) => (
                    "log(Size)",
                    "TSC Cycles / FLOP",
                    CYAN,
                    "Cycle Cost",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CompareGflops(points)) => (
                    "log(Size)",
                    "GFLOPS Ratio (%)",
                    MAGENTA,
                    "GFLOPS(+-Rel) w/IntelMKL",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CompareLatency(points)) => (
                    "log(Size)",
                    "Latency Ratio (%)",
                    YELLOW,
                    "Latency(+-Rel) w/IntelMKL",
                    points.as_slice(),
                ),
                None => ("X", "Y", WHITE, "N/A", &[]),
            };

        let (x_range, y_range) = if points.is_empty() {
            (0.0..1.0, 0.0..1.0)
        } else {
            let (mut x_min, mut x_max) = (points[0].0, points[0].0);
            let (mut y_min, mut y_max) = (points[0].1, points[0].1);

            for &(x, y) in points.iter().skip(1) {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }

            let x_span = (x_max - x_min).abs();
            let y_span = (y_max - y_min).abs();
            let x_padding = if x_span == 0.0 {
                x_min.abs().max(1.0) * 0.1
            } else {
                x_span * 0.1
            };
            let y_padding = if y_span == 0.0 {
                y_min.abs().max(1.0) * 0.1
            } else {
                y_span * 0.1
            };

            (
                (x_min - x_padding)..(x_max + x_padding),
                (y_min - y_padding)..(y_max + y_padding),
            )
        };

        let mut chart = ChartBuilder::on(&area)
            .caption(label, ("0xProto Nerd Font", 20).into_font().color(&color))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .margin(20)
            .build_cartesian_2d(x_range, y_range)?;

        chart
            .configure_mesh()
            .label_style(("0xProto Nerd Font", 14).into_font().color(&WHITE))
            .axis_desc_style(("0xProto Nerd Font", 16).into_font().color(&WHITE))
            .x_desc(x_label)
            .y_desc(y_label)
            .bold_line_style(WHITE.mix(0.3))
            .light_line_style(WHITE.mix(0.15))
            .draw()?;

        if !points.is_empty() {
            chart.draw_series(LineSeries::new(points.iter().copied(), color))?;

            chart.draw_series(
                points
                    .iter()
                    .map(|(x, y)| (*x, *y))
                    .map(|pos| Circle::new(pos, 5, color.filled())),
            )?;
        }
    }

    root.present()?;

    Ok(())
}

fn fmt_rel(val: f64, inverse: bool) -> colored::ColoredString {
    let s = format!("{val:>+.4}%");

    if inverse {
        if val < 0.0 {
            s.green()
        } else if val > 0.0 {
            s.red()
        } else {
            s.normal()
        }
    } else {
        if val > 0.0 {
            s.green()
        } else if val < 0.0 {
            s.red()
        } else {
            s.normal()
        }
    }
}

fn warmup(noise: &mut Noise, run_count: usize, sample_size: usize) {
    let mut a = vec![0.0f32; sample_size * sample_size];
    let mut x = vec![0.0f32; sample_size];
    let mut y = vec![0.0f32; sample_size];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    for i in 0..run_count {
        noise.fill_f32(&mut a);
        noise.fill_f32(&mut x);
        y.fill(1.0);

        {
            axpy(
                sample_size,
                (i % 7) as f32,
                &y,
                1,
                &mut vec![0.0f32; sample_size],
                1,
            );

            dot(sample_size, &y, 1, &vec![0.0f32; sample_size], 1);

            asum(sample_size, &y, 1);

            nrm2(sample_size, &y, 1);

            i_amax(sample_size, &y, 1);

            gemv(
                sample_size,
                sample_size,
                (i % 3) as f32,
                &a,
                sample_size,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                false,
            );

            gemv(
                sample_size,
                sample_size,
                (i % 3) as f32,
                &a,
                sample_size,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                true,
            );

            gemm(
                sample_size,
                sample_size,
                sample_size,
                (i % 3) as f32,
                &a,
                sample_size,
                &a,
                sample_size,
                (i % 5) as f32,
                &mut vec![1.0f32; sample_size * sample_size],
                sample_size,
                false,
                false,
            );

            gemm(
                sample_size,
                sample_size,
                sample_size,
                (i % 3) as f32,
                &a,
                sample_size,
                &a,
                sample_size,
                (i % 5) as f32,
                &mut vec![1.0f32; sample_size * sample_size],
                sample_size,
                true,
                false,
            );
        }

        noise.fill_f32(&mut a);
        noise.fill_f32(&mut x);
        y.fill(1.0);

        {
            axpy_intel_mkl(
                sample_size as i32,
                (i % 7) as f32,
                &y,
                1,
                &mut vec![0.0f32; sample_size],
                1,
            );

            dot_intel_mkl(sample_size as i32, &y, 1, &vec![0.0f32; sample_size], 1);

            asum_intel_mkl(sample_size as i32, &y, 1);

            nrm2_intel_mkl(sample_size as i32, &y, 1);

            i_amax_intel_mkl(sample_size as i32, &y, 1);

            gemv_intel_mkl(
                sample_size as i32,
                sample_size as i32,
                (i % 3) as f32,
                &a,
                sample_size as i32,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                false,
            );

            gemv_intel_mkl(
                sample_size as i32,
                sample_size as i32,
                (i % 3) as f32,
                &a,
                sample_size as i32,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                true,
            );

            gemm_intel_mkl(
                sample_size as i32,
                sample_size as i32,
                sample_size as i32,
                (i % 3) as f32,
                &a,
                sample_size as i32,
                &a,
                sample_size as i32,
                (i % 5) as f32,
                &mut vec![1.0f32; sample_size * sample_size],
                sample_size as i32,
                false,
                false,
            );

            gemm_intel_mkl(
                sample_size as i32,
                sample_size as i32,
                sample_size as i32,
                (i % 3) as f32,
                &a,
                sample_size as i32,
                &a,
                sample_size as i32,
                (i % 5) as f32,
                &mut vec![1.0f32; sample_size * sample_size],
                sample_size as i32,
                true,
                false,
            );
        }
    }
}
