mod harness;
mod utils;

use crate::harness::{MetricSet, run_bench};
use crate::utils::*;
use blas_rs::lvl1::{asum, axpy, dot, i_amax, nrm2};
use blas_rs::lvl2::gemv;
use blas_rs::lvl3::gemm;
use blas_rs::utils::*;
use std::error::Error;

use colored::Colorize;
use std::hint::black_box;
use std::path::Path;

extern crate intel_mkl_src;

/// Format relative performance metrics with color coding
#[inline(always)]
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

// TODO: bench is doing some weird thing, need to investigate
// Bench uses this as ref: https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark

fn warmup(noise: &mut Noise, r: usize, s: usize) {
    let mut a = vec![0.0f32; s * s];
    let mut x = vec![0.0f32; s];
    let mut y = vec![0.0f32; s];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    // Warmup phase
    for i in 0..r {
        noise.fill_f32(&mut a);
        noise.fill_f32(&mut x);
        y.fill(1.0);

        {
            axpy(s, (i % 7) as f32, &y, 1, &mut vec![0.0f32; s], 1);

            dot(s, &y, 1, &vec![0.0f32; s], 1);

            gemv(
                s,
                s,
                (i % 3) as f32,
                &a,
                s,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                false,
            );

            gemv(
                s,
                s,
                (i % 3) as f32,
                &a,
                s,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                true,
            );

            gemm(
                s,
                s,
                s,
                (i % 3) as f32,
                &a,
                s,
                &a,
                s,
                (i % 5) as f32,
                &mut vec![1.0f32; s * s],
                s,
                false,
                false,
            );
        }

        noise.fill_f32(&mut a);
        noise.fill_f32(&mut x);
        y.fill(1.0);

        {
            axpy_intel_mkl(s as i32, (i % 7) as f32, &y, 1, &mut vec![0.0f32; s], 1);

            dot_intel_mkl(s as i32, &y, 1, &vec![0.0f32; s], 1);

            gemv_intel_mkl(
                s as i32,
                s as i32,
                (i % 3) as f32,
                &a,
                s as i32,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                false,
            );

            gemv_intel_mkl(
                s as i32,
                s as i32,
                (i % 3) as f32,
                &a,
                s as i32,
                &x,
                1,
                (i % 5) as f32,
                &mut y,
                1,
                true,
            );

            gemm_intel_mkl(
                s as i32,
                s as i32,
                s as i32,
                (i % 3) as f32,
                &a,
                s as i32,
                &a,
                s as i32,
                (i % 5) as f32,
                &mut vec![1.0f32; s * s],
                s as i32,
                false,
                false,
            );
        }
    }
}

// TODO; add arithmetic intensity
enum BenchMetrics {
    Gflops(Vec<(f64, f64)>),
    Latency(Vec<(f64, f64)>),
    CompareGflops(Vec<(f64, f64)>),
    CompareLatency(Vec<(f64, f64)>),
    CacheEfficiency(Vec<(f64, f64)>), // LLC fit proxy
    CycleCost(Vec<(f64, f64)>), // TSC delta: invariant reference cycles, not turbo core clocks.
}

fn main() {
    unsafe {
        std::env::set_var("MKL_NUM_THREADS", "1");
    }

    // bit yap about axpy and dot, axpy is slightly overhead, due to ` _mm256_storeu_ps`, write heavy,
    // so its write-bandwidth bound, even it whole vectors fits in cache and also "cache" does not help much since those 8 lanes are not reused.
    // while dot is just most read heavy, and accumulated in 4 sum register, less overhead, finally the `cache miss` is heuristic metrics as said earlier

    // BENCH TOTAL; `target_time` = (sample_len * no of bench) seconds
    let target_time = 2.0;
    let size_sample = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];

    let plot_path = Path::new("./bench/plot");

    let kernels_to_run: Vec<String> = std::env::args().collect();

    if kernels_to_run.is_empty() {
        eprintln!("Usage: bencher <kernel1> <kernel2> ...");
        std::process::exit(1);
    }

    // pin core
    let cores = core_affinity::get_core_ids().expect("Failed to get Core IDs");
    println!("Available cores: {}", cores.len());
    let core = cores[4];
    core_affinity::set_for_current(core);

    println!("Heating up!!!");
    let mut noise = Noise::rng();
    warmup(&mut noise, 64, 2048);

    for k in kernels_to_run {
        match k.trim().to_lowercase().as_str() {
            "axpy" => {
                println!("Running AXPY benchmark...");
                bench_axpy(&mut noise, &size_sample, target_time, plot_path)
            }
            "asum" => {
                println!("Running ASUM benchmark...");
                bench_asum(&mut noise, &size_sample, target_time, plot_path)
            }
            "nrm2" => {
                println!("Running NRM2 benchmark...");
                bench_nrm2(&mut noise, &size_sample, target_time, plot_path)
            }
            "dot" => {
                println!("Running DOT benchmark...");
                bench_dot(&mut noise, &size_sample, target_time, plot_path)
            }
            "gemv" => {
                println!("Running GEMV benchmark...");
                bench_gemv(&mut noise, &size_sample, target_time, plot_path)
            }
            "gemv_t" => {
                println!("Running GEMV Transposed benchmark...");
                bench_gemv_t(&mut noise, &size_sample, target_time, plot_path)
            }
            "gemm" => {
                println!("Running GEMM benchmark...");
                bench_gemm_f_f(&mut noise, &size_sample, target_time, plot_path)
            }
            "i_amax" => {
                println!("Running I_AMAX benchmark...");
                bench_i_amax(&mut noise, &size_sample, target_time, plot_path)
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
            (i as f64, cyc_per_flop),
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

fn bench_nrm2(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];

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
            (i as f64, cyc_per_flop),
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
            (i as f64, cyc_per_flop),
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

fn bench_dot(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

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
            (i as f64, cyc_per_flop),
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

fn bench_i_amax(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];

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
            (i as f64, cyc_per_flop),
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

fn bench_gemv(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

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
            (i as f64, cyc_per_flop),
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

    match plot_bench(&metrics, &plot_path.join("gemv.png")) {
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
            (i as f64, cyc_per_flop),
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

fn bench_gemm_f_f(noise: &mut Noise, size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::with_capacity(6);
    let mut collector = MetricSet::with_capacity(size_sample.len());

    println!("| Size | Run | GFLOPS | %GFLOPS | %Latency | WSS Cache | Cyc/Flop |");
    println!("|------|-----|--------|---------|----------|-----------|----------|");

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i * i];
        let mut y1_buf = vec![0.0f32; i * i];
        let mut y2_buf = vec![0.0f32; i * i];

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
            (i as f64, cyc_per_flop),
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

    match plot_bench(&metrics, &plot_path.join("gemm_f_f.png")) {
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
