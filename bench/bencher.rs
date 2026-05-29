mod harness;
mod utils;

use crate::harness::{MetricSet, run_bench};
use crate::utils::{axpy_ob, dot_ob, gemv_ob};
use blas_rs::lvl1::{axpy, axpy_unsafe, dot, dot_unsafe};
use blas_rs::lvl2::gemv;
use blas_rs::utils::*;
use std::error::Error;
#[allow(unused)]
use std::f64::consts::PI;
use std::f64::consts::TAU;
use std::hint::black_box;
use std::path::Path;

extern crate openblas_src;

// Bench uses this as ref: https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark as ref

fn warmup(r: usize, s: usize) {
    let mut a = vec![0.0f32; s * s];
    let mut x = vec![0.0f32; s];
    let mut y = vec![0.0f32; s];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    gen_fill(&mut a);
    gen_fill(&mut x);
    y.fill(1.0);

    // Warmup phase
    for i in 0..r {
        {
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
            axpy(s, (i % 7) as f32, &y, 1, &mut vec![0.0f32; s], 1);

            dot(s, &y, 1, &vec![0.0f32; s], 1);
        }

        gen_fill(&mut a);
        gen_fill(&mut x);
        y.fill(1.0);

        {
            gemv_ob(
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

            gemv_ob(
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

            axpy_ob(s as i32, (i % 7) as f32, &y, 1, &mut vec![0.0f32; s], 1);

            dot_ob(s as i32, &y, 1, &vec![0.0f32; s], 1);
        }

        gen_fill(&mut a);
        gen_fill(&mut x);
        y.fill(1.0);
    }
}

enum BenchMetrics {
    Gflops(Vec<(f64, f64)>),
    Latency(Vec<(f64, f64)>),
    CacheEfficiency(Vec<(f64, f64)>), // LLC fit proxy
    // a bit heuristic, but we can see how much performance cost,
    // we pay per flop as working set grows, higher means more cache miss
    CacheMiss(Vec<(f64, f64)>),
    CompareGflops(Vec<(f64, f64)>),
    CompareLatency(Vec<(f64, f64)>),
}

fn main() {
    unsafe {
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    }

    warmup(32, 8192);

    // bit yap about axpy and dot, axpy is slightly overhead, due to` _mm256_storeu_ps`, write heavy,
    // so its write-bandwidth bound, even it whole vectors fits in cache and also "cache" does not help much since those 8 lanes are not reused.
    // while dot is just most read heavy, and accumulated in 4 sum register, less overhead, finally the `cache miss` is heuristic metrics as said earlier

    // BENCH TOTAL `target_time` = (sample_len * no of bench) seconds
    let target_time = TAU; // big enough to run least some runs in largest samples
    let size_sample = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

    let plot_path = Path::new("./bench/plot");

    let kernel: String = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "all".to_string())
        .to_lowercase();

    let kernels: Vec<&str> = if kernel == "all" {
        vec!["axpy", "dot", "gemv", "gemv_t"] // order
    } else {
        kernel.split(',').collect()
    };

    for k in kernels {
        match k.trim() {
            "axpy" => {
                println!("Running AXPY benchmark...");
                bench_axpy(&size_sample, target_time, plot_path)
            }
            "dot" => {
                println!("Running DOT benchmark...");
                bench_dot(&size_sample, target_time, plot_path)
            }
            "gemv" => {
                println!("Running GEMV benchmark...");
                bench_gemv(&size_sample, target_time, plot_path)
            }
            "gemv_t" => {
                println!("Running GEMV Transposed benchmark...");
                bench_gemv_t(&size_sample, target_time, plot_path)
            }
            _ => {
                eprintln!(
                    "Unknown kernel: '{}'. Specify: <kernel1>, <kernel2> or all",
                    k
                );
            }
        }
    }
}

fn bench_axpy(size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut bench_metrics: Vec<BenchMetrics> = Vec::new();

    let mut metrics_collector = MetricSet::new();

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y1_buf);

        gen_fill(&mut x_buf);
        y1_buf.fill(1.0);

        // Start time bound bench for both
        let rc = run_bench(
            || unsafe { axpy_unsafe(i, 3.0, &x_buf, 1, &mut y1_buf, 1) },
            target_time,
        );

        y2_buf.fill(1.0); // use diff buf to prevent overflow

        let rc_ob = run_bench(
            || axpy_ob(i as i32, 3.0, &x_buf, 1, &mut y2_buf, 1),
            target_time,
        );

        let working_kb = 3.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read/write
        let total_flops = 2.0 * i as f64 * rc; // 2n FLOPs per axpy call
        let total_flops_ob = 2.0 * i as f64 * rc_ob;

        let (gflops, gflops_ob, latency, latency_ob, cache_eff, ns_per_flop) = MetricSet::derive(
            rc,
            rc_ob,
            target_time,
            total_flops,
            total_flops_ob,
            working_kb,
        );

        let gflops_rel = (gflops - gflops_ob) / gflops_ob * 100.0;
        let latency_rel = (latency - latency_ob) / latency_ob * 100.0;

        metrics_collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            (working_kb.log10(), ns_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "S: {}, R: {}, Gflops: {}, Gflops_rel: {}%, Latency_rel: {}%, Cache fit: {} %",
            i, rc, gflops, gflops_rel, latency_rel, cache_eff
        );

        // reset, avoid hot cache possibility, we don't do TRUST ME BRO BENCH
        gen_fill(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }
    metrics_collector.finalize(&mut bench_metrics);

    match plot_bench(&bench_metrics, &plot_path.join("axpy.png")) {
        Ok(_) => println!("Exported"),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_dot(size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::new();

    let mut metrics_collector = MetricSet::new();

    for &i in size_sample {
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        gen_fill(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        y1_buf.fill(1.0);

        // Start time bound bench

        let rc = run_bench(
            || unsafe {
                dot_unsafe(i, &x_buf, 1, &y1_buf, 1);
            },
            target_time,
        );
        y2_buf.fill(1.0);

        let rc_ob = run_bench(
            || {
                dot_ob(i as i32, &x_buf, 1, &y2_buf, 1);
            },
            target_time,
        );

        let working_kb = 2.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read/write
        let total_flops = 2.0 * i as f64 * rc; // 2n FLOPs per axpy call
        let total_flops_ob = 2.0 * i as f64 * rc_ob;

        let (gflops, gflops_ob, latency, latency_ob, cache_eff, ns_per_flop) = MetricSet::derive(
            rc,
            rc_ob,
            target_time,
            total_flops,
            total_flops_ob,
            working_kb,
        );

        let gflops_rel = (gflops - gflops_ob) / gflops_ob * 100.0;
        let latency_rel = (latency - latency_ob) / latency_ob * 100.0;

        metrics_collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            (working_kb.log10(), ns_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "S: {}, R: {}, Gflops: {}, Gflops_rel: {}%, Latency_rel: {}%, Cache fit: {} %",
            i, rc, gflops, gflops_rel, latency_rel, cache_eff
        );

        // reset, avoid hot cache possibility, we don't do 'TRUST ME BRO' BENCH
        gen_fill(&mut x_buf);
        gen_fill(&mut y1_buf);
        gen_fill(&mut y2_buf);
    }

    metrics_collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("dot.png")) {
        Ok(_) => println!("Exported"),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_gemv(size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::new();

    let mut metrics_collector = MetricSet::new();

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        black_box(&mut a_buf);
        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        gen_fill(&mut a_buf);
        gen_fill(&mut x_buf);

        y1_buf.fill(1.0);

        // Start time bound bench
        let rc = run_bench(
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

        y2_buf.fill(1.0); // prevent overflow

        let rc_ob = run_bench(
            || {
                gemv_ob(
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
        let total_flops_ob = 2.0 * i.pow(2) as f64 * rc_ob;

        let (gflops, gflops_ob, latency, latency_ob, cache_eff, ns_per_flop) = MetricSet::derive(
            rc,
            rc_ob,
            target_time,
            total_flops,
            total_flops_ob,
            working_kbyte,
        );

        let gflops_rel = (gflops - gflops_ob) / gflops_ob * 100.0;
        let latency_rel = (latency - latency_ob) / latency_ob * 100.0;

        metrics_collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            (working_kbyte.log10(), ns_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "S: {}, R: {}, Gflops: {}, Gflops_rel: {}%, Latency_rel: {}%, Cache fit: {} %",
            i, rc, gflops, gflops_rel, latency_rel, cache_eff
        );

        // reset, avoid hot cache possibility
        gen_fill(&mut a_buf);
        gen_fill(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }

    metrics_collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("gemv.png")) {
        Ok(_) => println!("Exported"),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn bench_gemv_t(size_sample: &[usize], target_time: f64, plot_path: &Path) {
    let mut metrics: Vec<BenchMetrics> = Vec::new();

    let mut metrics_collector = MetricSet::new();

    for &i in size_sample {
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i];
        let mut y1_buf = vec![0.0f32; i];
        let mut y2_buf = vec![0.0f32; i];

        black_box(&mut a_buf);
        black_box(&mut x_buf);
        black_box(&mut y1_buf);
        black_box(&mut y2_buf);

        gen_fill(&mut a_buf);
        gen_fill(&mut x_buf);

        y1_buf.fill(1.0);

        // Start time bound bench
        let rc = run_bench(
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

        let rc_ob = run_bench(
            || {
                gemv_ob(
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
        let total_flops_ob = 2.0 * i.pow(2) as f64 * rc_ob;

        let (gflops, gflops_ob, latency, latency_ob, cache_eff, ns_per_flop) = MetricSet::derive(
            rc,
            rc_ob,
            target_time,
            total_flops,
            total_flops_ob,
            working_kbyte,
        );

        let gflops_rel = (gflops - gflops_ob) / gflops_ob * 100.0;
        let latency_rel = (latency - latency_ob) / latency_ob * 100.0;

        metrics_collector.collect(
            ((i as f64).log10(), gflops),
            ((i as f64).log10(), latency.log10()),
            ((i as f64).log10(), cache_eff),
            (working_kbyte.log10(), ns_per_flop),
            ((i as f64).log10(), gflops_rel),
            ((i as f64).log10(), latency_rel),
        );

        println!(
            "S: {}, R: {}, Gflops: {}, Gflops_rel: {}%, Latency_rel: {}%, Cache fit: {} %",
            i, rc, gflops, gflops_rel, latency_rel, cache_eff
        );

        // reset, avoid hot cache possibility
        gen_fill(&mut a_buf);
        gen_fill(&mut x_buf);
        y1_buf.fill(1.0);
        y2_buf.fill(1.0);
    }

    metrics_collector.finalize(&mut metrics);

    match plot_bench(&metrics, &plot_path.join("gemv_t.png")) {
        Ok(_) => println!("Exported"),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

#[inline(always)]
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
                    "Cache Fit",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CacheMiss(points)) => (
                    "log(Working Set KB)",
                    "ns / FLOP",
                    CYAN,
                    "Perf Cost",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CompareGflops(points)) => (
                    "log(Size)",
                    "GFLOPS Ratio (%)",
                    MAGENTA,
                    "GFLOPS(+-Rel) w/OpenBlas",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CompareLatency(points)) => (
                    "log(Size)",
                    "Latency Ratio (%)",
                    YELLOW,
                    "Latency(+-Rel) w/OpenBlas",
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
