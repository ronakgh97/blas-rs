use crate::BenchMetrics;
use crate::utils::MAX_L1L2_KB;
use std::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};
use std::time::Instant;

#[inline(always)]
/// Bench a function `f()` by running it repeatedly until `target_time` seconds have elapsed,
/// and returning `(runs, elapsed_secs, tsc_cycles)`.
pub fn run_bench<F>(mut f: F, target_time: f64) -> (f64, f64, f64)
where
    F: FnMut(),
{
    //tiny warmup
    {
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < 0.5 {
            f();
        }
    }

    let mut runs: u64 = 0;
    let start = Instant::now();

    let clock_start_mark = unsafe {
        _mm_lfence();
        _rdtsc()
    };

    while start.elapsed().as_secs_f64() < target_time {
        f();
        runs += 1;
    }

    let clock_end_mark = unsafe {
        let e = __rdtscp(&mut 0);
        _mm_lfence();
        e
    };

    (
        runs as f64,
        start.elapsed().as_secs_f64(),
        (clock_end_mark - clock_start_mark) as f64,
    )
}

pub struct MetricSet {
    gflops: Vec<(f64, f64)>,
    latency: Vec<(f64, f64)>,
    cache_eff: Vec<(f64, f64)>,
    cycle_cost: Vec<(f64, f64)>,
    compare_gflops: Vec<(f64, f64)>,
    compare_latency: Vec<(f64, f64)>,
}

impl MetricSet {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            gflops: Vec::with_capacity(capacity),
            latency: Vec::with_capacity(capacity),
            cache_eff: Vec::with_capacity(capacity),
            cycle_cost: Vec::with_capacity(capacity),
            compare_gflops: Vec::with_capacity(capacity),
            compare_latency: Vec::with_capacity(capacity),
        }
    }

    pub fn collect(
        &mut self,
        gflops: (f64, f64),
        latency: (f64, f64),
        cache_eff: (f64, f64),
        cycle_cost: (f64, f64),
        compare_gflops: (f64, f64),
        compare_latency: (f64, f64),
    ) {
        self.gflops.push(gflops);
        self.latency.push(latency);
        self.cache_eff.push(cache_eff);
        self.cycle_cost.push(cycle_cost);
        self.compare_gflops.push(compare_gflops);
        self.compare_latency.push(compare_latency);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        runs: f64,
        runs_mkl: f64,
        elapsed: f64,
        elapsed_mkl: f64,
        tsc: f64,
        tsc_mkl: f64,
        total_flops: f64,
        total_flops_mkl: f64,
        working_set_kb: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64) {
        // wall-time based
        let gflops = total_flops / elapsed / 1e9;
        let gflops_mkl = total_flops_mkl / elapsed_mkl / 1e9;
        let latency = elapsed / runs * 1e9;
        let latency_mkl = elapsed_mkl / runs_mkl * 1e9;
        let cache_eff = ((*MAX_L1L2_KB / working_set_kb) * 100.0).min(100.0);
        // TSC cycle based
        let cycles_per_flop = tsc / total_flops;
        let cycles_per_flop_mkl = tsc_mkl / total_flops_mkl;

        (
            gflops,
            gflops_mkl,
            latency,
            latency_mkl,
            cache_eff,
            cycles_per_flop,
            cycles_per_flop_mkl,
        )
    }

    pub fn finalize(self, bench_metrics: &mut Vec<BenchMetrics>) {
        bench_metrics.push(BenchMetrics::Gflops(self.gflops));
        bench_metrics.push(BenchMetrics::Latency(self.latency));
        bench_metrics.push(BenchMetrics::CacheEfficiency(self.cache_eff));
        bench_metrics.push(BenchMetrics::CycleCost(self.cycle_cost));
        bench_metrics.push(BenchMetrics::CompareGflops(self.compare_gflops));
        bench_metrics.push(BenchMetrics::CompareLatency(self.compare_latency));
    }
}
