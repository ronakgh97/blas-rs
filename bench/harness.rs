use crate::BenchMetrics;
use crate::utils::MAX_L1L2_KB;
use std::time::Instant;

#[inline(always)]
/// Bench a function `f` by running it repeatedly until `target_time` seconds have elapsed,
/// and returning the total number of runs and the actual elapsed time.
pub fn run_bench<F>(mut f: F, target_time: f64) -> (f64, f64)
where
    F: FnMut(),
{
    //tiny warmup
    {
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < 1.0 {
            f();
        }
    }

    let mut runs: u64 = 0;
    let start = Instant::now();

    while start.elapsed().as_secs_f64() < target_time {
        f();
        runs += 1;
    }

    (runs as f64, start.elapsed().as_secs_f64())
}

pub struct MetricSet {
    gflops: Vec<(f64, f64)>,
    avg_time: Vec<(f64, f64)>,
    cache_eff: Vec<(f64, f64)>,
    cache_miss: Vec<(f64, f64)>,
    compare_gflops: Vec<(f64, f64)>,
    compare_avg_time: Vec<(f64, f64)>,
}

impl MetricSet {
    pub fn new() -> Self {
        Self {
            gflops: Vec::with_capacity(12),
            avg_time: Vec::with_capacity(12),
            cache_eff: Vec::with_capacity(12),
            cache_miss: Vec::with_capacity(12),
            compare_gflops: Vec::with_capacity(12),
            compare_avg_time: Vec::with_capacity(12),
        }
    }

    pub fn collect(
        &mut self,
        gflops: (f64, f64),
        avg_time: (f64, f64),
        cache_eff: (f64, f64),
        cache_miss: (f64, f64),
        compare_gflops: (f64, f64),
        compare_avg_time: (f64, f64),
    ) {
        self.gflops.push(gflops);
        self.avg_time.push(avg_time);
        self.cache_eff.push(cache_eff);
        self.cache_miss.push(cache_miss);
        self.compare_gflops.push(compare_gflops);
        self.compare_avg_time.push(compare_avg_time);
    }

    #[inline]
    pub fn derive(
        runs: f64,
        runs_against: f64,
        toc: f64,
        toc_against: f64,
        total_flops: f64,
        total_flops_against: f64,
        working_kb: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let gflops = total_flops / toc / 1e9;
        let gflops_against = total_flops_against / toc_against / 1e9;
        let latency = toc / runs * 1e9;
        let latency_against = toc_against / runs_against * 1e9;
        let cache_eff = ((*MAX_L1L2_KB / working_kb) * 100.0).min(100.0);
        let ns_per_flop = (toc * 1e9) / total_flops;

        (
            gflops,
            gflops_against,
            latency,
            latency_against,
            cache_eff,
            ns_per_flop,
        )
    }

    pub fn finalize(self, bench_metrics: &mut Vec<BenchMetrics>) {
        bench_metrics.push(BenchMetrics::Gflops(self.gflops));
        bench_metrics.push(BenchMetrics::Latency(self.avg_time));
        bench_metrics.push(BenchMetrics::CacheEfficiency(self.cache_eff));
        bench_metrics.push(BenchMetrics::CacheMiss(self.cache_miss));
        bench_metrics.push(BenchMetrics::CompareGflops(self.compare_gflops));
        bench_metrics.push(BenchMetrics::CompareLatency(self.compare_avg_time));
    }
}
