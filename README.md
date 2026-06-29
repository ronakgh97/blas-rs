Rookie attempt to rewriting **BLAS FORTRAN 77** and **Intel Math Library** Kernels in modern rust. `ONLY x86_64`

> This will not cover all kernels for every single routine, but the most commonly used ones, (excluding complex
> type and only fp32) and I have **spammed** `_mm256*` intrinsics for all kernels, because I got I7 14650hx which does
> not support AVX-512 :( and lastly, this project is purely for learning source, the code is well written & documented,
> and I will add asm snippet [here](asm) for specific kernel & more refs for better understanding about rustc, x86, HPC
> and perf engineering.

refs I took:

- https://www.netlib.org/blas/ ← good for overview
- https://www.netlib.org/lapack/explore-html/ ←/
- https://icl.utk.edu/~mgates3/docs/lapack.html ←/
- https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/blas-routines.html ← good
  details, very clear
- https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html ← good man for intrinsics
- https://doc.rust-lang.org/core/arch/x86_64/index.html#functions ←/
- https://public.dhe.ibm.com/software/dw/cell/BLAS_Prog_Guide_API_v3.1.pdf <- api ref, mostly similar everywhere

TODO:

- lvl1: rotmg, rotm
- lvl2: gemv - all
- lvl3: gemm - all
- handle NaN, over/underflow, return vs panic and many more edge cases :(
- test code ref from [this](https://github.com/OpenMathLib/OpenBLAS/tree/develop) repo, currently ai-generated
- multithreading, GPU maybe?
- LESS BRANCHING, more SIMD, LESS FN CALLS, better DATA_READ, better CACHE
- MAKE GEMV & GEMM FAST FOR ALL PATHS!!!
- bench is somewhat UNFAIR AND NOT UNIFORM

To bench, run **[harness](./bench/bencher.rs)** using
`cargo bench --bench bencher <kernel> or all` [ref](https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark) and
build using `RUSTFLAGS target-cpu=x86-64-v3`, otherwise rustc will throw error.

Install [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline)
then copy the `.dll` or `.so` into the project root:

- windows > `Copy-Item "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\bin\libiomp5md.dll" .`
- arch linux > `cp /home/ronakgh97/intel/oneapi/compiler/latest/lib/libiomp5.so .`

> All are single threaded, ran on i7 14650hx, rust 1.96.0.

**Axpy**
![plot](bench/plot/axpy.png)

| Size  | Run       | GFLOPS             | Gflops_w/IntelMKL   | Latency_w/IntelMKL   | WSS Cache Fit |
|-------|-----------|--------------------|---------------------|----------------------|---------------|
| 128   | 200444736 | 8.166853140446882  | 35.60469391958493%  | -26.256240024183015% | 100%          |
| 256   | 182967692 | 14.90954863499651  | 29.707374430448624% | -22.903381215520827% | 100%          |
| 512   | 158822276 | 25.88400632328946  | 19.868010019469573% | -16.574906028925074% | 100%          |
| 1024  | 102353626 | 33.3620946547272   | -9.900970898622045% | 10.988987336901964%  | 100%          |
| 2048  | 78732738  | 51.32576461105222  | -0.59297421815088%  | 0.5965113768237997%  | 100%          |
| 4096  | 45148346  | 58.864289528297384 | -9.37985752546564%  | 10.35074241700902%   | 100%          |
| 8192  | 12958406  | 33.79026777499617  | 3.4783395931541348% | -3.3614180579529265% | 100%          |
| 16384 | 6169499   | 32.17510341402462  | 1.4684349155416352% | -1.4471839609666812% | 100%          |

**Dot**
![plot](bench/plot/dot.png)

| Size  | Run      | GFLOPS             | Gflops_w/IntelMKL    | Latency_w/IntelMKL   | WSS Cache Fit        |
|-------|----------|--------------------|----------------------|----------------------|----------------------|
| 128   | 14296793 | 74.560476619376    | 45.25317830526169%   | -31.154690612110635% | 100%                 |
| 256   | 3626624  | 75.65411136365991  | 1.0200344734208275%  | -1.0097348300640336% | 100%                 |
| 512   | 958167   | 79.95227713355987  | -3.787263838751373%  | 3.936343554770204%   | 100%                 |
| 1024  | 130700   | 43.62388254668084  | -3.2851400081118336% | 3.396727254102806%   | 51.02239532619279%   |
| 2048  | 26638    | 35.563727321564194 | 1.9815697268895662%  | -1.9430665091705137% | 12.774256460263286%  |
| 4096  | 3332     | 17.79273954659895  | 1.6728448770857642%  | -1.645321205586495%  | 3.195901439375457%   |
| 8192  | 730      | 15.581527220854792 | 1.5167628371123505%  | -1.4941008703617338% | 0.7992678462477121%  |
| 16384 | 183      | 15.629532440195318 | 4.679729553303714%   | -4.470521249217374%  | 0.19985354244217976% |

**Gemv**
![plot](bench/plot/gemv.png)

| Size  | Run      | GFLOPS             | Gflops_w/IntelMKL    | Latency_w/IntelMKL   | WSS Cache Fit        |
|-------|----------|--------------------|----------------------|----------------------|----------------------|
| 128   | 13675459 | 71.3201024193842   | 4.268020191150732%   | -4.093316611676632%  | 100%                 |
| 256   | 3510447  | 73.23057736572352  | -8.319051043098323%  | 9.073914633026996%   | 100%                 |
| 512   | 922166   | 76.94831744476373  | -7.46318740427445%   | 8.065101006752402%   | 100%                 |
| 1024  | 131174   | 43.7821562444634   | -3.8411086982812837% | 3.99454345436347%    | 51.02239532619279%   |
| 2048  | 26387    | 35.228949817595854 | -5.609158113425792%  | 5.942481284536169%   | 12.774256460263286%  |
| 4096  | 3303     | 17.6391133859561   | 2.0294378917208444%  | -1.9890709325230373% | 3.195901439375457%   |
| 8192  | 733      | 15.644653784477889 | 1.9995837742584672%  | -1.9603842489042462% | 0.7992678462477121%  |
| 16384 | 183      | 15.630144864527491 | 5.459202667614173%   | -5.176601500411932%  | 0.19985354244217976% |

**Gemv_T**
![plot](bench/plot/gemv_t.png)

| Size  | Run      | GFLOPS             | Gflops_w/IntelMKL    | Latency_w/IntelMKL    | WSS Cache Fit        |
|-------|----------|--------------------|----------------------|-----------------------|----------------------|
| 128   | 10251107 | 53.461460619935536 | -18.20981427099827%  | 22.26405785571079%    | 100%                 |
| 256   | 3208621  | 66.93424931951327  | -8.57786751447875%   | 9.382703379662736%    | 100%                 |
| 512   | 833889   | 69.58217242331276  | -6.055586355580537%  | 6.445924904594098%    | 100%                 |
| 1024  | 135292   | 45.15649762305524  | 0.4689591940686096%  | -0.4667702321497747%  | 51.02239532619279%   |
| 2048  | 24781    | 33.083852283542804 | -8.49737797495192%   | 9.286485771550721%    | 12.774256460263286%  |
| 4096  | 3255     | 17.378167168821662 | -0.0139636739858185% | 0.013965624100040343% | 3.195901439375457%   |
| 8192  | 718      | 15.330904989137172 | 1.0321720280180475%  | -1.0216270790771311%  | 0.7992678462477121%  |
| 16384 | 177      | 15.0682079956136   | -1.475670925019749%  | 1.4977731275862831%   | 0.19985354244217976% |

**Gemm_F_F**
![plot](bench/plot/gemm_f_f.png)

| Size  | Run    | GFLOPS             | Gflops_w/IntelMKL    | Latency_w/IntelMKL  | WSS Cache Fit     |
|-------|--------|--------------------|----------------------|---------------------|-------------------|
| 128   | 161746 | 107.97206188867901 | -18.604033489424108% | 22.856210555601482% | 100%              |
| 256   | 24346  | 130.0142859251401  | -6.137768280100085%  | 6.53912459530708%   | 100%              |
| 512   | 3188   | 136.1988075848359  | -4.726877458183129%  | 4.961396595465221%  | 40.9375%          |
| 1024  | 372    | 126.845770227491   | -11.005191658933485% | 12.366105241506723% | 10.234375%        |
| 2048  | 38     | 101.87507318632152 | -29.380126751437384% | 41.60319949602204%  | 2.55859375%       |
| 4096  | 3      | 56.74960819060317  | -60.278106342560825% | 151.75033411649008% | 0.6396484375%     |
| 8192  | 1      | 53.180359495912015 | -63.35565138753297%  | 172.89337588601128% | 0.159912109375%   |
| 16384 | 1      | 47.323985058902046 | -60.73901226188393%  | 154.70576712698488% | 0.03997802734375% |