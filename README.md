Rookie attempt to rewriting **BLAS FORTRAN 77** and **Intel Math Library** Kernels in modern native rust. `ONLY x86_64`

refs I took:

- https://www.netlib.org/blas/ ← good for overview
- https://www.netlib.org/lapack/explore-html/ ←/
- https://icl.utk.edu/~mgates3/docs/lapack.html ←/
- https://suif.stanford.edu/papers/lam-asplos91.pdf  <- good for understanding cache blocking techniques & optimizations
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
- multithreading, GPU maybe (NOT CUDA)?
- LESS BAD BRANCHING, more SIMD, LESS FN CALLS, better DATA_READ, better CACHE
- MAKE GEMV & GEMM FAST FOR ALL PATHS!!!
- bench is somewhat UNFAIR AND NOT UNIFORM

To bench, run **[harness](./bench/bencher.rs)** using
`cargo bench --bench bencher <kernel> or all` [ref](https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark) and
build using `RUSTFLAGS target-cpu=x86-64-v3`, otherwise rustc will throw error.

Install [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline)
then copy the `.dll` or `.so` into the project root:

- windows > `Copy-Item "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\bin\libiomp5md.dll" .`
- linux > `cp /home/ronakgh97/intel/oneapi/compiler/latest/lib/libiomp5.so .`
- mac > `sell that shit`

> All are single threaded, ran on i7 14650hx, rust 1.97.1, last updated on 8/7/2026

**Axpy**
![plot](bench/plot/axpy.png)

| Size  | Run       | GFLOPS             | Gflops_w/IntelMKL    | Latency_w/IntelMKL   | WSS Cache Fit |
|-------|-----------|--------------------|----------------------|----------------------|---------------|
| 128   | 187255932 | 7.629492809340103  | 10.912895832839984%  | -9.839158693761933%  | 100%          |
| 256   | 173547966 | 14.1419600920584   | 7.486771083284213%   | -6.965295364099484%  | 100%          |
| 512   | 151862451 | 24.749730821093852 | 1.873356543327374%   | -1.83890725395961%   | 100%          |
| 1024  | 122043465 | 39.779983627731504 | 14.620823686926965%  | -12.755818023836586% | 100%          |
| 2048  | 58962204  | 38.4373804382726   | -26.741202470379193% | 36.50237701426491%   | 100%          |
| 4096  | 42235529  | 55.06656584647125  | -14.269094530364459% | 16.644049718357774%  | 100%          |
| 8192  | 10024013  | 26.138435609014664 | -14.212387220691166% | 16.56694569325871%   | 100%          |
| 16384 | 4715347   | 24.591421180592963 | -17.214312473013063% | 20.793826792102738%  | 100%          |

**Dot**
![plot](bench/plot/dot.png)

| Size  | Run       | GFLOPS             | Gflops_w/IntelMKL   | Latency_w/IntelMKL   | WSS Cache Fit |
|-------|-----------|--------------------|---------------------|----------------------|---------------|
| 128   | 195724363 | 7.974527845064066  | 15.241024514513962% | -13.225346250366288% | 100%          |
| 256   | 192106380 | 15.654235916705563 | 21.511714415536932% | -17.703407872242437% | 100%          |
| 512   | 184358897 | 30.04582843091022  | 35.00790591137586%  | -25.930263620529264% | 100%          |
| 1024  | 172262260 | 56.14876627387122  | 61.02545000675933%  | -37.898015502641144% | 100%          |
| 2048  | 146304651 | 95.37580261375066  | 132.50196051640114% | -56.989609989570035% | 100%          |
| 4096  | 116067814 | 151.32889955594817 | 175.15762084025857% | -63.657194122181146% | 100%          |
| 8192  | 23734897  | 61.89098508547981  | 114.82158674555333% | -53.44974333587547%  | 100%          |
| 16384 | 12283986  | 64.06330579114818  | 111.12019618545372% | -52.633617338931735% | 100%          |

**Gemv**
![plot](bench/plot/gemv.png)

| Size  | Run     | GFLOPS             | Gflops_w/IntelMKL    | Latency_w/IntelMKL   | WSS Cache Fit        |
|-------|---------|--------------------|----------------------|----------------------|----------------------|
| 128   | 7665208 | 39.97550601582551  | -8.145060399881212%  | 8.867307991644111%   | 100%                 |
| 256   | 2431182 | 50.716283340863214 | 1.0187885272151849%  | -1.0085139032732424% | 100%                 |
| 512   | 842043  | 70.26256637735183  | -8.09471386248138%   | 8.807669507028347%   | 100%                 |
| 1024  | 101160  | 33.7641413623682   | 0.665279457940009%   | -0.6608827408242307% | 51.02239532619279%   |
| 2048  | 23600   | 31.506852394268957 | -0.7810677978064013% | 0.7872164923269553%  | 12.774256460263286%  |
| 4096  | 3318    | 17.713297860896997 | 3.4122531655549397%  | -3.2996603991329643% | 3.195901439375457%   |
| 8192  | 731     | 15.610736048678088 | 4.048336352195165%   | -3.8908227600025063% | 0.7992678462477121%  |
| 16384 | 183     | 15.556032392371709 | 6.320626079343515%   | -5.944872892891594%  | 0.19985354244217976% |

**Gemv_T**
![plot](bench/plot/gemv_t.png)

| Size  | Run     | GFLOPS             | Gflops_w/IntelMKL     | Latency_w/IntelMKL   | WSS Cache Fit        |
|-------|---------|--------------------|-----------------------|----------------------|----------------------|
| 128   | 7051057 | 36.77259325611539  | -19.109524913928215%  | 23.623949412578753%  | 100%                 |
| 256   | 2232249 | 46.566399632492164 | -10.513008386138068%  | 11.748085611707562%  | 100%                 |
| 512   | 838405  | 69.95904428675267  | -0.1411212468532423%  | 0.14132068035942652% | 100%                 |
| 1024  | 100107  | 33.41276537273465  | -2.4447008905789285%  | 2.5059642201874457%  | 51.02239532619279%   |
| 2048  | 25017   | 33.398641096847925 | -7.651033464134363%   | 8.284915090157408%   | 12.774256460263286%  |
| 4096  | 3456    | 18.449820337331435 | -3.1982348674938925%  | 3.3039013938599275%  | 3.195901439375457%   |
| 8192  | 774     | 16.520938936906674 | -0.08431781550658682% | 0.08438897044297171% | 0.7992678462477121%  |
| 16384 | 193     | 16.447078971453337 | 2.3507555534610596%   | -2.2967642405269455% | 0.19985354244217976% |

**Gemm_F_F**
![plot](bench/plot/gemm_f_f.png)

| Size  | Run    | GFLOPS             | Gflops_w/IntelMKL    | Latency_w/IntelMKL  | WSS Cache Fit     |
|-------|--------|--------------------|----------------------|---------------------|-------------------|
| 128   | 119011 | 79.44478975794888  | -39.03392812313379%  | 64.02565709329447%  | 100%              |
| 256   | 16508  | 88.15581207639944  | -35.87821683757666%  | 55.953242514631164% | 100%              |
| 512   | 2740   | 117.02672429550375 | -16.836658474730942% | 20.245288568179014% | 40.9375%          |
| 1024  | 333    | 113.77101765735884 | -18.93347886464331%  | 23.355484606314967% | 10.234375%        |
| 2048  | 36     | 97.5603789380732   | -31.131072712639313% | 45.203365202339555% | 2.55859375%       |
| 4096  | 3      | 61.348770612429576 | -57.12245989867806%  | 133.2223344988883%  | 0.6396484375%     |
| 8192  | 1      | 52.933379325077134 | -63.0638694477456%   | 170.7376178956475%  | 0.159912109375%   |
| 16384 | 1      | 52.86820748655343  | -63.90618647334251%  | 177.05578942536476% | 0.03997802734375% |

Wanna improve the [bench](bench/bencher.rs) and added more kernels perf test? Send me a PR