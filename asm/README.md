This dir/ contains handwritten ASM for some specific kernels,
this will be used for better insights/optimzation and comparing with rust compiler and `openblas`

> It will be written in NASM/Intel syntax, follows ubuntu abi register and
> you can use `nasm` and `gcc` to compile and link & test the API

Todo:

- lvl1: axpy, dot, i_amax, asum, nrm2
- lvl2: gemv, gemv_t
- lvl3: gemm_f_f