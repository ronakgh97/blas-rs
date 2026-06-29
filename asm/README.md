This dir/ contains handwritten asm for some specific kernels,
this will be used for better insights/optimzation and comparing with rust compiler and `openblas`

> It will be written in NASM/Intel syntax, follows ubuntu abi register and
> you can use `nasm` and `gcc` to compile and link & test the API

Todo:

- lvl1: axpy, dot, i_amax, asum, nrm2
- lvl2: gemv
- lvl3: none