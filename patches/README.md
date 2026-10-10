# llama.cpp patches this branch builds against (PH402 / sm_60)

The prompt path's MMQ kernels come from the llama.cpp checkout that `STRATA_GGML_DIR` points at. This branch expects llama.cpp at the commit Strata 0.1.41 pins (`3cf03257f219afbe7334045ff7c6a06ac68c627d`, the `GIT_TAG` in `CMakeLists.txt`), with these two patches applied in order:

1. `0001-sm_60-MMQ-the-dp4a-fallback-as-PRMT-mad.wide.s16-the.patch`
   - ggml's dp4a fallback for GPUs without `__dp4a`, rewritten as byte permutes plus 16-bit multiply-adds.
   - The integer sums are the same, so the output is the same bit for bit.
2. `0002-sm_60-MMQ-MoE-experts-read-in-place-offset-table-and.patch`
   - MMQ reads the MoE experts in place through a byte-offset table (`pt`).
   - The down product's K loop stops at K (`kstop`), so nothing past a matrix is read.
   - `src/prefill/mmq_inplace_*.cu` and `mmq::inplace()` need it. Against an unpatched checkout those files compile empty and the experts are gathered as before.

To apply:

```
git -C <llama.cpp> checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
git -C <llama.cpp> am <this folder>/0001-*.patch <this folder>/0002-*.patch
```

Then configure with `-DSTRATA_GGML_DIR=<llama.cpp> -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60`.
