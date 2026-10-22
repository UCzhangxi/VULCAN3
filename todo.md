# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
2. [unverified] A/B the CUDA solve guard against `bd97791` on a NAS GH200: it is built and the `ffi` tests pass there (2026-10-20); the bitwise A/B, the subnormal-pivot block and the solve time are not done.

## 1. vulcan3 on PyPI

The name was unclaimed on pypi.org on 2026-10-16 (404), as `vulcan-jax` was.
Every README installs with `-i https://test.pypi.org/simple/ --extra-index-url
https://pypi.org/simple/`, and the 2026-10-16 install tests took `psutil 8.0.0`,
`python-dateutil 3.9.0` and `radis 0.16.4` from TestPyPI. Claiming the name needs
the maintainer's pypi.org account. Commands: `uv build --out-dir <dir> .` in this
repo, then `uv run --with twine twine upload <dir>/*`. Publishing all four packages
on PyPI would let the READMEs drop the TestPyPI index. The 0.18.0 TestPyPI release
and its tag were made in a session whose record is not on daw; vulcan-retrieval's
todo lists confirming them. Done when the name is claimed (and, if decided, the
packages are published).

## 2. CUDA solve guard and build stamp

- On main (`2e947dc`): `bt_lu_solve_warp0` divides when any pivot of a layer
  leaves [2^-1000, 2^1000] (the factor's `bt_div_by_rcp` range) and runs the old
  multiply otherwise; the CUDA library's stamp is the source hash plus a hash of
  the arch and `nvcc --version`.
- Done 2026-10-20 at `a420570` [M]: `tools/gpu_kernel_test.pbs` (job 94875) built
  both libraries (nvcc 12.6 V12.6.85; the stamp is two 64-hex hashes) and ran
  `tests/test_solver_fast.py` on the GPU twice, plain and under XLA_CMDBUF=2's
  flags: 53 passed / 21 skipped each (16 compute-sanitizer, 3 full tier, 2
  stacked-rhs on the C++ handler), the `huge` (2^1003 pivot) and `tiny` cases
  included. vulcan-retrieval's v4 job (94946) found that library current at start.
- On a GH200 (environment as `tools/gpu_kernel_test.pbs` sets it), in two scratch
  clones at `bd97791` (old) and main (new): build both libraries with the
  production nvcc and `NVCC_ARCH=sm_90`; compare `cuobjdump --dump-resource-usage`
  for `bt_solve_kernel`; run `tools/ab_cuda_solve.py` (its docstring has the
  capture / run / compare steps). `tools/gpu_kernel_test.pbs` alone compares CUDA
  with CPU, not the old kernel with the new.
- The cleanup (`2216219`) removed the `tiny` guard case from
  `tests/test_solver_fast.py` (row 3 scaled by 2^-975), the only test of the
  Gauss-Jordan kernel's division fallback; restoring it against the CPU kernel's
  first layer is the maintainer's call.
- Done means: every A/B digest bitwise equal at 792 lanes, the subnormal-pivot
  block finite on the new kernel, solve time unchanged, the `ffi` tests passing;
  then delete `tools/ab_cuda_solve.py`.
