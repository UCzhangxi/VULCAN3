# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
2. [decision] The CUDA solve guard is bitwise and solves a subnormal pivot exactly, but costs ~7% solve time (notes "Efficiency program"); keep it (v4 ran with it) or recover the time.

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
- Measured on NAS GH200s at `da54541` [M]: `tools/gpu_kernel_test.pbs` (job 96221,
  nvcc 12.6): `tests/test_solver_fast.py` 33 passed / 13 skipped,
  vulcan-forward's `tests/test_ckd.py` 7 passed. `tools/ab_cuda_solve.py` against
  `bd97791` (jobs 96222, 96238): every digest bitwise equal, the subnormal-pivot
  block exact on the new kernel, solve time +6.6% (W39b) and +8.1% (random); the
  numbers are in notes "Efficiency program".
- The cleanup (`2216219`) removed the `tiny` guard case from
  `tests/test_solver_fast.py` (row 3 scaled by 2^-975), the only test of the
  Gauss-Jordan kernel's division fallback; restoring it against the CPU kernel's
  first layer is the maintainer's call.
- Decision: keep the guard as is, or recover the ~7% and re-measure with
  `tools/ab_cuda_solve.py`.
  Done when decided; then delete `tools/ab_cuda_solve.py`.
