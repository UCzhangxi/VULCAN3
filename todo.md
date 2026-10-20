# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Send the reply to the external user's 2026-10-14 report; the decisions it carries are settled (detail below).
2. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
3. [unverified] A/B the CUDA solve guard against `bd97791` on a NAS GH200: it is built and the `ffi` tests pass there (2026-10-20); the bitwise A/B, the subnormal-pivot block and the solve time are not done.

## 1. Reply to the external user

Their report came from a 0.3.1 tree. Not yet sent.

Decided by the maintainer (2026-10-20):
- The configs stay inside the package (a pip install needs them there);
  `vulcan3 --config <path>.yaml` runs a file from anywhere.
- The equilibrium start uses ExoGibbs (FastChem removed in `fcd4c28`, 0.15.0);
  say so in reply to the FastChem 4 offer.
- Earth: on main the clone's example converges once the element-budget term is
  off (`element_budget_tol: .inf`) and then matches upstream to 0.4% in the bulk
  species; an unconverged run now names the term that held it and this switch
  (notes "Withdrawn configs"). Their tree predates the term.

Also tell them (checked in the code on main):
- Ctrl+C opens a shell with the run state, and `use_plot_end` / `use_plot_evo`
  plot after a run (`cli.py`).
- The repo is `imalsky/VULCAN3`, package `vulcan3`. Configs are sectioned YAML.
- An extra element such as P in `atom_list` passes validation when the
  composition table and the network both carry it (`runtime_validation.py`); not
  run.
- `use_sat_surfaceH2O` set the initial H2O column but not the bottom H2O pin from
  `f93e7e0` (likely their 0.3.1 tree too) until the fix on main.

Ask for or settle:
- Earth: their inputs where they differ from the clone's example.
- Run-to-run differences: platform, jax version and two differing logs from a run
  on main. vulcan3 is bitwise reproducible on daw (two `vulcan3 -c HD189` runs
  bitwise equal at `7d24973`); upstream's trajectory depends on `PYTHONHASHSEED`.
- K2-18b: `thermo/SNCHO_DMS_photo_network_2025.txt` (not vendored; only the
  Tsai2024 network is) and confirmation of `atm_K2-18b-Nep100X-apo-H2Oclouds.txt`.
  Re-add recipe: notes "Withdrawn configs".

Done when the reply is sent and each answer is filed (a todo item if it clears
the bar, else notes).

## 2. vulcan3 on PyPI

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

## 3. CUDA solve guard and build stamp

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
- Done means: every A/B digest bitwise equal at 792 lanes, the subnormal-pivot
  block finite on the new kernel, solve time unchanged, the `ffi` tests passing;
  then delete `tools/ab_cuda_solve.py`.
