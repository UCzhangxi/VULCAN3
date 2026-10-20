# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Confirm the remaining departures from VULCAN master (projection, stage repair, `DT_MAX_S`, abundance file, retry cap; both referees keep them), schedule the owed measurements, and decide whether batches B-D of the provenance audit are owed.
2. [decision] Reply to the external user's 2026-10-14 report: what to tell them (the ExoGibbs seed, what is already on main) and what to ask for (Earth, run-to-run differences, K2-18b).
3. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
4. [robustness] The CUDA block-Thomas solve multiplies by 1/pivot with no division fallback, and the FFI build stamp ignores the CUDA arch and toolchain; both need a GPU host with nvcc (NAS) to change and test.

## 1. Confirm the departures from VULCAN master

Evidence: notes "Differences from VULCAN master" (each item, its provenance and the
referee verdicts) and "Solver conditioning". All are live on main and in every
production run:
- the atom-conservation projection of the chemistry RHS and Jacobian
  (`jax_step._project_chem_rhs` / `_project_chem_jac`, from ccc40f7);
- the Ros2 stage repair (`jax_step._repair_stage`) and `config.DT_MAX_S`;
- the certificate terms with no VULCAN 2 counterpart, `geom_conv_tol` and
  `element_budget_tol` (C21, C23);
- the FastChem solar abundance file swap (ccc40f7);
- the retry cap with force-accept (`batch_max_retries` 110).

Verdicts: both referees keep the projection, the stage repair, `DT_MAX_S`, the
abundance file and the retry cap. The two certificate terms are decided: keep
(notes "Decided by the maintainer, not to be re-opened").

Owed measurements (none blocks the v4 launch):
- Certificate-term A/B, no code change: ~50 v2 posterior draws (include lnKzz <
  -0.7), cold at nz 100, with and without `cfg_overrides` `geom_conv_tol: 1e9` and
  `element_budget_tol: 1e9` (both pass `vulcan3.validate_overrides`). In
  vulcan-retrieval an override file's `cfg_overrides` replaces the case's dict, so
  restate its other keys (Rp, rtol, photolysis cadence). Compare certified set,
  accept counts, `budget_drift_max` and binned depth in sigma.
- Stage repair on/off: ~20 v2 posterior draws cold, with `_repair_stage`
  monkeypatched to identity (not `_CHEM_PROJECTION_ENABLED`, which also turns off the
  projection); compare certified set, budget drift and depth in sigma, and record a
  |correction|/carrier histogram from the same runs. Both referees ask for this one.
- Projection on/off on real columns.
- A `dt_max` sweep 1e11..1e17 on difficult saved columns (above 1e15 needs
  `config.DT_MAX_S` raised in a scratch copy; a larger `dt_max` raises).
- Photolysis cadence: W39b 5/5 and the batched shared-tick cadence against cadence 1.
- Block-local pivoting: real large-dt stage matrices against LAPACK banded LU
  (`tests/_oracles.banded_solve`).

The audit covered batch A only (e46234a..7eda1af, 60 commits); the later 328
commits (batches B-D) were not audited. vulcan-forward's `conv_normal` and its
per-theta re-anchoring are the same question on the consumer side (vulcan-forward
todo, "conv_normal certificate terms"). Done when each item is confirmed or
scheduled for removal, the certificate-term decision is in notes "Design records",
and the B-D audit is done or declined.

## 2. Reply to the external user

Their report came from a 0.3.1 tree. Not yet sent.

Tell them (checked in the code on main):
- Ctrl+C opens a shell with the run state, and `use_plot_end` / `use_plot_evo`
  plot after a run (`cli.py`).
- The repo is `imalsky/VULCAN3`, package `vulcan3`. Configs are sectioned YAML.
- The configs stay inside the package (a pip install needs them there);
  `vulcan3 --config <path>.yaml` runs a file from anywhere.
- An extra element such as P in `atom_list` passes validation when the
  composition table and the network both carry it (`runtime_validation.py`); not
  run.
- The equilibrium seed is ExoGibbs, not FastChem (`fcd4c28`, 0.15.0). This bears
  on their offer to add FastChem 4, and on whether the condensation solver needs
  a JAX version.

Ask for or settle:
- Earth: their inputs, where they differ from the pinned clone's example. Their
  tree predates the element-budget term, so the term is not their failure; on main
  it also blocks open columns (notes "Stage repair and element budget"). The
  comparison and its recipe: notes "Withdrawn configs".
- Run-to-run differences: platform, jax version and two differing logs from a run
  on main. vulcan3 is bitwise reproducible on daw (two `vulcan3 -c HD189` runs
  bitwise equal at `7d24973`); upstream's trajectory depends on `PYTHONHASHSEED`.
- K2-18b: `thermo/SNCHO_DMS_photo_network_2025.txt` (not vendored; only the
  Tsai2024 network is) and confirmation of `atm_K2-18b-Nep100X-apo-H2Oclouds.txt`.
  Re-add recipe: notes "Withdrawn configs".

Done when the reply is sent and each answer is filed (a todo item if it clears
the bar, else notes).

## 3. vulcan3 on PyPI

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

## 4. CUDA solve reciprocal and the FFI build stamp

- Solve: `bt_lu_solve_warp0` (`csrc/block_thomas_cuda.cu`) takes `1.0 / u_ii` and
  multiplies, with no fallback; the kernel's own comment says a pivot below 2^-1024
  in magnitude gives inf/nan (1 / 1e-310 overflows where `t_i / u_ii` can be
  finite). The Gauss-Jordan factor already divides when `bt_div_by_rcp`'s range is
  left; the solve does not. No production solve is known to reach such a pivot
  (notes "Solver conditioning").
- Build stamp: `solver_fast._source_hash` covers the kernel source and
  `block_thomas_common.h` only. The `-arch` from `_nvcc_arch` and the nvcc version
  are not in it, so a library built for another arch or toolchain counts as
  current and is not rebuilt.
- nvcc is not on daw: change and test on NAS. Gates: the `ffi` tests in
  `tests/test_solver_fast.py`, and bitwise equality with the current kernel at the
  production shape (pivots in range).
- Done means: the solve divides outside the reciprocal's safe range (bitwise
  unchanged inside it), and the stamp covers the arch and the nvcc version.
