# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Confirm the remaining departures from VULCAN master (projection, stage repair, `DT_MAX_S`, abundance file, retry cap; both referees keep them), schedule the owed measurements, and decide whether batches B-D of the provenance audit are owed.
2. [decision] Reply to the external user's report of 2026-10-14 (not yet sent); items 3-5 wait on the files and logs it asks for.
3. [bug?] The external user reports that Earth converges in VULCAN but not in vulcan3 with the same inputs (their 0.3.1 tree, which predates the element-budget term). On main that term also blocks Earth: it assumes a closed column.
4. [bug?] The external user reports that vulcan3 runs differ from run to run on their machine.
5. [unverified] The external user reports that K2-18b (H2O clouds) converges in < 4000 steps; re-add it once it converges on main.
6. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.
7. [robustness] The CUDA block-Thomas solve multiplies by 1/pivot with no division fallback, and the FFI build stamp ignores the CUDA arch and toolchain; both need a GPU host with nvcc (NAS) to change and test.

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

The reply should ask for
`SNCHO_DMS_photo_network_2025.txt` and confirmation of the K2-18b atm file (item 5),
the Earth YAML, atm/BC files and commands (item 3), the exact P error text, and the
platform, jax version and two differing logs (item 4). Done when sent.

## 3. Earth: converges upstream, not here

Evidence: notes "Withdrawn configs" (2000 matched steps only; a full-length run
was never done) and "Do not retry" (the falsified cures).

That record and the user's 0.3.1 tree both predate the element-budget term
(0.9.0), so the term explains neither failure.

On main the term is a second blocker [I]: upstream's Earth example opens the
column (`use_botflux`, `use_fix_sp_bot` for H2O and CO2, `fix_species`;
`vulcan_cfg_Earth.py:98-99,117` in the pinned clone), and the recorded Earth run
gained H (`atom_loss[H] = 2.98e3`, notes "Withdrawn configs"). The term reads
each element relative to H, so that inflow shows as a drift near -1 in every
other element (tolerance 1e-2). Such a run cannot certify even if its chemistry
has converged, and it exits as "Maximal allowed steps exceeded" without naming
the term. Upstream has no such term.

Runs, all to `count_max` (~30-60 min each [E]; ask before running): upstream
from a disposable copy of the pinned clone (it needs the C13 fix and four
missing keys to start, notes "Withdrawn configs"); vulcan3 with the converted
YAML, `JAX_PLATFORMS=cpu vulcan3 -c <earth.yaml>`, from a scratch directory
holding the clone's `thermo/SNCHO_full_photo_network.txt` and
`atm/atm_Earth_Jan_Kzz.txt` (no longer in the package); and vulcan3 again with
`element_budget_tol: 1e9`. The conversion drops the refused `gs` (set `Mp`),
`fix_species_time`, `gibbs_text`, `ode_solver`, `use_print_delta` and
`fastchem_met_scale`, the explicit `dt_max` (1e17 exceeds `config.DT_MAX_S`;
derived it is 1e15) and `Ar` (C13). The user's Earth inputs (item 2) matter only
where they differ from the clone's. Done when the cause is a confirmed defect (a
"Parity ledger" entry), the codes agree, or the budget term is shown to be the
only blocker (then the maintainer decides how an open-column config certifies;
the term itself stays).

## 4. Run-to-run differences

Evidence: notes "Operational" and "Oracle policy": vulcan3 is bitwise
reproducible on daw CPU and GPU; upstream's trajectory depends on
`PYTHONHASHSEED`. Re-checked at `7d24973`: two `vulcan3 -c HD189` runs on one daw
core are bitwise equal in y, t, dt and longdy (1495 steps each). The user's tree
was 0.3.1. Waiting on platform, jax version and two differing logs from a run on
main (item 2). Done when reproduced and fixed, or shown to be their old tree or
upstream.

## 5. K2-18b re-add

Evidence: notes "Withdrawn configs". The user's YAML (`K2-18b-H2Ocloud.yaml`,
attached to their 2026-10-14 report) was written for the 0.3.1 tree: main refuses
it, naming 12 removed keys; with those gone it warns about and drops 12
display-only keys. It also pins `fix_species` with condensation on, which opens
the column for the element-budget term (item 3). Blocked on the user:
`thermo/SNCHO_DMS_photo_network_2025.txt` is not on daw (only
`SNCHO_DMS_photo_network_Tsai2024.txt` is vendored), and the atm file
`atm_K2-18b-Nep100X-apo-H2Oclouds.txt` must be confirmed. Then convert the YAML
(sectioned; keep `print_prog_num` and the supported plot keys), check what `remove_list
[891, 892]` selects by position (C14), add the network and atm file with hash pins
in `tests/science_sources.yaml`, and run `JAX_PLATFORMS=cpu vulcan3 -c K2-18b` on
daw (~0.13 s/step [E]: ~9 min to 4000 steps, ~45 min to the 20000 cap; ask
first), and again with `element_budget_tol: 1e9` if it does not certify. Done when end_case 1 with terminal hybrid phase 0 and the realised longdy
quoted; then update notes "Withdrawn configs" and "Do not retry", and the
CLAUDE.md line on Earth and K2-18b.

## 6. vulcan3 on PyPI

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

## 7. CUDA solve reciprocal and the FFI build stamp

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
