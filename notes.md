# VULCAN3 notes

Findings from the WASP-39b campaign (Oct 2026) that concern the solver itself. The
retrieval-side studies, runs and open items are in vulcan-retrieval's `notes.md` and
`todo.md`. The study outputs lived in `analysis/` on NAS, deleted 2026-10-19; a
reference to `analysis/<name>` anywhere means the `<name>` entry here or in
vulcan-retrieval's notes. [M] measured, [E] estimated.

## Speed work (all bitwise or roundoff-level, all on main)

Production shape: W39b SNCHO, nz 62, ni 89, GH200.

| Change | Commit | Effect |
|---|---|---|
| Gauss-Jordan block-Thomas factor + reciprocal-pivot solve | d992314 | 1.24x per step, 75.2 -> 60.7 ms/iter at 396 lanes; dlogL median 9e-8 |
| XLA: P3 column-layout Jacobian (always on), P2 grouped RHS (opt-in `VULCAN3_CHEM_RHS=grouped`) | 3154d6d | P1 input fence (~1%) and P4 sparse projection (flips low-Kzz lanes) dropped |
| Carry copies: y_time ring written once, photolysis cond spans the step | 309cca9 | 75.0 -> 58.9 ms/iter at 396 lanes |
| Lane count 792 (FFI_BLOCKS_PER_SM 3) | config | ms/iter 264:25.8 396:40.0 528:49.7 792:67.2 1188:104.4; queue throughput 2.03x stock-396 |
| Tail compaction (queue_compact) | 01f1558, d7021cb | pass 1 520 -> 344 s; the retry pass at width 320 is roundoff-level, not bitwise |
| speed2 kernel: REDUX argmax (bt_warp_argmax), bt_div_by_rcp, cp.async solve loader | a5b1215 | factor 29.1 -> 23.1 ms, solve 5.3 -> 4.6 ms at 792; pass 1 343.8 -> 300.6 s |
| speed2 tail loops run the main body (no unconditional ring copy) | eeb4873 | the ring copy was 5.6 GB/tick at 264 lanes, 2.8 at 132; with the speed2 retrieval changes, pass 1 344.0 -> 287.3 s |

Kernel limit [M]: the factor is latency-bound on the serial pivot chain (89 steps per
layer x 62 layers), at ~7% of FP64 peak (2.4 TFLOP/s useful of 33.5), occupancy 25%,
barrier stalls 42%. One pivot step ~1500 cycles before the speed2 kernel, 838 after;
the column chain alone has a floor of 623. Blocked, panel and tensor-core variants were
not faster (bgj8sf 29.3 ms, bgj8s 33.7, bgjl8 76.5), although DMMA reaches 64 TFLOP/s in
isolation. ~60% of the remaining factor time is the pivot chain [E]: further gains need
fewer serial steps.

Dropped: parallel-in-layer solvers (BCR 89 ms vs 39 ms block-Thomas at 396 lanes; OOM
at 1188), explicit-inverse solves (forward error ~600x), dt kick (lost certifications),
interpolated starts (|dlogL| up to 3 at rtol 0.2), tangent starts (1.13x), a 4-rung
compaction ladder (not bitwise).

## Grid resolution: a real VULCAN discretization effect

- At 62 layers (6.2/decade over 1e-9..7.6 bar) the grid smears a narrow SO2 depletion
  below a sharp OH front (~2e-4 bar) and over-produces SO2 by up to ~1 dex at
  1e-4..3e-3 bar. Only SO2 and SO change; convergence order p = 2.09 [M].
- Unmodified numpy VULCAN master (exoclime c870b1a) reproduces it on three posterior
  draws: SO2 column (1e-4..1e-1 bar) master minus vulcan3 <= 0.002-0.005 dex at every
  nz; 62 -> 300 layers moves draw 117 by -1.713 dex in both codes, draw 301 by -0.072 in
  both, draw 345 -0.878 (master) vs -0.914 (vulcan3). Master's own shipped W39b example
  is already converged (75 vs 600 layers: <= 0.03 dex). Not a port bug. (vulcan2_xcheck)
- Low-Kzz "bistability": two locally stable states that differ by where the
  photochemical front (H2 -> H, H2O photolysis, 1e-7..1e-6 bar) sits, by whole grid
  layers. At 100 layers 35/40 split lanes merge, at 150 layers 40/40. A 62-layer grid
  artifact, not physics. (bistable_policy)
- Non-uniform pressure grids are not supported (no pco-file input); a 12-line patch to
  state.py was prototyped and not kept.

## Differences from VULCAN master (vulcan2_audit, provenance audit)

- Convergence test: master's conv() terms plus stricter JAX-only terms (geometry
  tolerance, element budget, HC3N not ignored, mtol_conv 1e-20).
- Per-run settings that differ from master's W39b example: fixed rtol (0.2 presolve /
  0.05 certified; master 0.05 adaptive), cap-and-reject at 3000/5000 accepted steps
  (master stops at 1e4-2e4 and saves the unconverged state), dt_max 1e11 (1e17),
  photolysis keyed to ticks, the older network (the 2025 network changed no
  certification, +5% per tick), nz 62 over 1e-9..7.6 bar (master: 150 over 5e-9..50 bar).
- Provenance audit (2026-10-19), early history only (batch A: e46234a..7eda1af,
  Apr-Jul 2026, 60 commits; no NAS transcripts exist, so each item needs the user's
  confirmation). Status at main 2c08a01:
  - STILL ACTIVE: per-layer element-conservation projection of the chemistry RHS and
    Jacobian onto reservoirs (H2/H2O/CO/N2/H2S), switched on automatically for HD189 and
    W39b (ccc40f7, not in its commit message, not in master; jax_step.py:44-141, used at
    :141, :650, :707). Also live, from later unaudited commits: the Ros2 stage
    element-defect repair `_repair_stage` (jax_step.py:226, called :720, :738; "Not in
    VULCAN 2.0").
  - STILL ACTIVE: certificate terms with no VULCAN 2 counterpart, geom_conv_tol 1e-3 and
    element_budget_tol 1e-2 (default.yaml:176/180, W39b.yaml:170-171).
  - Still the default: FastChem solar_element_abundances.dat replaced in ccc40f7 (He and
    S changed, rocky elements -3; upstream's kept as ..._lodders2009.dat). For W39b the
    config's X_H values override it (ini_abun.py:311-322), so only He comes from the file.
    A row-order bug left carbon atomic for EQ-initialised runs ~05-07..06-08 (fixed
    45cc4c7).
  - Still in code, default now 110: retry cap with force-accept after
    batch_max_retries rejections (outer_loop.py:1074-1080); dt < dt_min ends a reject
    chain first, as in master.
  - Still in default.yaml from 3938cf9: rtol 0.2, use_adapt_rtol false, count_max 1e4
    (not checked against upstream's HD189 example).
  - Removed: the JAX-only "stall" convergence rule (89b5962/9c2bb75; removed c009555,
    2026-09-25, before the fixedmass and v2 runs); c957288 accept test on adapted rtol
    (reverted); bf01cd6 W39b default with wall_clock_max 1800 s (reverted); use_vm_mol
    true (now false); 9c2bb75 conver_ignore (now []). Kept: gravity G Mp/Rp^2.
  - e841d77: SNCHO reaction 949 k0 1e-20 -> 1e-29 (matches master's own fix).
  - The audit of the later 328 commits (batches B-D) was never done.

## Operational

- The JAX persistent compile cache key ignores XLA_PYTHON_CLIENT_MEM_FRACTION. Job 93577
  loaded entries compiled at fraction 0.04 and produced wrong pass-1 results (6114
  ticks / 1679 unconverged vs 7658 / 237). Any process at a fraction other than 0.90
  needs a private cache dir.
- Node cgh10n04 has a broken GPU: fail fast when `jax.default_backend() != 'gpu'`.
