# CLAUDE.md — VULCAN-JAX (package vulcan3)

## Docs

- Four root docs, nothing else: `README.md` (hand-written by the maintainer; never
  edit it unless asked for that edit), `notes.md`, `todo.md` and this file. The
  last three are tracked and public: write them for public reading (no personal
  names, contact details, account names or credentials). No other `.md` file in
  this tree. Policy and the no-bloat rules: the project-root `CLAUDE.md`.
- Read `todo.md` before work; open or close items in the same pass.
- `notes.md` holds what is true and why; this file holds the rules. Refer to a
  notes section by its name. Record every rejected approach under notes "Do not
  retry" in the same pass; end any behavior change with a doc pass on notes.md
  (and here if a rule changed).

## Project

A JAX port of VULCAN (exoclime), standalone: vendored data, no sibling at runtime,
one dependency beyond jax / numpy / scipy / PyYAML (`exogibbs` for the equilibrium
seed, installed `--no-deps` plus `pandas`). Port all of master's live physics;
dead-in-master paths are not ported; do not chase oracle tests for every knob.

## Upstream and oracles

- The VULCAN 2 reference is the pinned clones under `VULCAN_Project/.oracles/`
  (revisions in `tests/science_sources.yaml`), given as `$VULCAN_MASTER_DIR`; a
  dirty clone is refused (on daw use the copy named in notes "Current state").
  Never cite a hand-patched VULCAN-master copy for what upstream ships, or diff a
  disputed step count's config against one (notes "Oracle policy").
  `exoclime/VULCAN` now serves VULCAN 3: fetch from it only for VULCAN 3.
- `op.py:NNN` / `build_atm.py:NNN` citations are exoclime@80f75b9
  (vm_branch@84d010d where marked). Never write "upstream does X" from memory or
  a code comment: read the pinned source and cite the line, or measure.
- Static-vs-dynamic assumptions are the highest-risk class (refreshed in-loop:
  hydrostatic column, escape flux, vm, photo rates, conden rates; never: Kzz,
  Dzz, M, T, cross sections).
- To run master as an oracle, copy the whole tree (no symlinks: upstream setup
  writes into `thermo/` and `fastchem_vulcan/`), fix the cfg in the copy, and pin
  `PYTHONHASHSEED=0`.

## Parity rules

1. Match VULCAN master by default. An intentional divergence must fix a confirmed
   results-affecting defect; record it in notes "Parity ledger" (file:line or
   symbol on both sides) and "Differences from VULCAN master".
2. Never edit a pinned clone. A defect corrected on both sides is declared once
   in `tests/oracle.py::ORACLE_CODE_DELTAS`, applied to the disposable copy only.
3. Only results-affecting bugs go in the ledger. The target is scientific parity,
   not byte identity.

A physics-neutral change keeps HD189 1495 / HD209 1211 / W39b 1202
(`tests/test_step_count_gates.py`, full tier) on both `fast` and `reference`.
HD189_vulcan3's count is not a gate: an arm64 number that moves with any bit-level
change (notes "Stage repair and element budget"). A transport-kernel refactor
also needs vulcan-forward's full-tier `test_queue_is_differentiable`.

## Environment and commands

The root uv env `VULCAN_Project/.venv` (`uv sync` at the project root) installs
this package editable (standalone: `pip install -e . --no-deps`); conftest fails
collection on a non-editable install. `unset PYTHONSAFEPATH` before pytest; on
daw set `JAX_PLATFORMS=cpu` unless a GPU test is the point.

```bash
vulcan3 -c HD189                                                  # config name or YAML path
python -m pytest tests -n 12 -q --tb=short -ra                    # normal tier
VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q --tb=short -ra # full tier
ruff check . ; vulture . --min-confidence 80
python tools/audit_master_parity.py --master $VULCAN_MASTER_DIR
```

- Use `python -m pytest`, not the console script.
- CI `tests.yml` pins jax 0.6.2 (pyproject holds the floor) and runs the normal
  tier on three workers; `oracle.yml` (by hand) runs the full tier one process
  per slow test (notes "Evidence behind the numerical rules").
- A process at an `XLA_PYTHON_CLIENT_MEM_FRACTION` other than 0.90 uses a private
  JAX compile-cache dir (notes "Operational").
- Push only when the maintainer asks; after a push, check `imalsky/VULCAN3` CI.

## Architecture

```
outer_loop.OuterLoop                       one JIT'd lax.while_loop
  └── jax_step.jax_ros2_step
        ├── chem_funs codegen RHS (master-faithful term order)
        ├── chem.chem_jac_analytical
        └── solver_fast.factor / solve (default), or the reference pair
            solver.factor_block_thomas_diag_offdiag / solve_block_thomas_diag_offdiag
  photo.py kernels fire via lax.cond; setup = state.RunState.with_pre_loop_setup(cfg);
  .vul writer = legacy_io.Output.save_out(runstate, dname), RunState only.
```

- `integ(rs)` (`OuterLoop.__call__`) takes a fresh `RunState` only. `integ(rs,
  chunk=N)` (the CLI's path) runs the same loop through `runner_chunk` and must
  stay bitwise equal to it; never give `_runner` a chunk carry leaf.
- A config file overlays `default.yaml`, and `default.yaml` IS
  `default_config()`: a fix applied to named configs only silently changes every
  API user's run. Removed keys are refused with a remedy (`config._REMOVED_KEYS`;
  add every key you remove from `default.yaml`), also via
  `vulcan3.validate_overrides`; master's display-only keys (`config._IGNORED_KEYS`)
  warn and are dropped from a file, and an override dict refuses them.
- `make_config(...)` overrides need all three wirings: `_cfg_overlay` (setup),
  `OuterLoop(..., cfg=)` (runner), `Output(cfg=)` (output). Import-frozen knobs
  (`config._FROZEN_ENV`: `network`, `com_file`, `atom_list`) and `$VULCAN3_SOLVER`
  / `$VULCAN3_CHEM_RHS` are set in the environment before the first import; the
  CLI re-execs once to match a config, the Python API does not.
- `OuterLoop.run_batch` (freeze-on-done) and `run_queue` (the same loop refilled
  from a job queue; tail compaction `queue_compact` off by default): per-profile
  arrays ride the `ProfileVars` carry, never a closure (vmap does not batch
  closure constants). Only T-P may vary across a photo batch; conden config and
  `pref_indx` are closure-baked. `run_queue` is bitwise `run_batch` when
  `n_lanes >= n_jobs`; a fresh lane takes photolysis exactly once, on its own
  fields. Compare a batched entry against vmap-of-solo, never the solo runner.
- Test-only reference kernels live in `tests/_oracles.py`.

## Numerical rules (do not regress)

- float64 always (`jax_enable_x64` at import).
- The production Jacobian is `chem.chem_jac_analytical`: a left-to-right sum along
  static tables (`chem._jac_gather_tables`), no scatter-add and no `reduce`, one
  summation order in every program. It never raises to the 0 power (`y**0` has a
  NaN tangent at y=0).
- A where-zeroed numerator over a guarded denominator (the delta and longdy
  maxima) takes denominator 1, else a sub-atol density gives `0 * inf` in the
  batched tangent.
- Benchmark solver changes on REAL blocks from `output/*.vul` at the run's dt AND
  at `dt_max=1e11`, never synthetic ones; for retrieval throughput, at ni 89 under
  `jvp` (notes "Solver conditioning"). Jit checks take fixture arrays as
  ARGUMENTS, never closed over.
- `VULCAN3_SOLVER=fast` is the default, `ffi` the CUDA-kernel opt-in (production
  GH200), `reference` the parity pair; reverse mode through `ffi` is unsupported.
  Forward mode uses the matrix-free stage operator (`jax_ros2_step(matrix_free=
  True)`); a reverse-mode caller passes `matrix_free=False`, as
  `steady_state_grad` does. `VULCAN3_CHEM_RHS=grouped` stays opt-in until
  it is bitwise in full CPU runs too (notes "Efficiency program").
- `pos_cut`/`nega_cut` clipping stays OUTSIDE the JIT'd step and reads the
  POST-solve `ymix`. Asymmetric three-body M: forward and reverse slots are set
  independently. The codegen RHS is SymPy-faithful (no XLA rewrite of the
  multiply chains).
- The atom-conservation projection and the Ros2 stage repair
  (`jax_step._repair_stage`, C19) are default-on and NOT in VULCAN 2.0; both
  change the solution. `config.REPAIR_ABS_FLOOR`, `jax_step._DEFECT_FLOOR` and
  `jax_step._REPAIR_MAX_CELL_FRAC` are re-baseline knobs, not tuning ones; a
  relative floor or a dt gate is the wrong tool. `dt_max` is capped at
  `config.DT_MAX_S`; a larger explicit value raises (notes "Stage repair and
  element budget").
- Step acceptance uses the import-bound `rtol` (C16); bottom pins apply before
  delta/clip (C17); the photo-frequency switch waits for `trun_min` / `count_min`
  (C18); the escape Jacobian term exists only under `use_vm_mol` (C15); every
  config keeps `batch_max_retries` 110 (C11) (notes "Parity ledger").
- The certificate adds two terms NOT in VULCAN 2.0, both inert below
  `count_min`: geometry (C21; on the certificate candidate the runner refreshes
  the geometry whatever `update_frq` says and ends only if mu/g/Hp/dzi/Hpi moved
  less than `geom_conv_tol`) and the cumulative element budget (C23; each
  element's operator-weighted column RELATIVE TO H, accumulated per step on each
  step's own grid; never the absolute column, and `loss_eps` is no substitute).
  `vulcan3.conv_normal(final, cfg)` exports the whole certificate; a consumer
  calls it, never a copy (notes "Certificates").
- A forward-mode tangent is certified by `OuterLoop.run_jvp` (C22), not by
  `jax.jvp(runner)`; pass the direction in finite-difference-step units. Never a
  step floor on a differentiated map.
- A run ends only on the certificate or a cap (`use_conv_stall` is refused). The
  plateau counters are diagnostics that no consumer reads; they feed no predicate.
- `conver_ignore`: parity configs ship `[]`, the V3 preset `['HC3N']`; never pick
  a list because it reproduces a step count. Every shipped run converges on the
  LOOSE branch; quote the realised `longdy` next to any "converged" claim.
- `vm` (upwind moldiff) is refreshed IN-LOOP. A hybrid vm_mol result is a
  differentiable steady state only when `end_case == 1` AND terminal
  `hybrid_use_vm == 0`. The scheme matters only at low Kzz; HD189 is not evidence
  for it.
- Gravity derives from `Mp`/`Rp`; there is no `gs` knob.
- Differentiate the inner `integ._runner`, never `OuterLoop.__call__`. Reverse mode
  at the converged state is `steady_state_reaction_sensitivity` (the renormalised
  map, `steady_state_grad.SOLVER_MAP`; `body_dt` 1e7); on photo columns pass
  `photo_recompute_k=make_photo_recompute_k(integ._photo_static, final)` (notes
  "Adjoint record"). T-profile gradients rebuild `k_arr` via
  `rates_jax.build_rate_array` and the structure via `atm_setup.mu_dz_g_jax` /
  `atm_refresh.update_mu_dz_jax` and `atm_jax._mol_diff`.
  Run `audit_adjoint_scope` before trusting a gradient on a non-default config;
  never weaken it to pass a state.
- Keep the full-tier `test_forward_jvp_integration` (Kzz JVP vs re-converged FD);
  a single-step or reverse-mode check does not replace it (notes "Forward mode").
- Condensation (upstream `master_pin`) is NOT differentiable-through; the
  autodiff entry points refuse it (notes "Condensation contract (F1-F5)").
- Adding a field to `AtmStatic` touches three places in src (the NamedTuple,
  `make_atm_static`, `_ATM_STATIC_BATCH_AXES`) plus every direct `AtmStatic(`
  call in examples/ and tests/. Splice defaults into saved-fixture loaders
  rather than regenerating.
- Network id columns are OUTPUT; index photolysis by parser POSITION (C14).
  `cfg.remove_list` is positional in both codes.
- Every config declares `fastchem_solar_abundance_file`; match it before quoting
  a cross-code number (unmatched, composition dominates). The Lodders 2009 preset
  is not an agreement case with upstream's seed.
- Earth and K2-18b were removed (neither converged here); re-add only with a
  converging run (todo).

## Output and tests

- `.vul` = pickle with master's three top-level keys and the same public keys,
  shapes and dtypes, so `plot_py/` loads it. Byte identity is a non-goal.
- A function earns a test if it is hot-path, a numerical bridge to master, a
  jit/vmap/grad boundary, or a parser; the bar is "could fail silently and
  matter". Patterns: master-equivalence at machine tolerance, vmap consistency
  ~1e-12, gradient vs FD ~1e-6.
- The normal tier stays under 10 min on daw. A test that takes minutes or several
  GB is marked `slow` (full tier, `VULCAN3_RUN_SLOW=1`); a workflow never
  deselects a test for its cost. Master-oracle tests run only with
  `$VULCAN_MASTER_DIR` set, in subprocesses against a disposable copy.
- conftest fails on missing `tests/data` fixtures (`python tests/gen_fixtures.py
  --all`). `VULCAN3_ALLOW_MISSING_FIXTURES` is the developer escape hatch;
  `tests.yml` sets it, `oracle.yml` must not. Check a CI full-tier failure on daw
  only after `gen_fixtures.py --all --force`.
- Vendored inputs are hash-pinned in `tests/science_sources.yaml`; re-pin in the
  same commit as any edit.

## Style

- Magic numbers live in the config YAML; a literal in source gets a named
  constant with provenance.
- No dead code (`vulture --min-confidence 80` clean), but grep the pinned clone
  and the consumers before deleting a master-facing facade method (`op.py:951`
  calls `.mean_mass()` on the `Atm` facade; vulcan-forward calls
  `naming_solver`). No legacy fallbacks, no phase-history comments.
- Fail fast and loud: an unmet precondition raises with the offending value and
  the remedy; a skipped check announces itself.
- Vectorize with vmap; `lax.scan` only for true sequential dependencies. The one
  exception is the EQ seed (`custom_vmap` + `lax.map`, so a lane is bitwise
  independent of the batch width).
- Quote ratios, not ms pairs; name the host and code state for an ms pair, and
  believe a wall-time ratio only when `user+sys ~ real`.
