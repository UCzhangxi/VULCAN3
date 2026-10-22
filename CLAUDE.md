# CLAUDE.md — VULCAN-JAX (package vulcan3)

A JAX port of the VULCAN 2 photochemical-kinetics solver (exoclime), shipped as
`vulcan3` (GitHub `imalsky/VULCAN3`): float64, differentiable, vmap- and
GPU-capable, standalone (vendored data, no sibling repo at runtime). It is the
root of `vulcan3 <- vulcan-forward <- {vulcan-retrieval,
jwst-transit-authority}`, so a change here reaches every consumer.

## Docs

- Four root docs, nothing else: `README.md` (hand-written by the maintainer;
  never edit it unless asked), `notes.md` (what is true and why, by section;
  under 1000 lines: pay for a new line by trimming a duplicate or stale one),
  `todo.md` (open work: read it first, open or close items in the same pass) and
  this file (directions). The last three are tracked and public: no personal
  names except cited authors, no contact details, IDs, accounts, keys or chat
  quotes. No other `.md` file. Name a notes section by its heading, never a
  number.
- Record a rejected approach under notes "Do not retry" in the same pass; end a
  behavior change with a doc pass on notes.md (and here if a rule changed).
  Format: the `doc-upkeep` skill; policy: the project-root `CLAUDE.md`.

## Setup and commands

The root uv env `VULCAN_Project/.venv` (`uv sync` at the project root) installs
this package editable; standalone, `pip install -e . --no-deps` plus `exogibbs`
(`--no-deps`) and `pandas`, as `tests.yml` does. conftest refuses a
non-editable install.

```bash
vulcan3 -c HD189                                                  # config name or YAML path
python -m pytest tests -n 12 -q --tb=short -ra                    # normal tier
VULCAN3_RUN_SLOW=1 python -m pytest tests -n 12 -q --tb=short -ra # full tier
ruff check . ; vulture . --min-confidence 80
python tools/audit_master_parity.py --master $VULCAN_MASTER_DIR
```

- Run `python -m pytest` (the console script breaks the parity-audit imports)
  after `unset PYTHONSAFEPATH`, with an explicit `-n`. Tests default to CPU
  (`tests/conftest.py`); GPU use on daw: the project-root `CLAUDE.md`.
- Normal tier: everyday checks and CI, under 10 min on daw. Full tier
  (`VULCAN3_RUN_SLOW=1`): the `slow` tests; oracle tests need
  `$VULCAN_MASTER_DIR`.
- CI: `tests.yml` (every push) runs the normal tier on three workers with jax
  pinned at the version the tolerances are calibrated against (pyproject holds
  the floor); `oracle.yml` (release gate, by hand) runs the oracle comparisons
  and the full tier, one process per slow test (notes "Evidence behind the
  numerical rules"). Push only when asked; then check `imalsky/VULCAN3` CI to
  completion.

## Rules

### Physics and parity

- Match the pinned VULCAN 2 by default. Port all of master's live physics, not
  its dead paths; do not chase oracle tests for every knob.
- An intentional divergence must fix a confirmed, results-affecting defect;
  record it in notes "Parity ledger" (file:line or symbol on both sides) and
  "Differences from VULCAN master". Only results-affecting bugs go in the
  ledger: the target is scientific parity, not byte identity.
- The VULCAN 2 reference is the pinned clones under `VULCAN_Project/.oracles/`
  (revisions in `tests/science_sources.yaml`), passed as `$VULCAN_MASTER_DIR`;
  a dirty clone is refused. Never edit a clone: declare a defect corrected on
  both sides once in `tests/oracle.py::ORACLE_CODE_DELTAS`, which applies it to
  the disposable copy only.
- To run master as an oracle, copy the whole tree (no symlinks: upstream setup
  writes into `thermo/` and `fastchem_vulcan/`), fix the cfg in the copy, and
  pin `PYTHONHASHSEED=0`.
- Never cite a hand-patched VULCAN-master copy for what upstream ships, or diff
  a disputed step count's config against one (notes "Oracle policy").
  `exoclime/VULCAN` now serves VULCAN 3: fetch from it only for VULCAN 3.
- Never write "upstream does X" from memory or a code comment: read the pinned
  source and cite the line (`op.py:NNN` / `build_atm.py:NNN` citations are the
  pinned revision, notes "Parity ledger"), or measure.
- Static-vs-dynamic assumptions are the highest-risk class. Refreshed in-loop:
  hydrostatic column, escape flux, `vm`, photolysis rates, condensation rates.
  Never refreshed: Kzz, Dzz, M, T, cross sections.
- A physics-neutral change keeps the step-count gates
  (`tests/test_step_count_gates.py`, full tier) on both `fast` and `reference`.
  HD189_vulcan3's count is not a gate: it moves with any bit-level change (notes
  "Stage repair and element budget"). A transport-kernel refactor also needs
  vulcan-forward's full-tier `test_queue_is_differentiable`.
- Keep the parity items (notes "Parity ledger"): step acceptance uses the
  import-bound `rtol` (C16); bottom pins apply before delta and clip (C17); the
  photo-frequency switch waits for `trun_min` / `count_min` (C18); the escape
  Jacobian term exists only under `use_vm_mol` (C15); every config keeps
  `batch_max_retries` 110 (C11).
- `conver_ignore`: parity configs ship `[]`, the V3 preset `['HC3N']`; never
  pick a list because it reproduces a step count.
- Network id columns are output: index photolysis by parser position (C14).
  `cfg.remove_list` is positional in both codes.
- Every config declares `fastchem_solar_abundance_file`; match it before quoting
  a cross-code number (unmatched, composition dominates). The Lodders 2009
  preset is not an agreement case with upstream's seed.
- Gravity derives from `Mp`/`Rp`; there is no `gs` knob. `vm` (upwind moldiff)
  is refreshed in-loop; the scheme matters only at low Kzz, so HD189 is not
  evidence for it.
- Earth and K2-18b are withdrawn; re-add one only with a run that certifies on
  main (notes "Withdrawn configs").

### Numerics and solver

- float64 always (`jax_enable_x64` at import).
- The production Jacobian is `chem.chem_jac_analytical`: a left-to-right sum
  along static tables (`chem._jac_gather_tables`), no scatter-add and no
  `reduce`, one summation order in every program. Never raise to the 0 power
  (`y**0` has a NaN tangent at y=0).
- A where-zeroed numerator over a guarded denominator (the delta and longdy
  maxima) takes denominator 1, else a sub-atol density gives `0 * inf` in the
  batched tangent.
- `pos_cut`/`nega_cut` clipping stays outside the JIT'd step and reads the
  post-solve `ymix`. Asymmetric three-body M: set the forward and reverse slots
  independently. The codegen RHS stays SymPy-faithful (no XLA rewrite of the
  multiply chains).
- `VULCAN3_SOLVER=fast` is the default, `ffi` the CUDA-kernel opt-in
  (production GH200), `reference` the parity pair; reverse mode through `ffi` is
  unsupported. Forward mode uses the matrix-free stage operator
  (`jax_ros2_step(matrix_free=True)`); a reverse-mode caller passes
  `matrix_free=False`, as `steady_state_grad` does. `VULCAN3_CHEM_RHS=grouped`
  stays opt-in until it is bitwise in full CPU runs too (notes "Efficiency
  program").
- The atom-conservation projection and the Ros2 stage repair
  (`jax_step._repair_stage`, C19) are default-on, not in VULCAN 2.0, and change
  the solution. `config.REPAIR_ABS_FLOOR`, `jax_step._DEFECT_FLOOR` and
  `jax_step._REPAIR_MAX_CELL_FRAC` are re-baseline knobs, not tuning ones; a
  relative floor or a dt gate is the wrong tool. `dt_max` is capped at
  `config.DT_MAX_S`; a larger value raises (notes "Stage repair and element
  budget").
- Benchmark a solver change on real blocks from `output/*.vul`, at the run's dt
  and at `dt_max=1e11`, never on synthetic ones; benchmark retrieval throughput
  at the production shape under `jvp` (notes "Solver conditioning", "Efficiency
  program"). A jit check takes fixture arrays as arguments, never closed over.
- Vectorize with vmap; `lax.scan` only for a true sequential dependency. The
  exception is the EQ seed (`custom_vmap` + `lax.map`, so a lane is bitwise
  independent of the batch width).

### Certificates and differentiation

- A run ends only on the certificate or a cap (`use_conv_stall` is refused).
- The certificate has two terms not in VULCAN 2.0, both inert below
  `count_min`. Geometry (C21): on a certificate candidate the runner refreshes
  the geometry whatever `update_frq` says and ends only if mu/g/Hp/dzi/Hpi moved
  less than `geom_conv_tol`. Element budget (C23): each element's
  operator-weighted column relative to H, accumulated per step on that step's
  own grid; never the absolute column, and `loss_eps` is no substitute. It
  assumes a closed column: an open one (boundary fluxes, bottom or fix_species
  pins) sets `element_budget_tol: .inf`. An unconverged exit logs the failing
  terms (`outer_loop.certificate_failures`); keep that list empty exactly when
  `conv_normal` holds.
- `vulcan3.conv_normal(final, cfg)` exports the whole certificate; a consumer
  calls it, never a copy (notes "Certificates"). Every shipped run converges on
  the loose branch: quote the realised `longdy` next to any "converged" claim.
- A forward-mode tangent is certified by `OuterLoop.run_jvp` (C22), not by
  `jax.jvp(runner)`; pass the direction in finite-difference-step units. Never
  a step floor on a differentiated map.
- Differentiate the inner `integ._runner`, never `OuterLoop.__call__`. Reverse
  mode at the converged state is `steady_state_reaction_sensitivity` (the
  renormalised map `steady_state_grad.SOLVER_MAP`, default `BODY_MAP_DT`); on
  photo columns pass
  `photo_recompute_k=make_photo_recompute_k(integ._photo_static, final)` (notes
  "Adjoint record"). Run `audit_adjoint_scope` before trusting a gradient on a
  non-default config; never weaken it to pass a state.
- T-profile gradients rebuild `k_arr` via `rates_jax.build_rate_array` and the
  structure via `atm_setup.mu_dz_g_jax` / `atm_refresh.update_mu_dz_jax` and
  `atm_jax._mol_diff`.
- A hybrid vm_mol result is a differentiable steady state only when
  `end_case == 1` and the terminal `hybrid_use_vm == 0`. Condensation (upstream
  `master_pin`) is not differentiable-through; the autodiff entry points refuse
  it (notes "Condensation contract (F1-F5)").
- Keep the full-tier `test_forward_jvp_integration` (Kzz JVP vs re-converged
  FD); a single-step or reverse-mode check does not replace it (notes "Forward
  mode").

### Interfaces and contracts

- `integ(rs)` (`OuterLoop.__call__`) takes a fresh `RunState` only.
  `integ(rs, chunk=N)` (the CLI's path) runs the same loop through
  `runner_chunk` and must stay bitwise equal to it; never give `_runner` a chunk
  carry leaf.
- A config file overlays `default.yaml`, and `default.yaml` is
  `default_config()`: a fix applied to the named configs only silently changes
  every API user's run. Add every key you remove from `default.yaml` to
  `config._REMOVED_KEYS` (refused with a remedy, also by
  `vulcan3.validate_overrides`). Master's display-only keys
  (`config._IGNORED_KEYS`) warn and are dropped from a file; an override dict
  refuses them.
- `make_config(...)` overrides need all three wirings: `_cfg_overlay` (setup),
  `OuterLoop(..., cfg=)` (runner), `Output(cfg=)` (output).
- Setup never writes a config knob (`state._cfg_overlay` discards the write).
  The runner builds any setting derived in setup, as `OuterLoop._build_statics`
  builds the `use_sat_surfaceH2O` bottom pin from `sat_mix` (notes "Code
  contracts not obvious from the modules").
- Import-frozen knobs (`config._FROZEN_ENV`: `network`, `com_file`,
  `atom_list`) and `$VULCAN3_SOLVER` / `$VULCAN3_CHEM_RHS` are set in the
  environment before the first import; the CLI re-execs once to match a config,
  the Python API does not.
- `OuterLoop.run_batch` (freeze-on-done) and `run_queue` (the same loop
  refilled from a job queue; tail compaction `queue_compact` off by default):
  per-profile arrays ride the `ProfileVars` carry, never a closure (vmap does
  not batch closure constants). Only T-P may vary across a photo batch; conden
  config and `pref_indx` are closure-baked. `run_queue` is bitwise `run_batch`
  when `n_lanes >= n_jobs`; a fresh lane takes photolysis exactly once, on its
  own fields. Compare a batched entry against vmap-of-solo, never the solo
  runner.
- Adding a field to `AtmStatic` touches three places in src (the NamedTuple,
  `make_atm_static`, `_ATM_STATIC_BATCH_AXES`) plus every direct `AtmStatic(`
  call in examples/ and tests/. Splice defaults into saved-fixture loaders
  rather than regenerating.
- Add no runtime dependency beyond jax / numpy / scipy / PyYAML and `exogibbs`
  (notes "Charter, configs, API").

### Data, caches and tests

- Vendored inputs are hash-pinned in `tests/science_sources.yaml`; re-pin in
  the same commit as any edit.
- conftest fails on missing `tests/data` fixtures (`python tests/gen_fixtures.py
  --all`). `VULCAN3_ALLOW_MISSING_FIXTURES` is the developer escape hatch:
  `tests.yml` sets it, `oracle.yml` must not. Check a CI full-tier failure on
  daw only after `gen_fixtures.py --all --force`.
- A process at an `XLA_PYTHON_CLIENT_MEM_FRACTION` other than 0.90 uses its own
  JAX compile-cache dir: the cache key ignores the fraction (notes
  "Operational").
- `.vul` is a pickle with master's three top-level keys and the same public
  keys, shapes and dtypes, so master's `plot_py/` loads it. Byte identity is a
  non-goal.
- A function earns a test if it is hot-path, a numerical bridge to master, a
  jit/vmap/grad boundary, or a parser: the bar is "could fail silently and
  matter". Usual bars: master equivalence at machine tolerance, vmap
  consistency ~1e-12, gradient vs FD ~1e-6.
- The normal tier stays under 10 min on daw; a test that takes minutes or
  several GB is marked `slow` (full tier). A workflow never deselects a test for
  its cost. Master-oracle tests run only with `$VULCAN_MASTER_DIR` set, in
  subprocesses against a disposable copy. Test-only reference kernels live in
  `tests/_oracles.py`.

### Style

- Magic numbers live in the config YAML; a literal in source gets a named
  constant with provenance.
- No dead code (`vulture --min-confidence 80` clean), but grep the pinned clone
  and the consumers before deleting a master-facing facade method: vulture does
  not see those callers (notes "Do not retry"). No legacy fallbacks, no
  phase-history comments.
- Fail fast and loud: an unmet precondition raises with the offending value and
  the remedy; a skipped check announces itself.
- Quote ratios, not ms pairs; name the host and code state for an ms pair, and
  believe a wall-time ratio only when `user+sys ~ real`.

## Where things live

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

Config keys and refusals: `configs/default.yaml`, `config.py`,
`runtime_validation.py`. Known defects in every repo: notes "Known bugs, all repos".
