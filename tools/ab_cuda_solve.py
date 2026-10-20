"""A/B of the ffi block-Thomas kernel between two VULCAN-JAX trees (old, new).

  capture: python ab_cuda_solve.py capture SYSTEMS.npz FIXTURE_TREE
           (CPU; run with the W39b network env; FIXTURE_TREE/tests/data holds
           adj_state_w39b.npz)
  run:     PYTHONPATH=<tree>/src python ab_cuda_solve.py run SYSTEMS.npz OUT.npz <tree> [cpu]
           (refuses to run unless vulcan3 is imported from <tree>)
  compare: python ab_cuda_solve.py compare OLD.npz NEW.npz

`run` tiles the captured stage-1 systems to AB_LANES lanes (default 792, the
production batch), adds a random in-range batch at nz 62 (production), factors and
solves on the CUDA device (one rhs, and a stack of three on shared factors),
and records sha256 digests of lu, perm and the solutions, plus the median
solve time. It also solves one block with a subnormal pivot (2^-1030) whose
exact solution is all ones: old GPU inf and nan expected, new GPU all 1.0. The CPU kernel
gives nan there because XLA's CPU runtime flushes subnormals to zero.
"""
import hashlib
import os
import sys
import time

import numpy as np

mode = sys.argv[1]

if mode == "capture":
    out, fixture_tree = sys.argv[2], sys.argv[3]
    import jax
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.join(fixture_tree, "tests"))
    import _helpers
    from pathlib import Path
    _helpers.ROOT = Path(fixture_tree)  # load_adj_state reads ROOT/tests/data
    from test_solver_fast import capture_stage1
    dts = (1e-2, 1e2, 3.8e4, 1e6, 1e11)
    cases = [[np.asarray(a) for a in capture_stage1("adj_state_w39b.npz", "W39b", dt)] for dt in dts]
    np.savez(out, dts=np.array(dts),
             **{k: np.stack([c[i] for c in cases]) for i, k in enumerate(("diag", "sup", "sub", "rhs"))})
    print("captured", {k: v.shape for k, v in np.load(out).items()})

elif mode == "run":
    systems, out, tree = sys.argv[2], sys.argv[3], os.path.realpath(sys.argv[4])
    platform = sys.argv[5] if len(sys.argv) > 5 else "cuda"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import vulcan3.solver_fast as fm
    print("solver_fast:", fm.__file__)
    assert os.path.realpath(fm.__file__).startswith(tree + os.sep), (fm.__file__, tree)
    dev, cpu = jax.devices(platform)[0], jax.devices("cpu")[0]
    lanes = int(os.environ.get("AB_LANES", "792"))
    s = np.load(systems)
    rng = np.random.default_rng(0)
    nz, ni = 62, s["diag"].shape[2]  # production nz; the W39b fixture has its own (150)
    batches = {
        "w39b": [np.resize(s[k], (lanes,) + s[k].shape[1:]) for k in ("diag", "sup", "sub", "rhs")],
        "random": [rng.standard_normal((lanes, nz, ni, ni)) + 10.0 * np.eye(ni),
                   rng.standard_normal((lanes, nz - 1, ni)), rng.standard_normal((lanes, nz - 1, ni)),
                   rng.standard_normal((lanes, nz, ni))],
    }
    factor = jax.jit(fm._ffi_factor)
    solve = jax.jit(fm._ffi_solve)
    solve3 = jax.jit(jax.vmap(fm._ffi_solve, in_axes=(None, None, None, None, 0)))
    res = {}
    for name, (d, su, sb, r) in batches.items():
        d, su, sb, r = (jax.device_put(jnp.asarray(a), dev) for a in (d, su, sb, r))
        lu, perm = factor(d, su, sb)
        x = solve(lu, perm, su, sb, r)
        xs = solve3(lu, perm, su, sb, jnp.stack([r, 2.0 * r, -r]))
        for k, v in (("lu", lu), ("perm", perm), ("x", x), ("xs", xs)):
            a = np.asarray(v)
            res[f"{name}/{k}"] = np.array(hashlib.sha256(a.tobytes()).hexdigest())
            res[f"{name}/{k}_nonfinite"] = np.array(int((~np.isfinite(a)).sum()) if a.dtype.kind == "f" else 0)
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            solve(lu, perm, su, sb, r).block_until_ready()
            times.append(time.perf_counter() - t0)
        res[f"{name}/solve_ms"] = np.array(1e3 * np.median(times))
    # one block, pivot 2^-1030 (its reciprocal overflows), rhs chosen so x = 1
    d = np.eye(5)[None]
    d[0, 2, 2] = 2.0**-1030
    r = np.ones((1, 5))
    r[0, 2] = 2.0**-1030
    sup = sub = np.zeros((0, 5))
    for tag, where in (("subnormal_dev", dev), ("subnormal_cpu", cpu)):
        a = [jax.device_put(jnp.asarray(v), where) for v in (d, sup, sub, r)]
        lu, perm = factor(*a[:3])
        res[tag] = np.asarray(solve(lu, perm, a[1], a[2], a[3]))
    np.savez(out, **res)
    print({k: v for k, v in res.items() if k.endswith(("_ms", "nonfinite")) or k.startswith("subnormal")})

elif mode == "compare":
    old, new = (dict(np.load(p)) for p in sys.argv[2:4])
    same = True
    for k in sorted(old):
        if k.endswith(("/lu", "/perm", "/x", "/xs")):
            eq = old[k] == new[k]
            same &= bool(eq)
            print(f"{k:14s} {'bitwise equal' if eq else 'DIFFERENT'}  "
                  f"non-finite old {int(old[k + '_nonfinite'])} new {int(new[k + '_nonfinite'])}")
    for k in sorted(old):
        if k.endswith("solve_ms"):
            print(f"{k:16s} old {float(old[k]):.3f} ms  new {float(new[k]):.3f} ms")
    print("subnormal pivot: old", old["subnormal_dev"].ravel(), "new", new["subnormal_dev"].ravel(),
          "CPU kernel", new["subnormal_cpu"].ravel())
    print("ALL BITWISE EQUAL" if same else "BITWISE DIFFERENCES FOUND")
