# todo

Evidence for these items is in `notes.md` (this repo's, unless another repo is named).

1. [decision] Confirm the departures from VULCAN master whose provenance is not recorded, and decide whether batches B-D of the provenance audit are owed.
2. [decision] Reply to the external user's report (draft exists in the maintainer's mail); items 3-5 wait on the files and logs it asks for.
3. [bug?] The external user reports that Earth converges in VULCAN but not in vulcan3 with the same inputs.
4. [bug?] The external user reports that vulcan3 runs differ from run to run on their machine.
5. [unverified] The external user reports that K2-18b (H2O clouds) converges in < 4000 steps; re-add it once it converges on main.
6. [decision] Claim `vulcan3` on PyPI: the README installs put TestPyPI first, so a TestPyPI package can shadow its PyPI namesake.

## 1. Confirm the departures from VULCAN master

Evidence: notes "Differences from VULCAN master" (the items marked "provenance not
recorded"). vulcan-retrieval's todo checks them against the rule "keep VULCAN's
original convergence criteria"; all are live on main and in every production run:
- the atom-conservation projection of the chemistry RHS and Jacobian
  (`jax_step._project_chem_rhs` / `_project_chem_jac`, from ccc40f7);
- the Ros2 stage repair (`jax_step._repair_stage`) and `config.DT_MAX_S`;
- the certificate terms with no VULCAN 2 counterpart, `geom_conv_tol` and
  `element_budget_tol` (C21, C23);
- the FastChem solar abundance file swap (ccc40f7);
- the retry cap with force-accept (`batch_max_retries` 110).
The audit covered batch A only (e46234a..7eda1af, 60 commits); the later 328
commits (batches B-D) were not audited. vulcan-forward's `conv_normal` terms and
per-theta re-anchoring are the same question on the consumer side (vulcan-retrieval
todo, "Confirm"). Done when each item is confirmed or scheduled for removal, and
the B-D audit is either done or declined.

## 2. Reply to the external user

A reply is drafted in the maintainer's mail, corrected to the shipped state; its
time-zone and meeting-time placeholders are left for the maintainer. It asks for
`SNCHO_DMS_photo_network_2025.txt` and confirmation of the K2-18b atm file (item 5),
the Earth YAML, atm/BC files and commands (item 3), the exact P error text, and the
platform, jax version and two differing logs (item 4). Done when sent.

## 3. Earth: converges upstream, not here

Evidence: notes "Withdrawn configs" (2000 matched steps only; a full-length run
was never done) and "Do not retry" (the falsified cures). Waiting on the user's
Earth inputs (item 2). Then run both codes to `count_max`: upstream from a
disposable copy of the pinned clone (`.oracles/vulcan2_ncho/cfg_examples/vulcan_cfg_Earth.py`),
vulcan3 with the converted YAML, `JAX_PLATFORMS=cpu vulcan3 -c <earth.yaml>` from a
scratch directory. ~30-60 min each [E]; ask before running. Done when the cause is
a confirmed defect (a "Parity ledger" entry) or the two codes agree.

## 4. Run-to-run differences

Evidence: notes "Operational" and "Oracle policy": vulcan3 is bitwise
reproducible on daw CPU and GPU; upstream's trajectory depends on
`PYTHONHASHSEED`. Re-checked at `7d24973`: two `vulcan3 -c HD189` runs on one daw
core are bitwise equal in y, t, dt and longdy (1495 steps each). The user's tree
was 0.3.1. Waiting on platform, jax version and two differing logs from a run on
main (item 2). Done when reproduced and fixed, or shown to be their old tree or
upstream.

## 5. K2-18b re-add

Evidence: notes "Withdrawn configs". The user's YAML (`K2-18b-H2Ocloud.yaml`, in
the dated directory of their files at the workspace root) was written for the
0.3.1 tree: main refuses or drops 28 of its keys. Blocked on the user:
`thermo/SNCHO_DMS_photo_network_2025.txt` is not on daw (only
`SNCHO_DMS_photo_network_Tsai2024.txt` is vendored), and the atm file
`atm_K2-18b-Nep100X-apo-H2Oclouds.txt` must be confirmed. Then convert the YAML
(sectioned; keep `print_prog_num` and the plot keys), check what `remove_list
[891, 892]` selects by position (C14), add the network and atm file with hash pins
in `tests/science_sources.yaml`, and run `JAX_PLATFORMS=cpu vulcan3 -c K2-18b` on
daw (~0.13 s/step [E]: ~9 min to 4000 steps, ~45 min to the 20000 cap; ask
first). Done when end_case 1 with terminal hybrid phase 0 and the realised longdy
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
