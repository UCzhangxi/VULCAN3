"""Every knob `runtime_validation` bound-checks must be declared in every
shipped config, and each check refuses its silently wrong values.

`_validate_numerical_bounds` checks what a config declares and skips the
rest, so the YAML is the only home of each value. That holds only while
every shipped config declares every checked knob and the validator has no
literal getattr defaults; both are pinned here.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "src" / "vulcan3" / "configs"
VALIDATION = ROOT / "src" / "vulcan3" / "runtime_validation.py"

# Read only when use_condense is on, so a config that leaves it off need not
# declare them. Everything else is unconditional.
_CONDITIONAL = {"start_conden_time", "stop_conden_time"}


def _validator_source() -> str:
    return VALIDATION.read_text()


def _checked_knobs() -> set[str]:
    """Knob names `runtime_validation` reads, from the source itself."""
    from vulcan3.runtime_validation import _COUNTERS, _FINITE_POSITIVE

    names = set(re.findall(r'_declared\(cfg,\s*"([A-Za-z_]+)"', _validator_source()))
    assert len(names) > 5, f"knob scrape found only {len(names)}; parser broke"
    return names | set(_COUNTERS) | set(_FINITE_POSITIVE)


def _config_files() -> list[Path]:
    files = sorted(CONFIGS.glob("*.yaml"))
    assert files, f"no shipped configs under {CONFIGS}"
    return files


@pytest.mark.parametrize("cfg_path", _config_files(), ids=lambda p: p.stem)
def test_every_shipped_config_declares_every_checked_knob(cfg_path):
    declared = set(yaml.safe_load(cfg_path.read_text()) or {})
    required = _checked_knobs() - _CONDITIONAL
    # dt_max is derived by config.py, never written in YAML.
    required.discard("dt_max")
    missing = sorted(required - declared)
    assert not missing, (
        f"{cfg_path.name} does not declare {missing}. runtime_validation "
        f"bound-checks these, and it SKIPS a knob a config does not declare "
        f"instead of substituting a literal -- so an undeclared knob silently "
        f"loses its check. Declare it in the YAML."
    )


def test_the_validator_carries_no_literal_config_defaults():
    """No `getattr(cfg, "knob", <number>)` may come back.

    Such a literal is a second copy of a YAML-owned value. If the two drift,
    this module validates a number the run never uses.
    """
    tree = ast.parse(_validator_source())
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3):
            continue
        target, name, default = node.args
        if not (isinstance(target, ast.Name) and target.id == "cfg"):
            continue
        if isinstance(default, ast.Constant) and isinstance(
                default.value, (int, float)) and not isinstance(
                default.value, bool):
            offenders.append(
                f"line {node.lineno}: getattr(cfg, "
                f"{getattr(name, 'value', '?')!r}, {default.value!r})")
    assert not offenders, (
        "numeric literal config defaults are back in runtime_validation.py:\n  "
        + "\n  ".join(offenders)
        + "\nUse _declared(cfg, key) and skip an undeclared knob instead.")


_VALID = dict(nz=10, P_b=1e9, P_t=1e-2, Mp=1.9e30, Rp=8e9, rtol=0.25,
              geom_conv_tol=1e-3, element_budget_tol=1e-2, dt_min=1e-14,
              dt_max=1e15, sl_angle=1.0, humidity=0.5, count_min=120,
              count_max=10000, update_frq=100, conv_step=500,
              batch_max_retries=110, ini_update_photo_frq=100,
              final_update_photo_frq=5)
_NAN = float("nan")


@pytest.mark.parametrize("bad", [
    dict(nz=2), dict(P_b=1e-3), dict(P_t=0.0), dict(Mp=_NAN), dict(Rp=-1.0),
    dict(rtol=float("inf")), dict(geom_conv_tol=0.0), dict(element_budget_tol=_NAN),
    dict(element_budget_tol=-float("inf")),
    dict(dt_min=1e16), dict(sl_angle=58.0), dict(humidity=50.0),
    *(dict(**{k: v}) for k in ("count_min", "batch_max_retries",
                               "ini_update_photo_frq", "final_update_photo_frq")
      for v in (0, 1.5, _NAN, 2**31)),
], ids=repr)
def test_silently_wrong_values_are_refused(bad):
    """Each value refused alone; the valid baseline passes clean."""
    from types import SimpleNamespace
    from vulcan3.runtime_validation import _validate_numerical_bounds

    assert _validate_numerical_bounds(SimpleNamespace(**_VALID)) == []
    # .inf is the budget term's documented off switch.
    off = SimpleNamespace(**{**_VALID, "element_budget_tol": float("inf")})
    assert _validate_numerical_bounds(off) == []
    assert len(_validate_numerical_bounds(SimpleNamespace(**{**_VALID, **bad}))) == 1
