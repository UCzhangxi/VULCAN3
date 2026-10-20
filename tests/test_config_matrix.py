"""Breadth coverage of config flags not exercised by the per-flag tests.

Each case targets one flag (or a small group), runs the relevant pre-loop
setup, and asserts something useful about the state. Per-case detail lives in
each test's docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import load_tpk_state, set_cfg



def test_T_cross_sp_path_finite_positive():
    """T-dependent absp cross-section path is finite and non-negative for
    every (sp, layer) pair.
    """
    import vulcan3.photo_setup as photo_setup

    data_var, data_atm, _ = load_tpk_state()
    import vulcan3.legacy_io as op

    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)

    set_cfg(T_cross_sp=["CO2", "H2O", "NH3"])
    static = photo_setup._build_photo_static_dense(data_var, data_atm)

    absp_T_cross = np.asarray(static.absp_T_cross)
    assert absp_T_cross.shape[0] == 3, (
        f"expected 3 T-dep species rows, got {absp_T_cross.shape}"
    )
    assert np.all(np.isfinite(absp_T_cross))
    assert np.all(absp_T_cross >= 0.0)
    assert np.any(absp_T_cross > 0.0), "T-dep cross-sections all zero"


def test_use_vm_mol_populates_vm():
    """``use_vm_mol=True`` writes finite, non-zero advective velocity into
    ``atm.vm``.
    """
    import vulcan3.legacy_io as op
    from vulcan3.ini_abun import InitialAbun

    set_cfg(use_vm_mol=True)
    data_var, data_atm, make_atm = load_tpk_state()
    # f_mu_dz needs ymix; populate via const_mix to avoid the EQ seed.
    set_cfg(ini_mix="const_mix", const_mix={"H2": 0.9, "He": 0.0838, "H2O": 1e-3})
    data_var = InitialAbun().ini_y(data_var, data_atm)
    data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
    make_atm.mol_diff(data_atm)

    vm = np.asarray(data_atm.vm)
    # vm is the interface-centered drift velocity: one entry per cell interface.
    assert vm.shape == (data_var.y.shape[0] - 1, data_var.y.shape[1])
    assert np.all(np.isfinite(vm))
    # use_vm_mol should produce nonzero advective velocity for at least
    # one (layer, species) pair given non-isothermal HD189 Tco.
    assert np.any(np.abs(vm) > 0.0), "vm all-zero with use_vm_mol=True"


@pytest.mark.parametrize(
    "flag,file_attr,file_path,target_sp",
    [
        (
            "use_topflux",
            "top_BC_flux_file",
            "atm/BC_top_Jupiter.txt",
            "H",
        ),
        (
            "use_botflux",
            "bot_BC_flux_file",
            "atm/BC_bot_Earth.txt",
            "CO",
        ),
    ],
)
def test_bc_flux_loaded_from_file(flag, file_attr, file_path, target_sp):
    """``use_topflux=True`` / ``use_botflux=True`` populates
    ``atm.top_flux`` / ``atm.bot_flux`` from the matching cfg file.
    Verifies non-zero flux for at least one network species.
    """
    import vulcan3.composition as composition

    species_list = list(composition.species)
    set_cfg(**{flag: True, file_attr: file_path})
    _, data_atm, make_atm = load_tpk_state()
    make_atm.BC_flux(data_atm)

    arr_name = "top_flux" if flag == "use_topflux" else "bot_flux"
    arr = np.asarray(getattr(data_atm, arr_name))
    assert arr.shape == (len(species_list),)
    idx = species_list.index(target_sp)
    assert arr[idx] != 0.0, (
        f"expected nonzero {arr_name} for {target_sp} from {file_path}"
    )
    assert np.any(arr != 0.0)
