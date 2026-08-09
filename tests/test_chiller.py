"""Unit tests for the DOE-2 chiller performance curves
(src/cooling_twin/models/chiller.py).

Test pattern 1 of 4 (05_ENGINEERING_STANDARDS.md SS3): known-answer
tests for the two pure functions, plus input-validation tests for every
`ValueError` branch -- the same pattern used for RCParams' INV-5 checks
in L4.2/tests/test_rc.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from cooling_twin.models.chiller import ChillerCurves, biquadratic, chiller_power

# Same coefficients as chiller.py's __main__ demo -- see that file's
# comment for the 3-constraint derivation (EIRFPLR(1.0)=1.0, vertex in
# [0.6, 0.8], EIRFPLR(0.2)=2.0). Duplicated here deliberately: this
# module's job is to verify the SHAPE this specific triple produces,
# independent of whether the demo block happens to still construct it
# the same way.
_L5_2_EIR_FPLR: tuple[float, float, float] = (3.05, -6.05, 4.00)

# Flat curves: CAPFT == EIRFT == 1.0 everywhere, EIRFPLR == 1.0 everywhere.
# Isolates chiller_power()'s arithmetic from any particular curve shape --
# with these, power must equal q_load_kw / cop_ref exactly, for any load
# and any temperatures.
_FLAT: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_FLAT_PLR: tuple[float, float, float] = (1.0, 0.0, 0.0)


def _flat_curves(cop_ref: float = 5.0, q_ref_kw: float = 1000.0) -> ChillerCurves:
    return ChillerCurves(
        cap_ft=_FLAT,
        eir_ft=_FLAT,
        eir_fplr=_FLAT_PLR,
        q_ref_kw=q_ref_kw,
        cop_ref=cop_ref,
    )


def test_biquadratic_known_answer() -> None:
    """f(2, 3) with coeffs (1,2,3,4,5,6) hand-computed:
    1 + 2*2 + 3*2^2 + 4*3 + 5*3^2 + 6*2*3 = 1+4+12+12+45+36 = 110.
    """
    assert biquadratic(2.0, 3.0, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)) == pytest.approx(110.0)


def test_biquadratic_at_origin_returns_c0() -> None:
    assert biquadratic(0.0, 0.0, (7.0, 1.0, 1.0, 1.0, 1.0, 1.0)) == pytest.approx(7.0)


def test_chiller_curves_rejects_cop_above_ten() -> None:
    """INV-1: 0 < COP < 10."""
    with pytest.raises(ValueError, match="INV-1"):
        _flat_curves(cop_ref=15.0)


def test_chiller_curves_rejects_zero_cop() -> None:
    with pytest.raises(ValueError, match="INV-1"):
        _flat_curves(cop_ref=0.0)


def test_chiller_curves_rejects_non_positive_rated_capacity() -> None:
    with pytest.raises(ValueError, match="q_ref_kw"):
        _flat_curves(q_ref_kw=0.0)


def test_chiller_power_matches_cop_ref_with_flat_curves() -> None:
    """With every curve flat (== 1.0), power is just q_load / cop_ref,
    independent of load level or temperature -- known-answer check that
    the CAPFT/EIRFT/EIRFPLR combination reduces correctly.
    """
    curves = _flat_curves(cop_ref=5.0, q_ref_kw=1000.0)
    power_kw = chiller_power(250.0, t_chw_supply_c=6.7, t_cond_water_c=29.4, curves=curves)
    assert power_kw == pytest.approx(50.0)  # 250 / 5.0


def test_chiller_power_independent_of_temperature_when_curves_flat() -> None:
    curves = _flat_curves(cop_ref=5.0, q_ref_kw=1000.0)
    power_a = chiller_power(250.0, t_chw_supply_c=6.7, t_cond_water_c=29.4, curves=curves)
    power_b = chiller_power(250.0, t_chw_supply_c=9.0, t_cond_water_c=35.0, curves=curves)
    assert power_a == pytest.approx(power_b)


def test_chiller_power_rejects_negative_load() -> None:
    curves = _flat_curves()
    with pytest.raises(ValueError, match="q_load_kw"):
        chiller_power(-1.0, t_chw_supply_c=6.7, t_cond_water_c=29.4, curves=curves)


def test_chiller_power_rejects_capft_extrapolation_to_zero_capacity() -> None:
    """A CAPFT curve that crosses zero outside its fitted range must
    raise, not silently divide by zero / return a negative PLR.
    """
    collapsing_cap_ft = (1.0, -1.0, 0.0, 0.0, 0.0, 0.0)  # 1.0 - x, hits 0 at x=1
    curves = ChillerCurves(
        cap_ft=collapsing_cap_ft,
        eir_ft=_FLAT,
        eir_fplr=_FLAT_PLR,
        q_ref_kw=1000.0,
        cop_ref=5.0,
    )
    with pytest.raises(ValueError, match="CAPFT collapsed"):
        chiller_power(100.0, t_chw_supply_c=5.0, t_cond_water_c=0.0, curves=curves)


def test_chiller_power_rejects_plr_above_inv6_limit() -> None:
    """INV-6: PLR <= 1.05. 1200 kW load against a 1000 kW rated chiller
    (flat CAPFT=1) is PLR=1.2 -- must raise, not silently overload.
    """
    curves = _flat_curves(cop_ref=5.0, q_ref_kw=1000.0)
    with pytest.raises(ValueError, match="INV-6"):
        chiller_power(1200.0, t_chw_supply_c=6.7, t_cond_water_c=29.4, curves=curves)


def test_chiller_power_accepts_plr_at_inv6_boundary() -> None:
    """INV-6's bound is inclusive (PLR <= 1.05, not < 1.05) -- exactly
    1050 kW against a 1000 kW rated chiller must NOT raise.
    """
    curves = _flat_curves(cop_ref=5.0, q_ref_kw=1000.0)
    power_kw = chiller_power(1050.0, t_chw_supply_c=6.7, t_cond_water_c=29.4, curves=curves)
    assert power_kw == pytest.approx(1050.0 / 5.0)


def test_eir_fplr_curve_peaks_in_60_to_80_percent_band() -> None:
    """L5.2's curriculum-mandated verification method: plot EIRFPLR
    (equivalently, COP) against PLR and check it peaks at 60-80% and
    collapses below 25% (02_CURRICULUM.md L5.2). Encoded as a permanent
    test so a future coefficient change can't silently break the shape.
    """
    c0, c1, c2 = _L5_2_EIR_FPLR
    plr = np.linspace(0.0, 1.05, 1000)
    eir_fplr = c0 + c1 * plr + c2 * plr**2

    peak_efficiency_plr = plr[np.argmin(eir_fplr)]
    assert 0.6 <= peak_efficiency_plr <= 0.8

    below_25pct = eir_fplr[plr < 0.25]
    at_or_above_25pct = eir_fplr[plr >= 0.25]
    assert below_25pct.min() > at_or_above_25pct.max()


def test_eir_fplr_curve_stays_within_inv1_cop_bounds() -> None:
    """A curve that satisfies the shape check above could still dip low
    enough to push COP above 10 (INV-1) somewhere PLR sweeps through --
    verify that never happens for the cop_ref this project demonstrates
    with (6.0).
    """
    c0, c1, c2 = _L5_2_EIR_FPLR
    cop_ref = 6.0
    plr = np.linspace(0.0, 1.05, 1000)
    eir_fplr = c0 + c1 * plr + c2 * plr**2

    cop = cop_ref / eir_fplr
    assert (cop > 0).all()
    assert (cop < 10).all()


def test_chiller_power_rejects_load_above_inv4_limit() -> None:
    """INV-4: Q_load <= 1.1 * q_ref_kw, a separate absolute check from
    INV-6's curve-relative PLR bound. Uses a CAPFT curve reporting MORE
    than nameplate capacity (cap_f=1.15) so PLR stays comfortably under
    1.05 (not tripping INV-6) while Q_load still exceeds 110% of the
    1000 kW nameplate rating.
    """
    boosted_cap_ft = (1.15, 0.0, 0.0, 0.0, 0.0, 0.0)  # constant 1.15 everywhere
    curves = ChillerCurves(
        cap_ft=boosted_cap_ft,
        eir_ft=_FLAT,
        eir_fplr=_FLAT_PLR,
        q_ref_kw=1000.0,
        cop_ref=5.0,
    )
    # q_avail=1150kW, so PLR=1150/1150=1.0 (<=1.05, INV-6 satisfied),
    # but Q_load=1150kW > 1.1*1000kW=1100kW (INV-4 violated).
    with pytest.raises(ValueError, match="INV-4"):
        chiller_power(1150.0, t_chw_supply_c=6.7, t_cond_water_c=29.4, curves=curves)


def test_chiller_power_accepts_load_at_inv4_boundary() -> None:
    """INV-4's bound is inclusive (Q_load <= 1.1*q_ref_kw) -- exactly
    at the boundary must NOT raise.
    """
    boosted_cap_ft = (1.15, 0.0, 0.0, 0.0, 0.0, 0.0)
    curves = ChillerCurves(
        cap_ft=boosted_cap_ft,
        eir_ft=_FLAT,
        eir_fplr=_FLAT_PLR,
        q_ref_kw=1000.0,
        cop_ref=5.0,
    )
    # Q_load=1100kW is exactly 1.1*1000kW; PLR=1100/1150=0.9565 (<=1.05).
    power_kw = chiller_power(1100.0, t_chw_supply_c=6.7, t_cond_water_c=29.4, curves=curves)
    assert power_kw == pytest.approx(1100.0 / 5.0)
