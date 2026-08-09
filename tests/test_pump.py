"""Unit tests for the pump affinity-law model
(src/cooling_twin/models/pump.py).

Test pattern 1 of 4 (05_ENGINEERING_STANDARDS.md SS3): known-answer
tests, including a direct encoding of L1.4/03_DOMAIN_REFERENCE.md's
"2x flow = 8x power" rule of thumb, plus input-validation tests for
every ValueError branch.
"""

from __future__ import annotations

import pytest

from cooling_twin.models.pump import PumpParams, pump_power


def _params(flow_rated: float = 1.0, power_rated: float = 100.0) -> PumpParams:
    return PumpParams(flow_rated_m3_per_s=flow_rated, power_rated_kw=power_rated)


def test_pump_params_rejects_non_positive_flow() -> None:
    with pytest.raises(ValueError, match="flow_rated_m3_per_s"):
        _params(flow_rated=0.0)


def test_pump_params_rejects_non_positive_power() -> None:
    with pytest.raises(ValueError, match="power_rated_kw"):
        _params(power_rated=-1.0)


def test_pump_power_variable_speed_halving_flow_gives_eighth_power() -> None:
    """L1.4 / 03_DOMAIN_REFERENCE.md SS1: 'Pump power ~ flow^3, flow 2x
    = power 8x' -- checked in both directions from the same rated point.
    """
    params = _params(flow_rated=1.0, power_rated=100.0)
    power_full = pump_power(1.0, params, variable_speed=True)
    power_half = pump_power(0.5, params, variable_speed=True)
    assert power_full == pytest.approx(100.0)
    assert power_half == pytest.approx(12.5)
    assert power_full / power_half == pytest.approx(8.0)


def test_pump_power_variable_speed_matches_rated_at_rated_flow() -> None:
    params = _params(flow_rated=2.0, power_rated=50.0)
    assert pump_power(2.0, params, variable_speed=True) == pytest.approx(50.0)


def test_pump_power_fixed_speed_is_flow_independent() -> None:
    """A fixed-speed (valve-throttled) pump draws its rated power at
    ANY valid flow -- that's the whole reason VFDs save energy.
    """
    params = _params(flow_rated=1.0, power_rated=100.0)
    power_full = pump_power(1.0, params, variable_speed=False)
    power_half = pump_power(0.5, params, variable_speed=False)
    power_low = pump_power(0.1, params, variable_speed=False)
    assert power_full == power_half == power_low == pytest.approx(100.0)


def test_pump_power_rejects_negative_flow() -> None:
    params = _params()
    with pytest.raises(ValueError, match="flow_m3_per_s"):
        pump_power(-0.1, params, variable_speed=True)


def test_pump_power_rejects_flow_above_rated() -> None:
    params = _params(flow_rated=1.0)
    with pytest.raises(ValueError, match="exceeds"):
        pump_power(1.5, params, variable_speed=True)


def test_pump_power_rejects_flow_above_rated_even_at_fixed_speed() -> None:
    """The rated-flow ceiling is a physical limit on the pump/motor,
    not an artifact of the variable-speed branch -- must be enforced
    the same way regardless of `variable_speed`.
    """
    params = _params(flow_rated=1.0)
    with pytest.raises(ValueError, match="exceeds"):
        pump_power(1.5, params, variable_speed=False)


def test_pump_power_zero_flow_variable_speed_is_zero() -> None:
    params = _params(flow_rated=1.0, power_rated=100.0)
    assert pump_power(0.0, params, variable_speed=True) == pytest.approx(0.0)
