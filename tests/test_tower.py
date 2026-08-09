"""Unit tests for the cooling tower outlet model
(src/cooling_twin/models/tower.py).

Test pattern 1 of 4 (05_ENGINEERING_STANDARDS.md SS3): a known-answer
test tying the function directly to 03_DOMAIN_REFERENCE.md's documented
condenser-water floors, INV-3 enforcement across several wet-bulb
values, and input-validation tests for the ValueError branch.
"""

from __future__ import annotations

import pytest

from cooling_twin.models.tower import cooling_tower_outlet


def test_cooling_tower_outlet_matches_documented_tropical_floor() -> None:
    """27degC wet bulb (03_DOMAIN_REFERENCE.md SS3's tropical mean) plus
    a 4K approach (SS1's optimisation-range upper end) should reproduce
    SS3's documented ~31degC tropical condenser water floor.
    """
    assert cooling_tower_outlet(t_wet_bulb_c=27.0, approach_k=4.0) == pytest.approx(31.0)


def test_cooling_tower_outlet_matches_documented_temperate_floor() -> None:
    """11degC wet bulb (within SS3's 8-13degC temperate range) plus a
    4K approach should reproduce SS3's documented ~15degC floor.
    """
    assert cooling_tower_outlet(t_wet_bulb_c=11.0, approach_k=4.0) == pytest.approx(15.0)


@pytest.mark.parametrize("t_wet_bulb_c", [-10.0, 0.0, 15.0, 27.0, 35.0])
@pytest.mark.parametrize("approach_k", [0.1, 2.5, 4.0, 10.0])
def test_cooling_tower_outlet_respects_inv3(t_wet_bulb_c: float, approach_k: float) -> None:
    """INV-3: T_condenser_water > T_wet_bulb, for any valid input."""
    t_cw = cooling_tower_outlet(t_wet_bulb_c, approach_k)
    assert t_cw > t_wet_bulb_c


def test_cooling_tower_outlet_rejects_zero_approach() -> None:
    with pytest.raises(ValueError, match="INV-3"):
        cooling_tower_outlet(t_wet_bulb_c=27.0, approach_k=0.0)


def test_cooling_tower_outlet_rejects_negative_approach() -> None:
    with pytest.raises(ValueError, match="INV-3"):
        cooling_tower_outlet(t_wet_bulb_c=27.0, approach_k=-1.0)
