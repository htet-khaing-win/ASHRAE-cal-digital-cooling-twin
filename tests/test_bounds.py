"""Tests for data-derived parameter bounds (calibration/bounds.py).

Known-answer throughout. The reference building is synthetic and its
truth is stated in one place:

    load(t) = CONSTANT_KW + SLOPE_KW_PER_K * max(t - SWITCH_ON_C, 0)

Below `SWITCH_ON_C` the weather contributes nothing, so the load is
exactly `CONSTANT_KW` and the cold-weather floor must recover it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cooling_twin.calibration.bounds import (
    INTERNAL_GAIN_BOUND_MULTIPLIER,
    cold_weather_floor_w_per_m2,
    internal_gain_upper_bound,
)

CONSTANT_KW = 100.0
SLOPE_KW_PER_K = 20.0
SWITCH_ON_C = 18.0
AREA_M2 = 1000.0


def _synthetic_year(
    constant_kw: float = CONSTANT_KW, hours: int = 8784
) -> tuple[pd.Series, pd.Series]:
    """A year of outdoor temperature and the load it produces."""
    index = pd.date_range("2016-01-01", periods=hours, freq="h", tz="UTC")
    position = np.arange(hours, dtype=float)
    t_out = pd.Series(
        SWITCH_ON_C
        + 12.0 * np.sin(2 * np.pi * position / (24 * 365))
        + 4.0 * np.sin(2 * np.pi * position / 24),
        index=index,
    )
    load = pd.Series(
        constant_kw + SLOPE_KW_PER_K * np.clip(t_out - SWITCH_ON_C, 0.0, None),
        index=index,
    )
    return load, t_out


def test_floor_recovers_a_known_constant_load() -> None:
    """100 kW over 1,000 m2 must come back as 100 W/m2."""
    load, t_out = _synthetic_year()

    assert cold_weather_floor_w_per_m2(load, t_out, AREA_M2) == pytest.approx(100.0, rel=1e-6)


def test_floor_scales_with_the_constant_not_the_weather_term() -> None:
    """Doubling the constant doubles the floor; the slope is irrelevant."""
    load_low, t_out = _synthetic_year(constant_kw=100.0)
    load_high, _ = _synthetic_year(constant_kw=200.0)

    low = cold_weather_floor_w_per_m2(load_low, t_out, AREA_M2)
    high = cold_weather_floor_w_per_m2(load_high, t_out, AREA_M2)

    assert high == pytest.approx(2.0 * low, rel=1e-6)


def test_floor_is_near_zero_for_a_purely_weather_driven_building() -> None:
    """The case where the constant-term diagnosis would be WRONG.

    A building with no constant load must report a floor near zero, so
    the derived bound collapses instead of inventing headroom.
    """
    load, t_out = _synthetic_year(constant_kw=0.0)

    assert cold_weather_floor_w_per_m2(load, t_out, AREA_M2) < 1.0


def test_floor_normalises_by_floor_area() -> None:
    load, t_out = _synthetic_year()

    small = cold_weather_floor_w_per_m2(load, t_out, floor_area_m2=500.0)
    large = cold_weather_floor_w_per_m2(load, t_out, floor_area_m2=2000.0)

    assert small == pytest.approx(4.0 * large, rel=1e-6)


def test_floor_ignores_thin_bins() -> None:
    """One freak cold hour must not set the bound.

    A single very cold hour at an absurdly low load lands in its own bin
    with 1 hour in it; the `min_bin_hours` cut must discard that bin
    rather than average it into the floor.
    """
    load, t_out = _synthetic_year()
    t_out.iloc[0] = -40.0
    load.iloc[0] = 1.0

    assert cold_weather_floor_w_per_m2(load, t_out, AREA_M2) == pytest.approx(100.0, rel=0.05)


def test_floor_rejects_a_series_too_short_to_bin() -> None:
    load, t_out = _synthetic_year(hours=200)

    with pytest.raises(ValueError, match="temperature bins"):
        cold_weather_floor_w_per_m2(load, t_out, AREA_M2)


def test_floor_rejects_non_positive_area() -> None:
    load, t_out = _synthetic_year()

    with pytest.raises(ValueError, match="floor_area_m2"):
        cold_weather_floor_w_per_m2(load, t_out, floor_area_m2=0.0)


def test_floor_rejects_non_overlapping_inputs() -> None:
    load, t_out = _synthetic_year()
    shifted = t_out.copy()
    shifted.index = shifted.index + pd.Timedelta(days=400)

    with pytest.raises(ValueError, match="do not overlap"):
        cold_weather_floor_w_per_m2(load, shifted, AREA_M2)


def test_bound_is_the_multiplier_times_the_floor() -> None:
    load, t_out = _synthetic_year()

    floor = cold_weather_floor_w_per_m2(load, t_out, AREA_M2)
    bound = internal_gain_upper_bound(load, t_out, AREA_M2)

    assert bound == pytest.approx(INTERNAL_GAIN_BOUND_MULTIPLIER * floor, rel=1e-9)


def test_bound_rejects_a_multiplier_that_would_forbid_the_measured_load() -> None:
    """At or below 1x, the bound excludes load the building demonstrably draws."""
    load, t_out = _synthetic_year()

    with pytest.raises(ValueError, match="multiplier must be > 1"):
        internal_gain_upper_bound(load, t_out, AREA_M2, multiplier=1.0)


def test_bound_derivation_is_identical_for_two_buildings_with_the_same_shape() -> None:
    """No per-building judgement: same data in, same bound out."""
    load, t_out = _synthetic_year()

    first = internal_gain_upper_bound(load, t_out, AREA_M2)
    second = internal_gain_upper_bound(load.copy(), t_out.copy(), AREA_M2)

    assert first == second
