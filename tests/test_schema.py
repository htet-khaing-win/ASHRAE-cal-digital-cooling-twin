import pandas as pd
import pytest

from cooling_twin.data.schema import BuildingTimeSeries, validate_schema


def _valid_ts(**overrides: object) -> BuildingTimeSeries:
    """A minimal, contract-compliant fixture. Tests override one field
    at a time to trigger exactly one violation per test.
    """
    idx = pd.date_range("2016-01-01", periods=48, freq="h", tz="America/New_York")
    base = dict(
        building_id="Fox_education_Theodore",
        timestamp=idx,
        chilledwater_kwh=pd.Series(100.0, index=idx),
        electricity_kwh=None,
        air_temp_c=pd.Series(20.0, index=idx),
        dew_temp_c=pd.Series(14.0, index=idx),
        wet_bulb_c=pd.Series(16.0, index=idx),
        humidity_ratio=pd.Series(0.01, index=idx),
        rh_pct=pd.Series(65.0, index=idx),
        floor_area_m2=5000.0,
        primary_use="Education",
        year_built=1998,
        site_id="Fox",
    )
    base.update(overrides)
    return BuildingTimeSeries(**base)  # type: ignore[arg-type]


def test_valid_fixture_passes() -> None:
    validate_schema(_valid_ts())  # should not raise


def test_schema_rejects_naive_timestamp() -> None:
    naive_idx = pd.date_range("2016-01-01", periods=48, freq="h")  # no tz
    with pytest.raises(ValueError, match="tz-aware"):
        validate_schema(_valid_ts(timestamp=naive_idx))


def test_schema_rejects_non_hourly_index() -> None:
    idx = pd.date_range("2016-01-01", periods=48, freq="h", tz="America/New_York")
    idx = idx.delete(5)  # punch a hole -> irregular spacing
    ts = _valid_ts(timestamp=idx, chilledwater_kwh=pd.Series(100.0, index=idx))
    with pytest.raises(ValueError, match="hourly"):
        validate_schema(ts)


def test_schema_rejects_rh_out_of_bounds() -> None:
    idx = pd.date_range("2016-01-01", periods=48, freq="h", tz="America/New_York")
    bad_rh = pd.Series(65.0, index=idx)
    bad_rh.iloc[10] = 142.0
    with pytest.raises(ValueError, match="INV-7"):
        validate_schema(_valid_ts(rh_pct=bad_rh))


def test_schema_rejects_dew_wet_dry_ordering_violation() -> None:
    idx = pd.date_range("2016-01-01", periods=48, freq="h", tz="America/New_York")
    # dew point above dry-bulb air temp is physically impossible
    bad_dew = pd.Series(14.0, index=idx)
    bad_dew.iloc[3] = 25.0  # exceeds air_temp_c = 20.0
    with pytest.raises(ValueError, match="INV-9"):
        validate_schema(_valid_ts(dew_temp_c=bad_dew))


def test_schema_rejects_nonpositive_floor_area() -> None:
    with pytest.raises(ValueError, match="floor_area_m2"):
        validate_schema(_valid_ts(floor_area_m2=0.0))


def test_schema_rejects_mismatched_series_length() -> None:
    idx = pd.date_range("2016-01-01", periods=48, freq="h", tz="America/New_York")
    short_series = pd.Series(100.0, index=idx[:40])  # 40, not 48
    with pytest.raises(ValueError, match="length"):
        validate_schema(_valid_ts(chilledwater_kwh=short_series))