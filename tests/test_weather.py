"""Unit tests for src/cooling_twin/data/weather.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cooling_twin import SEED
from cooling_twin.data.weather import (
    STANDARD_ATMOSPHERIC_PRESSURE_PA,
    _validate_inv9,
    add_psychrometric_features,
    cross_correlate_lag,
    join_weather,
    load_weather,
    validate_timezone_alignment,
)


def _write_weather_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2016-01-01", periods=4, freq="h")
    pd.DataFrame(
        {
            "site_id": ["Fox", "Fox", "Panther", "Panther"],
            "timestamp": list(idx[:2].strftime("%Y-%m-%d %H:%M:%S")) * 2,
            "airTemperature": [10.0, 12.0, 8.0, 9.0],
            "dewTemperature": [2.0, 3.0, 1.0, 1.5],
            "seaLvlPressure": [1013.0, 1013.0, 1010.0, 1010.0],
        }
    ).to_csv(path, index=False)


def test_load_weather_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="weather.csv"):
        load_weather(tmp_path)


def test_load_weather_reads_all_sites(tmp_path: Path) -> None:
    _write_weather_csv(tmp_path / "data" / "weather" / "weather.csv")

    df = load_weather(tmp_path)

    assert set(df["site_id"]) == {"Fox", "Panther"}
    assert len(df) == 4


def test_load_weather_filters_by_site_id(tmp_path: Path) -> None:
    _write_weather_csv(tmp_path / "data" / "weather" / "weather.csv")

    df = load_weather(tmp_path, site_id="Fox")

    assert set(df["site_id"]) == {"Fox"}
    assert len(df) == 2


def test_load_weather_unknown_site_id_raises(tmp_path: Path) -> None:
    _write_weather_csv(tmp_path / "data" / "weather" / "weather.csv")

    with pytest.raises(ValueError, match="No weather rows"):
        load_weather(tmp_path, site_id="Nonexistent")


def test_add_psychrometric_features_produces_physically_ordered_output() -> None:
    df = pd.DataFrame(
        {
            "airTemperature": [25.0, 30.0],
            "dewTemperature": [15.0, 20.0],
            "seaLvlPressure": [1013.25, 1013.25],
        }
    )

    result = add_psychrometric_features(df)

    assert {"rh_pct", "humidity_ratio", "wet_bulb_c", "enthalpy_j_per_kg"} <= set(result.columns)
    assert (result["rh_pct"] > 0).all() and (result["rh_pct"] <= 100).all()
    assert (result["dewTemperature"] <= result["wet_bulb_c"] + 1e-6).all()
    assert (result["wet_bulb_c"] <= result["airTemperature"] + 1e-6).all()


def test_add_psychrometric_features_missing_pressure_uses_standard_atmosphere() -> None:
    with_pressure = pd.DataFrame(
        {
            "airTemperature": [25.0],
            "dewTemperature": [15.0],
            "seaLvlPressure": [STANDARD_ATMOSPHERIC_PRESSURE_PA / 100.0],
        }
    )
    without_pressure = pd.DataFrame(
        {
            "airTemperature": [25.0],
            "dewTemperature": [15.0],
            "seaLvlPressure": [float("nan")],
        }
    )

    result_with = add_psychrometric_features(with_pressure)
    result_without = add_psychrometric_features(without_pressure)

    assert result_with["rh_pct"].iloc[0] == pytest.approx(result_without["rh_pct"].iloc[0])


def test_add_psychrometric_features_skips_nan_rows() -> None:
    df = pd.DataFrame(
        {
            "airTemperature": [25.0, float("nan")],
            "dewTemperature": [15.0, 20.0],
            "seaLvlPressure": [1013.25, 1013.25],
        }
    )

    result = add_psychrometric_features(df)

    assert pd.isna(result["rh_pct"].iloc[1])
    assert pd.isna(result["wet_bulb_c"].iloc[1])
    assert not pd.isna(result["rh_pct"].iloc[0])


def test_validate_inv9_passes_on_correctly_ordered_data() -> None:
    df = pd.DataFrame(
        {"dewTemperature": [10.0], "wet_bulb_c": [15.0], "airTemperature": [20.0]}
    )
    _validate_inv9(df)  # should not raise


def test_validate_inv9_raises_on_violation() -> None:
    # wet_bulb_c above airTemperature is physically impossible.
    df = pd.DataFrame(
        {"dewTemperature": [10.0], "wet_bulb_c": [25.0], "airTemperature": [20.0]}
    )
    with pytest.raises(ValueError, match="INV-9"):
        _validate_inv9(df)


def _building_df(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "building_id": "Fox_education_Theodore",
            "meter_reading": np.arange(len(timestamps), dtype=float),
        }
    )


def test_join_weather_inner_joins_on_timestamp() -> None:
    idx = pd.date_range("2016-01-01", periods=3, freq="h")
    building_df = _building_df(idx)
    weather_df = pd.DataFrame(
        {
            "site_id": ["Fox"] * 3,
            "timestamp": idx,
            "airTemperature": [10.0, 11.0, 12.0],
        }
    )

    merged = join_weather(building_df, weather_df, site_id="Fox")

    assert len(merged) == 3
    assert "site_id" not in merged.columns
    assert "airTemperature" in merged.columns


def test_join_weather_no_rows_for_site_raises() -> None:
    idx = pd.date_range("2016-01-01", periods=3, freq="h")
    building_df = _building_df(idx)
    weather_df = pd.DataFrame(
        {"site_id": ["Panther"] * 3, "timestamp": idx, "airTemperature": [10.0, 11.0, 12.0]}
    )

    with pytest.raises(ValueError, match="No weather rows"):
        join_weather(building_df, weather_df, site_id="Fox")


def test_join_weather_drops_unmatched_timestamps() -> None:
    building_idx = pd.date_range("2016-01-01", periods=5, freq="h")
    building_df = _building_df(building_idx)
    # Weather only covers the last 2 of the 5 building timestamps.
    weather_df = pd.DataFrame(
        {
            "site_id": ["Fox"] * 2,
            "timestamp": building_idx[-2:],
            "airTemperature": [10.0, 11.0],
        }
    )

    merged = join_weather(building_df, weather_df, site_id="Fox")

    assert len(merged) == 2


def _lagged_series(true_lag: int, n: int = 300) -> tuple[pd.Series, pd.Series]:
    """temp is a smooth random walk; load = temp shifted by true_lag hours,
    i.e. load(t) = temp(t - true_lag) -- load lags behind temp.
    """
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    temp = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx)
    load = temp.shift(true_lag)
    return load, temp


def test_cross_correlate_lag_recovers_known_lag() -> None:
    load, temp = _lagged_series(true_lag=3)

    result = cross_correlate_lag(load, temp, max_lag_hours=6)

    best_row = result.loc[result["correlation"].idxmax()]
    assert int(best_row["lag_hours"]) == 3
    assert best_row["correlation"] == pytest.approx(1.0, abs=1e-9)


def test_cross_correlate_lag_mismatched_index_raises() -> None:
    idx_a = pd.date_range("2016-01-01", periods=200, freq="h")
    idx_b = pd.date_range("2016-02-01", periods=200, freq="h")
    load = pd.Series(1.0, index=idx_a)
    temp = pd.Series(1.0, index=idx_b)

    with pytest.raises(ValueError, match="identical DatetimeIndex"):
        cross_correlate_lag(load, temp)


def test_cross_correlate_lag_insufficient_overlap_raises() -> None:
    idx = pd.date_range("2016-01-01", periods=50, freq="h")
    load = pd.Series(1.0, index=idx)
    temp = pd.Series(1.0, index=idx)

    with pytest.raises(ValueError, match="too little data"):
        cross_correlate_lag(load, temp, max_lag_hours=6)


def test_validate_timezone_alignment_passes_within_range() -> None:
    load, temp = _lagged_series(true_lag=2)

    best_lag = validate_timezone_alignment(load, temp, expected_lag_range_hours=(0, 6))

    assert best_lag == 2


def test_validate_timezone_alignment_raises_outside_range() -> None:
    load, temp = _lagged_series(true_lag=-5)  # load LEADS temp -- physically backwards

    with pytest.raises(ValueError, match="TIMEZONE ALIGNMENT CHECK FAILED"):
        validate_timezone_alignment(load, temp, expected_lag_range_hours=(0, 6))
