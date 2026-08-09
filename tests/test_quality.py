"""Unit tests for src/cooling_twin/data/quality.py -- C1, C2, C7."""
import numpy as np
import pandas as pd
import pytest

from cooling_twin import SEED
from cooling_twin.data.quality import (
    check_extreme_spikes,
    check_negative_values,
    check_relative_humidity_bounds,
    detect_change_points,
    detect_flatlines,
    drop_sparse_months,
    handle_gaps,
    load_cleaning_config,
    run_cleaning_pipeline,
)


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2016-01-01", periods=len(values), freq="h")
    return pd.Series(values, index=idx)


def test_c1_flags_negative_values():
    """Known-answer: exactly the negative entries are nulled and flagged."""
    s = _series([100.0, -5.0, 200.0, -0.1, 150.0])
    cleaned, flags = check_negative_values(s)

    assert cleaned.isna().sum() == 2
    assert pd.isna(cleaned.iloc[1])
    assert pd.isna(cleaned.iloc[3])
    assert cleaned.iloc[0] == 100.0  # untouched values survive unchanged
    assert len(flags) == 2
    assert all(f.rule == "C1" for f in flags)


def test_c1_empty_series_raises():
    with pytest.raises(ValueError, match="empty"):
        check_negative_values(pd.Series([], dtype=float))


def test_c2_flags_spike_above_threshold():
    """1000 normal points + one 50x spike; spike must be the only flag.

    n must be large relative to 1/(1 - percentile/100) = 1000 for the
    p99.9 threshold estimate to not be contaminated by the spike itself
    -- see the docstring note below for why n=100 fails this.
    """
    normal = [300.0 + (i % 7) for i in range(1000)]  # values ~300-306
    s = _series(normal + [15000.0])
    config = {"c2_spike": {"percentile": 99.9, "multiplier": 3.0}}

    cleaned, flags = check_extreme_spikes(s, config)

    assert len(flags) == 1
    assert flags[0].rule == "C2"
    assert pd.isna(cleaned.iloc[-1])
    assert cleaned.iloc[0] == 300.0


def test_c2_insufficient_data_raises():
    config = {"c2_spike": {"percentile": 99.9, "multiplier": 3.0}}
    with pytest.raises(ValueError, match="100 non-NaN"):
        check_extreme_spikes(_series([1.0, 2.0, 3.0]), config)


def test_c7_flags_rh_out_of_bounds():
    """Known-answer: RH of 105 and -3 violate [0, 100]; 50 and 0 and 100 don't."""
    s = _series([50.0, 105.0, -3.0, 0.0, 100.0])
    config = {"c7_physical_bounds": {"rh_pct_min": 0.0, "rh_pct_max": 100.0}}

    cleaned, flags = check_relative_humidity_bounds(s, config)

    assert len(flags) == 2
    assert pd.isna(cleaned.iloc[1])
    assert pd.isna(cleaned.iloc[2])
    assert cleaned.iloc[3] == 0.0    # boundary values are NOT violations
    assert cleaned.iloc[4] == 100.0  # boundary values are NOT violations


def test_c3_flags_nonzero_stuck_run():
    """A 8-hour identical nonzero run triggers C3, not C4."""
    values = [300.0, 305.0] + [250.0] * 8 + [310.0, 298.0]
    s = _series(values)
    config = {
        "c3_stuck_sensor": {"min_run_hours": 6},
        "c4_meter_offline": {"min_zero_run_hours": 24},
    }

    cleaned, flags = detect_flatlines(s, config)

    assert len(flags) == 1
    assert flags[0].rule == "C3"
    assert cleaned.iloc[2:10].isna().all()
    assert cleaned.iloc[0] == 300.0
    assert cleaned.iloc[-1] == 298.0


def test_c4_flags_zero_run_not_c3():
    """A 30-hour zero run triggers C4 only, even though it also exceeds
    the C3 threshold -- C4 is the more specific diagnosis and wins."""
    values = [300.0] + [0.0] * 30 + [295.0]
    s = _series(values)
    config = {
        "c3_stuck_sensor": {"min_run_hours": 6},
        "c4_meter_offline": {"min_zero_run_hours": 24},
    }

    cleaned, flags = detect_flatlines(s, config)

    assert len(flags) == 1
    assert flags[0].rule == "C4"
    assert cleaned.iloc[1:31].isna().all()


def test_short_zero_run_below_c4_but_above_c3_still_flags_c3():
    """A 10-hour zero run: too short for C4 (24h) but long enough for C3 (6h)."""
    values = [300.0] + [0.0] * 10 + [295.0]
    s = _series(values)
    config = {
        "c3_stuck_sensor": {"min_run_hours": 6},
        "c4_meter_offline": {"min_zero_run_hours": 24},
    }

    cleaned, flags = detect_flatlines(s, config)

    assert len(flags) == 1
    assert flags[0].rule == "C3"


def test_run_below_both_thresholds_not_flagged():
    """A 3-hour identical run is below both C3 (6h) and C4 (24h) -- clean."""
    values = [300.0, 250.0, 250.0, 250.0, 310.0]
    s = _series(values)
    config = {
        "c3_stuck_sensor": {"min_run_hours": 6},
        "c4_meter_offline": {"min_zero_run_hours": 24},
    }

    cleaned, flags = detect_flatlines(s, config)

    assert len(flags) == 0
    assert cleaned.equals(s)


def test_nan_run_is_ignored_not_flagged():
    """A run of NaN values is a gap (C5/C6 territory), not a flatline."""
    values = [300.0] + [float("nan")] * 10 + [295.0]
    s = _series(values)
    config = {
        "c3_stuck_sensor": {"min_run_hours": 6},
        "c4_meter_offline": {"min_zero_run_hours": 24},
    }

    cleaned, flags = detect_flatlines(s, config)

    assert len(flags) == 0


def test_flatlines_empty_series_raises():
    config = {
        "c3_stuck_sensor": {"min_run_hours": 6},
        "c4_meter_offline": {"min_zero_run_hours": 24},
    }
    with pytest.raises(ValueError, match="empty"):
        detect_flatlines(pd.Series([], dtype=float), config)

def test_short_gap_is_interpolated() -> None:
    idx = pd.date_range("2016-01-01", periods=6, freq="h")
    s = pd.Series([10.0, 12.0, np.nan, np.nan, 18.0, 20.0], index=idx)
    filled, log = handle_gaps(s, max_interpolate_hours=3)
    assert filled.isna().sum() == 0
    assert log.loc[0, "action"] == "interpolated"
    assert filled.iloc[2] == pytest.approx(14.0)  # linear between 12 and 18


def test_long_gap_is_left_as_nan() -> None:
    idx = pd.date_range("2016-01-01", periods=10, freq="h")
    values = [10.0, 12.0] + [np.nan] * 6 + [20.0, 22.0]
    s = pd.Series(values, index=idx)
    filled, log = handle_gaps(s, max_interpolate_hours=3)
    assert filled.iloc[2:8].isna().all()
    assert log.loc[0, "action"] == "left_as_nan"
    assert log.loc[0, "length_hours"] == 6


def test_leading_trailing_nan_not_extrapolated() -> None:
    idx = pd.date_range("2016-01-01", periods=5, freq="h")
    s = pd.Series([np.nan, np.nan, 12.0, 14.0, np.nan], index=idx)
    filled, _ = handle_gaps(s, max_interpolate_hours=3)
    assert filled.isna().tolist() == [True, True, False, False, True]


def test_handle_gaps_rejects_non_monotonic_index() -> None:
    idx = pd.DatetimeIndex(["2016-01-02", "2016-01-01"])
    s = pd.Series([1.0, 2.0], index=idx)
    with pytest.raises(ValueError, match="monotonic"):
        handle_gaps(s, max_interpolate_hours=3)

def test_detects_single_level_shift() -> None:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2016-01-01", periods=24 * 200, freq="h")
    values = np.concatenate(
        [
            100 + rng.normal(0, 3, 24 * 100),  # regime 1: mean ~100
            160 + rng.normal(0, 3, 24 * 100),  # regime 2: mean ~160
        ]
    )
    s = pd.Series(values, index=idx)

    segmented, change_points = detect_change_points(s, penalty=30.0, min_segment_days=60)

    assert len(change_points) == 1
    # detected shift should land within a few days of the true break at day 100
    true_break = idx[24 * 100]
    assert abs((change_points[0] - true_break).days) <= 5
    assert segmented["segment_id"].nunique() == 2


def test_no_shift_found_when_none_exists() -> None:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2016-01-01", periods=24 * 200, freq="h")
    s = pd.Series(100 + rng.normal(0, 3, len(idx)), index=idx)

    _, change_points = detect_change_points(s, penalty=30.0, min_segment_days=60)
    assert change_points == []  # no shift is a correct, unremarkable result


def test_change_points_rejects_non_monotonic_index() -> None:
    idx = pd.DatetimeIndex(["2016-01-02", "2016-01-01"])
    s = pd.Series([1.0, 2.0], index=idx)
    with pytest.raises(ValueError, match="monotonic"):
        detect_change_points(s, penalty=30.0, min_segment_days=60)

def _synthetic_meter_and_rh(
    idx: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[pd.Series, pd.Series]:
    """Shared fixture: a plausible meter series + a plausible RH series,
    same index -- avoids repeating this setup across the pipeline tests.
    """
    meter = pd.Series(100 + rng.normal(0, 5, len(idx)), index=idx)
    rh = pd.Series(60 + rng.normal(0, 5, len(idx)), index=idx).clip(0, 100)
    return meter, rh


def test_run_cleaning_pipeline_order_and_completeness() -> None:
    """Every rule group's key should appear in logs, in a fixed order,
    and the pipeline should not raise on a well-formed synthetic series.
    """
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2016-01-01", periods=24 * 400, freq="h")
    s, rh = _synthetic_meter_and_rh(idx, rng)
    s.iloc[500:503] = -10.0       # C1
    s.iloc[1000:1008] = s.iloc[999]  # C3
    s.iloc[2000:2005] = np.nan    # C5/C6

    config = load_cleaning_config()
    cleaned, logs = run_cleaning_pipeline(s, rh, config)

    assert list(logs.keys()) == [
        "C1_C2_C7_physical_bounds", "C3_C4_flatlines",
        "C5_C6_gaps", "C8_change_points", "C9_sparse_months",
    ]
    assert len(cleaned) == len(s)
    assert len(logs["C1_C2_C7_physical_bounds"]) == 3  # exactly the 3 negative C1 points


def test_run_cleaning_pipeline_is_deterministic() -> None:
    """Same input + same config -> byte-identical output.

    M3 gate artifact (06_ASSESSMENT.md checklist item, next_action item 4
    in 07_PROGRESS.md) -- required before M3 GATE can be marked closed.
    """
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2016-01-01", periods=24 * 400, freq="h")
    s, rh = _synthetic_meter_and_rh(idx, rng)
    config = load_cleaning_config()

    cleaned_1, logs_1 = run_cleaning_pipeline(s, rh, config)
    cleaned_2, logs_2 = run_cleaning_pipeline(s, rh, config)

    pd.testing.assert_series_equal(cleaned_1, cleaned_2)
    for key in logs_1:
        pd.testing.assert_frame_equal(
            logs_1[key].reset_index(drop=True), logs_2[key].reset_index(drop=True)
        )


def test_drop_sparse_months() -> None:
    idx = pd.date_range("2016-01-01", periods=24 * 60, freq="h")  # Jan + Feb 2016
    s = pd.Series(100.0, index=idx)
    jan_mask = idx.month == 1
    s.loc[jan_mask & (idx.day > 10)] = np.nan  # Jan: only first 10 days valid -> < 50%

    result, dropped_log = drop_sparse_months(s, month_min_valid_pct=50.0)
    assert result.loc[idx.month == 1].isna().all()
    assert not result.loc[idx.month == 2].isna().any()
    assert len(dropped_log) == 1