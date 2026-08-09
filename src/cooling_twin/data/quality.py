"""Data quality rules (C1-C9) per 04_DATA_CONTRACT.md SS4.

This module implements physical bound checking first (C1, C2, C7) because
domain-bound violations are unambiguous: unlike a statistical outlier,
a negative energy reading or an out-of-range relative humidity is wrong
regardless of what the rest of the series looks like.

Rules C3-C6, C8, C9 (flatlines, gaps, change points) are added in later
lessons as the module grows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CLEANING_CONFIG_PATH = Path("config/cleaning.yaml")


@dataclass(frozen=True)
class CleaningFlag:
    """A single record of a value removed by a cleaning rule.

    Attributes:
        rule: The rule ID that triggered (e.g. "C1", "C2", "C7").
        timestamp: When the flagged reading occurred.
        original_value: The value before it was set to NaN.
        reason: Human-readable explanation, included in the L3.7 report.
    """

    rule: str
    timestamp: pd.Timestamp
    original_value: float
    reason: str


def load_cleaning_config(path: Path = DEFAULT_CLEANING_CONFIG_PATH) -> dict:
    """Load cleaning thresholds from config/cleaning.yaml.

    Args:
        path: Path to the cleaning config YAML file.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaning config not found at {path}. "
            "This file must exist before any C-rule can run -- thresholds "
            "are never hardcoded (05_ENGINEERING_STANDARDS.md SS2)."
        )
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"Cleaning config at {path} must parse to a YAML mapping, got {type(config).__name__}"
        )
    return config


def check_negative_values(series: pd.Series) -> tuple[pd.Series, list[CleaningFlag]]:
    """C1: flag and null out negative meter readings.

    Args:
        series: Meter reading series (e.g. chilled water kWh), tz-aware
            DatetimeIndex.

    Returns:
        Tuple of (cleaned series with negatives set to NaN, list of flags).

    Raises:
        ValueError: If series is empty.
    """
    if series.empty:
        raise ValueError("check_negative_values: series is empty.")

    negative_mask = series < 0
    flags = [
        CleaningFlag(
            rule="C1",
            timestamp=ts,
            original_value=float(val),
            reason="negative energy reading -- physically impossible",
        )
        for ts, val in series[negative_mask].items()
    ]

    cleaned = series.copy()
    cleaned[negative_mask] = float("nan")

    if flags:
        logger.info("C1: flagged %d negative reading(s)", len(flags))

    return cleaned, flags


def check_extreme_spikes(
    series: pd.Series, config: dict
) -> tuple[pd.Series, list[CleaningFlag]]:
    """C2: flag and null out values exceeding a multiple of the p99.9 percentile.

    Threshold is relative to THIS series' own distribution, not an
    absolute value -- a spike is defined by how far it departs from a
    building's own typical range, not a fixed kWh number that would be
    wrong for a building of different size.

    Args:
        series: Meter reading series.
        config: Parsed cleaning.yaml; must contain
            ``c2_spike.percentile`` and ``c2_spike.multiplier``.

    Returns:
        Tuple of (cleaned series with spikes set to NaN, list of flags).

    Raises:
        ValueError: If series is empty, or if fewer than 100 non-NaN
            points exist (percentile estimate would be unreliable).
    """
    if series.empty:
        raise ValueError("check_extreme_spikes: series is empty.")
    if series.notna().sum() < 100:
        raise ValueError(
            "check_extreme_spikes: fewer than 100 non-NaN points -- "
            "the p99.9 percentile estimate would not be reliable."
        )

    percentile = config["c2_spike"]["percentile"]
    multiplier = config["c2_spike"]["multiplier"]

    threshold = series.quantile(percentile / 100.0) * multiplier
    spike_mask = series > threshold

    flags = [
        CleaningFlag(
            rule="C2",
            timestamp=ts,
            original_value=float(val),
            reason=(
                f"value {val:.1f} exceeds p{percentile} * {multiplier} "
                f"= {threshold:.1f}"
            ),
        )
        for ts, val in series[spike_mask].items()
    ]

    cleaned = series.copy()
    cleaned[spike_mask] = float("nan")

    if flags:
        logger.info(
            "C2: flagged %d spike(s) above threshold %.1f", len(flags), threshold
        )

    return cleaned, flags


def check_relative_humidity_bounds(
    rh_pct: pd.Series, config: dict
) -> tuple[pd.Series, list[CleaningFlag]]:
    """C7 (INV-7): flag and null out relative humidity outside [0, 100].

    Args:
        rh_pct: Relative humidity series, percent, from
            add_psychrometric_features() (L2.3).
        config: Parsed cleaning.yaml; must contain
            ``c7_physical_bounds.rh_pct_min`` / ``rh_pct_max``.

    Returns:
        Tuple of (cleaned series with violations set to NaN, list of flags).

    Raises:
        ValueError: If rh_pct is empty.
    """
    if rh_pct.empty:
        raise ValueError("check_relative_humidity_bounds: series is empty.")

    lo = config["c7_physical_bounds"]["rh_pct_min"]
    hi = config["c7_physical_bounds"]["rh_pct_max"]

    violation_mask = (rh_pct < lo) | (rh_pct > hi)
    flags = [
        CleaningFlag(
            rule="C7",
            timestamp=ts,
            original_value=float(val),
            reason=f"RH {val:.1f}% outside physical bound [{lo}, {hi}] (INV-7)",
        )
        for ts, val in rh_pct[violation_mask].items()
    ]

    cleaned = rh_pct.copy()
    cleaned[violation_mask] = float("nan")

    if flags:
        logger.warning("C7 / INV-7: flagged %d RH violation(s)", len(flags))

    return cleaned, flags


def validate_physical_bounds(
    meter_kwh: pd.Series,
    rh_pct: pd.Series,
    config: dict | None = None,
    config_path: Path = DEFAULT_CLEANING_CONFIG_PATH,
) -> tuple[pd.Series, pd.Series, list[CleaningFlag]]:
    """Run C1, C2, C7 together -- the physical-bound gate for raw meter data.

    Args:
        meter_kwh: Meter reading series (chilled water or electricity).
        rh_pct: Relative humidity series, same index as meter_kwh.
        config: Pre-loaded cleaning config. If None, loaded from
            config_path.
        config_path: Used only if config is None.

    Returns:
        Tuple of (cleaned meter_kwh, cleaned rh_pct, combined list of
        CleaningFlag from all three rules, in rule order C1 -> C2 -> C7).
    """
    if config is None:
        config = load_cleaning_config(config_path)

    meter_after_c1, flags_c1 = check_negative_values(meter_kwh)
    meter_after_c2, flags_c2 = check_extreme_spikes(meter_after_c1, config)
    rh_after_c7, flags_c7 = check_relative_humidity_bounds(rh_pct, config)

    all_flags = flags_c1 + flags_c2 + flags_c7
    return meter_after_c2, rh_after_c7, all_flags

def _find_runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Find (start_idx, length) for every maximal run of True in mask.

    Args:
        mask: Boolean series, positional (integer-indexed after reset).

    Returns:
        List of (start_position, run_length) tuples, one per maximal
        contiguous run of True values. Empty list if mask is all False.
    """
    if not mask.any():
        return []

    values = mask.to_numpy()
    change_points = [0] + list(
        (values[1:] != values[:-1]).nonzero()[0] + 1
    ) + [len(values)]

    runs = []
    for start, end in zip(change_points[:-1], change_points[1:], strict=True):
        if values[start]:
            runs.append((start, end - start))
    return runs


def detect_flatlines(
    series: pd.Series, config: dict
) -> tuple[pd.Series, list[CleaningFlag]]:
    """C3 + C4: detect stuck sensors and offline meters via flatline runs.

    C3 flags any run of >= min_run_hours identical values, regardless of
    value. C4 flags runs of >= min_zero_run_hours that are specifically
    zero. A run can only trigger one rule -- if a zero-valued run is long
    enough to trigger both (e.g. 30 hours of zeros), it is reported as C4
    only, since "meter offline" is the more specific and more useful
    diagnosis than the generic "stuck" label.

    Args:
        series: Meter reading series, tz-aware DatetimeIndex, no gaps
            (regular hourly frequency expected).
        config: Parsed cleaning.yaml; must contain
            ``c3_stuck_sensor.min_run_hours`` and
            ``c4_meter_offline.min_zero_run_hours``.

    Returns:
        Tuple of (cleaned series with flagged runs set to NaN, list of
        CleaningFlag -- one flag per flagged run, not per point, so a
        142-hour stuck run produces exactly one flag with the run's
        start/end recorded in the reason string).

    Raises:
        ValueError: If series is empty.
    """
    if series.empty:
        raise ValueError("detect_flatlines: series is empty.")

    min_run_c3 = config["c3_stuck_sensor"]["min_run_hours"]
    min_run_c4 = config["c4_meter_offline"]["min_zero_run_hours"]

    is_same_as_prev = series.eq(series.shift())
    is_same_as_prev.iloc[0] = False  # first point can never start a "same as prev" run
    run_id = (~is_same_as_prev).cumsum()

    positions = pd.RangeIndex(len(series))
    grouped = positions.to_series(index=series.index).groupby(run_id.values)

    cleaned = series.copy()
    flags: list[CleaningFlag] = []

    for _, group_positions in grouped:
        run_len = len(group_positions)
        run_value = series.iloc[group_positions.iloc[0]]

        if pd.isna(run_value):
            continue  # NaN runs are gaps (C5/C6), not flatlines

        is_zero_run = run_value == 0
        triggers_c4 = is_zero_run and run_len >= min_run_c4
        triggers_c3 = run_len >= min_run_c3

        if not triggers_c3 and not triggers_c4:
            continue

        start_ts = series.index[group_positions.iloc[0]]
        end_ts = series.index[group_positions.iloc[-1]]
        rule = "C4" if triggers_c4 else "C3"
        reason = (
            f"{rule}: {run_len}h identical run"
            f"{' at zero (meter offline)' if triggers_c4 else ' (stuck sensor)'}"
            f", value={run_value}, {start_ts} to {end_ts}"
        )

        flags.append(
            CleaningFlag(
                rule=rule, timestamp=start_ts, original_value=float(run_value), reason=reason
            )
        )
        cleaned.iloc[group_positions] = float("nan")

    if flags:
        c3_count = sum(1 for f in flags if f.rule == "C3")
        c4_count = sum(1 for f in flags if f.rule == "C4")
        logger.info(
            "Flatline detection: %d C3 run(s), %d C4 run(s) flagged", c3_count, c4_count
        )

    return cleaned, flags

def handle_gaps(
    series: pd.Series,
    max_interpolate_hours: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Fill short gaps by linear interpolation; leave long gaps as NaN.

    Implements C5 (gap <= max_interpolate_hours: linear interpolate) and
    C6 (gap > max_interpolate_hours: leave as NaN -- do not fabricate).

    Args:
        series: Hourly series with a monotonic DatetimeIndex. May already
            contain NaN from C1-C4, or from genuinely missing readings.
        max_interpolate_hours: Threshold in hours (config/cleaning.yaml,
            gap_handling.max_interpolate_hours). Gaps at or under this
            length are interpolated; gaps longer than this are left NaN.

    Returns:
        A tuple of:
            filled:  the series with short gaps interpolated, long gaps
                     (and any leading/trailing NaN) still NaN.
            gap_log: one row per gap found, with start, end, length_hours,
                     and the action taken. Feeds the L3.7 quality report.

    Raises:
        ValueError: If the index is not a monotonic DatetimeIndex.
    """
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_monotonic_increasing:
        raise ValueError("series must have a monotonic DatetimeIndex")

    is_gap = series.isna()
    gap_id = (is_gap != is_gap.shift()).cumsum()

    # Interpolate every interior gap first (never extrapolates past the
    # first/last valid point -- limit_area="inside" handles that).
    filled = series.interpolate(method="linear", limit_area="inside")

    records: list[dict[str, object]] = []
    for _, group in series.groupby(gap_id):
        if not is_gap.loc[group.index].iloc[0]:
            continue  # a run of valid values, not a gap

        gap_len = len(group)
        start, end = group.index[0], group.index[-1]

        if gap_len <= max_interpolate_hours:
            action = "interpolated"
        else:
            action = "left_as_nan"
            filled.loc[group.index] = np.nan  # revert: C6, refuse to fabricate

        records.append(
            {"start": start, "end": end, "length_hours": gap_len, "action": action}
        )

    gap_log = pd.DataFrame.from_records(
        records, columns=["start", "end", "length_hours", "action"]
    )

    n_long = int((gap_log["action"] == "left_as_nan").sum()) if not gap_log.empty else 0
    if n_long:
        logger.warning(
            "handle_gaps: %d gap(s) exceed %dh and were left as NaN, not interpolated",
            n_long,
            max_interpolate_hours,
        )

    return filled, gap_log

def detect_change_points(
    series: pd.Series,
    penalty: float,
    min_segment_days: int,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Detect level shifts and segment the series around them.

    Implements C8: identifies structural breaks (meter replacement,
    tenant change, retrofit) so calibration can be scoped to a single
    physical regime instead of averaging across a break that no single
    RC parameter set can describe.

    Detection runs on a daily-mean aggregate (not raw hourly values) to
    avoid the diurnal cycle being mistaken for a level shift, using
    PELT (Pruned Exact Linear Time) with an L2 cost function -- this
    finds an unknown number of breakpoints without requiring you to
    specify how many shifts to look for in advance.

    Args:
        series: Hourly series, tz-aware DatetimeIndex, may contain NaN
            (from C1-C6). NaN days are excluded before detection.
        penalty: PELT penalty term (config/cleaning.yaml:
            change_point_detection.penalty). Higher = fewer, larger
            shifts detected; lower = more, smaller shifts.
        min_segment_days: Minimum days between change points
            (config/cleaning.yaml: change_point_detection.min_segment_days).
            Prevents PELT from carving out implausibly short regimes.

    Returns:
        A tuple of:
            segmented: DataFrame indexed like `series`, with the
                original value and an integer `segment_id` column.
            change_points: Timestamps marking the start of each new
                segment. Empty list means no level shift was detected --
                this is a normal, good outcome, not a failure.

    Raises:
        ValueError: If the index is not a monotonic DatetimeIndex.
    """
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_monotonic_increasing:
        raise ValueError("series must have a monotonic DatetimeIndex")

    daily = series.resample("D").mean().dropna()

    if len(daily) < 2 * min_segment_days:
        logger.info("detect_change_points: series too short to segment meaningfully")
        segmented = pd.DataFrame({"value": series, "segment_id": 0})
        return segmented, []

    signal = daily.to_numpy().reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=min_segment_days, jump=1).fit(signal)
    breakpoints = algo.predict(pen=penalty)[:-1]  # drop the trailing len(signal) marker

    change_points = [daily.index[bp] for bp in breakpoints]

    segment_id = pd.Series(0, index=series.index, dtype=int)
    for i, cp in enumerate(change_points, start=1):
        segment_id.loc[cp:] = i

    segmented = pd.DataFrame({"value": series, "segment_id": segment_id})

    if change_points:
        logger.warning(
            "detect_change_points: %d level shift(s) detected at %s -- "
            "review against weather record before treating as confirmed",
            len(change_points),
            [str(cp.date()) for cp in change_points],
        )

    return segmented, change_points

def drop_sparse_months(
    series: pd.Series,
    month_min_valid_pct: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Drop any calendar month with less than `month_min_valid_pct` valid data.

    Implements C9. Runs after all other cleaning, since a month's validity
    should reflect the final, fully-cleaned state of the series.

    Args:
        series: Cleaned hourly series (post C1-C8).
        month_min_valid_pct: Threshold from config/cleaning.yaml
            (reporting.month_min_valid_pct).

    Returns:
        (result, dropped_log): result has entire low-validity months set
        to NaN; dropped_log has one row per dropped month with its
        valid-data percentage.
    """
    result = series.copy()
    monthly_valid_pct = series.notna().groupby(series.index.to_period("M")).mean() * 100
    dropped = monthly_valid_pct[monthly_valid_pct < month_min_valid_pct]

    for period in dropped.index:
        mask = series.index.to_period("M") == period
        result.loc[mask] = np.nan

    dropped_log = dropped.rename("valid_pct").reset_index()
    dropped_log.columns = ["month", "valid_pct"]

    if not dropped.empty:
        logger.warning(
            "drop_sparse_months: %d month(s) dropped for < %.0f%% valid data",
            len(dropped), month_min_valid_pct,
        )
    return result, dropped_log


def _flags_to_log(flags: list[CleaningFlag]) -> pd.DataFrame:
    """Convert a list of CleaningFlag into the DataFrame shape every other
    rule-group log already has (handle_gaps()'s gap_log,
    drop_sparse_months()'s dropped_log).

    check_negative_values/check_extreme_spikes/check_relative_humidity_bounds
    (bundled by validate_physical_bounds) and detect_flatlines all return
    list[CleaningFlag], not a DataFrame -- without this conversion,
    generate_quality_report.py's _rule_summary_row() would crash on
    log.empty / log.columns, since a plain list has neither.

    Args:
        flags: Flags from one or more C-rule checks, already in rule order.

    Returns:
        One row per flag, columns timestamp/rule/original_value/reason.
        Empty (correctly-shaped, zero-row) DataFrame if flags is empty --
        never omit the columns just because nothing was flagged.
    """
    return pd.DataFrame.from_records(
        [
            {
                "timestamp": f.timestamp,
                "rule": f.rule,
                "original_value": f.original_value,
                "reason": f.reason,
            }
            for f in flags
        ],
        columns=["timestamp", "rule", "original_value", "reason"],
    )


def run_cleaning_pipeline(
    raw: pd.Series,
    rh_pct: pd.Series,
    config: dict,
) -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    """Run C1-C9 in the fixed, physically-motivated order and collect logs.

    Order is not arbitrary -- see 01_LEARNING_PATH lesson notes / L3.7
    Concept section for why bounds must precede flatline detection, why
    gap handling must follow all fault detection, and why change-point
    detection runs last, on the fully cleaned series.

    Args:
        raw: The uncleaned hourly chilledwater_kwh series, straight from
            the weather-joined load (post L2.3/L2.4, pre any C-rule).
        rh_pct: The matching relative-humidity series, same index as raw
            (from the same weather join) -- required by C7, which is part
            of the C1/C2/C7 physical-bounds gate this pipeline runs first.
            Its own cleaned output is not returned: RH is validated here
            only to flag physically-impossible readings for the report;
            the RH series that modeling code actually consumes is
            schema-validated separately, at L3.6's validate_schema() gate.
        config: Parsed cleaning.yaml (the FULL dict -- each individual
            check function indexes its own relevant top-level key itself,
            e.g. check_extreme_spikes reads config["c2_spike"]).

    Returns:
        (cleaned, logs): `cleaned` is the final chilledwater_kwh series
        after all nine rules. `logs` maps rule-group name -> its
        DataFrame log, for the report to tabulate.
    """
    logs: dict[str, pd.DataFrame] = {}

    bounded, _rh_bounded, bounds_flags = validate_physical_bounds(raw, rh_pct, config)
    logs["C1_C2_C7_physical_bounds"] = _flags_to_log(bounds_flags)

    deflatlined, flatline_flags = detect_flatlines(bounded, config)
    logs["C3_C4_flatlines"] = _flags_to_log(flatline_flags)

    filled, gap_log = handle_gaps(deflatlined, config["gap_handling"]["max_interpolate_hours"])
    logs["C5_C6_gaps"] = gap_log

    _, change_points = detect_change_points(
        filled,
        config["change_point_detection"]["penalty"],
        config["change_point_detection"]["min_segment_days"],
    )
    logs["C8_change_points"] = pd.DataFrame({"change_point": change_points})

    final, month_log = drop_sparse_months(filled, config["reporting"]["month_min_valid_pct"])
    logs["C9_sparse_months"] = month_log

    return final, logs