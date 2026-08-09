"""Weather data loading and psychrometric feature engineering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import psychrolib

logger = logging.getLogger(__name__)

# psychrolib's unit system is process-global state -- set it once at import
# time rather than inside every function, since calling SetUnitSystem
# repeatedly is redundant and invites a mismatched call being added later
# by mistake in a new function.
psychrolib.SetUnitSystem(psychrolib.SI)

# ASHRAE standard atmospheric pressure at sea level (Pa). Used ONLY when a
# weather row's seaLvlPressure reading is missing. Named so the fallback is
# traceable in review, per 05_ENGINEERING_STANDARDS.md SS2 ("no magic numbers").
STANDARD_ATMOSPHERIC_PRESSURE_PA: float = 101_325.0

WEATHER_RELATIVE_PATH = Path("data/weather/weather.csv")


def load_weather(bdg2_root: Path, site_id: str | None = None) -> pd.DataFrame:
    """Load BDG2's combined hourly weather file.

    Args:
        bdg2_root: Path to the root of the extracted BDG2 repository
            (the directory containing ``data/weather/weather.csv``).
        site_id: If given, return only rows for this site. If ``None``,
            return the full 19-site file.

    Returns:
        DataFrame with ``site_id``, ``timestamp`` (tz-naive, local time per
        BDG2 convention), ``airTemperature``, ``dewTemperature``,
        ``seaLvlPressure``, plus the remaining raw BDG2 weather columns.

    Raises:
        FileNotFoundError: If ``weather.csv`` is not found under
            ``bdg2_root``.
        ValueError: If ``site_id`` is given but no rows match it.
    """
    weather_path = bdg2_root / WEATHER_RELATIVE_PATH
    if not weather_path.exists():
        raise FileNotFoundError(
            f"weather.csv not found at {weather_path}. "
            "Check bdg2_root points at the extracted BDG2 repo root."
        )

    df = pd.read_csv(weather_path, parse_dates=["timestamp"])

    if site_id is not None:
        df = df[df["site_id"] == site_id].copy()
        if df.empty:
            raise ValueError(f"No weather rows found for site_id={site_id!r}")

    logger.info(
        "Loaded weather: %d rows, %d site(s)", len(df), df["site_id"].nunique()
    )
    return df


def add_psychrometric_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Derive RH, humidity ratio, wet bulb, and enthalpy via psychrolib.

    Args:
        weather_df: Weather DataFrame with ``airTemperature`` (dry bulb, °C),
            ``dewTemperature`` (°C), and ``seaLvlPressure`` (hPa, may
            contain NaN) columns.

    Returns:
        A copy of ``weather_df`` with four new columns:
        ``rh_pct``, ``humidity_ratio``, ``wet_bulb_c``, ``enthalpy_j_per_kg``.

    Raises:
        ValueError: If any derived row violates INV-9
            (``T_dew_point <= T_wet_bulb <= T_dry_bulb``).
    """
    df = weather_df.copy()

    pressure_pa = df["seaLvlPressure"].where(
        df["seaLvlPressure"].notna(),
        STANDARD_ATMOSPHERIC_PRESSURE_PA / 100.0,  # keep units consistent, convert below
    ) * 100.0  # hPa -> Pa

    rh_pct: list[float] = []
    humidity_ratio: list[float] = []
    wet_bulb_c: list[float] = []
    enthalpy_j_per_kg: list[float] = []

    for t_dry, t_dew, p_pa in zip(
        df["airTemperature"], df["dewTemperature"], pressure_pa, strict=True
    ):
        if pd.isna(t_dry) or pd.isna(t_dew):
            rh_pct.append(float("nan"))
            humidity_ratio.append(float("nan"))
            wet_bulb_c.append(float("nan"))
            enthalpy_j_per_kg.append(float("nan"))
            continue

        rh_frac = psychrolib.GetRelHumFromTDewPoint(t_dry, t_dew)
        w = psychrolib.GetHumRatioFromTDewPoint(t_dew, p_pa)
        t_wb = psychrolib.GetTWetBulbFromRelHum(t_dry, rh_frac, p_pa)
        h = psychrolib.GetMoistAirEnthalpy(t_dry, w)

        rh_pct.append(rh_frac * 100.0)
        humidity_ratio.append(w)
        wet_bulb_c.append(t_wb)
        enthalpy_j_per_kg.append(h)

    df["rh_pct"] = rh_pct
    df["humidity_ratio"] = humidity_ratio
    df["wet_bulb_c"] = wet_bulb_c
    df["enthalpy_j_per_kg"] = enthalpy_j_per_kg

    _validate_inv9(df)
    return df


def _validate_inv9(df: pd.DataFrame) -> None:
    """Raise if T_dew <= T_wetbulb <= T_dry is violated (INV-9).

    Only rows where all three values are present are checked; NaN rows are
    skipped rather than treated as violations.
    """
    valid = df.dropna(subset=["dewTemperature", "wet_bulb_c", "airTemperature"])
    tolerance_c = 1e-6
    broken = valid[
        (valid["dewTemperature"] > valid["wet_bulb_c"] + tolerance_c)
        | (valid["wet_bulb_c"] > valid["airTemperature"] + tolerance_c)
    ]
    if not broken.empty:
        raise ValueError(
            f"INV-9 violated on {len(broken)} row(s): "
            "T_dew_point <= T_wet_bulb <= T_dry_bulb failed. "
            "Check the seaLvlPressure unit conversion (hPa -> Pa) first -- "
            "a unit bug here is the most common cause."
        )


def join_weather(
    building_df: pd.DataFrame, weather_df: pd.DataFrame, site_id: str
) -> pd.DataFrame:
    """Join a building's meter data onto weather on (site_id, timestamp).

    Args:
        building_df: Long-format meter DataFrame with a ``timestamp``
            column (as returned by ``load_meter``).
        weather_df: Weather DataFrame with psychrometric features already
            added by ``add_psychrometric_features``.
        site_id: The BDG2 site this building belongs to (e.g. ``"Fox"``),
            read from the building's row in metadata -- never parsed from
            the building_id string.

    Returns:
        ``building_df`` inner-joined with the given site's weather rows on
        ``timestamp``.

    Raises:
        ValueError: If the join drops more than 1% of building_df's rows,
            which signals a timestamp or site mismatch rather than genuine
            missing weather coverage.
    """
    site_weather = weather_df[weather_df["site_id"] == site_id]
    if site_weather.empty:
        raise ValueError(f"No weather rows for site_id={site_id!r}")

    merged = building_df.merge(
        site_weather.drop(columns=["site_id"]), on="timestamp", how="inner"
    )

    dropped_frac = 1 - len(merged) / len(building_df)
    if dropped_frac > 0.01:
        logger.warning(
            "Weather join dropped %.2f%% of building rows for site %s -- "
            "check timestamp alignment before proceeding to L2.4.",
            dropped_frac * 100,
            site_id,
        )

    return merged

def cross_correlate_lag(
    load: pd.Series, temp: pd.Series, max_lag_hours: int = 12
) -> pd.DataFrame:
    """Cross-correlate a load series against a temperature series by lag.

    Computes Pearson correlation between ``load`` and ``temp`` shifted by
    each integer lag in ``[-max_lag_hours, +max_lag_hours]``. Both series
    must share the same DatetimeIndex (hourly, aligned).

    Args:
        load: Hourly load series (e.g. chilled water meter reading), with
            a DatetimeIndex.
        temp: Hourly outdoor dry-bulb temperature series, same index.
        max_lag_hours: How far in each direction to search.

    Returns:
        DataFrame with columns ``lag_hours`` and ``correlation``. A
        positive lag means load is being compared against PAST
        temperature (load lags temp) -- the physically expected case.

    Raises:
        ValueError: If ``load`` and ``temp`` do not share an index, or if
            fewer than 100 overlapping non-NaN points remain after
            shifting (too little data for a meaningful correlation).
    """
    if not load.index.equals(temp.index):
        raise ValueError(
            "load and temp must share an identical DatetimeIndex -- "
            "align them (e.g. via join_weather) before calling this."
        )

    lags = range(-max_lag_hours, max_lag_hours + 1)
    correlations: list[float] = []
    for lag in lags:
        shifted_temp = temp.shift(lag)
        valid = pd.DataFrame({"load": load, "temp": shifted_temp}).dropna()
        if len(valid) < 100:
            raise ValueError(
                f"Only {len(valid)} overlapping points at lag={lag}h -- "
                "too little data for a meaningful cross-correlation."
            )
        correlations.append(valid["load"].corr(valid["temp"]))

    return pd.DataFrame({"lag_hours": list(lags), "correlation": correlations})


def validate_timezone_alignment(
    load: pd.Series,
    temp: pd.Series,
    expected_lag_range_hours: tuple[int, int] = (0, 6),
    max_lag_hours: int = 12,
) -> int:
    """Blocking gate: confirm load lags temperature by a physically sane amount.

    This is the mandatory check from 04_DATA_CONTRACT.md SS3. Thermal mass
    delay between an outdoor temperature swing and the resulting cooling
    load is well-documented to fall in the 1-3 hour range; a wider 0-6
    hour tolerance is used here to allow for building-to-building
    variation without masking a genuine misalignment.

    Args:
        load: Hourly load series with a DatetimeIndex.
        temp: Hourly outdoor dry-bulb temperature series, same index.
        expected_lag_range_hours: Inclusive (min, max) lag considered
            physically plausible.
        max_lag_hours: Search window passed to cross_correlate_lag.

    Returns:
        The best (highest-correlation) lag in hours, if it falls inside
        the expected range.

    Raises:
        ValueError: If the best lag falls outside the expected range.
            This is a hard stop -- do not catch and continue. A pipeline
            that proceeds past this point with misaligned data produces
            results that fail silently, not loudly.
    """
    result = cross_correlate_lag(load, temp, max_lag_hours=max_lag_hours)
    best_row = result.loc[result["correlation"].idxmax()]
    best_lag = int(best_row["lag_hours"])
    best_corr = float(best_row["correlation"])

    lo, hi = expected_lag_range_hours
    if not (lo <= best_lag <= hi):
        raise ValueError(
            f"TIMEZONE ALIGNMENT CHECK FAILED.\n"
            f"  Best cross-correlation lag: {best_lag}h (r={best_corr:.3f})\n"
            f"  Expected range: [{lo}, {hi}]h (thermal-mass delay)\n"
            f"  This is NOT a modeling problem -- it means the weather "
            f"join in join_weather() is misaligned in time. Do not proceed "
            f"to L3+ until this is fixed. Common causes: weather timestamps "
            f"in UTC while meter timestamps are local, or an off-by-one-day "
            f"join error."
        )

    logger.info(
        "Timezone alignment OK: best lag = %dh (r=%.3f, within [%d, %d])",
        best_lag, best_corr, lo, hi,
    )
    return best_lag