"""The single interface boundary between data engineering (M2-M3) and
modeling code (M4+). Modeling code must accept only BuildingTimeSeries --
never a raw CSV or DataFrame -- so this file is the contract in
04_DATA_CONTRACT.md SS6, made executable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

_HOURLY = pd.Timedelta(hours=1)
_TEMP_TOLERANCE_C = 1e-6  # floating-point slack for INV-9's <= comparisons


@dataclass(frozen=True, eq=False)
class BuildingTimeSeries:
    """The validated, contract-compliant representation of one building.

    Frozen so that once a module receives this object, it cannot be
    silently mutated by a later stage in a pipeline -- any transformation
    must produce a new BuildingTimeSeries, not edit this one in place.

    Attributes:
        building_id: BDG2 building identifier, e.g. "Fox_education_Theodore".
        timestamp: tz-aware, strictly hourly, monotonic increasing index.
        chilledwater_kwh: Target variable for calibration.
        electricity_kwh: Optional whole-building electricity meter.
        air_temp_c: Dry-bulb outdoor air temperature.
        dew_temp_c: Dew point temperature.
        wet_bulb_c: Derived via psychrolib (L2.3) -- not hand-rolled.
        humidity_ratio: Derived, kg water / kg dry air.
        rh_pct: Derived relative humidity, expected in [0, 100].
        floor_area_m2: Must be > 0.
        primary_use: BDG2 primaryspaceusage label.
        year_built: May be unknown for a given building.
        site_id: BDG2 site identifier, used for the weather join.
    """

    building_id: str
    timestamp: pd.DatetimeIndex
    chilledwater_kwh: pd.Series
    electricity_kwh: pd.Series | None
    air_temp_c: pd.Series
    dew_temp_c: pd.Series
    wet_bulb_c: pd.Series
    humidity_ratio: pd.Series
    rh_pct: pd.Series
    floor_area_m2: float
    primary_use: str
    year_built: int | None
    site_id: str


def validate_schema(ts: BuildingTimeSeries) -> None:
    """Raise if `ts` violates the data contract. Call before any modeling.

    Checks, in order: index shape (tz-aware, hourly, monotonic, no
    duplicates), that every series is aligned to that index, the value
    bounds this schema can check (INV-7, INV-9), and the metadata fields.
    Physical invariants that depend on fields not present in this schema
    (INV-1 COP, INV-3 condenser water, INV-5 R/C, INV-6 PLR, INV-8 energy
    balance) are enforced later, at the pipeline stage that introduces
    those quantities -- this gate only checks what it can see.

    Args:
        ts: The object to validate.

    Raises:
        ValueError: On the first contract violation found. The message
            names which check failed and why -- see 03_DOMAIN_REFERENCE.md
            for the physical reasoning behind INV-7 and INV-9.
    """
    if not isinstance(ts.timestamp, pd.DatetimeIndex):
        raise ValueError(f"timestamp must be a DatetimeIndex, got {type(ts.timestamp)}")
    if ts.timestamp.tz is None:
        raise ValueError("timestamp index must be tz-aware, got tz-naive")
    if ts.timestamp.has_duplicates:
        raise ValueError("timestamp index contains duplicate timestamps")
    if not ts.timestamp.is_monotonic_increasing:
        raise ValueError("timestamp index must be monotonic increasing")

    step_sizes = ts.timestamp.to_series().diff().dropna().unique()
    if len(step_sizes) != 1 or step_sizes[0] != _HOURLY:
        raise ValueError(
            f"timestamp index must be strictly hourly; found step size(s) {list(step_sizes)}"
        )

    series_fields: dict[str, pd.Series] = {
        "chilledwater_kwh": ts.chilledwater_kwh,
        "air_temp_c": ts.air_temp_c,
        "dew_temp_c": ts.dew_temp_c,
        "wet_bulb_c": ts.wet_bulb_c,
        "humidity_ratio": ts.humidity_ratio,
        "rh_pct": ts.rh_pct,
    }
    if ts.electricity_kwh is not None:
        series_fields["electricity_kwh"] = ts.electricity_kwh

    for name, series in series_fields.items():
        if len(series) != len(ts.timestamp):
            raise ValueError(
                f"{name} has length {len(series)}, timestamp has length "
                f"{len(ts.timestamp)} -- they must match"
            )
        if not series.index.equals(ts.timestamp):
            raise ValueError(f"{name}'s index is not identical to ts.timestamp")

    valid_rh = ts.rh_pct.dropna()
    if not valid_rh.between(0, 100).all():
        bad = valid_rh[~valid_rh.between(0, 100)]
        raise ValueError(
            f"rh_pct has {len(bad)} value(s) outside [0, 100] -- violates INV-7, "
            f"e.g. {bad.iloc[0]:.2f} at {bad.index[0]}"
        )

    aligned = pd.notna(ts.dew_temp_c) & pd.notna(ts.wet_bulb_c) & pd.notna(ts.air_temp_c)
    dew, wet, dry = ts.dew_temp_c[aligned], ts.wet_bulb_c[aligned], ts.air_temp_c[aligned]
    ordering_ok = (dew <= wet + _TEMP_TOLERANCE_C) & (wet <= dry + _TEMP_TOLERANCE_C)
    if not ordering_ok.all():
        bad_idx = ordering_ok[~ordering_ok].index[0]
        raise ValueError(
            f"dew_temp_c <= wet_bulb_c <= air_temp_c violated at {bad_idx} "
            f"(dew={dew[bad_idx]:.2f}, wet={wet[bad_idx]:.2f}, dry={dry[bad_idx]:.2f}) "
            f"-- violates INV-9"
        )

    if ts.floor_area_m2 <= 0:
        raise ValueError(f"floor_area_m2 must be positive, got {ts.floor_area_m2}")
    if not ts.building_id:
        raise ValueError("building_id must be a non-empty string")
    if not ts.primary_use:
        raise ValueError("primary_use must be a non-empty string")

    logger.info(
        "validate_schema: %s passed all contract checks (%d hours)",
        ts.building_id,
        len(ts.timestamp),
    )