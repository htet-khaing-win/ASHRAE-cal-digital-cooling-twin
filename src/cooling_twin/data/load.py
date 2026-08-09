"""BDG2 dataset loading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# WSL filesystem only -- never /mnt/c/ (see 04_DATA_CONTRACT.md SS1, L0.1)
BDG2_ROOT = Path.home() / "data" / "bdg2"
METADATA_PATH = BDG2_ROOT / "data" / "metadata" / "metadata.csv"
METERS_RAW_DIR = BDG2_ROOT / "data" / "meters" / "raw"

# Meter types present in BDG2 (04_DATA_CONTRACT.md SS1)
METER_TYPES = (
    "electricity", "chilledwater", "steam", "hotwater",
    "gas", "water", "solar", "irrigation",
)

# Synthetic fallback sizing -- just enough to exercise the interface when
# BDG2 hasn't been downloaded yet
_SYNTHETIC_N_BUILDINGS = 5
_SYNTHETIC_N_HOURS = 24 * 7


def load_metadata(path: Path = METADATA_PATH) -> pd.DataFrame:
    """Load BDG2 building metadata.

    Args:
        path: Path to metadata.csv. Defaults to the standard BDG2_ROOT layout.

    Returns:
        DataFrame indexed by building_id, one row per building, with columns
        including site_id, primaryspaceusage, sqft, yearbuilt, timezone, and
        one Yes/NaN presence column per meter type.
    """
    if not path.exists():
        logger.warning(
            "BDG2 metadata not found at %s -- generating synthetic metadata "
            "so the pipeline can still be demonstrated. Run "
            "scripts/download_data.py to fetch the real dataset.",
            path,
        )
        return _synthetic_metadata()

    df = pd.read_csv(path)
    df = df.set_index("building_id")
    logger.info("Loaded metadata for %d buildings from %s", len(df), path)
    return df


def load_meter(meter_type: str, meters_dir: Path = METERS_RAW_DIR) -> pd.DataFrame:
    """Load one meter type in long format.

    BDG2 ships each meter type as a wide CSV (one column per building_id).
    This melts it to long format -- (timestamp, building_id, meter_reading)
    -- which is the shape every downstream stage expects.

    Args:
        meter_type: One of METER_TYPES, e.g. "chilledwater".
        meters_dir: Directory containing the raw meter CSVs.

    Returns:
        Long-format DataFrame with columns timestamp, building_id, meter_reading.

    Raises:
        ValueError: If meter_type is not a recognised BDG2 meter type.
    """
    if meter_type not in METER_TYPES:
        raise ValueError(
            f"Unknown meter type {meter_type!r}. Expected one of {METER_TYPES}."
        )

    path = meters_dir / f"{meter_type}.csv"
    if not path.exists():
        logger.warning(
            "BDG2 meter file not found at %s -- generating synthetic %s "
            "readings so the pipeline can still be demonstrated.",
            path, meter_type,
        )
        return _synthetic_meter(meter_type)

    wide = pd.read_csv(path)
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], format="%Y-%m-%d %H:%M:%S")
    long = wide.melt(id_vars="timestamp", var_name="building_id", value_name="meter_reading")
    logger.info(
        "Loaded %s meter: %d readings across %d buildings",
        meter_type, len(long), long["building_id"].nunique(),
    )
    return long


def list_buildings_with_meter(metadata: pd.DataFrame, meter_type: str) -> list[str]:
    """Buildings whose metadata marks a given meter type as present.

    Args:
        metadata: DataFrame from load_metadata(), indexed by building_id.
        meter_type: One of METER_TYPES.

    Returns:
        Sorted list of building_id values with that meter present.

    Raises:
        ValueError: If meter_type is unrecognised or the column is missing.
    """
    if meter_type not in METER_TYPES:
        raise ValueError(
            f"Unknown meter type {meter_type!r}. Expected one of {METER_TYPES}."
        )
    if meter_type not in metadata.columns:
        raise ValueError(
            f"metadata has no {meter_type!r} column -- was it loaded correctly?"
        )
    present = metadata[metadata[meter_type].notna()]
    return sorted(present.index.tolist())


def _synthetic_metadata() -> pd.DataFrame:
    """Small synthetic metadata table, shaped like real BDG2 metadata."""
    rng = np.random.default_rng(42)
    building_ids = [f"Synthetic_office_Building{i}" for i in range(_SYNTHETIC_N_BUILDINGS)]
    df = pd.DataFrame(
        {
            "building_id": building_ids,
            "site_id": ["SyntheticSite"] * _SYNTHETIC_N_BUILDINGS,
            "primaryspaceusage": ["Office"] * _SYNTHETIC_N_BUILDINGS,
            "sqft": rng.uniform(20_000, 200_000, _SYNTHETIC_N_BUILDINGS).round(0),
            "yearbuilt": rng.integers(1970, 2020, _SYNTHETIC_N_BUILDINGS),
            "timezone": ["America/New_York"] * _SYNTHETIC_N_BUILDINGS,
            "electricity": ["Yes"] * _SYNTHETIC_N_BUILDINGS,
            "chilledwater": ["Yes"] * _SYNTHETIC_N_BUILDINGS,
        }
    ).set_index("building_id")
    return df


def _synthetic_meter(meter_type: str) -> pd.DataFrame:
    """Small synthetic meter series, one week hourly, shaped like real BDG2 data."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2016-01-01", periods=_SYNTHETIC_N_HOURS, freq="h")
    building_ids = [f"Synthetic_office_Building{i}" for i in range(_SYNTHETIC_N_BUILDINGS)]
    rows = []
    for building_id in building_ids:
        base = rng.uniform(50, 200)
        readings = base + rng.normal(0, base * 0.1, _SYNTHETIC_N_HOURS)
        rows.append(
            pd.DataFrame(
                {"timestamp": timestamps, "building_id": building_id, "meter_reading": readings}
            )
        )
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metadata = load_metadata()
    chw = load_meter("chilledwater")
    buildings = list_buildings_with_meter(metadata, "chilledwater")
    print(f"Metadata: {len(metadata)} buildings")
    print(f"Chilled water meter: {len(chw)} readings, {chw['building_id'].nunique()} buildings")
    preview = buildings[:5]
    suffix = "..." if len(buildings) > 5 else ""
    print(f"Buildings with chilledwater meter: {preview}{suffix}")