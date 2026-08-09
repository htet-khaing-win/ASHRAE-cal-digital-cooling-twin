"""Unit tests for src/cooling_twin/data/load.py."""
from pathlib import Path

import pandas as pd
import pytest

from cooling_twin.data.load import (
    _SYNTHETIC_N_BUILDINGS,
    _SYNTHETIC_N_HOURS,
    list_buildings_with_meter,
    load_metadata,
    load_meter,
)


def _write_metadata_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "building_id": ["Fox_education_Theodore", "Panther_education_Aurora"],
            "site_id": ["Fox", "Panther"],
            "primaryspaceusage": ["Education", "Education"],
            "sqft": [122_600.0, 86_233.0],
            "yearbuilt": [1971, 2013],
            "timezone": ["America/New_York", "America/New_York"],
            "electricity": ["Yes", "Yes"],
            "chilledwater": ["Yes", None],
        }
    ).to_csv(path, index=False)


def _write_wide_meter_csv(path: Path) -> None:
    idx = pd.date_range("2016-01-01", periods=3, freq="h")
    pd.DataFrame(
        {
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "Fox_education_Theodore": [100.0, 105.0, None],
            "Panther_education_Aurora": [50.0, None, 55.0],
        }
    ).to_csv(path, index=False)


def test_load_metadata_reads_real_csv(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    _write_metadata_csv(path)

    df = load_metadata(path)

    assert df.index.name == "building_id"
    assert set(df.index) == {"Fox_education_Theodore", "Panther_education_Aurora"}
    assert df.loc["Fox_education_Theodore", "site_id"] == "Fox"


def test_load_metadata_missing_file_falls_back_to_synthetic(tmp_path: Path) -> None:
    df = load_metadata(tmp_path / "does_not_exist.csv")

    assert len(df) == _SYNTHETIC_N_BUILDINGS
    assert df.index.name == "building_id"
    assert (df["primaryspaceusage"] == "Office").all()


def test_load_meter_melts_wide_to_long(tmp_path: Path) -> None:
    path = tmp_path / "chilledwater.csv"
    _write_wide_meter_csv(path)

    long = load_meter("chilledwater", meters_dir=tmp_path)

    assert list(long.columns) == ["timestamp", "building_id", "meter_reading"]
    assert len(long) == 6  # 3 timestamps * 2 buildings
    assert set(long["building_id"]) == {"Fox_education_Theodore", "Panther_education_Aurora"}
    fox_first = long[
        (long["building_id"] == "Fox_education_Theodore")
        & (long["timestamp"] == pd.Timestamp("2016-01-01 00:00:00"))
    ]
    assert fox_first["meter_reading"].iloc[0] == 100.0


def test_load_meter_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown meter type"):
        load_meter("nonsense_meter_type")


def test_load_meter_missing_file_falls_back_to_synthetic(tmp_path: Path) -> None:
    long = load_meter("chilledwater", meters_dir=tmp_path)

    assert set(long.columns) == {"timestamp", "building_id", "meter_reading"}
    assert long["building_id"].nunique() == _SYNTHETIC_N_BUILDINGS
    assert len(long) == _SYNTHETIC_N_BUILDINGS * _SYNTHETIC_N_HOURS
    assert long["meter_reading"].notna().all()


def test_list_buildings_with_meter_returns_sorted_present_only(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    _write_metadata_csv(path)
    metadata = load_metadata(path)

    result = list_buildings_with_meter(metadata, "chilledwater")

    assert result == ["Fox_education_Theodore"]  # Panther has NaN chilledwater flag


def test_list_buildings_with_meter_unknown_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    _write_metadata_csv(path)
    metadata = load_metadata(path)

    with pytest.raises(ValueError, match="Unknown meter type"):
        list_buildings_with_meter(metadata, "nonsense_meter_type")


def test_list_buildings_with_meter_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    _write_metadata_csv(path)
    metadata = load_metadata(path).drop(columns=["chilledwater"])

    with pytest.raises(ValueError, match="no 'chilledwater' column"):
        list_buildings_with_meter(metadata, "chilledwater")
