"""Unit tests for src/cooling_twin/data/select.py."""
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import yaml

from cooling_twin.data.select import (
    MAX_MISSING_FRACTION,
    BuildingCandidate,
    _apply_hard_filters,
    _missing_fraction_by_building,
    _soft_score,
    _year_presence_by_building,
    score_candidates,
    select_buildings,
    write_buildings_yaml,
)


def _long_meter(rows: Sequence[tuple[str, str, float | None]]) -> pd.DataFrame:
    """rows: (timestamp_str, building_id, meter_reading)."""
    return pd.DataFrame(
        [
            {"timestamp": pd.Timestamp(ts), "building_id": bid, "meter_reading": val}
            for ts, bid, val in rows
        ]
    )


def test_missing_fraction_by_building_only_counts_required_years() -> None:
    rows = [
        ("2016-01-01 00:00", "A", 1.0),
        ("2016-01-01 01:00", "A", None),  # missing, counts
        ("2015-01-01 00:00", "A", None),  # outside REQUIRED_YEARS, excluded
    ]
    result = _missing_fraction_by_building(_long_meter(rows), years=(2016,))
    assert result["A"] == pytest.approx(0.5)


def test_year_presence_by_building() -> None:
    rows = [
        ("2016-01-01 00:00", "A", 1.0),
        ("2017-01-01 00:00", "B", 1.0),
    ]
    long = _long_meter(rows)
    has_2016 = _year_presence_by_building(long, 2016)
    has_2017 = _year_presence_by_building(long, 2017)
    assert has_2016["A"] and not has_2016.get("B", False)
    assert has_2017["B"] and not has_2017.get("A", False)


def test_year_presence_false_when_all_null() -> None:
    rows = [("2016-01-01 00:00", "A", None)]
    has_2016 = _year_presence_by_building(_long_meter(rows), 2016)
    assert not has_2016["A"]


def _filters(**overrides: object) -> list[str]:
    base = dict(
        building_id="Alpha_education_A",
        site_id="AX",
        primary_use="Education",
        floor_area_m2=1000.0,
        missing_cw=0.0,
        missing_elec=0.0,
        has_2016=True,
        has_2017=True,
    )
    base.update(overrides)
    return cast(list[str], _apply_hard_filters(**base))


def test_apply_hard_filters_all_pass_is_empty() -> None:
    assert _filters() == []


def test_apply_hard_filters_excluded_site() -> None:
    failures = _filters(site_id="Moose")
    assert any("Moose" in f and "excluded" in f for f in failures)


def test_apply_hard_filters_excluded_building_id() -> None:
    failures = _filters(building_id="Hog_office_Darline")
    assert any("Hog_office_Darline" in f and "excluded" in f for f in failures)


def test_apply_hard_filters_missing_chilledwater_over_threshold() -> None:
    failures = _filters(missing_cw=MAX_MISSING_FRACTION + 0.01)
    assert any("chilledwater missing" in f for f in failures)


def test_apply_hard_filters_missing_electricity_over_threshold() -> None:
    failures = _filters(missing_elec=MAX_MISSING_FRACTION + 0.01)
    assert any("electricity missing" in f for f in failures)


def test_apply_hard_filters_floor_area_missing() -> None:
    failures = _filters(floor_area_m2=float("nan"))
    assert any("floor_area_m2" in f for f in failures)


def test_apply_hard_filters_primary_use_missing() -> None:
    failures = _filters(primary_use=None)
    assert any("primaryspaceusage" in f for f in failures)


def test_apply_hard_filters_missing_year_coverage() -> None:
    failures = _filters(has_2017=False)
    assert any("both 2016 and 2017" in f for f in failures)


def test_soft_score_preferred_use_and_year_and_cleanliness() -> None:
    score, reasons = _soft_score("Education", 2000, missing_cw=0.0)
    assert score == pytest.approx(2.0 + 1.0 + 1.0)
    assert any("Education" in r for r in reasons)
    assert any("year_built=2000" in r for r in reasons)


def test_soft_score_non_preferred_use_no_bonus() -> None:
    score, _ = _soft_score("Retail", None, missing_cw=0.0)
    assert score == pytest.approx(1.0)  # only the cleanliness bonus


def test_soft_score_cleanliness_scales_with_missing_fraction() -> None:
    clean_score, _ = _soft_score("Education", None, missing_cw=0.0)
    dirtier_score, _ = _soft_score("Education", None, missing_cw=MAX_MISSING_FRACTION / 2)
    assert dirtier_score < clean_score


def _write_scoring_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Metadata + chilledwater/electricity CSVs covering every hard-filter
    branch: a clean pass, a non-preferred-use pass, a missing-meter-flag
    exclusion, an over-threshold missing fraction, a site exclusion, a
    building_id exclusion, and a missing-year-coverage failure.
    """
    metadata_path = tmp_path / "metadata.csv"
    meters_dir = tmp_path / "meters"
    meters_dir.mkdir()

    pd.DataFrame(
        [
            dict(building_id="Alpha_education_A", site_id="AX", primaryspaceusage="Education",
                 sqft=100_000.0, yearbuilt=2000, electricity="Yes", chilledwater="Yes"),
            dict(building_id="Beta_education_B", site_id="BY", primaryspaceusage="Education",
                 sqft=90_000.0, yearbuilt=1995, electricity="Yes", chilledwater="Yes"),
            dict(building_id="Gamma_retail_C", site_id="CZ", primaryspaceusage="Retail",
                 sqft=80_000.0, yearbuilt=1980, electricity="Yes", chilledwater="Yes"),
            dict(building_id="Delta_education_D", site_id="DW", primaryspaceusage="Education",
                 sqft=70_000.0, yearbuilt=2010, electricity="Yes", chilledwater=None),
            dict(building_id="Epsilon_education_E", site_id="EV", primaryspaceusage="Education",
                 sqft=60_000.0, yearbuilt=2005, electricity="Yes", chilledwater="Yes"),
            dict(building_id="Zeta_education_Z", site_id="Moose", primaryspaceusage="Education",
                 sqft=50_000.0, yearbuilt=1990, electricity="Yes", chilledwater="Yes"),
            dict(building_id="Hog_office_Darline", site_id="HD", primaryspaceusage="Office",
                 sqft=40_000.0, yearbuilt=1985, electricity="Yes", chilledwater="Yes"),
            dict(building_id="Theta_education_T", site_id="TT", primaryspaceusage="Education",
                 sqft=30_000.0, yearbuilt=1975, electricity="Yes", chilledwater="Yes"),
        ]
    ).to_csv(metadata_path, index=False)

    timestamps = [
        "2016-01-01 00:00:00", "2016-01-01 01:00:00",
        "2017-01-01 00:00:00", "2017-01-01 01:00:00",
    ]

    def wide(col_values: dict[str, list[float | None]]) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": timestamps, **col_values})

    wide(
        {
            "Alpha_education_A": [100.0, 102.0, 101.0, 103.0],
            "Beta_education_B": [80.0, 81.0, 82.0, 83.0],
            "Gamma_retail_C": [70.0, 71.0, 72.0, 73.0],
            "Epsilon_education_E": [None, None, None, 60.0],  # 75% missing -> fails threshold
            "Zeta_education_Z": [10.0, 11.0, 12.0, 13.0],
            "Hog_office_Darline": [10.0, 11.0, 12.0, 13.0],
            "Theta_education_T": [90.0, 91.0, None, None],  # no 2017 coverage
        }
    ).to_csv(meters_dir / "chilledwater.csv", index=False)

    wide(
        {
            "Alpha_education_A": [50.0, 51.0, 52.0, 53.0],
            "Beta_education_B": [40.0, 41.0, 42.0, 43.0],
            "Gamma_retail_C": [30.0, 31.0, 32.0, 33.0],
            "Epsilon_education_E": [20.0, 21.0, 22.0, 23.0],
            "Zeta_education_Z": [5.0, 6.0, 7.0, 8.0],
            "Hog_office_Darline": [5.0, 6.0, 7.0, 8.0],
            "Theta_education_T": [10.0, 11.0, 12.0, 13.0],
        }
    ).to_csv(meters_dir / "electricity.csv", index=False)

    return metadata_path, meters_dir


def test_score_candidates_end_to_end(tmp_path: Path) -> None:
    metadata_path, meters_dir = _write_scoring_fixture(tmp_path)

    candidates = score_candidates(metadata_path=metadata_path, meters_dir=meters_dir)
    by_id = {c.building_id: c for c in candidates}

    assert by_id["Alpha_education_A"].passed_hard_filters
    assert by_id["Alpha_education_A"].soft_score == pytest.approx(4.0)

    assert by_id["Beta_education_B"].passed_hard_filters
    assert by_id["Beta_education_B"].soft_score == pytest.approx(4.0)

    assert by_id["Gamma_retail_C"].passed_hard_filters
    assert by_id["Gamma_retail_C"].soft_score == pytest.approx(2.0)  # no preferred-use bonus

    assert not by_id["Delta_education_D"].passed_hard_filters
    assert any("meter absent" in f for f in by_id["Delta_education_D"].hard_filter_failures)

    epsilon_failures = by_id["Epsilon_education_E"].hard_filter_failures
    assert not by_id["Epsilon_education_E"].passed_hard_filters
    assert any("chilledwater missing" in f for f in epsilon_failures)

    assert not by_id["Zeta_education_Z"].passed_hard_filters
    assert any("Moose" in f for f in by_id["Zeta_education_Z"].hard_filter_failures)

    assert not by_id["Hog_office_Darline"].passed_hard_filters
    assert any("Hog_office_Darline" in f for f in by_id["Hog_office_Darline"].hard_filter_failures)

    assert not by_id["Theta_education_T"].passed_hard_filters
    assert any("both 2016 and 2017" in f for f in by_id["Theta_education_T"].hard_filter_failures)

    selection = select_buildings(candidates, n_generalisation=2)
    # tie-break: alphabetically first
    assert selection["primary"][0].building_id == "Alpha_education_A"
    generalisation_ids = {c.building_id for c in selection["generalisation"]}
    assert generalisation_ids == {"Beta_education_B", "Gamma_retail_C"}


def _candidate(
    building_id: str, site_id: str, soft_score: float, missing_cw: float = 0.0
) -> BuildingCandidate:
    return BuildingCandidate(
        building_id=building_id,
        site_id=site_id,
        primary_use="Education",
        floor_area_m2=1000.0,
        year_built=2000,
        missing_frac_chilledwater=missing_cw,
        missing_frac_electricity=0.0,
        passed_hard_filters=True,
        soft_score=soft_score,
    )


def test_select_buildings_prefers_distinct_sites() -> None:
    candidates = [
        _candidate("P", "site1", 4.0),
        _candidate("Q", "site1", 3.9),  # same site as primary -- skipped while distinct available
        _candidate("R", "site2", 3.0),
        _candidate("S", "site3", 2.0),
    ]
    selection = select_buildings(candidates, n_generalisation=2)
    assert selection["primary"][0].building_id == "P"
    assert [c.building_id for c in selection["generalisation"]] == ["R", "S"]


def test_select_buildings_falls_back_to_same_site_when_too_few_distinct() -> None:
    candidates = [
        _candidate("P", "site1", 4.0),
        _candidate("Q", "site1", 3.0),
        _candidate("T", "site1", 2.0),
    ]
    selection = select_buildings(candidates, n_generalisation=2)
    assert selection["primary"][0].building_id == "P"
    assert [c.building_id for c in selection["generalisation"]] == ["Q", "T"]


def test_select_buildings_raises_when_too_few_pass() -> None:
    candidates = [_candidate("P", "site1", 4.0)]
    with pytest.raises(ValueError, match="Only 1 candidates passed"):
        select_buildings(candidates, n_generalisation=2)


def test_select_buildings_deterministic_tie_break_by_building_id() -> None:
    candidates = [
        _candidate("Zeta", "site1", 3.0, missing_cw=0.05),
        _candidate("Alpha", "site2", 3.0, missing_cw=0.05),  # exact tie -> building_id decides
        _candidate("Beta", "site3", 1.0),
    ]
    selection = select_buildings(candidates, n_generalisation=2)
    assert selection["primary"][0].building_id == "Alpha"


def test_write_buildings_yaml_round_trips(tmp_path: Path) -> None:
    selection = {
        "primary": [_candidate("Fox_education_Theodore", "Fox", 4.0)],
        "generalisation": [
            _candidate("Panther_education_Aurora", "Panther", 3.13, missing_cw=0.0866)
        ],
    }
    out_path = tmp_path / "nested" / "buildings.yaml"

    write_buildings_yaml(selection, out_path)

    payload = yaml.safe_load(out_path.read_text())
    assert payload["primary"][0]["building_id"] == "Fox_education_Theodore"
    assert payload["primary"][0]["soft_score"] == 4.0
    assert payload["generalisation"][0]["missing_pct_chilledwater"] == pytest.approx(8.66)
