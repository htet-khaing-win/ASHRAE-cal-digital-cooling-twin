"""Unit tests for src/cooling_twin/data/select.py."""
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import yaml

from cooling_twin import SEED
from cooling_twin.data.select import (
    MAX_CEILING_CVRMSE_PCT,
    MAX_MISSING_FRACTION,
    MAX_STABILITY_RATIO,
    BuildingCandidate,
    ScreenMetrics,
    _apply_hard_filters,
    _cooling_to_electricity_ratio,
    _mean_intensity_by_building,
    _missing_fraction_by_building,
    _screen_soft_score,
    _soft_score,
    _warn_if_climate_undiverse,
    _year_presence_by_building,
    operational_stability,
    score_candidates,
    select_buildings,
    weather_conditional_mean,
    weather_explainability_ceiling,
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


# --------------------------------------------------------------------
# ADR-012 screen: explainability ceiling and operational stability
# --------------------------------------------------------------------


def _hourly_weather(hours: int = 8784, amplitude: float = 10.0) -> pd.DataFrame:
    """A year of smooth diurnal + seasonal outdoor temperature."""
    index = pd.date_range("2016-01-01", periods=hours, freq="h", tz="UTC")
    position = np.arange(hours, dtype=float)
    temperature = (
        20.0
        + amplitude * np.sin(2 * np.pi * position / 24.0)
        + 10.0 * np.sin(2 * np.pi * position / (24 * 365))
    )
    return pd.DataFrame({"airTemperature": temperature}, index=index)


def test_ceiling_is_near_zero_when_load_is_a_function_of_temperature() -> None:
    """A perfectly weather-driven building has almost no irreducible error.

    Not exactly zero: the estimator replaces each 2 degC bin with its
    mean, so a linear load carries a discretisation floor of a couple of
    percent. That floor is the price of a NON-PARAMETRIC ceiling -- it
    assumes no functional form, and in exchange it slightly over-states
    the error. Over-stating is the safe direction: it can only make the
    screen more conservative about admitting a building.
    """
    weather = _hourly_weather()
    load = pd.Series(50.0 * weather["airTemperature"] + 100.0, index=weather.index)

    ceiling = weather_explainability_ceiling(
        load, weather, {"airTemperature": np.arange(-10.0, 50.0, 2.0)}
    )

    assert ceiling < 4.0


def test_ceiling_recovers_the_noise_level_it_cannot_explain() -> None:
    """With 10% white noise added, the ceiling lands near 10%."""
    weather = _hourly_weather()
    signal = 50.0 * weather["airTemperature"] + 2000.0
    rng = np.random.default_rng(SEED)
    load = pd.Series(signal + rng.normal(0.0, 0.10 * signal.mean(), len(signal)),
                     index=weather.index)

    ceiling = weather_explainability_ceiling(
        load, weather, {"airTemperature": np.arange(-10.0, 50.0, 2.0)}
    )

    assert 8.0 < ceiling < 13.0


def test_ceiling_flags_a_building_no_model_can_fit() -> None:
    """Load driven by something the drivers do not contain."""
    weather = _hourly_weather()
    rng = np.random.default_rng(SEED)
    unrelated = pd.Series(rng.normal(1000.0, 400.0, len(weather)), index=weather.index)

    ceiling = weather_explainability_ceiling(
        unrelated, weather, {"airTemperature": np.arange(-10.0, 50.0, 2.0)}
    )

    assert ceiling > MAX_CEILING_CVRMSE_PCT


def test_ceiling_rejects_drivers_it_cannot_bin() -> None:
    weather = _hourly_weather()
    load = pd.Series(1.0, index=weather.index)

    with pytest.raises(ValueError, match="none of the binned drivers"):
        weather_explainability_ceiling(load, weather, {"windSpeed": [0.0, 10.0]})


def test_ceiling_refuses_to_report_from_too_few_scored_hours() -> None:
    """A ceiling from a handful of hours would top the ranking on noise."""
    weather = _hourly_weather(hours=200)
    load = pd.Series(50.0 * weather["airTemperature"], index=weather.index)

    with pytest.raises(ValueError, match="could be scored"):
        weather_explainability_ceiling(
            load, weather, {"airTemperature": np.arange(-10.0, 50.0, 2.0)}
        )


def test_flatlined_meter_scores_an_excellent_ceiling() -> None:
    """The trap this screen must never be run without M3's quality rules.

    A stuck sensor is trivially predictable, so it looks like the most
    calibratable building in the dataset. Hog_office_Darline scored the
    best ceiling in the real BDG2 screen (4.8%) and is exactly that.
    """
    weather = _hourly_weather()
    flatlined = pd.Series(500.0, index=weather.index)

    ceiling = weather_explainability_ceiling(
        flatlined, weather, {"airTemperature": np.arange(-10.0, 50.0, 2.0)}
    )

    assert ceiling < 1.0  # excellent -- and completely worthless


def test_stability_is_one_for_a_building_with_no_excursion() -> None:
    index = pd.date_range("2016-01-01", periods=24 * 120, freq="h", tz="UTC")
    steady = pd.Series(1000.0, index=index)

    assert operational_stability(steady) == pytest.approx(1.0)


def test_stability_finds_a_sustained_doubling() -> None:
    """Theodore's September signature, synthesised."""
    index = pd.date_range("2016-01-01", periods=24 * 120, freq="h", tz="UTC")
    load = pd.Series(1000.0, index=index)
    event = (load.index >= "2016-03-01") & (load.index < "2016-03-12")
    load[event] = 2000.0

    assert operational_stability(load) == pytest.approx(2.0, rel=0.05)


def test_stability_ignores_a_single_hot_day() -> None:
    """One spike is weather; a fortnight at double load is not."""
    index = pd.date_range("2016-01-01", periods=24 * 120, freq="h", tz="UTC")
    load = pd.Series(1000.0, index=index)
    load[(load.index >= "2016-03-01") & (load.index < "2016-03-02")] = 3000.0

    assert operational_stability(load) < 1.25


def test_stability_rejects_a_mostly_zero_meter() -> None:
    index = pd.date_range("2016-01-01", periods=24 * 120, freq="h", tz="UTC")
    dead = pd.Series(0.0, index=index)

    with pytest.raises(ValueError, match="not positive"):
        operational_stability(dead)


def test_stability_rejects_a_series_shorter_than_the_window() -> None:
    index = pd.date_range("2016-01-01", periods=24 * 5, freq="h", tz="UTC")

    with pytest.raises(ValueError, match="at least 11 days"):
        operational_stability(pd.Series(1000.0, index=index))


def test_screen_soft_score_rewards_headroom_and_steadiness() -> None:
    generous = ScreenMetrics(
        ceiling_cvrmse_pct=15.0, stability_ratio=1.0,
        humid_hours_frac=0.5, companion_meters=("steam",), screened_hours=8700,
    )
    marginal = ScreenMetrics(
        ceiling_cvrmse_pct=29.0, stability_ratio=1.55,
        humid_hours_frac=0.5, companion_meters=(), screened_hours=8700,
    )

    assert _screen_soft_score(generous)[0] > _screen_soft_score(marginal)[0]


def test_climate_diversity_warning_fires_for_an_all_dry_portfolio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dry = ScreenMetrics(
        ceiling_cvrmse_pct=20.0, stability_ratio=1.0,
        humid_hours_frac=0.02, companion_meters=(), screened_hours=8700,
    )
    selected = [
        BuildingCandidate(
            building_id=f"b{i}", site_id=f"s{i}", primary_use="Education",
            floor_area_m2=1000.0, year_built=2000, missing_frac_chilledwater=0.0,
            missing_frac_electricity=0.0, passed_hard_filters=True, screen=dry,
        )
        for i in range(3)
    ]

    with caplog.at_level("WARNING"):
        _warn_if_climate_undiverse(selected)

    assert "humid site" in caplog.text


# --------------------------------------------------------------------
# Intensity plausibility -- the screen is scale-invariant, this is not
# --------------------------------------------------------------------


def test_apply_hard_filters_rejects_an_impossible_intensity() -> None:
    """The Eagle site reports ~24,000 W/m2 across ~90 buildings.

    CV(RMSE) cannot see this: a meter in the wrong unit is exactly as
    explainable as a correct one, so without this filter the screen
    ranks a unit error as a strong candidate.
    """
    failures = _filters(intensity_w_per_m2=24_306.0)

    assert len(failures) == 1
    assert "unit convention" in failures[0]


def test_apply_hard_filters_rejects_a_meter_that_barely_registers() -> None:
    failures = _filters(intensity_w_per_m2=0.4)

    assert len(failures) == 1
    assert "W/m2 outside" in failures[0]


def test_apply_hard_filters_accepts_a_plausible_laboratory_intensity() -> None:
    """Q6: Fox's laboratories run a median 161.6 W/m2."""
    assert _filters(intensity_w_per_m2=161.6) == []


def test_apply_hard_filters_rejects_an_uncomputable_intensity() -> None:
    failures = _filters(intensity_w_per_m2=float("nan"))

    assert len(failures) == 1
    assert "not computable" in failures[0]


def test_mean_intensity_uses_the_same_area_source_as_floor_area_m2() -> None:
    """Both derive from sqft, so a building is filtered on what it reports."""
    meter = _long_meter(
        [("2016-01-01T00:00:00", "b1", 100.0), ("2016-01-01T01:00:00", "b1", 200.0)]
    )
    metadata = pd.DataFrame({"sqft": [10_763.9]}, index=["b1"])  # ~1000 m2

    intensity = _mean_intensity_by_building(meter, metadata, 2016)

    # mean 150 kW over 1000 m2 = 150 W/m2
    assert intensity["b1"] == pytest.approx(150.0, rel=1e-3)


def test_apply_hard_filters_rejects_a_meter_serving_more_than_the_building() -> None:
    """Cooling 15x the building's own electricity is a scope problem.

    Every watt of electricity ends up as heat inside the building, so
    electricity bounds internal gains. Q6 verified Fox_education_Theodore
    at 7.93 as a genuine high-outside-air laboratory, which is the anchor
    for the 10.0 limit.
    """
    failures = _filters(cooling_to_electricity=15.08)

    assert len(failures) == 1
    assert "serving more than this building" in failures[0]


def test_apply_hard_filters_accepts_a_verified_high_outside_air_ratio() -> None:
    """Theodore's 7.93 must stay admissible -- Q6 proved it genuine."""
    assert _filters(cooling_to_electricity=7.93) == []


def test_cooling_to_electricity_ratio_uses_annual_totals() -> None:
    cooling = _long_meter(
        [("2016-01-01T00:00:00", "b1", 100.0), ("2016-01-01T01:00:00", "b1", 200.0)]
    )
    electricity = _long_meter(
        [("2016-01-01T00:00:00", "b1", 10.0), ("2016-01-01T01:00:00", "b1", 20.0)]
    )

    ratio = _cooling_to_electricity_ratio(cooling, electricity, 2016)

    assert ratio["b1"] == pytest.approx(10.0)


def test_cooling_to_electricity_ratio_drops_a_zero_denominator() -> None:
    """An undefined ratio must be absent, not infinite."""
    cooling = _long_meter([("2016-01-01T00:00:00", "b1", 100.0)])
    electricity = _long_meter([("2016-01-01T00:00:00", "b1", 0.0)])

    assert "b1" not in _cooling_to_electricity_ratio(cooling, electricity, 2016).index


def test_stability_ignores_a_seasonal_swing_the_weather_explains() -> None:
    """The defect the residual form fixes.

    A building in a hot climate spends its summer far above its annual
    median day. On raw load that scores as an excursion; against the
    weather expectation it scores as nothing, which is correct -- the
    weather explains it.
    """
    weather = _hourly_weather()
    seasonal = pd.Series(50.0 * weather["airTemperature"] + 2000.0, index=weather.index)

    raw = operational_stability(seasonal)
    against_weather = operational_stability(seasonal, seasonal)

    assert raw > 1.0
    assert against_weather < 0.01


def test_stability_still_catches_an_unexplained_regime() -> None:
    """Theodore's September, synthesised: load doubles, weather does not."""
    weather = _hourly_weather()
    expected = pd.Series(50.0 * weather["airTemperature"] + 2000.0, index=weather.index)
    measured = expected.copy()
    event = (measured.index >= "2016-03-01") & (measured.index < "2016-03-12")
    measured[event] = measured[event] * 2.0

    assert operational_stability(measured, expected) > MAX_STABILITY_RATIO


def test_weather_conditional_mean_reproduces_a_binned_function() -> None:
    weather = _hourly_weather()
    load = pd.Series(50.0 * weather["airTemperature"] + 100.0, index=weather.index)

    predicted = weather_conditional_mean(
        load, weather, {"airTemperature": np.arange(-10.0, 50.0, 2.0)}
    )

    assert predicted.index.equals(load.index)
    # Each 2 degC bin is replaced by its mean, so no hour can differ from
    # its bin mean by more than the load span across one whole bin:
    # 50 kW/K * 2 K = 100 kW. (Half that would be the bound only if the
    # temperatures inside every bin were symmetrically distributed, which
    # a diurnal-plus-seasonal signal is not.)
    assert (predicted - load).abs().max() < 100.0
