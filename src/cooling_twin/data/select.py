"""Building selection for the cooling twin project.

Applies the hard filters and soft-preference ranking defined in
04_DATA_CONTRACT.md SS2, and writes the selected buildings plus the
recorded reason for each selection to config/buildings.yaml.

Interface contract with L2.1's load.py (verified against the real file,
not assumed):
    load_metadata(path=METADATA_PATH) -> pd.DataFrame
        indexed by building_id; columns include site_id, primaryspaceusage,
        sqft, yearbuilt, timezone, and one Yes/NaN column per meter type.
    load_meter(meter_type, meters_dir=METERS_RAW_DIR) -> pd.DataFrame
        LONG format: columns timestamp, building_id, meter_reading.
    list_buildings_with_meter(metadata_df, meter_type) -> list[str]
        reads the metadata Yes/NaN flag column, not the raw meter file.

BDG2 reports floor area in sqft; the data contract (04_DATA_CONTRACT.md SS6)
specifies floor_area_m2, so the conversion happens once, here, at the
boundary -- nothing downstream ever sees sqft.

KNOWN LIMITATION -- read before rerunning from scratch:
This script's hard filters and soft scoring run purely against metadata.csv
and the meter completeness fraction. They do NOT run the L2.4 timezone
cross-correlation gate or a stuck-sensor check -- those require joining
weather data and are performed separately (see weather.py). Two exclusions
below (EXCLUDED_SITES, EXCLUDED_BUILDING_IDS) were discovered that way,
AFTER an initial run of this script, and are hardcoded here so a fresh run
does not silently reselect a combination already known to fail those later
checks. This is a stopgap, not a fix: a rerun of this script alone still
cannot discover a *new* timezone or stuck-sensor problem in a *different*
candidate. Folding L2.4 into this script's ranking loop is a reasonable
M9 (production packaging) task, not done here to keep L2.2 and L2.4 as
separate, teachable steps. See 07_PROGRESS.md ADR-004, ADR-005.

Run:
    PYTHONPATH=src python3 -m cooling_twin.data.select
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

# Layering note: the data layer importing one pure function from the
# calibration layer is deliberate. `cvrmse` has no data-layer dependency,
# and two definitions of CV(RMSE) in one repository -- one for selecting
# buildings, one for grading them -- is a worse problem than the import
# direction.
from cooling_twin.calibration.metrics import cvrmse
from cooling_twin.data.load import (
    BDG2_ROOT,
    METADATA_PATH,
    METERS_RAW_DIR,
    list_buildings_with_meter,
    load_metadata,
    load_meter,
)
from cooling_twin.data.quality import load_cleaning_config, run_cleaning_pipeline
from cooling_twin.data.weather import add_psychrometric_features, load_weather

logger = logging.getLogger(__name__)

# --- Unit conversion -----------------------------------------------------
# BDG2 metadata ships floor area in sqft; the data contract requires m2.
# Exact factor, not a rounded approximation.
SQFT_TO_M2 = 0.09290304

# --- Hard filter thresholds ------------------------------------------------
# Every number here traces back to 04_DATA_CONTRACT.md SS2.
MAX_MISSING_FRACTION = 0.10  # missing data < 10% per meter
REQUIRED_YEARS = (2016, 2017)

# The screen runs on this year alone. 2017 is the held-out test set
# (ADR-002): a building chosen because its 2017 load is predictable
# would be a building chosen using 2017.
TRAIN_YEAR = 2016

# Mean cooling intensity a real building can plausibly show, W/m2.
#
# This filter exists because CV(RMSE) -- and therefore the whole
# explainability screen -- is SCALE-INVARIANT: a meter reporting in the
# wrong unit is exactly as "explainable" as a correct one, and the
# screen will happily rank it first. Measured across the 521 BDG2
# buildings with a 2016 chilledwater series, the median site runs
# 0-99 W/m2, and the Eagle site runs a median of 24,306 W/m2 -- about
# 400x anything physical, across every one of its ~90 buildings. That
# is a site-level unit convention, not a building load.
#
# 1,000 W/m2 is deliberately generous: it admits a dense data hall,
# which no BDG2 education building approaches. 5 W/m2 excludes meters
# that barely register, where there is no cooling system to model.
# Q6's evidence anchors the plausible band: Fox's laboratories run a
# median 161.6 W/m2 and its non-laboratories 74.1 W/m2.
MIN_INTENSITY_W_PER_M2 = 5.0
MAX_INTENSITY_W_PER_M2 = 1000.0

# Annual chilled-water thermal energy divided by the building's own
# electricity. A METER SCOPE check, not an efficiency one.
#
# Every watt of electricity a building draws ends up as heat inside it,
# so electricity is an upper bound on internal gains. Cooling far in
# excess of it must arrive through the envelope or as outside air --
# possible, and real for a high-outside-air laboratory, but there is a
# limit past which the arithmetic says the meter is not measuring this
# building alone.
#
# ANCHOR: Fox_education_Theodore sits at 7.93, and Q6 established with
# four independent lines of evidence that its high load is genuine. A
# verified high-outside-air laboratory therefore sits under 8, and 10
# leaves headroom above the hardest verified case. The measured
# distribution over 491 buildings has a median of 6.62.
#
# HONESTY NOTE: unlike the thresholds above, this one was added AFTER a
# selection run -- the calibration attempt on Bull_office_Anne (ratio
# 15.08) produced a fitted constant of 290 kW against a 13.8 W/m2
# electricity draw, which is not interpretable as a building's internal
# gain. The threshold is set from the Theodore anchor rather than from
# the list of buildings it excludes, but the amendment is recorded in
# ADR-012 either way.
MAX_COOLING_TO_ELECTRICITY_RATIO = 10.0

# --- Known exclusions from post-selection validation ------------------------
# These were NOT caught by the hard filters above -- they only surfaced
# after running L2.4's timezone cross-correlation gate and a stuck-sensor
# check against the top-ranked candidates (a step this script does not
# perform itself; see the module docstring). Recorded here so a fresh run
# of this script does not silently reproduce a selection already known to
# fail downstream. See 07_PROGRESS.md ADR-004 / ADR-005.
EXCLUDED_SITES: set[str] = {
    "Moose",  # ADR-005: 3/3 independently-tested Moose buildings failed
              # the L2.4 timezone gate with consistent negative peak lag
              # (-5h, -4h, -2h) and an identical 684-row weather join gap
              # -- a site-level weather file fault, not a per-building one.
}
EXCLUDED_BUILDING_IDS: set[str] = {
    "Hog_office_Darline",  # ADR-004: empirically flattest load shape
                            # candidate (cv=0.054), but its 7h identical-
                            # value run is a C3 stuck-sensor artifact, not
                            # genuine equipment-driven load.
}

# --- Retained negative cases (ADR-012) --------------------------------------
# Buildings the screen REJECTS that are kept in config/buildings.yaml
# anyway, with their rejection reasons, because they carry findings the
# project reports.
#
# This is the honesty mechanism for a mid-project re-selection. Choosing
# new buildings after the old ones failed is selection-on-outcome, and
# the standard critique -- "you calibrated the buildings that were easy
# to calibrate" -- is unanswerable if the failures quietly disappear
# from the repository. Keeping the rejected building, its numbers, and
# the reason it was rejected is what makes the re-selection auditable
# rather than convenient.
NEGATIVE_CASE_BUILDING_IDS: tuple[str, ...] = (
    "Fox_education_Theodore",  # M6 was run to completion here: CV(RMSE)
                               # 42.25%, beaten by a 2-parameter linear
                               # regression, an unexplained 11-day load
                               # doubling in September 2016, and a
                               # weather-explainability ceiling of 38.3%
                               # that no model structure could beat.
)

# --- Soft preference weights ------------------------------------------------
PREFERRED_USES = {"Office", "Education"}
SOFT_WEIGHT_PREFERRED_USE = 2.0
SOFT_WEIGHT_YEAR_BUILT_PRESENT = 1.0
SOFT_WEIGHT_CLEANLINESS = 1.0  # proxy for "few flatline/zero streaks"

# --- Screen thresholds (ADR-012) --------------------------------------------
#
# The v1 criteria above score data COMPLETENESS. They never asked whether
# the load is EXPLAINABLE by the drivers a cooling twin actually has, and
# the two turn out to be nearly unrelated: Fox_education_Theodore has 0.0%
# missing data and sits at the 49th percentile of explainability, which is
# why M6 stalled at CV(RMSE) 42% against a 30% gate.
#
# Every threshold below was set from the measured distribution over the
# 379 screenable BDG2 buildings BEFORE any candidate list was read, and is
# recorded in ADR-012 -- so the screen is pre-registered rather than tuned
# until a preferred building appears.

# The G14 hourly acceptance limit. A building whose ceiling is above this
# cannot pass the gate no matter how good the model is, so admitting one
# is committing to a failure.
MAX_CEILING_CVRMSE_PCT = 30.0

# Largest 11-day excursion from the WEATHER EXPECTATION, as a fraction
# of the median day's load. See `operational_stability` for why the
# residual and not the raw load.
#
# The measured separation is wide enough that the exact cut hardly
# matters: Theodore's September regime scores 1.66, while every other
# building examined -- including two whose RAW statistic exceeded the
# old 1.6 limit purely because they sit in hot climates -- lands between
# 0.09 and 0.30. 0.5 sits in the empty gap between those groups.
#
#   Fox_education_Theodore  ceiling 38.3%  raw 3.17  residual 1.66  REJECT
#   Fox_education_Claude    ceiling 10.0%  raw 1.84  residual 0.19  admit
#   Hog_public_Brad         ceiling 11.3%  raw 1.69  residual 0.09  admit
#   Bull_education_Luke     ceiling 13.7%  raw 1.46  residual 0.10  admit
#
# HONESTY NOTE: the raw-load version of this statistic was the original
# pre-registered one, and it was replaced after a run -- it rejected
# Claude and Brad, whose excursions the weather fully explains, and in
# doing so collapsed the portfolio to a single site. The replacement
# measures what the failure message always claimed to measure. Recorded
# in ADR-012.
MAX_STABILITY_RATIO = 0.5
STABILITY_WINDOW_DAYS = 11

# Hours of usable data needed before either statistic means anything.
MIN_SCREEN_HOURS = 8000
MIN_SCORED_HOURS = 500

# The M3 gate's own threshold (06_ASSESSMENT.md): a building whose
# cleaning removes more than this is not a calibration candidate. Applied
# INSIDE the screen because a stuck sensor scores an excellent ceiling --
# see `_screen_one`.
MAX_REMOVED_FRACTION = 0.15

# Humidity ratio above which an hour is counted as "humid" -- the
# threshold at which latent load stops being negligible for a coil
# leaving air near 12.8 degC (03_DOMAIN_REFERENCE.md SS3).
HUMID_HUMIDITY_RATIO = 0.012
HUMID_SITE_MIN_FRAC = 0.40
DRY_SITE_MAX_FRAC = 0.10

# The screen's weights. Explainability dominates deliberately: it is the
# criterion whose absence caused the M6 setback.
SOFT_WEIGHT_EXPLAINABILITY = 3.0
SOFT_WEIGHT_STABILITY = 1.0
SOFT_WEIGHT_COMPANION_METER = 0.5

# Companion meters worth points. A heating meter alongside chilled water
# is what makes simultaneous heating and cooling DETECTABLE -- the exact
# signature found at Theodore in September 2016.
COMPANION_METERS = ("steam", "hotwater", "water")


@dataclass(frozen=True)
class ScreenMetrics:
    """What the ADR-012 screen measures about one building-year.

    Attributes:
        ceiling_cvrmse_pct: Held-out weather-explainability ceiling --
            the best CV(RMSE) ANY model driven by these weather
            variables could reach on this building. Model structure and
            optimiser effort cannot beat it.
        stability_ratio: Largest `STABILITY_WINDOW_DAYS` rolling mean
            divided by the annual daily median. 1.0 is a building with
            no operational excursion; Theodore's September event is 3.17.
        humid_hours_frac: Fraction of hours above `HUMID_HUMIDITY_RATIO`
            at this building's site. Drives the climate-diversity
            constraint, and decides which buildings can exercise a
            latent-load term at all.
        companion_meters: Which of `COMPANION_METERS` this building has.
        screened_hours: Usable hours the metrics were computed from.
    """

    ceiling_cvrmse_pct: float
    stability_ratio: float
    humid_hours_frac: float
    companion_meters: tuple[str, ...]
    screened_hours: int


@dataclass(frozen=True)
class BuildingCandidate:
    """One building's filter result and soft score, with reasons recorded."""

    building_id: str
    site_id: str
    primary_use: str
    floor_area_m2: float
    year_built: int | None
    missing_frac_chilledwater: float
    missing_frac_electricity: float
    passed_hard_filters: bool
    hard_filter_failures: list[str] = field(default_factory=list)
    soft_score: float = 0.0
    soft_reasons: list[str] = field(default_factory=list)
    sub_use: str = ""
    screen: ScreenMetrics | None = None


def _missing_fraction_by_building(
    meter_long: pd.DataFrame, years: tuple[int, ...]
) -> pd.Series:
    """Fraction of missing hourly readings per building, across given years.

    meter_long is long-format (timestamp, building_id, meter_reading) from
    load_meter(). melt() preserves one row per (timestamp, building_id) pair
    that existed in the wide CSV's grid, with NaN where a reading is absent
    -- so isna().mean() per building over the year-filtered rows is exactly
    the missing fraction.

    Args:
        meter_long: Long-format meter DataFrame.
        years: Years to include in the denominator.

    Returns:
        Series indexed by building_id, missing fraction in [0, 1]. A
        building entirely absent from meter_long simply won't appear here --
        callers must handle that with a fallback default.
    """
    year_mask = meter_long["timestamp"].dt.year.isin(years)
    subset = meter_long.loc[year_mask, ["building_id", "meter_reading"]].copy()
    subset["is_missing"] = subset["meter_reading"].isna()
    return subset.groupby("building_id")["is_missing"].mean()


def _year_presence_by_building(meter_long: pd.DataFrame, year: int) -> pd.Series:
    """Whether each building has at least one non-null reading in `year`."""
    year_mask = meter_long["timestamp"].dt.year == year
    subset = meter_long.loc[year_mask, ["building_id", "meter_reading"]].copy()
    subset["is_present"] = subset["meter_reading"].notna()
    return subset.groupby("building_id")["is_present"].any()


def _mean_intensity_by_building(
    meter_long: pd.DataFrame, metadata: pd.DataFrame, year: int
) -> pd.Series:
    """Mean cooling intensity per building, W/m2, for one year.

    Floor area comes from `sqft`, converted here -- the same source and
    the same conversion `floor_area_m2` uses elsewhere in this module.
    Deriving it from the `sqm` column instead would let a building be
    filtered on one area and reported with another if the two ever
    disagree.

    Args:
        meter_long: Long-format chilledwater readings.
        metadata: Indexed by building_id, with an `sqft` column.
        year: Year to average over.

    Returns:
        Series indexed by building_id. Buildings with no floor area or
        no readings are absent, and callers must treat that as a
        failure rather than as a pass.
    """
    subset = meter_long[meter_long["timestamp"].dt.year == year]
    mean_kw = subset.groupby("building_id")["meter_reading"].mean()
    area_m2 = metadata["sqft"].reindex(mean_kw.index) * SQFT_TO_M2
    return (1000.0 * mean_kw / area_m2.replace(0.0, np.nan)).dropna()


def _cooling_to_electricity_ratio(
    cw_long: pd.DataFrame, elec_long: pd.DataFrame, year: int
) -> pd.Series:
    """Annual chilled-water energy divided by electricity, per building.

    Args:
        cw_long: Long-format chilledwater readings.
        elec_long: Long-format electricity readings.
        year: Year to total over.

    Returns:
        Series indexed by building_id. Buildings with no electricity, or
        with a zero total on either meter, are absent -- the ratio is
        undefined there, and callers must treat absence as a failure.
    """
    cooling = cw_long[cw_long["timestamp"].dt.year == year].groupby("building_id")[
        "meter_reading"
    ].sum()
    electricity = elec_long[elec_long["timestamp"].dt.year == year].groupby("building_id")[
        "meter_reading"
    ].sum()
    paired = pd.concat([cooling.rename("cooling"), electricity.rename("electricity")], axis=1)
    paired = paired[(paired["electricity"] > 0) & (paired["cooling"] > 0)]
    return paired["cooling"] / paired["electricity"]


def _apply_hard_filters(
    building_id: str,
    site_id: str,
    primary_use: object,
    floor_area_m2: float,
    missing_cw: float,
    missing_elec: float,
    has_2016: bool,
    has_2017: bool,
    intensity_w_per_m2: float | None = None,
    cooling_to_electricity: float | None = None,
) -> list[str]:
    """Return failed hard filters for one building; empty list = passed.

    Checks the known-exclusion lists (EXCLUDED_SITES, EXCLUDED_BUILDING_IDS)
    first, before the data-contract filters, so a rerun of this script
    cannot silently reselect a building or site already known to fail the
    downstream L2.4 timezone gate or a stuck-sensor check.
    """
    failures: list[str] = []
    if site_id in EXCLUDED_SITES:
        failures.append(f"site {site_id} excluded -- see ADR-005 in 07_PROGRESS.md")
    if building_id in EXCLUDED_BUILDING_IDS:
        failures.append(
            f"building {building_id} excluded -- see ADR-004 in 07_PROGRESS.md"
        )
    if missing_cw > MAX_MISSING_FRACTION:
        failures.append(f"chilledwater missing {missing_cw:.1%} > {MAX_MISSING_FRACTION:.0%}")
    if missing_elec > MAX_MISSING_FRACTION:
        failures.append(f"electricity missing {missing_elec:.1%} > {MAX_MISSING_FRACTION:.0%}")
    if pd.isna(floor_area_m2):
        failures.append("floor_area_m2 missing (sqft absent in metadata)")
    if not primary_use or (isinstance(primary_use, float) and pd.isna(primary_use)):
        failures.append("primaryspaceusage missing")
    if not (has_2016 and has_2017):
        failures.append("data not present in both 2016 and 2017")
    if intensity_w_per_m2 is not None:
        # `None` means "not measured" -- used by unit tests that exercise
        # the other filters in isolation. score_candidates always passes
        # a value, so a real run is never unchecked.
        if pd.isna(intensity_w_per_m2):
            failures.append("mean cooling intensity not computable (no readings or area)")
        elif not (
            MIN_INTENSITY_W_PER_M2 <= intensity_w_per_m2 <= MAX_INTENSITY_W_PER_M2
        ):
            failures.append(
                f"mean cooling intensity {intensity_w_per_m2:,.0f} W/m2 outside "
                f"[{MIN_INTENSITY_W_PER_M2:.0f}, {MAX_INTENSITY_W_PER_M2:.0f}] -- "
                "a unit convention or metering error, not a building load"
            )
    if cooling_to_electricity is not None:
        if pd.isna(cooling_to_electricity):
            failures.append("cooling/electricity ratio not computable (a meter totals zero)")
        elif cooling_to_electricity > MAX_COOLING_TO_ELECTRICITY_RATIO:
            failures.append(
                f"cooling/electricity ratio {cooling_to_electricity:.2f} > "
                f"{MAX_COOLING_TO_ELECTRICITY_RATIO:.1f} -- the chilled-water meter "
                "is probably serving more than this building, so its floor area "
                "is the wrong normaliser for every per-area parameter"
            )
    return failures


def _soft_score(
    primary_use: object, year_built: int | None, missing_cw: float
) -> tuple[float, list[str]]:
    """Compute the soft preference score and the reason behind each point."""
    score = 0.0
    reasons: list[str] = []

    if primary_use in PREFERRED_USES:
        score += SOFT_WEIGHT_PREFERRED_USE
        reasons.append(f"primary_use={primary_use} (simple occupancy pattern)")

    if year_built is not None:
        score += SOFT_WEIGHT_YEAR_BUILT_PRESENT
        reasons.append(f"year_built={year_built} present")

    # Real flatline detection doesn't exist until L3.1-L3.3. Missing-hours
    # fraction is used as an explicit, named proxy in the meantime.
    cleanliness_bonus = SOFT_WEIGHT_CLEANLINESS * (1.0 - missing_cw / MAX_MISSING_FRACTION)
    score += cleanliness_bonus
    reasons.append(
        f"chilledwater missing only {missing_cw:.1%} "
        "(cleanliness proxy, not a real flatline check yet)"
    )

    return score, reasons


def weather_explainability_ceiling(
    load_kw: pd.Series,
    drivers: pd.DataFrame,
    bin_edges: Mapping[str, Sequence[float] | npt.NDArray[np.float64]],
    min_scored_hours: int = MIN_SCORED_HOURS,
) -> float:
    """Best CV(RMSE) any model on these drivers could reach for this building.

    Bins the drivers, fits the conditional mean of the load in each bin
    on ODD days, and scores those means on EVEN days. Because the
    conditional mean is the minimiser of squared error, and because it
    is fitted and scored on disjoint days, the result is a held-out
    estimate of the floor that no model structure and no amount of
    optimiser effort can go below.

    Why this belongs in SELECTION and not in diagnosis: a building whose
    ceiling is above the G14 limit cannot pass the gate, so admitting it
    to the portfolio is committing to a failure before any modelling
    starts. That is exactly what happened with
    `Fox_education_Theodore` -- 0.0% missing data, and a ceiling of
    38.3% against a 30% gate.

    IMPORTANT -- this does NOT replace the M3 quality rules, and running
    it alone is unsafe. A flatlined meter is trivially predictable and
    therefore scores an EXCELLENT ceiling: the best score in the whole
    BDG2 screen (4.8%) belongs to `Hog_office_Darline`, the building
    ADR-004 excluded for a C3 stuck-sensor artifact. Run
    `detect_flatlines()` and `validate_physical_bounds()` first, or this
    metric will recommend broken meters.

    Args:
        load_kw: Measured load, indexed by tz-aware hourly timestamps.
        drivers: Driver columns aligned to `load_kw`'s index.
        bin_edges: Bin edges per driver column. A driver absent here is
            ignored.
        min_scored_hours: Refuse to report a ceiling from fewer scored
            hours than this.

    Returns:
        CV(RMSE) percent. Lower is better.

    Raises:
        ValueError: If `drivers` has no usable column, or if fewer than
            `min_scored_hours` hours can be scored -- a ceiling from a
            handful of hours is noise, and returning it silently would
            let a building with almost no data top the ranking.
    """
    columns = [name for name in bin_edges if name in drivers.columns]
    if not columns:
        raise ValueError(
            f"none of the binned drivers {list(bin_edges)} are present in "
            f"drivers ({list(drivers.columns)})"
        )

    frame = drivers[columns].join(load_kw.rename("load"), how="inner").dropna()
    if frame.empty:
        raise ValueError("no overlapping hours between load and drivers")

    keys = [
        pd.cut(frame[name], np.asarray(bin_edges[name], dtype=float), include_lowest=True)
        for name in columns
    ]
    # Odd days fit, even days score. Splitting by DAY rather than by
    # random hour matters: adjacent hours are ~0.99 correlated, so a
    # random hourly split would put a point's own neighbours in the
    # training set and report a ceiling far below the truth.
    is_scored = frame.index.dayofyear % 2 == 0
    fitted = frame[~is_scored].groupby([key[~is_scored] for key in keys], observed=True)[
        "load"
    ].mean()

    scored_keys = [key[is_scored] for key in keys]
    index = pd.MultiIndex.from_arrays(scored_keys) if len(keys) > 1 else scored_keys[0]
    predicted = fitted.reindex(index).to_numpy()
    measured = frame.loc[is_scored, "load"].to_numpy()

    covered = np.isfinite(predicted)
    if int(covered.sum()) < min_scored_hours:
        raise ValueError(
            f"only {int(covered.sum())} hours could be scored (need "
            f"{min_scored_hours}); the bins are too fine or the series too "
            "short for this ceiling to mean anything"
        )
    return cvrmse(measured[covered], predicted[covered], n_params=0)


def weather_conditional_mean(
    load_kw: pd.Series,
    drivers: pd.DataFrame,
    bin_edges: Mapping[str, Sequence[float] | npt.NDArray[np.float64]],
) -> pd.Series:
    """Load predicted from binned weather alone, fitted on all hours.

    The companion to `weather_explainability_ceiling`: that function
    holds days out to estimate a floor, this one fits everything to
    produce the best weather-only explanation of each hour. In-sample is
    correct here -- the output is a baseline to subtract, not a
    performance claim.

    Args:
        load_kw: Measured load, indexed by hourly timestamps.
        drivers: Driver columns aligned to `load_kw`'s index.
        bin_edges: Bin edges per driver column.

    Returns:
        Predicted load, indexed like the overlapping hours.

    Raises:
        ValueError: If no binned driver is present in `drivers`.
    """
    columns = [name for name in bin_edges if name in drivers.columns]
    if not columns:
        raise ValueError(
            f"none of the binned drivers {list(bin_edges)} are present in "
            f"drivers ({list(drivers.columns)})"
        )
    frame = drivers[columns].join(load_kw.rename("load"), how="inner").dropna()
    keys = [
        pd.cut(frame[name], np.asarray(bin_edges[name], dtype=float), include_lowest=True)
        for name in columns
    ]
    return frame.groupby(keys, observed=True)["load"].transform("mean")


def operational_stability(
    load_kw: pd.Series,
    weather_explained_kw: pd.Series | None = None,
    window_days: int = STABILITY_WINDOW_DAYS,
) -> float:
    """Largest sustained UNEXPLAINED excursion, relative to the median day.

    Catches the signature that broke the M6 calibration: an operational
    regime that appears for days or weeks and then disappears. At
    Theodore, 11 days in September carried 9.5 CV(RMSE) points from 5%
    of the hours, while the building's own electricity did not move.

    Measured on the RESIDUAL when `weather_explained_kw` is given, and
    this is the part that matters. On raw load the statistic cannot tell
    a July peak from a broken valve: a strongly seasonal building spends
    its summer well above its annual median day and is penalised for
    being in a hot climate. Fox_education_Claude scores 1.84 on raw load
    yet has a weather-explainability ceiling of 10.0% -- its excursion
    IS the weather. Differencing against what the weather explains
    leaves exactly the regime changes the failure message claims to
    describe.

    A rolling window rather than a per-day maximum, because one hot day
    is weather; a fortnight at double load is not.

    Args:
        load_kw: Measured load, indexed by hourly timestamps.
        weather_explained_kw: Prediction from weather alone, aligned to
            `load_kw`. When `None`, the statistic falls back to raw
            load and therefore mixes season with excursion -- useful
            only for a quick look, never for a gate.
        window_days: Excursion length to look for.

    Returns:
        Ratio >= 0. 1.0 means the worst window departs from the weather
        expectation by as much as a typical day's load, which is
        enormous; a well-behaved building is well under 0.5.

    Raises:
        ValueError: If the series covers fewer than `window_days` days,
            or its median is not positive (a mostly-zero meter would
            otherwise divide by ~0 and report an enormous ratio).
    """
    daily = load_kw.resample("D").mean().dropna()
    if len(daily) < window_days:
        raise ValueError(
            f"need at least {window_days} days to measure a {window_days}-day "
            f"excursion, got {len(daily)}"
        )
    median = float(daily.median())
    if median <= 0.0:
        raise ValueError(
            "daily median load is not positive -- this meter is mostly zero "
            "and the C3/C4 rules should have caught it before the screen"
        )

    if weather_explained_kw is None:
        return float(daily.rolling(window_days, center=True).mean().max() / median)

    residual = (load_kw - weather_explained_kw).dropna()
    daily_residual = residual.resample("D").mean().dropna()
    if len(daily_residual) < window_days:
        raise ValueError(
            f"need at least {window_days} days of residual, got {len(daily_residual)}"
        )
    worst = float(daily_residual.rolling(window_days, center=True).mean().abs().max())
    return worst / median


def _screen_one(
    building_id: str,
    site_id: str,
    metadata: pd.DataFrame,
    load_kw: pd.Series,
    site_weather: pd.DataFrame,
    cleaning_config: dict[str, object],
) -> ScreenMetrics:
    """Compute the screen metrics for one building-year, on CLEANED load.

    The cleaning pipeline runs FIRST, and that ordering is not a detail.
    A stuck sensor is trivially predictable, so on raw data it scores a
    near-perfect explainability ceiling and a perfect stability ratio --
    it looks like the best building in the dataset. This was not
    hypothetical: the first run of this screen ranked
    `Fox_education_Gloria` top on raw data, and the M3 rules then found
    its meter stuck at 2315.8281 kW for runs of up to 503 hours across
    eight months of 2016.

    Raises:
        ValueError: If cleaning removes more than `MAX_REMOVED_FRACTION`
            of the year, or too few hours survive to measure anything.
    """
    temperature = site_weather["airTemperature"]
    humidity = site_weather["humidity_ratio"]

    cleaned, _ = run_cleaning_pipeline(
        load_kw, site_weather["rh_pct"].reindex(load_kw.index), cleaning_config
    )
    removed_fraction = float(cleaned.isna().mean())
    if removed_fraction > MAX_REMOVED_FRACTION:
        raise ValueError(
            f"{building_id}: M3 cleaning removes {removed_fraction:.1%} of the "
            f"year (> {MAX_REMOVED_FRACTION:.0%}) -- stuck sensors and gaps, "
            "not a building this screen can rank"
        )

    drivers = pd.DataFrame(
        {"airTemperature": temperature, "humidity_ratio": humidity}
    ).dropna()
    aligned = drivers.join(cleaned.rename("load"), how="inner").dropna()
    aligned = aligned[aligned["load"] > 0]
    if len(aligned) < MIN_SCREEN_HOURS:
        raise ValueError(
            f"{building_id}: only {len(aligned)} usable hours after the "
            f"weather join (need {MIN_SCREEN_HOURS})"
        )

    # 2 degC temperature bins and humidity quintiles: fine enough to
    # follow a cooling curve, coarse enough that every bin keeps enough
    # hours to average. Quantile edges rather than fixed ones because
    # humidity ratio spans two orders of magnitude between sites.
    t_edges = np.arange(
        np.floor(aligned["airTemperature"].min()) - 1.0,
        np.ceil(aligned["airTemperature"].max()) + 2.0,
        2.0,
    )
    w_edges = np.unique(np.quantile(aligned["humidity_ratio"], np.linspace(0.0, 1.0, 6)))
    w_edges[0] -= 1e-9
    w_edges[-1] += 1e-9

    bin_edges = {"airTemperature": t_edges, "humidity_ratio": w_edges}
    drivers_only = aligned[["airTemperature", "humidity_ratio"]]
    ceiling = weather_explainability_ceiling(aligned["load"], drivers_only, bin_edges)
    stability = operational_stability(
        aligned["load"],
        weather_conditional_mean(aligned["load"], drivers_only, bin_edges),
    )
    humid_fraction = float((humidity > HUMID_HUMIDITY_RATIO).mean())

    row = metadata.loc[building_id] if building_id in metadata.index else pd.Series(dtype=object)
    companions = tuple(
        meter for meter in COMPANION_METERS if pd.notna(row.get(meter, float("nan")))
    )
    logger.debug(
        "%s (%s): ceiling %.1f%%, stability %.2f, humid hours %.0f%%",
        building_id,
        site_id,
        ceiling,
        stability,
        100 * humid_fraction,
    )
    return ScreenMetrics(
        ceiling_cvrmse_pct=ceiling,
        stability_ratio=stability,
        humid_hours_frac=humid_fraction,
        companion_meters=companions,
        screened_hours=len(aligned),
    )


def _screen_soft_score(metrics: ScreenMetrics) -> tuple[float, list[str]]:
    """Points and reasons contributed by the ADR-012 screen."""
    score = 0.0
    reasons: list[str] = []

    headroom = (MAX_CEILING_CVRMSE_PCT - metrics.ceiling_cvrmse_pct) / MAX_CEILING_CVRMSE_PCT
    score += SOFT_WEIGHT_EXPLAINABILITY * headroom
    reasons.append(
        f"weather-explainability ceiling {metrics.ceiling_cvrmse_pct:.1f}% "
        f"(G14 limit {MAX_CEILING_CVRMSE_PCT:.0f}% -- headroom {100 * headroom:.0f}%)"
    )

    steadiness = (MAX_STABILITY_RATIO - metrics.stability_ratio) / MAX_STABILITY_RATIO
    score += SOFT_WEIGHT_STABILITY * max(steadiness, 0.0)
    reasons.append(
        f"worst {STABILITY_WINDOW_DAYS}-day excursion {metrics.stability_ratio:.2f}x "
        "the median day"
    )

    if metrics.companion_meters:
        score += SOFT_WEIGHT_COMPANION_METER * len(metrics.companion_meters)
        reasons.append(
            f"companion meters {', '.join(metrics.companion_meters)} "
            "(makes simultaneous heating/cooling detectable)"
        )

    reasons.append(f"site humid hours {100 * metrics.humid_hours_frac:.0f}%")
    return score, reasons


def screen_candidates(
    candidates: list[BuildingCandidate],
    year: int,
    metadata_path: Path = METADATA_PATH,
    meters_dir: Path = METERS_RAW_DIR,
    weather_root: Path = BDG2_ROOT,
) -> list[BuildingCandidate]:
    """Enrich hard-filter survivors with the ADR-012 screen and re-score.

    Buildings that fail the screen have `passed_hard_filters` set to
    False with the reason recorded, so a rejection stays as auditable as
    a selection.

    The screen runs on `year` ONLY. Computing it on the held-out year
    would leak the test set (ADR-002) -- a building chosen because its
    2017 load is predictable is a building chosen using 2017.

    Args:
        candidates: Output of `score_candidates()`.
        year: Training year to screen on.
        metadata_path: Path to metadata.csv.
        meters_dir: Directory containing the raw meter CSVs.
        weather_root: BDG2 root, for the site weather files.

    Returns:
        A new list; input candidates are not mutated.
    """
    metadata = load_metadata(metadata_path)
    chilled = load_meter("chilledwater", meters_dir)
    chilled = chilled[chilled["timestamp"].dt.year == year]

    weather = add_psychrometric_features(load_weather(weather_root))
    weather = weather[weather["timestamp"].dt.year == year]
    weather_by_site = {
        site: frame.set_index("timestamp")[
            ["airTemperature", "humidity_ratio", "rh_pct"]
        ].sort_index()
        for site, frame in weather.groupby("site_id")
    }
    cleaning_config = load_cleaning_config()
    load_by_building = {
        building_id: frame.set_index("timestamp")["meter_reading"].sort_index()
        for building_id, frame in chilled.groupby("building_id")
    }

    screened: list[BuildingCandidate] = []
    for candidate in candidates:
        if not candidate.passed_hard_filters:
            screened.append(candidate)
            continue

        sub_use = (
            str(metadata.loc[candidate.building_id, "sub_primaryspaceusage"])
            if candidate.building_id in metadata.index
            else ""
        )

        site_weather = weather_by_site.get(candidate.site_id)
        load_kw = load_by_building.get(candidate.building_id)
        if site_weather is None or load_kw is None:
            screened.append(
                replace(
                    candidate,
                    passed_hard_filters=False,
                    hard_filter_failures=[
                        *candidate.hard_filter_failures,
                        f"no {year} weather/meter data to screen against",
                    ],
                    sub_use=sub_use,
                )
            )
            continue

        try:
            metrics = _screen_one(
                candidate.building_id,
                candidate.site_id,
                metadata,
                load_kw,
                site_weather,
                cleaning_config,
            )
        except ValueError as error:
            screened.append(
                replace(
                    candidate,
                    passed_hard_filters=False,
                    hard_filter_failures=[*candidate.hard_filter_failures, str(error)],
                    sub_use=sub_use,
                )
            )
            continue

        failures = []
        if metrics.ceiling_cvrmse_pct > MAX_CEILING_CVRMSE_PCT:
            failures.append(
                f"explainability ceiling {metrics.ceiling_cvrmse_pct:.1f}% > "
                f"{MAX_CEILING_CVRMSE_PCT:.0f}% -- no model on these drivers can "
                "pass G14 for this building"
            )
        if metrics.stability_ratio > MAX_STABILITY_RATIO:
            failures.append(
                f"worst {STABILITY_WINDOW_DAYS}-day excursion "
                f"{metrics.stability_ratio:.2f}x median > {MAX_STABILITY_RATIO}x -- "
                "an unexplained operational regime no weather model reproduces"
            )

        screen_score, screen_reasons = _screen_soft_score(metrics)
        screened.append(
            replace(
                candidate,
                passed_hard_filters=not failures,
                hard_filter_failures=[*candidate.hard_filter_failures, *failures],
                soft_score=candidate.soft_score + (0.0 if failures else screen_score),
                soft_reasons=[*candidate.soft_reasons, *screen_reasons],
                sub_use=sub_use,
                screen=metrics,
            )
        )

    survivors = sum(candidate.passed_hard_filters for candidate in screened)
    logger.info("Screen (%d): %d candidates survive", year, survivors)
    return screened


def score_candidates(
    metadata_path: Path = METADATA_PATH,
    meters_dir: Path = METERS_RAW_DIR,
) -> list[BuildingCandidate]:
    """Run every metadata row through the hard filters and soft scoring.

    Args:
        metadata_path: Path to metadata.csv.
        meters_dir: Directory containing the raw meter CSVs.

    Returns:
        Every candidate, passed or not -- rejections stay auditable too.
    """
    metadata = load_metadata(metadata_path)
    cw_long = load_meter("chilledwater", meters_dir)
    elec_long = load_meter("electricity", meters_dir)

    cw_buildings = set(list_buildings_with_meter(metadata, "chilledwater"))
    elec_buildings = set(list_buildings_with_meter(metadata, "electricity"))

    missing_cw_by_building = _missing_fraction_by_building(cw_long, REQUIRED_YEARS)
    missing_elec_by_building = _missing_fraction_by_building(elec_long, REQUIRED_YEARS)
    has_2016_by_building = _year_presence_by_building(cw_long, 2016)
    has_2017_by_building = _year_presence_by_building(cw_long, 2017)
    intensity_by_building = _mean_intensity_by_building(cw_long, metadata, TRAIN_YEAR)
    ratio_by_building = _cooling_to_electricity_ratio(cw_long, elec_long, TRAIN_YEAR)

    candidates: list[BuildingCandidate] = []
    for building_id, row in metadata.iterrows():
        primary_use = row.get("primaryspaceusage")
        year_built_raw = row.get("yearbuilt")
        year_built = int(year_built_raw) if pd.notna(year_built_raw) else None
        sqft = row.get("sqft")
        floor_area_m2 = float(sqft) * SQFT_TO_M2 if pd.notna(sqft) else float("nan")
        site_id = row.get("site_id", "")

        if building_id not in cw_buildings or building_id not in elec_buildings:
            missing_meters = [
                m for m, present in
                (("chilledwater", building_id in cw_buildings),
                 ("electricity", building_id in elec_buildings))
                if not present
            ]
            candidates.append(
                BuildingCandidate(
                    building_id=building_id,
                    site_id=site_id,
                    primary_use=primary_use or "",
                    floor_area_m2=floor_area_m2,
                    year_built=year_built,
                    missing_frac_chilledwater=1.0,
                    missing_frac_electricity=1.0,
                    passed_hard_filters=False,
                    hard_filter_failures=[
                        f"{m} meter absent (metadata flag)" for m in missing_meters
                    ],
                )
            )
            continue

        missing_cw = float(missing_cw_by_building.get(building_id, 1.0))
        missing_elec = float(missing_elec_by_building.get(building_id, 1.0))
        has_2016 = bool(has_2016_by_building.get(building_id, False))
        has_2017 = bool(has_2017_by_building.get(building_id, False))

        failures = _apply_hard_filters(
            building_id, site_id, primary_use, floor_area_m2,
            missing_cw, missing_elec, has_2016, has_2017,
            intensity_w_per_m2=float(intensity_by_building.get(building_id, float("nan"))),
            cooling_to_electricity=float(ratio_by_building.get(building_id, float("nan"))),
        )
        passed = len(failures) == 0
        soft_score, soft_reasons = (
            (0.0, []) if not passed else _soft_score(primary_use, year_built, missing_cw)
        )

        candidates.append(
            BuildingCandidate(
                building_id=building_id,
                site_id=site_id,
                primary_use=primary_use or "",
                floor_area_m2=floor_area_m2,
                year_built=year_built,
                missing_frac_chilledwater=missing_cw,
                missing_frac_electricity=missing_elec,
                passed_hard_filters=passed,
                hard_filter_failures=failures,
                soft_score=soft_score,
                soft_reasons=soft_reasons,
            )
        )

    logger.info(
        "Scored %d candidates, %d passed hard filters",
        len(candidates),
        sum(c.passed_hard_filters for c in candidates),
    )
    return candidates


def select_buildings(
    candidates: list[BuildingCandidate],
    n_generalisation: int = 2,
    require_distinct_sites: bool = True,
) -> dict[str, list[BuildingCandidate]]:
    """Rank passing candidates and split into primary / generalisation roles.

    Args:
        candidates: Output of score_candidates().
        n_generalisation: How many additional buildings to select.
        require_distinct_sites: If True, prefer generalisation buildings from
            sites other than the primary's and each other's, since the role
            exists to test cross-site transfer (04_DATA_CONTRACT.md SS2).
            Falls back to same-site candidates, with a logged warning, only
            if too few distinct sites pass the hard filters.

    Returns:
        {"primary": [...], "generalisation": [...]}

    Raises:
        ValueError: If fewer than 1 + n_generalisation candidates pass.
    """
    passing = sorted(
        (c for c in candidates if c.passed_hard_filters),
        # Deterministic tie-break: soft_score desc, then the continuous
        # cleanliness proxy asc (finer-grained than the rounded score),
        # then building_id asc as a final, arbitrary-but-stable key.
        # Omitting a key here is what let ties fall back to metadata.csv's
        # row order -- which is grouped by site -- last time.
        key=lambda c: (-c.soft_score, c.missing_frac_chilledwater, c.building_id),
    )
    required = 1 + n_generalisation
    if len(passing) < required:
        raise ValueError(
            f"Only {len(passing)} candidates passed hard filters; "
            f"need at least {required} (1 primary + {n_generalisation} generalisation)."
        )

    primary = passing[0]
    generalisation: list[BuildingCandidate] = []
    used_sites = {primary.site_id}

    for c in passing[1:]:
        if len(generalisation) >= n_generalisation:
            break
        if require_distinct_sites and c.site_id in used_sites:
            continue
        generalisation.append(c)
        used_sites.add(c.site_id)

    if len(generalisation) < n_generalisation:
        chosen_ids = {primary.building_id} | {c.building_id for c in generalisation}
        remaining = [c for c in passing[1:] if c.building_id not in chosen_ids]
        shortfall = n_generalisation - len(generalisation)
        logger.warning(
            "Only %d distinct sites available among passing candidates; "
            "filling %d generalisation slot(s) with same-site buildings. "
            "This weakens the cross-site transfer claim -- record it in "
            "07_PROGRESS.md.",
            len(used_sites),
            shortfall,
        )
        generalisation.extend(remaining[:shortfall])

    selected = [primary, *generalisation]
    _warn_if_climate_undiverse(selected)
    return {"primary": [primary], "generalisation": generalisation}


def _warn_if_climate_undiverse(selected: list[BuildingCandidate]) -> None:
    """Warn when the selection cannot support the humid-vs-dry comparison.

    A portfolio drawn entirely from dry sites can never demonstrate that
    the latent term matters, and one drawn entirely from humid sites
    cannot show that it is safe to omit. The check warns rather than
    raises: the ranking is the primary criterion, and silently
    reordering buildings to satisfy a climate quota would make the
    selection unreproducible from the scores alone.
    """
    fractions = [
        candidate.screen.humid_hours_frac
        for candidate in selected
        if candidate.screen is not None
    ]
    if not fractions:
        return
    if not any(fraction >= HUMID_SITE_MIN_FRAC for fraction in fractions):
        logger.warning(
            "No selected building sits at a humid site (>= %.0f%% of hours above "
            "w=%.3f). The latent-load work (ADR-011) and the humid-vs-dry "
            "comparison cannot be demonstrated on this portfolio.",
            100 * HUMID_SITE_MIN_FRAC,
            HUMID_HUMIDITY_RATIO,
        )
    if not any(fraction <= DRY_SITE_MAX_FRAC for fraction in fractions):
        logger.warning(
            "No selected building sits at a dry site (<= %.0f%% humid hours); "
            "the comparison has no control arm.",
            100 * DRY_SITE_MAX_FRAC,
        )


def collect_negative_cases(
    candidates: list[BuildingCandidate],
    building_ids: tuple[str, ...] = NEGATIVE_CASE_BUILDING_IDS,
) -> list[BuildingCandidate]:
    """Pull the retained negative cases out of a screened candidate list.

    Their `hard_filter_failures` are the screen's own words for why they
    were rejected, so the record cannot drift from the criteria that
    produced it.

    Args:
        candidates: Output of `screen_candidates()`.
        building_ids: Which buildings to retain.

    Returns:
        The matching candidates, in `building_ids` order. A named
        building that is not in `candidates` is skipped with a warning
        rather than raising -- a missing negative case must not be able
        to break a selection run.
    """
    by_id = {candidate.building_id: candidate for candidate in candidates}
    retained = []
    for building_id in building_ids:
        candidate = by_id.get(building_id)
        if candidate is None:
            logger.warning(
                "negative case %s is not among the scored candidates; it will "
                "be absent from config/buildings.yaml",
                building_id,
            )
            continue
        if candidate.passed_hard_filters:
            logger.warning(
                "negative case %s now PASSES the screen -- either the data or "
                "the thresholds changed. Re-read ADR-012 before shipping this.",
                building_id,
            )
        retained.append(candidate)
    return retained


def write_buildings_yaml(selection: dict[str, list[BuildingCandidate]], out_path: Path) -> None:
    """Write config/buildings.yaml with the selected buildings and their reasons.

    Args:
        selection: Output of select_buildings(), optionally with a
            `negative_case` role added from `collect_negative_cases()`.
        out_path: Destination, e.g. config/buildings.yaml.
    """
    payload: dict[str, object] = {}
    for role, items in selection.items():
        payload[role] = [
            {
                "building_id": c.building_id,
                "site_id": c.site_id,
                "primary_use": c.primary_use,
                "sub_use": c.sub_use,
                "floor_area_m2": round(c.floor_area_m2, 1),
                "missing_pct_chilledwater": round(c.missing_frac_chilledwater * 100, 2),
                "soft_score": round(c.soft_score, 2),
                **(
                    {}
                    if c.passed_hard_filters
                    else {"rejected_because": c.hard_filter_failures}
                ),
                **(
                    {}
                    if c.screen is None
                    else {
                        "ceiling_cvrmse_pct": round(c.screen.ceiling_cvrmse_pct, 1),
                        "stability_ratio": round(c.screen.stability_ratio, 2),
                        "humid_hours_pct": round(100 * c.screen.humid_hours_frac, 1),
                        "companion_meters": list(c.screen.companion_meters),
                    }
                ),
                "reasons": c.soft_reasons,
            }
            for c in items
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Stage 1: metadata + completeness (v1 criteria, unchanged).
    candidates = score_candidates()
    # Stage 2: the ADR-012 screen, on the TRAINING year only.
    candidates = screen_candidates(candidates, year=TRAIN_YEAR)
    selection = select_buildings(candidates, n_generalisation=2)
    # Retained rejections travel with the selection -- see ADR-012.
    selection["negative_case"] = collect_negative_cases(candidates)
    write_buildings_yaml(selection, Path("config/buildings.yaml"))

    for role, items in selection.items():
        for c in items:
            screen = c.screen
            detail = (
                ""
                if screen is None
                else (
                    f", ceiling={screen.ceiling_cvrmse_pct:.1f}%"
                    f", stability={screen.stability_ratio:.2f}"
                    f", humid={100 * screen.humid_hours_frac:.0f}%"
                )
            )
            print(f"{role}: {c.building_id} (site={c.site_id}, score={c.soft_score:.2f}{detail})")