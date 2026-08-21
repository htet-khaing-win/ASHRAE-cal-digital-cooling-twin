"""Everything the dashboard reads or recomputes, and nothing it invents.

`scripts/build_dashboard.py` (the static predecessor this replaces) had
one rule worth keeping: every number on the page traces to an artifact
on disk. This module keeps that rule and extends it to cover what a
static page cannot -- a scrubbable predicted-vs-actual series and a
live setpoint slider need HOURLY numbers no summary JSON stores.

Two kinds of "real" data, both used, never blended silently:

    ARTIFACT NUMBERS   read verbatim from reports/calibration_runs/*.json
                        (train/test CV(RMSE), physics/ML decomposition,
                        conformal coverage). Whatever those files say is
                        what MODULE 06/07/08 measured -- this module does
                        not recompute them, because recomputing a number
                        the project already scored and reporting a
                        possibly-different value under the same name is
                        worse than an honest cache-read.

    LIVE RECOMPUTATION  the hourly predicted load, the plant electric
                        power, and the what-if slider's counterfactual +
                        conformal interval do not exist as saved series
                        anywhere on disk (only their annual summaries do),
                        so they are recomputed here by calling the SAME
                        functions the project's own scripts call --
                        `cooling_twin.twin.counterfactual` and
                        `cooling_twin.twin.uncertainty` -- on the SAME
                        frozen parameters the gate artifact recorded and
                        the SAME BDG2 rows `scripts/run_calibration.py`
                        loads. Nothing here is a new model.

Everything is `st.cache_data`/`st.cache_resource`'d: BDG2 loading and
ODE simulation are the slow steps, and Streamlit re-executes this module
top-to-bottom on every widget interaction.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARTIFACTS_DIR = REPO_ROOT / "reports" / "calibration_runs"
BUILDINGS_PATH = REPO_ROOT / "config" / "buildings.yaml"
CALIBRATION_CONFIG_PATH = REPO_ROOT / "config" / "calibration.yaml"
PLANT_CONFIG_PATH = REPO_ROOT / "config" / "plant.yaml"

# `scripts/*.py` are not a package -- they import each other by relying on
# the caller's sys.path, the same trick `scripts/twin_setup.py` uses on
# itself. One insertion here, at import time, is enough for every
# `from run_calibration import ...` below to resolve exactly the way it
# does when the script is run directly.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from open_test_set import TEST_YEAR, frozen_parameters, load_with_spin_up  # noqa: E402
from run_calibration import load_config, load_training_data  # noqa: E402

from cooling_twin.models.plant import (  # noqa: E402
    PlantParams,
    build_plant_params,
    load_plant_config,
)
from cooling_twin.models.rc import DEFAULT_SUPPLY_HUMIDITY_RATIO  # noqa: E402
from cooling_twin.twin.counterfactual import (  # noqa: E402
    CalibratedTwin,
    Scenario,
    simulate_setpoint_change,
)
from cooling_twin.twin.uncertainty import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BLOCK_HOURS,
    block_bootstrap_ci,
    conformal_interval,
    conformal_quantile,
    normalising_scale,
    time_ordered_split,
)

TRAIN_YEAR = 2016
CALIBRATION_FRACTION = 0.7  # matches scripts/run_counterfactual.py exactly

BANNER_TEXT = (
    "Historical replay on 2016–2017 BDG2 meter and weather data. "
    "No live actuation, no real-time feed, no hardware connection. "
    "Every number below is either read from a run artifact in "
    "reports/calibration_runs/ or recomputed live from that same recorded "
    "data through the project's calibrated (2016) and frozen (2017) "
    "parameters."
)

ROLE_LABELS = {
    "primary": "Primary",
    "generalisation": "Generalisation",
    "negative_case": "Negative case (ADR-014)",
}


# --------------------------------------------------------------------- #
# Static artifacts -- read verbatim, cached for the process lifetime.
# --------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_buildings_config() -> dict[str, list[dict[str, Any]]]:
    """`config/buildings.yaml`, the current (post-gate) role labelling.

    Used for display roles rather than the gate artifact's `role` field:
    the gate artifact is an immutable record of what the buildings were
    CALLED when the gate ran, and Hog was relabelled `negative_case`
    (ADR-014) only after it failed there. Showing the artifact's stale
    label would silently un-flag the negative case this dashboard is
    required to surface.
    """
    if not BUILDINGS_PATH.exists():
        raise FileNotFoundError(f"{BUILDINGS_PATH} not found")
    raw = yaml.safe_load(BUILDINGS_PATH.read_text(encoding="utf-8"))
    return {
        role: [{**entry, "role": role} for entry in (raw.get(role) or [])]
        for role in ("primary", "generalisation", "negative_case")
    }


@st.cache_data(show_spinner=False)
def building_roster() -> list[dict[str, str]]:
    """`[{building_id, site_id, role}]`, one row per selectable building.

    Restricted to buildings that were actually calibrated (i.e. have a
    `calibration_*.json` artifact and a gate record): `Fox_education_Theodore`
    sits in `buildings.yaml`'s `negative_case` list too, but it was
    screened out BEFORE calibration (ADR-012) and has no frozen
    parameters to build a live twin from -- listing it as selectable
    would need a fabricated model to show anything.
    """
    config = load_buildings_config()
    gate = load_artifact("gate_2017_opened.json")
    gated_ids = {record["building_id"] for record in gate.get("buildings", [])}
    return [
        {"building_id": entry["building_id"], "site_id": entry["site_id"], "role": role}
        for role, entries in config.items()
        for entry in entries
        if entry["building_id"] in gated_ids
    ]


@st.cache_data(show_spinner=False)
def load_artifact(name: str) -> dict[str, Any]:
    """One JSON artifact from `reports/calibration_runs/`, by filename.

    Args:
        name: Filename, e.g. `"gate_2017_opened.json"`.

    Returns:
        The parsed JSON.

    Raises:
        FileNotFoundError: If the artifact has not been generated. The
            dashboard must say so on the page rather than fabricate a
            number in its place.
    """
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run the script that produces it "
            f"(see Makefile's `counterfactual` target) before loading the dashboard."
        )
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def artifact_building_record(artifact: dict[str, Any], building_id: str) -> dict[str, Any] | None:
    """Pull one building's entry out of an artifact's `buildings` list."""
    buildings: list[dict[str, Any]] = artifact.get("buildings", [])
    for record in buildings:
        if record["building_id"] == building_id:
            return record
    return None


@st.cache_data(show_spinner=False)
def equifinality_candidates(building_id: str) -> list[dict[str, float]]:
    """Behavioural parameter sets from L6.8's equifinality study, if any.

    Only `Fox_education_Claude` has one on disk (it is the only building
    L6.8 was run for). Returning an empty list for every other building
    is the honest answer, not a placeholder -- `run_counterfactual.py`
    does the same and logs why (see its `parameter_ensemble_summary`).
    """
    candidates: list[dict[str, float]] = []
    for path in sorted(ARTIFACTS_DIR.glob("equifinality_*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("building_id") != building_id:
            continue
        for candidate in record.get("candidates", []):
            if "parameters" in candidate:
                candidates.append(candidate["parameters"])
        if candidates:
            break
    return candidates


# --------------------------------------------------------------------- #
# Live recomputation -- same functions the project's own scripts call.
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class YearSeries:
    """One building-year, measured and predicted, at hourly resolution.

    Attributes:
        index: Timestamp index.
        measured_kw: Meter-derived cooling load.
        predicted_kw: The calibrated twin's cooling load at those same
            hours -- for 2017 this is the FROZEN 2016 parameter set
            re-solved on 2017 weather (spin-up discarded), never refit.
        outdoor_c: Outdoor dry-bulb, for the temperature axis.
        floor_area_m2: Conditioned floor area.
        twin: The `CalibratedTwin`, kept for the what-if panel so the
            slider re-solves the ODE instead of re-loading BDG2 per drag.
        plant: The plant sized from this building-year's measured peak.
    """

    index: pd.DatetimeIndex
    measured_kw: npt.NDArray[np.float64]
    predicted_kw: npt.NDArray[np.float64]
    outdoor_c: npt.NDArray[np.float64]
    floor_area_m2: float
    twin: CalibratedTwin
    plant: PlantParams


@st.cache_resource(show_spinner="Solving the calibrated twin against BDG2 weather...")
def load_year_series(building_id: str, site_id: str, year: int) -> YearSeries:
    """Rebuild one building-year's measured and predicted series.

    Mirrors `scripts/open_test_set.py`'s `evaluate()` exactly for both
    years: 2016 is loaded and solved directly; 2017 is preceded by a
    72-hour spin-up tail of December 2016 (same weather-only warm-up
    the gate itself uses) so the envelope's thermal state has settled
    before the first scored hour, then the spin-up hours are discarded.

    Args:
        building_id: BDG2 identifier.
        site_id: Its BDG2 site.
        year: 2016 (training) or 2017 (held-out test).

    Returns:
        A `YearSeries` covering exactly `year`'s hours.

    Raises:
        FileNotFoundError: If the building has no frozen 2016 parameters.
        ValueError: If `year` is neither 2016 nor 2017, or if cleaning
            leaves no usable rows.
    """
    if year not in (TRAIN_YEAR, TEST_YEAR):
        raise ValueError(f"year must be {TRAIN_YEAR} or {TEST_YEAR}, got {year}")

    config = load_config(CALIBRATION_CONFIG_PATH)
    parameters, _artifact_name = frozen_parameters(ARTIFACTS_DIR, building_id)

    if year == TRAIN_YEAR:
        frame, floor_area_m2 = load_training_data(building_id, site_id, TRAIN_YEAR)
        scored_offset = 0
    else:
        frame, scored_offset = load_with_spin_up(building_id, site_id, config)
        _, floor_area_m2 = load_training_data(building_id, site_id, TRAIN_YEAR)

    twin = CalibratedTwin(
        t_seconds=(frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
        t_ambient_c=frame["airTemperature"].to_numpy(dtype=float),
        humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
        wet_bulb_c=frame["wet_bulb_c"].to_numpy(dtype=float),
        floor_area_m2=floor_area_m2,
        parameters=parameters,
        envelope_capacity_ratio=float(config["fixed"]["envelope_capacity_ratio"]),
        ceiling_height_m=float(config["fixed"]["ceiling_height_m"]),
        supply_humidity_ratio=float(
            config.get("ventilation", {}).get(
                "supply_humidity_ratio", DEFAULT_SUPPLY_HUMIDITY_RATIO
            )
        ),
        building_id=building_id,
        year=year,
    )
    predicted_full = twin.predict_load_kw()

    measured_kw = frame["load_kwh"].to_numpy(dtype=float)[scored_offset:]
    predicted_kw = predicted_full[scored_offset:]
    outdoor_c = frame["airTemperature"].to_numpy(dtype=float)[scored_offset:]
    scored_index = frame.index[scored_offset:]

    # The twin passed to the what-if panel must simulate over exactly the
    # SCORED hours (never the spin-up tail): a scenario re-run including
    # December 2016 would compare 2017 electricity against a baseline
    # that partly is not 2017.
    if scored_offset:
        twin = CalibratedTwin(
            t_seconds=(scored_index - scored_index[0]).total_seconds().to_numpy(dtype=float),
            t_ambient_c=outdoor_c,
            humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float)[scored_offset:],
            wet_bulb_c=frame["wet_bulb_c"].to_numpy(dtype=float)[scored_offset:],
            floor_area_m2=floor_area_m2,
            parameters=parameters,
            envelope_capacity_ratio=twin.envelope_capacity_ratio,
            ceiling_height_m=twin.ceiling_height_m,
            supply_humidity_ratio=twin.supply_humidity_ratio,
            building_id=building_id,
            year=year,
        )

    plant_config = load_plant_config(PLANT_CONFIG_PATH)
    plant = build_plant_params(float(measured_kw.max()), plant_config)

    return YearSeries(
        index=scored_index,
        measured_kw=measured_kw,
        predicted_kw=predicted_kw,
        outdoor_c=outdoor_c,
        floor_area_m2=floor_area_m2,
        twin=twin,
        plant=plant,
    )


@dataclass(frozen=True)
class SetpointWhatIf:
    """One setpoint delta, with every uncertainty source that applies.

    Attributes:
        delta_c: The zone setpoint change simulated, K.
        total_change_pct: Point-estimate change in total plant
            electricity, percent -- `simulate_setpoint_change()`'s own
            number, not re-derived.
        conformal_lower_kw / conformal_upper_kw: Hourly conformal band
            on the SCENARIO cooling load, propagated from the twin's own
            physics residual on training-year calibration hours.
        bootstrap_lower_pct / bootstrap_upper_pct: 90% interval on the
            annual-mean total-electricity change, from a week-block
            bootstrap of the hourly scenario-minus-baseline series.
        parameter_ensemble_pct: `total_change_pct` re-run on every
            behavioural parameter set from L6.8's equifinality study, or
            `None` if the building has no such study.
        n_parameter_sets: How many parameter sets `parameter_ensemble_pct`
            was built from.
    """

    delta_c: float
    total_change_pct: float
    conformal_lower_kw: npt.NDArray[np.float64]
    conformal_upper_kw: npt.NDArray[np.float64]
    scenario_load_kw: npt.NDArray[np.float64]
    bootstrap_lower_pct: float
    bootstrap_upper_pct: float
    parameter_ensemble_pct: list[float] | None
    n_parameter_sets: int


def compute_setpoint_what_if(
    building_id: str,
    year_series: YearSeries,
    delta_c: float,
    alpha: float = DEFAULT_ALPHA,
) -> SetpointWhatIf:
    """Run the zone-setpoint slider through the real M8 machinery.

    Reuses `simulate_setpoint_change()` (L8.2) for the point estimate and
    plant split, `conformal_interval()` (L8.3) for the hourly band, and
    `block_bootstrap_ci()` for the annual-mean band -- the same three
    functions `scripts/run_counterfactual.py` calls, on the same
    TRAINING-year calibration split (ADR-002: the twin is never
    intervened on using the held-out year, regardless of which year the
    dashboard's train/test toggle is showing).

    Args:
        building_id: BDG2 identifier, for the equifinality lookup.
        year_series: The building's training-year twin and plant (2016 --
            callers must pass the 2016 `YearSeries` even when the
            dashboard is displaying 2017, per ADR-002).
        delta_c: Zone setpoint change to simulate, K. 0.0 is the
            no-op baseline scenario.
        alpha: Conformal / bootstrap miss rate.

    Returns:
        A `SetpointWhatIf`.
    """
    scenario = Scenario(
        name=f"zone_setpoint_{delta_c:+.1f}k".replace(".", "_"),
        description=f"Zone setpoint {delta_c:+.1f} K",
        zone_setpoint_delta_c=delta_c,
    )
    result = simulate_setpoint_change(year_series.twin, scenario, year_series.plant)

    calibration, _scored = time_ordered_split(
        year_series.measured_kw.size, CALIBRATION_FRACTION, embargo_hours=DEFAULT_BLOCK_HOURS
    )
    physics_residual = year_series.measured_kw - result.baseline_load_kw
    scale_calibration = normalising_scale(result.baseline_load_kw[calibration])
    quantile = conformal_quantile(physics_residual[calibration], alpha, scale=scale_calibration)
    scale_full = normalising_scale(result.scenario_load_kw)
    n_calibration = len(range(*calibration.indices(year_series.measured_kw.size)))
    interval = conformal_interval(
        result.scenario_load_kw,
        quantile,
        alpha=alpha,
        n_calibration=n_calibration,
        scale=scale_full,
    )

    bootstrap_lo, bootstrap_hi = block_bootstrap_ci(
        result.hourly_total_delta_kw, alpha=alpha, block_hours=DEFAULT_BLOCK_HOURS
    )
    baseline_total_mean = float(result.baseline_plant.total_kw.mean())
    bootstrap_lo_pct = 100.0 * bootstrap_lo / baseline_total_mean if baseline_total_mean else 0.0
    bootstrap_hi_pct = 100.0 * bootstrap_hi / baseline_total_mean if baseline_total_mean else 0.0

    candidates = equifinality_candidates(building_id)
    ensemble_pct: list[float] | None = None
    if candidates and scenario.touches_load:
        ensemble_pct = []
        for params in candidates:
            twin_variant = CalibratedTwin(
                t_seconds=year_series.twin.t_seconds,
                t_ambient_c=year_series.twin.t_ambient_c,
                humidity_ratio=year_series.twin.humidity_ratio,
                wet_bulb_c=year_series.twin.wet_bulb_c,
                floor_area_m2=year_series.twin.floor_area_m2,
                parameters=params,
                envelope_capacity_ratio=year_series.twin.envelope_capacity_ratio,
                ceiling_height_m=year_series.twin.ceiling_height_m,
                supply_humidity_ratio=year_series.twin.supply_humidity_ratio,
                building_id=building_id,
                year=year_series.twin.year,
            )
            variant_result = simulate_setpoint_change(twin_variant, scenario, year_series.plant)
            ensemble_pct.append(variant_result.total_change_pct)

    return SetpointWhatIf(
        delta_c=delta_c,
        total_change_pct=result.total_change_pct,
        conformal_lower_kw=interval.lower,
        conformal_upper_kw=interval.upper,
        scenario_load_kw=result.scenario_load_kw,
        bootstrap_lower_pct=bootstrap_lo_pct,
        bootstrap_upper_pct=bootstrap_hi_pct,
        parameter_ensemble_pct=ensemble_pct,
        n_parameter_sets=len(candidates),
    )
