"""Does a companion meter explain the residual? (Q10, and Q8's method)

    python scripts/check_companion_meter.py

Q10 asks whether Hog_education_Cathleen's U-shaped residual is PHYSICS
-- a balance-point term the model has no shape for -- or METERING, a
chilled-water reading that is not this building's cooling in deep
winter. The two produce an identical residual shape and only one of them
is a reason to change the model, so the question has to be settled
before any structure work begins. Running it produced a THIRD answer
that neither option covered; see `clipping_checks`.

`config/buildings.yaml` records a companion meter for every selected
building: `steam` for Cathleen and Luke, `hotwater` for Claude and
Theodore. If the unexplained cooling rises when the building is also
HEATING, that is evidence for simultaneous heating and cooling, which is
a physical term. If it does not, the winter rise has to be explained
some other way, and a meter fault moves to the front.

TWO THINGS THIS SCRIPT REFUSES TO DO.

  1. It never uses a companion meter's MAGNITUDE. Q8 established that
     the Fox hot-water meter's units are wrong (Claude's mean reads
     3,215 W/m2) and nothing has established that any other companion
     meter is better. Every statistic here is therefore scale-free: rank
     correlations, and a median split that depends only on ordering. A
     monotone unit error changes none of them.
  2. It never touches 2017. `year_residual` is year-agnostic, and this
     script only ever hands it the configured training year -- there is
     no argument that could make it do otherwise.

LUKE IS THE CONTROL, and that is why the script runs every building
rather than the one under investigation. Luke has the same `steam`
companion meter and NO U-shaped residual. If the matched-temperature
steam effect shows up on Luke as well, then whatever it measures is not
what makes Cathleen different, and the Q10 verdict has to be read
accordingly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import (  # noqa: E402
    IncompatibleArtifactError,
    frozen_parameters,
)
from investigate_ushape import year_residual  # noqa: E402
from run_calibration import load_config  # noqa: E402

from cooling_twin.analysis.residual import (  # noqa: E402
    MIN_BIN_COUNT,
    TEMPERATURE_BIN_WIDTH_K,
    band_edges_from_quantiles,
    effective_sample_size,
    matched_band_split,
)
from cooling_twin.data.load import load_meter  # noqa: E402

logger = logging.getLogger("check_companion_meter")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_2_companion_meter.png")

BUILDING_GROUPS = ("primary", "generalisation", "negative_case")

# Minimum share of hours a companion meter must cover before its
# correlations are worth quoting. Below this the comparison is being
# made on a subset that may not represent the year.
MIN_COVERAGE_PCT = 80.0

# A cold-band load whose coefficient of variation is below this is
# suspiciously flat for a real thermal load, and consistent with a meter
# reading a fixed bypass flow rather than the building.
FLAT_LOAD_CV = 0.10

# Hours within this fraction of the year's 1st-percentile load count as
# "sitting on the floor". A bypass or minimum-flow artifact piles hours
# up there; a genuine winter load does not.
FLOOR_TOLERANCE = 0.05

HIGH_COLOUR = "#b2182b"
LOW_COLOUR = "#2171b5"
ZERO_COLOUR = "#525252"


def buildings_with_companions(path: Path) -> list[dict[str, Any]]:
    """Every selected building that declares a companion meter."""
    if not path.exists():
        raise FileNotFoundError(f"building selection not found at {path}")
    selection = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        {
            "building_id": entry["building_id"],
            "site_id": entry["site_id"],
            "role": role,
            "companion_meters": list(entry.get("companion_meters") or []),
        }
        for role in BUILDING_GROUPS
        for entry in (selection.get(role) or [])
        if entry.get("companion_meters")
    ]


def companion_series(
    meter: str, building_id: str, index: pd.DatetimeIndex
) -> npt.NDArray[np.float64]:
    """One companion meter aligned to the residual's index, or NaN.

    Args:
        meter: BDG2 meter name, e.g. `"steam"`.
        building_id: The building.
        index: Timestamps to align to.

    Returns:
        The meter reading per hour, NaN where it has none.

    Raises:
        ValueError: If the building has no rows in that meter file.
    """
    frame = load_meter(meter)
    frame = frame[frame["building_id"] == building_id]
    if frame.empty:
        raise ValueError(f"{building_id} has no {meter} readings")
    series = frame.set_index("timestamp")["meter_reading"].sort_index()
    return series.reindex(index).to_numpy(dtype=float)


def rank_correlation(
    first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]
) -> float:
    """Spearman correlation, computed as Pearson on the ranks.

    Reported alongside Pearson because it is the one that survives a
    wrong unit convention: any monotone rescaling of a companion meter
    leaves the ranks untouched. Written out rather than imported so the
    tie handling is visible -- `scipy.stats.rankdata`'s average method,
    which is what a meter stuck at one value needs.
    """
    from scipy.stats import rankdata

    return float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])


def metering_checks(
    load_kw: npt.NDArray[np.float64],
    cold: npt.NDArray[np.bool_],
    t_ambient_c: npt.NDArray[np.float64],
) -> dict[str, float | bool]:
    """Evidence for the alternative explanation: a meter artifact.

    A genuine winter cooling load varies -- it is driven by something.
    A bypass flow, a shared loop, or a stuck reading does not. Three
    cheap discriminators, none of which needs a companion meter:

      * the coefficient of variation of the cold-band load;
      * how many cold hours sit on the year's floor;
      * whether the cold-band load responds to temperature at all.

    Args:
        load_kw: Measured chilled-water load for the whole year.
        cold: Mask selecting the cold band.
        t_ambient_c: Outdoor dry bulb.

    Returns:
        The three measurements and a combined flag.
    """
    cold_load = load_kw[cold]
    floor = float(np.percentile(load_kw, 1.0))
    on_floor = float(
        np.mean(np.abs(cold_load - floor) <= FLOOR_TOLERANCE * max(abs(floor), 1.0))
    )
    coefficient_of_variation = float(cold_load.std() / abs(cold_load.mean()))
    responds = float(np.corrcoef(cold_load, t_ambient_c[cold])[0, 1])
    return {
        "cold_band_load_cv": coefficient_of_variation,
        "cold_hours_on_annual_floor_pct": on_floor * 100.0,
        "cold_band_corr_temperature": responds,
        "looks_like_a_flat_meter": bool(
            coefficient_of_variation < FLAT_LOAD_CV or on_floor > 0.5
        ),
    }


def clipping_checks(
    predicted_kw: npt.NDArray[np.float64],
    measured_kw: npt.NDArray[np.float64],
    cold: npt.NDArray[np.bool_],
    max_clipped_fraction: float,
) -> dict[str, Any]:
    """The third explanation: the model's own floor.

    `inverse_cooling_load` clips its output at zero, which is physically
    correct -- a cooling coil cannot heat -- but it means the model has a
    hard floor that a real building need not share. If a building keeps a
    weather-insensitive base cooling load through a freezing winter, the
    model's deltaT terms drive its prediction negative, the clip catches
    it at zero, and the residual is FORCED positive by arithmetic rather
    than by any missing physics.

    That mechanism is invisible in a correlation and invisible in a
    binned profile: both show a rising cold arm, which is exactly what a
    missing balance-point term would also show. The distinguishing
    measurement is simply how often the prediction is exactly zero.

    L6.6's objective already penalises clipping beyond
    `max_clipped_fraction`, so a building sitting ON that limit is
    reporting a binding constraint -- the same kind of statement as a
    parameter pinned at its bound (Q7), and it must be read the same way.

    Args:
        predicted_kw: Model output, already clipped at zero.
        measured_kw: Metered load over the same hours.
        cold: Mask selecting the cold band.
        max_clipped_fraction: The objective's allowance.

    Returns:
        Clipped fractions and what the meter was reading meanwhile.
    """
    clipped = predicted_kw <= 0.0
    annual = float(clipped.mean())
    return {
        "clipped_fraction_year": annual,
        "clipped_fraction_cold_band": float(clipped[cold].mean()),
        "objective_max_clipped_fraction": max_clipped_fraction,
        "clipping_constraint_is_binding": bool(annual >= 0.9 * max_clipped_fraction),
        "measured_mean_while_clipped_kw": (
            float(measured_kw[clipped].mean()) if clipped.any() else 0.0
        ),
        "measured_cv_while_clipped": (
            float(measured_kw[clipped].std() / measured_kw[clipped].mean())
            if clipped.any() and measured_kw[clipped].mean() != 0.0
            else 0.0
        ),
    }


def check(
    building: dict[str, Any], config: dict[str, Any], artifacts: Path
) -> dict[str, Any]:
    """Run the companion-meter test for one building, training year only."""
    building_id = building["building_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(
        artifacts, building_id, train_year, names
    )

    series = year_residual(building, config, parameters, train_year)
    residual = series["residual_kw"]
    temperature = series["outdoor_dry_bulb"]
    ratio = effective_sample_size(residual) / residual.size

    lower_edge, _upper = band_edges_from_quantiles(temperature)
    cold = temperature < lower_edge

    record: dict[str, Any] = {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "year": train_year,
        "hours": int(residual.size),
        "cold_band_below_degC": lower_edge,
        "cold_band_hours": int(cold.sum()),
        "cold_band_mean_residual_kw": float(residual[cold].mean()),
        "effective_sample_ratio": ratio,
        "metering_checks": metering_checks(series["measured_kw"], cold, temperature),
        "clipping_checks": clipping_checks(
            series["predicted_kw"],
            series["measured_kw"],
            cold,
            float(config["objective"]["max_clipped_fraction"]),
        ),
        "companions": {},
    }

    for meter in building["companion_meters"]:
        try:
            companion = companion_series(meter, building_id, series["index"])
        except ValueError as error:
            logger.info("%s: %s", building_id, error)
            continue

        usable = np.isfinite(companion)
        coverage = float(usable.mean() * 100.0)
        if coverage < MIN_COVERAGE_PCT:
            logger.warning(
                "%s %s: only %.1f%% coverage, below the %.0f%% floor -- "
                "correlations not computed",
                building_id,
                meter,
                coverage,
                MIN_COVERAGE_PCT,
            )
            record["companions"][meter] = {"coverage_pct": coverage, "usable": False}
            continue

        split = matched_band_split(
            residual[usable],
            temperature[usable],
            companion[usable],
            control="outdoor_dry_bulb",
            probe=meter,
            control_width=TEMPERATURE_BIN_WIDTH_K,
            min_bin_count=MIN_BIN_COUNT,
            effective_sample_ratio=ratio,
        )
        cold_usable = usable & cold
        record["companions"][meter] = {
            "coverage_pct": coverage,
            "usable": True,
            "pearson_all_hours": float(
                np.corrcoef(residual[usable], companion[usable])[0, 1]
            ),
            "spearman_all_hours": rank_correlation(residual[usable], companion[usable]),
            "pearson_cold_band": float(
                np.corrcoef(residual[cold_usable], companion[cold_usable])[0, 1]
            ),
            "spearman_cold_band": rank_correlation(
                residual[cold_usable], companion[cold_usable]
            ),
            "matched_split": {
                "weighted_difference_kw": split.weighted_difference_kw,
                "weighted_difference_sem_kw": split.weighted_difference_sem_kw,
                "probe_raises_residual": split.probe_raises_residual,
                "bins": int(split.centres.size),
                "centres": split.centres.tolist(),
                "differences_kw": split.differences.tolist(),
                "means_low": split.means_low.tolist(),
                "means_high": split.means_high.tolist(),
            },
            "_split": split,
            "_companion": companion,
        }

    record["_series"] = series
    record["_cold"] = cold
    return record


def plot_splits(records: list[dict[str, Any]], path: Path) -> Path:
    """Residual against temperature, split by the companion meter.

    One panel per building-meter. Two lines that lie on top of each
    other mean the companion explains nothing the temperature did not
    already explain; two that separate mean it does.
    """
    panels = [
        (record, meter, payload)
        for record in records
        for meter, payload in record["companions"].items()
        if payload.get("usable")
    ]
    if not panels:
        raise SystemExit("no companion meter had enough coverage to plot")

    figure, axes = plt.subplots(
        1, len(panels), figsize=(4.8 * len(panels), 4.2), squeeze=False
    )
    for axis, (record, meter, payload) in zip(axes[0], panels, strict=True):
        split = payload["_split"]
        axis.plot(
            split.centres,
            split.means_low,
            marker="o",
            markersize=3,
            color=LOW_COLOUR,
            label=f"low {meter} half",
        )
        axis.plot(
            split.centres,
            split.means_high,
            marker="o",
            markersize=3,
            color=HIGH_COLOUR,
            label=f"high {meter} half",
        )
        axis.axvline(
            record["cold_band_below_degC"],
            color=ZERO_COLOUR,
            linestyle=":",
            linewidth=1,
        )
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
        verdict = "RAISES" if split.probe_raises_residual else "no effect"
        axis.set_title(
            f"{record['building_id']}\n{meter}: {verdict} "
            f"({split.weighted_difference_kw:+.0f} "
            f"+/- {split.weighted_difference_sem_kw:.0f} kW)",
            fontsize=9,
        )
        axis.set_xlabel("outdoor dry bulb, degC (dotted: cold band edge)", fontsize=8)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0][0].set_ylabel(
        "residual after the within-bin\ntemperature trend is removed, kW", fontsize=9
    )
    figure.suptitle(
        "Q10 -- does the companion meter explain the residual at MATCHED "
        "outdoor temperature? Training year only."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: run the companion-meter test on every building."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])

    records = []
    for building in buildings_with_companions(arguments.buildings):
        try:
            records.append(check(building, config, artifacts))
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)

    if not records:
        raise SystemExit("no building had both a calibration artifact and a companion")

    correlations = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "meter": meter,
                "cover %": round(payload["coverage_pct"], 1),
                "pearson all": round(payload["pearson_all_hours"], 3),
                "spearman all": round(payload["spearman_all_hours"], 3),
                "pearson cold": round(payload["pearson_cold_band"], 3),
                "spearman cold": round(payload["spearman_cold_band"], 3),
            }
            for record in records
            for meter, payload in record["companions"].items()
            if payload.get("usable")
        ]
    ).set_index(["building", "meter"])
    logger.info(
        "--- 1. raw correlation (CONFOUNDED: both rise as it gets colder) ---\n%s",
        correlations.to_string(),
    )

    matched = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "meter": meter,
                "high-low kW": round(
                    payload["matched_split"]["weighted_difference_kw"], 1
                ),
                "+/- kW": round(
                    payload["matched_split"]["weighted_difference_sem_kw"], 1
                ),
                "bins": payload["matched_split"]["bins"],
                "raises residual?": payload["matched_split"]["probe_raises_residual"],
            }
            for record in records
            for meter, payload in record["companions"].items()
            if payload.get("usable")
        ]
    ).set_index(["building", "meter"])
    logger.info(
        "--- 2. THE TEST: at MATCHED outdoor temperature, with the within-bin "
        "temperature trend removed ---\n%s",
        matched.to_string(),
    )

    artifacts_table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "cold band below": round(record["cold_band_below_degC"], 1),
                "cold hours": record["cold_band_hours"],
                "cold mean resid kW": round(record["cold_band_mean_residual_kw"], 0),
                "cold load CV": round(
                    float(record["metering_checks"]["cold_band_load_cv"]), 3
                ),
                "on floor %": round(
                    float(record["metering_checks"]["cold_hours_on_annual_floor_pct"]), 1
                ),
                "cold corr T": round(
                    float(record["metering_checks"]["cold_band_corr_temperature"]), 3
                ),
                "flat meter?": record["metering_checks"]["looks_like_a_flat_meter"],
            }
            for record in records
        ]
    ).set_index("building")
    logger.info(
        "--- 3. the alternative explanation: does the cold-band load look "
        "like a meter artifact? ---\n%s",
        artifacts_table.to_string(),
    )

    clipping = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "clipped year %": round(
                    record["clipping_checks"]["clipped_fraction_year"] * 100.0, 2
                ),
                "allowed %": round(
                    record["clipping_checks"]["objective_max_clipped_fraction"] * 100.0,
                    2,
                ),
                "clipped cold band %": round(
                    record["clipping_checks"]["clipped_fraction_cold_band"] * 100.0, 1
                ),
                "meter read while clipped kW": round(
                    record["clipping_checks"]["measured_mean_while_clipped_kw"], 0
                ),
                "its CV": round(
                    record["clipping_checks"]["measured_cv_while_clipped"], 2
                ),
                "binding?": record["clipping_checks"]["clipping_constraint_is_binding"],
            }
            for record in records
        ]
    ).set_index("building")
    logger.info(
        "--- 4. the third explanation: the MODEL's floor. The inverse model "
        "clips at zero, so a building with a base load in freezing weather "
        "gets a positive residual by arithmetic ---\n%s",
        clipping.to_string(),
    )

    plot_splits(records, arguments.figure)

    # Deliberately NOT a verdict. Session 018 recorded what happens when a
    # diagnostic table is applied mechanically: `diagnose()` labelled
    # Cathleen "overfitting -> remove capacity", the exact opposite of the
    # correct action. The evidence is printed; the reading is written by
    # hand, into Q10, with the control building in view.
    logger.info(
        "--- how to read this ---\n"
        "  Table 2 is the test. Table 1 is the trap it exists to escape:\n"
        "  a heating meter and a cold-weather residual both rise as it gets\n"
        "  colder, so they correlate whether or not one causes the other.\n"
        "  BULL_EDUCATION_LUKE IS THE CONTROL -- same steam meter, no\n"
        "  U-shaped residual. An effect that appears on Luke as well is not\n"
        "  what makes Cathleen different.\n"
        "  Table 4 is a THIRD explanation that neither Q10 option covered:\n"
        "  a residual forced positive by the model's own clip at zero.\n"
        "  Magnitudes are never used (Q8): the split depends only on the\n"
        "  companion meter's ORDER within each temperature bin."
    )

    out_path = artifacts / "companion_meter_check_2016.json"
    out_path.write_text(
        json.dumps(
            {
                "year": int(config["train_year"]),
                "note": (
                    "Training year only. Companion-meter magnitudes are never "
                    "used -- only rank correlations and a within-bin median "
                    "split (Q8: companion meter units are not trusted)."
                ),
                "buildings": [
                    {
                        key: (
                            {
                                meter: {
                                    k: v
                                    for k, v in payload.items()
                                    if not k.startswith("_")
                                }
                                for meter, payload in value.items()
                            }
                            if key == "companions"
                            else value
                        )
                        for key, value in record.items()
                        if not key.startswith("_")
                    }
                    for record in records
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
