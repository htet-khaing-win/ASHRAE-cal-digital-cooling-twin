"""Is the U-shaped residual real, and is it on both years? (L7.1c/L7.2)

    python scripts/investigate_ushape.py --reread-test-set

L7.1b found the training-year residual rising at BOTH ends of the
outdoor temperature range and falling in the middle, on Claude and on
Cathleen. Three things follow, and this script answers them in order:

  1. Is the shape a shape, or is it two noisy end bins? -- the quadratic
     term and the three band means, `analysis/residual.py`.
  2. Does it appear on the held-out year too? -- the question that
     separates a STRUCTURAL fault from an artefact of the fitted year.
     A fault present on both years in equal measure cannot be
     overfitting; that reasoning is ADR-014's, applied to a shape rather
     than to a seasonal bias.
  3. Is what remains learnable, or is it noise? -- ACF and Ljung-Box,
     pre-registering L7.2's `residual_diagnostics()`.

ON THE TEST YEAR -- read before running.

  2017 was opened once, at L6.10, and ADR-002 permits that once. This
  script READS IT AGAIN. That is a re-read, not a held-out result, and
  every 2017 number it prints is labelled `re-read` in the output and in
  the artifact. Two things make the re-read defensible rather than a
  quiet second opening:

    * NO model, parameter or bound decision may be taken from it. The
      2016 numbers decide what to build; 2017 only answers "is the same
      shape there?", which selects nothing.
    * Nothing here can fit. The optimiser is not imported, and the
      parameters are read frozen from the 2016 artifacts.

  The access must still be logged in 07_PROGRESS.md's Test Set Access
  Log. The flag is required so that logging it is a deliberate act.

Band edges are derived on the TRAINING year and reused for the test
year. Recomputing them per year would move the bands with the weather,
and comparing two years across bands that are not the same bands answers
nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import (  # noqa: E402
    IncompatibleArtifactError,
    frozen_parameters,
    selected_buildings,
)
from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)

from cooling_twin.analysis.residual import (  # noqa: E402
    CurvatureFit,
    ResidualDiagnostics,
    ResidualProfile,
    autocorrelation,
    band_edges_from_quantiles,
    decompose_residual,
    fit_residual_curvature,
    residual_diagnostics,
)
from cooling_twin.calibration.crossval import DEFAULT_SPIN_UP_HOURS  # noqa: E402
from cooling_twin.calibration.metrics import ashrae_g14_pass  # noqa: E402

logger = logging.getLogger("investigate_ushape")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
TEST_YEAR = 2017

SHAPE_FIGURE_PATH = Path("reports/figures/l7_1c_ushape_both_years.png")
ACF_FIGURE_PATH = Path("reports/figures/l7_2_residual_acf.png")

# The three drivers under investigation, with the profile name they carry
# in `decompose_residual` and the units the curvature is reported in.
INVESTIGATED = (
    ("outdoor_dry_bulb", "degC"),
    ("humidity_ratio", "g/kg"),
    ("predicted_load", "kW"),
)

TRAIN_COLOUR = "#2171b5"
TEST_COLOUR = "#b2182b"
BAND_COLOUR = "#f0f0f0"
ZERO_COLOUR = "#525252"

# Lags for the ACF panel: a full week of hourly lags.
ACF_MAX_LAG = 168


def year_residual(
    building: dict[str, str],
    config: dict[str, Any],
    parameters: dict[str, float],
    year: int,
) -> dict[str, Any]:
    """Residual and drivers for one building-year, at frozen parameters.

    The SAME function serves both years. Two code paths -- one for the
    training year and one for the test year -- would be the obvious
    shape, and any difference between them would show up as a difference
    between the years, which is exactly the thing being measured.

    For the test year the simulation starts `DEFAULT_SPIN_UP_HOURS`
    before 1 January using training-year drivers, and those hours are
    discarded before anything is computed. This is the L6.9/L6.10
    treatment, kept identical so the residual here is the residual the
    gate scored.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        parameters: Frozen calibrated parameters.
        year: Year to simulate.

    Returns:
        A mapping with the index, measured, predicted and residual
        series, the two weather drivers, and the G14 verdict.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])

    frame, floor_area_m2 = load_training_data(building_id, site_id, year)
    offset = 0
    if year != train_year:
        spin_up_frame, _ = load_training_data(building_id, site_id, train_year)
        spin_up = spin_up_frame.tail(DEFAULT_SPIN_UP_HOURS)
        frame = pd.concat([spin_up, frame]).sort_index()
        offset = len(spin_up)

    objective = CalibrationObjective(
        (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
        frame["airTemperature"].to_numpy(dtype=float),
        frame["load_kwh"].to_numpy(dtype=float),
        floor_area_m2,
        config,
        outdoor_humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
    )
    vector = np.array([parameters[name] for name in names], dtype=float)
    simulated, _raw = objective.predict(vector)

    predicted = simulated[offset:]
    measured = objective.observed_kw[offset:]
    index = frame.index[offset:]
    verdict = ashrae_g14_pass(measured, predicted, n_params=len(names))

    return {
        "year": year,
        "index": index,
        "measured_kw": measured,
        "predicted_kw": predicted,
        "residual_kw": measured - predicted,
        "outdoor_dry_bulb": frame["airTemperature"].to_numpy(dtype=float)[offset:],
        "humidity_ratio": frame["humidity_ratio"].to_numpy(dtype=float)[offset:] * 1000.0,
        "verdict": verdict,
    }


def curvature_row(
    building_id: str,
    series: dict[str, Any],
    fit: CurvatureFit,
    profile: ResidualProfile,
    is_reread: bool,
) -> dict[str, Any]:
    """One line of the curvature table, per building-year-driver.

    `turn at` and `mass below` exist because the band verdict alone
    misled the eye once already. The three bands are MASS-weighted, so an
    arm that turns up over a handful of extreme hours cannot lift its
    band and the verdict correctly reads `False` -- while the plotted
    profile visibly turns up and invites the opposite conclusion. Both
    readings are true and they answer different questions, so both are
    printed: where the residual bottoms out, and what share of the year
    lies on the far side of it.
    """
    low, middle, high = fit.band_means_kw
    turning_bin = int(np.argmin(profile.means))
    return {
        "building": building_id,
        "year": f"{series['year']}{' (re-read)' if is_reread else ''}",
        "driver": fit.driver,
        "low band": round(low, 0),
        "mid band": round(middle, 0),
        "high band": round(high, 0),
        "low lift": round(fit.low_lift_kw, 0),
        "high lift": round(fit.high_lift_kw, 0),
        "quad": round(fit.quadratic_kw_per_unit2, 3),
        "vertex": None if fit.vertex is None else round(fit.vertex, 1),
        "R2 quad": round(fit.r_squared, 3),
        "R2 line": round(fit.linear_r_squared, 3),
        "turn at": round(float(profile.centres[turning_bin]), 1),
        "mass below %": round(
            100.0
            * float(profile.counts[:turning_bin].sum())
            / float(profile.counts.sum()),
            1,
        ),
        "U?": fit.is_u_shaped,
    }


def investigate(
    building: dict[str, str], config: dict[str, Any], artifacts: Path
) -> dict[str, Any]:
    """Run all four questions for one building, on both years."""
    building_id = building["building_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(
        artifacts, building_id, train_year, names
    )

    series_by_year = {
        year: year_residual(building, config, parameters, year)
        for year in (train_year, TEST_YEAR)
    }

    # Derived on the TRAINING year, then reused. See the module docstring.
    edges = {
        driver: band_edges_from_quantiles(
            _driver_values(series_by_year[train_year], driver)
        )
        for driver, _unit in INVESTIGATED
    }

    record: dict[str, Any] = {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "parameters": parameters,
        "band_edges_from_year": train_year,
        "band_edges": {driver: list(pair) for driver, pair in edges.items()},
        "years": {},
    }
    fits: dict[int, dict[str, CurvatureFit]] = {}
    decompositions = {}

    for year, series in series_by_year.items():
        is_reread = year != train_year
        year_fits = {
            driver: fit_residual_curvature(
                series["residual_kw"],
                _driver_values(series, driver),
                driver=driver,
                band_edges=edges[driver],
            )
            for driver, _unit in INVESTIGATED
        }
        fits[year] = year_fits

        decompositions[year] = decompose_residual(
            series["index"],
            series["measured_kw"],
            series["predicted_kw"],
            t_ambient_c=series["outdoor_dry_bulb"],
            humidity_ratio_kg_per_kg=series["humidity_ratio"] / 1000.0,
            label=f"{building_id} {year}{' re-read' if is_reread else ''}",
        )
        diagnostics = residual_diagnostics(
            series["residual_kw"],
            label=f"{building_id} {year}{' re-read' if is_reread else ''}",
        )

        record["years"][str(year)] = {
            "is_reread_of_held_out_year": is_reread,
            "hours": int(series["residual_kw"].size),
            "cvrmse_pct": series["verdict"].cvrmse_pct,
            "nmbe_pct": series["verdict"].nmbe_pct,
            "passed": series["verdict"].passed,
            "curvature": {
                driver: _fit_record(fit) for driver, fit in year_fits.items()
            },
            "diagnostics": _diagnostics_record(diagnostics),
        }

    record["u_shape_on_both_years"] = all(
        fits[year]["outdoor_dry_bulb"].is_u_shaped for year in series_by_year
    )
    return record | {
        "_series": series_by_year,
        "_decompositions": decompositions,
        "_fits": fits,
    }


def _driver_values(series: dict[str, Any], driver: str) -> npt.NDArray[np.float64]:
    """The driver array behind a profile name."""
    if driver == "predicted_load":
        return np.asarray(series["predicted_kw"], dtype=float)
    return np.asarray(series[driver], dtype=float)


def _fit_record(fit: CurvatureFit) -> dict[str, Any]:
    """Flatten a `CurvatureFit` for JSON."""
    return {
        "quadratic_kw_per_unit2": fit.quadratic_kw_per_unit2,
        "linear_kw_per_unit": fit.linear_kw_per_unit,
        "vertex": fit.vertex,
        "r_squared": fit.r_squared,
        "linear_r_squared": fit.linear_r_squared,
        "band_edges": list(fit.band_edges),
        "band_means_kw": list(fit.band_means_kw),
        "band_sems_kw": list(fit.band_sems_kw),
        "band_counts": list(fit.band_counts),
        "low_lift_kw": fit.low_lift_kw,
        "high_lift_kw": fit.high_lift_kw,
        "is_u_shaped": fit.is_u_shaped,
    }


def _diagnostics_record(diagnostics: ResidualDiagnostics) -> dict[str, Any]:
    """Flatten a `ResidualDiagnostics` for JSON."""
    return {
        "acf": {str(lag): value for lag, value in diagnostics.acf.items()},
        "ljung_box_q": diagnostics.ljung_box_q,
        "ljung_box_lags": diagnostics.ljung_box_lags,
        "ljung_box_p": diagnostics.ljung_box_p,
        "daily_variance_share": diagnostics.daily_variance_share,
        "white_noise_variance_share": diagnostics.white_noise_variance_share,
        "survives_daily_averaging": diagnostics.survives_daily_averaging,
    }


def plot_both_years(records: list[dict[str, Any]], path: Path) -> Path:
    """Binned residual against three drivers, both years overlaid.

    Overlaid rather than side by side. Two adjacent panels would let a
    difference in y-scale read as a difference in the residual, and the
    whole question here is whether the two years carry the SAME shape.
    """
    n_rows, n_cols = len(records), len(INVESTIGATED)
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.4 * n_cols, 3.2 * n_rows), sharey="row", squeeze=False
    )

    for row, record in enumerate(records):
        for column, (driver, unit) in enumerate(INVESTIGATED):
            axis = axes[row][column]
            verdicts: list[str] = []
            lower, upper = record["band_edges"][driver]
            # The bands are shaded, not drawn as lines, because they are
            # regions of the driver -- and shading the OUTER two makes
            # the middle band read as the reference it is.
            axis.axvspan(-1e9, lower, color=BAND_COLOUR, zorder=0)
            axis.axvspan(upper, 1e9, color=BAND_COLOUR, zorder=0)

            for year, colour in (
                (min(record["_decompositions"]), TRAIN_COLOUR),
                (max(record["_decompositions"]), TEST_COLOUR),
            ):
                profile = record["_decompositions"][year].profiles[driver]
                fit = record["_fits"][year][driver]
                is_reread = record["years"][str(year)]["is_reread_of_held_out_year"]
                axis.errorbar(
                    profile.centres,
                    profile.means,
                    yerr=profile.sems,
                    marker="o",
                    markersize=3,
                    linewidth=1.6,
                    capsize=2,
                    color=colour,
                    label=f"{year}{' (re-read)' if is_reread else ''}",
                )
                verdicts.append("U" if fit.is_u_shaped else "no U")

            axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
            axis.set_xlim(
                min(
                    float(d.profiles[driver].centres.min())
                    for d in record["_decompositions"].values()
                ),
                max(
                    float(d.profiles[driver].centres.max())
                    for d in record["_decompositions"].values()
                ),
            )
            # The verdict goes in the title, not the legend: it is a
            # property of the panel, and burying it in a colour key is
            # how a reader ends up reading the shape instead of the test.
            axis.set_title(
                f"{driver}   2016: {verdicts[0]}   2017: {verdicts[1]}",
                fontsize=9,
                color=TEST_COLOUR if "no U" not in verdicts else "black",
            )
            axis.set_xlabel(f"{driver}, {unit}", fontsize=9)
            axis.grid(alpha=0.3, zorder=1)
            axis.tick_params(labelsize=8)
            axis.legend(fontsize=8, loc="best")
        axes[row][0].set_ylabel(f"{record['building_id']}\nresidual, kW", fontsize=9)

    figure.suptitle(
        "Does the residual's shape survive the held-out year?\n"
        "Shaded = outer terciles of the 2016 driver, reused unchanged for "
        "2017.  Above zero = the model under-predicts.\n"
        "2017 is a RE-READ of an already-opened test set (ADR-002), not a "
        "held-out result.",
        fontsize=11,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def plot_acf(records: list[dict[str, Any]], path: Path) -> Path:
    """Autocorrelation to one week of lags, per building, both years."""
    figure, axes = plt.subplots(
        1, len(records), figsize=(4.6 * len(records), 3.6), sharey=True, squeeze=False
    )
    lags = tuple(range(1, ACF_MAX_LAG + 1))

    for column, record in enumerate(records):
        axis = axes[0][column]
        for year, colour in (
            (min(record["_series"]), TRAIN_COLOUR),
            (max(record["_series"]), TEST_COLOUR),
        ):
            acf = autocorrelation(record["_series"][year]["residual_kw"], lags)
            is_reread = record["years"][str(year)]["is_reread_of_held_out_year"]
            axis.plot(
                list(acf),
                list(acf.values()),
                linewidth=1.4,
                color=colour,
                label=f"{year}{' (re-read)' if is_reread else ''}",
            )
        for marker in (24, 48, 168):
            axis.axvline(marker, color=ZERO_COLOUR, linestyle=":", linewidth=0.8)
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_title(record["building_id"], fontsize=10)
        axis.set_xlabel("lag, hours (dotted: 24, 48, 168)", fontsize=9)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0][0].set_ylabel("autocorrelation of the residual", fontsize=9)

    figure.suptitle(
        "L7.2 -- is the residual white noise? White noise sits on zero at "
        "every lag."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: curvature and diagnostics, both years, every building."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--shape-figure", type=Path, default=SHAPE_FIGURE_PATH)
    parser.add_argument("--acf-figure", type=Path, default=ACF_FIGURE_PATH)
    parser.add_argument(
        "--reread-test-set",
        action="store_true",
        help="Required. Reads the 2017 data, which ADR-002 opened once "
        "already. Every number from it is a re-read, and the access must "
        "be logged in 07_PROGRESS.md.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    if not arguments.reread_test_set:
        raise SystemExit(
            f"refusing to run without --reread-test-set. This script reads "
            f"the {TEST_YEAR} data, which ADR-002 permits opening once and "
            "which was opened at L6.10. Running it again is a RE-READ: "
            "nothing it produces is a held-out result, and no model or "
            "parameter decision may be taken from it. Pass the flag only "
            "when that is what you mean to do, and log the access."
        )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    logger.warning(
        "RE-READING the held-out year (%d). Not a second opening: the "
        "optimiser is not imported and no decision may be taken from these "
        "numbers. Log this access in 07_PROGRESS.md.",
        TEST_YEAR,
    )

    records = []
    for building in selected_buildings(arguments.buildings):
        try:
            records.append(investigate(building, config, artifacts))
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)

    if not records:
        raise SystemExit("no building had a calibration artifact to analyse")

    curvature = pd.DataFrame(
        [
            curvature_row(
                record["building_id"],
                record["_series"][year],
                fit,
                record["_decompositions"][year].profiles[driver],
                record["years"][str(year)]["is_reread_of_held_out_year"],
            )
            for record in records
            for year in sorted(record["_series"])
            for driver, fit in record["_fits"][year].items()
        ]
    ).set_index(["building", "driver", "year"])
    logger.info(
        "--- 1-3. curvature and bands, both years (kW; lift = band minus "
        "middle band) ---\n%s",
        curvature.sort_index().to_string(),
    )

    verdicts = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "U on 2016": record["years"][str(train_year)]["curvature"][
                    "outdoor_dry_bulb"
                ]["is_u_shaped"],
                "U on 2017 (re-read)": record["years"][str(TEST_YEAR)]["curvature"][
                    "outdoor_dry_bulb"
                ]["is_u_shaped"],
                "both": record["u_shape_on_both_years"],
                "2016 CV %": round(record["years"][str(train_year)]["cvrmse_pct"], 2),
                "2017 CV %": round(record["years"][str(TEST_YEAR)]["cvrmse_pct"], 2),
            }
            for record in records
        ]
    ).set_index("building")
    logger.info("--- 3. is the U on BOTH years? ---\n%s", verdicts.to_string())

    diagnostics_table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "year": year,
                "rho(1)": round(record["years"][str(year)]["diagnostics"]["acf"]["1"], 3),
                "rho(24)": round(
                    record["years"][str(year)]["diagnostics"]["acf"]["24"], 3
                ),
                "rho(168)": round(
                    record["years"][str(year)]["diagnostics"]["acf"]["168"], 3
                ),
                "LB Q": round(record["years"][str(year)]["diagnostics"]["ljung_box_q"]),
                "LB p": f"{record['years'][str(year)]['diagnostics']['ljung_box_p']:.1e}",
                "daily var share": round(
                    record["years"][str(year)]["diagnostics"]["daily_variance_share"], 3
                ),
                "vs white noise 0.042": record["years"][str(year)]["diagnostics"][
                    "survives_daily_averaging"
                ],
            }
            for record in records
            for year in sorted(record["_series"])
        ]
    ).set_index(["building", "year"])
    logger.info(
        "--- 4. L7.2 pre-registered: is the residual random? ---\n%s",
        diagnostics_table.to_string(),
    )

    plot_both_years(records, arguments.shape_figure)
    plot_acf(records, arguments.acf_figure)

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    out_path = artifacts / "ushape_investigation_2016_2017.json"
    out_path.write_text(
        json.dumps(
            {
                "generated_utc": stamp,
                "train_year": train_year,
                "test_year": TEST_YEAR,
                "test_year_status": (
                    "RE-READ of an already-opened held-out year (ADR-002; "
                    "opened at L6.10 on 2026-08-13). Not a held-out result. "
                    "No model, parameter or bound decision was taken from "
                    "it -- it answers only whether the 2016 shape is also "
                    "present on 2017. Log this access in 07_PROGRESS.md."
                ),
                "band_definition": (
                    "Terciles of the TRAINING year's driver, reused unchanged "
                    "for the test year."
                ),
                "buildings": [
                    {key: value for key, value in record.items() if not key.startswith("_")}
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
