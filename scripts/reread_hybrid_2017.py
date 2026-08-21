"""Does the learnt correction survive a year it has never seen? (L7.3 re-read)

    python scripts/reread_hybrid_2017.py --reread-test-set

THIS IS A RE-READ, NOT A GATE. `open_test_set.py` spent the one clean
opening of 2017 that ADR-002 allows, at L6.10, on the physics model.
07_PROGRESS.md's standing rule applies to everything after it: any 2017
number produced following a model change is a re-read, never a clean
held-out result. Every number this script writes is labelled that way in
its artifact, and none of them may be substituted for the gate figures
in reports/02_calibration.md.

Be precise about WHY it is only a re-read, because the usual reason does
not apply here. The hybrid never sees 2017 data: the physics parameters
are frozen from 2016 and the residual model is fitted on 2016 alone. The
leak is not in the data, it is in the ANALYST. `hybrid.py`'s feature set
was chosen after L7.1 and L7.2 had already read 2017's residual
structure -- the hockey stick, the humidity finding, the daily variance
shares in the *rr* rows of reports/06_residual_curvature.md. Choices
made with knowledge of the answer cannot be scored as though they were
made blind, however clean the code path is.

WHAT THIS ANSWERS THAT BLOCKED CV CANNOT. L7.3's out-of-fold ML share
(3.32% on Claude) is a WITHIN-YEAR number: the correction was fitted on
January-to-September and scored on October-to-December. That is not the
question anyone deploying a twin cares about. The question is whether a
correction learnt on one year still corrects the next one, and only a
different year can answer it. Year-to-year weather variation is exactly
what kills learnt corrections, and it is invisible to any split of 2016.

TWO STRUCTURAL DIFFERENCES FROM L7.3, both deliberate:

  1. NO FOLDS. The residual model is fitted on ALL of 2016. Folds were a
     substitute for a held-out set; with a real one in hand, holding
     data back would only weaken the model being tested.
  2. AN EXTRAPOLATION CENSUS. A tree ensemble goes flat outside its
     training range, so the share of 2017 hours lying outside 2016's
     observed feature range is reported. That number bounds how much of
     2017 the correction could even in principle have got right, and it
     is the empirical form of the argument for learning the residual
     rather than the load.
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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import IncompatibleArtifactError, frozen_parameters  # noqa: E402
from fit_hybrid_residual import selected_buildings  # noqa: E402
from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)
from tune_residual_model import model_factory_from  # noqa: E402

from cooling_twin.analysis.hybrid import (  # noqa: E402
    build_features,
    default_model_factory,
    fit_hybrid,
    variance_decomposition,
)
from cooling_twin.analysis.residual import (  # noqa: E402
    Binning,
    residual_diagnostics,
    residual_profile,
)
from cooling_twin.calibration.crossval import DEFAULT_SPIN_UP_HOURS  # noqa: E402
from cooling_twin.calibration.metrics import cvrmse, nmbe  # noqa: E402

logger = logging.getLogger("reread_hybrid_2017")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_3_hybrid_reread_2017.png")
TEST_YEAR = 2017

# Continuous features whose 2017 range is compared against 2016's. The
# cyclic and binary features are excluded because they cannot leave
# their training range by construction -- an hour of day in 2017 is an
# hour of day in 2016, and reporting them as "0% out of range" would
# pad the census with three guaranteed zeros.
CENSUS_FEATURES = (
    "outdoor_dry_bulb_c",
    "outdoor_dry_bulb_24h_mean_c",
    "humidity_ratio_g_per_kg",
)

TRAIN_COLOUR = "#2171b5"
BEFORE_COLOUR = "#b2182b"
AFTER_COLOUR = "#1a9850"
ZERO_COLOUR = "#525252"


def extrapolation_census(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> dict[str, float]:
    """Share of test hours lying outside the training range, per feature.

    A histogram-based gradient booster assigns any value beyond its
    outermost bin edge to that bin, so its prediction there is the edge
    leaf's value -- constant, no matter how far outside. These fractions
    are therefore an upper bound on how much of the test year the
    correction could have responded to.

    Args:
        train_features: Features over the fitting year.
        test_features: Features over the year being predicted.

    Returns:
        `{feature: fraction of test hours outside, ..., "any": ...}`.
    """
    outside = np.zeros(len(test_features), dtype=bool)
    census = {}
    for name in CENSUS_FEATURES:
        low = float(np.nanmin(train_features[name].to_numpy(dtype=float)))
        high = float(np.nanmax(train_features[name].to_numpy(dtype=float)))
        values = test_features[name].to_numpy(dtype=float)
        beyond = (values < low) | (values > high)
        census[name] = float(np.mean(beyond))
        outside |= np.nan_to_num(beyond, nan=False)
    census["any"] = float(np.mean(outside))
    return census


def evaluate(
    building: dict[str, str],
    config: dict[str, Any],
    artifacts: Path,
    *,
    tuned_params: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit the correction on 2016, apply it to 2017, and score both.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.
        tuned_params: Hyperparameters from the A/B run, or None for
            `hybrid.default_model_factory`.

    Returns:
        `(record, series)` -- the JSON-shaped record and the arrays the
        figure needs.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    n_params = len(names)
    parameters, artifact_name = frozen_parameters(
        artifacts, building_id, train_year, names
    )
    vector = np.array([parameters[name] for name in names], dtype=float)

    train_frame, floor_area_m2 = load_training_data(building_id, site_id, train_year)
    test_frame, _ = load_training_data(building_id, site_id, TEST_YEAR)

    # The physics model carries envelope state, so the test simulation
    # starts in December 2016 and those hours are discarded -- the L6.9
    # treatment `open_test_set.py` uses. It leaks nothing: the spin-up
    # consumes training-year WEATHER, never test-year measurements.
    combined = pd.concat([train_frame.tail(DEFAULT_SPIN_UP_HOURS), test_frame]).sort_index()
    scored_offset = len(train_frame.tail(DEFAULT_SPIN_UP_HOURS))

    def objective_for(frame: pd.DataFrame) -> CalibrationObjective:
        return CalibrationObjective(
            (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
            frame["airTemperature"].to_numpy(dtype=float),
            frame["load_kwh"].to_numpy(dtype=float),
            floor_area_m2,
            config,
            outdoor_humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
        )

    train_objective = objective_for(train_frame)
    train_predicted, _raw = train_objective.predict(vector)
    train_measured = train_objective.observed_kw
    train_t = train_frame["airTemperature"].to_numpy(dtype=float)
    train_w = train_frame["humidity_ratio"].to_numpy(dtype=float)

    test_objective = objective_for(combined)
    simulated, _raw = test_objective.predict(vector)
    test_predicted = simulated[scored_offset:]
    test_measured = test_objective.observed_kw[scored_offset:]
    test_index = combined.index[scored_offset:]
    test_t = combined["airTemperature"].to_numpy(dtype=float)[scored_offset:]

    # --- the 2016 within-year number, for the comparison ---------------
    # Re-run rather than read from hybrid_2016.json so the two arms come
    # from one code path and one set of hyperparameters. If this
    # disagrees with the stored artifact, the comparison below is
    # between two different models and means nothing.
    factory = model_factory_from(tuned_params) if tuned_params else default_model_factory()
    within_year = fit_hybrid(
        train_frame.index,
        train_measured,
        train_predicted,
        t_outdoor_c=train_t,
        humidity_ratio_kg_per_kg=train_w,
        n_physics_params=n_params,
        label=f"{building_id} {train_year} within-year",
        model_factory=factory,
    )

    # --- fit on ALL of 2016, predict 2017 ------------------------------
    train_features = build_features(train_frame.index, train_t, train_w)
    # Features for the test year are built over the COMBINED frame and
    # then sliced, so the 24-hour weather mean at 01:00 on 1 January
    # 2017 is a real mean of December 2016 weather rather than a NaN.
    # Training-year weather is an exogenous input, not a measurement of
    # the test year, so this leaks nothing.
    test_features = build_features(
        combined.index,
        combined["airTemperature"].to_numpy(dtype=float),
        combined["humidity_ratio"].to_numpy(dtype=float),
    ).iloc[scored_offset:]

    model = factory()
    model.fit(train_features.to_numpy(dtype=float), train_measured - train_predicted)
    correction = np.asarray(
        model.predict(test_features.to_numpy(dtype=float)), dtype=float
    )
    test_hybrid = np.maximum(test_predicted + correction, 0.0)
    clipped = float(np.mean(test_predicted + correction < 0.0))

    across_year = variance_decomposition(
        test_measured,
        test_predicted,
        test_hybrid,
        label=f"{building_id} {TEST_YEAR} re-read",
    )
    census = extrapolation_census(train_features, test_features)
    before = residual_diagnostics(
        test_measured - test_predicted, label=f"{building_id} {TEST_YEAR} physics residual"
    )
    after = residual_diagnostics(
        test_measured - test_hybrid, label=f"{building_id} {TEST_YEAR} hybrid residual"
    )

    physics_cv = cvrmse(test_measured, test_predicted, n_params=n_params)
    hybrid_cv = cvrmse(test_measured, test_hybrid, n_params=n_params)

    logger.warning(
        "%s RE-READ %d: ML share %.2f%% within %d (out-of-fold) -> %.2f%% across "
        "years; CV(RMSE) %.2f%% -> %.2f%%; %.1f%% of test hours outside the "
        "training feature range",
        building_id,
        TEST_YEAR,
        within_year.decomposition.ml_pct,
        train_year,
        across_year.ml_pct,
        physics_cv,
        hybrid_cv,
        100.0 * census["any"],
    )

    record = {
        "building_id": building_id,
        "role": building["role"],
        "reread": True,
        "not_a_gate_number": (
            "ADR-002's single clean opening was spent at L6.10 on the physics "
            "model. The hybrid's feature set was chosen after L7.1/L7.2 had "
            "read 2017. This is an analyst-mediated leak, not a data leak."
        ),
        "calibration_artifact": artifact_name,
        "hyperparameters": tuned_params or "hybrid.default_model_factory",
        "train_year": train_year,
        "test_year": TEST_YEAR,
        "test_hours": int(test_measured.size),
        "within_year_out_of_fold": {
            "ml_pct": within_year.decomposition.ml_pct,
            "physics_pct": within_year.decomposition.physics_pct,
            "unexplained_pct": within_year.decomposition.unexplained_pct,
            "hours_scored": within_year.n_hours_scored,
        },
        "across_year_reread": {
            "ml_pct": across_year.ml_pct,
            "physics_pct": across_year.physics_pct,
            "unexplained_pct": across_year.unexplained_pct,
        },
        "ml_share_retained_pct": (
            100.0 * across_year.ml_pct / within_year.decomposition.ml_pct
            if within_year.decomposition.ml_pct != 0.0
            else float("nan")
        ),
        "physics_cvrmse_pct": physics_cv,
        "hybrid_cvrmse_pct": hybrid_cv,
        "physics_nmbe_pct": nmbe(test_measured, test_predicted, n_params=n_params),
        "hybrid_nmbe_pct": nmbe(test_measured, test_hybrid, n_params=n_params),
        "clipped_fraction": clipped,
        "extrapolation_census": census,
        "daily_variance_share_before": before.daily_variance_share,
        "daily_variance_share_after": after.daily_variance_share,
        "white_noise_variance_share": after.white_noise_variance_share,
    }
    series = {
        "t_outdoor_c": test_t,
        "residual_before": test_measured - test_predicted,
        "residual_after": test_measured - test_hybrid,
        "train_t_outdoor_c": train_t,
        "index": test_index.to_numpy(),
    }
    return record, series


def plot_reread(
    results: list[tuple[dict[str, Any], dict[str, np.ndarray]]], path: Path
) -> Path:
    """Per building: what the correction did on a year it never saw."""
    n_rows = len(results)
    figure, axes = plt.subplots(n_rows, 2, figsize=(11.0, 3.6 * n_rows), squeeze=False)

    for row, (record, series) in enumerate(results):
        # --- the residual shape on the test year -----------------------
        axis = axes[row][0]
        mean_load = float(np.abs(series["residual_before"]).mean() + 1.0)
        for values, colour, name in (
            (series["residual_before"], BEFORE_COLOUR, "physics residual"),
            (series["residual_after"], AFTER_COLOUR, "hybrid residual"),
        ):
            profile = residual_profile(
                values,
                series["t_outdoor_c"],
                name=name,
                unit="outdoor dry bulb, degC",
                binning=Binning.FIXED_WIDTH,
                normaliser_kw=mean_load,
            )
            axis.errorbar(
                profile.centres,
                profile.means,
                yerr=profile.sems,
                marker="o",
                markersize=3,
                linewidth=1.6,
                capsize=2,
                color=colour,
                label=f"{name} (swing {profile.swing_kw:.0f} kW)",
            )
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_xlabel(f"outdoor dry bulb, degC ({TEST_YEAR})", fontsize=8)
        axis.set_ylabel(f"{record['building_id']}\nmean residual, kW", fontsize=8)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.set_title(
            f"CV(RMSE) {record['physics_cvrmse_pct']:.2f}% -> "
            f"{record['hybrid_cvrmse_pct']:.2f}%  (RE-READ)",
            fontsize=9,
        )

        # --- how much of the ML share survived the year change ---------
        axis = axes[row][1]
        within = record["within_year_out_of_fold"]["ml_pct"]
        across = record["across_year_reread"]["ml_pct"]
        axis.bar(
            [f"within {record['train_year']}\n(out-of-fold)", f"across to {TEST_YEAR}\n(re-read)"],
            [within, across],
            color=[TRAIN_COLOUR, AFTER_COLOUR if across > 0 else BEFORE_COLOUR],
            width=0.55,
        )
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_ylabel("ML share of variance, %", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.3, axis="y")
        axis.set_title(
            f"{record['ml_share_retained_pct']:.0f}% of the ML share retained; "
            f"{100.0 * record['extrapolation_census']['any']:.1f}% of hours "
            "out of training range",
            fontsize=9,
        )

    figure.suptitle(
        f"RE-READ of {TEST_YEAR} (ADR-002's clean opening was spent at L6.10). "
        "Correction fitted on 2016 only; physics parameters frozen."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def load_tuned_params(artifacts: Path, train_year: int) -> dict[str, dict[str, Any]]:
    """Deployment hyperparameters per building, if the A/B run kept them.

    Only configurations that CLEARED the pre-registered rule are used.
    A tuned config that lost the A/B test but gets used anyway would make
    the A/B test decorative.

    Args:
        artifacts: Artifact directory.
        train_year: The tuning run's year.

    Returns:
        `{building_id: params}` for buildings whose tuning was kept.
    """
    path = artifacts / f"hybrid_tuning_{train_year}.json"
    if not path.exists():
        logger.info("no tuning artifact at %s -- using hybrid.py's defaults", path)
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    kept = {
        entry["building_id"]: entry["deployment_params"]
        for entry in record["buildings"]
        if entry["worth_keeping"]
    }
    logger.info(
        "tuning artifact found: %d of %d buildings cleared the rule (%s)",
        len(kept),
        len(record["buildings"]),
        ", ".join(kept) or "none",
    )
    return kept


def main() -> None:
    """Entry point: re-read the test year with the hybrid, and log it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument(
        "--reread-test-set",
        action="store_true",
        help=(
            "Required. Reading the test year is deliberate even as a re-read, "
            "and every access is logged in 07_PROGRESS.md."
        ),
    )
    parser.add_argument(
        "--use-tuned",
        action="store_true",
        help=(
            "Use the deployment hyperparameters from hybrid_tuning_<year>.json, "
            "but only for buildings whose tuning cleared the pre-registered rule."
        ),
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    if not arguments.reread_test_set:
        raise SystemExit(
            f"refusing to run without --reread-test-set. This script reads "
            f"{TEST_YEAR}. ADR-002's one clean opening was spent at L6.10, so "
            "everything here is a RE-READ and must be labelled one wherever it "
            "is quoted. Pass the flag only when that is what you mean to do."
        )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    logger.warning(
        "RE-READING THE TEST YEAR (%d) with the L7.3 hybrid. This is NOT a "
        "gate result and must never be substituted for the L6.10 figures in "
        "reports/02_calibration.md. Log this access in 07_PROGRESS.md.",
        TEST_YEAR,
    )

    tuned = load_tuned_params(artifacts, train_year) if arguments.use_tuned else {}

    results = []
    for building in selected_buildings(arguments.buildings, ("primary", "generalisation")):
        try:
            record, series = evaluate(
                building,
                config,
                artifacts,
                tuned_params=tuned.get(building["building_id"]),
            )
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        results.append((record, series))

    if not results:
        raise SystemExit("no building had a calibration artifact to analyse")

    table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                f"ML % within {train_year}": round(
                    record["within_year_out_of_fold"]["ml_pct"], 2
                ),
                f"ML % across to {TEST_YEAR}": round(
                    record["across_year_reread"]["ml_pct"], 2
                ),
                "retained %": round(record["ml_share_retained_pct"], 0),
                "CV physics %": round(record["physics_cvrmse_pct"], 2),
                "CV hybrid %": round(record["hybrid_cvrmse_pct"], 2),
                "out of range %": round(
                    100.0 * record["extrapolation_census"]["any"], 1
                ),
            }
            for record, _ in results
        ]
    ).set_index("building")
    logger.info(
        "--- RE-READ %d: does the correction transfer across years? ---\n%s",
        TEST_YEAR,
        table.to_string(),
    )

    plot_reread(results, arguments.figure)

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S+0000")
    out_path = artifacts / f"hybrid_reread_{TEST_YEAR}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "reread_utc": stamp,
                "test_year": TEST_YEAR,
                "train_year": train_year,
                "is_gate_result": False,
                "note": (
                    "RE-READ. ADR-002's single clean opening was spent at L6.10 "
                    "on the physics model. The hybrid never sees test-year data "
                    "-- physics parameters frozen from the training year, "
                    "residual model fitted on the training year alone -- but its "
                    "feature set was chosen after L7.1/L7.2 had read 2017. "
                    "Timestamped rather than overwritten so re-reads accumulate "
                    "visibly instead of replacing one another."
                ),
                "buildings": [record for record, _ in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
