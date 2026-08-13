"""Open the held-out year and score the calibration against it (L6.10).

    python scripts/open_test_set.py --open-test-set

THIS IS THE GATE. It is the one script in the repo that reads 2017, and
running it spends a resource that cannot be refilled: after this, no
number measured on 2017 is a held-out number ever again. ADR-002 allows
exactly one opening, and this script exists so that opening is a
deliberate, logged, reproducible act rather than a `--year 2017` typed
into a calibration run one evening.

Three protections are structural rather than procedural:

  1. `--open-test-set` is REQUIRED. There is no default that touches the
     test year, so the access cannot happen by running the file.
  2. Nothing here can fit. The optimiser is not imported -- not
     `calibrate`, not `differential_evolution`, nothing that could move
     a parameter. Parameters are READ from the 2016 artifacts and frozen.
  3. The baselines are the 2016 fits from L6.4, evaluated on 2017
     through stored COEFFICIENTS. `BaselineFit` was built to hold
     coefficients rather than predictions for exactly this moment: a
     baseline refitted on the test year would see data the model cannot,
     and would flatter itself.

Spin-up: the simulation starts 72 h BEFORE 1 January 2017 using December
2016 drivers, and those hours are discarded before scoring. This is the
L6.9 treatment and it leaks nothing -- the spin-up consumes training-year
weather to settle the envelope state, never test-year measurements.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)

from cooling_twin.calibration.baseline import (  # noqa: E402
    MIN_RELATIVE_IMPROVEMENT_PCT,
    fit_annual_mean,
    fit_linear_regression,
    relative_cvrmse_improvement_pct,
)
from cooling_twin.calibration.crossval import DEFAULT_SPIN_UP_HOURS  # noqa: E402
from cooling_twin.calibration.metrics import (  # noqa: E402
    DataInterval,
    G14Verdict,
    ashrae_g14_pass,
    g14_thresholds,
)

logger = logging.getLogger("open_test_set")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
TEST_YEAR = 2017


def selected_buildings(path: Path) -> list[dict[str, str]]:
    """The three buildings the gate covers, primary first.

    Args:
        path: `config/buildings.yaml`.

    Returns:
        `[{"building_id": ..., "site_id": ..., "role": ...}]`.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If no primary building is declared.
    """
    if not path.exists():
        raise FileNotFoundError(f"building selection not found at {path}")
    selection = yaml.safe_load(path.read_text(encoding="utf-8"))
    primary = selection.get("primary") or []
    if not primary:
        raise ValueError(f"{path} declares no primary building")
    groups = (
        ("primary", primary),
        ("generalisation", selection.get("generalisation") or []),
    )
    return [
        {"building_id": entry["building_id"], "site_id": entry["site_id"], "role": role}
        for role, entries in groups
        for entry in entries
    ]


def frozen_parameters(directory: Path, building_id: str) -> tuple[dict[str, float], str]:
    """Read one building's calibrated parameters from its 2016 artifact.

    Read, never re-derived. The gate scores the parameters that were
    actually reported, so a re-run that happened to land elsewhere
    cannot quietly replace them.

    Args:
        directory: Where calibration artifacts live.
        building_id: The building.

    Returns:
        `(parameters, artifact_name)`.

    Raises:
        FileNotFoundError: If the building has no artifact.
        ValueError: If the artifact is not from the training year.
    """
    for path in sorted(directory.glob("calibration_*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record.get("metadata", {})
        if metadata.get("building_id") != building_id:
            continue
        if int(metadata.get("year", -1)) == TEST_YEAR:
            raise ValueError(
                f"{path.name} was calibrated on {TEST_YEAR}. That should be "
                "impossible -- run_calibration.py refuses any year but the "
                "training year. Investigate before trusting anything here."
            )
        return {name: float(value) for name, value in record["parameters"].items()}, path.name
    raise FileNotFoundError(f"no calibration artifact for {building_id} in {directory}")


def load_with_spin_up(
    building_id: str, site_id: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    """Load the test year preceded by a spin-up tail of the training year.

    Args:
        building_id: BDG2 identifier.
        site_id: Its site.
        config: Calibration config, for the training year.

    Returns:
        `(frame, scored_offset)` -- the frame covers spin-up plus the
        whole test year, and scoring starts at `scored_offset`.

    Raises:
        ValueError: If either year is missing after cleaning.
    """
    train_year = int(config["train_year"])
    train_frame, _ = load_training_data(building_id, site_id, train_year)
    test_frame, _ = load_training_data(building_id, site_id, TEST_YEAR)

    spin_up = train_frame.tail(DEFAULT_SPIN_UP_HOURS)
    combined = pd.concat([spin_up, test_frame]).sort_index()
    return combined, len(spin_up)


def evaluate(
    building: dict[str, str], config: dict[str, Any], artifacts: Path
) -> dict[str, Any]:
    """Score one building's frozen 2016 parameters on 2017.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.

    Returns:
        A record of everything the report needs.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    parameters, artifact_name = frozen_parameters(artifacts, building_id)
    names = tuple(config["parameters"])
    n_params = len(names)

    train_frame, floor_area_m2 = load_training_data(
        building_id, site_id, int(config["train_year"])
    )
    combined, scored_offset = load_with_spin_up(building_id, site_id, config)

    def objective_for(frame: pd.DataFrame) -> CalibrationObjective:
        return CalibrationObjective(
            (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
            frame["airTemperature"].to_numpy(dtype=float),
            frame["load_kwh"].to_numpy(dtype=float),
            floor_area_m2,
            config,
            outdoor_humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
        )

    vector = np.array([parameters[name] for name in names], dtype=float)

    # --- training year, for the train/test comparison -------------------
    train_objective = objective_for(train_frame)
    train_predicted, _raw = train_objective.predict(vector)
    train_measured = train_objective.observed_kw
    train_verdict = ashrae_g14_pass(train_measured, train_predicted, n_params=n_params)

    # --- the held-out year ----------------------------------------------
    test_objective = objective_for(combined)
    simulated, _raw = test_objective.predict(vector)
    test_predicted = simulated[scored_offset:]
    test_measured = test_objective.observed_kw[scored_offset:]
    test_index = combined.index[scored_offset:]
    test_outdoor = combined["airTemperature"].to_numpy(dtype=float)[scored_offset:]
    test_verdict = ashrae_g14_pass(test_measured, test_predicted, n_params=n_params)

    # --- baselines: fitted on 2016, evaluated on 2017 --------------------
    train_outdoor = train_frame["airTemperature"].to_numpy(dtype=float)
    mean_fit = fit_annual_mean(train_measured)
    regression_fit = fit_linear_regression(train_outdoor, train_measured)
    baselines = {}
    for fit in (mean_fit, regression_fit):
        verdict = ashrae_g14_pass(
            test_measured, fit.predict(test_outdoor), n_params=fit.n_params
        )
        baselines[fit.name] = verdict

    best_baseline_cvrmse = min(verdict.cvrmse_pct for verdict in baselines.values())
    improvement = relative_cvrmse_improvement_pct(
        best_baseline_cvrmse, test_verdict.cvrmse_pct
    )

    logger.info(
        "%s (%s): TRAIN NMBE %+.2f%% CV %.2f%% [%s]  ->  TEST NMBE %+.2f%% CV %.2f%% [%s]",
        building_id,
        building["role"],
        train_verdict.nmbe_pct,
        train_verdict.cvrmse_pct,
        "PASS" if train_verdict.passed else "FAIL",
        test_verdict.nmbe_pct,
        test_verdict.cvrmse_pct,
        "PASS" if test_verdict.passed else "FAIL",
    )

    return {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "parameters": parameters,
        "train_year": int(config["train_year"]),
        "train_hours": int(len(train_measured)),
        "test_hours": int(len(test_measured)),
        "test_mean_kw": float(test_measured.mean()),
        "train": _verdict_record(train_verdict),
        "test": _verdict_record(test_verdict),
        "baselines_on_test": {
            name: _verdict_record(verdict) for name, verdict in baselines.items()
        },
        "relative_improvement_pct": improvement,
        "beats_baseline_requirement": improvement >= MIN_RELATIVE_IMPROVEMENT_PCT,
        "seasonal_test_nmbe_pct": _seasonal_nmbe(
            test_index, test_measured, test_predicted, n_params
        ),
        "seasonal_train_nmbe_pct": _seasonal_nmbe(
            train_frame.index, train_measured, train_predicted, n_params
        ),
        "degradation_cvrmse_pp": test_verdict.cvrmse_pct - train_verdict.cvrmse_pct,
    }


def _verdict_record(verdict: G14Verdict) -> dict[str, Any]:
    """Flatten a `G14Verdict` for JSON."""
    return {
        "nmbe_pct": verdict.nmbe_pct,
        "cvrmse_pct": verdict.cvrmse_pct,
        "passed": verdict.passed,
        "meets_stretch_target": verdict.meets_stretch_target,
    }


def _seasonal_nmbe(
    index: pd.DatetimeIndex,
    measured: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
    n_params: int,
) -> dict[str, float]:
    """NMBE by meteorological season on the test year.

    The annual number can be near zero while the seasons are far from
    it -- measured on the training year in
    `reports/05_fold2_diagnosis.md`. Whether that signature repeats on
    2017 is the single most informative extra number the gate can
    produce, so it is computed here rather than left for M7.
    """
    from cooling_twin.calibration.metrics import nmbe

    seasons = {
        "winter (DJF)": (12, 1, 2),
        "spring (MAM)": (3, 4, 5),
        "summer (JJA)": (6, 7, 8),
        "autumn (SON)": (9, 10, 11),
    }
    months = index.month.to_numpy()
    return {
        label: round(nmbe(measured[mask], predicted[mask], n_params), 2)
        for label, group in seasons.items()
        if (mask := np.isin(months, group)).any()
    }


# A training fit this close to the G14 CV(RMSE) limit is not "good on
# train" -- it is marginal, and a normal year-to-year difference tips it
# over. Distinguishing the two matters because the prescribed actions are
# opposite: overfitting says REMOVE capacity, structure error says ADD it.
MARGINAL_HEADROOM_PCT = 3.0

# Degradation from train to test that counts as an overfitting signal,
# in percentage points. Matches crossval.OVERFITTING_GAP_PCT so the
# within-year and across-year verdicts use one threshold.
OVERFITTING_DEGRADATION_PP = 5.0


def diagnose(results: list[dict[str, Any]]) -> list[str]:
    """Apply 06_ASSESSMENT.md's four failure signatures to the outcome.

    The table's first row -- "good on train, poor on test -> overfitting"
    -- cannot be applied mechanically, and applying it mechanically is
    how a structural fault gets the wrong prescription. Two refinements,
    both learned by running this on real data:

      * "Good on train" must mean good, not merely inside the limit.
        A fit sitting 1.3 points under a 30% threshold has no headroom,
        and the ordinary difference between two weather years moves it
        across.
      * A seasonal bias pattern that is the SAME on both years cannot be
        overfitting. Overfitting is a fault that appears on data the fit
        never saw; a fault present on the training year in equal measure
        is structure, hidden by an annual NMBE that averages the seasons
        against each other.

    Args:
        results: Per-building records from `evaluate`.

    Returns:
        Human-readable findings, empty if the gate passes cleanly.
    """
    _nmbe_limit_pct, cvrmse_limit_pct = g14_thresholds(DataInterval.HOURLY)
    findings = []
    for record in results:
        name = record["building_id"]
        train, test = record["train"], record["test"]
        train_seasonal = record.get("seasonal_train_nmbe_pct", {})
        test_seasonal = record.get("seasonal_test_nmbe_pct", {})
        worst_train_season = max((abs(v) for v in train_seasonal.values()), default=0.0)
        worst_test_season = max((abs(v) for v in test_seasonal.values()), default=0.0)

        # Reported whether or not the building passed: a seasonal fault
        # this size is a finding even behind a passing annual number.
        if worst_train_season > cvrmse_limit_pct / 3.0:
            findings.append(
                f"{name}: seasonal NMBE reaches {worst_train_season:.1f}% on the "
                f"TRAINING year and {worst_test_season:.1f}% on test, against an "
                f"annual {train['nmbe_pct']:+.2f}% / {test['nmbe_pct']:+.2f}% -- "
                "MODEL STRUCTURE ERROR, present on both years and hidden by the "
                "annual average. Action: increase order, add latent term, check "
                "inputs. NOT overfitting: the fault is not new on the test year."
            )

        if test["passed"]:
            continue

        degradation = record["degradation_cvrmse_pp"]
        headroom = cvrmse_limit_pct - train["cvrmse_pct"]
        if train["passed"] and headroom < MARGINAL_HEADROOM_PCT:
            findings.append(
                f"{name}: train {train['cvrmse_pct']:.2f}% left only "
                f"{headroom:.2f} points of headroom under the {cvrmse_limit_pct:.0f}% "
                f"limit, and test came in {degradation:+.2f} pp higher. MARGINAL "
                "ON TRAIN, not a collapse -- the model was never comfortably "
                "inside the standard on this building."
            )
        elif train["passed"] and degradation > OVERFITTING_DEGRADATION_PP:
            findings.append(
                f"{name}: good on train ({train['cvrmse_pct']:.2f}%), poor on test "
                f"({test['cvrmse_pct']:.2f}%, {degradation:+.2f} pp) -- OVERFITTING "
                "signature. Action: fewer parameters, lower model order, regularise."
            )
        elif not train["passed"]:
            findings.append(
                f"{name}: poor on both ({train['cvrmse_pct']:.2f}% / "
                f"{test['cvrmse_pct']:.2f}%) -- MODEL STRUCTURE ERROR. Action: "
                "increase order, add latent term, check inputs."
            )

        if abs(test["nmbe_pct"]) > 10.0 and test["cvrmse_pct"] <= cvrmse_limit_pct:
            findings.append(
                f"{name}: large NMBE ({test['nmbe_pct']:+.2f}%) with acceptable "
                f"CV(RMSE) ({test['cvrmse_pct']:.2f}%) -- SYSTEMATIC BIAS. Action: "
                "check scaling, units, weather join."
            )
    return findings


def main() -> None:
    """Entry point: open the test year, score, diagnose, record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument(
        "--open-test-set",
        action="store_true",
        help="Required. Opening the held-out year is a one-way door "
        "(ADR-002) and must be typed deliberately.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    if not arguments.open_test_set:
        raise SystemExit(
            "refusing to run without --open-test-set. This script reads the "
            f"{TEST_YEAR} data, which ADR-002 permits exactly once. Once it has "
            "run, no number measured on that year is held out any more. Pass "
            "the flag only when that is what you mean to do."
        )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    buildings = selected_buildings(arguments.buildings)

    logger.warning(
        "OPENING THE HELD-OUT TEST SET (%d) for %d buildings. ADR-002 permits "
        "this once; log it in 07_PROGRESS.md.",
        TEST_YEAR,
        len(buildings),
    )

    results = [evaluate(building, config, artifacts) for building in buildings]

    table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "train NMBE %": round(record["train"]["nmbe_pct"], 2),
                "train CV %": round(record["train"]["cvrmse_pct"], 2),
                "test NMBE %": round(record["test"]["nmbe_pct"], 2),
                "test CV %": round(record["test"]["cvrmse_pct"], 2),
                "G14": "PASS" if record["test"]["passed"] else "FAIL",
                "vs baseline %": round(record["relative_improvement_pct"], 1),
            }
            for record in results
        ]
    ).set_index("building")
    logger.info("--- THE GATE ---\n%s", table.to_string())

    baseline_table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                **{
                    name: round(verdict["cvrmse_pct"], 2)
                    for name, verdict in record["baselines_on_test"].items()
                },
                "calibrated RC": round(record["test"]["cvrmse_pct"], 2),
            }
            for record in results
        ]
    ).set_index("building")
    logger.info("--- test-year CV(RMSE) against baselines ---\n%s", baseline_table.to_string())

    for record in results:
        logger.info(
            "%s seasonal test NMBE: %s",
            record["building_id"],
            record["seasonal_test_nmbe_pct"],
        )

    passed = all(record["test"]["passed"] for record in results)
    beats = all(record["beats_baseline_requirement"] for record in results)
    findings = diagnose(results)

    if passed:
        logger.info(
            "PRIMARY GATE: PASS -- all %d buildings meet both G14 hourly "
            "criteria on the held-out year.",
            len(results),
        )
    else:
        logger.warning("PRIMARY GATE: FAIL on at least one building.")
    for finding in findings:
        logger.warning("signature -- %s", finding)
    if not beats:
        logger.warning(
            "SUPPORTING REQUIREMENT NOT MET: >= %.0f%% relative CV(RMSE) "
            "improvement on the best baseline. Met by %s.",
            MIN_RELATIVE_IMPROVEMENT_PCT,
            [r["building_id"] for r in results if r["beats_baseline_requirement"]] or "none",
        )

    record = {
        "opened_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_year": TEST_YEAR,
        "interval": DataInterval.HOURLY.value,
        "primary_gate_passed": passed,
        "beats_baseline_requirement_all": beats,
        "failure_signatures": findings,
        "buildings": results,
    }
    out_path = artifacts / f"gate_{TEST_YEAR}_opened.json"
    if out_path.exists():
        logger.warning(
            "%s already exists -- the test set has been opened before. The "
            "first opening is the one that counts; this run is a re-read and "
            "must be reported as such.",
            out_path.name,
        )
        out_path = out_path.with_name(
            f"gate_{TEST_YEAR}_reread_{record['opened_utc'].replace(':', '')}.json"
        )
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
