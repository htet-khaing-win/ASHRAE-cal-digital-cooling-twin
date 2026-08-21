"""Does the tuner's answer depend on how long you let it search?

    python scripts/sweep_trial_budget.py --mlflow

Runs the nested A/B at several Optuna trial budgets -- 25, 50, 75, 100,
200, 300 by default -- and asks one question: DO THE CHOSEN PARAMETER
SETS DIFFER? That is a stability question, not a selection one, and the
distinction decides how the output may be used.

  ALLOWED    "the deployment config stops moving after ~N trials, so N
             is enough for this search space" -- a statement about
             convergence.
  ALLOWED    "the config never settles, so the objective is flat and the
             hyperparameters barely matter" -- also a real finding, and
             the more likely one here given the ML layer explains 3.3%
             of Claude's variance.
  FORBIDDEN  "budget 200 gave the best nested score, so we use 200."
             Sweeping six budgets and keeping the winner is tuning the
             tuner: the reported score becomes the maximum of six draws
             and is biased upward, exactly the bias the nested loop
             exists to remove. The pre-registered budget stays
             `tune_residual_model.DEFAULT_N_TRIALS`, and this sweep does
             not change it.

ONE STUDY SERVES EVERY BUDGET. `TPESampler(seed=SEED)` is sequential and
deterministic and the objective is deterministic, so the first 25 trials
of a 300-trial study ARE the 25 trials a 25-trial study would run. The
sweep therefore runs one study per tuning pass at the largest budget and
reads the running best at each checkpoint: 300 trials of work instead of
25+50+75+100+200+300 = 750, with identical results.
`tests/test_trial_budget_prefix.py` asserts that equality rather than
asking you to believe this paragraph.

COST. Roughly 300 trials x 3 inner fits x (5 outer folds + 1 deployment
pass) x 2 buildings ~ 11,000 model fits. Measured at ~0.1-0.3 s each,
that is 20-50 minutes on one core, single-threaded. Well short of
needing a warning, but do not expect it to finish while you make tea.
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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tracking  # noqa: E402
from analyse_residuals import IncompatibleArtifactError, frozen_parameters  # noqa: E402
from fit_hybrid_residual import selected_buildings  # noqa: E402
from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)
from tune_residual_model import (  # noqa: E402
    DEFAULT_N_FOLDS,
    DEFAULT_N_INNER_FOLDS,
    DEFAULT_N_TRIALS,
    MAX_GAP_WIDENING_PCT,
    MIN_ML_SHARE_GAIN_PCT,
    model_factory_from,
    nested_correction,
    tune,
)

from cooling_twin import SEED  # noqa: E402
from cooling_twin.analysis.hybrid import (  # noqa: E402
    DEFAULT_EMBARGO_HOURS,
    build_features,
    fit_hybrid,
    variance_decomposition,
)
from cooling_twin.calibration.crossval import expanding_window_folds  # noqa: E402
from cooling_twin.calibration.metrics import cvrmse  # noqa: E402

logger = logging.getLogger("sweep_trial_budget")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_3_trial_budget_sweep.png")
DEFAULT_BUDGETS = (25, 50, 75, 100, 200, 300)

# Hyperparameters compared between budgets. Kept as an explicit tuple
# rather than read from whatever the last trial happened to contain, so
# that adding a knob to the search space without adding it here is a
# visible omission rather than a silently shorter comparison.
TRACKED_PARAMS = (
    "max_depth",
    "max_leaf_nodes",
    "min_samples_leaf",
    "learning_rate",
    "max_iter",
    "l2_regularization",
)

# Parameters whose sensible comparison is multiplicative, not additive:
# a learning rate moving 0.01 -> 0.02 is the same size of change as
# 0.10 -> 0.20, and an absolute difference would call the first one
# negligible. Stability is therefore measured in log space for these.
LOG_SCALE_PARAMS = (
    "learning_rate",
    "l2_regularization",
    "min_samples_leaf",
    "max_leaf_nodes",
)

PHYSICS_COLOUR = "#2171b5"
ML_COLOUR = "#e08214"
STABLE_COLOUR = "#1a9850"
ZERO_COLOUR = "#525252"


def parameter_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    """How far apart two configurations are, as a single number.

    Each tracked parameter is normalised to a comparable scale -- log
    ratio for the multiplicative ones, relative difference for the rest
    -- and the mean absolute change is returned. It is a diagnostic, not
    a metric with units: 0.0 means identical, and larger means the
    search landed somewhere else. What matters is whether it FALLS
    towards zero as the budget grows.

    Args:
        left: One configuration.
        right: Another.

    Returns:
        Mean normalised absolute difference over `TRACKED_PARAMS`.
    """
    changes = []
    for name in TRACKED_PARAMS:
        a, b = float(left[name]), float(right[name])
        if name in LOG_SCALE_PARAMS:
            changes.append(abs(np.log(b / a)) if a > 0 and b > 0 else float("nan"))
        else:
            changes.append(abs(b - a) / max(abs(a), 1.0))
    return float(np.nanmean(changes))


def analyse(
    building: dict[str, str],
    config: dict[str, Any],
    artifacts: Path,
    *,
    budgets: tuple[int, ...],
    n_folds: int,
    n_inner_folds: int,
) -> list[dict[str, Any]]:
    """Run every budget for one building from a single set of studies.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.
        budgets: Trial budgets to report at.
        n_folds: Outer folds.
        n_inner_folds: Inner folds.

    Returns:
        One record per budget, in budget order.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(
        artifacts, building_id, train_year, names
    )

    frame, floor_area_m2 = load_training_data(building_id, site_id, train_year)
    t_outdoor = frame["airTemperature"].to_numpy(dtype=float)
    humidity = frame["humidity_ratio"].to_numpy(dtype=float)
    objective = CalibrationObjective(
        (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
        t_outdoor,
        frame["load_kwh"].to_numpy(dtype=float),
        floor_area_m2,
        config,
        outdoor_humidity_ratio=humidity,
    )
    predicted, _raw = objective.predict(
        np.array([parameters[name] for name in names], dtype=float)
    )
    measured = objective.observed_kw
    residual = measured - predicted
    features = build_features(frame.index, t_outdoor, humidity)
    matrix = features.to_numpy(dtype=float)

    # The A arm is budget-independent -- L7.3's hand-chosen
    # hyperparameters do not know what a trial is -- so it is computed
    # once and compared against every budget.
    baseline = fit_hybrid(
        frame.index,
        measured,
        predicted,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=humidity,
        n_physics_params=len(names),
        label=f"{building_id} defaults",
        n_folds=n_folds,
    )

    largest = max(budgets)
    outer_folds = expanding_window_folds(
        measured.size,
        n_folds=n_folds,
        spin_up_hours=0,
        embargo_hours=DEFAULT_EMBARGO_HOURS,
    )
    nested_by_budget = nested_correction(
        features,
        residual,
        outer_folds,
        n_trials=largest,
        n_inner_folds=n_inner_folds,
        label=building_id,
        checkpoints=budgets,
    )
    deployment_by_budget = tune(
        features,
        residual,
        n_trials=largest,
        n_inner_folds=n_inner_folds,
        label=f"{building_id} deployment",
        checkpoints=budgets,
    )

    records = []
    previous: dict[str, Any] | None = None
    previous_folds: list[dict[str, Any]] | None = None
    for budget in budgets:
        correction, scored, per_fold = nested_by_budget[budget]
        deployment = deployment_by_budget[budget]
        hybrid = np.maximum(predicted + correction, 0.0)
        nested = variance_decomposition(
            measured[scored],
            predicted[scored],
            hybrid[scored],
            label=f"{building_id} nested @{budget}",
        )

        in_sample_model = model_factory_from(deployment)()
        in_sample_model.fit(matrix[scored], residual[scored])
        in_sample_hybrid = np.maximum(
            predicted[scored]
            + np.asarray(in_sample_model.predict(matrix[scored]), dtype=float),
            0.0,
        )
        in_sample = variance_decomposition(
            measured[scored],
            predicted[scored],
            in_sample_hybrid,
            label=f"{building_id} nested in-sample @{budget}",
        )

        gain = nested.ml_pct - baseline.decomposition.ml_pct
        gap = in_sample.ml_pct - nested.ml_pct
        widening = gap - baseline.memorisation_gap_pct
        distance = parameter_distance(previous, deployment) if previous else float("nan")
        # The DEPLOYMENT config and the PER-FOLD configs are different
        # objects and they move independently. `nested_ml_pct` is driven
        # by the per-fold ones; the deployment config is what would be
        # shipped. Reporting only the deployment distance next to the
        # nested score invites the reading "the config did not move, so
        # why did the score?" -- which happened on Claude at 50 trials,
        # where the deployment config was byte-identical to the 25-trial
        # one while the score fell 0.6 pp. Both are reported.
        fold_distance = (
            float(
                np.mean(
                    [
                        parameter_distance(before, after)
                        for before, after in zip(previous_folds, per_fold, strict=True)
                    ]
                )
            )
            if previous_folds
            else float("nan")
        )

        record = {
            "building_id": building_id,
            "role": building["role"],
            "calibration_artifact": artifact_name,
            "year": train_year,
            "n_trials": budget,
            "n_outer_folds": n_folds,
            "n_inner_folds": n_inner_folds,
            "hours_scored": int(scored.sum()),
            "default_ml_pct": baseline.decomposition.ml_pct,
            "nested_ml_pct": nested.ml_pct,
            "nested_physics_pct": nested.physics_pct,
            "ml_share_gain_pct": gain,
            "memorisation_gap_pct": gap,
            "gap_widening_pct": widening,
            "default_cvrmse_pct": baseline.hybrid_cvrmse_pct,
            "nested_cvrmse_pct": cvrmse(
                measured[scored], hybrid[scored], n_params=len(names)
            ),
            "worth_keeping": bool(
                gain >= MIN_ML_SHARE_GAIN_PCT and widening <= MAX_GAP_WIDENING_PCT
            ),
            "deployment_params": deployment,
            "per_outer_fold_params": per_fold,
            "param_distance_from_previous_budget": distance,
            "per_fold_param_distance_from_previous_budget": fold_distance,
        }
        records.append(record)
        previous = deployment
        previous_folds = per_fold

        with tracking.run(
            f"{building_id}-trials{budget}",
            tags={
                "building_id": building_id,
                "role": building["role"],
                "train_year": str(train_year),
                "seed": str(SEED),
                "script": "sweep_trial_budget.py",
            },
        ):
            tracking.log_params(
                {
                    "n_trials": budget,
                    "n_outer_folds": n_folds,
                    "n_inner_folds": n_inner_folds,
                    "embargo_hours": DEFAULT_EMBARGO_HOURS,
                    "building_id": building_id,
                    **{f"best_{key}": deployment[key] for key in TRACKED_PARAMS},
                }
            )
            tracking.log_metrics(
                {
                    "nested_ml_pct": nested.ml_pct,
                    "default_ml_pct": baseline.decomposition.ml_pct,
                    "ml_share_gain_pct": gain,
                    "nested_physics_pct": nested.physics_pct,
                    "memorisation_gap_pct": gap,
                    "gap_widening_pct": widening,
                    "nested_cvrmse_pct": record["nested_cvrmse_pct"],
                    "default_cvrmse_pct": baseline.hybrid_cvrmse_pct,
                    "param_distance_from_previous_budget": distance,
                    "per_fold_param_distance_from_previous_budget": fold_distance,
                    "worth_keeping": float(record["worth_keeping"]),
                },
                step=budget,
            )
            tracking.log_json(record, f"{building_id}_trials{budget}.json")

        logger.info(
            "%s @%d trials: nested ML %.2f%% (gain %+.2f pp), gap %.2f pp, "
            "CV %.2f%%, config move deployment %.3f / per-fold %.3f -- %s",
            building_id,
            budget,
            nested.ml_pct,
            gain,
            gap,
            record["nested_cvrmse_pct"],
            distance,
            fold_distance,
            "KEEP TUNED" if record["worth_keeping"] else "KEEP DEFAULTS",
        )

    return records


def plot_sweep(records: list[dict[str, Any]], path: Path) -> Path:
    """Left: does the score move with budget. Right: does the config."""
    frame = pd.DataFrame(records)
    buildings = list(dict.fromkeys(frame["building_id"]))
    figure, axes = plt.subplots(
        len(buildings), 2, figsize=(11.0, 3.6 * len(buildings)), squeeze=False
    )

    for row, building_id in enumerate(buildings):
        rows = frame[frame["building_id"] == building_id]

        axis = axes[row][0]
        axis.plot(
            rows["n_trials"], rows["nested_ml_pct"], marker="o", color=ML_COLOUR,
            label="nested ML share (tuned)",
        )
        axis.axhline(
            float(rows["default_ml_pct"].iloc[0]),
            color=PHYSICS_COLOUR,
            linestyle="--",
            label="L7.3 defaults",
        )
        axis.axhline(
            float(rows["default_ml_pct"].iloc[0]) + MIN_ML_SHARE_GAIN_PCT,
            color=STABLE_COLOUR,
            linestyle=":",
            label=f"rule: +{MIN_ML_SHARE_GAIN_PCT:.0f} pp to keep tuned",
        )
        axis.set_xlabel("Optuna trials", fontsize=8)
        axis.set_xticks(list(rows["n_trials"]))
        axis.set_ylabel(f"{building_id}\nML share of variance, %", fontsize=8)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.set_title("does a longer search score better?", fontsize=9)

        axis = axes[row][1]
        axis.plot(
            rows["n_trials"],
            rows["param_distance_from_previous_budget"],
            marker="o",
            color=STABLE_COLOUR,
            label="deployment config (what ships)",
        )
        if "per_fold_param_distance_from_previous_budget" in rows:
            axis.plot(
                rows["n_trials"],
                rows["per_fold_param_distance_from_previous_budget"],
                marker="s",
                linestyle="--",
                color=ML_COLOUR,
                label="per-fold configs (what drives the score)",
            )
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_xlabel("Optuna trials", fontsize=8)
        axis.set_xticks(list(rows["n_trials"]))
        axis.set_ylabel("config change from previous budget", fontsize=8)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.set_title(
            "does the chosen config settle?  (0 = identical)", fontsize=9
        )

    figure.suptitle(
        "Trial-budget sweep. A STABILITY diagnostic -- the budget is NOT "
        "selected from these curves (see the module docstring)."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: sweep the budgets, log, plot, record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Optuna trial budgets to compare.",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-inner-folds", type=int, default=DEFAULT_N_INNER_FOLDS)
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log each budget as an MLflow run in ./mlruns.",
    )
    parser.add_argument("--experiment", default=tracking.DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--replot",
        action="store_true",
        help=(
            "Redraw the figure and tables from the stored artifact instead of "
            "re-running the sweep. Every derived column is recomputed from the "
            "per-budget configurations the artifact already holds, so a "
            "reporting fix costs seconds rather than another full search."
        ),
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    if arguments.mlflow:
        tracking.enable(arguments.experiment)

    budgets = tuple(sorted(set(arguments.budgets)))
    logger.warning(
        "This sweep is a STABILITY diagnostic. The pre-registered budget stays "
        "%d trials; do NOT select a budget from these results -- keeping the "
        "best of %d budgets makes the reported score the maximum of %d draws.",
        DEFAULT_N_TRIALS,
        len(budgets),
        len(budgets),
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    if arguments.replot:
        records = replot_records(artifacts, train_year)
        report(records, artifacts, train_year, arguments.figure, write_ab=False)
        return

    records: list[dict[str, Any]] = []
    for building in selected_buildings(arguments.buildings, ("primary", "generalisation")):
        try:
            records.extend(
                analyse(
                    building,
                    config,
                    artifacts,
                    budgets=budgets,
                    n_folds=arguments.n_folds,
                    n_inner_folds=arguments.n_inner_folds,
                )
            )
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue

    if not records:
        raise SystemExit("no building had a calibration artifact to analyse")

    report(records, artifacts, train_year, arguments.figure, budgets=budgets)


def report(
    records: list[dict[str, Any]],
    artifacts: Path,
    train_year: int,
    figure_path: Path,
    *,
    budgets: tuple[int, ...] | None = None,
    write_ab: bool = True,
) -> None:
    """Log the tables, draw the figure, write the artifacts.

    Split out of `main` so `--replot` can reach it without re-running a
    search that takes half an hour to produce numbers the artifact
    already holds.

    Args:
        records: One record per building-budget.
        artifacts: Artifact directory.
        train_year: The fitting year.
        figure_path: Where the figure goes.
        budgets: Budgets covered, for the artifact. Inferred from the
            records when replotting.
        write_ab: Whether to rewrite the A/B artifact at the
            pre-registered budget. False on a replot: redrawing a figure
            must not silently rewrite the file a downstream script reads.
    """
    budgets = budgets or tuple(sorted({record["n_trials"] for record in records}))

    table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "trials": record["n_trials"],
                "nested ML %": round(record["nested_ml_pct"], 2),
                "gain pp": round(record["ml_share_gain_pct"], 2),
                "gap pp": round(record["memorisation_gap_pct"], 2),
                "CV %": round(record["nested_cvrmse_pct"], 2),
                "move (ship)": round(record["param_distance_from_previous_budget"], 3),
                "move (folds)": round(
                    record.get(
                        "per_fold_param_distance_from_previous_budget", float("nan")
                    ),
                    3,
                ),
                "keep": record["worth_keeping"],
            }
            for record in records
        ]
    ).set_index(["building", "trials"])
    logger.info("--- trial-budget sweep, %d ---\n%s", train_year, table.to_string())

    chosen = pd.DataFrame(
        [
            {"building": record["building_id"], "trials": record["n_trials"],
             **{key: record["deployment_params"][key] for key in TRACKED_PARAMS}}
            for record in records
        ]
    ).set_index(["building", "trials"])
    logger.info("--- deployment config by budget ---\n%s", chosen.to_string())

    plot_sweep(records, figure_path)

    out_path = artifacts / f"hybrid_trial_budget_sweep_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "budgets": list(budgets),
                "seed": SEED,
                "purpose": (
                    "STABILITY DIAGNOSTIC. Do the chosen hyperparameters settle "
                    "as the search budget grows? The budget is NOT selected from "
                    "these results -- that would be tuning the tuner, and the "
                    "reported score would become the maximum of len(budgets) "
                    "draws. The pre-registered budget remains "
                    f"tune_residual_model.DEFAULT_N_TRIALS = {DEFAULT_N_TRIALS}."
                ),
                "equivalence_note": (
                    "One Optuna study per tuning pass, run to max(budgets), with "
                    "the running best read off at each budget. Exactly equal to "
                    "separate studies because TPESampler(seed) is sequential and "
                    "the objective is deterministic -- see "
                    "tests/test_trial_budget_prefix.py."
                ),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)

    # The pre-registered budget's rows ARE the A/B result, in the shape
    # `reread_hybrid_2017.py --use-tuned` reads. Written here so that
    # answering the A/B question does not mean paying for the same 50
    # trials a second time under a different script name. Only the
    # pre-registered budget is written -- writing the best budget's rows
    # would smuggle the selection this sweep is forbidden from making.
    if write_ab and DEFAULT_N_TRIALS in budgets:
        ab_path = artifacts / f"hybrid_tuning_{train_year}.json"
        ab_path.write_text(
            json.dumps(
                {
                    "year": train_year,
                    "decision_rule": {
                        "min_ml_share_gain_pct": MIN_ML_SHARE_GAIN_PCT,
                        "max_gap_widening_pct": MAX_GAP_WIDENING_PCT,
                        "registered": "before the first run, in code",
                    },
                    "note": (
                        f"Written by sweep_trial_budget.py at the PRE-REGISTERED "
                        f"budget of {DEFAULT_N_TRIALS} trials, not at whichever "
                        "budget scored best. Identical to running "
                        "tune_residual_model.py --n-trials "
                        f"{DEFAULT_N_TRIALS} -- see the prefix equivalence in "
                        "tests/test_trial_budget_prefix.py."
                    ),
                    "buildings": [
                        record for record in records
                        if record["n_trials"] == DEFAULT_N_TRIALS
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "A/B artifact at the pre-registered %d-trial budget: %s",
            DEFAULT_N_TRIALS,
            ab_path,
        )

    logger.info(
        "browse any tracked runs with: mlflow ui --backend-store-uri %s",
        tracking.DEFAULT_TRACKING_URI,
    )


def replot_records(artifacts: Path, train_year: int) -> list[dict[str, Any]]:
    """Load a stored sweep and recompute its derived columns.

    Derived columns are recomputed rather than trusted, so a fix to how
    a diagnostic is calculated reaches an existing artifact instead of
    only applying to runs made after the fix. The measured quantities --
    scores, shares, chosen configurations -- are read as they were
    recorded and never recomputed, because those came from the search.

    Args:
        artifacts: Artifact directory.
        train_year: The fitting year.

    Returns:
        The records, with distance columns rebuilt.

    Raises:
        FileNotFoundError: If no sweep artifact exists.
    """
    path = artifacts / f"hybrid_trial_budget_sweep_{train_year}.json"
    if not path.exists():
        raise FileNotFoundError(f"no sweep artifact at {path} -- run the sweep first")
    records = json.loads(path.read_text(encoding="utf-8"))["records"]

    by_building: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_building.setdefault(record["building_id"], []).append(record)

    for rows in by_building.values():
        rows.sort(key=lambda record: record["n_trials"])
        for index, record in enumerate(rows):
            if index == 0:
                record["param_distance_from_previous_budget"] = float("nan")
                record["per_fold_param_distance_from_previous_budget"] = float("nan")
                continue
            previous = rows[index - 1]
            record["param_distance_from_previous_budget"] = parameter_distance(
                previous["deployment_params"], record["deployment_params"]
            )
            record["per_fold_param_distance_from_previous_budget"] = float(
                np.mean(
                    [
                        parameter_distance(before, after)
                        for before, after in zip(
                            previous["per_outer_fold_params"],
                            record["per_outer_fold_params"],
                            strict=True,
                        )
                    ]
                )
            )
    logger.info("replotted %d records from %s", len(records), path)
    return records


if __name__ == "__main__":
    main()
