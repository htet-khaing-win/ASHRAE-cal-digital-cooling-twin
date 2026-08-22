"""Does tuning the residual model's hyperparameters buy anything? (L7.3 A/B)

    python scripts/tune_residual_model.py

An A/B test with the decision rule written down BEFORE the run, not
after. The hypothesis being tested is that L7.3's hand-chosen
hyperparameters are leaving accuracy on the table; the prior is that
they are not, because the ML layer explains 3.3% of Claude's variance
and 1.1% of Luke's, and a fraction of a small number is a smaller
number.

THE TRAP THIS SCRIPT EXISTS TO AVOID. The obvious way to tune is to
search hyperparameters against the out-of-fold score and keep whichever
wins. That destroys the out-of-fold score. Fifty trials selected on it
means fifty models were fitted to it, and it is no longer an estimate of
performance on unseen hours -- it is the maximum of fifty noisy draws,
which is biased upward by construction. It is the same mistake as tuning
on the test set, one level down, and it would corrupt the single number
L7.3 exists to produce.

NESTED CROSS-VALIDATION is the fix, and this script runs BOTH loops
because they answer different questions:

    outer fold 1  |--- train ---|~~|== score ==|.................|
                    |
                    +-- inner folds tune HERE, and only here
                        the outer scored block is never seen by
                        the search that chose the hyperparameters

  1. THE NESTED ESTIMATE (`nested`). For each outer fold, tune on that
     fold's training block using inner folds, fit the winner on the
     whole outer training block, predict the outer scored block. The
     resulting out-of-fold ML share is an honest estimate of the
     PROCEDURE "tune, then fit" -- it is what you would get if you ran
     the whole pipeline on a new year. It does NOT give you one
     configuration, because each outer fold may choose a different one.
     That is not a defect; it is what the estimate means.

  2. THE DEPLOYMENT CONFIGURATION (`deployment`). One tuning pass over
     all of 2016 with inner folds only, to pick the single config you
     would actually ship. Its inner score is optimistically biased and
     is NOT reported as performance -- estimate 1 is the performance.

Reporting estimate 2's score as the result is the most common way this
analysis is got wrong in industry, and it is exactly the number a
reviewer will ask you to defend.

THE DECISION RULE, fixed here in code so it cannot be softened after
seeing the answer -- see `TunedVerdict.worth_keeping`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import IncompatibleArtifactError, frozen_parameters  # noqa: E402
from fit_hybrid_residual import selected_buildings  # noqa: E402
from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)

from cooling_twin import SEED  # noqa: E402
from cooling_twin.analysis.hybrid import (  # noqa: E402
    DEFAULT_EMBARGO_HOURS,
    DEFAULT_N_FOLDS,
    ModelFactory,
    build_features,
    fit_hybrid,
    out_of_fold_correction,
    variance_decomposition,
)
from cooling_twin.calibration.crossval import (  # noqa: E402
    TimeFold,
    expanding_window_folds,
)
from cooling_twin.calibration.metrics import cvrmse  # noqa: E402

logger = logging.getLogger("tune_residual_model")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")

# --- THE PRE-REGISTERED DECISION RULE --------------------------------
#
# Written before the first run. A tuned configuration is kept only if it
# clears BOTH of these, and the numbers are here rather than in the
# report so that moving them is a code change with a diff, not a
# sentence rewritten after the result came in.
#
# 1.0 percentage point on the ML share, because anything smaller is
# invisible in CV(RMSE) (0.3 points at these magnitudes) and would not
# change a single conclusion in the project. The threshold is stated in
# the ML share rather than in CV(RMSE) because the ML share is what
# tuning can actually move -- the physics share is fixed by frozen
# parameters and cannot respond to a learning rate.
MIN_ML_SHARE_GAIN_PCT = 1.0

# The memorisation gap must not widen. A tuned config that raises the
# out-of-fold share by widening the gap has bought accuracy with
# capacity, and capacity is what fails first on a new year. Allowing a
# small tolerance would make the rule unfalsifiable in practice.
MAX_GAP_WIDENING_PCT = 0.0

# Trials per tuning pass. Fifty is enough for six parameters with TPE,
# and the whole run is ~2,000 fits at ~0.2 s each. Raising this raises
# the selection bias on the DEPLOYMENT config without raising the
# quality of the NESTED estimate, which is the one being reported.
DEFAULT_N_TRIALS = 50

# Inner folds. Three rather than five: the inner loop runs inside an
# outer training block, and the first outer block is only ~1,460 hours.
DEFAULT_N_INNER_FOLDS = 3

# The inner loop uses the same embargo as the outer. An inner split
# without one would select hyperparameters that interpolate neighbouring
# hours -- i.e. it would reliably choose the most over-fitting config in
# the search space, which is worse than not tuning at all.
INNER_EMBARGO_HOURS = DEFAULT_EMBARGO_HOURS

# Shortest inner validation window, hours. `crossval`'s own floor is 336
# and it CANNOT be met here: the first outer fold trains on ~1,460 hours,
# and three inner folds plus a 168 h embargo leave ~197 hours per inner
# window. Lowering the floor for the inner loop specifically is
# defensible where lowering it for the outer loop would not be -- the
# inner score only RANKS configurations against each other and is never
# reported as performance, while the outer score is the number that
# leaves this script. A week is the shortest window that still contains
# a weekend, which matters because `is_weekend` is a feature.
INNER_MIN_VALIDATE_HOURS = 168


@dataclass(frozen=True)
class TunedVerdict:
    """The A/B result for one building, and whether it clears the rule.

    Attributes:
        building_id: The building.
        default_ml_pct: Out-of-fold ML share with L7.3's hand-chosen
            hyperparameters.
        nested_ml_pct: Out-of-fold ML share of the tune-then-fit
            procedure, from the nested loop.
        default_gap_pct: L7.3's memorisation gap, percentage points.
        nested_gap_pct: The tuned procedure's memorisation gap.
        default_cvrmse_pct: Hybrid CV(RMSE) with the defaults.
        nested_cvrmse_pct: Hybrid CV(RMSE) with the tuned procedure.
        deployment_params: The single configuration a deployment would
            ship, from the separate all-of-2016 tuning pass. Reported
            for the record whether or not the rule is cleared.
    """

    building_id: str
    default_ml_pct: float
    nested_ml_pct: float
    default_gap_pct: float
    nested_gap_pct: float
    default_cvrmse_pct: float
    nested_cvrmse_pct: float
    deployment_params: dict[str, Any]

    @property
    def ml_share_gain_pct(self) -> float:
        """Percentage points of ML share the tuning bought."""
        return self.nested_ml_pct - self.default_ml_pct

    @property
    def gap_widening_pct(self) -> float:
        """How much wider the memorisation gap got. Negative is better."""
        return self.nested_gap_pct - self.default_gap_pct

    @property
    def worth_keeping(self) -> bool:
        """The pre-registered rule, evaluated.

        Both conditions, not either: a gain bought by widening the gap
        is capacity rather than learning, and capacity is what fails on
        a year the model has not seen.
        """
        return (
            self.ml_share_gain_pct >= MIN_ML_SHARE_GAIN_PCT
            and self.gap_widening_pct <= MAX_GAP_WIDENING_PCT
        )


def model_factory_from(params: dict[str, Any], seed: int = SEED) -> ModelFactory:
    """Build a model factory from a trial's parameters.

    `early_stopping=False` is forced here and is NOT in the search
    space. Letting the search turn it on would let it choose sklearn's
    random 10% holdout, which on an hourly series scores by
    interpolating neighbours -- the search would find that it "works"
    and select it every time.

    Args:
        params: Hyperparameters.
        seed: Estimator seed.

    Returns:
        A factory producing a fresh unfitted estimator per call.
    """

    def build() -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            early_stopping=False, random_state=seed, **params
        )

    return build


def _suggest(trial: Any) -> dict[str, Any]:
    """The search space.

    Ranges bracket L7.3's hand-chosen values rather than starting from
    them, so the search can say "your defaults were at the edge of a
    sensible range" -- which is a finding -- instead of only ever
    confirming a neighbourhood you already picked.

    Args:
        trial: An Optuna trial.

    Returns:
        Hyperparameters for `HistGradientBoostingRegressor`.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=100),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-3, 10.0, log=True
        ),
    }


def tune(
    features: pd.DataFrame,
    residual: np.ndarray,
    *,
    n_trials: int,
    n_inner_folds: int,
    label: str,
    checkpoints: tuple[int, ...] = (),
) -> dict[int, dict[str, Any]]:
    """Search hyperparameters against inner-fold RMSE only.

    ONE study is run, to `n_trials`, and the best configuration is read
    off after each checkpoint. That is not an approximation of running
    separate studies -- it is EXACTLY equal to them. `TPESampler(seed=...)`
    is a sequential deterministic process: trial `i`'s suggestion depends
    only on the seed and on trials `0..i-1`, and the objective here is
    deterministic, so the first 25 trials of a 300-trial study are the
    same 25 trials a 25-trial study would have run. Sweeping six budgets
    therefore costs one study at the largest budget rather than the sum
    of all six -- 300 trials instead of 750.
    `tests/test_trial_budget_prefix.py` asserts the equality rather than
    trusting this paragraph.

    Args:
        features: Feature matrix for the region being tuned on.
        residual: Physics residual over the same region.
        n_trials: Optuna trials to run.
        n_inner_folds: Inner expanding-window folds.
        label: For the log.
        checkpoints: Trial counts to report the running best at. Empty
            means report only at `n_trials`. Values above `n_trials` are
            refused rather than silently clipped -- a caller asking for
            a 300-trial result from a 100-trial study has a bug, and
            returning the 100-trial answer under a "300" key would hide
            it.

    Returns:
        `{checkpoint: best hyperparameters after that many trials}`.

    Raises:
        ValueError: If a checkpoint is not a positive integer at most
            `n_trials`.
    """
    import optuna

    wanted = tuple(sorted(set(checkpoints or (n_trials,))))
    for checkpoint in wanted:
        if checkpoint < 1 or checkpoint > n_trials:
            raise ValueError(
                f"checkpoint {checkpoint} must be between 1 and n_trials "
                f"({n_trials})"
            )

    inner_folds = expanding_window_folds(
        len(features),
        n_folds=n_inner_folds,
        spin_up_hours=0,
        embargo_hours=INNER_EMBARGO_HOURS,
        min_validate_hours=INNER_MIN_VALIDATE_HOURS,
    )

    def objective(trial: Any) -> float:
        correction, scored = out_of_fold_correction(
            features, residual, inner_folds, model_factory=model_factory_from(_suggest(trial))
        )
        # Plain RMSE of the corrected residual on the inner scored
        # hours. Not the ML share: the share's denominator is the
        # measured load's variance, which is constant across trials, so
        # the two orderings are identical and RMSE is the cheaper,
        # clearer objective to state.
        return float(np.sqrt(np.mean((residual[scored] - correction[scored]) ** 2)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_at: dict[int, dict[str, Any]] = {}
    for checkpoint in wanted:
        prefix = study.trials[:checkpoint]
        winner = min(prefix, key=lambda trial: trial.value)
        best_at[checkpoint] = dict(winner.params)
        logger.info(
            "%s: best inner RMSE %.1f kW after %d trials -- %s",
            label,
            winner.value,
            checkpoint,
            winner.params,
        )
    return best_at


def nested_correction(
    features: pd.DataFrame,
    residual: np.ndarray,
    outer_folds: tuple[TimeFold, ...],
    *,
    n_trials: int,
    n_inner_folds: int,
    label: str,
    checkpoints: tuple[int, ...] = (),
) -> dict[int, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    """Out-of-fold correction where each fold tuned on its own past only.

    Args:
        features: Full-year feature matrix.
        residual: Full-year physics residual.
        outer_folds: The reporting folds.
        n_trials: Trials per outer fold.
        n_inner_folds: Inner folds inside each outer training block.
        label: For the log.
        checkpoints: Trial budgets to produce a result for. See `tune`
            for why one study serves all of them.

    Returns:
        `{checkpoint: (correction, scored_mask, per_fold_params)}`.
    """
    wanted = tuple(sorted(set(checkpoints or (n_trials,))))
    results = {
        checkpoint: (
            np.zeros(residual.size, dtype=float),
            np.zeros(residual.size, dtype=bool),
            [],
        )
        for checkpoint in wanted
    }

    for fold in outer_folds:
        best_at = tune(
            features.iloc[fold.train_slice],
            residual[fold.train_slice],
            n_trials=n_trials,
            n_inner_folds=n_inner_folds,
            label=f"{label} outer fold {fold.number}",
            checkpoints=wanted,
        )
        for checkpoint, best in best_at.items():
            correction, scored, chosen = results[checkpoint]
            chosen.append(best)
            fold_correction, fold_scored = out_of_fold_correction(
                features, residual, (fold,), model_factory=model_factory_from(best)
            )
            correction[fold_scored] = fold_correction[fold_scored]
            scored |= fold_scored

    return results


def analyse(
    building: dict[str, str],
    config: dict[str, Any],
    artifacts: Path,
    *,
    n_trials: int,
    n_folds: int,
    n_inner_folds: int,
) -> tuple[TunedVerdict, dict[str, Any]]:
    """Run the A/B test for one building.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.
        n_trials: Optuna trials per tuning pass.
        n_folds: Outer folds.
        n_inner_folds: Inner folds.

    Returns:
        `(verdict, record)`.
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

    # --- A: L7.3's defaults, re-run so both arms share one code path ---
    arm_a = fit_hybrid(
        frame.index,
        measured,
        predicted,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=humidity,
        n_physics_params=len(names),
        label=f"{building_id} defaults",
        n_folds=n_folds,
    )

    # --- B: the nested tune-then-fit procedure -------------------------
    features = build_features(frame.index, t_outdoor, humidity)
    outer_folds = expanding_window_folds(
        measured.size,
        n_folds=n_folds,
        spin_up_hours=0,
        embargo_hours=DEFAULT_EMBARGO_HOURS,
    )
    correction, scored, per_fold = nested_correction(
        features,
        residual,
        outer_folds,
        n_trials=n_trials,
        n_inner_folds=n_inner_folds,
        label=building_id,
    )[n_trials]
    hybrid = np.maximum(predicted + correction, 0.0)
    nested = variance_decomposition(
        measured[scored], predicted[scored], hybrid[scored], label=f"{building_id} nested"
    )

    # The tuned arm's in-sample twin, for its memorisation gap. Fitted
    # with the DEPLOYMENT config, which is the one a single in-sample
    # fit would use -- the nested arm has no single config by design.
    deployment_params = tune(
        features,
        residual,
        n_trials=n_trials,
        n_inner_folds=n_inner_folds,
        label=f"{building_id} deployment",
    )[n_trials]
    in_sample_model = model_factory_from(deployment_params)()
    matrix = features.to_numpy(dtype=float)
    in_sample_model.fit(matrix[scored], residual[scored])
    in_sample_hybrid = np.maximum(
        predicted[scored] + np.asarray(in_sample_model.predict(matrix[scored]), dtype=float),
        0.0,
    )
    nested_in_sample = variance_decomposition(
        measured[scored],
        predicted[scored],
        in_sample_hybrid,
        label=f"{building_id} nested in-sample",
    )

    verdict = TunedVerdict(
        building_id=building_id,
        default_ml_pct=arm_a.decomposition.ml_pct,
        nested_ml_pct=nested.ml_pct,
        default_gap_pct=arm_a.memorisation_gap_pct,
        nested_gap_pct=nested_in_sample.ml_pct - nested.ml_pct,
        default_cvrmse_pct=arm_a.hybrid_cvrmse_pct,
        nested_cvrmse_pct=cvrmse(
            measured[scored], hybrid[scored], n_params=len(names)
        ),
        deployment_params=deployment_params,
    )

    logger.info(
        "%s: ML share %.2f%% (defaults) -> %.2f%% (tuned, nested), gain %+.2f pp; "
        "gap %.2f -> %.2f pp; CV(RMSE) %.2f%% -> %.2f%%; VERDICT %s",
        building_id,
        verdict.default_ml_pct,
        verdict.nested_ml_pct,
        verdict.ml_share_gain_pct,
        verdict.default_gap_pct,
        verdict.nested_gap_pct,
        verdict.default_cvrmse_pct,
        verdict.nested_cvrmse_pct,
        "KEEP TUNED" if verdict.worth_keeping else "KEEP DEFAULTS",
    )

    record = {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "year": train_year,
        "hours_scored": int(scored.sum()),
        "n_trials": n_trials,
        "n_outer_folds": n_folds,
        "n_inner_folds": n_inner_folds,
        "default": {
            "ml_pct": verdict.default_ml_pct,
            "physics_pct": arm_a.decomposition.physics_pct,
            "memorisation_gap_pct": verdict.default_gap_pct,
            "hybrid_cvrmse_pct": verdict.default_cvrmse_pct,
        },
        "nested_tuned": {
            "ml_pct": verdict.nested_ml_pct,
            "physics_pct": nested.physics_pct,
            "memorisation_gap_pct": verdict.nested_gap_pct,
            "hybrid_cvrmse_pct": verdict.nested_cvrmse_pct,
            "per_outer_fold_params": per_fold,
        },
        "deployment_params": deployment_params,
        "ml_share_gain_pct": verdict.ml_share_gain_pct,
        "gap_widening_pct": verdict.gap_widening_pct,
        "worth_keeping": verdict.worth_keeping,
    }
    return verdict, record


def main() -> None:
    """Entry point: run the A/B test, apply the rule, record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-inner-folds", type=int, default=DEFAULT_N_INNER_FOLDS)
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logger.info(
        "Decision rule, fixed before this run: keep the tuned config only if "
        "the nested out-of-fold ML share gains >= %.1f pp AND the memorisation "
        "gap widens by <= %.1f pp.",
        MIN_ML_SHARE_GAIN_PCT,
        MAX_GAP_WIDENING_PCT,
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    records = []
    for building in selected_buildings(arguments.buildings, ("primary", "generalisation")):
        try:
            _verdict, record = analyse(
                building,
                config,
                artifacts,
                n_trials=arguments.n_trials,
                n_folds=arguments.n_folds,
                n_inner_folds=arguments.n_inner_folds,
            )
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        records.append(record)

    if not records:
        raise SystemExit("no building had a calibration artifact to analyse")

    table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "ML % default": round(record["default"]["ml_pct"], 2),
                "ML % tuned": round(record["nested_tuned"]["ml_pct"], 2),
                "gain pp": round(record["ml_share_gain_pct"], 2),
                "gap default": round(record["default"]["memorisation_gap_pct"], 2),
                "gap tuned": round(record["nested_tuned"]["memorisation_gap_pct"], 2),
                "CV default": round(record["default"]["hybrid_cvrmse_pct"], 2),
                "CV tuned": round(record["nested_tuned"]["hybrid_cvrmse_pct"], 2),
                "keep tuned": record["worth_keeping"],
            }
            for record in records
        ]
    ).set_index("building")
    logger.info("--- A/B: hand-chosen vs tuned, %d ---\n%s", train_year, table.to_string())

    out_path = artifacts / f"hybrid_tuning_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "decision_rule": {
                    "min_ml_share_gain_pct": MIN_ML_SHARE_GAIN_PCT,
                    "max_gap_widening_pct": MAX_GAP_WIDENING_PCT,
                    "registered": "before the first run, in code",
                },
                "note": (
                    "nested_tuned is the honest estimate of the tune-then-fit "
                    "PROCEDURE; deployment_params is the single config a "
                    "deployment would ship, and its own inner score is "
                    "optimistically biased and deliberately not reported."
                ),
                "defaults": {
                    "source": "cooling_twin.analysis.hybrid.default_model_factory",
                },
                "buildings": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)

    if not any(record["worth_keeping"] for record in records):
        logger.info(
            "No building cleared the rule. The defaults stay. Record the "
            "negative result -- a measured 'tuning did not help' is a stronger "
            "claim than an untested default."
        )


if __name__ == "__main__":
    main()
