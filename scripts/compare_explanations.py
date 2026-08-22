"""Attribution vs decomposition, on the same model, per building (L7.4b).

    python scripts/compare_explanations.py

L7.3 said how much of each building the hybrid explains. L7.4a built the
other kind of explanation -- exact Shapley attribution -- and showed on
synthetic data that it survives having nothing to explain. This script
runs both on the REAL buildings and puts them in one table, so the
project's answer to "just show me a SHAP plot" is a measurement rather
than an opinion.

FOUR MODELS ARE FITTED PER BUILDING, and the fourth is the point:

  1. the physics model, parameters READ FROZEN from the 2016 artifact;
  2. the out-of-fold correction (L7.3), which produces the ML share;
  3. the DEPLOYMENT correction -- one ensemble fitted on every hour of
     the year, which is the model a practitioner would ship and the
     model a SHAP plot is normally drawn from;
  4. the CONTROL -- the same ensemble, same hyperparameters, fitted on
     the same residual with its HOURS SHUFFLED. Nothing about the
     building survives that shuffle, so every honest diagnostic must
     collapse on it. Whatever attribution the control still produces is
     the part of the real building's attribution that is not evidence of
     anything.

The same two structural protections the rest of M7 carries:

  1. ADR-002. Training year only. Nothing here touches 2017.
  2. The optimiser is not imported. Physics parameters are READ.

And ADR-015's exclusion of `Hog_education_Cathleen`, for the reason
`fit_hybrid_residual.py` gives at length: its residual is the inverse
model's clip at zero, and an attribution over a booster that learnt a
base load would read as a healthy explanation of a broken model.
`--include-negative-case` runs it anyway.
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

from analyse_residuals import (  # noqa: E402
    IncompatibleArtifactError,
    frozen_parameters,
)
from fit_hybrid_residual import selected_buildings  # noqa: E402
from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)

from cooling_twin import SEED  # noqa: E402
from cooling_twin.analysis.explain import (  # noqa: E402
    ShapleyExplanation,
    explanation_comparison,
    rank_agreement,
    sample_rows,
    shapley_values_kw,
)
from cooling_twin.analysis.hybrid import (  # noqa: E402
    DEFAULT_EMBARGO_HOURS,
    DEFAULT_N_FOLDS,
    build_features,
    default_model_factory,
    fit_hybrid,
    permutation_importance_kw,
)

logger = logging.getLogger("compare_explanations")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_4_explanation_comparison.png")

DEFAULT_GROUPS = ("primary", "generalisation")
NEGATIVE_CASE_GROUP = "negative_case"

# Hours explained, and background hours the base value averages over.
# 64 x 150 x 120 = 1.15M predictions per model, an order of magnitude
# under explain.MAX_MODEL_EVALUATIONS and about ten seconds per arm on
# this machine. Both are stated here rather than defaulted inside the
# library so the report can quote them.
N_EXPLAIN_HOURS = 150
N_BACKGROUND_HOURS = 120

# Separate seed for the explained-hours draw, so the explained and
# background samples are not the same rows in the same order.
EXPLAIN_SEED = SEED + 1

REAL_COLOUR = "#2171b5"
CONTROL_COLOUR = "#d6604d"
PHYSICS_COLOUR = "#2171b5"
ML_COLOUR = "#e08214"
UNEXPLAINED_COLOUR = "#bdbdbd"
ZERO_COLOUR = "#525252"


def explain_one_model(
    features: pd.DataFrame,
    target_kw: np.ndarray,
    folds: Any,
    *,
    label: str,
) -> tuple[ShapleyExplanation, pd.DataFrame]:
    """Fit a deployment correction on every hour, then explain it.

    Fitting on every hour is deliberate and is what makes the comparison
    fair to attribution: this is the model a practitioner ships, and
    explaining the shipped model is what the method is for. The held-out
    column beside it comes from `permutation_importance_kw`, which
    refits per fold -- so the table compares an in-sample explanation of
    the deployed model against an out-of-sample measurement of the same
    learner on the same features, which is exactly the comparison a
    reviewer needs.

    Args:
        features: Exogenous feature frame for the whole year.
        target_kw: What the correction is fitted on, kW.
        folds: Folds for the held-out permutation importance.
        label: Building, year and arm, for logs.

    Returns:
        `(explanation, comparison_frame)`.
    """
    model = default_model_factory()()
    model.fit(features.to_numpy(dtype=float), target_kw)

    explanation = shapley_values_kw(
        model,
        sample_rows(features, N_EXPLAIN_HOURS, seed=EXPLAIN_SEED),
        sample_rows(features, N_BACKGROUND_HOURS),
        label=label,
    )
    comparison = explanation_comparison(
        explanation, permutation_importance_kw(features, target_kw, folds)
    )
    logger.info("%s\n%s", label, comparison.round(2).to_string())
    return explanation, comparison


def analyse(
    building: dict[str, str],
    config: dict[str, Any],
    artifacts: Path,
    *,
    n_folds: int,
    embargo_hours: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Decompose, explain, and run the shuffled control for one building.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.
        n_folds: Expanding-window folds.
        embargo_hours: Hours dropped between training and scoring.

    Returns:
        `(record, frames)` -- a JSON-shaped summary, and the two
        comparison frames the figure draws.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(artifacts, building_id, train_year, names)

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

    label = f"{building_id} {train_year}"
    hybrid = fit_hybrid(
        frame.index,
        measured,
        predicted,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=humidity,
        n_physics_params=len(names),
        label=label,
        n_folds=n_folds,
        embargo_hours=embargo_hours,
    )

    features = build_features(frame.index, t_outdoor, humidity)
    real, real_comparison = explain_one_model(
        features, residual, hybrid.folds, label=f"{label} deployment"
    )
    # The control. Shuffled with its own generator so the arm is
    # reproducible independently of anything fitted before it.
    shuffled = np.random.default_rng(SEED).permutation(residual)
    control, control_comparison = explain_one_model(
        features, shuffled, hybrid.folds, label=f"{label} shuffled control"
    )

    real_total = float(sum(real.mean_abs_kw.values()))
    control_total = float(sum(control.mean_abs_kw.values()))
    record = {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "year": train_year,
        "hours_total": hybrid.n_hours_total,
        "hours_scored": hybrid.n_hours_scored,
        "n_explain_hours": N_EXPLAIN_HOURS,
        "n_background_hours": N_BACKGROUND_HOURS,
        "physics_pct": hybrid.decomposition.physics_pct,
        "ml_pct_out_of_fold": hybrid.decomposition.ml_pct,
        "ml_pct_in_sample": hybrid.in_sample_decomposition.ml_pct,
        "unexplained_pct": hybrid.decomposition.unexplained_pct,
        "memorisation_gap_pct": hybrid.memorisation_gap_pct,
        "attribution_kw": dict(real.mean_abs_kw),
        "attribution_kw_control": dict(control.mean_abs_kw),
        "permutation_kw": {
            str(name): float(value)
            for name, value in zip(
                real_comparison.index, real_comparison["permutation kW"], strict=True
            )
        },
        "permutation_kw_control": {
            str(name): float(value)
            for name, value in zip(
                control_comparison.index, control_comparison["permutation kW"], strict=True
            )
        },
        "base_value_kw": real.base_value_kw,
        "base_value_kw_control": control.base_value_kw,
        "efficiency_error_kw": real.efficiency_error_kw,
        "off_manifold_fraction": dict(real.off_manifold_fraction),
        "rank_agreement": rank_agreement(real_comparison),
        "rank_agreement_control": rank_agreement(control_comparison),
        # The headline of this script. How much of the attribution
        # magnitude a model with NOTHING to explain still produces,
        # relative to the real one. A number near 1 means the plot would
        # look the same either way.
        "control_attribution_ratio": control_total / real_total,
        "attribution_total_kw": real_total,
        "attribution_total_kw_control": control_total,
        "top_feature": real.ranking[0],
        "top_feature_control": control.ranking[0],
    }
    logger.info(
        "%s: attribution total %.1f kW real vs %.1f kW on the shuffled control "
        "(%.0f%%); ML share out-of-fold %.2f%%",
        building_id,
        real_total,
        control_total,
        100.0 * record["control_attribution_ratio"],
        hybrid.decomposition.ml_pct,
    )
    return record, {"real": real_comparison, "control": control_comparison}


def plot_comparisons(
    results: list[tuple[dict[str, Any], dict[str, pd.DataFrame]]], path: Path
) -> Path:
    """One row per building: attribution, held-out cost, decomposition.

    The first two panels carry the same six features on the same axis
    scale per row, real against shuffled control. A reader who covers up
    the legend cannot tell the two attribution panels apart; the held-out
    panel beside it is unambiguous. That is the argument, drawn.
    """
    n_rows = len(results)
    figure, axes = plt.subplots(n_rows, 3, figsize=(15.0, 3.9 * n_rows), squeeze=False)
    offsets = np.array([-0.2, 0.2])

    for row, (record, frames) in enumerate(results):
        names = list(frames["real"].index)
        positions = np.arange(len(names), dtype=float)

        # --- panel 1: the attribution, real vs control ---
        axis = axes[row][0]
        axis.barh(
            positions + offsets[0],
            [record["attribution_kw"][name] for name in names],
            height=0.38,
            color=REAL_COLOUR,
            label="deployment model",
        )
        axis.barh(
            positions + offsets[1],
            [record["attribution_kw_control"][name] for name in names],
            height=0.38,
            color=CONTROL_COLOUR,
            label="shuffled control",
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(names, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel("mean |attribution|, kW", fontsize=8)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3, axis="x")
        axis.tick_params(labelsize=8)
        axis.set_title(
            f"attribution: {record['attribution_total_kw']:.0f} kW real vs "
            f"{record['attribution_total_kw_control']:.0f} kW control "
            f"({100.0 * record['control_attribution_ratio']:.0f}%)",
            fontsize=9,
        )

        # --- panel 2: the held-out cost, same features, same order ---
        axis = axes[row][1]
        axis.barh(
            positions + offsets[0],
            [record["permutation_kw"][name] for name in names],
            height=0.38,
            color=REAL_COLOUR,
        )
        axis.barh(
            positions + offsets[1],
            [record["permutation_kw_control"][name] for name in names],
            height=0.38,
            color=CONTROL_COLOUR,
        )
        axis.axvline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_yticks(positions)
        axis.set_yticklabels([], fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel("held-out RMSE cost of scrambling, kW", fontsize=8)
        axis.grid(alpha=0.3, axis="x")
        axis.tick_params(labelsize=8)
        axis.set_title("the same six features, measured on unseen hours", fontsize=9)

        # --- panel 3: what the decomposition says instead ---
        axis = axes[row][2]
        left = 0.0
        for share, colour, name in (
            (record["physics_pct"], PHYSICS_COLOUR, "physics"),
            (record["ml_pct_out_of_fold"], ML_COLOUR, "ML"),
            (record["unexplained_pct"], UNEXPLAINED_COLOUR, "unexplained"),
        ):
            axis.barh(0.0, share, left=left, color=colour, height=0.5, label=name)
            if share > 4.0:
                axis.text(
                    left + share / 2.0,
                    0.0,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if colour != UNEXPLAINED_COLOUR else "black",
                )
            left += share
        axis.barh(
            1.0,
            record["ml_pct_in_sample"],
            color=ML_COLOUR,
            height=0.5,
            alpha=0.5,
        )
        axis.text(
            record["ml_pct_in_sample"] + 1.0,
            1.0,
            f"ML in-sample {record['ml_pct_in_sample']:.1f}%",
            va="center",
            fontsize=7,
        )
        axis.set_yticks([0.0, 1.0])
        axis.set_yticklabels(["out-of-fold", "in-sample ML"], fontsize=8)
        axis.set_ylim(-0.6, 1.6)
        axis.set_xlim(0.0, 100.0)
        axis.set_xlabel("share of measured variance, %", fontsize=8)
        axis.legend(fontsize=7, loc="lower right")
        axis.tick_params(labelsize=8)
        axis.set_title(
            f"ML out-of-fold {record['ml_pct_out_of_fold']:.1f}%  "
            f"(gap {record['memorisation_gap_pct']:+.1f} pp)",
            fontsize=9,
        )

        axes[row][0].set_ylabel(record["building_id"], fontsize=9)

    figure.suptitle(
        "The same learner, explained two ways. Left: attribution, computed from the "
        "model alone. Right: what the model is worth against the meter."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: decompose, explain, control, log, plot, record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--embargo-hours", type=int, default=DEFAULT_EMBARGO_HOURS)
    parser.add_argument(
        "--include-negative-case",
        action="store_true",
        help="Also explain Hog_education_Cathleen (ADR-015). Read the docstring.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    groups = DEFAULT_GROUPS
    if arguments.include_negative_case:
        groups = (*DEFAULT_GROUPS, NEGATIVE_CASE_GROUP)
        logger.warning(
            "Including the negative case. An attribution over its correction "
            "describes a booster relearning the clip-at-zero base load of "
            "ADR-015, and must not be reported as an explanation of the building."
        )

    results = []
    for building in selected_buildings(arguments.buildings, groups):
        try:
            record, frames = analyse(
                building,
                config,
                artifacts,
                n_folds=arguments.n_folds,
                embargo_hours=arguments.embargo_hours,
            )
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        results.append((record, frames))

    if not results:
        raise SystemExit("no building had a calibration artifact to explain")

    summary = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "top by attribution": record["top_feature"],
                "top on control": record["top_feature_control"],
                "attribution kW": round(record["attribution_total_kw"], 1),
                "control kW": round(record["attribution_total_kw_control"], 1),
                "control / real": round(record["control_attribution_ratio"], 2),
                "ML % (oof)": round(record["ml_pct_out_of_fold"], 2),
                "ML % (in-sample)": round(record["ml_pct_in_sample"], 2),
                "rank agree": round(record["rank_agreement"], 2),
                "off-manifold": round(max(record["off_manifold_fraction"].values()), 2),
            }
            for record, _ in results
        ]
    ).set_index("building")
    logger.info("--- attribution vs decomposition, %d ---\n%s", train_year, summary.to_string())

    plot_comparisons(results, arguments.figure)

    out_path = artifacts / f"explanations_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "note": (
                    "Training year only (ADR-002). Physics parameters read frozen. "
                    "The attribution explains a correction fitted on every hour -- "
                    "the deployment model -- which is what a SHAP plot normally "
                    "shows. The control arm is the identical learner fitted on the "
                    "same residual with its hours shuffled: any attribution it "
                    "still produces is not evidence about the building."
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
