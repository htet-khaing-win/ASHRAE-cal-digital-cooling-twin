"""Learn each model's residual and decompose physics / ML / unexplained (L7.3).

    python scripts/fit_hybrid_residual.py

Reads each building's FROZEN 2016 parameters from its calibration
artifact, simulates the physics model, fits a gradient-boosted
correction to what is left over, and reports how much of the measured
load each layer explains -- with every ML number measured out of fold.

THE SAME TWO STRUCTURAL PROTECTIONS `analyse_residuals.py` carries, for
the same two reasons:

  1. ADR-002. Training year only. Nothing here touches 2017.
  2. The optimiser is not imported. Parameters are READ and frozen. If
     the physics were re-fitted after seeing this decomposition, the
     physics share would be a number chosen with the answer in hand.

AND ONE MORE, NEW HERE. `Hog_education_Cathleen` is EXCLUDED by default,
and that is a decision, not an oversight. ADR-015 established that its
U-shaped residual is the inverse model's clip at zero binding on 4.95%
of the year -- below -15 degC the model predicts exactly zero against a
meter reading a steady 444 kW. A gradient booster handed that residual
would learn the base load in an afternoon and report a large, healthy ML
share. The defect would vanish from the numbers while remaining entirely
present in the model, which is the exact opposite of what M7 exists to
do. The finding is the deliverable; laundering it into an accuracy
figure would destroy it. `--include-negative-case` runs it anyway, for
the demonstration -- read the warning it logs before quoting anything it
produces.
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
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import (  # noqa: E402
    IncompatibleArtifactError,
    frozen_parameters,
)
from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)

from cooling_twin.analysis.hybrid import (  # noqa: E402
    DEFAULT_EMBARGO_HOURS,
    DEFAULT_N_FOLDS,
    FEATURE_NAMES,
    HybridResult,
    build_features,
    fit_hybrid,
    permutation_importance_kw,
)
from cooling_twin.analysis.residual import (  # noqa: E402
    Binning,
    residual_profile,
)

logger = logging.getLogger("fit_hybrid_residual")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_3_hybrid_decomposition.png")

# Buildings whose residual is a legitimate learning target: the ones
# whose model holds G14 on the held-out year. See the module docstring
# for why the negative case is not in this tuple.
DEFAULT_GROUPS = ("primary", "generalisation")
NEGATIVE_CASE_GROUP = "negative_case"

PHYSICS_COLOUR = "#2171b5"
ML_COLOUR = "#e08214"
UNEXPLAINED_COLOUR = "#bdbdbd"
BEFORE_COLOUR = "#b2182b"
AFTER_COLOUR = "#1a9850"
ZERO_COLOUR = "#525252"


def selected_buildings(path: Path, groups: tuple[str, ...]) -> list[dict[str, str]]:
    """Buildings to analyse, in report order.

    Args:
        path: `config/buildings.yaml`.
        groups: Which roles to include.

    Returns:
        `[{"building_id": ..., "site_id": ..., "role": ...}]`.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If no primary building is declared.
    """
    if not path.exists():
        raise FileNotFoundError(f"building selection not found at {path}")
    selection = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not selection.get("primary"):
        raise ValueError(f"{path} declares no primary building")
    return [
        {"building_id": entry["building_id"], "site_id": entry["site_id"], "role": role}
        for role in groups
        for entry in (selection.get(role) or [])
    ]


def analyse(
    building: dict[str, str],
    config: dict[str, Any],
    artifacts: Path,
    *,
    n_folds: int,
    embargo_hours: int,
) -> tuple[HybridResult, dict[str, np.ndarray], dict[str, Any]]:
    """Fit and decompose one building's hybrid.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.
        n_folds: Expanding-window folds.
        embargo_hours: Hours dropped between training and scoring.

    Returns:
        `(result, series, record)` -- the fitted hybrid, the arrays the
        figure needs (kept OUT of the JSON record, which is a summary
        and must stay readable), and a JSON-shaped record.
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

    label = f"{building_id} {train_year}"
    result = fit_hybrid(
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
    importance = permutation_importance_kw(features, measured - predicted, result.folds)
    logger.info(
        "%s permutation importance (kW RMSE, averaged over folds): %s",
        building_id,
        {name: round(value, 2) for name, value in importance.items()},
    )

    record = {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "year": train_year,
        "hours_total": result.n_hours_total,
        "hours_scored": result.n_hours_scored,
        "scored_fraction": result.scored_fraction,
        "n_folds": n_folds,
        "embargo_hours": embargo_hours,
        "features": list(FEATURE_NAMES),
        "out_of_fold": {
            "physics_pct": result.decomposition.physics_pct,
            "ml_pct": result.decomposition.ml_pct,
            "unexplained_pct": result.decomposition.unexplained_pct,
        },
        "in_sample": {
            "physics_pct": result.in_sample_decomposition.physics_pct,
            "ml_pct": result.in_sample_decomposition.ml_pct,
            "unexplained_pct": result.in_sample_decomposition.unexplained_pct,
        },
        "memorisation_gap_pct": result.memorisation_gap_pct,
        "physics_cvrmse_pct": result.physics_cvrmse_pct,
        "hybrid_cvrmse_pct": result.hybrid_cvrmse_pct,
        "physics_nmbe_pct": result.physics_nmbe_pct,
        "hybrid_nmbe_pct": result.hybrid_nmbe_pct,
        "cvrmse_improvement_pct": result.cvrmse_improvement_pct,
        "clipped_fraction": result.clipped_fraction,
        "fold_ml_pct": list(result.fold_ml_pct),
        "n_folds_harmed": result.n_folds_harmed,
        "daily_variance_share_before": result.diagnostics_before.daily_variance_share,
        "daily_variance_share_after": result.diagnostics_after.daily_variance_share,
        "white_noise_variance_share": (
            result.diagnostics_after.white_noise_variance_share
        ),
        "acf_before": {str(lag): value for lag, value in result.diagnostics_before.acf.items()},
        "acf_after": {str(lag): value for lag, value in result.diagnostics_after.acf.items()},
        "permutation_importance_kw": dict(importance),
        "ml_dominates": result.decomposition.ml_dominates,
    }
    series = {
        "t_outdoor_c": t_outdoor,
        # Both taken directly from the measured series rather than one
        # derived from the other: `measured - physics` and
        # `(measured - hybrid) + correction` differ on any hour the clip
        # at zero engaged, and a panel that silently used the wrong one
        # would understate the correction exactly where it misbehaved.
        "residual_before": measured - predicted,
        "residual_after": measured - result.hybrid_kw,
    }
    return result, series, record


def plot_hybrids(
    results: list[tuple[dict[str, Any], HybridResult, dict[str, np.ndarray]]], path: Path
) -> Path:
    """One row per building: the split, the shape it removed, the drivers.

    The middle panel is the one that carries the argument. A share is a
    single number and a reader has to take it on trust; the residual
    profile against outdoor temperature shows the same claim as a shape,
    and if the correction only flattened the curve where the physics was
    already fine, that is visible immediately.
    """
    n_rows = len(results)
    figure, axes = plt.subplots(
        n_rows, 3, figsize=(14.0, 3.6 * n_rows), squeeze=False
    )

    for row, (record, result, series) in enumerate(results):
        scored = result.scored_mask
        t_outdoor = series["t_outdoor_c"]

        # --- panel 1: the decomposition, out-of-fold vs in-sample ---
        axis = axes[row][0]
        for offset, key in ((0.0, "out_of_fold"), (1.0, "in_sample")):
            shares = record[key]
            left = 0.0
            for share, colour in (
                (shares["physics_pct"], PHYSICS_COLOUR),
                (shares["ml_pct"], ML_COLOUR),
                (shares["unexplained_pct"], UNEXPLAINED_COLOUR),
            ):
                axis.barh(offset, share, left=left, color=colour, height=0.55)
                if share > 4.0:
                    axis.text(
                        left + share / 2.0,
                        offset,
                        f"{share:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if colour != UNEXPLAINED_COLOUR else "black",
                    )
                left += share
        axis.set_yticks([0.0, 1.0])
        axis.set_yticklabels(["out-of-fold", "in-sample"], fontsize=8)
        axis.set_ylim(-0.6, 1.6)
        axis.set_xlim(0.0, 100.0)
        axis.set_xlabel("share of measured variance, %", fontsize=8)
        axis.set_title(
            f"physics {record['out_of_fold']['physics_pct']:.0f}%  /  "
            f"ML {record['out_of_fold']['ml_pct']:.0f}%  /  "
            f"unexplained {record['out_of_fold']['unexplained_pct']:.0f}%",
            fontsize=9,
        )
        axis.tick_params(labelsize=8)

        # --- panel 2: the temperature profile, before and after ---
        axis = axes[row][1]
        mean_load = float(result.hybrid_kw[scored].mean())
        residual_before = series["residual_before"][scored]
        residual_after = series["residual_after"][scored]
        for values, colour, name in (
            (residual_before, BEFORE_COLOUR, "physics residual"),
            (residual_after, AFTER_COLOUR, "hybrid residual"),
        ):
            profile = residual_profile(
                values,
                t_outdoor[scored],
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
        axis.set_xlabel("outdoor dry bulb, degC", fontsize=8)
        axis.set_ylabel("mean residual, kW", fontsize=8)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.set_title(
            f"daily variance share {record['daily_variance_share_before']:.2f}"
            f" -> {record['daily_variance_share_after']:.2f}"
            f"  (noise {record['white_noise_variance_share']:.2f})",
            fontsize=9,
        )

        # --- panel 3: what the correction needed ---
        axis = axes[row][2]
        importance = record["permutation_importance_kw"]
        names = list(importance)[::-1]
        axis.barh(names, [importance[name] for name in names], color=ML_COLOUR)
        axis.axvline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_xlabel("held-out RMSE cost of scrambling, kW", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.3, axis="x")
        axis.set_title(
            f"CV(RMSE) {record['physics_cvrmse_pct']:.1f}% -> "
            f"{record['hybrid_cvrmse_pct']:.1f}%",
            fontsize=9,
        )

        axes[row][0].set_ylabel(record["building_id"], fontsize=9)

    figure.suptitle(
        "Physics / ML / unexplained, training year, frozen physics parameters. "
        "Every ML number is out-of-fold unless labelled in-sample."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: fit, decompose, log, plot, record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--embargo-hours", type=int, default=DEFAULT_EMBARGO_HOURS)
    parser.add_argument(
        "--include-negative-case",
        action="store_true",
        help=(
            "Also fit Hog_education_Cathleen, whose residual is a documented "
            "structural defect (ADR-015). Read the module docstring first."
        ),
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
            "Including the negative case. Its residual is the clip-at-zero "
            "defect of ADR-015, not a missing physical term. A large ML share "
            "there measures how easily a booster learns a base load, and must "
            "NOT be reported as evidence the hybrid works."
        )

    results = []
    for building in selected_buildings(arguments.buildings, groups):
        try:
            result, series, record = analyse(
                building,
                config,
                artifacts,
                n_folds=arguments.n_folds,
                embargo_hours=arguments.embargo_hours,
            )
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        results.append((record, result, series))

    if not results:
        raise SystemExit("no building had a calibration artifact to analyse")

    ranking = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "physics %": round(record["out_of_fold"]["physics_pct"], 1),
                "ML % (oof)": round(record["out_of_fold"]["ml_pct"], 1),
                "ML % (in-sample)": round(record["in_sample"]["ml_pct"], 1),
                "unexplained %": round(record["out_of_fold"]["unexplained_pct"], 1),
                "CV physics %": round(record["physics_cvrmse_pct"], 2),
                "CV hybrid %": round(record["hybrid_cvrmse_pct"], 2),
                "daily var before": round(record["daily_variance_share_before"], 3),
                "daily var after": round(record["daily_variance_share_after"], 3),
                "folds harmed": f"{record['n_folds_harmed']}/{record['n_folds']}",
            }
            for record, _, _ in results
        ]
    ).set_index("building")
    logger.info("--- hybrid decomposition, %d ---\n%s", train_year, ranking.to_string())

    plot_hybrids(results, arguments.figure)

    out_path = artifacts / f"hybrid_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "note": (
                    "Training year only (ADR-002). Physics parameters read "
                    "frozen; the optimiser is not imported. Every ML share "
                    "labelled out_of_fold is measured on hours the residual "
                    "model was not fitted on."
                ),
                "buildings": [record for record, _, _ in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
