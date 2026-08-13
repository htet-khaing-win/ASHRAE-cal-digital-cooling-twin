"""Decompose the calibrated model's residual, per building (L7.1b).

    python scripts/analyse_residuals.py

Runs L7.1's `decompose_residual()` over the TRAINING year for every
building the project carries -- the two that hold ASHRAE G14 on the
held-out year, and the one that does not -- using each building's frozen
2016 parameters read from its calibration artifact.

TWO STRUCTURAL PROTECTIONS, for two different reasons:

  1. ADR-002. This script reads the training year only. It never touches
     2017, so nothing it produces is a re-read of the held-out year (see
     07_PROGRESS.md's note that any 2017 number produced after a model
     change is a re-read and must be labelled as one).
  2. The optimiser is not imported. This script's job is to find out
     what the model is missing; a script that can also re-fit invites
     the loop where a diagnosis is immediately "fixed" by moving a
     parameter, and the finding is lost. Parameters are READ, frozen,
     and simulated -- the same discipline `open_test_set.py` applies for
     the opposite reason.

One calibration config serves every building. Only the BOUNDS differ per
building, and bounds do not enter a prediction -- the parameter names,
the fixed values and the ventilation assumption are shared, which is
exactly the arrangement `open_test_set.py` already relies on.
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

from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)

from cooling_twin.analysis.residual import (  # noqa: E402
    MIN_STRUCTURE_RATIO,
    ResidualDecomposition,
    decompose_residual,
    linear_residual_slopes,
)
from cooling_twin.calibration.metrics import ashrae_g14_pass  # noqa: E402

logger = logging.getLogger("analyse_residuals")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_1_residual_decomposition.png")

# Groups read from config/buildings.yaml, in report order. The negative
# case is INCLUDED deliberately: a building whose model failed is the
# one whose residual has the most to say, and excluding it would leave
# the decomposition demonstrated only where it has least to find.
BUILDING_GROUPS = ("primary", "generalisation", "negative_case")

# Drivers in a fixed column order for the figure, so the same driver is
# always the same panel across buildings and across runs.
DRIVER_ORDER = (
    "month",
    "hour_of_day",
    "predicted_load",
    "outdoor_dry_bulb",
    "humidity_ratio",
)

PASS_COLOUR = "#2171b5"
FAIL_COLOUR = "#b2182b"
ZERO_COLOUR = "#525252"


def selected_buildings(path: Path) -> list[dict[str, str]]:
    """Every building with a calibration to analyse, primary first.

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
    if not selection.get("primary"):
        raise ValueError(f"{path} declares no primary building")
    return [
        {"building_id": entry["building_id"], "site_id": entry["site_id"], "role": role}
        for role in BUILDING_GROUPS
        for entry in (selection.get(role) or [])
    ]


class IncompatibleArtifactError(Exception):
    """Raised when an artifact belongs to a different model structure.

    Its own exception type, not a `ValueError`, so `main` can skip a
    building whose only calibration predates the current model while
    still failing hard on a wrong-year artifact. Both would be
    `ValueError` otherwise, and one of them must never be swallowed.
    """


def frozen_parameters(
    directory: Path, building_id: str, train_year: int, names: tuple[str, ...]
) -> tuple[dict[str, float], str]:
    """Read one building's calibrated parameters from its artifact.

    Read, never re-derived: the residual analysed here must belong to the
    parameter set that was actually reported, not to whatever a fresh run
    would land on today.

    An artifact whose parameter NAMES do not match the config's is from a
    different model structure and is skipped, newest first. Theodore is
    exactly this case -- its only run is the superseded 4-parameter fit
    from before ADR-011 added `vent_flow_kg_per_s`, and simulating it
    with today's 5-parameter model is not possible. Matching on names
    rather than on count is deliberate: two structures can have the same
    parameter count and mean entirely different things.

    Args:
        directory: Where calibration artifacts live.
        building_id: The building.
        train_year: The only year an artifact may have been fitted on.
        names: The parameter names the current model expects.

    Returns:
        `(parameters, artifact_name)`.

    Raises:
        FileNotFoundError: If the building has no artifact at all.
        IncompatibleArtifactError: If it has artifacts, but none matches
            the current model's parameter set.
        ValueError: If a matching artifact was not fitted on the
            training year.
    """
    found_any = False
    for path in sorted(directory.glob("calibration_*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record.get("metadata", {})
        if metadata.get("building_id") != building_id:
            continue
        found_any = True
        parameters = {name: float(value) for name, value in record["parameters"].items()}
        if set(parameters) != set(names):
            logger.info(
                "%s: skipping %s -- fitted %s, current model expects %s",
                building_id,
                path.name,
                sorted(parameters),
                sorted(names),
            )
            continue
        if int(metadata.get("year", -1)) != train_year:
            raise ValueError(
                f"{path.name} was fitted on {metadata.get('year')}, not the "
                f"training year {train_year}. Investigate before trusting "
                "anything downstream of it."
            )
        return parameters, path.name
    if found_any:
        raise IncompatibleArtifactError(
            f"{building_id} has calibration artifacts, but none was fitted "
            f"with the current parameter set {sorted(names)}"
        )
    raise FileNotFoundError(f"no calibration artifact for {building_id} in {directory}")


def analyse(
    building: dict[str, str], config: dict[str, Any], artifacts: Path
) -> tuple[ResidualDecomposition, dict[str, Any]]:
    """Decompose one building's training-year residual.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.

    Returns:
        `(decomposition, record)` -- the decomposition for plotting, and
        a JSON-shaped record of everything worth keeping.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(
        artifacts, building_id, train_year, names
    )

    frame, floor_area_m2 = load_training_data(building_id, site_id, train_year)
    objective = CalibrationObjective(
        (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
        frame["airTemperature"].to_numpy(dtype=float),
        frame["load_kwh"].to_numpy(dtype=float),
        floor_area_m2,
        config,
        outdoor_humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
    )
    vector = np.array([parameters[name] for name in names], dtype=float)
    predicted, _raw = objective.predict(vector)
    measured = objective.observed_kw

    # Recomputed rather than read from the artifact. If this disagrees
    # with the reported number, the residual being decomposed is not the
    # residual that was reported, and every finding below is about the
    # wrong model.
    verdict = ashrae_g14_pass(measured, predicted, n_params=len(names))

    decomposition = decompose_residual(
        frame.index,
        measured,
        predicted,
        t_ambient_c=frame["airTemperature"].to_numpy(dtype=float),
        humidity_ratio_kg_per_kg=frame["humidity_ratio"].to_numpy(dtype=float),
        label=f"{building_id} {train_year}",
    )
    slopes = linear_residual_slopes(
        decomposition.residual_kw,
        frame["airTemperature"].to_numpy(dtype=float),
        frame["humidity_ratio"].to_numpy(dtype=float),
    )

    logger.info(
        "--- %s (%s) -- TRAINING year CV(RMSE) %.2f%% NMBE %+.2f%% [%s] ---\n%s",
        building_id,
        building["role"],
        verdict.cvrmse_pct,
        verdict.nmbe_pct,
        "PASS" if verdict.passed else "FAIL",
        decomposition.summary().to_string(),
    )
    logger.info(
        "%s residual slopes: %+.2f kW/K, %+.2f kW per g/kg (mean %+.1f kW)",
        building_id,
        slopes["slope_kw_per_K"],
        slopes["slope_kw_per_g_per_kg"],
        slopes["mean_residual_kw"],
    )

    record = {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "parameters": parameters,
        "year": train_year,
        "hours": int(measured.size),
        "mean_measured_kw": decomposition.mean_measured_kw,
        "mean_residual_kw": decomposition.mean_residual_kw,
        "train_cvrmse_pct": verdict.cvrmse_pct,
        "train_nmbe_pct": verdict.nmbe_pct,
        "train_passed": verdict.passed,
        "structured_drivers": list(decomposition.structured_drivers),
        "slopes": slopes,
        "profiles": {
            name: {
                "unit": profile.unit,
                "explained_fraction": profile.explained_fraction,
                "noise_floor": profile.noise_floor,
                "structure_ratio": profile.structure_ratio,
                "swing_kw": profile.swing_kw,
                "swing_pct_of_mean_load": profile.swing_pct_of_mean_load,
                "structured": profile.structured,
                "centres": profile.centres.tolist(),
                "counts": profile.counts.tolist(),
                "means": profile.means.tolist(),
                "sems": profile.sems.tolist(),
            }
            for name, profile in decomposition.profiles.items()
        },
    }
    return decomposition, record


def plot_decompositions(
    decompositions: list[tuple[dict[str, Any], ResidualDecomposition]],
    path: Path,
) -> Path:
    """One row per building, one column per driver, shared y within a row.

    Sharing y ACROSS a row and not across the figure is the whole point:
    within one building the five drivers must be comparable at a glance,
    but Claude's mean load is six times Cathleen's and a shared figure
    scale would flatten the smaller building into a straight line.
    """
    n_rows, n_cols = len(decompositions), len(DRIVER_ORDER)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 2.9 * n_rows),
        sharey="row",
        squeeze=False,
    )

    for row, (record, decomposition) in enumerate(decompositions):
        for column, driver in enumerate(DRIVER_ORDER):
            axis = axes[row][column]
            profile = decomposition.profiles[driver]
            colour = FAIL_COLOUR if profile.structured else PASS_COLOUR
            axis.errorbar(
                profile.centres,
                profile.means,
                yerr=profile.sems,
                marker="o",
                markersize=3,
                linewidth=1.6,
                capsize=2,
                color=colour,
            )
            axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
            # The ratio is on every panel because a shape that looks
            # dramatic on a tight y-axis may be entirely inside its own
            # noise floor, and the eye cannot tell.
            axis.set_title(
                f"{driver}  (x{profile.structure_ratio:.0f})",
                fontsize=9,
                color=colour,
            )
            axis.set_xlabel(profile.unit, fontsize=8)
            axis.tick_params(labelsize=8)
            axis.grid(alpha=0.3)
        verdict = "PASS" if record["train_passed"] else "FAIL"
        axes[row][0].set_ylabel(
            f"{record['building_id']}\nresidual, kW\n"
            f"(train CV {record['train_cvrmse_pct']:.1f}% {verdict})",
            fontsize=8,
        )

    figure.suptitle(
        "Where each model's error lives -- training year, frozen parameters. "
        f"Red = structured (>= {MIN_STRUCTURE_RATIO:.0f}x its noise floor). "
        "Above zero = the model under-predicts."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: decompose every building's residual, log, record."""
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
    train_year = int(config["train_year"])

    results = []
    for building in selected_buildings(arguments.buildings):
        try:
            decomposition, record = analyse(building, config, artifacts)
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            # A building listed for the record but never calibrated with
            # the current model is expected -- Theodore's only run is the
            # superseded 4-parameter fit -- and is not a reason to
            # abandon the buildings that follow it.
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        results.append((record, decomposition))

    if not results:
        raise SystemExit("no building had a calibration artifact to analyse")

    ranking = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "train CV %": round(record["train_cvrmse_pct"], 2),
                "mean residual kW": round(record["mean_residual_kw"], 1),
                **{
                    driver: round(record["profiles"][driver]["structure_ratio"], 1)
                    for driver in DRIVER_ORDER
                },
                "kW/K": round(record["slopes"]["slope_kw_per_K"], 1),
                "kW per g/kg": round(record["slopes"]["slope_kw_per_g_per_kg"], 1),
            }
            for record, _ in results
        ]
    ).set_index("building")
    logger.info(
        "--- structure ratio by driver (x noise floor; >= %.0f = structured) ---\n%s",
        MIN_STRUCTURE_RATIO,
        ranking.to_string(),
    )

    plot_decompositions(results, arguments.figure)

    out_path = artifacts / f"residuals_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "note": (
                    "Training year only (ADR-002). Parameters read frozen from "
                    "the calibration artifacts; the optimiser is not imported."
                ),
                "min_structure_ratio": MIN_STRUCTURE_RATIO,
                "buildings": [record for record, _ in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
