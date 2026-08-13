"""Cross-validate the calibration inside the training year (L6.9b).

    python scripts/run_crossval.py --config config/calibration.yaml

Fits the model independently on each expanding-window fold of the
TRAINING year and scores it on the block that follows. The gap between
training and validation CV(RMSE) is the estimate of how the calibration
would behave on a year it has not seen -- produced without spending
2017, which is opened once, at L6.10 (ADR-002).

Every fold is calibrated FROM SCRATCH. Warm-starting from the full-year
optimum would be five times faster and would leak the validation window
into the fit through the starting point; `cross_validate`'s signature
makes it impossible to do by accident, and this script does not do it
deliberately either.

The objective, the data pipeline and the bound resolution are imported
from `run_calibration.py` -- one definition, so a fold is scored by
exactly the machinery that produced the headline number it qualifies.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
    resolve_bounds,
)

from cooling_twin.calibration.crossval import (  # noqa: E402
    DEFAULT_SPIN_UP_HOURS,
    OVERFITTING_GAP_PCT,
    CrossValidationResult,
    TimeFold,
    cross_validate,
    expanding_window_folds,
    neighbour_leak_test,
    random_folds,
)
from cooling_twin.calibration.optimize import calibrate  # noqa: E402

logger = logging.getLogger("run_crossval")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l6_9_crossval.png")

# Single hue, light to dark, for training vs validation: the pair is
# ordered (one is a subset of what the other measures), not categorical,
# and the bars are directly labelled so nothing rests on colour alone.
TRAIN_COLOUR = "#c6dbef"
VALIDATE_COLOUR = "#2171b5"
REFERENCE_COLOUR = "#b2182b"


class FoldCalibration:
    """Calibrates and predicts one fold, sharing L6.6's objective.

    Attributes:
        bounds: The box every fold searches -- the same one the
            headline calibration used.
        n_evaluations: Objective calls across all folds so far.
    """

    def __init__(
        self,
        objective: CalibrationObjective,
        bounds: dict[str, tuple[float, float]],
        optimiser: dict[str, Any],
    ) -> None:
        self._objective = objective
        self.bounds = bounds
        self._optimiser = optimiser
        self.n_evaluations = 0
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        """One artifact-shaped record per fold, for the run log."""
        return self._records

    def fit(self, fold: TimeFold) -> dict[str, float]:
        """Run the full two-stage search on this fold's training window.

        Args:
            fold: The partition being fitted. Only `train_slice` is
                read -- the fold carries no information about the
                window it will be scored on.

        Returns:
            The fitted parameters.
        """
        window = FoldObjective(self._objective, fold.train_slice)
        result = calibrate(
            window,
            self.bounds,
            popsize=int(self._optimiser["popsize"]),
            maxiter=int(self._optimiser["maxiter"]),
            tol=float(self._optimiser["tol"]),
            workers=int(self._optimiser["workers"]),
            metadata={"fold": fold.number, "train_hours": fold.n_train},
        )
        self.n_evaluations += result.n_evaluations
        self._records.append(
            {
                "fold": fold.number,
                "train_hours": fold.n_train,
                "validate_hours": fold.n_validate,
                "objective": result.objective_value,
                "parameters": result.parameters,
                "pinned_parameters": list(result.pinned_parameters),
                "n_evaluations": result.n_evaluations,
                "elapsed_seconds": round(result.elapsed_seconds, 1),
            }
        )
        return result.parameters

    def predict(
        self, parameters: dict[str, float], window: slice
    ) -> npt.NDArray[np.float64]:
        """Simulate one parameter set over an arbitrary window."""
        vector = np.array([parameters[name] for name in self.bounds], dtype=float)
        clipped_kw, _raw = self._objective.predict(vector, window=window)
        return clipped_kw


class FoldObjective:
    """The L6.6 objective restricted to one window, picklable.

    A class rather than a lambda for the same reason as
    `CalibrationObjective`: `calibrate(workers=-1)` hands this to a
    process pool, and a closure cannot be pickled.
    """

    def __init__(self, objective: CalibrationObjective, window: slice) -> None:
        self._objective = objective
        self._window = window

    def __call__(self, vector: npt.NDArray[np.float64]) -> float:
        """Score a parameter vector on this window alone."""
        return self._objective.evaluate(vector, window=self._window)


def plot_folds(
    result: CrossValidationResult,
    bounds: dict[str, tuple[float, float]],
    full_year_cvrmse_pct: float,
    building_id: str,
    path: Path,
) -> Path:
    """Two panels: the gap per fold, and each parameter across folds."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    positions = np.arange(len(result.scores))
    width = 0.38
    axes[0].bar(
        positions - width / 2,
        [score.train_cvrmse_pct for score in result.scores],
        width,
        color=TRAIN_COLOUR,
        edgecolor="white",
        label="train (own fold)",
    )
    axes[0].bar(
        positions + width / 2,
        [score.validate_cvrmse_pct for score in result.scores],
        width,
        color=VALIDATE_COLOUR,
        edgecolor="white",
        label="validate (unseen block)",
    )
    axes[0].axhline(
        full_year_cvrmse_pct,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=1.5,
        label=f"full-year fit ({full_year_cvrmse_pct:.2f}%)",
    )
    axes[0].axhline(30.0, color="k", linestyle=":", linewidth=1, label="G14 hourly limit")
    for index, score in enumerate(result.scores):
        # The gap AND the G14 verdict. A fold can have a small gap and
        # still fail the standard, so the panel must not let the gap
        # alone read as a pass.
        axes[0].annotate(
            f"{score.gap_pct:+.1f} pp"
            + ("" if score.passes_g14() else "\nG14 FAIL (NMBE)"),
            (index, max(score.train_cvrmse_pct, score.validate_cvrmse_pct)),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            color="black" if score.passes_g14() else REFERENCE_COLOUR,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        [f"fold {score.fold.number}\n{score.fold.n_train} h train" for score in result.scores]
    )
    axes[0].set_ylabel("CV(RMSE), %")
    axes[0].set_title("1. does it hold up on hours it never saw?")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2 -- each parameter's fitted value per fold, normalised to
    # its own bounds so five different units share one axis.
    stability = result.parameter_stability()
    for name in result.parameter_names:
        low, high = bounds[name]
        axes[1].plot(
            [score.fold.number for score in result.scores],
            [
                (score.parameters[name] - low) / (high - low)
                for score in result.scores
            ],
            marker="o",
            label=f"{name} ({stability[name][0]:.3g}–{stability[name][1]:.3g})",
        )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xticks([score.fold.number for score in result.scores])
    axes[1].set_xlabel("fold")
    axes[1].set_ylabel("fitted value, fraction of its own bounds")
    axes[1].set_title(
        "2. do the parameters stay put?\n(flat = stable, sloped = tracking the season)"
    )
    axes[1].legend(fontsize=7, loc="best")
    axes[1].grid(alpha=0.3)

    figure.suptitle(
        f"Within-2016 cross-validation: {building_id} "
        f"(mean gap {result.mean_gap_pct:+.2f} pp)"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: load, split, cross-validate, report, write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--spin-up-hours", type=int, default=DEFAULT_SPIN_UP_HOURS)
    parser.add_argument("--embargo-hours", type=int, default=0)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    year = int(config["train_year"])
    frame, floor_area_m2 = load_training_data(config["building_id"], config["site_id"], year)

    objective = CalibrationObjective(
        (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
        frame["airTemperature"].to_numpy(dtype=float),
        frame["load_kwh"].to_numpy(dtype=float),
        floor_area_m2,
        config,
        outdoor_humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
    )
    measured = objective.observed_kw
    bounds = resolve_bounds(config, frame, floor_area_m2)

    folds = expanding_window_folds(
        len(measured),
        n_folds=arguments.n_folds,
        spin_up_hours=arguments.spin_up_hours,
        embargo_hours=arguments.embargo_hours,
    )

    # The leak check on THIS building's real load, before anything is
    # fitted: if the blocked split were as easy as a random one, the
    # cross-validation below would be measuring smoothness.
    scattered = random_folds(len(measured), n_folds=arguments.n_folds)
    blocked = [np.arange(f.validate_start, f.validate_stop) for f in folds]
    leak_random = neighbour_leak_test(measured, scattered)
    leak_blocked = neighbour_leak_test(measured, blocked)
    logger.info(
        "leak check on real load -- knowledge-free model scores %.2f%% under a "
        "RANDOM split (%s G14) and %.2f%% under the blocked split used below",
        leak_random,
        "passes" if leak_random <= 30.0 else "fails",
        leak_blocked,
    )

    runner = FoldCalibration(objective, bounds, config["optimiser"])
    result = cross_validate(
        runner.fit,
        runner.predict,
        measured,
        folds,
        n_params=len(bounds),
        overfitting_gap_pct=OVERFITTING_GAP_PCT,
    )

    logger.info("--- folds ---\n%s", result.summary())
    logger.info(
        "--- parameter stability across folds ---\n%s",
        "\n".join(
            f"  {name:<28}{low:12.3f} .. {high:10.3f}"
            f"   {100 * (high - low) / (bounds[name][1] - bounds[name][0]):6.1f}% of bounds"
            for name, (low, high) in result.parameter_stability().items()
        ),
    )

    record = {
        "building_id": config["building_id"],
        "year": year,
        "n_folds": arguments.n_folds,
        "spin_up_hours": arguments.spin_up_hours,
        "embargo_hours": arguments.embargo_hours,
        "leak_check": {"random_split_pct": leak_random, "blocked_split_pct": leak_blocked},
        "overfitting_gap_pct": result.overfitting_gap_pct,
        "mean_validate_cvrmse_pct": result.mean_validate_cvrmse_pct,
        "worst_validate_cvrmse_pct": result.worst_validate_cvrmse_pct,
        "folds_failing_g14": list(result.folds_failing_g14),
        "mean_gap_pct": result.mean_gap_pct,
        "overfitting_suspected": result.overfitting_suspected,
        "folds": [
            {
                **fold_record,
                "train_cvrmse_pct": score.train_cvrmse_pct,
                "validate_cvrmse_pct": score.validate_cvrmse_pct,
                "train_nmbe_pct": score.train_nmbe_pct,
                "validate_nmbe_pct": score.validate_nmbe_pct,
                "gap_pct": score.gap_pct,
            }
            for fold_record, score in zip(runner.records, result.scores, strict=True)
        ],
        "parameter_stability": {
            name: list(pair) for name, pair in result.parameter_stability().items()
        },
        "total_evaluations": runner.n_evaluations,
    }

    # The full-year fit, for the dashed reference line: what the
    # headline number is, scored on the data it was fitted to.
    artifacts_dir = Path(config["artifacts"]["directory"])
    full_year = _latest_full_year_cvrmse(artifacts_dir, config["building_id"])
    record["full_year_cvrmse_pct"] = full_year

    plot_folds(result, bounds, full_year, config["building_id"], arguments.figure)

    stem = f"crossval_{config['building_id']}_{year}_{arguments.n_folds}fold.json"
    out_path = artifacts_dir / stem
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("artifact: %s", out_path)


def _latest_full_year_cvrmse(directory: Path, building_id: str) -> float:
    """CV(RMSE) of the most recent full-year calibration for this building.

    Args:
        directory: Where `run_calibration.py` writes.
        building_id: The building.

    Returns:
        Its training CV(RMSE), percent.

    Raises:
        FileNotFoundError: If no calibration artifact matches.
    """
    for path in sorted(directory.glob("calibration_*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("metadata", {}).get("building_id") == building_id:
            return float(record["breakdown"]["cvrmse_pct"])
    raise FileNotFoundError(f"no calibration artifact for {building_id} in {directory}")


if __name__ == "__main__":
    main()
