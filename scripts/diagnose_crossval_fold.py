"""Diagnose why one cross-validation fold fails (L6.9 follow-up).

    python scripts/diagnose_crossval_fold.py --fold 2

Fold 2 of the Fox_education_Claude 4-fold run validates on 26 May to
7 August -- peak Tempe summer -- having trained on January to May only.
It scores CV(RMSE) 15.68% (inside G14's 30%) with NMBE +11.93%, which is
OUTSIDE the +-10% bias limit: the model under-predicts summer cooling by
roughly 12%.

Three explanations are live and this script is built to separate them
rather than to argue about them:

    H1  capacity      -- `vent_flow` and `ua_envelope` are pinned at
                         their ceilings, so the weather-driven terms
                         cannot grow enough for a Tempe summer.
    H2  extrapolation -- the constant term (`internal_gain`, traded off
                         against `t_setpoint`) was fitted on cool months
                         and is simply too small; nothing structural.
    H3  structure     -- a reheat load (Q8) that the model has no term
                         for, and which does not scale with any driver
                         the model has.

The discriminating experiments, in order of decisiveness:

    1. Score the SAME window with the FULL-YEAR parameters. If the bias
       disappears, this model structure CAN represent a Tempe summer and
       the fold's problem is the fit, not the form (H2 over H1/H3).
    2. Swap one fold-2 parameter at a time to its full-year value and
       re-score. Whichever swap removes the bias names the parameter
       responsible.
    3. Decompose the residual against the drivers the model already has
       (temperature, humidity) and one it does not (the hot-water meter,
       Q8's reheat proxy). Structure left in the residual against a
       driver the model HAS is a parameter problem; structure against
       one it does NOT have is H3.

ADR-002: 2016 only. Nothing here touches the held-out year.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
    resolve_bounds,
)

from cooling_twin.calibration.crossval import (  # noqa: E402
    TimeFold,
    expanding_window_folds,
)
from cooling_twin.calibration.metrics import cvrmse, nmbe  # noqa: E402
from cooling_twin.data.load import load_meter  # noqa: E402

logger = logging.getLogger("diagnose_fold")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l6_9_fold_residuals.png")

# Meteorological seasons, by month. Named here rather than inline so the
# season table and the figure cannot disagree about which months are
# "summer" on a site whose cooling season runs well past September.
SEASONS = {
    "winter (DJF)": (12, 1, 2),
    "spring (MAM)": (3, 4, 5),
    "summer (JJA)": (6, 7, 8),
    "autumn (SON)": (9, 10, 11),
}

RESIDUAL_COLOUR = "#2171b5"
BIN_COLOUR = "#b2182b"
ZERO_COLOUR = "#525252"

# Outdoor-temperature bins for the residual trend line, degC. Wide
# enough that each bin holds hundreds of hours on this site.
TEMP_BIN_EDGES = np.arange(0.0, 50.0, 2.5)


def fold_residuals(
    objective: CalibrationObjective,
    parameters: dict[str, float],
    names: tuple[str, ...],
    fold: TimeFold,
) -> npt.NDArray[np.float64]:
    """Measured minus predicted over a fold's SCORED window.

    Spin-up is simulated and discarded, exactly as `cross_validate`
    does -- a diagnostic that scored the spin-up hours would be
    diagnosing the initial condition.

    Args:
        objective: The calibration objective, holding the drivers.
        parameters: Parameter set to simulate.
        names: Parameter order.
        fold: The fold whose validation window is scored.

    Returns:
        Residuals in kW, G14 sign convention (measured - predicted).
    """
    vector = np.array([parameters[name] for name in names], dtype=float)
    simulated, _raw = objective.predict(vector, window=fold.simulate_slice)
    predicted = simulated[fold.scored_offset :]
    return objective.observed_kw[fold.validate_slice] - predicted


def score_window(
    objective: CalibrationObjective,
    parameters: dict[str, float],
    names: tuple[str, ...],
    fold: TimeFold,
    n_params: int,
) -> tuple[float, float]:
    """`(cvrmse_pct, nmbe_pct)` for one parameter set on one fold."""
    vector = np.array([parameters[name] for name in names], dtype=float)
    simulated, _raw = objective.predict(vector, window=fold.simulate_slice)
    predicted = simulated[fold.scored_offset :]
    measured = objective.observed_kw[fold.validate_slice]
    return (
        cvrmse(measured, predicted, n_params),
        nmbe(measured, predicted, n_params),
    )


def season_table(
    index: pd.DatetimeIndex,
    measured: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
    n_params: int,
) -> pd.DataFrame:
    """Score one prediction series season by season.

    Args:
        index: Timestamps aligned with the series.
        measured: Measured load, kW.
        predicted: Model output, kW.
        n_params: Calibrated parameter count.

    Returns:
        One row per season with hours, mean load, CV(RMSE) and NMBE.
    """
    rows = []
    for label, months in SEASONS.items():
        mask = np.isin(index.month.to_numpy(), months)
        if not mask.any():
            continue
        rows.append(
            {
                "season": label,
                "hours": int(mask.sum()),
                "mean kW": round(float(measured[mask].mean()), 0),
                "CV(RMSE) %": round(cvrmse(measured[mask], predicted[mask], n_params), 2),
                "NMBE %": round(nmbe(measured[mask], predicted[mask], n_params), 2),
            }
        )
    return pd.DataFrame(rows).set_index("season")


def decompose_residual(
    residual_kw: npt.NDArray[np.float64],
    t_ambient_c: npt.NDArray[np.float64],
    humidity_ratio: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Regress the residual on the drivers the model already has.

    A residual that is a flat positive offset says the CONSTANT term is
    too small. One that slopes with temperature says a weather-driven
    term is too small. One that slopes with humidity says the latent
    term is. The three are separable because the regression reports all
    three coefficients at once, and the mean is reported beside them so
    a large slope on a small mean is not mistaken for the main effect.

    Args:
        residual_kw: Measured minus predicted, kW.
        t_ambient_c: Outdoor dry bulb, degC.
        humidity_ratio: Outdoor humidity ratio, kg/kg.

    Returns:
        Intercept, slopes, and the correlations behind them.
    """
    design = np.column_stack(
        [np.ones_like(residual_kw), t_ambient_c, humidity_ratio * 1000.0]
    )
    coefficients, *_ = np.linalg.lstsq(design, residual_kw, rcond=None)
    return {
        "mean_residual_kw": float(residual_kw.mean()),
        "intercept_kw": float(coefficients[0]),
        "slope_kw_per_K": float(coefficients[1]),
        "slope_kw_per_g_per_kg": float(coefficients[2]),
        "corr_temperature": float(np.corrcoef(residual_kw, t_ambient_c)[0, 1]),
        "corr_humidity": float(np.corrcoef(residual_kw, humidity_ratio)[0, 1]),
    }


def hot_water_correlation(
    building_id: str, index: pd.DatetimeIndex, residual_kw: npt.NDArray[np.float64]
) -> float | None:
    """Correlate the residual with the hot-water meter (Q8's reheat proxy).

    The Fox hot-water meter's UNITS are wrong (Q8: Claude's mean reads
    3,215 W/m2), so only timing and correlation are usable -- never
    magnitude. A correlation is exactly what is wanted here: if the
    unexplained cooling rises when the building is also heating, that is
    evidence for a reheat term the model does not have.

    Args:
        building_id: BDG2 identifier.
        index: Timestamps of the residual.
        residual_kw: Measured minus predicted, kW.

    Returns:
        Pearson correlation, or None if the meter has no overlapping
        data.
    """
    meter = load_meter("hotwater")
    meter = meter[meter["building_id"] == building_id]
    if meter.empty:
        return None
    series = meter.set_index("timestamp")["meter_reading"].sort_index()
    aligned = series.reindex(index)
    usable = aligned.notna().to_numpy() & np.isfinite(residual_kw)
    if usable.sum() < 100 or aligned[usable].std() == 0:
        return None
    return float(np.corrcoef(residual_kw[usable], aligned[usable].to_numpy())[0, 1])


def plot_residuals(
    index: pd.DatetimeIndex,
    residuals: dict[int, tuple[TimeFold, npt.NDArray[np.float64]]],
    t_ambient_c: npt.NDArray[np.float64],
    failing_fold: int,
    building_id: str,
    path: Path,
) -> Path:
    """Residual against outdoor temperature, one panel per fold."""
    figure, axes = plt.subplots(
        1, len(residuals), figsize=(4.2 * len(residuals), 4.4), sharey=True
    )
    for axis, (number, (fold, residual)) in zip(axes, sorted(residuals.items()), strict=True):
        temperature = t_ambient_c[fold.validate_slice]
        axis.scatter(temperature, residual, s=4, alpha=0.25, color=RESIDUAL_COLOUR)
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)

        # Binned means: the scatter shows spread, the line shows bias,
        # and bias is the thing under investigation.
        which = np.digitize(temperature, TEMP_BIN_EDGES)
        centres, means = [], []
        for bin_index in np.unique(which):
            mask = which == bin_index
            if mask.sum() >= 20:
                centres.append(temperature[mask].mean())
                means.append(residual[mask].mean())
        axis.plot(centres, means, color=BIN_COLOUR, marker="o", markersize=4, linewidth=2)

        start, stop = index[fold.validate_start], index[fold.validate_stop - 1]
        axis.set_title(
            f"fold {number}{'  -- G14 FAIL' if number == failing_fold else ''}\n"
            f"{start:%d %b} to {stop:%d %b}, mean residual {residual.mean():+.0f} kW",
            fontsize=10,
            color=BIN_COLOUR if number == failing_fold else "black",
        )
        axis.set_xlabel("outdoor dry bulb, degC")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("residual (measured - predicted), kW")
    figure.suptitle(
        f"Where each fold's error lives: {building_id} 2016 "
        "(above zero = the model under-predicts)"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: reproduce the folds, then run the three experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fold", type=int, default=2, help="Fold under investigation.")
    parser.add_argument("--n-folds", type=int, default=4)
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
    bounds = resolve_bounds(config, frame, floor_area_m2)
    names = tuple(bounds)
    n_params = len(names)
    index = frame.index
    measured = objective.observed_kw
    t_ambient = frame["airTemperature"].to_numpy(dtype=float)
    humidity = frame["humidity_ratio"].to_numpy(dtype=float)

    artifacts = Path(config["artifacts"]["directory"])
    crossval = json.loads(
        (
            artifacts
            / f"crossval_{config['building_id']}_{year}_{arguments.n_folds}fold.json"
        ).read_text(encoding="utf-8")
    )
    fold_parameters = {entry["fold"]: entry["parameters"] for entry in crossval["folds"]}
    folds = {
        fold.number: fold
        for fold in expanding_window_folds(len(measured), n_folds=arguments.n_folds)
    }
    target = folds[arguments.fold]

    full_year = _full_year_parameters(artifacts, config["building_id"])
    logger.info("full-year parameters: %s", {k: round(v, 3) for k, v in full_year.items()})

    # --- 1. parameters, fold by fold ------------------------------------
    comparison = pd.DataFrame(fold_parameters).T
    comparison.index.name = "fold"
    comparison.loc["full year"] = pd.Series(full_year)
    comparison["bound"] = ""
    logger.info(
        "--- 1. calibrated parameters by fold ---\n%s",
        comparison.round(2).to_string(),
    )
    logger.info(
        "bounds: %s",
        {name: (round(low, 2), round(high, 1)) for name, (low, high) in bounds.items()},
    )

    # --- 2. residual against temperature, per fold ----------------------
    residuals = {
        number: (fold, fold_residuals(objective, fold_parameters[number], names, fold))
        for number, fold in folds.items()
    }
    plot_residuals(
        index, residuals, t_ambient, arguments.fold, config["building_id"], arguments.figure
    )

    # --- 3. seasons, under the full-year fit ----------------------------
    full_vector = np.array([full_year[name] for name in names], dtype=float)
    full_predicted, _raw = objective.predict(full_vector)
    logger.info(
        "--- 2. the FULL-YEAR fit, scored season by season ---\n%s",
        season_table(index, measured, full_predicted, n_params).to_string(),
    )

    # --- 4. the decisive swap -------------------------------------------
    own_cv, own_nmbe = score_window(
        objective, fold_parameters[arguments.fold], names, target, n_params
    )
    full_cv, full_nmbe = score_window(objective, full_year, names, target, n_params)
    logger.info(
        "--- 3. the same window, two parameter sets ---\n"
        "  fold-%d parameters   CV(RMSE) %6.2f%%  NMBE %+7.2f%%\n"
        "  full-year parameters CV(RMSE) %6.2f%%  NMBE %+7.2f%%",
        arguments.fold,
        own_cv,
        own_nmbe,
        full_cv,
        full_nmbe,
    )

    # --- 5. one parameter at a time -------------------------------------
    rows = []
    for name in names:
        swapped = dict(fold_parameters[arguments.fold])
        swapped[name] = full_year[name]
        swap_cv, swap_nmbe = score_window(objective, swapped, names, target, n_params)
        rows.append(
            {
                "swapped to full-year value": name,
                "from": round(fold_parameters[arguments.fold][name], 2),
                "to": round(full_year[name], 2),
                "CV(RMSE) %": round(swap_cv, 2),
                "NMBE %": round(swap_nmbe, 2),
                "bias removed pp": round(abs(own_nmbe) - abs(swap_nmbe), 2),
            }
        )
    ablation = pd.DataFrame(rows).set_index("swapped to full-year value")
    logger.info(
        "--- 4. swap ONE fold-%d parameter to its full-year value ---\n%s",
        arguments.fold,
        ablation.sort_values("bias removed pp", ascending=False).to_string(),
    )

    # --- 6. what is left in the residual --------------------------------
    target_residual = residuals[arguments.fold][1]
    parts = decompose_residual(
        target_residual,
        t_ambient[target.validate_slice],
        humidity[target.validate_slice],
    )
    logger.info(
        "--- 5. fold-%d residual against the drivers the model HAS ---\n%s",
        arguments.fold,
        "\n".join(f"  {key:<26}{value:>12.4f}" for key, value in parts.items()),
    )

    # The bias and the SHAPE of the error are different faults. Removing
    # the mean bias with one parameter does not mean the residual has
    # gone -- if it still slopes with temperature afterwards, the model's
    # weather response is too flat and that is a separate finding.
    best_swap = ablation["bias removed pp"].idxmax()
    repaired = dict(fold_parameters[arguments.fold])
    repaired[best_swap] = full_year[best_swap]
    repaired_residual = fold_residuals(objective, repaired, names, target)
    repaired_parts = decompose_residual(
        repaired_residual,
        t_ambient[target.validate_slice],
        humidity[target.validate_slice],
    )
    logger.info(
        "--- 5b. the same residual AFTER swapping %s ---\n%s",
        best_swap,
        "\n".join(f"  {key:<26}{value:>12.4f}" for key, value in repaired_parts.items()),
    )

    reheat = hot_water_correlation(
        config["building_id"], index[target.validate_slice], target_residual
    )
    logger.info(
        "--- 6. against a driver the model does NOT have (Q8 reheat proxy) ---\n"
        "  corr(residual, hotwater) = %s",
        "unavailable" if reheat is None else f"{reheat:+.3f}",
    )

    record = {
        "building_id": config["building_id"],
        "year": year,
        "fold": arguments.fold,
        "fold_window": [
            str(index[target.validate_start]),
            str(index[target.validate_stop - 1]),
        ],
        "own_parameters": {"cvrmse_pct": own_cv, "nmbe_pct": own_nmbe},
        "full_year_parameters_same_window": {"cvrmse_pct": full_cv, "nmbe_pct": full_nmbe},
        "ablation": ablation.reset_index().to_dict(orient="records"),
        "residual_decomposition": parts,
        "best_swap": best_swap,
        "residual_decomposition_after_swap": repaired_parts,
        "corr_residual_hotwater": reheat,
        "season_table_full_year_fit": season_table(
            index, measured, full_predicted, n_params
        ).reset_index().to_dict(orient="records"),
    }
    out_path = artifacts / f"fold{arguments.fold}_diagnosis_{config['building_id']}_{year}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("artifact: %s", out_path)


def _full_year_parameters(directory: Path, building_id: str) -> dict[str, float]:
    """Parameters from the most recent full-year calibration."""
    for path in sorted(directory.glob("calibration_*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("metadata", {}).get("building_id") == building_id:
            return {name: float(value) for name, value in record["parameters"].items()}
    raise FileNotFoundError(f"no calibration artifact for {building_id} in {directory}")


if __name__ == "__main__":
    main()
