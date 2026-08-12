"""Calibrate the RC model against one building's measured cooling load.

    python scripts/run_calibration.py --config config/calibration.yaml

Runs the L6.7 two-stage search (differential evolution, then L-BFGS-B)
against the objective L6.6 designed, on the TRAINING year only, and
writes a JSON artifact for the run. Every number that steers the run
comes from the config file; nothing is hardcoded here.

The test-set lock (ADR-002) is enforced in code, not by discipline:
this script refuses to run on any year other than `train_year`. That
refusal is the point -- 2017 is opened once, deliberately, at L6.10,
and a script that could quietly evaluate on it is a leak waiting for a
tired evening.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from cooling_twin.calibration.baseline import fit_annual_mean, fit_linear_regression
from cooling_twin.calibration.bounds import internal_gain_upper_bound
from cooling_twin.calibration.metrics import DataInterval, ashrae_g14_pass
from cooling_twin.calibration.optimize import (
    INFEASIBLE_OBJECTIVE,
    ObjectiveBreakdown,
    calibrate,
    clipping_violation,
    g14_objective,
    write_artifact,
)
from cooling_twin.data.load import BDG2_ROOT, load_metadata, load_meter
from cooling_twin.data.quality import load_cleaning_config, run_cleaning_pipeline
from cooling_twin.data.weather import (
    add_psychrometric_features,
    join_weather,
    load_weather,
)
from cooling_twin.models.rc import (
    DEFAULT_SUPPLY_HUMIDITY_RATIO,
    inverse_cooling_load,
)

logger = logging.getLogger("run_calibration")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the calibration config.

    Args:
        path: Path to the calibration YAML.

    Returns:
        The parsed config.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it does not parse to a mapping, or a required
            top-level key is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration config not found at {path}. Bounds and optimiser "
            "settings are never hardcoded (05_ENGINEERING_STANDARDS.md SS2)."
        )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must parse to a YAML mapping")
    missing = {
        "building_id",
        "site_id",
        "train_year",
        "parameters",
        "fixed",
        "objective",
        "optimiser",
        "artifacts",
    } - set(config)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    return config


def load_training_data(building_id: str, site_id: str, year: int) -> tuple[pd.DataFrame, float]:
    """Load, join and clean one building-year, reusing the M2/M3 pipeline.

    Args:
        building_id: BDG2 building identifier.
        site_id: Its site, for the weather join.
        year: The year to load. Must be the configured training year --
            the caller enforces that, see `main`.

    Returns:
        `(frame, floor_area_m2)` where `frame` is indexed by timestamp
        and carries `load_kwh` and `airTemperature`.

    Raises:
        ValueError: If no usable rows survive the cleaning pipeline.
    """
    floor_area_m2 = float(load_metadata().loc[building_id, "sqm"])

    meter = load_meter("chilledwater")
    meter = meter[meter["building_id"] == building_id]
    weather = add_psychrometric_features(load_weather(BDG2_ROOT, site_id=site_id))

    joined = join_weather(meter, weather, site_id).set_index("timestamp").sort_index()
    joined = joined[joined.index.year == year]

    cleaned, _ = run_cleaning_pipeline(
        joined["meter_reading"], joined["rh_pct"], load_cleaning_config()
    )
    frame = joined.assign(load_kwh=cleaned).dropna(
        subset=["load_kwh", "airTemperature", "humidity_ratio"]
    )
    if frame.empty:
        raise ValueError(
            f"no usable rows for {building_id} in {year} after cleaning -- "
            "check the meter/weather join before calibrating anything."
        )

    logger.info(
        "%s %d: %d hours, mean load %.1f kW, floor area %.0f m2",
        building_id,
        year,
        len(frame),
        frame["load_kwh"].mean(),
        floor_area_m2,
    )
    return frame, floor_area_m2


class CalibrationObjective:
    """The L6.6 objective, closed over one building-year.

    Written as a class with `__call__` rather than the obvious nested
    closure for one reason: `calibrate(workers=-1)` hands the objective
    to a process pool, and a pool pickles it. A nested function is not
    picklable, and the failure surfaces as an opaque pickling error from
    inside SciPy rather than as anything resembling the actual problem.
    An instance holding only arrays and plain config values pickles
    fine, and each worker pays the copy once.

    Attributes:
        observed_kw: Measured load, kW, for the calibration window.
    """

    def __init__(
        self,
        t_seconds: npt.NDArray[np.float64],
        t_ambient_c: npt.NDArray[np.float64],
        observed_kw: npt.NDArray[np.float64],
        floor_area_m2: float,
        config: dict[str, Any],
        outdoor_humidity_ratio: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self._t_seconds = t_seconds
        self._t_ambient_c = t_ambient_c
        self._outdoor_humidity_ratio = outdoor_humidity_ratio
        self.observed_kw = observed_kw
        self._floor_area_m2 = floor_area_m2
        self._parameter_names = tuple(config["parameters"])
        self._supply_humidity_ratio = float(
            config.get("ventilation", {}).get(
                "supply_humidity_ratio", DEFAULT_SUPPLY_HUMIDITY_RATIO
            )
        )
        self._capacity_ratio = float(config["fixed"]["envelope_capacity_ratio"])
        self._ceiling_height_m = float(config["fixed"]["ceiling_height_m"])
        objective_config = config["objective"]
        self._interval = DataInterval(objective_config["interval"])
        self._nmbe_weight = float(objective_config["nmbe_weight"])
        self._penalty_weight = float(objective_config["penalty_weight"])
        self._max_clipped_fraction = float(objective_config["max_clipped_fraction"])
        self._n_params = len(config["parameters"])

    def predict(
        self, vector: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Run the inverse model at one parameter vector.

        Args:
            vector: Parameter values in the config's parameter order.
                Looked up BY NAME rather than unpacked positionally, so
                that adding a parameter (ADR-011's ventilation flow) is
                a config change and not a silent re-ordering bug.

        Returns:
            `(clipped_kw, raw_kw)` as `inverse_cooling_load` returns.

        Raises:
            RuntimeError: If the ODE solver fails.
        """
        values = dict(zip(self._parameter_names, (float(v) for v in vector), strict=True))
        vent_flow = values.get("vent_flow_kg_per_s", 0.0)
        return inverse_cooling_load(
            self._t_seconds,
            self._t_ambient_c,
            ua_envelope_w_per_m2k=values["ua_envelope_w_per_m2k"],
            r_internal_ratio=values["r_internal_ratio"],
            internal_gain_w_per_m2=values["internal_gain_w_per_m2"],
            t_setpoint_c=values["t_setpoint_c"],
            floor_area_m2=self._floor_area_m2,
            envelope_capacity_ratio=self._capacity_ratio,
            ceiling_height_m=self._ceiling_height_m,
            vent_flow_kg_per_s=vent_flow,
            outdoor_humidity_ratio=(
                None if vent_flow <= 0.0 else self._outdoor_humidity_ratio
            ),
            supply_humidity_ratio=self._supply_humidity_ratio,
        )

    def breakdown(self, vector: npt.NDArray[np.float64]) -> ObjectiveBreakdown:
        """Decompose the objective at one parameter vector, for the log."""
        clipped_kw, raw_kw = self.predict(vector)
        return g14_objective(
            self.observed_kw,
            clipped_kw,
            n_params=self._n_params,
            violations={
                "clipped_hours": clipping_violation(
                    raw_kw, max_clipped_fraction=self._max_clipped_fraction
                )
            },
            interval=self._interval,
            nmbe_weight=self._nmbe_weight,
            penalty_weight=self._penalty_weight,
        )

    def __call__(self, vector: npt.NDArray[np.float64]) -> float:
        """The scalar the optimiser minimises."""
        try:
            return self.breakdown(vector).total
        except RuntimeError as error:
            # A parameter set the solver cannot integrate is infeasible,
            # not fatal: the optimiser must be able to walk away from it.
            logger.debug("infeasible candidate %s: %s", vector, error)
            return INFEASIBLE_OBJECTIVE


AUTO_BOUND = "auto"


def resolve_bounds(
    config: dict[str, Any], frame: pd.DataFrame, floor_area_m2: float
) -> dict[str, tuple[float, float]]:
    """Turn the config's parameter block into concrete numeric bounds.

    A bound written as `auto` is DERIVED FROM THE DATA rather than typed
    in. Only `internal_gain_w_per_m2` supports it today, and the
    derivation is `calibration/bounds.py`'s: a fixed multiple of the
    cooling load that survives the coldest hours of the year.

    The point of `auto` is that no building gets a bound chosen to suit
    it. Three successive hand-set values for this one parameter (15,
    200, 60, 120 W/m2) each looked physical when written and each turned
    out to be anchored on the wrong measurement; the rule replaces the
    judgement, and applies the same way to every building including the
    ones it does not flatter.

    Args:
        config: Parsed calibration config.
        frame: The training data, carrying `load_kwh` and
            `airTemperature`.
        floor_area_m2: Conditioned floor area.

    Returns:
        `{name: (lower, upper)}` in the config's parameter order.

    Raises:
        ValueError: If a parameter other than `internal_gain_w_per_m2`
            asks for `auto`, or if a bound is not a number.
    """
    resolved: dict[str, tuple[float, float]] = {}
    for name, spec in config["parameters"].items():
        upper = spec["upper"]
        if isinstance(upper, str) and upper.strip().lower() == AUTO_BOUND:
            if name != "internal_gain_w_per_m2":
                raise ValueError(
                    f"{name}: `auto` is only defined for internal_gain_w_per_m2. "
                    "Deriving a bound needs a measurement that isolates that "
                    "parameter, and there is none for this one."
                )
            upper_value = internal_gain_upper_bound(
                frame["load_kwh"], frame["airTemperature"], floor_area_m2
            )
            logger.info(
                "derived %s upper bound: %.1f W/m2 (auto, from the cold-weather floor)",
                name,
                upper_value,
            )
        else:
            upper_value = float(upper)
        resolved[name] = (float(spec["lower"]), upper_value)
    return resolved


def report_against_baselines(
    frame: pd.DataFrame,
    observed_kw: npt.NDArray[np.float64],
    predicted_kw: npt.NDArray[np.float64],
    n_params: int,
) -> pd.DataFrame:
    """Score the calibrated model beside L6.4's two naive baselines.

    A CV(RMSE) means nothing on its own -- 22% is excellent against a
    48% baseline and embarrassing against a 20% one. Both baselines are
    refitted here on exactly the rows the calibration used, so the
    comparison is like for like.
    """
    outdoor_c = frame["airTemperature"].to_numpy(dtype=float)
    mean_fit = fit_annual_mean(observed_kw)
    regression_fit = fit_linear_regression(outdoor_c, observed_kw)

    rows = []
    for name, n_p, prediction in (
        (mean_fit.name, mean_fit.n_params, mean_fit.predict(outdoor_c)),
        (regression_fit.name, regression_fit.n_params, regression_fit.predict(outdoor_c)),
        ("calibrated RC (L6.7)", n_params, predicted_kw),
    ):
        verdict = ashrae_g14_pass(observed_kw, prediction, n_params=n_p)
        rows.append(
            {
                "model": name,
                "p": n_p,
                "NMBE %": round(verdict.nmbe_pct, 2),
                "CV(RMSE) %": round(verdict.cvrmse_pct, 2),
                "G14": "PASS" if verdict.passed else "FAIL",
            }
        )
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    """Entry point: load config, calibrate, report, write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to calibrate on. Defaults to the config's train_year; "
        "any other value is refused (ADR-002 test-set lock).",
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    train_year = int(config["train_year"])
    year = train_year if arguments.year is None else int(arguments.year)
    if year != train_year:
        raise SystemExit(
            f"refusing to calibrate on {year}: the configured training year is "
            f"{train_year} and every other year is held out (ADR-002). The test "
            "set is opened once, deliberately, at L6.10 -- and that access is "
            "logged in 07_PROGRESS.md when it happens."
        )

    frame, floor_area_m2 = load_training_data(config["building_id"], config["site_id"], year)
    objective = CalibrationObjective(
        (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
        frame["airTemperature"].to_numpy(dtype=float),
        frame["load_kwh"].to_numpy(dtype=float),
        floor_area_m2,
        config,
        outdoor_humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
    )
    observed_kw = objective.observed_kw

    bounds = resolve_bounds(config, frame, floor_area_m2)
    optimiser = config["optimiser"]

    result = calibrate(
        objective,
        bounds,
        breakdown_fn=objective.breakdown,
        popsize=int(optimiser["popsize"]),
        maxiter=int(optimiser["maxiter"]),
        tol=float(optimiser["tol"]),
        workers=int(optimiser["workers"]),
        metadata={
            "building_id": config["building_id"],
            "site_id": config["site_id"],
            "year": year,
            "n_hours": int(len(frame)),
            "mean_load_kw": float(observed_kw.mean()),
            "floor_area_m2": floor_area_m2,
            "objective": "g14_budget_with_clipping_penalty",
            "config_path": str(arguments.config),
            "derived_bounds": {
                name: list(pair)
                for name, pair in bounds.items()
                if isinstance(config["parameters"][name]["upper"], str)
            },
        },
    )

    predicted_kw, _ = objective.predict(np.asarray(result.best_parameters, dtype=float))
    comparison = report_against_baselines(
        frame, observed_kw, predicted_kw, n_params=len(bounds)
    )

    logger.info("--- calibrated parameters ---\n%s", result.summary())
    logger.info("--- against L6.4's baselines ---\n%s", comparison.to_string())
    if result.pinned_parameters:
        logger.warning(
            "pinned: %s -- read this as a statement about the bounds or the "
            "model structure, not as a calibrated value (07_PROGRESS.md Q7)",
            ", ".join(result.pinned_parameters),
        )

    path = write_artifact(result, config["artifacts"]["directory"])
    logger.info("artifact: %s", path)


if __name__ == "__main__":
    main()
