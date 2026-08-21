"""Score a 3P change-point baseline against the calibrated RC model.

    python scripts/fit_change_point_baseline.py

Authorised by ADR-015. Q10 established that Cathleen's residual is the
inverse model's clip at zero meeting a weather-insensitive base load the
RC form cannot hold. That is an argument. This turns it into a
measurement:

    load = base + slope * max(0, T - T_cp)          [3 parameters]

is the shape her data actually has -- flat through winter, rising in
summer -- and it is the standard ASHRAE/IMT inverse model for exactly
that. If a THREE-parameter change-point model beats the FIVE-parameter
physics model on this building, the finding is no longer "the RC form
struggles here"; it is "a simpler model of the right shape does better,
and here is the shape".

WHAT THIS IS NOT. A change to the twin. `fit_change_point` lives in
`calibration/baseline.py` alongside the annual mean and the linear
regression, is scored through the same `ashrae_g14_pass()`, and changes
no model, parameter or bound. ADR-015 authorised it on the TRAINING YEAR
ONLY and this script has no argument that could make it read another
year.

The comparison is run on all three buildings, not just Cathleen. A
baseline that beat the RC model everywhere would be a statement about
the RC model in general rather than about this building, and the only
way to know which is to look.
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
    selected_buildings,
)
from investigate_ushape import year_residual  # noqa: E402
from run_calibration import load_config  # noqa: E402

from cooling_twin.calibration.baseline import (  # noqa: E402
    fit_annual_mean,
    fit_change_point,
    fit_linear_regression,
    relative_cvrmse_improvement_pct,
)
from cooling_twin.calibration.metrics import ashrae_g14_pass  # noqa: E402

logger = logging.getLogger("change_point_baseline")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_2_change_point_baseline.png")

MEASURED_COLOUR = "#969696"
RC_COLOUR = "#2171b5"
CHANGE_POINT_COLOUR = "#b2182b"

# Bin width for the plotted load-vs-temperature curves, K. Same as the
# residual profiles use, so the two figures can be read side by side.
PLOT_BIN_WIDTH_K = 2.5


def binned(
    temperature: np.ndarray, values: np.ndarray, width: float = PLOT_BIN_WIDTH_K
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` per temperature bin, for plotting."""
    codes = np.floor(temperature / width).astype(int)
    unique = np.unique(codes)
    centres, means = [], []
    for code in unique:
        mask = codes == code
        if mask.sum() < 20:
            continue
        centres.append(float(temperature[mask].mean()))
        means.append(float(values[mask].mean()))
    return np.array(centres), np.array(means)


def compare(
    building: dict[str, str], config: dict[str, Any], artifacts: Path
) -> dict[str, Any]:
    """Score every baseline and the calibrated model on one building."""
    building_id = building["building_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(
        artifacts, building_id, train_year, names
    )

    series = year_residual(building, config, parameters, train_year)
    measured = series["measured_kw"]
    temperature = series["outdoor_dry_bulb"]
    rc_predicted = series["predicted_kw"]

    mean_fit = fit_annual_mean(measured)
    regression_fit = fit_linear_regression(temperature, measured)
    change_point_fit = fit_change_point(temperature, measured)

    scored = {
        fit.name: (fit, ashrae_g14_pass(measured, fit.predict(temperature), n_params=fit.n_params))
        for fit in (mean_fit, regression_fit, change_point_fit)
    }
    rc_verdict = ashrae_g14_pass(measured, rc_predicted, n_params=len(names))

    return {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "year": train_year,
        "hours": int(measured.size),
        "mean_load_kw": float(measured.mean()),
        "calibrated_rc": {
            "n_params": len(names),
            "cvrmse_pct": rc_verdict.cvrmse_pct,
            "nmbe_pct": rc_verdict.nmbe_pct,
            "passed": rc_verdict.passed,
        },
        "baselines": {
            name: {
                "n_params": fit.n_params,
                "cvrmse_pct": verdict.cvrmse_pct,
                "nmbe_pct": verdict.nmbe_pct,
                "passed": verdict.passed,
                "change_point_degC": fit.change_point,
                "coefficients": fit.coefficients.tolist(),
            }
            for name, (fit, verdict) in scored.items()
        },
        "change_point_vs_rc_pct": relative_cvrmse_improvement_pct(
            rc_verdict.cvrmse_pct, scored[change_point_fit.name][1].cvrmse_pct
        ),
        "_series": series,
        "_change_point_fit": change_point_fit,
    }


def plot_comparison(records: list[dict[str, Any]], path: Path) -> Path:
    """Measured, RC and change-point load against outdoor temperature."""
    figure, axes = plt.subplots(
        1, len(records), figsize=(4.8 * len(records), 4.2), squeeze=False
    )
    for axis, record in zip(axes[0], records, strict=True):
        series = record["_series"]
        temperature = series["outdoor_dry_bulb"]
        fit = record["_change_point_fit"]

        for values, colour, label in (
            (series["measured_kw"], MEASURED_COLOUR, "measured"),
            (series["predicted_kw"], RC_COLOUR, "calibrated RC (p=5)"),
            (fit.predict(temperature), CHANGE_POINT_COLOUR, "change-point (p=3)"),
        ):
            centres, means = binned(temperature, values)
            axis.plot(centres, means, marker="o", markersize=3, color=colour, label=label)

        axis.axvline(
            fit.change_point, color=CHANGE_POINT_COLOUR, linestyle=":", linewidth=1
        )
        axis.axhline(0.0, color="#525252", linewidth=1)
        rc = record["calibrated_rc"]["cvrmse_pct"]
        cp = record["baselines"][fit.name]["cvrmse_pct"]
        axis.set_title(
            f"{record['building_id']}\nRC {rc:.2f}%  vs  change-point {cp:.2f}%",
            fontsize=9,
            color=CHANGE_POINT_COLOUR if cp < rc else "black",
        )
        axis.set_xlabel(
            f"outdoor dry bulb, degC (dotted: T_cp {fit.change_point:.1f})", fontsize=8
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0][0].set_ylabel("mean cooling load, kW", fontsize=9)
    figure.suptitle(
        "ADR-015: a 3-parameter change-point BASELINE against the "
        "5-parameter calibrated RC model. Training year only."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: fit and score the change-point baseline."""
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

    records = []
    for building in selected_buildings(arguments.buildings):
        try:
            records.append(compare(building, config, artifacts))
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)

    if not records:
        raise SystemExit("no building had a calibration artifact to compare against")

    rows = []
    for record in records:
        rows.append(
            {
                "building": record["building_id"],
                "model": "calibrated RC (physics)",
                "p": record["calibrated_rc"]["n_params"],
                "CV(RMSE) %": round(record["calibrated_rc"]["cvrmse_pct"], 2),
                "NMBE %": round(record["calibrated_rc"]["nmbe_pct"], 2),
                "G14": "PASS" if record["calibrated_rc"]["passed"] else "FAIL",
            }
        )
        for name, payload in record["baselines"].items():
            rows.append(
                {
                    "building": record["building_id"],
                    "model": name,
                    "p": payload["n_params"],
                    "CV(RMSE) %": round(payload["cvrmse_pct"], 2),
                    "NMBE %": round(payload["nmbe_pct"], 2),
                    "G14": "PASS" if payload["passed"] else "FAIL",
                }
            )
    table = pd.DataFrame(rows).set_index(["building", "model"])
    logger.info("--- every model on the training year ---\n%s", table.to_string())

    verdicts = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "T_cp degC": round(
                    float(record["_change_point_fit"].change_point or np.nan), 1
                ),
                "base kW": round(float(record["_change_point_fit"].coefficients[0]), 0),
                "slope kW/K": round(
                    float(record["_change_point_fit"].coefficients[1]), 1
                ),
                "RC CV %": round(record["calibrated_rc"]["cvrmse_pct"], 2),
                "3P CV %": round(
                    record["baselines"][record["_change_point_fit"].name]["cvrmse_pct"],
                    2,
                ),
                "3P beats RC by %": round(record["change_point_vs_rc_pct"], 1),
            }
            for record in records
        ]
    ).set_index("building")
    logger.info(
        "--- the ADR-015 comparison: 3 parameters against 5 ---\n%s",
        verdicts.to_string(),
    )
    logger.info(
        "--- how to read this ---\n"
        "  A POSITIVE '3P beats RC by %%' means a three-parameter curve of the\n"
        "  right shape outperforms the five-parameter physics model. On a\n"
        "  building where that happens, the gap is not calibration quality --\n"
        "  it is the model's FORM, which is what ADR-015 records and declines\n"
        "  to fix. A negative value means the physics is earning its extra\n"
        "  parameters, which is the expected and desirable case."
    )

    plot_comparison(records, arguments.figure)

    out_path = artifacts / f"change_point_baseline_{int(config['train_year'])}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": int(config["train_year"]),
                "authorised_by": "ADR-015",
                "note": (
                    "A BASELINE in L6.4's family, scored through the same "
                    "ashrae_g14_pass(). Changes no model, parameter or bound. "
                    "Training year only."
                ),
                "buildings": [
                    {k: v for k, v in record.items() if not k.startswith("_")}
                    for record in records
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
