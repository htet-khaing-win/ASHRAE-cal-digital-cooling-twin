"""Correlation vs intervention -- what a twin answers that regression cannot (L8.1).

    python scripts/compare_correlation_intervention.py

Two demonstrations, in this order:

1. **A known-truth case.** Synthetic data where the causal structure is
   built by hand, so the right answer is available to check against.
   Outdoor temperature drives BOTH the humidity and the cooling load;
   a regression of load on humidity therefore reports a slope that is
   mostly temperature wearing humidity's name. The confounded estimate,
   the adjusted estimate and the truth are printed side by side.

2. **The real building.** The same question asked of
   Fox_education_Claude's 2016 record, where no ground truth exists --
   and then asked of the calibrated twin as an INTERVENTION, which is
   the only way it can be answered at all.

The second demonstration ends somewhere the first cannot: the zone
setpoint. It is not recorded in BDG2 and it does not vary in the record,
so the observational question "what was the load when the setpoint was
25 degC" has no data behind it whatsoever -- no amount of regression
recovers a coefficient for a variable that never moves. The twin still
answers `do(setpoint = 25)`, because the setpoint appears in its
equations rather than in its data. That gap is the argument for M8.

ADR-002: training year only.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from run_calibration import load_config  # noqa: E402
from twin_setup import (  # noqa: E402
    BUILDINGS_PATH,
    DEFAULT_GROUPS,
    IncompatibleArtifactError,
    load_twin,
    selected_buildings,
)

from cooling_twin import SEED  # noqa: E402

logger = logging.getLogger("compare_correlation_intervention")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l8_1_correlation_vs_intervention.png")

# The intervention size for the humidity demonstration: +1 g/kg of
# outdoor humidity ratio, holding temperature fixed. Stated in g/kg
# because that is the unit the M7 humid/dry finding is reported in.
HUMIDITY_STEP_G_PER_KG = 1.0
_G_PER_KG = 1000.0

# Setpoint intervention for the "regression cannot answer this at all"
# demonstration.
SETPOINT_STEP_K = 1.0

# Synthetic demonstration: a deliberately simple linear world, so that
# the confounded and adjusted estimates can be compared against numbers
# that are true by construction rather than by argument.
SYNTHETIC_HOURS = 8760
TRUE_TEMPERATURE_COEFF_KW_PER_K = 300.0
TRUE_HUMIDITY_COEFF_KW_PER_G_PER_KG = 120.0
HUMIDITY_ON_TEMPERATURE_G_PER_KG_PER_K = 0.45
SYNTHETIC_BASE_LOAD_KW = 2000.0
SYNTHETIC_NOISE_KW = 250.0

CONFOUNDED_COLOUR = "#b2182b"
ADJUSTED_COLOUR = "#4393c3"
INTERVENTION_COLOUR = "#1a9850"
TRUTH_COLOUR = "#252525"


def ols_slope(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Univariate least-squares slope of `y` on `x`, with an intercept.

    Hand-rolled rather than pulled from a library for one reason: this
    script's argument is about what a regression COEFFICIENT means, and
    the reader should be able to see that nothing clever is happening
    inside it.

    Args:
        x: Regressor.
        y: Response.

    Returns:
        The slope.

    Raises:
        ValueError: If the arrays are mismatched, shorter than 2, or if
            `x` has no variance (the coefficient does not exist).
    """
    if x.shape != y.shape:
        raise ValueError(f"x {x.shape} and y {y.shape} must have the same shape")
    if x.size < 2:
        raise ValueError("need at least 2 points to fit a slope")
    if np.ptp(x) == 0.0:
        raise ValueError(
            "x is constant, so no slope exists. This is not an edge case to "
            "handle -- it is the whole point of L8.1: a variable that never "
            "moves in the record has no observational coefficient at all."
        )
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[1])


def multiple_slopes(
    regressors: dict[str, npt.NDArray[np.float64]], y: npt.NDArray[np.float64]
) -> dict[str, float]:
    """Least-squares slopes with every regressor held mutually constant.

    "Adjusting for the confounder" in its simplest honest form. It
    recovers the truth in the synthetic demonstration because there the
    confounder is known, measured and linear -- three conditions that
    are never all true on a real building, which is exactly why the
    real-data section does not stop here.

    Args:
        regressors: Named regressors, all the same length as `y`.
        y: Response.

    Returns:
        `{name: slope}`.

    Raises:
        ValueError: If a regressor is mismatched or the design is
            singular.
    """
    names = list(regressors)
    columns = [np.ones_like(y)]
    for name in names:
        column = regressors[name]
        if column.shape != y.shape:
            raise ValueError(f"regressor {name!r} {column.shape} does not match y {y.shape}")
        columns.append(column)
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return dict(zip(names, (float(value) for value in coefficients[1:]), strict=True))


def synthetic_confounding_demo(seed: int = SEED) -> dict[str, Any]:
    """A world where the causal truth is known, so the bias is measurable.

    Structure, built in this order so that it is a DAG and not a
    circular definition::

        temperature  ->  humidity
        temperature  ->  load
        humidity     ->  load

    Temperature is the confounder of the humidity-load relationship. The
    naive slope of load on humidity therefore picks up the true humidity
    effect PLUS the temperature effect routed through humidity, and the
    size of that contamination is known exactly:

        bias = TRUE_TEMPERATURE_COEFF / HUMIDITY_ON_TEMPERATURE

    Args:
        seed: Passed to `np.random.default_rng` (L0.3).

    Returns:
        A JSON-shaped record of the three estimates and the truth.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(SYNTHETIC_HOURS)
    temperature = 20.0 + 12.0 * np.sin(2.0 * np.pi * (hours - 2400) / SYNTHETIC_HOURS) + rng.normal(
        0.0, 2.0, SYNTHETIC_HOURS
    )
    # Humidity is CAUSED by temperature plus its own independent noise.
    humidity = (
        4.0
        + HUMIDITY_ON_TEMPERATURE_G_PER_KG_PER_K * temperature
        + rng.normal(0.0, 1.5, SYNTHETIC_HOURS)
    )
    load = (
        SYNTHETIC_BASE_LOAD_KW
        + TRUE_TEMPERATURE_COEFF_KW_PER_K * temperature
        + TRUE_HUMIDITY_COEFF_KW_PER_G_PER_KG * humidity
        + rng.normal(0.0, SYNTHETIC_NOISE_KW, SYNTHETIC_HOURS)
    )

    confounded = ols_slope(humidity, load)
    adjusted = multiple_slopes({"temperature": temperature, "humidity": humidity}, load)
    # For load = a*T + b*w + noise with w = c*T + u, the univariate slope
    # of load on w is  b + a*c*var(T)/var(w). The second term is the
    # contamination, and it is a closed form -- so the demonstration is a
    # known-answer check, not an illustration.
    expected_bias = (
        TRUE_TEMPERATURE_COEFF_KW_PER_K
        * HUMIDITY_ON_TEMPERATURE_G_PER_KG_PER_K
        * float(np.var(temperature))
        / float(np.var(humidity))
    )

    record = {
        "true_humidity_coeff_kw_per_g_per_kg": TRUE_HUMIDITY_COEFF_KW_PER_G_PER_KG,
        "true_temperature_coeff_kw_per_k": TRUE_TEMPERATURE_COEFF_KW_PER_K,
        "confounded_slope_kw_per_g_per_kg": confounded,
        "adjusted_slope_kw_per_g_per_kg": adjusted["humidity"],
        "adjusted_temperature_slope_kw_per_k": adjusted["temperature"],
        "confounded_overstatement_x": confounded / TRUE_HUMIDITY_COEFF_KW_PER_G_PER_KG,
        "predicted_bias_kw_per_g_per_kg": expected_bias,
        "corr_temperature_humidity": float(np.corrcoef(temperature, humidity)[0, 1]),
        "n_hours": SYNTHETIC_HOURS,
        "seed": seed,
    }
    logger.info(
        "synthetic: truth %.1f kW per g/kg; naive regression says %.1f (%.1fx too "
        "large); adjusting for the confounder recovers %.1f",
        TRUE_HUMIDITY_COEFF_KW_PER_G_PER_KG,
        confounded,
        record["confounded_overstatement_x"],
        adjusted["humidity"],
    )
    return record


def real_building_demo(bundle: Any) -> dict[str, Any]:
    """The same question on real data, then asked as an intervention.

    Args:
        bundle: A `TwinBundle` from `twin_setup.load_twin`.

    Returns:
        A JSON-shaped record.
    """
    temperature = bundle.twin.t_ambient_c
    humidity_g_per_kg = bundle.twin.humidity_ratio * _G_PER_KG
    measured = bundle.measured_kw

    observational = ols_slope(humidity_g_per_kg, measured)
    adjusted = multiple_slopes(
        {"temperature": temperature, "humidity": humidity_g_per_kg}, measured
    )

    # THE INTERVENTION. Everything is held exactly as it was -- the same
    # weather, the same hours, the same parameters -- and one driver is
    # moved. This is not a regression on the twin's output; the twin is
    # re-solved with a modified input.
    baseline_kw = bundle.twin.predict_load_kw()
    # `replace` on a frozen dataclass: every other driver is carried
    # across by identity, so nothing but the humidity can differ between
    # the two arms. Rebuilding the twin field by field would leave one
    # more place for a typo to become a "finding".
    intervened = replace(
        bundle.twin,
        humidity_ratio=bundle.twin.humidity_ratio + HUMIDITY_STEP_G_PER_KG / _G_PER_KG,
    )
    humidity_intervention_kw = float(
        (intervened.predict_load_kw() - baseline_kw).mean() / HUMIDITY_STEP_G_PER_KG
    )

    setpoint_up = bundle.twin.predict_load_kw(setpoint_delta_c=SETPOINT_STEP_K)
    setpoint_intervention_kw = float((setpoint_up - baseline_kw).mean() / SETPOINT_STEP_K)

    record = {
        "building_id": bundle.building_id,
        "role": bundle.role,
        "year": bundle.twin.year,
        "n_hours": int(measured.size),
        "calibration_artifact": bundle.artifact_name,
        "observational_slope_kw_per_g_per_kg": observational,
        "adjusted_slope_kw_per_g_per_kg": adjusted["humidity"],
        "adjusted_temperature_slope_kw_per_k": adjusted["temperature"],
        "interventional_kw_per_g_per_kg": humidity_intervention_kw,
        "interventional_kw_per_k_setpoint": setpoint_intervention_kw,
        "corr_temperature_humidity": float(np.corrcoef(temperature, humidity_g_per_kg)[0, 1]),
        "setpoint_variance_in_data": 0.0,
        "mean_measured_kw": float(measured.mean()),
    }
    logger.info(
        "%s: observational %.1f kW per g/kg -> adjusted %.1f -> INTERVENTIONAL %.1f "
        "(outdoor T/w correlation %.2f)",
        bundle.building_id,
        observational,
        adjusted["humidity"],
        humidity_intervention_kw,
        record["corr_temperature_humidity"],
    )
    logger.info(
        "%s: do(setpoint +%.0f K) = %.1f kW per K. There is NO observational "
        "counterpart to this number -- the setpoint has zero variance in the "
        "record, so its regression coefficient does not exist.",
        bundle.building_id,
        SETPOINT_STEP_K,
        setpoint_intervention_kw,
    )
    return record


def plot_comparison(
    synthetic: dict[str, Any], buildings: list[dict[str, Any]], path: Path
) -> Path:
    """Left: the known-truth case. Right: the same three estimates per building."""
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    axis = axes[0]
    labels = ["regression\non humidity", "regression\nadjusted for T", "truth\n(by construction)"]
    values = [
        synthetic["confounded_slope_kw_per_g_per_kg"],
        synthetic["adjusted_slope_kw_per_g_per_kg"],
        synthetic["true_humidity_coeff_kw_per_g_per_kg"],
    ]
    axis.bar(labels, values, color=[CONFOUNDED_COLOUR, ADJUSTED_COLOUR, TRUTH_COLOUR])
    axis.axhline(
        synthetic["true_humidity_coeff_kw_per_g_per_kg"],
        color=TRUTH_COLOUR,
        linestyle="--",
        linewidth=1,
    )
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    axis.set_ylabel("kW per g/kg of humidity ratio", fontsize=9)
    axis.set_title(
        "Synthetic world, causal truth known:\n"
        f"the naive slope is {synthetic['confounded_overstatement_x']:.1f}x the true effect",
        fontsize=10,
    )
    axis.tick_params(labelsize=8)
    axis.grid(alpha=0.3, axis="y")

    axis = axes[1]
    width = 0.26
    positions = np.arange(len(buildings))
    for offset, key, colour, label in (
        (-width, "observational_slope_kw_per_g_per_kg", CONFOUNDED_COLOUR, "observational"),
        (0.0, "adjusted_slope_kw_per_g_per_kg", ADJUSTED_COLOUR, "adjusted for outdoor T"),
        (width, "interventional_kw_per_g_per_kg", INTERVENTION_COLOUR, "twin, do(w + 1 g/kg)"),
    ):
        axis.bar(
            positions + offset,
            [record[key] for record in buildings],
            width=width,
            color=colour,
            label=label,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([record["building_id"] for record in buildings], fontsize=7)
    axis.axhline(0.0, color=TRUTH_COLOUR, linewidth=1)
    axis.set_ylabel("kW per g/kg of humidity ratio", fontsize=9)
    axis.set_title(
        "Real buildings, no ground truth available:\n"
        "the three answers disagree, and only the third is an intervention",
        fontsize=10,
    )
    axis.legend(fontsize=8)
    axis.tick_params(labelsize=8)
    axis.grid(alpha=0.3, axis="y")

    figure.suptitle(
        "L8.1 correlation vs intervention -- training year (2016), physics parameters frozen"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: synthetic demo, real demo, figure, artifact."""
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

    synthetic = synthetic_confounding_demo()

    records = []
    for building in selected_buildings(arguments.buildings, DEFAULT_GROUPS):
        try:
            bundle = load_twin(building, config, artifacts)
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        records.append(real_building_demo(bundle))

    if not records:
        raise SystemExit("no building had a usable calibration artifact")

    plot_comparison(synthetic, records, arguments.figure)

    out_path = artifacts / f"correlation_vs_intervention_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "note": (
                    "Training year only (ADR-002). Physics parameters read "
                    "frozen. The interventional column is the twin re-solved "
                    "with one driver moved; it is not a regression on the "
                    "twin's output."
                ),
                "humidity_step_g_per_kg": HUMIDITY_STEP_G_PER_KG,
                "setpoint_step_k": SETPOINT_STEP_K,
                "synthetic": synthetic,
                "buildings": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
