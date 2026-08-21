"""Does the model structure transfer from a dry site to a humid one? (L7.5)

    python scripts/compare_site_humidity.py

The project's two buildings sit in climates that differ by a factor of
nine in humid hours -- Fox 5.8% against Bull 51.3% of the year above
`select.HUMID_HUMIDITY_RATIO` (12 g/kg), the same threshold M2 used to
choose them. That pairing was deliberate (01_LEARNING_PATH.md S9), and
this script is what it was for.

THE STRUCTURE UNDER TEST is ADR-011's ventilation term:

    latent_kw = vent_flow * h_fg * max(w_outdoor - w_supply, 0)

`w_supply` is a STATED ASSUMPTION fixed at 9.2 g/kg and never fitted --
saturated air off a ~12.8 degC coil. It is not a scale factor, it is a
THRESHOLD, and a threshold interacts with the climate it is applied in:

    dry site   median outdoor w BELOW the threshold  -> term mostly OFF
    humid site median outdoor w ABOVE the threshold  -> term mostly ON

So the identical model structure becomes a different model at the two
sites, without anyone changing a line. Three measurements, per building:

  A. REGIME SPLIT. The year is cut into humid and dry HOURS at the same
     12 g/kg, and both model structures -- physics alone, and L7.3's
     physics + ML hybrid -- are scored inside each. A site-level
     comparison has n = 1 per group and can only ever be a case study;
     an hour-level split has thousands of hours per cell and supports a
     within-building statement. Both are reported, labelled.
  B. TRIGGER SWEEP. `w_supply` is swept with every CALIBRATED parameter
     FROZEN, and the humidity signal left in the residual is measured at
     each value with the L7.2 matched split. This is a SENSITIVITY
     STUDY, not a re-calibration: no value found here is adopted, and
     adopting one would change ADR-011's stated assumption and fall
     under ADR-015.
  C. LATENT ACTIVITY. How much of each year the term is switched on at
     all, and what share of the predicted load it carries.

ADR-002: training year only. The optimiser is not imported.

`Hog_education_Cathleen` IS included here, unlike in the M7 hybrid work.
The reason is the mirror image of ADR-015's: a humidity diagnostic on
the PHYSICS residual is a statement about the structure, and a structure
claim tested on two buildings when a third is available is weaker for
no reason. Its CV(RMSE) column is NOT interpretable -- that building's
error is dominated by the clip-at-zero defect of ADR-015 -- so the
hybrid arm skips it and the report says so at every mention.
"""

from __future__ import annotations

import argparse
import copy
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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import (  # noqa: E402
    IncompatibleArtifactError,
    frozen_parameters,
    selected_buildings,
)
from check_humidity_trigger import latent_activity  # noqa: E402
from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
)

from cooling_twin.analysis.hybrid import (  # noqa: E402
    DEFAULT_EMBARGO_HOURS,
    DEFAULT_N_FOLDS,
    fit_hybrid,
)
from cooling_twin.analysis.residual import (  # noqa: E402
    TEMPERATURE_BIN_WIDTH_K,
    effective_sample_size,
    matched_band_split,
)
from cooling_twin.calibration.metrics import cvrmse, nmbe  # noqa: E402
from cooling_twin.data.select import HUMID_HUMIDITY_RATIO  # noqa: E402

logger = logging.getLogger("compare_site_humidity")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_5_humid_vs_dry.png")

# Buildings whose HYBRID numbers may be quoted (ADR-015 excludes the
# negative case from any learnt correction). Every other group is
# analysed on the physics residual -- `analyse_residuals.selected_buildings`
# returns all three roles, which is what this script wants and is why it
# imports that one rather than `fit_hybrid_residual`'s filtered version.
HYBRID_GROUPS = ("primary", "generalisation")

# Supply humidity ratios to sweep, kg/kg. The range brackets ADR-011's
# 9.2 g/kg on both sides:
#   6.0  a colder coil, or an outdoor-air fraction below 100%
#   9.2  THE STATED ASSUMPTION (12.8 degC saturated)
#  12.0  the project's own humid-hour threshold
#  16.0  a coil doing almost no dehumidification
# Chosen as physical statements, not as a grid around an optimum: the
# point is the SHAPE of the response, and a grid centred on the answer
# would beg the question.
TRIGGER_SWEEP_G_PER_KG: tuple[float, ...] = (4.0, 6.0, 8.0, 9.2, 12.0, 16.0)

# Minimum hours in a regime before its metrics are reported. Below this
# the CV(RMSE) of a subset is a statement about a handful of days.
MIN_REGIME_HOURS = 200

_G_PER_KG = 1000.0
_PERCENT = 100.0

DRY_COLOUR = "#2171b5"
HUMID_COLOUR = "#b2182b"
ZERO_COLOUR = "#525252"
BUILDING_COLOURS = ("#2171b5", "#b2182b", "#e08214")


def regime_metrics(
    measured_kw: npt.NDArray[np.float64],
    physics_kw: npt.NDArray[np.float64],
    hybrid_kw: npt.NDArray[np.float64] | None,
    mask: npt.NDArray[np.bool_],
    *,
    n_params: int,
    regime: str,
    label: str,
) -> dict[str, Any] | None:
    """Score both model structures inside one humidity regime.

    Both structures are scored on the SAME hours through the same
    metrics, which is the only way the comparison means anything -- see
    L6.4, where the baselines and the model were deliberately put
    through one `predict()`.

    The ML share is computed against the REGIME's own variance, so the
    two regimes' shares are comparable in sign and not in magnitude.
    A humid-hour share of 8% and a dry-hour share of 2% do not mean the
    correction did four times as much work in humid hours; they mean it
    explained more of a smaller (or larger) pot. The absolute kilowatts
    are printed alongside for that reason.

    Args:
        measured_kw: Metered load, kW.
        physics_kw: Calibrated physics prediction, kW.
        hybrid_kw: Physics + out-of-fold correction, or None when the
            hybrid arm is not run for this building.
        mask: Hours belonging to the regime.
        n_params: Calibrated parameter count, for `n - p`.
        regime: `"dry"` or `"humid"` -- the key the figure and the report
            look up. Kept separate from `label` so that changing the
            wording of a log line cannot silently empty a panel.
        label: The human-readable version, for logs.

    Returns:
        The regime's metrics, or None if it holds too few hours.
    """
    hours = int(mask.sum())
    if hours < MIN_REGIME_HOURS:
        logger.info("%s: %d hours, below %d -- not scored", label, hours, MIN_REGIME_HOURS)
        return None

    measured, physics = measured_kw[mask], physics_kw[mask]
    record: dict[str, Any] = {
        "regime": regime,
        "label": label,
        "hours": hours,
        "mean_measured_kw": float(measured.mean()),
        "mean_residual_kw": float((measured - physics).mean()),
        "physics_cvrmse_pct": cvrmse(measured, physics, n_params=n_params),
        "physics_nmbe_pct": nmbe(measured, physics, n_params=n_params),
        "hybrid_cvrmse_pct": None,
        "ml_pct": None,
    }
    if hybrid_kw is not None:
        hybrid = hybrid_kw[mask]
        ss_total = float(((measured - measured.mean()) ** 2).sum())
        ss_physics = float(((measured - physics) ** 2).sum())
        ss_hybrid = float(((measured - hybrid) ** 2).sum())
        record["hybrid_cvrmse_pct"] = cvrmse(measured, hybrid, n_params=n_params)
        record["ml_pct"] = _PERCENT * (ss_physics - ss_hybrid) / ss_total
    return record


def humidity_effect(
    residual_kw: npt.NDArray[np.float64],
    t_outdoor_c: npt.NDArray[np.float64],
    humidity_g_per_kg: npt.NDArray[np.float64],
    *,
    effective_sample_ratio: float,
) -> dict[str, Any]:
    """How much humidity signal is left in the residual, at matched temperature.

    Whole year, not the humid hours alone. The matched split IS the
    humid-versus-dry comparison -- it bins on outdoor temperature,
    removes the within-bin temperature trend, and splits each bin at its
    own median humidity -- so restricting it to humid hours first would
    throw away the dry half it needs to compare against.
    """
    split = matched_band_split(
        residual_kw,
        t_outdoor_c,
        humidity_g_per_kg,
        control="outdoor_dry_bulb",
        probe="humidity_ratio",
        control_width=TEMPERATURE_BIN_WIDTH_K,
        effective_sample_ratio=effective_sample_ratio,
    )
    return {
        "weighted_difference_kw": split.weighted_difference_kw,
        "weighted_difference_sem_kw": split.weighted_difference_sem_kw,
        "humid_hours_run_higher": split.probe_raises_residual,
        "bins": int(split.centres.size),
    }


def analyse(
    building: dict[str, str],
    config: dict[str, Any],
    artifacts: Path,
    *,
    n_folds: int,
    embargo_hours: int,
) -> dict[str, Any]:
    """Regime split, trigger sweep and latent activity for one building.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Calibration config.
        artifacts: Artifact directory.
        n_folds: Expanding-window folds for the hybrid arm.
        embargo_hours: Hours dropped between training and scoring.

    Returns:
        A JSON-shaped record.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(artifacts, building_id, train_year, names)
    vector = np.array([parameters[name] for name in names], dtype=float)

    # Loaded ONCE. The sweep below rebuilds the objective per trigger
    # value but never re-reads the meter: a sweep that reloaded would
    # spend most of its time in pandas and would invite someone to
    # shorten the sweep instead.
    frame, floor_area_m2 = load_training_data(building_id, site_id, train_year)
    seconds = (frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float)
    t_outdoor = frame["airTemperature"].to_numpy(dtype=float)
    humidity = frame["humidity_ratio"].to_numpy(dtype=float)
    load = frame["load_kwh"].to_numpy(dtype=float)

    def objective_at(supply_humidity_ratio: float) -> CalibrationObjective:
        """The same objective with one stated assumption changed."""
        variant = copy.deepcopy(config)
        variant.setdefault("ventilation", {})["supply_humidity_ratio"] = (
            supply_humidity_ratio
        )
        return CalibrationObjective(
            seconds,
            t_outdoor,
            load,
            floor_area_m2,
            variant,
            outdoor_humidity_ratio=humidity,
        )

    baseline_supply = float(config["ventilation"]["supply_humidity_ratio"])
    objective = objective_at(baseline_supply)
    predicted, _raw = objective.predict(vector)
    measured = objective.observed_kw
    residual = measured - predicted

    # One autocorrelation correction per building, measured on the
    # baseline residual and reused across the sweep. The alternative --
    # recomputing it at every trigger value -- would let the error bars
    # move for a reason unrelated to the humidity signal being measured,
    # and the sweep is a comparison of the CENTRES.
    ratio = effective_sample_size(residual) / residual.size
    humid_hours = humidity > HUMID_HUMIDITY_RATIO

    hybrid_kw: npt.NDArray[np.float64] | None = None
    scored: npt.NDArray[np.bool_] = np.ones(measured.size, dtype=bool)
    hybrid_record: dict[str, Any] | None = None
    if building["role"] in HYBRID_GROUPS:
        hybrid = fit_hybrid(
            frame.index,
            measured,
            predicted,
            t_outdoor_c=t_outdoor,
            humidity_ratio_kg_per_kg=humidity,
            n_physics_params=len(names),
            label=f"{building_id} {train_year}",
            n_folds=n_folds,
            embargo_hours=embargo_hours,
        )
        hybrid_kw, scored = hybrid.hybrid_kw, hybrid.scored_mask
        hybrid_record = {
            "physics_pct": hybrid.decomposition.physics_pct,
            "ml_pct": hybrid.decomposition.ml_pct,
            "unexplained_pct": hybrid.decomposition.unexplained_pct,
        }
    else:
        logger.warning(
            "%s: hybrid arm SKIPPED (ADR-015). Its physics residual is the "
            "clip-at-zero defect, so a learnt correction there measures how "
            "quickly a booster relearns a base load.",
            building_id,
        )

    # Regimes are scored on the hours the hybrid covers, so that the two
    # structures are compared on ONE set of hours. For the negative case
    # `scored` is all-true and the physics columns cover the whole year.
    threshold_g_per_kg = HUMID_HUMIDITY_RATIO * _G_PER_KG
    regimes = [
        record
        for regime, mask, label in (
            ("dry", scored & ~humid_hours, f"dry hours (w <= {threshold_g_per_kg:.0f} g/kg)"),
            ("humid", scored & humid_hours, f"humid hours (w > {threshold_g_per_kg:.0f} g/kg)"),
        )
        if (
            record := regime_metrics(
                measured,
                predicted,
                hybrid_kw,
                mask,
                n_params=len(names),
                regime=regime,
                label=f"{building_id} {label}",
            )
        )
        is not None
    ]

    sweep = []
    for trigger_g_per_kg in TRIGGER_SWEEP_G_PER_KG:
        trigger = trigger_g_per_kg / _G_PER_KG
        swept_objective = objective_at(trigger)
        swept_predicted, _ = swept_objective.predict(vector)
        swept_residual = measured - swept_predicted
        row = {
            "supply_humidity_ratio_g_per_kg": trigger_g_per_kg,
            "is_stated_assumption": bool(np.isclose(trigger, baseline_supply)),
            "cvrmse_pct": cvrmse(measured, swept_predicted, n_params=len(names)),
            "nmbe_pct": nmbe(measured, swept_predicted, n_params=len(names)),
            **latent_activity(
                humidity, swept_predicted, float(parameters["vent_flow_kg_per_s"]), trigger
            ),
            **humidity_effect(
                swept_residual,
                t_outdoor,
                humidity * _G_PER_KG,
                effective_sample_ratio=ratio,
            ),
        }
        sweep.append(row)
        logger.info(
            "%s w_supply %.1f g/kg: latent active %.1f%% of hours, CV(RMSE) %.2f%%, "
            "residual humidity effect %+.0f +/- %.0f kW",
            building_id,
            trigger_g_per_kg,
            row["hours_latent_active_pct"],
            row["cvrmse_pct"],
            row["weighted_difference_kw"],
            row["weighted_difference_sem_kw"],
        )

    record = {
        "building_id": building_id,
        "site_id": site_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "year": train_year,
        "hours": int(measured.size),
        "hours_scored": int(scored.sum()),
        "humid_hours_pct": float(humid_hours.mean() * _PERCENT),
        "humid_threshold_g_per_kg": HUMID_HUMIDITY_RATIO * _G_PER_KG,
        "outdoor_humidity_p50_g_per_kg": float(np.median(humidity) * _G_PER_KG),
        "vent_flow_kg_per_s": float(parameters["vent_flow_kg_per_s"]),
        "effective_sample_ratio": ratio,
        "whole_year": {
            "physics_cvrmse_pct": cvrmse(measured, predicted, n_params=len(names)),
            "physics_nmbe_pct": nmbe(measured, predicted, n_params=len(names)),
        },
        "hybrid": hybrid_record,
        "regimes": regimes,
        "trigger_sweep": sweep,
    }
    logger.info(
        "%s (%s site, %.1f%% humid hours): %s",
        building_id,
        "humid" if record["humid_hours_pct"] >= 40.0 else "dry",
        record["humid_hours_pct"],
        pd.DataFrame(regimes)[
            ["label", "hours", "physics_nmbe_pct", "physics_cvrmse_pct", "ml_pct"]
        ].round(2).to_string(index=False),
    )
    return record


def plot_site_humidity(records: list[dict[str, Any]], path: Path) -> Path:
    """Three panels: the regimes, the trigger's reach, the signal it leaves.

    Panel 3 is the one that carries the finding. It plots the humidity
    signal remaining in the residual against the trigger value, with the
    stated assumption marked -- so "the threshold is arbitrary" stops
    being an argument and becomes a curve that either crosses zero
    somewhere useful or does not.
    """
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    stated = next(
        row["supply_humidity_ratio_g_per_kg"]
        for row in records[0]["trigger_sweep"]
        if row["is_stated_assumption"]
    )

    # --- panel 1: NMBE by regime ---
    axis = axes[0]
    labels, dry_values, humid_values = [], [], []
    for record in records:
        by_regime = {row["regime"]: row for row in record["regimes"]}
        if "dry" not in by_regime or "humid" not in by_regime:
            continue
        labels.append(record["building_id"].split("_")[-1])
        dry_values.append(by_regime["dry"]["physics_nmbe_pct"])
        humid_values.append(by_regime["humid"]["physics_nmbe_pct"])
    positions = np.arange(len(labels), dtype=float)
    axis.bar(positions - 0.2, dry_values, width=0.38, color=DRY_COLOUR, label="dry hours")
    axis.bar(positions + 0.2, humid_values, width=0.38, color=HUMID_COLOUR, label="humid hours")
    axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("physics NMBE, %", fontsize=8)
    axis.set_title(
        f"bias by humidity regime (split at {records[0]['humid_threshold_g_per_kg']:.0f} g/kg)",
        fontsize=9,
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3, axis="y")
    axis.tick_params(labelsize=8)

    # --- panel 2: how much of the year the term is on ---
    axis = axes[1]
    for colour, record in zip(BUILDING_COLOURS, records, strict=False):
        triggers = [row["supply_humidity_ratio_g_per_kg"] for row in record["trigger_sweep"]]
        active = [row["hours_latent_active_pct"] for row in record["trigger_sweep"]]
        axis.plot(
            triggers,
            active,
            marker="o",
            markersize=4,
            color=colour,
            label=(
                f"{record['building_id'].split('_')[-1]} "
                f"({record['humid_hours_pct']:.0f}% humid hours)"
            ),
        )
    axis.axvline(stated, color=ZERO_COLOUR, linestyle="--", linewidth=1)
    axis.text(stated + 0.15, 5.0, "ADR-011\nassumption", fontsize=7, color=ZERO_COLOUR)
    axis.set_xlabel("supply humidity ratio (the trigger), g/kg", fontsize=8)
    axis.set_ylabel("hours the latent term is active, %", fontsize=8)
    axis.set_title("one threshold, three climates", fontsize=9)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    axis.tick_params(labelsize=8)

    # --- panel 3: humidity signal left in the residual ---
    axis = axes[2]
    for colour, record in zip(BUILDING_COLOURS, records, strict=False):
        triggers = [row["supply_humidity_ratio_g_per_kg"] for row in record["trigger_sweep"]]
        effect = np.array(
            [row["weighted_difference_kw"] for row in record["trigger_sweep"]], dtype=float
        )
        sem = np.array(
            [row["weighted_difference_sem_kw"] for row in record["trigger_sweep"]], dtype=float
        )
        axis.errorbar(
            triggers,
            effect,
            yerr=2.0 * sem,
            marker="o",
            markersize=4,
            capsize=3,
            linewidth=1.5,
            color=colour,
            label=record["building_id"].split("_")[-1],
        )
    axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
    axis.axvline(stated, color=ZERO_COLOUR, linestyle="--", linewidth=1)
    axis.set_xlabel("supply humidity ratio (the trigger), g/kg", fontsize=8)
    axis.set_ylabel("humid-hour residual excess at matched\ntemperature, kW", fontsize=8)
    axis.set_title("what the term failed to remove (+/- 2 SEM)", fontsize=9)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    axis.tick_params(labelsize=8)

    figure.suptitle(
        "One model structure, two climates. Training year, calibrated parameters "
        "FROZEN -- the sweep changes a stated assumption, not a fitted value."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: split by regime, sweep the trigger, log, plot, record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--embargo-hours", type=int, default=DEFAULT_EMBARGO_HOURS)
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    records = []
    for building in selected_buildings(arguments.buildings):
        try:
            records.append(
                analyse(
                    building,
                    config,
                    artifacts,
                    n_folds=arguments.n_folds,
                    embargo_hours=arguments.embargo_hours,
                )
            )
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue

    if not records:
        raise SystemExit("no building had a calibration artifact to analyse")

    summary = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "site": record["site_id"],
                "humid hours %": round(record["humid_hours_pct"], 1),
                "outdoor w p50": round(record["outdoor_humidity_p50_g_per_kg"], 1),
                "latent active %": round(
                    next(
                        row["hours_latent_active_pct"]
                        for row in record["trigger_sweep"]
                        if row["is_stated_assumption"]
                    ),
                    1,
                ),
                "latent share %": round(
                    next(
                        row["latent_share_of_predicted_pct"]
                        for row in record["trigger_sweep"]
                        if row["is_stated_assumption"]
                    ),
                    1,
                ),
                "humidity left kW": round(
                    next(
                        row["weighted_difference_kw"]
                        for row in record["trigger_sweep"]
                        if row["is_stated_assumption"]
                    ),
                    0,
                ),
                "CV(RMSE) %": round(record["whole_year"]["physics_cvrmse_pct"], 2),
            }
            for record in records
        ]
    ).set_index("building")
    logger.info(
        "--- site humidity, %d, at the stated assumption ---\n%s",
        train_year,
        summary.to_string(),
    )

    plot_site_humidity(records, arguments.figure)

    out_path = artifacts / f"site_humidity_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "humid_threshold_g_per_kg": HUMID_HUMIDITY_RATIO * _G_PER_KG,
                "note": (
                    "Training year only (ADR-002). Calibrated parameters read "
                    "frozen; the optimiser is not imported. The trigger sweep "
                    "varies ADR-011's STATED ASSUMPTION only and is a sensitivity "
                    "study -- no value here is adopted, and adopting one would "
                    "fall under ADR-015. The negative case's CV(RMSE) is not "
                    "interpretable (clip-at-zero defect) and it has no hybrid arm."
                ),
                "buildings": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
