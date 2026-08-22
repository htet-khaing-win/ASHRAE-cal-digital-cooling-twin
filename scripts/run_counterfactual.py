"""Setpoint what-if scenarios, with intervals and a trade-off sweep (L8.2, L8.3).

    python scripts/run_counterfactual.py

For each building whose model holds ASHRAE G14 on the held-out year:

  1. Run five interventions against one shared baseline.
  2. Sweep chilled-water supply temperature to locate the point where
     the chiller's saving is cancelled by the pump's penalty -- once
     assuming the coils are rebalanced, once assuming the return
     temperature is fixed. The two curves are the trade-off.
  3. Attach uncertainty to every number, from three separate sources
     that measure three different things (see `INTERVAL_SOURCES`).

ADR-002: training year only. A counterfactual has no ground truth on
any year -- the world in which the setpoint was 1 K higher was not
recorded -- so opening 2017 for it would spend the project's scarcest
asset on a question no data can settle.

NOTHING HERE IS VALIDATED, and the word is used deliberately. The
cooling-load model is validated (M6, on 2017). The conversion from
cooling load to electricity is not, and cannot be from this dataset:
BDG2 records no chiller sub-meter. See `config/plant.yaml`.
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
import pandas as pd  # noqa: E402
from run_calibration import load_config  # noqa: E402
from twin_setup import (  # noqa: E402
    BUILDINGS_PATH,
    DEFAULT_GROUPS,
    IncompatibleArtifactError,
    TwinBundle,
    load_twin,
    selected_buildings,
)

from cooling_twin.models.plant import PlantParams, plant_electric_kw  # noqa: E402
from cooling_twin.twin.counterfactual import (  # noqa: E402
    DEFAULT_SCENARIOS,
    CounterfactualResult,
    Scenario,
    compare_scenarios,
    simulate_setpoint_change,
)
from cooling_twin.twin.uncertainty import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BLOCK_HOURS,
    block_bootstrap_ci,
    conformal_interval,
    conformal_quantile,
    normalising_scale,
    time_ordered_split,
)

logger = logging.getLogger("run_counterfactual")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
DEFAULT_SCENARIO_FIGURE = Path("reports/figures/l8_2_scenarios.png")
DEFAULT_TRADEOFF_FIGURE = Path("reports/figures/l8_2_chiller_pump_tradeoff.png")

# Chilled-water supply increases swept for the trade-off, K. Stops at
# +3.3 K because 03_DOMAIN_REFERENCE.md SS1 gives 6.7-10.0 degC as the
# optimisation range for this setpoint, and extrapolating a plant model
# past its own documented operating range is how a counterfactual
# becomes fiction.
CHW_SWEEP_DELTA_K = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.3)

# The pump's share of HVAC energy, per 03_DOMAIN_REFERENCE.md SS1's
# 8-12% band. The midpoint is what config/plant.yaml uses; the two edges
# are run to show how much of the trade-off's answer is carried by that
# one assumption.
PUMP_SHARE_BAND_PCT = (8.0, 10.0, 12.0)

# Calibration share for the split-conformal quantile. The remaining 30%
# -- the last ~2,600 hours of the training year -- is what coverage is
# measured on and what the what-if bands are reported over.
CALIBRATION_FRACTION = 0.7

INTERVAL_SOURCES = {
    "conformal_band_kw": (
        "The twin's own hourly error against the meter, as a "
        "distribution-free interval. Brackets ONE hour's load. Assumes "
        "the error distribution is unchanged by the intervention -- an "
        "assumption, not a result."
    ),
    "block_bootstrap_pct": (
        "Sampling variability of the ANNUAL MEAN saving, resampled in "
        "week-long blocks so the residual's autocorrelation survives. "
        "Says nothing about the model being wrong."
    ),
    "parameter_ensemble_pct": (
        "The same intervention re-run on every behavioural parameter "
        "set from L6.8's equifinality study -- parameter sets that fit "
        "the meter equally well and describe different buildings. This "
        "is the term that dominates, and the only one that measures "
        "anything structural."
    ),
}

CHILLER_COLOUR = "#2171b5"
PUMP_COLOUR = "#e08214"
TOTAL_COLOUR = "#252525"
ZERO_COLOUR = "#525252"
BAND_COLOUR = "#bdbdbd"


def scenario_intervals(
    bundle: TwinBundle,
    result: CounterfactualResult,
    alpha: float,
    block_hours: int,
) -> dict[str, Any]:
    """Attach the three kinds of interval to one scenario.

    Args:
        bundle: The building's twin, plant and measurements.
        result: The scenario to bracket.
        alpha: Miss rate for both the conformal interval and the
            bootstrap.
        block_hours: Bootstrap block length.

    Returns:
        A JSON-shaped record of the intervals.
    """
    calibration, scored = time_ordered_split(
        bundle.measured_kw.size, CALIBRATION_FRACTION, embargo_hours=DEFAULT_BLOCK_HOURS
    )
    physics_residual = bundle.measured_kw - result.baseline_load_kw

    # Normalised scores: the model's error grows with the load, so a
    # constant band is simultaneously too wide on quiet hours and too
    # narrow on the hours a decision is actually about.
    scale_calibration = normalising_scale(result.baseline_load_kw[calibration])
    quantile = conformal_quantile(physics_residual[calibration], alpha, scale=scale_calibration)

    scenario_scored = result.scenario_load_kw[scored]
    interval = conformal_interval(
        scenario_scored,
        quantile,
        alpha=alpha,
        n_calibration=int(physics_residual[calibration].size),
        scale=normalising_scale(scenario_scored),
    )

    hourly_delta = result.hourly_total_delta_kw
    baseline_mean_kw = float(result.baseline_plant.total_kw.mean())
    lower_kw, upper_kw = block_bootstrap_ci(
        hourly_delta, alpha=alpha, block_hours=block_hours
    )
    return {
        "conformal": {
            "alpha": alpha,
            "quantile_relative": quantile,
            "n_calibration": interval.n_calibration,
            "n_scored": int(scenario_scored.size),
            "median_half_width_kw": float(np.median(interval.width) / 2.0),
            "median_half_width_pct_of_mean_load": (
                50.0 * float(np.median(interval.width)) / float(scenario_scored.mean())
                if scenario_scored.mean()
                else float("nan")
            ),
        },
        "block_bootstrap_pct": {
            "point": result.total_change_pct,
            "lower": 100.0 * lower_kw / baseline_mean_kw,
            "upper": 100.0 * upper_kw / baseline_mean_kw,
            "block_hours": block_hours,
        },
    }


def parameter_ensemble(
    bundle: TwinBundle,
    scenarios: tuple[Scenario, ...],
    artifacts: Path,
) -> dict[str, Any] | None:
    """Re-run every scenario on L6.8's equifinal parameter sets.

    THE MOST IMPORTANT UNCERTAINTY IN M8, and the one a confidence
    interval on the fit does not contain. L6.8 found parameter sets that
    score within 5% of the calibrated objective and describe physically
    different buildings -- a tight envelope with large gains, a leaky
    one with small gains. They agree about the past by construction.
    They are under no obligation to agree about an intervention, and
    where they disagree, the disagreement IS the honest error bar.

    Args:
        bundle: The building.
        scenarios: The interventions to re-run.
        artifacts: Directory holding `equifinality_*.json`.

    Returns:
        `{scenario_name: {...}}` plus the parameter sets used, or None
        when the building has no equifinality study.
    """
    candidates: list[dict[str, float]] = []
    source = None
    for path in sorted(artifacts.glob("equifinality_*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("building_id") != bundle.building_id:
            continue
        source = path.name
        for candidate in record.get("candidates", []):
            if candidate.get("behavioural"):
                candidates.append(candidate["parameters"])
        break

    if not candidates:
        logger.info(
            "%s: no equifinality study found -- the structural interval is "
            "UNAVAILABLE for this building, not zero.",
            bundle.building_id,
        )
        return None

    logger.info(
        "%s: re-running %d scenario(s) on %d behavioural parameter set(s) from %s",
        bundle.building_id,
        len(scenarios),
        len(candidates),
        source,
    )
    per_scenario: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}
    for parameters in candidates:
        twin = replace(bundle.twin, parameters=parameters)
        baseline = twin.predict_load_kw()
        for scenario in scenarios:
            result = simulate_setpoint_change(
                twin, scenario, bundle.plant, baseline_load_kw=baseline
            )
            per_scenario[scenario.name].append(result.total_change_pct)

    return {
        "source": source,
        "n_parameter_sets": len(candidates),
        "parameters": candidates,
        "total_change_pct": {
            name: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "values": [float(value) for value in values],
            }
            for name, values in per_scenario.items()
        },
    }


def _with_pump_share(plant: PlantParams, share_pct: float, reference_pct: float) -> PlantParams:
    """A copy of the plant with the pump re-sized to a different energy share.

    Pump power at rated speed scales linearly with the assumed share, so
    the plant is rebuilt by scaling `power_rated_kw` rather than by
    re-reading the config -- which keeps every other assumption
    provably identical between the three runs.
    """
    scaled = replace(
        plant.chw_pump,
        power_rated_kw=plant.chw_pump.power_rated_kw * share_pct / reference_pct,
    )
    return replace(plant, chw_pump=scaled)


def chw_tradeoff_sweep(
    bundle: TwinBundle,
    baseline_load_kw: npt.NDArray[np.float64],
    reference_share_pct: float,
) -> dict[str, Any]:
    """Sweep chilled-water supply temperature; find where the pump wins.

    Two curves, differing in ONE assumption -- whether the coils can
    still return water at the design temperature once the supply is
    warmer. Under `coils_hold` the flow never changes and the chiller's
    gain is kept in full. Under `return_fixed` the delta-T narrows, and
    since flow sits in the denominator of `Q = m cp dT` and the affinity
    law cubes it, the pump's penalty grows far faster than the chiller's
    gain.

    Args:
        bundle: The building.
        baseline_load_kw: The shared baseline (the sweep does not touch
            the zone, so the load is identical throughout).
        reference_share_pct: The pump share `config/plant.yaml` was
            built with, used to rescale for the sensitivity band.

    Returns:
        A JSON-shaped record of the sweep.
    """
    rows: list[dict[str, Any]] = []
    for share_pct in PUMP_SHARE_BAND_PCT:
        plant = _with_pump_share(bundle.plant, share_pct, reference_share_pct)
        baseline_plant = plant_electric_kw(baseline_load_kw, bundle.twin.wet_bulb_c, plant)
        baseline_total = float(baseline_plant.total_kw.sum())
        for return_fixed in (False, True):
            for delta_k in CHW_SWEEP_DELTA_K:
                if return_fixed and delta_k >= plant.design_delta_t_k:
                    continue
                delta_t_k = (
                    plant.design_delta_t_k - delta_k if return_fixed else plant.design_delta_t_k
                )
                operation = plant_electric_kw(
                    baseline_load_kw,
                    bundle.twin.wet_bulb_c,
                    plant,
                    t_chw_supply_c=plant.t_chw_supply_c + delta_k,
                    chw_delta_t_k=delta_t_k,
                )
                rows.append(
                    {
                        "pump_share_pct": share_pct,
                        "mode": "return_fixed" if return_fixed else "coils_hold",
                        "chw_supply_delta_k": delta_k,
                        "chw_delta_t_k": delta_t_k,
                        "chiller_change_pct": 100.0
                        * (
                            float(operation.chiller_kw.sum())
                            / float(baseline_plant.chiller_kw.sum())
                            - 1.0
                        ),
                        "pump_change_pct": 100.0
                        * (
                            float(operation.pump_kw.sum()) / float(baseline_plant.pump_kw.sum())
                            - 1.0
                        ),
                        "total_change_pct": 100.0
                        * (float(operation.total_kw.sum()) / baseline_total - 1.0),
                        "n_hours_flow_capped": operation.n_hours_flow_capped,
                        "n_hours_cop_capped": operation.n_hours_cop_capped,
                    }
                )

    frame = pd.DataFrame(rows)
    break_even = {}
    for (share_pct, mode), group in frame.groupby(["pump_share_pct", "mode"]):
        # The zero-delta point is the baseline compared with itself and
        # is exactly 0.0 by construction. Leaving it in makes every
        # sweep "cross zero" at 0 K, which is a sign-detection artifact
        # and not a break-even point.
        group = group[group["chw_supply_delta_k"] > 0.0].sort_values("chw_supply_delta_k")
        totals = group["total_change_pct"].to_numpy()
        deltas = group["chw_supply_delta_k"].to_numpy()
        crossing = np.where(np.diff(np.sign(totals)) != 0)[0]
        key = f"{mode}@{share_pct:.0f}pct"
        entry: dict[str, Any] = {
            "sign_at_first_step": "saves" if totals[0] < 0 else "costs",
            "first_step_k": float(deltas[0]),
            "total_pct_at_first_step": float(totals[0]),
            "total_pct_at_last_step": float(totals[-1]),
            "last_step_k": float(deltas[-1]),
        }
        if crossing.size == 0:
            # No crossing means one component wins across the whole
            # documented operating range -- which is a stronger
            # statement than a break-even point, not a missing one.
            entry["crossing_k"] = None
        else:
            index = int(crossing[0])
            span = totals[index + 1] - totals[index]
            fraction = 0.0 if span == 0 else -totals[index] / span
            entry["crossing_k"] = float(
                deltas[index] + fraction * (deltas[index + 1] - deltas[index])
            )
        break_even[key] = entry
    return {"rows": rows, "break_even_chw_delta_k": break_even}


def _ensemble_range(record: dict[str, Any], scenario_name: str) -> str:
    """The equifinal spread for one scenario, or an explicit `unavailable`.

    "unavailable" rather than a blank or a zero: a building with no
    equifinality study has an UNMEASURED structural uncertainty, which
    is not the same as a small one.
    """
    ensemble = record.get("parameter_ensemble")
    if not ensemble:
        return "unavailable"
    spread = ensemble["total_change_pct"][scenario_name]
    return f"[{spread['min']:+.2f}, {spread['max']:+.2f}]"


def plot_scenarios(records: list[dict[str, Any]], path: Path) -> Path:
    """One row per building: every scenario, split into chiller and pump."""
    figure, axes = plt.subplots(
        len(records), 1, figsize=(11.0, 3.8 * len(records)), squeeze=False
    )
    for row, record in enumerate(records):
        axis = axes[row][0]
        scenarios = record["scenarios"]
        names = [entry["name"] for entry in scenarios]
        positions = np.arange(len(names))
        width = 0.26
        axis.bar(
            positions - width,
            [entry["chiller_change_pct"] for entry in scenarios],
            width=width,
            color=CHILLER_COLOUR,
            label="chiller",
        )
        axis.bar(
            positions,
            [entry["pump_change_pct"] for entry in scenarios],
            width=width,
            color=PUMP_COLOUR,
            label="CHW pump",
        )
        totals = [entry["total_change_pct"] for entry in scenarios]
        axis.bar(positions + width, totals, width=width, color=TOTAL_COLOUR, label="plant total")

        # Error bars from the parameter ensemble where it exists -- the
        # only interval here that measures anything structural.
        ensemble = record.get("parameter_ensemble")
        if ensemble:
            spread = ensemble["total_change_pct"]
            lows = [totals[i] - spread[name]["min"] for i, name in enumerate(names)]
            highs = [spread[name]["max"] - totals[i] for i, name in enumerate(names)]
            axis.errorbar(
                positions + width,
                totals,
                yerr=[np.abs(lows), np.abs(highs)],
                fmt="none",
                ecolor=ZERO_COLOUR,
                capsize=4,
                linewidth=1.2,
            )
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_xticks(positions)
        axis.set_xticklabels([name.replace("_", "\n") for name in names], fontsize=7)
        axis.set_ylabel("annual change, %", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8, ncol=3)
        axis.set_title(
            f"{record['building_id']} -- bars are the twin's point estimate; "
            "whiskers are equifinal parameter sets that fit the meter equally well",
            fontsize=9,
        )
    figure.suptitle(
        "L8.2 counterfactual scenarios. Cooling-load model calibrated (M6); "
        "load-to-electricity conversion UNCALIBRATED (config/plant.yaml)."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def plot_tradeoff(records: list[dict[str, Any]], path: Path) -> Path:
    """The trade-off itself: chiller gain against pump penalty, swept."""
    figure, axes = plt.subplots(
        1, len(records), figsize=(6.2 * len(records), 4.6), squeeze=False
    )
    for column, record in enumerate(records):
        axis = axes[0][column]
        frame = pd.DataFrame(record["chw_tradeoff"]["rows"])
        midpoint = frame[frame["pump_share_pct"] == 10.0]
        for mode, style in (("coils_hold", "-"), ("return_fixed", "--")):
            group = midpoint[midpoint["mode"] == mode].sort_values("chw_supply_delta_k")
            axis.plot(
                group["chw_supply_delta_k"],
                group["chiller_change_pct"],
                style,
                color=CHILLER_COLOUR,
                label=f"chiller, {mode}",
                linewidth=1.6,
            )
            axis.plot(
                group["chw_supply_delta_k"],
                group["pump_change_pct"],
                style,
                color=PUMP_COLOUR,
                label=f"pump, {mode}",
                linewidth=1.6,
            )
            axis.plot(
                group["chw_supply_delta_k"],
                group["total_change_pct"],
                style,
                color=TOTAL_COLOUR,
                label=f"total, {mode}",
                linewidth=2.2,
            )
        # Sensitivity band on the total: the same sweep at the 8% and 12%
        # edges of the documented pump-share range.
        for mode in ("coils_hold", "return_fixed"):
            low = frame[(frame["pump_share_pct"] == 8.0) & (frame["mode"] == mode)].sort_values(
                "chw_supply_delta_k"
            )
            high = frame[
                (frame["pump_share_pct"] == 12.0) & (frame["mode"] == mode)
            ].sort_values("chw_supply_delta_k")
            axis.fill_between(
                low["chw_supply_delta_k"],
                low["total_change_pct"],
                high["total_change_pct"],
                color=BAND_COLOUR,
                alpha=0.45,
                linewidth=0,
            )
        axis.axhline(0.0, color=ZERO_COLOUR, linewidth=1)
        axis.set_xlabel("chilled-water supply setpoint increase, K", fontsize=9)
        axis.set_ylabel("annual plant electricity change, %", fontsize=9)
        axis.legend(fontsize=7, ncol=2)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        entry = record["chw_tradeoff"]["break_even_chw_delta_k"]["return_fixed@10pct"]
        if entry["crossing_k"] is not None:
            subtitle = f"return fixed: chiller and pump cancel at +{entry['crossing_k']:.2f} K"
        else:
            subtitle = (
                f"return fixed: the change {entry['sign_at_first_step']} energy at every "
                f"step out to +{entry['last_step_k']:.1f} K "
                f"({entry['total_pct_at_last_step']:+.1f}%)"
            )
        axis.set_title(f"{record['building_id']}\n{subtitle}", fontsize=10)
    figure.suptitle(
        "L8.2 chiller-versus-pump trade-off. Solid = coils rebalanced (delta-T held); "
        "dashed = return temperature fixed. Grey band = pump 8-12% of HVAC energy."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def analyse(
    bundle: TwinBundle,
    artifacts: Path,
    alpha: float,
    block_hours: int,
    reference_share_pct: float,
) -> dict[str, Any]:
    """Run every scenario, the sweep and the intervals for one building."""
    results = compare_scenarios(bundle.twin, bundle.plant, DEFAULT_SCENARIOS)
    baseline = results[0].baseline_load_kw

    scenarios = []
    for result in results:
        record = result.to_dict()
        record["intervals"] = scenario_intervals(bundle, result, alpha, block_hours)
        scenarios.append(record)

    ensemble = parameter_ensemble(bundle, DEFAULT_SCENARIOS, artifacts)
    tradeoff = chw_tradeoff_sweep(bundle, baseline, reference_share_pct)

    return {
        "building_id": bundle.building_id,
        "role": bundle.role,
        "year": bundle.twin.year,
        "calibration_artifact": bundle.artifact_name,
        "n_hours": int(bundle.measured_kw.size),
        "mean_measured_kw": float(bundle.measured_kw.mean()),
        "mean_baseline_load_kw": float(baseline.mean()),
        "plant": {
            "n_chillers": bundle.plant.n_chillers,
            "q_ref_kw_per_chiller": bundle.plant.curves.q_ref_kw,
            "capacity_kw": bundle.plant.capacity_kw,
            "t_chw_supply_c": bundle.plant.t_chw_supply_c,
            "design_delta_t_k": bundle.plant.design_delta_t_k,
            "tower_approach_k": bundle.plant.tower_approach_k,
            "pump_rated_kw": bundle.plant.chw_pump.power_rated_kw,
            "pump_rated_flow_m3_per_s": bundle.plant.chw_pump.flow_rated_m3_per_s,
            "variable_speed": bundle.plant.variable_speed,
        },
        "baseline_mwh": results[0].baseline_plant.annual_mwh,
        "baseline_plant_cop": float(baseline.sum() / results[0].baseline_plant.total_kw.sum()),
        "scenarios": scenarios,
        "parameter_ensemble": ensemble,
        "chw_tradeoff": tradeoff,
    }


def main() -> None:
    """Entry point: scenarios, sweep, intervals, figures, artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--scenario-figure", type=Path, default=DEFAULT_SCENARIO_FIGURE)
    parser.add_argument("--tradeoff-figure", type=Path, default=DEFAULT_TRADEOFF_FIGURE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS)
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    from cooling_twin.models.plant import load_plant_config

    reference_share_pct = float(load_plant_config()["pump"]["share_of_hvac_pct"])

    records = []
    for building in selected_buildings(arguments.buildings, DEFAULT_GROUPS):
        try:
            bundle = load_twin(building, config, artifacts)
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        records.append(
            analyse(
                bundle,
                artifacts,
                alpha=arguments.alpha,
                block_hours=arguments.block_hours,
                reference_share_pct=reference_share_pct,
            )
        )

    if not records:
        raise SystemExit("no building had a usable calibration artifact")

    summary = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "scenario": entry["name"],
                "load %": round(entry["load_change_pct"], 2),
                "chiller %": round(entry["chiller_change_pct"], 2),
                "pump %": round(entry["pump_change_pct"], 2),
                "total %": round(entry["total_change_pct"], 2),
                "bootstrap 90% CI": (
                    f"[{entry['intervals']['block_bootstrap_pct']['lower']:+.2f}, "
                    f"{entry['intervals']['block_bootstrap_pct']['upper']:+.2f}]"
                ),
                "equifinal range": _ensemble_range(record, entry["name"]),
            }
            for record in records
            for entry in record["scenarios"]
        ]
    )
    logger.info(
        "--- counterfactual scenarios, %d ---\n%s", train_year, summary.to_string(index=False)
    )

    plot_scenarios(records, arguments.scenario_figure)
    plot_tradeoff(records, arguments.tradeoff_figure)

    out_path = artifacts / f"counterfactual_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "note": (
                    "Training year only (ADR-002). Physics parameters read "
                    "frozen. The cooling-load model is calibrated and was "
                    "tested on 2017 (M6); the load-to-electricity conversion "
                    "is UNCALIBRATED and cannot be validated from BDG2, which "
                    "records no chiller sub-meter. No result here is a "
                    "measured saving."
                ),
                "interval_sources": INTERVAL_SOURCES,
                "alpha": arguments.alpha,
                "buildings": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
