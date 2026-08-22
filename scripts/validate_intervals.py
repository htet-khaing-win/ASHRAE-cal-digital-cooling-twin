"""Conformal prediction intervals, and whether they actually cover (L8.3).

    python scripts/validate_intervals.py

06_ASSESSMENT.md's M8 gate asks for two things: conformal prediction
implemented, and empirical coverage >= 90% at alpha = 0.1. The second is
the one that can fail, so this script is built around measuring it three
ways rather than once:

  MARGINAL      one number over the whole scoring block. This is what
                the gate asks for and what the guarantee is about.
  CONDITIONAL   coverage within each month and within each load
                quintile. The guarantee says nothing about these, and
                they are what an operator experiences.
  DIRECTION     the same split run backwards (calibrate on the late
                block, score on the early one), because a single split
                on a seasonal series can flatter or punish by accident.

A Gaussian interval is scored alongside as a control. It is not a straw
man: `mean +- 1.645 sigma` is what most reporting does, it is one line
of code, and the point is to show what its assumption costs on a
residual that M7 measured as neither normal nor independent.

ADR-002: training year only.

DISCLOSED BIAS, and it runs one way. The physics parameters were fitted
on the WHOLE of 2016, so every hour scored here was seen by the
optimiser. Split conformal's guarantee needs the calibration and test
scores to be exchangeable, which they are with respect to the SPLIT --
but both blocks are in-sample with respect to the point model. Coverage
reported here is therefore an optimistic estimate of what the same
intervals would deliver on an unseen year, and the honest way to close
the gap would be a 2017 re-read that ADR-002 does not permit for a
question this one cannot answer.
"""

from __future__ import annotations

import argparse
import json
import logging
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

from cooling_twin.twin.uncertainty import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BLOCK_HOURS,
    ConformalInterval,
    conformal_interval,
    conformal_quantile,
    coverage_by_group,
    interleaved_block_split,
    mondrian_interval,
    mondrian_quantiles,
    normalising_scale,
    time_ordered_split,
    validate_coverage,
)

logger = logging.getLogger("validate_intervals")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l8_3_conformal_coverage.png")

CALIBRATION_FRACTION = 0.7
N_LOAD_BINS = 5

# Mondrian grouping: 5 K bands of outdoor dry bulb. Wide enough that
# each band holds hundreds of calibration hours, narrow enough to
# separate the regimes M7 found -- Claude's residual is flat below
# ~16 degC and climbs at +54 kW/K above it, which one pooled quantile
# has to average over.
MONDRIAN_BIN_WIDTH_K = 5.0
MONDRIAN_MIN_GROUP = 100

# 1.645 is the one-sided 95% normal quantile, so +-1.645 sigma is the
# textbook 90% interval. Named rather than inlined so the assumption is
# visible in the code that depends on it.
GAUSSIAN_Z_90 = 1.6448536269514722

CONSTANT_COLOUR = "#4393c3"
NORMALISED_COLOUR = "#1a9850"
MONDRIAN_COLOUR = "#762a83"
GAUSSIAN_COLOUR = "#b2182b"
TARGET_COLOUR = "#252525"


def temperature_bins(t_outdoor_c: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    """Fixed-width outdoor-temperature bins, as Mondrian groups.

    Fixed width rather than quantile-based, and defined over the WHOLE
    year rather than per block, so that a bin means the same band of
    weather in the calibration block and the scoring block. Quantile
    bins would shift with whichever block they were computed on and the
    two would stop being comparable.
    """
    edges = np.arange(
        MONDRIAN_BIN_WIDTH_K * np.floor(t_outdoor_c.min() / MONDRIAN_BIN_WIDTH_K),
        t_outdoor_c.max() + MONDRIAN_BIN_WIDTH_K,
        MONDRIAN_BIN_WIDTH_K,
    )
    return np.digitize(t_outdoor_c, edges[1:-1])


def _interval_set(
    measured: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
    calibration: npt.NDArray[np.bool_],
    scored: npt.NDArray[np.bool_],
    alpha: float,
    groups: npt.NDArray[np.int_],
) -> dict[str, ConformalInterval]:
    """Build the four intervals compared throughout this script."""
    residual = measured - predicted
    calibration_residual = residual[calibration]
    scored_prediction = predicted[scored]
    scale_calibration = normalising_scale(predicted[calibration])
    scale_scored = normalising_scale(scored_prediction)

    constant_q = conformal_quantile(calibration_residual, alpha)
    normalised_q = conformal_quantile(calibration_residual, alpha, scale=scale_calibration)
    group_quantiles, pooled = mondrian_quantiles(
        calibration_residual,
        groups[calibration],
        alpha,
        min_group=MONDRIAN_MIN_GROUP,
        scale=scale_calibration,
    )
    # The Gaussian control, expressed as an interval object so it goes
    # through exactly the same coverage code as the others.
    gaussian_half_width = GAUSSIAN_Z_90 * float(np.std(calibration_residual, ddof=1))

    return {
        "conformal_constant": conformal_interval(
            scored_prediction,
            constant_q,
            alpha=alpha,
            n_calibration=int(calibration_residual.size),
        ),
        "conformal_normalised": conformal_interval(
            scored_prediction,
            normalised_q,
            alpha=alpha,
            n_calibration=int(calibration_residual.size),
            scale=scale_scored,
        ),
        "conformal_mondrian_temperature": mondrian_interval(
            scored_prediction,
            groups[scored],
            group_quantiles,
            pooled,
            alpha=alpha,
            n_calibration=int(calibration_residual.size),
            scale=scale_scored,
        ),
        "gaussian_control": conformal_interval(
            scored_prediction,
            gaussian_half_width,
            alpha=alpha,
            n_calibration=int(calibration_residual.size),
        ),
    }


def _mask_from_slice(n: int, window: slice) -> npt.NDArray[np.bool_]:
    """Boolean mask for a slice, so all three split regimes share one shape."""
    mask = np.zeros(n, dtype=bool)
    mask[window] = True
    return mask


def build_splits(n: int) -> dict[str, tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]]:
    """The three calibration/scoring splits this script compares.

    They answer three different questions and it matters which one a
    claim is made under:

      `contiguous_forward`   deployment. Calibrate on the past, predict
                             the future. The honest simulation of using
                             this interval next week.
      `contiguous_reversed`  the same split backwards. A coverage number
                             that survives only one direction was about
                             the calendar, not the method.
      `interleaved_weeks`    alternating week-long blocks, so both sides
                             span every season. This is the version
                             whose exchangeability assumption the data
                             can actually support.

    Args:
        n: Series length.

    Returns:
        `{name: (calibration_mask, scoring_mask)}`.
    """
    early, late = time_ordered_split(n, CALIBRATION_FRACTION, embargo_hours=DEFAULT_BLOCK_HOURS)
    interleaved_calibration, interleaved_scored = interleaved_block_split(
        n, block_hours=DEFAULT_BLOCK_HOURS, calibration_fraction=CALIBRATION_FRACTION
    )
    return {
        "contiguous_forward": (_mask_from_slice(n, early), _mask_from_slice(n, late)),
        "contiguous_reversed": (_mask_from_slice(n, late), _mask_from_slice(n, early)),
        "interleaved_weeks": (interleaved_calibration, interleaved_scored),
    }


def analyse(bundle: TwinBundle, alpha: float) -> dict[str, Any]:
    """Build, score and stress the intervals for one building."""
    measured = bundle.measured_kw
    predicted = bundle.twin.predict_load_kw()
    groups = temperature_bins(bundle.twin.t_ambient_c)
    months_all = np.asarray(bundle.index.month)

    record: dict[str, Any] = {
        "building_id": bundle.building_id,
        "role": bundle.role,
        "year": bundle.twin.year,
        "calibration_artifact": bundle.artifact_name,
        "alpha": alpha,
        "n_hours": int(measured.size),
        "embargo_hours": DEFAULT_BLOCK_HOURS,
        "mondrian_bin_width_k": MONDRIAN_BIN_WIDTH_K,
        "splits": {},
    }

    for split_name, (calibration, scored) in build_splits(measured.size).items():
        intervals = _interval_set(measured, predicted, calibration, scored, alpha, groups)
        scored_measured = measured[scored]
        scored_index = bundle.index[scored]
        months = months_all[scored]
        # Quintiles of MEASURED load: the question "is the interval
        # honest on the hours that matter" is about the big hours, and
        # binning on the prediction would let the model choose its own
        # examination.
        edges = np.quantile(scored_measured, np.linspace(0.0, 1.0, N_LOAD_BINS + 1))
        load_bin = np.clip(np.digitize(scored_measured, edges[1:-1]), 0, N_LOAD_BINS - 1)

        methods: dict[str, Any] = {}
        for name, interval in intervals.items():
            coverage = validate_coverage(scored_measured, interval)
            by_month = coverage_by_group(scored_measured, interval, months)
            by_load = coverage_by_group(scored_measured, interval, load_bin)
            methods[name] = {
                "marginal": coverage.to_dict(),
                "quantile": interval.quantile,
                "normalised": interval.normalised,
                "by_month": {
                    str(int(month)): result.to_dict() for month, result in sorted(by_month.items())
                },
                "by_load_quintile": {
                    str(int(index) + 1): result.to_dict()
                    for index, result in sorted(by_load.items())
                },
                "worst_conditional_pct": min(
                    min(result.empirical_pct for result in by_month.values()),
                    min(result.empirical_pct for result in by_load.values()),
                ),
            }
            logger.info(
                "%s [%s] %-32s %s",
                bundle.building_id,
                split_name,
                name,
                coverage.summary(),
            )

        record["splits"][split_name] = {
            "n_calibration": int(calibration.sum()),
            "n_scored": int(scored.sum()),
            "scored_window": [str(scored_index[0]), str(scored_index[-1])],
            "methods": methods,
        }
        if split_name == "contiguous_forward":
            record["series"] = {
                "index": [str(value) for value in scored_index],
                "measured_kw": scored_measured.tolist(),
                "predicted_kw": intervals["conformal_normalised"].prediction.tolist(),
                "lower_kw": intervals["conformal_normalised"].lower.tolist(),
                "upper_kw": intervals["conformal_normalised"].upper.tolist(),
            }
    return record


def plot_coverage(records: list[dict[str, Any]], path: Path) -> Path:
    """Left: per-month coverage under the interleaved split. Right: the band."""
    figure, axes = plt.subplots(
        len(records), 2, figsize=(13.0, 4.0 * len(records)), squeeze=False
    )

    for row, record in enumerate(records):
        axis = axes[row][0]
        target = 100.0 * (1.0 - record["alpha"])
        split = record["splits"]["interleaved_weeks"]
        months = sorted(split["methods"]["conformal_normalised"]["by_month"], key=int)
        positions = np.arange(len(months))
        width = 0.22
        for offset, name, colour in (
            (-1.5 * width, "conformal_constant", CONSTANT_COLOUR),
            (-0.5 * width, "conformal_normalised", NORMALISED_COLOUR),
            (0.5 * width, "conformal_mondrian_temperature", MONDRIAN_COLOUR),
            (1.5 * width, "gaussian_control", GAUSSIAN_COLOUR),
        ):
            values = [
                split["methods"][name]["by_month"][month]["empirical_pct"] for month in months
            ]
            axis.bar(
                positions + offset,
                values,
                width=width,
                color=colour,
                label=(
                    f"{name.replace('conformal_', '')} "
                    f"({split['methods'][name]['marginal']['empirical_pct']:.1f}% marginal)"
                ),
            )
        axis.axhline(target, color=TARGET_COLOUR, linestyle="--", linewidth=1.2)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"m{month}" for month in months], fontsize=8)
        axis.set_ylim(0.0, 105.0)
        axis.set_ylabel("coverage, %", fontsize=8)
        axis.legend(fontsize=6, ncol=2)
        axis.grid(alpha=0.3, axis="y")
        axis.tick_params(labelsize=8)
        forward = record["splits"]["contiguous_forward"]["methods"]["conformal_normalised"]
        reversed_split = record["splits"]["contiguous_reversed"]["methods"][
            "conformal_normalised"
        ]
        axis.set_title(
            f"{record['building_id']} -- interleaved weeks, per month "
            f"(dashed = {target:.0f}% target).\n"
            f"Same method on a contiguous split: {forward['marginal']['empirical_pct']:.1f}% "
            f"forward, {reversed_split['marginal']['empirical_pct']:.1f}% reversed",
            fontsize=9,
        )

        axis = axes[row][1]
        series = record["series"]
        window = slice(0, min(14 * 24, len(series["measured_kw"])))
        hours = np.arange(len(series["measured_kw"][window]))
        axis.fill_between(
            hours,
            series["lower_kw"][window],
            series["upper_kw"][window],
            color=NORMALISED_COLOUR,
            alpha=0.25,
            label="90% conformal interval",
        )
        axis.plot(
            hours,
            series["predicted_kw"][window],
            color=NORMALISED_COLOUR,
            linewidth=1.2,
            label="twin",
        )
        axis.plot(
            hours, series["measured_kw"][window], color=TARGET_COLOUR, linewidth=1.0, label="meter"
        )
        axis.set_xlabel("hours from the start of the scoring block", fontsize=8)
        axis.set_ylabel("cooling load, kW", fontsize=8)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
        axis.set_title(
            f"two weeks from {record['splits']['contiguous_forward']['scored_window'][0][:10]} "
            "(forward split, normalised interval; median half-width "
            f"{forward['marginal']['median_width'] / 2:.0f} kW)",
            fontsize=9,
        )

    figure.suptitle(
        "L8.3 conformal coverage, training year, time-ordered split. Point model "
        "fitted on the whole year: coverage here is optimistic for an unseen year."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: build intervals, measure coverage, plot, record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])
    train_year = int(config["train_year"])

    records = []
    for building in selected_buildings(arguments.buildings, DEFAULT_GROUPS):
        try:
            bundle = load_twin(building, config, artifacts)
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)
            continue
        records.append(analyse(bundle, arguments.alpha))

    if not records:
        raise SystemExit("no building had a usable calibration artifact")

    summary = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "split": split_name,
                "method": name,
                "marginal %": round(method["marginal"]["empirical_pct"], 2),
                "gate": "PASS" if method["marginal"]["passed"] else "FAIL",
                "worst month/quintile %": round(method["worst_conditional_pct"], 2),
                "median width kW": round(method["marginal"]["median_width"], 0),
                "width % of mean": round(method["marginal"]["mean_relative_width_pct"], 1),
            }
            for record in records
            for split_name, split in record["splits"].items()
            for name, method in split["methods"].items()
        ]
    )
    logger.info("--- conformal coverage, %d ---\n%s", train_year, summary.to_string(index=False))

    plot_coverage(records, arguments.figure)

    # The series are for the figure only -- keeping 2,400 hours x 4
    # columns x 2 buildings in the artifact would make it unreadable and
    # unreviewable, which is how an artifact stops being read.
    for record in records:
        record.pop("series", None)

    out_path = artifacts / f"conformal_coverage_{train_year}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": train_year,
                "note": (
                    "Training year only (ADR-002). Split conformal on a "
                    "time-ordered split with a 168-hour embargo. The point "
                    "model was fitted on the whole year, so these coverage "
                    "numbers are OPTIMISTIC for an unseen year; the "
                    "conditional columns are where the method's assumption "
                    "actually shows strain."
                ),
                "alpha": arguments.alpha,
                "calibration_fraction": CALIBRATION_FRACTION,
                "buildings": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
