"""Probe one calibrated result for parameter sets that fit just as well.

    python scripts/run_equifinality.py --config config/calibration.yaml

Takes the calibration artifact written by `run_calibration.py`, runs the
L6.8 equifinality probe against the SAME objective, the SAME data and
the SAME bounds, and reports three things the calibration itself cannot:

    1. the behavioural range of every fitted parameter,
    2. which parameters trade off against which,
    3. how far a downstream retrofit answer moves across parameter sets
       the data cannot rank.

Everything about the run is reused rather than reimplemented -- the
objective, the data pipeline and the bound resolution are imported from
`run_calibration.py`. A probe that scored the calibration with a second
implementation of the objective would be measuring the difference
between the two implementations.

The ADR-002 test-set lock applies here exactly as it does to the
calibration: this script refuses any year but `train_year`. An
equifinality study is still model fitting, and doing it on 2017 would
burn the held-out year for a diagnostic.
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

# `run_calibration` is a script, not an installed module. Importing it by
# path keeps ONE definition of the objective and the data pipeline; the
# alternative -- copying `CalibrationObjective` into this file -- would
# let the two drift, and a probe of a stale objective is worse than no
# probe.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_calibration import (  # noqa: E402
    CalibrationObjective,
    load_config,
    load_training_data,
    resolve_bounds,
)

from cooling_twin.calibration.equifinality import (  # noqa: E402
    DEFAULT_BEHAVIOURAL_TOLERANCE,
    DEFAULT_N_STARTS,
    EquifinalityStudy,
    OutcomeSpread,
    collect_behavioural_sets,
    compensating_pairs,
    most_divergent_pair,
    outcome_spread,
    parameter_spread,
)
from cooling_twin.calibration.metrics import (  # noqa: E402
    DataInterval,
    ashrae_g14_pass,
)

logger = logging.getLogger("run_equifinality")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l6_8_equifinality.png")

# Retrofits priced across the behavioural sets. Each scales ONE fitted
# parameter by a factor: a ventilation setback (VAV fume hoods, occupancy
# sensing) and an envelope upgrade (glazing, insulation). 30% is a round
# number chosen for legibility, not an engineering estimate for either
# measure -- the interesting quantity is the SPREAD of the answer, which
# is proportional to the factor and so does not depend on it.
RETROFITS = {
    "ventilation setback -30%": ("vent_flow_kg_per_s", 0.70),
    "envelope upgrade -30% UA": ("ua_envelope_w_per_m2k", 0.70),
}

# The verdict is ORDINAL -- identified, weakly identified, unidentified --
# so it is drawn as one hue getting darker, not as a traffic light.
# Red/green is the classic colour-vision trap and there is no polarity
# here to justify two hues anyway. "bound-limited" is a different KIND of
# statement rather than a worse degree of the same one, so it gets grey
# plus a hatch: a second, non-colour channel, which is also what keeps
# the panel readable in print. Every bar is labelled with its verdict in
# text as well, so nothing is encoded by colour alone.
VERDICT_COLOURS = {
    "identified": "#c6dbef",
    "weakly identified": "#6baed6",
    "unidentified": "#2171b5",
    "bound-limited": "#9e9e9e",
}
BEHAVIOURAL_MARK = "#2171b5"
CALIBRATED_MARK = "#b2182b"


class RetrofitSaving:
    """Predicted annual cooling saving from scaling one fitted parameter.

    The counterfactual is model against model, not model against
    measurement: each behavioural set predicts its OWN baseline energy
    and its own retrofitted energy, and the saving is the difference
    between them. Comparing a retrofit prediction against the measured
    data instead would fold the model's own fit error into the answer.

    Attributes:
        parameter: Name of the parameter the retrofit changes.
        factor: Multiplier applied to it.
    """

    def __init__(
        self,
        objective: CalibrationObjective,
        parameter_names: tuple[str, ...],
        parameter: str,
        factor: float,
    ) -> None:
        if parameter not in parameter_names:
            raise KeyError(f"{parameter!r} is not calibrated: {parameter_names}")
        if factor <= 0.0:
            raise ValueError(f"factor must be > 0, got {factor}")
        self._objective = objective
        self._index = parameter_names.index(parameter)
        self.parameter = parameter
        self.factor = factor

    def _annual_kwh(self, vector: npt.NDArray[np.float64]) -> float:
        """Annual cooling energy at one parameter set, kWh.

        Hourly samples, so a sum of kW is a sum of kWh -- over the hours
        that survived cleaning, which is not quite a calendar year. That
        is fine here because both terms of the saving use the same hours.
        """
        clipped_kw, _raw_kw = self._objective.predict(vector)
        return float(np.sum(clipped_kw))

    def __call__(self, vector: npt.NDArray[np.float64]) -> float:
        """Saving as a percentage of that set's own baseline energy."""
        baseline = self._annual_kwh(vector)
        if baseline <= 0.0:
            raise ValueError(
                f"baseline annual energy is {baseline} kWh at {vector} -- a "
                "parameter set predicting no cooling at all cannot be given a "
                "percentage saving."
            )
        retrofitted = np.asarray(vector, dtype=float).copy()
        retrofitted[self._index] *= self.factor
        return 100.0 * (baseline - self._annual_kwh(retrofitted)) / baseline


def load_reference(
    artifact_path: Path,
    config: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    year: int,
) -> tuple[list[float], dict[str, Any]]:
    """Read the calibrated vector to be probed, and check it belongs here.

    Four things are verified before the vector is used: the building,
    the year, the parameter names and the bounds. A study run against a
    vector from a different building, or inside a different box, would
    produce a spread that looks like a finding and describes nothing.

    Args:
        artifact_path: JSON written by `run_calibration.py`.
        config: Parsed calibration config.
        bounds: Bounds resolved for THIS run.
        year: The year being probed.

    Returns:
        `(reference_vector, artifact_record)`.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the artifact does not match this configuration.
    """
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"no calibration artifact at {artifact_path}. Run "
            "scripts/run_calibration.py first -- there is nothing to probe "
            "until something has been calibrated."
        )
    record = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = record.get("metadata", {})

    if metadata.get("building_id") != config["building_id"]:
        raise ValueError(
            f"artifact is for {metadata.get('building_id')!r} but the config "
            f"says {config['building_id']!r}"
        )
    if int(metadata.get("year", -1)) != year:
        raise ValueError(
            f"artifact is for {metadata.get('year')} but this run probes {year}"
        )

    artifact_bounds = record["bounds"]
    if tuple(artifact_bounds) != tuple(bounds):
        raise ValueError(
            f"parameter names differ: artifact {tuple(artifact_bounds)} vs "
            f"config {tuple(bounds)}"
        )
    for name, (lower, upper) in bounds.items():
        recorded = artifact_bounds[name]
        if not np.allclose(recorded, [lower, upper], rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"{name}: the artifact was calibrated in [{recorded[0]}, "
                f"{recorded[1]}] but this run resolves [{lower}, {upper}]. The "
                "probe must use the box the calibration used, or the spread it "
                "reports belongs to a different calibration."
            )

    reference = [float(record["parameters"][name]) for name in bounds]
    logger.info(
        "probing %s (objective %.6f, %s)",
        artifact_path.name,
        record["objective_value"],
        ", ".join(f"{name}={value:.3f}" for name, value in zip(bounds, reference, strict=True)),
    )
    return reference, record


def latest_artifact(directory: Path, building_id: str) -> Path:
    """Most recent calibration artifact for one building.

    Args:
        directory: Where `run_calibration.py` writes.
        building_id: The building whose artifact is wanted.

    Returns:
        Path to the newest matching artifact, by filename (the names
        carry an ISO timestamp, so lexical order is chronological).

    Raises:
        FileNotFoundError: If no artifact in `directory` is for this
            building.
    """
    matching = [
        path
        for path in sorted(directory.glob("calibration_*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("metadata", {}).get(
            "building_id"
        )
        == building_id
    ]
    if not matching:
        raise FileNotFoundError(
            f"no calibration artifact for {building_id} in {directory}"
        )
    return matching[-1]


def spread_table(study: EquifinalityStudy) -> str:
    """Format the behavioural range of every parameter, with a pinning flag.

    A parameter that every behavioural set leaves on the same BOUND has
    a span of zero and would otherwise be reported as "identified" -- by
    the box, not by the data. The flag says which.
    """
    lines = [
        f"{'parameter':<28}{'behavioural range':>26}{'% of box':>10}  verdict",
    ]
    for name, spread in parameter_spread(study).items():
        lines.append(
            f"{name:<28}{spread.minimum:11.3f} .. {spread.maximum:10.3f}"
            f"{spread.span_fraction:9.1%}  {spread.verdict}"
            + (f"  [{spread.pinned_at.upper()} BOUND]" if spread.pinned_at else "")
        )
    return "\n".join(lines)


def plot_study(
    study: EquifinalityStudy,
    savings: dict[str, Any],
    building_id: str,
    path: Path,
) -> Path:
    """Three panels: the ridge, the spread, and the price of the spread.

    Args:
        study: A completed study.
        savings: `{label: OutcomeSpread}` for each retrofit priced.
        building_id: For the title.
        path: Where to write the PNG.

    Returns:
        The path written.
    """
    spreads = parameter_spread(study)
    names = study.parameter_names
    matrix = study.matrix
    reference = study.reference_set

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1 -- the compensating pair if there are enough sets to find
    # one, otherwise the two widest-roaming parameters. The figure must
    # not depend on the correlation being reportable: a study too small
    # to correlate is exactly the study whose picture is most needed.
    try:
        pairs = compensating_pairs(study)
    except ValueError:
        pairs = ()
    if pairs:
        x_name, y_name = pairs[0][0], pairs[0][1]
    else:
        widest = sorted(
            spreads.values(), key=lambda spread: -spread.span_fraction
        )[:2]
        x_name, y_name = widest[0].name, widest[1].name
    x_index, y_index = names.index(x_name), names.index(y_name)
    axes[0].scatter(
        matrix[:, x_index],
        matrix[:, y_index],
        s=70,
        facecolors="none",
        edgecolors=BEHAVIOURAL_MARK,
        linewidths=1.6,
        label="behavioural sets",
    )
    if reference is not None:
        axes[0].scatter(
            reference.parameters[x_index],
            reference.parameters[y_index],
            s=140,
            marker="*",
            color=CALIBRATED_MARK,
            label="calibrated (L6.7)",
            zorder=3,
        )
    # Zoomed to the behavioural family, not to the bounds. Drawn against
    # the full box the family is a single speck and the trade-off -- the
    # only thing this panel exists to show -- is invisible. The box is
    # stated on the axis instead, so nothing is hidden: panel 2 is where
    # the family's size RELATIVE to the box is read.
    for axis, name, index in (
        (axes[0].set_xlabel, x_name, x_index),
        (axes[0].set_ylabel, y_name, y_index),
    ):
        low, high = study.bounds[index]
        axis(f"{name}\n(bounds {low:g} to {high:g})")
    for setter, index in ((axes[0].set_xlim, x_index), (axes[0].set_ylim, y_index)):
        values = matrix[:, index]
        margin = 0.25 * (values.max() - values.min()) or 1.0
        setter(values.min() - margin, values.max() + margin)
    axes[0].set_title(
        f"1. every point fits within {study.tolerance:.0%} of the best"
        + (
            f"\nr = {pairs[0][2]:+.3f}"
            if pairs
            else f"\n{len(study.sets)} sets -- too few to correlate"
        )
    )
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2 -- how much of its own box each parameter roams.
    ordered = sorted(spreads.values(), key=lambda spread: spread.span_fraction)
    bars = axes[1].barh(
        [spread.name for spread in ordered],
        [100 * spread.span_fraction for spread in ordered],
        color=[VERDICT_COLOURS[spread.verdict] for spread in ordered],
        hatch=["//" if spread.pinned_at else "" for spread in ordered],
        edgecolor="white",
    )
    for bar, spread in zip(bars, ordered, strict=True):
        if spread.pinned_at:
            bar.set_edgecolor(VERDICT_COLOURS["unidentified"])
    for index, spread in enumerate(ordered):
        percent = 100 * spread.span_fraction
        label = spread.verdict + (f" (at {spread.pinned_at} bound)" if spread.pinned_at else "")
        # The axis is a percentage and so cannot be widened to make room:
        # a long label on a long bar goes INSIDE it instead of running
        # off the panel.
        inside = percent > 55.0
        axes[1].text(
            percent - 2.0 if inside else percent + 1.5,
            index,
            label,
            va="center",
            ha="right" if inside else "left",
            color="white" if inside else "black",
            fontsize=8,
        )
    axes[1].set_xlim(0, 100)
    axes[1].axvline(10, color="k", linestyle=":", linewidth=1)
    axes[1].axvline(50, color="k", linestyle=":", linewidth=1)
    axes[1].set_xlabel("behavioural range, % of the parameter's own bounds")
    axes[1].set_title("2. what the data actually identifies")
    axes[1].grid(axis="x", alpha=0.3)

    # Panel 3 -- the same retrofit, priced by every admissible set.
    for offset, spread in enumerate(savings.values()):
        axes[2].scatter(
            spread.values,
            np.full(len(spread.values), offset),
            s=60,
            facecolors="none",
            edgecolors=BEHAVIOURAL_MARK,
            linewidths=1.6,
            label="behavioural sets" if offset == 0 else None,
        )
        if spread.reference is not None:
            axes[2].scatter(
                spread.reference,
                offset,
                s=140,
                marker="*",
                color=CALIBRATED_MARK,
                zorder=3,
                label="calibrated (L6.7)" if offset == 0 else None,
            )
        axes[2].annotate(
            f"{spread.minimum:.1f}% to {spread.maximum:.1f}%",
            (0.5 * (spread.minimum + spread.maximum), offset + 0.24),
            ha="center",
            fontsize=9,
        )
    axes[2].set_yticks(range(len(savings)))
    axes[2].set_yticklabels(list(savings), fontsize=9)
    axes[2].set_ylim(-0.55, len(savings) - 0.25)
    lows = [spread.minimum for spread in savings.values()]
    highs = [spread.maximum for spread in savings.values()]
    pad = 0.12 * (max(highs) - min(lows) or 1.0)
    axes[2].set_xlim(min(lows) - pad, max(highs) + pad)
    axes[2].set_xlabel("predicted annual cooling saving, %")
    axes[2].set_title("3. the answer the twin would give")
    axes[2].legend(loc="lower right", fontsize=8)
    axes[2].grid(axis="x", alpha=0.3)

    figure.suptitle(
        f"Equifinality: {building_id}, {len(study.sets)} parameter sets the "
        f"calibration cannot tell apart "
        f"(restarts drawn from {study.start_spread:.0%} of each bound width)"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: load, probe, price, report, write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Calibration artifact to probe. Defaults to the most recent one "
        "for this config's building.",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_BEHAVIOURAL_TOLERANCE)
    parser.add_argument("--n-starts", type=int, default=DEFAULT_N_STARTS)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--start-spread",
        type=float,
        default=1.0,
        help="Fraction of each bound width the restarts are drawn from, "
        "centred on the calibrated vector. 1.0 (default) scatters over the "
        "whole box and asks whether a DISTANT rival fits as well; a small "
        "value maps the ridge the optimum sits on. Different questions.",
    )
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        help="Redraw the figure from an existing equifinality artifact and "
        "exit. The study records every refinement, so regenerating a report "
        "figure costs nothing -- there is no reason to re-run 40 refinements "
        "because a label was in the wrong place.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    if arguments.replot is not None:
        record = json.loads(arguments.replot.read_text(encoding="utf-8"))
        restored = EquifinalityStudy.from_dict(record)
        plot_study(
            restored,
            {
                label: OutcomeSpread(
                    values=tuple(entry["values"]),
                    minimum=entry["min"],
                    maximum=entry["max"],
                    median=entry["median"],
                    reference=entry["calibrated"],
                )
                for label, entry in record["retrofit_savings_pct"].items()
            },
            record["building_id"],
            arguments.figure,
        )
        return

    config = load_config(arguments.config)
    year = int(config["train_year"])
    artifacts_dir = Path(config["artifacts"]["directory"])

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

    artifact_path = arguments.artifact or latest_artifact(
        artifacts_dir, config["building_id"]
    )
    reference, _record = load_reference(artifact_path, config, bounds, year)

    study = collect_behavioural_sets(
        objective,
        bounds,
        reference,
        tolerance=arguments.tolerance,
        n_starts=arguments.n_starts,
        workers=arguments.workers,
        start_spread=arguments.start_spread,
    )

    logger.info("--- what the data identifies ---\n%s", spread_table(study))

    # The CV(RMSE) span the behavioural threshold actually corresponds
    # to. "Within 5% of the objective" means nothing to a reviewer;
    # "every one of these scores between 11.7% and 12.4% CV(RMSE)" does.
    scores = [
        ashrae_g14_pass(
            objective.observed_kw,
            objective.predict(np.asarray(entry.parameters))[0],
            n_params=len(bounds),
            interval=DataInterval(config["objective"]["interval"]),
        )
        for entry in study.sets
    ]
    logger.info(
        "CV(RMSE) across the behavioural sets: %.2f%% to %.2f%% (all %s G14)",
        min(score.cvrmse_pct for score in scores),
        max(score.cvrmse_pct for score in scores),
        "PASS" if all(score.passed for score in scores) else "MIXED on",
    )

    # How many refinements the threshold admits is a judgement, so it is
    # reported at three values rather than one. Free: `rethreshold` uses
    # the refinements already run.
    logger.info(
        "--- sensitivity to the behavioural threshold ---\n%s",
        "\n".join(
            f"  tolerance {alternative:5.1%}  ->  "
            f"{len(study.rethreshold(alternative).sets):2d} behavioural sets"
            for alternative in (0.02, arguments.tolerance, 0.10, 0.20)
        ),
    )

    try:
        for name_a, name_b, correlation in compensating_pairs(study):
            logger.info(
                "compensating pair: %s <-> %s   r = %+.3f", name_a, name_b, correlation
            )
    except ValueError as error:
        # Not fatal, and not silently skipped either: a study too small
        # to correlate is a statement about the run, and the fix
        # (more starts) belongs in the log where it will be read.
        logger.warning("correlations not reported -- %s", error)

    if study.rejected:
        rejected = sorted(entry.objective for entry in study.rejected)
        logger.info(
            "%d refinements landed OUTSIDE the threshold, objectives %.4f to "
            "%.4f (threshold %.4f) -- other basins, not near misses, which is "
            "the same multi-modality L6.7's global stage exists for",
            len(rejected),
            rejected[0],
            rejected[-1],
            study.threshold,
        )

    savings = {}
    for label, (parameter, factor) in RETROFITS.items():
        savings[label] = outcome_spread(
            study, RetrofitSaving(objective, study.parameter_names, parameter, factor)
        )
        logger.info("%-26s %s", label, savings[label].summary("%"))

    widest = max(
        parameter_spread(study).values(), key=lambda spread: spread.span_fraction
    )
    low, high = most_divergent_pair(study, widest.name)
    logger.info(
        "--- the exhibit: two admissible readings of this building ---\n%s",
        "\n".join(
            f"  {label}: objective {entry.objective:.4f}  "
            + "  ".join(
                f"{name}={value:.2f}"
                for name, value in zip(study.parameter_names, entry.parameters, strict=True)
            )
            for label, entry in (("A", low), ("B", high))
        ),
    )

    plot_study(study, savings, config["building_id"], arguments.figure)

    record = study.to_dict()
    record["building_id"] = config["building_id"]
    record["year"] = year
    record["calibration_artifact"] = artifact_path.name
    record["cvrmse_pct_range"] = [
        min(score.cvrmse_pct for score in scores),
        max(score.cvrmse_pct for score in scores),
    ]
    record["tolerance_sensitivity"] = {
        f"{alternative:.3f}": len(study.rethreshold(alternative).sets)
        for alternative in (0.02, arguments.tolerance, 0.10, 0.20)
    }
    record["retrofit_savings_pct"] = {
        label: {
            "min": spread.minimum,
            "max": spread.maximum,
            "median": spread.median,
            "calibrated": spread.reference,
            "values": list(spread.values),
        }
        for label, spread in savings.items()
    }
    stem = artifact_path.stem.removeprefix("calibration_")
    # The sampling window goes in the FILENAME, not just the record:
    # a whole-box study and a ridge study of the same calibration are
    # different results and must not overwrite each other.
    window = f"spread{round(100 * arguments.start_spread):03d}"
    out_path = artifacts_dir / f"equifinality_{stem}_{window}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()
