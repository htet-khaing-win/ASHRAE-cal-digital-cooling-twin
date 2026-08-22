"""Distribution-free prediction intervals, and what they are worth here.

A point prediction is not decision support. "Raising the setpoint saves
1.9%" and "raising the setpoint saves 1.9%, and the model's own hourly
error is +-1,900 kW" are different statements, and only the second one
lets someone decide whether to act.

**Split conformal prediction** is the method, for three reasons that
matter in this project:

1. It assumes nothing about the error distribution. The residual here
   is demonstrably not Gaussian and demonstrably not independent (M7
   measured a daily variance share of 0.34-0.80 against a white-noise
   0.042), so a `mean +- 1.96 sigma` interval would be a statement about
   a distribution this residual does not have.
2. It wraps the model instead of replacing it. The calibrated physics
   is untouched; conformal only reads its residuals.
3. Its guarantee is finite-sample and exact, not asymptotic:
   `P(y in interval) >= 1 - alpha` for any n, given exchangeability.

THAT LAST CONDITION IS THE WHOLE STORY, and this module refuses to hide
it. Exchangeability fails for time series: hours are autocorrelated,
and a chilled-water plant in July is not exchangeable with the same
plant in January. The marginal guarantee survives reasonably well in
practice; the CONDITIONAL guarantee -- 90% coverage in every month, at
every load level -- does not, and it is the conditional one an operator
actually needs. `coverage_by_group()` exists to measure that gap rather
than to let it pass unmeasured, and `validate_coverage()` reports the
marginal number beside it so the two are never confused.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cooling_twin import SEED

logger = logging.getLogger(__name__)

_PERCENT = 100.0

# Default miss rate. 0.1 -> 90% target coverage, which is what
# 06_ASSESSMENT.md's M8 gate is written against.
DEFAULT_ALPHA = 0.1

# Block length for the moving-block bootstrap, in hours. One week.
# Chosen to exceed the residual's documented structure rather than to
# suit an answer: M7 measured autocorrelation surviving to lag 168, and
# a bootstrap whose blocks are shorter than the dependence it resamples
# over will report an interval that is too narrow.
DEFAULT_BLOCK_HOURS = 168

DEFAULT_BOOTSTRAP_RESAMPLES = 2000

# Floor applied to a normalising scale, as a fraction of its own mean.
# Without it, an hour whose predicted load is near zero gets an interval
# near zero -- infinite confidence exactly where the model has none.
_MIN_SCALE_FRACTION = 0.05


@dataclass(frozen=True)
class ConformalInterval:
    """A prediction interval per hour, plus the constant that produced it.

    The `quantile` is carried alongside the arrays deliberately. It is a
    single number that fully describes the interval's width (before any
    normalisation), it is the number that changes when the calibration
    set changes, and reporting an interval without it makes the interval
    impossible to audit.

    Attributes:
        lower: Lower endpoint per hour.
        upper: Upper endpoint per hour.
        prediction: The point prediction the interval was built around.
        quantile: The conformal score quantile (kW, or dimensionless if
            a `scale` was used).
        alpha: Target miss rate.
        n_calibration: Calibration points the quantile was computed on.
        normalised: Whether a per-hour scale was applied.
    """

    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    prediction: npt.NDArray[np.float64]
    quantile: float
    alpha: float
    n_calibration: int
    normalised: bool = False

    @property
    def width(self) -> npt.NDArray[np.float64]:
        """Interval width per hour."""
        return self.upper - self.lower

    @property
    def target_coverage_pct(self) -> float:
        """The coverage the interval is built to deliver, percent."""
        return _PERCENT * (1.0 - self.alpha)


@dataclass(frozen=True)
class CoverageResult:
    """Measured coverage of an interval on data it was not calibrated on.

    Attributes:
        empirical_pct: Share of points inside the interval.
        target_pct: `100 * (1 - alpha)`.
        n: Points scored.
        mean_width: Mean interval width, same units as the target.
        median_width: Median interval width -- the honest "typical"
            number when a few hours are extremely wide.
        mean_relative_width_pct: Mean width as a percent of the mean
            measured value. An interval is only useful relative to the
            thing it brackets.
    """

    empirical_pct: float
    target_pct: float
    n: int
    mean_width: float
    median_width: float
    mean_relative_width_pct: float

    @property
    def passed(self) -> bool:
        """Whether empirical coverage reached the target.

        No tolerance band is applied. Conformal's guarantee is one-sided
        (`>= 1 - alpha`), so falling short is a real failure of the
        exchangeability assumption and must not be rounded away.
        """
        return self.empirical_pct >= self.target_pct

    def summary(self) -> str:
        """One line, for logs and reports."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}: {self.empirical_pct:.2f}% coverage against a "
            f"{self.target_pct:.0f}% target on {self.n} points; median width "
            f"{self.median_width:.1f} ({self.mean_relative_width_pct:.1f}% of "
            "the mean measured value)"
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-shaped record."""
        return {
            "empirical_pct": self.empirical_pct,
            "target_pct": self.target_pct,
            "n": self.n,
            "mean_width": self.mean_width,
            "median_width": self.median_width,
            "mean_relative_width_pct": self.mean_relative_width_pct,
            "passed": self.passed,
        }


def _validated_residuals(residuals: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Shared validation, so every entry point agrees on what is valid."""
    values = np.asarray(residuals, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("residuals must contain at least one point")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "residuals must be finite. A NaN here would propagate into the "
            "quantile and produce an interval of infinite width that still "
            "looks like a number."
        )
    return values


def conformal_quantile(
    residuals: npt.ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    scale: npt.ArrayLike | None = None,
) -> float:
    """The split-conformal score quantile: how wide the interval must be.

    The conformity score is the absolute residual (optionally divided by
    a per-point scale), and the quantile taken is

        ceil((n + 1) * (1 - alpha)) / n

    NOT the plain `1 - alpha` empirical quantile, and the difference is
    the entire guarantee. The `(n + 1)` accounts for the unseen test
    point being exchangeable with the n calibration points: with n = 100
    and alpha = 0.1 it takes the 91st largest score rather than the 90th.
    On large n the two agree to a fraction of a percent; on small n the
    plain quantile under-covers, and it under-covers in the direction
    that makes a model look better than it is.

    Args:
        residuals: Calibration residuals `measured - predicted`. These
            must come from data the point model did NOT fit, or the
            interval inherits the model's optimism.
        alpha: Target miss rate, in (0, 1).
        scale: Optional per-point normaliser (see
            `normalising_scale()`). Produces intervals whose width
            varies with the hour instead of one constant band.

    Returns:
        The quantile of the conformity scores.

    Raises:
        ValueError: If `alpha` is outside (0, 1), if the residuals are
            empty or non-finite, if `scale` is mismatched or
            non-positive, or if n is too small for the requested alpha
            (`ceil((n+1)(1-alpha)) > n`) -- a case that silently returns
            the maximum score, and therefore an interval with no
            guarantee at all, if it is not caught.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    scores = np.abs(_validated_residuals(residuals))
    if scale is not None:
        scale_values = np.asarray(scale, dtype=float).ravel()
        if scale_values.shape != scores.shape:
            raise ValueError(
                f"scale {scale_values.shape} must match residuals {scores.shape}"
            )
        if np.any(scale_values <= 0.0):
            raise ValueError("scale must be strictly positive everywhere")
        scores = scores / scale_values

    n = scores.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        raise ValueError(
            f"n={n} calibration points cannot support alpha={alpha}: the "
            f"required rank is {rank}. Split conformal needs at least "
            f"{math.ceil(1.0 / alpha) - 1} points before its guarantee means "
            "anything; with fewer, the interval is unbounded."
        )
    return float(np.sort(scores)[rank - 1])


def normalising_scale(
    prediction: npt.ArrayLike, min_fraction: float = _MIN_SCALE_FRACTION
) -> npt.NDArray[np.float64]:
    """A per-hour scale making interval width follow predicted load.

    Constant-width intervals are the default because they are the
    version with the clean guarantee, but they are wrong in a specific
    and visible way: the model's error is much larger on a 15,000 kW
    summer afternoon than on a 3,000 kW spring night, so one band is too
    wide on the quiet hours and too narrow on the hours anyone cares
    about. Normalising by the prediction restores that structure while
    keeping the marginal guarantee intact -- the score is still
    exchangeable, it is just measured in relative terms.

    The floor is not cosmetic: an hour predicted near zero would
    otherwise receive an interval of near-zero width, claiming perfect
    confidence exactly where the inverse model's clip has destroyed the
    information.

    Args:
        prediction: Point predictions, kW.
        min_fraction: Floor, as a fraction of the mean prediction.

    Returns:
        A strictly positive scale per point.

    Raises:
        ValueError: If the predictions are empty, non-finite, or if
            `min_fraction` is not positive.
    """
    if min_fraction <= 0.0:
        raise ValueError(f"min_fraction must be > 0, got {min_fraction}")
    values = _validated_residuals(prediction)
    floor = min_fraction * float(np.abs(values).mean())
    if floor <= 0.0:
        raise ValueError("prediction is identically zero; there is nothing to scale by")
    return np.maximum(np.abs(values), floor)


def conformal_interval(
    prediction: npt.ArrayLike,
    quantile: float,
    alpha: float = DEFAULT_ALPHA,
    n_calibration: int = 0,
    scale: npt.ArrayLike | None = None,
    lower_clip: float | None = 0.0,
) -> ConformalInterval:
    """Build the interval around a prediction from a calibrated quantile.

    Args:
        prediction: Point predictions to bracket, kW.
        quantile: From `conformal_quantile()`, computed on data the
            point model did not fit.
        alpha: The miss rate the quantile was computed at. Carried
            through for reporting; it is not re-applied here.
        n_calibration: Size of the calibration set, for the record.
        scale: Per-hour normaliser. Must be the same KIND of scale the
            quantile was computed with, or the interval is meaningless.
        lower_clip: Physical floor applied to the lower endpoint. `0.0`
            by default because a chilled-water plant cannot deliver
            negative cooling, so an interval reaching below zero claims
            something impossible. This makes coverage CONSERVATIVE (the
            interval can only get narrower where the floor binds) and
            never optimistic. Pass `None` to disable.

    Returns:
        A `ConformalInterval`.

    Raises:
        ValueError: If `quantile` is negative or non-finite, or if
            `scale` does not match the predictions.
    """
    if not np.isfinite(quantile) or quantile < 0.0:
        raise ValueError(f"quantile must be finite and >= 0, got {quantile}")

    values = _validated_residuals(prediction)
    half_width = np.full_like(values, float(quantile))
    normalised = scale is not None
    if scale is not None:
        scale_values = np.asarray(scale, dtype=float).ravel()
        if scale_values.shape != values.shape:
            raise ValueError(
                f"scale {scale_values.shape} must match prediction {values.shape}"
            )
        if np.any(scale_values <= 0.0):
            raise ValueError("scale must be strictly positive everywhere")
        half_width = float(quantile) * scale_values

    lower = values - half_width
    if lower_clip is not None:
        lower = np.maximum(lower, lower_clip)
    return ConformalInterval(
        lower=lower,
        upper=values + half_width,
        prediction=values,
        quantile=float(quantile),
        alpha=alpha,
        n_calibration=int(n_calibration),
        normalised=normalised,
    )


def validate_coverage(measured: npt.ArrayLike, interval: ConformalInterval) -> CoverageResult:
    """Measure how often the interval actually contained the truth.

    The guarantee is only a guarantee under exchangeability, which this
    data violates. Measuring is therefore not a formality: it is the
    only evidence that the assumption held well enough to be useful, and
    a shortfall is a finding about the residual, not a bug in the
    method.

    Args:
        measured: Observed values, aligned with the interval.
        interval: The interval to score.

    Returns:
        A `CoverageResult`.

    Raises:
        ValueError: If the shapes disagree or the measurements are
            non-finite.
    """
    values = _validated_residuals(measured)
    if values.shape != interval.prediction.shape:
        raise ValueError(
            f"measured {values.shape} does not align with the interval "
            f"{interval.prediction.shape}. Scoring an interval against the "
            "wrong hours produces a coverage number that means nothing."
        )

    inside = (values >= interval.lower) & (values <= interval.upper)
    mean_measured = float(np.abs(values).mean())
    result = CoverageResult(
        empirical_pct=_PERCENT * float(inside.mean()),
        target_pct=interval.target_coverage_pct,
        n=int(values.size),
        mean_width=float(interval.width.mean()),
        median_width=float(np.median(interval.width)),
        mean_relative_width_pct=(
            _PERCENT * float(interval.width.mean()) / mean_measured
            if mean_measured
            else float("nan")
        ),
    )
    logger.info("coverage: %s", result.summary())
    return result


def coverage_by_group(
    measured: npt.ArrayLike,
    interval: ConformalInterval,
    groups: Sequence[Any],
) -> dict[Any, CoverageResult]:
    """Coverage within each group -- the conditional check.

    Marginal coverage is an average, and an average of 90% is equally
    consistent with "90% everywhere" and with "99% in winter, 70% in
    summer". The second is what a time-series conformal interval
    actually does, and it is the one that matters: nobody operates a
    plant on the annual average of a guarantee.

    Args:
        measured: Observed values.
        interval: The interval to score.
        groups: A label per point (month, load decile, season...).

    Returns:
        `{group: CoverageResult}`, ordered by first appearance.

    Raises:
        ValueError: If `groups` does not align with the interval.
    """
    labels = np.asarray(groups)
    if labels.shape != interval.prediction.shape:
        raise ValueError(
            f"groups {labels.shape} must align with the interval "
            f"{interval.prediction.shape}"
        )
    values = _validated_residuals(measured)

    results: dict[Any, CoverageResult] = {}
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        subset = ConformalInterval(
            lower=interval.lower[mask],
            upper=interval.upper[mask],
            prediction=interval.prediction[mask],
            quantile=interval.quantile,
            alpha=interval.alpha,
            n_calibration=interval.n_calibration,
            normalised=interval.normalised,
        )
        inside = (values[mask] >= subset.lower) & (values[mask] <= subset.upper)
        mean_measured = float(np.abs(values[mask]).mean())
        results[label] = CoverageResult(
            empirical_pct=_PERCENT * float(inside.mean()),
            target_pct=subset.target_coverage_pct,
            n=int(mask.sum()),
            mean_width=float(subset.width.mean()),
            median_width=float(np.median(subset.width)),
            mean_relative_width_pct=(
                _PERCENT * float(subset.width.mean()) / mean_measured
                if mean_measured
                else float("nan")
            ),
        )
    return results


def time_ordered_split(
    n: int, calibration_fraction: float = 0.7, embargo_hours: int = 0
) -> tuple[slice, slice]:
    """Split a series into a calibration block and a later scoring block.

    Time-ordered, never random. A random split would let 14:00 calibrate
    the interval that 15:00 is then scored against -- two readings from
    the same afternoon of the same building, which are about as
    independent as two photographs of the same room. Coverage measured
    that way is reliably several points too good and never reproduces in
    deployment.

    The embargo drops hours at the boundary. It matters less here than
    in M7's fold construction (the point model is not refitted between
    the blocks), but a residual with autocorrelation surviving to lag
    168 still leaks across an abutting boundary.

    Args:
        n: Length of the series.
        calibration_fraction: Share of the series used to compute the
            conformal quantile.
        embargo_hours: Hours dropped between the two blocks.

    Returns:
        `(calibration_slice, scoring_slice)`.

    Raises:
        ValueError: If the fraction is outside (0, 1), if `n` is too
            small, or if the embargo consumes the scoring block.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )
    if embargo_hours < 0:
        raise ValueError(f"embargo_hours must be >= 0, got {embargo_hours}")
    split = int(calibration_fraction * n)
    start = split + embargo_hours
    if split < 1 or start >= n:
        raise ValueError(
            f"n={n} with calibration_fraction={calibration_fraction} and "
            f"embargo_hours={embargo_hours} leaves one of the two blocks empty"
        )
    return slice(0, split), slice(start, n)


def interleaved_block_split(
    n: int, block_hours: int = DEFAULT_BLOCK_HOURS, calibration_fraction: float = 0.7
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Split into alternating blocks so both halves span the whole year.

    The contiguous split above simulates DEPLOYMENT: calibrate on the
    past, predict the future. This one asks a different and equally
    legitimate question -- "does the interval cover a randomly chosen
    hour of this year" -- and it is the version whose assumption the
    data can actually support, because both blocks then see every
    season.

    Which one to report depends on the claim being made, and reporting
    only the flattering one is the failure mode this pair exists to
    prevent. `validate_intervals.py` runs both.

    Blocks rather than individual hours: assigning single hours would
    put 14:00 in calibration and 15:00 in scoring, and the resulting
    coverage would be measured against a neighbour rather than against
    an unseen hour.

    No embargo is applied at the block boundaries, and that is a known
    residual leak: the last calibration hour of a block abuts the first
    scoring hour of the next. With week-long blocks that is 2 hours in
    168, so the effect is small -- but it is not zero, and it biases
    coverage upward rather than down.

    Args:
        n: Length of the series.
        block_hours: Block length; blocks are dealt alternately.
        calibration_fraction: Share of blocks assigned to calibration.

    Returns:
        `(calibration_mask, scoring_mask)`, disjoint and covering.

    Raises:
        ValueError: If the fraction is outside (0, 1), if `block_hours`
            is not positive, or if either mask ends up empty.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )
    if block_hours < 1:
        raise ValueError(f"block_hours must be >= 1, got {block_hours}")

    block_index = np.arange(n) // block_hours
    n_blocks = int(block_index.max()) + 1
    # Deterministic dealing rather than a random draw: the same series
    # must produce the same split on every run (L0.3), and a stride
    # keeps calibration blocks evenly spread through the year instead of
    # clumping the way a random draw sometimes does.
    stride = max(int(round(1.0 / (1.0 - calibration_fraction))), 2)
    scoring_blocks = set(range(0, n_blocks, stride))
    scoring = np.isin(block_index, list(scoring_blocks))
    calibration = ~scoring
    if not calibration.any() or not scoring.any():
        raise ValueError(
            f"n={n} with block_hours={block_hours} leaves one block empty; "
            "shorten the block or lengthen the series"
        )
    return calibration, scoring


def mondrian_quantiles(
    residuals: npt.ArrayLike,
    groups: Sequence[Any],
    alpha: float = DEFAULT_ALPHA,
    min_group: int = 100,
    scale: npt.ArrayLike | None = None,
) -> tuple[dict[Any, float], float]:
    """One conformal quantile per group -- coverage where it is needed.

    Standard (Mondrian) conformal, and the principled answer to the
    conditional-coverage gap `coverage_by_group()` measures. Instead of
    one quantile for the whole year, each group gets its own, so a
    regime whose errors are twice as large gets an interval twice as
    wide rather than borrowing the average.

    The grouping should be something the interval is allowed to know at
    prediction time. Outdoor temperature bins qualify -- the weather is
    an input to the twin, and M7 located the residual's structure in
    exactly that variable. The measured load does NOT qualify: it is
    what the interval is trying to bracket.

    Groups too small to support the requested alpha fall back to the
    pooled quantile. Silently keeping a group of 12 points at
    alpha = 0.1 would produce its maximum score, an interval with no
    guarantee, and no warning.

    Args:
        residuals: Calibration residuals.
        groups: A group label per residual.
        alpha: Target miss rate.
        min_group: Smallest group that gets its own quantile.
        scale: Optional per-point normaliser.

    Returns:
        `({group: quantile}, pooled_quantile)`.

    Raises:
        ValueError: If `groups` does not align with `residuals`, or for
            any problem raised by `conformal_quantile`.
    """
    values = _validated_residuals(residuals)
    labels = np.asarray(groups)
    if labels.shape != values.shape:
        raise ValueError(f"groups {labels.shape} must align with residuals {values.shape}")
    scale_values = None if scale is None else np.asarray(scale, dtype=float).ravel()

    pooled = conformal_quantile(values, alpha, scale=scale_values)
    quantiles: dict[Any, float] = {}
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        if int(mask.sum()) < min_group:
            logger.info(
                "group %r has %d calibration points (< %d) and falls back to the "
                "pooled quantile",
                label,
                int(mask.sum()),
                min_group,
            )
            quantiles[label] = pooled
            continue
        quantiles[label] = conformal_quantile(
            values[mask], alpha, scale=None if scale_values is None else scale_values[mask]
        )
    return quantiles, pooled


def mondrian_interval(
    prediction: npt.ArrayLike,
    groups: Sequence[Any],
    quantiles: dict[Any, float],
    pooled_quantile: float,
    alpha: float = DEFAULT_ALPHA,
    n_calibration: int = 0,
    scale: npt.ArrayLike | None = None,
    lower_clip: float | None = 0.0,
) -> ConformalInterval:
    """Apply per-group quantiles to a prediction.

    A group never seen during calibration takes the pooled quantile.
    That is a documented compromise rather than a solution: an unseen
    group is precisely the case where the calibration data has nothing
    to say, and the pooled width is a guess whose only merit is being
    the least-informed one available.

    Args:
        prediction: Point predictions.
        groups: A group label per prediction.
        quantiles: From `mondrian_quantiles`.
        pooled_quantile: Fallback for unseen groups.
        alpha: The miss rate the quantiles were computed at.
        n_calibration: Calibration size, for the record.
        scale: Per-point normaliser, same kind as calibration used.
        lower_clip: Physical floor on the lower endpoint.

    Returns:
        A `ConformalInterval` whose `quantile` field carries the POOLED
        value -- the per-group widths differ, so no single number
        describes them, and reporting one group's would be misleading.

    Raises:
        ValueError: If `groups` does not align with `prediction`.
    """
    values = _validated_residuals(prediction)
    labels = np.asarray(groups)
    if labels.shape != values.shape:
        raise ValueError(f"groups {labels.shape} must align with prediction {values.shape}")

    per_point = np.array(
        [float(quantiles.get(label, pooled_quantile)) for label in labels.tolist()]
    )
    half_width = per_point
    normalised = scale is not None
    if scale is not None:
        scale_values = np.asarray(scale, dtype=float).ravel()
        if scale_values.shape != values.shape:
            raise ValueError(f"scale {scale_values.shape} must match prediction {values.shape}")
        if np.any(scale_values <= 0.0):
            raise ValueError("scale must be strictly positive everywhere")
        half_width = per_point * scale_values

    lower = values - half_width
    if lower_clip is not None:
        lower = np.maximum(lower, lower_clip)
    return ConformalInterval(
        lower=lower,
        upper=values + half_width,
        prediction=values,
        quantile=float(pooled_quantile),
        alpha=alpha,
        n_calibration=int(n_calibration),
        normalised=normalised,
    )


def block_bootstrap_ci(
    values: npt.ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    block_hours: int = DEFAULT_BLOCK_HOURS,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """Interval for the MEAN of an autocorrelated hourly series.

    Used for the one quantity conformal cannot bracket: the annual mean
    of a counterfactual difference. Conformal brackets a single future
    observation; the question "how big is the saving over a year" is
    about the mean of 8,760 dependent observations, which is a different
    interval with a different width.

    The blocks are the point. An ordinary i.i.d. bootstrap resamples
    single hours, destroys the autocorrelation, and reports a standard
    error that can be several times too small -- a narrow interval
    around the right centre, which is the most misleading output an
    uncertainty method can produce. Moving blocks of a week preserve the
    dependence within a block.

    WHAT IT DOES NOT COVER: this is sampling variability of the mean
    given the model. It says nothing about the model being wrong, and
    on a counterfactual the model being wrong is the dominant term. Use
    it beside the parameter-ensemble spread, never instead of it.

    Args:
        values: Hourly series (e.g. scenario-minus-baseline kW).
        alpha: Miss rate; the interval is `[alpha/2, 1 - alpha/2]`.
        block_hours: Block length.
        n_resamples: Bootstrap replicates.
        seed: Passed to `np.random.default_rng` (L0.3).

    Returns:
        `(lower, upper)` for the mean.

    Raises:
        ValueError: If the series is shorter than one block, if
            `block_hours` or `n_resamples` is not positive, or if
            `alpha` is outside (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if block_hours < 1:
        raise ValueError(f"block_hours must be >= 1, got {block_hours}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    series = _validated_residuals(values)
    n = series.size
    if n < block_hours:
        raise ValueError(
            f"series has {n} points, shorter than one {block_hours}-hour block. "
            "Shortening the block to fit would silently discard the "
            "autocorrelation this function exists to preserve."
        )

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_hours)
    starts = rng.integers(0, n - block_hours + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(block_hours)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        indices = (starts[index][:, None] + offsets).ravel()[:n]
        means[index] = series[indices].mean()

    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # A synthetic series with the two properties that break naive
    # intervals: heteroscedastic error (bigger when the load is bigger)
    # and strong autocorrelation (a daily cycle, not white noise).
    rng_demo = np.random.default_rng(SEED)
    hours_demo = np.arange(8760)
    truth = 5000.0 + 3000.0 * np.sin(2.0 * np.pi * (hours_demo - 2400) / 8760) + 800.0 * np.sin(
        2.0 * np.pi * hours_demo / 24.0
    )
    noise = rng_demo.normal(0.0, 0.08 * truth)
    measured_demo = truth + noise
    predicted_demo = truth  # a perfect structural model, imperfect hourly

    # Time-ordered split: calibrate the quantile on the first 70%, score
    # coverage on the last 30%. A RANDOM split would let a July hour
    # calibrate the interval for the July hour beside it and report a
    # coverage that no deployment ever sees.
    split = int(0.7 * measured_demo.size)
    calibration = measured_demo[:split] - predicted_demo[:split]

    for label, scale_cal, scale_test in (
        ("constant width", None, None),
        (
            "normalised by prediction",
            normalising_scale(predicted_demo[:split]),
            normalising_scale(predicted_demo[split:]),
        ),
    ):
        q = conformal_quantile(calibration, DEFAULT_ALPHA, scale=scale_cal)
        interval_demo = conformal_interval(
            predicted_demo[split:],
            q,
            alpha=DEFAULT_ALPHA,
            n_calibration=calibration.size,
            scale=scale_test,
        )
        coverage = validate_coverage(measured_demo[split:], interval_demo)
        logger.info("%-26s %s", label + ":", coverage.summary())

        month = ((hours_demo[split:] // 730) % 12) + 1
        by_month = coverage_by_group(measured_demo[split:], interval_demo, month.tolist())
        worst = min(by_month.items(), key=lambda item: item[1].empirical_pct)
        logger.info(
            "%-26s worst block coverage %.1f%% (block %s, n=%d) -- marginal "
            "coverage hides this",
            "",
            worst[1].empirical_pct,
            worst[0],
            worst[1].n,
        )

    # The mean of a difference, with and without blocks -- the i.i.d.
    # bootstrap is shown ONLY to demonstrate how much too narrow it is.
    difference = 0.02 * truth + rng_demo.normal(0.0, 50.0, size=truth.size)
    block_lo, block_hi = block_bootstrap_ci(difference, block_hours=DEFAULT_BLOCK_HOURS)
    iid_lo, iid_hi = block_bootstrap_ci(difference, block_hours=1)
    logger.info(
        "mean difference %.1f kW: week-block CI [%.1f, %.1f] (width %.1f) vs "
        "hour-block CI [%.1f, %.1f] (width %.1f -- too narrow by %.1fx)",
        float(difference.mean()),
        block_lo,
        block_hi,
        block_hi - block_lo,
        iid_lo,
        iid_hi,
        iid_hi - iid_lo,
        (block_hi - block_lo) / (iid_hi - iid_lo),
    )
