"""Naive baselines -- the reference every calibrated result is judged against.

A CV(RMSE) of 22% is not a result. It is a number waiting for a
comparison. If the annual mean alone scores 48%, then 22% is real work;
if the annual mean scores 25%, the entire physics model has bought
almost nothing and something is wrong with either the model or the
building's load being trivially flat.

`06_ASSESSMENT.md`'s M6 supporting requirements make this explicit: the
calibrated model must beat the best baseline by at least 30% relative
CV(RMSE). That check cannot be run until the baselines exist, so they
are built before the optimiser (L6.7), not after it.

Two baselines are provided, in increasing order of how much they know:

    annual mean         knows the target's mean and nothing else
    linear regression   knows outdoor temperature (or any features given)

Both are ordinary least squares fits and both return a `BaselineFit`
that predicts through the same `.predict()` call, so the SAME
`ashrae_g14_pass()` scores the baselines and the calibrated model. A
baseline scored by a different code path than the model it is meant to
be compared against is not a comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_PERCENT = 100.0

# 06_ASSESSMENT.md, M6 supporting requirements: "Calibrated model beats
# best baseline by >= 30% relative CV(RMSE)". Same reasoning as L6.3's
# G14 thresholds -- this is an acceptance criterion, so it lives in code
# where widening it is a visible, deliberate act.
MIN_RELATIVE_IMPROVEMENT_PCT = 30.0


@dataclass(frozen=True)
class BaselineFit:
    """A fitted baseline model.

    Holds the fitted coefficients rather than the predictions, so the
    same fit can be applied to a different period -- fitting on 2016 and
    predicting 2017 is the only honest way to score a baseline against a
    held-out year (L6.10), and a class that stored predictions could not
    do it.

    Attributes:
        name: Human-readable label, used in report tables.
        coefficients: `[intercept, slope_1, ..., slope_k]`. Length is
            `n_params`.
        n_params: Number of fitted parameters `p`, for the `n - p`
            denominator in `metrics.cvrmse`. Counting this correctly is
            what keeps the baseline comparison fair -- see the L6.4
            rationale.
    """

    name: str
    coefficients: npt.NDArray[np.float64]
    n_params: int
    change_point: float | None = None

    def predict(self, features: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Predict the target for the given feature rows.

        Args:
            features: `(n,)` or `(n, k)` array of predictor values. The
                annual-mean baseline uses only the row count and
                ignores the values -- that is precisely what makes it a
                naive baseline, and keeping the signature uniform is
                what lets a comparison loop score every baseline and
                the calibrated model through one code path.

        Returns:
            Predicted values, shape `(n,)`.

        Raises:
            ValueError: If the number of feature columns does not match
                the columns this baseline was fitted on.
        """
        x = np.asarray(features, dtype=float)
        if self.change_point is not None:
            # 3P: the feature is the EXCESS above the change point, so
            # the model is flat below it. Applied here rather than by the
            # caller so that a change-point fit predicts through the same
            # `.predict()` as every other baseline -- the property this
            # whole module exists to preserve.
            x = np.maximum(0.0, x - self.change_point)
        # Width comes from the COEFFICIENT count, not from `n_params`.
        # The two are equal for the OLS baselines but not for the
        # change-point fit, whose breakpoint is a fitted parameter that
        # does not appear in the design matrix. Counting it in n_params
        # is what keeps `cvrmse`'s n - p denominator honest; excluding it
        # here is what keeps the matrix the right shape.
        design = _design_matrix(x, n_slopes=len(self.coefficients) - 1)
        return np.asarray(design @ self.coefficients, dtype=float)


def _design_matrix(
    features: npt.ArrayLike,
    n_slopes: int,
) -> npt.NDArray[np.float64]:
    """Build `[1, x_1, ..., x_k]` and check it has the expected width.

    Args:
        features: `(n,)` or `(n, k)` predictor array.
        n_slopes: How many feature columns are expected, i.e. `p - 1`.
            Pass 0 for an intercept-only (annual mean) model.

    Returns:
        Design matrix of shape `(n, n_slopes + 1)`.

    Raises:
        ValueError: If `features` is empty, not 1- or 2-dimensional,
            non-finite, or has a different number of columns than
            `n_slopes`.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"features must be 1- or 2-dimensional, got {x.ndim} dimensions")
    if x.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "features must be finite -- a NaN outdoor temperature would "
            "propagate silently through lstsq into every coefficient."
        )

    n_rows = x.shape[0]
    if n_slopes == 0:
        return np.ones((n_rows, 1), dtype=float)
    if x.shape[1] != n_slopes:
        raise ValueError(
            f"this baseline was fitted on {n_slopes} feature column(s) but "
            f"was given {x.shape[1]} -- predicting with a different feature "
            "set than the fit used produces a plausible-looking, meaningless "
            "prediction."
        )
    return np.column_stack([np.ones(n_rows, dtype=float), x])


def _fit_ols(
    design: npt.NDArray[np.float64],
    measured: npt.NDArray[np.float64],
    name: str,
) -> BaselineFit:
    """Solve `design @ beta = measured` in the least-squares sense.

    Uses `np.linalg.lstsq` (SVD-based) rather than the normal equations
    `(X'X)^-1 X'y`, which squares the condition number and loses
    precision on collinear features, and rather than adding
    scikit-learn as a dependency for what is one call.

    Args:
        design: `(n, p)` design matrix, intercept column included.
        measured: `(n,)` target values.
        name: Label for the resulting `BaselineFit`.

    Returns:
        The fitted baseline.

    Raises:
        ValueError: If the design matrix is rank-deficient (collinear or
            constant features), which would otherwise yield an
            arbitrary member of a family of equally-good solutions.
    """
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design, measured, rcond=None
    )
    n_params = design.shape[1]
    if rank < n_params:
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < {n_params} "
            "columns) -- a constant or collinear feature was passed. "
            "lstsq would return one arbitrary solution out of infinitely "
            "many, and its coefficients would be uninterpretable."
        )
    return BaselineFit(
        name=name,
        coefficients=np.asarray(coefficients, dtype=float),
        n_params=n_params,
    )


def _validated_target(measured: npt.ArrayLike, n_rows: int) -> npt.NDArray[np.float64]:
    """Check the target array and its alignment with the design matrix."""
    y = np.asarray(measured, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"measured must be 1-dimensional, got {y.ndim} dimensions")
    if y.size != n_rows:
        raise ValueError(
            f"measured has {y.size} points but features has {n_rows} rows -- "
            "these must be aligned observations, not merely similar lengths."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("measured must be finite -- clean NaN in M3, not here")
    return y


def fit_annual_mean(measured: npt.ArrayLike) -> BaselineFit:
    """Fit the weakest defensible baseline: a constant at the mean.

    This is the floor. Any model that cannot beat it has learned
    nothing about the building. Note it fits exactly ONE parameter --
    the mean itself -- so `n_params` is 1, not 0.

    Args:
        measured: Target values `y` from the training period.

    Returns:
        A `BaselineFit` whose `predict()` returns the training mean for
        every row.

    Raises:
        ValueError: If `measured` is empty, non-1-D, or non-finite.
    """
    y = np.asarray(measured, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"measured must be 1-dimensional, got {y.ndim} dimensions")
    if y.size == 0:
        raise ValueError("measured must contain at least one point")
    if not np.all(np.isfinite(y)):
        raise ValueError("measured must be finite -- clean NaN in M3, not here")

    design = np.ones((y.size, 1), dtype=float)
    return _fit_ols(design, y, name="annual mean")


def fit_linear_regression(
    features: npt.ArrayLike,
    measured: npt.ArrayLike,
    name: str = "linear regression",
) -> BaselineFit:
    """Fit an ordinary least squares baseline on the given features.

    With outdoor dry-bulb temperature as the single feature this is the
    classic utility-analysis regression: cooling load rises roughly
    linearly with outdoor temperature above a balance point. It knows
    nothing about thermal mass, occupancy, or humidity -- which is
    exactly what the RC model is supposed to add, and therefore exactly
    what the comparison is measuring.

    Args:
        features: `(n,)` or `(n, k)` predictor array from the training
            period, e.g. outdoor dry-bulb temperature.
        measured: `(n,)` target values, aligned with `features`.
        name: Label for the resulting `BaselineFit`.

    Returns:
        The fitted baseline, with `n_params = k + 1` (slopes plus
        intercept).

    Raises:
        ValueError: If the arrays are misaligned, empty, non-finite, or
            if the features are collinear or constant.
    """
    x = np.asarray(features, dtype=float)
    n_slopes = 1 if x.ndim == 1 else x.shape[1] if x.ndim == 2 else 0
    if x.ndim == 2 and n_slopes == 0:
        raise ValueError(
            "features has zero columns -- fitting no slopes would silently "
            "produce an annual-mean baseline under a regression's name. "
            "Call fit_annual_mean() if that is what you want."
        )
    design = _design_matrix(features, n_slopes=n_slopes)
    y = _validated_target(measured, n_rows=design.shape[0])
    return _fit_ols(design, y, name=name)


# Candidate change points are quantiles of the building's OWN outdoor
# temperature, from the 5th to the 95th percentile in 1-point steps. A
# fixed degC grid was rejected for the same reason fixed band edges were
# (L7.1c): this portfolio runs from -24 to +44 degC, so one grid either
# misses a building's range entirely or wastes most of its candidates.
CHANGE_POINT_QUANTILES = np.arange(0.05, 0.96, 0.01)

# Hours that must lie above a candidate change point before it is
# considered. Below this the sloped segment is fitted on a handful of
# points, and the search will happily put the breakpoint in the extreme
# tail where two outliers define the slope.
MIN_SEGMENT_HOURS = 200


def fit_change_point(
    t_ambient_c: npt.ArrayLike,
    measured: npt.ArrayLike,
    name: str = "change-point (3P cooling)",
) -> BaselineFit:
    """Fit the ASHRAE/IMT three-parameter cooling change-point model.

        load = base + slope * max(0, T - T_cp)

    This is the standard inverse model for a load with a floor: flat
    below the change point, rising linearly above it. It is the shape
    utility-bill analysis has used for decades, and it is added here
    because it is the shape `Hog_education_Cathleen`'s data actually has
    -- a weather-insensitive base load through winter, rising with
    temperature in summer. The RC model cannot express that: its only
    constant term competes against a large negative deltaT term in cold
    weather, so it clips at zero instead (Q10, ADR-015).

    Scoring a 3-parameter change-point model against a 5-parameter
    physics model is the sharpest available statement of the finding. It
    is a BASELINE, not a change to the twin -- ADR-015 authorises it on
    the training year only.

    T_cp is found by grid search rather than by gradient descent: the
    sum of squares is piecewise-smooth in the breakpoint with flat
    stretches between data points, so a gradient method stalls, and the
    grid is cheap (91 candidates, one `lstsq` each). This is also what
    the reference implementations do.

    Args:
        t_ambient_c: Outdoor dry-bulb temperature from the training
            period.
        measured: Target values, aligned with `t_ambient_c`.
        name: Label for the resulting `BaselineFit`.

    Returns:
        A `BaselineFit` with `n_params = 3` -- base, slope AND the
        breakpoint, which is fitted and must be counted. Under-counting
        it would flatter the baseline against the model it is being
        compared with, which is the one direction of error this module
        must not make.

    Raises:
        ValueError: If the arrays are misaligned, empty or non-finite,
            or if no candidate breakpoint leaves `MIN_SEGMENT_HOURS`
            hours above it.
    """
    temperature = np.asarray(t_ambient_c, dtype=float)
    if temperature.ndim != 1:
        raise ValueError(
            f"t_ambient_c must be 1-dimensional, got {temperature.ndim} dimensions"
        )
    if temperature.size == 0:
        raise ValueError("t_ambient_c must contain at least one point")
    if not np.all(np.isfinite(temperature)):
        raise ValueError("t_ambient_c must be finite -- clean NaN in M3, not here")
    y = _validated_target(measured, n_rows=temperature.size)

    candidates = np.unique(np.quantile(temperature, CHANGE_POINT_QUANTILES))
    best: tuple[float, float, npt.NDArray[np.float64]] | None = None
    for candidate in candidates:
        above = temperature > candidate
        if above.sum() < MIN_SEGMENT_HOURS:
            continue
        excess = np.maximum(0.0, temperature - candidate)
        design = np.column_stack([np.ones_like(excess), excess])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        sum_of_squares = float(((y - design @ coefficients) ** 2).sum())
        if best is None or sum_of_squares < best[0]:
            best = (sum_of_squares, float(candidate), coefficients)

    if best is None:
        raise ValueError(
            f"no candidate change point leaves {MIN_SEGMENT_HOURS} hours above "
            "it -- this temperature series is too short or too narrow for a "
            "change-point model."
        )

    _sse, change_point, coefficients = best
    if coefficients[1] <= 0.0:
        # A cooling change-point model with a non-positive slope is
        # describing a heating load, or a meter that is not measuring
        # what it is assumed to. Fitted and returned either way -- the
        # caller is comparing baselines, not accepting one -- but never
        # silently.
        logger.warning(
            "%s: fitted slope is %.2f kW/K, not positive. A cooling "
            "change-point model that slopes DOWN with temperature is a "
            "statement about the meter, not about the building.",
            name,
            coefficients[1],
        )
    if change_point <= candidates[0] or change_point >= candidates[-1]:
        logger.warning(
            "%s: change point %.1f degC sits on the edge of the searched "
            "range [%.1f, %.1f]. The data does not support a breakpoint -- "
            "this fit has degenerated towards a straight line or a constant.",
            name,
            change_point,
            candidates[0],
            candidates[-1],
        )

    logger.info(
        "%s: base %.1f kW, slope %.2f kW/K above %.1f degC",
        name,
        coefficients[0],
        coefficients[1],
        change_point,
    )
    return BaselineFit(
        name=name,
        coefficients=np.asarray(coefficients, dtype=float),
        n_params=3,
        change_point=change_point,
    )


def relative_cvrmse_improvement_pct(
    baseline_cvrmse_pct: float,
    model_cvrmse_pct: float,
) -> float:
    """How much better the model is than the baseline, in relative percent.

    `(baseline - model) / baseline * 100`. Relative, not absolute:
    going from 48% to 34% (a 29% relative gain) and from 10% to 7% (a
    30% relative gain) are comparable amounts of work, while their
    absolute gaps -- 14 points versus 3 -- are not.

    Args:
        baseline_cvrmse_pct: The baseline's CV(RMSE), percent.
        model_cvrmse_pct: The candidate model's CV(RMSE), percent.

    Returns:
        Relative improvement in percent. Negative if the model is worse
        than the baseline.

    Raises:
        ValueError: If `baseline_cvrmse_pct` is not strictly positive.
    """
    if baseline_cvrmse_pct <= 0.0:
        raise ValueError(
            f"baseline_cvrmse_pct must be > 0, got {baseline_cvrmse_pct}. "
            "A baseline with zero error means the target is constant or the "
            "baseline saw the answer -- either way the comparison is void."
        )
    return (baseline_cvrmse_pct - model_cvrmse_pct) / baseline_cvrmse_pct * _PERCENT


def beats_baseline(
    baseline_cvrmse_pct: float,
    model_cvrmse_pct: float,
    min_improvement_pct: float = MIN_RELATIVE_IMPROVEMENT_PCT,
) -> bool:
    """Apply 06_ASSESSMENT.md's >= 30% relative CV(RMSE) requirement.

    Args:
        baseline_cvrmse_pct: The BEST baseline's CV(RMSE), percent --
            not the weakest. Beating only the annual mean is not the
            requirement.
        model_cvrmse_pct: The calibrated model's CV(RMSE), percent.
        min_improvement_pct: Required relative improvement. Defaults to
            the project's 30%.

    Returns:
        True if the model clears the bar. Inclusive at the boundary.

    Raises:
        ValueError: See `relative_cvrmse_improvement_pct`.
    """
    improvement = relative_cvrmse_improvement_pct(baseline_cvrmse_pct, model_cvrmse_pct)
    return improvement >= min_improvement_pct


if __name__ == "__main__":
    from cooling_twin import set_seed
    from cooling_twin.calibration.metrics import cvrmse

    logging.basicConfig(level=logging.INFO)

    # A synthetic year whose load depends on outdoor temperature plus a
    # daily occupancy cycle the temperature alone cannot explain. That
    # residual structure is what a physics model is supposed to capture
    # and a temperature regression is not.
    rng = set_seed()
    hours = np.arange(8760)
    outdoor_c = 15.0 + 12.0 * np.sin(2 * np.pi * (hours - 2160) / 8760)
    occupancy = np.where((hours % 24 >= 8) & (hours % 24 < 18), 1.0, 0.35)
    load_kw = 400.0 + 60.0 * outdoor_c + 500.0 * occupancy + rng.normal(0, 80, 8760)

    mean_fit = fit_annual_mean(load_kw)
    regression_fit = fit_linear_regression(outdoor_c, load_kw)

    logger.info("--- baselines fitted on a synthetic year (n=8760) ---")
    for fit in (mean_fit, regression_fit):
        score = cvrmse(load_kw, fit.predict(outdoor_c), n_params=fit.n_params)
        logger.info(
            "%-20s p=%d  coefficients=%s  CV(RMSE)=%.2f%%",
            fit.name,
            fit.n_params,
            np.round(fit.coefficients, 2),
            score,
        )

    best_baseline = min(
        cvrmse(load_kw, f.predict(outdoor_c), n_params=f.n_params)
        for f in (mean_fit, regression_fit)
    )

    logger.info("--- what a calibrated model would have to reach ---")
    logger.info(
        "best baseline CV(RMSE) = %.2f%%  ->  model must be <= %.2f%%",
        best_baseline,
        best_baseline * (1 - MIN_RELATIVE_IMPROVEMENT_PCT / _PERCENT),
    )
    for candidate in (best_baseline * 0.95, best_baseline * 0.70, best_baseline * 0.50):
        logger.info(
            "candidate CV(RMSE)=%6.2f%%  improvement=%+6.2f%%  beats baseline: %s",
            candidate,
            relative_cvrmse_improvement_pct(best_baseline, candidate),
            beats_baseline(best_baseline, candidate),
        )
